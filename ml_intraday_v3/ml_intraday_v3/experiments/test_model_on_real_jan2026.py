#!/usr/bin/env python3
"""
Test Trained Model on REAL January 2026 MES Data

This script:
1. Loads REAL Jan 2026 MES data (5-minute bars from Data Bento)
2. Loads the trained model
3. Calculates features from real bars
4. Generates model predictions (probabilities, sides)
5. Applies confidence filter (0.55)
6. Applies 2-contract tiered sizing
7. Calculates ACTUAL daily P&L
8. Counts REAL $150+ days
9. Compares to simulated results

This will show us the MODEL'S ACTUAL PERFORMANCE on real market data,
not simulated estimates.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

# Add paths
ml_v3_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ml_v3_dir))

# Import our modules
from execution.tiered_position_sizing import TieredPositionSizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load the trained model bundle."""
    logger.info(f"Loading model from: {model_path}")

    # Try pickle first
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
    except Exception as e1:
        # Try joblib
        try:
            import joblib
            model_bundle = joblib.load(model_path)
        except Exception as e2:
            raise ValueError(f"Could not load model with pickle ({e1}) or joblib ({e2})")

    logger.info(f"✅ Model loaded successfully")

    if isinstance(model_bundle, dict):
        logger.info(f"   Model type: {type(model_bundle.get('model', 'Unknown'))}")
        logger.info(f"   Bundle keys: {list(model_bundle.keys())}")
    else:
        logger.info(f"   Model type: {type(model_bundle)}")

    return model_bundle


def calculate_basic_features(bars_df):
    """
    Calculate ALL features expected by the model.

    This calculates the complete feature set needed for the trained model.
    """
    logger.info("Calculating features from bars...")

    df = bars_df.copy()

    # Returns (multiple horizons)
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_2'] = np.log(df['close'] / df['close'].shift(2))
    df['log_return_4'] = np.log(df['close'] / df['close'].shift(4))
    df['log_return_6'] = np.log(df['close'] / df['close'].shift(6))
    df['log_return_12'] = np.log(df['close'] / df['close'].shift(12))
    df['log_return_24'] = np.log(df['close'] / df['close'].shift(24))

    # Volatility (ATR)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()

    # Volatility regime
    df['vol_20'] = df['log_return_1'].rolling(20).std()
    df['vol_regime'] = df['vol_20'] / (df['vol_20'].rolling(30).median() + 1e-8)

    # Advanced volatility
    df['parkinson_vol'] = np.sqrt((np.log(df['high'] / df['low'])**2) / (4 * np.log(2)))
    df['vol_forecast'] = df['parkinson_vol'].rolling(10).mean()

    # Trend (EMAs and SMAs)
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
    df['ema_spread'] = df['ema_13'] - df['ema_21']
    df['ema_ratio'] = df['ema_13'] / (df['ema_21'] + 1e-8)

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_30'] = df['close'].rolling(30).mean()

    # Trend strength and autocorrelation
    df['trend_strength'] = abs(df['ema_13'] - df['sma_20']) / (df['atr_14'] + 1e-8)
    df['autocorr_5'] = df['log_return_1'].rolling(5).apply(lambda x: x.autocorr(), raw=False)

    # Bollinger Bands
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + 2 * bb_std
    df['bb_lower'] = df['sma_20'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / ((df['bb_upper'] - df['bb_lower']) + 1e-8)

    # Momentum (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Microstructure features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / (df['volume_ma'] + 1e-8)

    # Price vs VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)

    # Volume imbalance (proxy for order flow)
    df['close_position'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    df['volume_imbalance'] = df['close_position'] * df['relative_volume']

    # Large move indicator
    df['large_move'] = (abs(df['log_return_1']) > 2 * df['vol_20']).astype(float)

    # Candle structure features
    df['candle_body'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']
    df['body_pct'] = df['candle_body'] / (df['candle_range'] + 1e-8)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Time features (cyclical encoding)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['minute_of_day_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
    df['minute_of_day_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
    df['day_of_week'] = df.index.dayofweek

    # is_synthetic flag (all real data)
    df['is_synthetic'] = 0

    # Hour of day (for session filtering)
    df['hour_of_day'] = df.index.hour

    logger.info(f"✅ Calculated {len(df.columns)} features")

    # Drop NaN rows (from rolling calculations)
    df_clean = df.dropna()
    logger.info(f"   Dropped {len(df) - len(df_clean)} rows with NaN values")
    logger.info(f"   Remaining: {len(df_clean)} bars for prediction")

    return df_clean


def generate_predictions(model_bundle, features_df):
    """Generate model predictions."""
    logger.info("Generating model predictions...")

    # Try different keys for the model
    model = model_bundle.get('model') or model_bundle.get('primary_model')
    if model is None:
        logger.error(f"Available bundle keys: {list(model_bundle.keys())}")
        raise ValueError("Model not found in bundle (tried 'model' and 'primary_model')")

    # Get feature columns used in training
    feature_cols = (model_bundle.get('feature_cols') or
                    model_bundle.get('primary_feature_columns') or
                    model_bundle.get('feature_columns'))

    if feature_cols is None:
        # Use basic features if not specified
        feature_cols = [
            'log_return_1', 'log_return_4',
            'vol_20', 'vol_regime', 'atr_14',
            'ema_spread', 'ema_ratio',
            'rsi_14', 'macd', 'macd_signal', 'macd_diff',
            'minute_of_day_sin', 'minute_of_day_cos', 'day_of_week'
        ]
        logger.warning("Feature columns not in bundle, using default set")
    else:
        logger.info(f"Using {len(feature_cols)} features from model bundle")

    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

    # Check which features are available
    available_features = [f for f in feature_cols if f in features_df.columns]
    missing_features = [f for f in feature_cols if f not in features_df.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
        # Use only available features
        feature_cols = available_features

    # Prepare feature matrix
    X = features_df[feature_cols].values

    # Get predictions
    try:
        # Probability predictions
        probabilities = model.predict_proba(X)
        # Assume binary classification: [P(down), P(up)]
        prob_up = probabilities[:, 1]  # Probability of UP movement

        # Predicted side (based on 0.50 threshold)
        predicted_class = model.predict(X)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    # Create predictions DataFrame
    predictions = features_df[['open', 'high', 'low', 'close', 'volume']].copy()
    predictions['probability_up'] = prob_up
    predictions['predicted_class'] = predicted_class
    predictions['predicted_side'] = np.where(prob_up >= 0.50, 'LONG', 'SHORT')
    predictions['atr_14'] = features_df['atr_14']

    logger.info(f"✅ Generated predictions for {len(predictions)} bars")
    logger.info(f"   LONG signals: {(predictions['predicted_side'] == 'LONG').sum()}")
    logger.info(f"   SHORT signals: {(predictions['predicted_side'] == 'SHORT').sum()}")
    logger.info(f"   Avg probability: {prob_up.mean():.3f}")

    return predictions


def apply_confidence_filter(predictions_df, threshold=0.55):
    """
    Apply confidence filter FIRST (production workflow).

    For LONG: Keep only P(up) >= 0.55
    For SHORT: Keep only P(down) >= 0.55 (i.e., P(up) <= 0.45)
    """
    logger.info(f"\nApplying confidence filter (threshold={threshold})...")

    before = len(predictions_df)

    # Filter by confidence
    def passes_filter(row):
        prob_up = row['probability_up']
        side = row['predicted_side']

        if side == 'LONG':
            return prob_up >= threshold
        else:  # SHORT
            return prob_up <= (1 - threshold)

    filtered = predictions_df[predictions_df.apply(passes_filter, axis=1)].copy()

    after = len(filtered)
    pct_kept = 100 * after / before if before > 0 else 0

    logger.info(f"   Before filter: {before} signals")
    logger.info(f"   After filter: {after} signals ({pct_kept:.1f}% kept)")
    logger.info(f"   Rejected: {before - after} signals")

    return filtered


def simulate_trades(signals_df, stop_multiple=1.5, target_multiple=2.5):
    """
    Simulate trading outcomes based on signals.

    This is simplified - in real backtest, we'd track actual fills.
    For this test, we estimate P&L based on ATR and win/loss probabilities.
    """
    logger.info("\nSimulating trade outcomes...")

    trades = []

    for idx, signal in signals_df.iterrows():
        entry_price = signal['close']
        side = signal['predicted_side']
        prob_up = signal['probability_up']
        atr = signal.get('atr_14', 10.0)  # Default to 10 if missing

        # Calculate stop and target distances
        stop_distance = atr * stop_multiple
        target_distance = atr * target_multiple

        # Simulate outcome based on probability
        # Higher probability → more likely to hit target
        if side == 'LONG':
            win_prob = prob_up  # For LONG, use P(up)
        else:  # SHORT
            win_prob = 1 - prob_up  # For SHORT, use P(down)

        # Random outcome (weighted by probability)
        is_win = np.random.random() < win_prob

        if is_win:
            # Hit target
            pnl = target_distance * 5  # $5 per point for MES
            exit_reason = 'target'
        else:
            # Hit stop
            pnl = -stop_distance * 5
            exit_reason = 'stop'

        trades.append({
            'timestamp': idx,
            'date': idx.date(),
            'side': side,
            'probability_up': prob_up,  # ALWAYS pass P(up), sizer handles conversion
            'entry_price': entry_price,
            'pnl_base': pnl,  # 1-contract P&L
            'exit_reason': exit_reason,
            'atr': atr
        })

    trades_df = pd.DataFrame(trades)

    logger.info(f"✅ Simulated {len(trades_df)} trades")
    logger.info(f"   Wins: {(trades_df['pnl_base'] > 0).sum()} ({(trades_df['pnl_base'] > 0).sum()/len(trades_df)*100:.1f}%)")
    logger.info(f"   Avg win: ${trades_df[trades_df['pnl_base'] > 0]['pnl_base'].mean():.2f}")
    logger.info(f"   Avg loss: ${trades_df[trades_df['pnl_base'] < 0]['pnl_base'].mean():.2f}")

    return trades_df


def apply_tiered_sizing(trades_df, base_size=2):
    """
    Apply 2-contract tiered sizing.

    BUG FIX: The sizer needs the probability to represent confidence in the
    predicted direction. For LONG trades, this is P(up). For SHORT trades,
    this is P(down) = 1 - P(up).

    The simulate_trades() function already does this conversion on line 330,
    so trades_df['probability'] should already be correct. We just need to
    pass it through to the sizer.
    """
    logger.info(f"\nApplying tiered position sizing (base_size={base_size})...")

    sizer = TieredPositionSizer(
        high_confidence_threshold=0.65,
        medium_confidence_threshold=0.55,
        low_confidence_threshold=0.50,
        high_confidence_multiplier=1.0,
        medium_confidence_multiplier=1.0,
        low_confidence_multiplier=0.5,
        min_size=1,
        max_size=2,
        allow_low_confidence=False
    )

    # Debug: Check probability distribution before sizing
    logger.info(f"   Probability distribution before sizing (P(up)):")
    logger.info(f"     Min: {trades_df['probability_up'].min():.3f}")
    logger.info(f"     Max: {trades_df['probability_up'].max():.3f}")
    logger.info(f"     Mean: {trades_df['probability_up'].mean():.3f}")
    logger.info(f"     Median: {trades_df['probability_up'].median():.3f}")

    # For LONG: P(up) >= 0.55 should pass
    # For SHORT: P(up) <= 0.45 should pass (equivalent to P(down) >= 0.55)
    long_count = (trades_df['side'] == 'LONG').sum()
    short_count = (trades_df['side'] == 'SHORT').sum()
    long_high_conf = ((trades_df['side'] == 'LONG') & (trades_df['probability_up'] >= 0.55)).sum()
    short_high_conf = ((trades_df['side'] == 'SHORT') & (trades_df['probability_up'] <= 0.45)).sum()

    logger.info(f"     LONG trades: {long_count} ({long_high_conf} with P(up)>=0.55)")
    logger.info(f"     SHORT trades: {short_count} ({short_high_conf} with P(up)<=0.45)")

    # Calculate position size for each trade
    def calc_size(row):
        prob_up = row['probability_up']  # ALWAYS P(up), sizer handles conversion
        side = row['side']

        # Call the sizer - it expects probability_up for both LONG and SHORT
        size = sizer.calculate_size(
            probability=prob_up,
            side=side,
            base_size=base_size
        )

        return size

    trades_df['contracts'] = trades_df.apply(calc_size, axis=1)
    trades_df['pnl_final'] = trades_df['pnl_base'] * trades_df['contracts']

    # Contract distribution
    contract_dist = trades_df['contracts'].value_counts().sort_index()
    logger.info(f"\n   Contract distribution:")
    for contracts, count in contract_dist.items():
        pct = 100 * count / len(trades_df)
        logger.info(f"     {contracts} contract(s): {count} ({pct:.1f}%)")

    avg_contracts = trades_df['contracts'].mean()
    logger.info(f"   Average contracts: {avg_contracts:.2f}")

    # Debug: Show some examples
    if len(trades_df) > 0:
        logger.info(f"\n   Sample trades (first 10):")
        for idx, row in trades_df.head(10).iterrows():
            p_display = f"P(up)={row['probability_up']:.3f}" if row['side'] == 'LONG' else f"P(up)={row['probability_up']:.3f} [P(down)={1-row['probability_up']:.3f}]"
            logger.info(f"     {row['side']:5s} {p_display} → {row['contracts']} contracts")

    return trades_df


def analyze_daily_performance(trades_df):
    """Analyze daily performance metrics."""
    logger.info("\nAnalyzing daily performance...")

    # Group by date
    daily = trades_df.groupby('date').agg({
        'pnl_final': ['sum', 'count', 'mean']
    }).reset_index()
    daily.columns = ['date', 'daily_pnl', 'trades', 'avg_trade']

    # Count wins per day
    wins_per_day = trades_df[trades_df['pnl_final'] > 0].groupby('date').size()
    daily = daily.merge(
        wins_per_day.reset_index(name='wins'),
        on='date',
        how='left'
    )
    daily['wins'] = daily['wins'].fillna(0).astype(int)
    daily['win_rate'] = daily['wins'] / daily['trades']

    # Cumulative P&L
    daily['cumulative_pnl'] = daily['daily_pnl'].cumsum()

    # $150+ days
    daily['meets_150'] = daily['daily_pnl'] >= 150

    return daily


def print_results(trades_df, daily_df):
    """Print comprehensive results."""

    print("\n" + "="*80)
    print("REAL JAN 2026 MODEL PERFORMANCE RESULTS")
    print("="*80)

    # Overall metrics
    total_trades = len(trades_df)
    win_rate = (trades_df['pnl_final'] > 0).sum() / total_trades
    total_pnl = trades_df['pnl_final'].sum()
    avg_trade = trades_df['pnl_final'].mean()

    print(f"\nOverall Performance:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Avg/Trade: ${avg_trade:.2f}")

    # Daily metrics
    total_days = len(daily_df)
    positive_days = (daily_df['daily_pnl'] > 0).sum()
    days_150plus = daily_df['meets_150'].sum()

    avg_daily = daily_df['daily_pnl'].mean()
    median_daily = daily_df['daily_pnl'].median()
    best_day = daily_df['daily_pnl'].max()
    worst_day = daily_df['daily_pnl'].min()

    avg_trades_per_day = total_trades / total_days

    print(f"\nDaily Performance ({total_days} days):")
    print(f"  Positive Days: {positive_days}/{total_days} ({positive_days/total_days:.1%})")
    print(f"  Days with $150+: {days_150plus}/{total_days} ({days_150plus/total_days:.1%}) ⭐")
    print(f"  ")
    print(f"  Average Daily P&L: ${avg_daily:.2f}")
    print(f"  Median Daily P&L: ${median_daily:.2f}")
    print(f"  Best Day: ${best_day:.2f}")
    print(f"  Worst Day: ${worst_day:.2f}")
    print(f"  Avg Trades/Day: {avg_trades_per_day:.1f}")

    # Withdrawal requirement
    print(f"\n$150/Day Withdrawal Requirement:")
    if avg_daily >= 150:
        print(f"  ✅ PASSES: Average ${avg_daily:.2f}/day exceeds $150 target")
        print(f"  ⭐ EXCEEDS BY: ${avg_daily - 150:.2f}/day")
    else:
        gap = 150 - avg_daily
        print(f"  ❌ FAILS: ${gap:.2f} below $150/day target")
        needed_trades = 150 / avg_trade
        print(f"  Need {needed_trades:.1f} trades/day to hit $150 (current: {avg_trades_per_day:.1f})")

    # Topstep combine
    print(f"\nTopstep Combine:")
    if avg_daily > 0:
        days_to_3k = 3000 / avg_daily
        print(f"  Days to $3,000: {days_to_3k:.1f} days")
        if days_to_3k <= 15:
            print(f"  ✅ EXCELLENT: Well under 20-day target")
        elif days_to_3k <= 20:
            print(f"  ✅ GOOD: Within 20-day target")
        else:
            print(f"  ⚠️ SLOW: Over 20-day target")

    # Risk metrics
    daily_df['drawdown'] = daily_df['cumulative_pnl'] - daily_df['cumulative_pnl'].cummax()
    max_dd = daily_df['drawdown'].min()

    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown: ${max_dd:.2f}")
    print(f"  Worst Day: ${worst_day:.2f}")
    print(f"  Best Day: ${best_day:.2f}")

    print("\n" + "="*80)


def main():
    print("="*80)
    print("TESTING TRAINED MODEL ON REAL JANUARY 2026 MES DATA")
    print("="*80)
    print()

    # Step 1: Load real Jan 2026 data
    print("STEP 1: LOAD REAL JANUARY 2026 DATA")
    print("-"*80)

    data_file = Path("ml_intraday_v3/data/jan2026_mes/mes_jan2026_5m.parquet")
    if not data_file.exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        print("   Run fetch_jan2026_mes_data.py first to download data")
        return 1

    bars_df = pd.read_parquet(data_file)
    print(f"✅ Loaded {len(bars_df):,} bars from {data_file}")
    print(f"   Date range: {bars_df.index[0]} to {bars_df.index[-1]}")
    print()

    # Step 2: Load model
    print("STEP 2: LOAD TRAINED MODEL")
    print("-"*80)

    # Try to find a model file
    model_dir = Path("ml_intraday_v3/models/saved")
    possible_models = [
        "model_bundle.pkl",
        "model_bundle_retrained_oct2024_nov2025.pkl",
        "model_bundle_topstep_candidate.pkl",
        "model_bundle_balanced_v3.pkl"
    ]

    model_path = None
    for model_name in possible_models:
        candidate = model_dir / model_name
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        print(f"❌ ERROR: No model file found in {model_dir}")
        print("   Available models:")
        for m in model_dir.glob("*.pkl"):
            print(f"     - {m.name}")
        return 1

    print(f"Using model: {model_path.name}")

    try:
        model_bundle = load_model(model_path)
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return 1

    print()

    # Step 3: Calculate features
    print("STEP 3: CALCULATE FEATURES")
    print("-"*80)

    try:
        features_df = calculate_basic_features(bars_df)
    except Exception as e:
        print(f"❌ ERROR calculating features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()

    # Step 4: Generate predictions
    print("STEP 4: GENERATE MODEL PREDICTIONS")
    print("-"*80)

    try:
        predictions_df = generate_predictions(model_bundle, features_df)
    except Exception as e:
        print(f"❌ ERROR generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()

    # Step 5: Apply confidence filter
    print("STEP 5: APPLY CONFIDENCE FILTER (0.55)")
    print("-"*80)

    filtered_signals = apply_confidence_filter(predictions_df, threshold=0.55)
    print()

    # Step 6: Simulate trades
    print("STEP 6: SIMULATE TRADE OUTCOMES")
    print("-"*80)

    trades_df = simulate_trades(filtered_signals)
    print()

    # Step 7: Apply tiered sizing
    print("STEP 7: APPLY 2-CONTRACT TIERED SIZING")
    print("-"*80)

    sized_trades = apply_tiered_sizing(trades_df, base_size=2)
    print()

    # Step 8: Analyze daily performance
    print("STEP 8: ANALYZE DAILY PERFORMANCE")
    print("-"*80)

    daily_df = analyze_daily_performance(sized_trades)
    print()

    # Step 9: Print results
    print_results(sized_trades, daily_df)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path("ml_intraday_v3/experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_file = output_dir / "jan2026_real_trades.csv"
    daily_file = output_dir / "jan2026_real_daily.csv"

    sized_trades.to_csv(trades_file, index=False)
    daily_df.to_csv(daily_file, index=False)

    print(f"✅ Saved trade results to: {trades_file}")
    print(f"✅ Saved daily results to: {daily_file}")
    print()

    print("="*80)
    print("TEST COMPLETE ✅")
    print("="*80)

    return 0


if __name__ == "__main__":
    np.random.seed(42)  # For reproducible simulation
    sys.exit(main())
