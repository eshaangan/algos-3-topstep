"""
Backtest ML model on Jan 2026 MES 5-min data WITH CUSUM event filtering.

Compares FIXED features (all 34 generated correctly) vs BROKEN features
(19 features replaced with training medians, simulating NaN + median imputation).

CUSUM filtering ensures we only generate predictions on structurally
significant bars (where cumulative price change exceeds ATR-based threshold),
matching the live system's LiveEventDetector logic.

Usage:
    cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
    python -m ml_intraday_v3.backtest_jan2026_fixed
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
import joblib

# Ensure imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml_intraday_v3"))

from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "ml_intraday_v3" / "model_bundle_retrained_oct2024_nov2025.pkl"
DATA_PATH = PROJECT_ROOT / "ml_intraday_v3" / "ml_intraday_v3" / "data" / "jan2026_mes" / "mes_jan2026_5m.parquet"
FEATURES_CONFIG_PATH = PROJECT_ROOT / "ml_intraday_v3" / "configs" / "features.yaml"

# Trade parameters
POINT_VALUE = 5.0          # $5 per point for MES
TICK_SIZE = 0.25           # MES tick size in points
TICK_VALUE = POINT_VALUE * TICK_SIZE  # $1.25 per tick
COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1.5
SLIPPAGE_DOLLARS = SLIPPAGE_TICKS * TICK_VALUE  # $1.875
ROUND_TRIP_COST = 2 * (COMMISSION_PER_SIDE + SLIPPAGE_DOLLARS)  # $4.99 per RT
PT_ATR_MULT = 2.0          # Profit target = 2x ATR
SL_ATR_MULT = 1.5          # Stop loss = 1.5x ATR
MAX_HOLD_BARS = 24         # Time stop: 24 bars = 2 hours
WARMUP_BARS = 100          # Warmup period for feature computation
CONFIDENCE_LONG = 0.55     # LONG if p_target > 0.55
CONFIDENCE_SHORT = 0.45    # SHORT if p_target < 0.45
NUM_CONTRACTS = 2          # 2 contracts per trade (matching original backtest)

# CUSUM parameters
CUSUM_ATR_PERIOD = 14
CUSUM_ATR_MULT = 0.8

# 19 features that were broken (NaN -> median imputation) in old config
BROKEN_FEATURES = [
    'log_return_2', 'log_return_6', 'log_return_12', 'log_return_24',
    'ema_34', 'sma_20', 'sma_30', 'trend_strength', 'autocorr_5',
    'bb_position', 'volume_imbalance', 'price_vs_vwap', 'relative_volume',
    'large_move', 'candle_body', 'candle_range', 'body_pct',
    'upper_wick', 'lower_wick',
]


def load_data() -> pd.DataFrame:
    """Load Jan 2026 5m bars."""
    df = pd.read_parquet(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("No DatetimeIndex and no 'timestamp' column")
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()
    return df


def load_features_config() -> dict:
    """Load features.yaml config."""
    with open(FEATURES_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def generate_all_features(bars_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate features for all bars at once using build_features."""
    features_df = build_features(bars_df, "5m", config)
    return features_df


def cusum_events(close_prices: pd.Series, atr_values: pd.Series, mult: float = 0.8) -> list:
    """
    Return list of bar indices where CUSUM events fire.

    Tracks positive and negative cumulative sums of price changes.
    Signals an event when either exceeds threshold = mult * current_ATR.
    Resets the triggered accumulator to zero on event.
    """
    events = []
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, len(close_prices)):
        atr_val = atr_values.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        threshold = mult * atr_val
        diff = close_prices.iloc[i] - close_prices.iloc[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos >= threshold:
            events.append(i)
            s_pos = 0.0
        elif s_neg <= -threshold:
            events.append(i)
            s_neg = 0.0
    return events


def simulate_trade(
    bars_df: pd.DataFrame,
    entry_bar_idx: int,
    side: int,
    atr_at_signal: float,
) -> dict:
    """
    Simulate a single trade with ATR-based stops.

    Entry at next bar's open after signal.
    Exit at PT, SL, time stop, or end of data.

    Returns dict with trade details.
    """
    entry_idx = entry_bar_idx + 1
    if entry_idx >= len(bars_df):
        return None

    entry_price = bars_df.iloc[entry_idx]["open"]
    entry_time = bars_df.index[entry_idx]

    pt_distance = PT_ATR_MULT * atr_at_signal
    sl_distance = SL_ATR_MULT * atr_at_signal

    if side == 1:  # LONG
        target_price = entry_price + pt_distance
        stop_price = entry_price - sl_distance
    else:  # SHORT
        target_price = entry_price - pt_distance
        stop_price = entry_price + sl_distance

    exit_price = None
    exit_reason = None
    exit_time = None

    for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, len(bars_df))):
        bar = bars_df.iloc[j]
        bar_time = bars_df.index[j]
        bar_is_bullish = bar["close"] >= bar["open"]

        if side == 1:  # LONG
            if bar_is_bullish:
                # O->H->L->C: check target first, then stop
                if bar["high"] >= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_time = bar_time
                    break
                if bar["low"] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_time = bar_time
                    break
            else:
                # O->L->H->C: check stop first, then target
                if bar["low"] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_time = bar_time
                    break
                if bar["high"] >= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_time = bar_time
                    break
        else:  # SHORT
            if bar_is_bullish:
                # O->H->L->C: check stop first (high), then target (low)
                if bar["high"] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_time = bar_time
                    break
                if bar["low"] <= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_time = bar_time
                    break
            else:
                # O->L->H->C: check target first (low), then stop (high)
                if bar["low"] <= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_time = bar_time
                    break
                if bar["high"] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    exit_time = bar_time
                    break

    # Time stop: exit at close of last allowed bar
    if exit_price is None:
        last_bar_idx = min(entry_idx + MAX_HOLD_BARS - 1, len(bars_df) - 1)
        exit_price = bars_df.iloc[last_bar_idx]["close"]
        exit_reason = "time_stop"
        exit_time = bars_df.index[last_bar_idx]

    # Calculate P&L
    if side == 1:
        raw_pnl_points = exit_price - entry_price
    else:
        raw_pnl_points = entry_price - exit_price

    raw_pnl_dollars = raw_pnl_points * POINT_VALUE * NUM_CONTRACTS
    net_pnl_dollars = raw_pnl_dollars - ROUND_TRIP_COST * NUM_CONTRACTS

    holding_bars = (
        bars_df.index.get_loc(exit_time) - bars_df.index.get_loc(entry_time) + 1
        if exit_time in bars_df.index and entry_time in bars_df.index
        else 0
    )

    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": "LONG" if side == 1 else "SHORT",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "atr": atr_at_signal,
        "exit_reason": exit_reason,
        "raw_pnl": raw_pnl_dollars,
        "net_pnl": net_pnl_dollars,
        "holding_bars": holding_bars,
    }


def run_backtest(
    bars_df: pd.DataFrame,
    features_df: pd.DataFrame,
    predictor: LiveModelPredictor,
    cusum_event_indices: list,
    label: str = "FIXED",
) -> dict:
    """
    Run the backtest loop with CUSUM event filtering.

    Only evaluates bars that are CUSUM events AND after warmup.
    For each CUSUM event bar:
      1. Get prediction from model
      2. Apply confidence filter
      3. If signal and no open position, simulate trade
      4. Track P&L
    """
    feature_cols = predictor.feature_columns
    trades = []
    in_position_until_idx = -1  # Bar index when current position exits

    n_cusum_events_evaluated = 0
    n_signals = 0
    n_long_signals = 0
    n_short_signals = 0
    n_skipped_in_position = 0
    n_skipped_warmup = 0

    cusum_set = set(cusum_event_indices)

    for i in range(len(bars_df)):
        # Only evaluate CUSUM event bars
        if i not in cusum_set:
            continue

        # Skip warmup period
        if i < WARMUP_BARS:
            n_skipped_warmup += 1
            continue

        # Skip if still in a position
        if i <= in_position_until_idx:
            n_skipped_in_position += 1
            continue

        n_cusum_events_evaluated += 1

        # Get features for this bar
        feat_row = features_df.iloc[i]

        # Get prediction
        prediction = predictor.predict(feat_row)

        p_target = prediction.get("p_target", 0.5)
        p_stop = prediction.get("p_stop", 0.5)
        side = prediction.get("side", 0)

        # Apply confidence filter
        signal_side = 0
        if p_target > CONFIDENCE_LONG:
            signal_side = 1  # LONG
        elif p_target < CONFIDENCE_SHORT:
            signal_side = -1  # SHORT

        if signal_side == 0:
            continue

        n_signals += 1
        if signal_side == 1:
            n_long_signals += 1
        else:
            n_short_signals += 1

        # Get ATR for stop/target calculation
        atr_val = feat_row.get("atr_14", np.nan)
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Simulate trade
        trade = simulate_trade(bars_df, i, signal_side, atr_val)
        if trade is None:
            continue

        trade["p_target"] = p_target
        trade["p_stop"] = p_stop
        trade["signal_bar_idx"] = i
        trades.append(trade)

        # Block new entries until this trade exits
        if trade["exit_time"] in bars_df.index:
            in_position_until_idx = bars_df.index.get_loc(trade["exit_time"])
        else:
            in_position_until_idx = i + MAX_HOLD_BARS

    # Compute statistics
    results = compute_stats(trades, bars_df, label)
    results["n_cusum_events_total"] = len(cusum_event_indices)
    results["n_cusum_events_after_warmup"] = len([e for e in cusum_event_indices if e >= WARMUP_BARS])
    results["n_cusum_events_evaluated"] = n_cusum_events_evaluated
    results["n_signals"] = n_signals
    results["n_long_signals"] = n_long_signals
    results["n_short_signals"] = n_short_signals
    results["n_skipped_in_position"] = n_skipped_in_position
    results["n_skipped_warmup"] = n_skipped_warmup
    results["trades"] = trades

    return results


def compute_stats(trades: list, bars_df: pd.DataFrame, label: str) -> dict:
    """Compute backtest statistics from trade list."""
    if not trades:
        return {
            "label": label,
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "daily_pnl": {},
        }

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n_trades = len(trades)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Max drawdown
    equity_curve = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Daily P&L
    daily_pnl = defaultdict(float)
    for t in trades:
        day = t["entry_time"].strftime("%Y-%m-%d")
        daily_pnl[day] += t["net_pnl"]
    daily_pnl = dict(sorted(daily_pnl.items()))

    # Sharpe ratio (annualized from daily)
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Trades by side
    long_trades = [t for t in trades if t["side"] == "LONG"]
    short_trades = [t for t in trades if t["side"] == "SHORT"]
    long_pnl = sum(t["net_pnl"] for t in long_trades)
    short_pnl = sum(t["net_pnl"] for t in short_trades)
    long_wins = sum(1 for t in long_trades if t["net_pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["net_pnl"] > 0)

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1

    return {
        "label": label,
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / n_trades, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "long_win_rate": round(long_wins / len(long_trades) * 100, 1) if long_trades else 0,
        "short_win_rate": round(short_wins / len(short_trades) * 100, 1) if short_trades else 0,
        "exit_reasons": dict(exit_reasons),
        "daily_pnl": daily_pnl,
        "avg_holding_bars": round(np.mean([t["holding_bars"] for t in trades]), 1),
        "round_trip_cost": ROUND_TRIP_COST * NUM_CONTRACTS,
    }


def create_broken_features(
    features_df: pd.DataFrame,
    feature_columns: list,
    medians: list,
) -> pd.DataFrame:
    """
    Simulate the BROKEN features.yaml by setting 19 features to their
    TRAINING median (from the model's preprocessor_state).

    This mimics what happens when features are NaN and the preprocessor's
    median imputation fills them all with the same constant value from training.
    """
    broken_df = features_df.copy()
    for feat in BROKEN_FEATURES:
        if feat in broken_df.columns and feat in feature_columns:
            idx = feature_columns.index(feat)
            broken_df[feat] = medians[idx]
    return broken_df


def print_results(results: dict):
    """Pretty-print backtest results."""
    label = results["label"]
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS: {label}")
    print(f"{'='*70}")
    print(f"  Contracts:        {NUM_CONTRACTS}")
    print(f"  Trades:           {results['n_trades']}")
    print(f"    LONG:           {results.get('n_long', 'N/A')} (win rate: {results.get('long_win_rate', 0):.1f}%)")
    print(f"    SHORT:          {results.get('n_short', 'N/A')} (win rate: {results.get('short_win_rate', 0):.1f}%)")
    print(f"  CUSUM events:     {results.get('n_cusum_events_total', 'N/A')} total, {results.get('n_cusum_events_after_warmup', 'N/A')} after warmup")
    print(f"  Events evaluated: {results.get('n_cusum_events_evaluated', 'N/A')} (skipped in-position: {results.get('n_skipped_in_position', 0)})")
    print(f"  Signals total:    {results.get('n_signals', 'N/A')} (L:{results.get('n_long_signals', 'N/A')} / S:{results.get('n_short_signals', 'N/A')})")
    print(f"  Total P&L:        ${results['total_pnl']:+,.2f}")
    print(f"    LONG P&L:       ${results.get('long_pnl', 0):+,.2f}")
    print(f"    SHORT P&L:      ${results.get('short_pnl', 0):+,.2f}")
    print(f"  Avg P&L/trade:    ${results['avg_pnl']:+,.2f}")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")
    print(f"  Avg Win:          ${results['avg_win']:+,.2f}")
    print(f"  Avg Loss:         ${results['avg_loss']:+,.2f}")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print(f"  Max Drawdown:     ${results['max_drawdown']:+,.2f}")
    print(f"  Sharpe Ratio:     {results['sharpe']:.2f}")
    print(f"  Avg Hold (bars):  {results.get('avg_holding_bars', 'N/A')}")
    print(f"  RT Cost/trade:    ${results['round_trip_cost']:.2f} ({NUM_CONTRACTS} contracts)")

    print(f"\n  Exit Reasons:")
    for reason, count in sorted(results.get("exit_reasons", {}).items()):
        pct = count / results["n_trades"] * 100 if results["n_trades"] > 0 else 0
        print(f"    {reason:15s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n  Daily P&L:")
    daily = results.get("daily_pnl", {})
    for day, pnl in daily.items():
        print(f"    {day}: ${pnl:+10.2f}")

    # Summary
    n_days = len(daily) if daily else 1
    avg_daily = results["total_pnl"] / n_days
    winning_days = sum(1 for p in daily.values() if p > 0)
    print(f"\n  Trading days:     {n_days}")
    print(f"  Avg daily P&L:    ${avg_daily:+,.2f}")
    print(f"  Winning days:     {winning_days}/{n_days} ({winning_days/n_days*100:.0f}%)")
    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("  Jan 2026 Backtest: FIXED vs BROKEN Features (CUSUM + 2 Contracts)")
    print("=" * 70)
    print(f"  Model:       {MODEL_PATH.name}")
    print(f"  Data:        {DATA_PATH.name}")
    print(f"  Bars:        5-min MES")
    print(f"  Contracts:   {NUM_CONTRACTS}")
    print(f"  PT:          {PT_ATR_MULT}x ATR, SL: {SL_ATR_MULT}x ATR")
    print(f"  Confidence:  LONG > {CONFIDENCE_LONG}, SHORT < {CONFIDENCE_SHORT}")
    print(f"  Max Hold:    {MAX_HOLD_BARS} bars (2 hours)")
    print(f"  Costs:       ${COMMISSION_PER_SIDE}/side commission + {SLIPPAGE_TICKS} ticks slippage")
    print(f"  RT Cost:     ${ROUND_TRIP_COST:.2f}/contract x {NUM_CONTRACTS} = ${ROUND_TRIP_COST * NUM_CONTRACTS:.2f}")
    print(f"  CUSUM:       ATR period={CUSUM_ATR_PERIOD}, mult={CUSUM_ATR_MULT}")
    print()

    # Load data
    print("Loading data...")
    bars_df = load_data()
    print(f"  {len(bars_df)} bars from {bars_df.index[0]} to {bars_df.index[-1]}")

    # Load config
    print("Loading features config...")
    config = load_features_config()

    # Generate features
    print("Generating features (all 34)...")
    features_df = generate_all_features(bars_df, config)
    print(f"  Generated {len(features_df.columns)} feature columns")
    print(f"  Feature columns: {list(features_df.columns)}")

    # Load predictor
    predictor = LiveModelPredictor(MODEL_PATH)
    model_feats = predictor.feature_columns
    missing = [f for f in model_feats if f not in features_df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    else:
        print(f"  All {len(model_feats)} model features present.")

    # Check NaN counts after warmup
    after_warmup = features_df.iloc[WARMUP_BARS:]
    nan_counts = after_warmup[model_feats].isna().sum()
    if nan_counts.any():
        print(f"\n  NaN counts in model features (after warmup):")
        for feat, cnt in nan_counts[nan_counts > 0].items():
            print(f"    {feat}: {cnt}")
    else:
        print(f"  No NaN in model features after warmup bar {WARMUP_BARS}.")

    # -----------------------------------------------------------------------
    # Compute CUSUM events
    # -----------------------------------------------------------------------
    print("\nComputing CUSUM events...")
    atr_col = f"atr_{CUSUM_ATR_PERIOD}"
    if atr_col not in features_df.columns:
        raise ValueError(f"ATR column '{atr_col}' not found in features. Available: {list(features_df.columns)}")

    cusum_event_indices = cusum_events(
        bars_df["close"],
        features_df[atr_col],
        mult=CUSUM_ATR_MULT,
    )
    cusum_after_warmup = [e for e in cusum_event_indices if e >= WARMUP_BARS]
    print(f"  Total CUSUM events: {len(cusum_event_indices)}")
    print(f"  After warmup ({WARMUP_BARS} bars): {len(cusum_after_warmup)}")
    print(f"  Event rate: {len(cusum_after_warmup) / (len(bars_df) - WARMUP_BARS) * 100:.1f}% of bars")

    # -----------------------------------------------------------------------
    # Run 1: FIXED features (correct feature generation) + CUSUM
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Running backtest with FIXED features + CUSUM...")
    print("-" * 70)
    fixed_results = run_backtest(
        bars_df, features_df, predictor, cusum_event_indices,
        label="FIXED (all 34 features correct, CUSUM, 2 contracts)",
    )
    print_results(fixed_results)

    # -----------------------------------------------------------------------
    # Run 2: BROKEN features (19 features set to training median) + CUSUM
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Running backtest with BROKEN features (19 set to training median) + CUSUM...")
    print("-" * 70)

    # Get training medians from model bundle
    bundle = joblib.load(MODEL_PATH)
    training_medians = bundle["primary_preprocessor"]["medians"]
    training_feature_cols = bundle["primary_feature_columns"]

    broken_features_df = create_broken_features(features_df, training_feature_cols, training_medians)

    # Verify the broken features are constant
    print("  Verifying broken features are constant (training median):")
    for feat in BROKEN_FEATURES[:5]:
        if feat in broken_features_df.columns and feat in training_feature_cols:
            idx = training_feature_cols.index(feat)
            vals = broken_features_df[feat].iloc[WARMUP_BARS:].unique()
            print(f"    {feat}: {len(vals)} unique val(s) -> {vals[0]:.6f} (training median: {training_medians[idx]:.6f})")

    broken_results = run_backtest(
        bars_df, broken_features_df, predictor, cusum_event_indices,
        label="BROKEN (19 features = training median, CUSUM, 2 contracts)",
    )
    print_results(broken_results)

    # -----------------------------------------------------------------------
    # Comparison summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY (CUSUM filtered, 2 contracts)")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'FIXED':>15s} {'BROKEN':>15s} {'Delta':>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    fixed_daily = fixed_results.get("daily_pnl", {})
    broken_daily = broken_results.get("daily_pnl", {})
    fixed_avg_daily = fixed_results["total_pnl"] / max(len(fixed_daily), 1)
    broken_avg_daily = broken_results["total_pnl"] / max(len(broken_daily), 1)

    comparisons = [
        ("Trades", fixed_results["n_trades"], broken_results["n_trades"]),
        ("Total P&L ($)", fixed_results["total_pnl"], broken_results["total_pnl"]),
        ("Avg P&L/trade ($)", fixed_results["avg_pnl"], broken_results["avg_pnl"]),
        ("Avg P&L/day ($)", round(fixed_avg_daily, 2), round(broken_avg_daily, 2)),
        ("Win Rate (%)", fixed_results["win_rate"], broken_results["win_rate"]),
        ("Avg Win ($)", fixed_results["avg_win"], broken_results["avg_win"]),
        ("Avg Loss ($)", fixed_results["avg_loss"], broken_results["avg_loss"]),
        ("Profit Factor", fixed_results["profit_factor"], broken_results["profit_factor"]),
        ("Max Drawdown ($)", fixed_results["max_drawdown"], broken_results["max_drawdown"]),
        ("Sharpe Ratio", fixed_results["sharpe"], broken_results["sharpe"]),
    ]

    for metric, fixed_val, broken_val in comparisons:
        delta = fixed_val - broken_val if isinstance(fixed_val, (int, float)) else "N/A"
        if isinstance(delta, float):
            print(f"  {metric:<25s} {fixed_val:>15.2f} {broken_val:>15.2f} {delta:>+15.2f}")
        else:
            print(f"  {metric:<25s} {fixed_val:>15} {broken_val:>15} {delta:>15}")

    print(f"  {'='*70}")

    # Save results to JSON
    output_path = PROJECT_ROOT / "ml_intraday_v3" / "backtest_results" / "jan2026_fixed_vs_broken_cusum.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def serialize_results(res):
        r = {k: v for k, v in res.items() if k != "trades"}
        r["trade_summary"] = []
        for t in res.get("trades", []):
            ts = {k: v for k, v in t.items()}
            ts["entry_time"] = str(ts["entry_time"])
            ts["exit_time"] = str(ts["exit_time"])
            r["trade_summary"].append(ts)
        return r

    output = {
        "run_time": datetime.now().isoformat(),
        "model": str(MODEL_PATH.name),
        "data": str(DATA_PATH.name),
        "params": {
            "pt_atr_mult": PT_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT,
            "max_hold_bars": MAX_HOLD_BARS,
            "confidence_long": CONFIDENCE_LONG,
            "confidence_short": CONFIDENCE_SHORT,
            "round_trip_cost_per_contract": ROUND_TRIP_COST,
            "num_contracts": NUM_CONTRACTS,
            "cusum_atr_period": CUSUM_ATR_PERIOD,
            "cusum_atr_mult": CUSUM_ATR_MULT,
        },
        "fixed": serialize_results(fixed_results),
        "broken": serialize_results(broken_results),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
