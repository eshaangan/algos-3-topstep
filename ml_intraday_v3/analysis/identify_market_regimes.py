#!/usr/bin/env python3
"""
Market Regime Identification

Analyzes historical data to classify periods as:
- BULL: Strong uptrend
- BEAR: Strong downtrend  
- VOLATILE: High volatility, choppy
- RANGING: Low volatility, sideways

Used to ensure balanced training data across all market conditions.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_regime_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for regime classification.
    
    Returns DataFrame with regime indicators.
    """
    df = bars.copy()
    
    # Trend indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_20'] = (df['close'] - df['sma_20']) / df['sma_20']  # % above/below SMA
    df['trend_50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Momentum
    df['return_20'] = df['close'].pct_change(20)
    df['return_50'] = df['close'].pct_change(50)
    
    # Volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['atr_14'] = calculate_atr(df, 14)
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # Range vs trend
    df['hh_20'] = df['high'].rolling(20).max()
    df['ll_20'] = df['low'].rolling(20).min()
    df['range_20'] = (df['hh_20'] - df['ll_20']) / df['close']
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr


def classify_regime(row: pd.Series) -> str:
    """
    Classify a single bar's regime based on indicators.
    
    Rules:
    - BULL: trend_50 > 2% AND volatility < 30%
    - BEAR: trend_50 < -2% AND volatility < 30%
    - VOLATILE: volatility > 40%
    - RANGING: abs(trend_50) < 2% AND volatility < 30%
    """
    trend_50 = row['trend_50']
    volatility = row['volatility_20']
    
    if pd.isna(trend_50) or pd.isna(volatility):
        return 'UNKNOWN'
    
    # High volatility overrides trend
    if volatility > 0.40:
        return 'VOLATILE'
    
    # Trending markets
    if trend_50 > 0.02:
        return 'BULL'
    elif trend_50 < -0.02:
        return 'BEAR'
    
    # Ranging market
    if abs(trend_50) < 0.02 and volatility < 0.30:
        return 'RANGING'
    
    return 'MIXED'


def identify_regime_periods(
    bars: pd.DataFrame,
    min_period_days: int = 10
) -> pd.DataFrame:
    """
    Identify continuous regime periods.
    
    Args:
        bars: OHLCV bars
        min_period_days: Minimum days for a regime period
    
    Returns:
        DataFrame with regime periods
    """
    logger.info("Calculating regime indicators...")
    df = calculate_regime_indicators(bars)
    
    logger.info("Classifying regimes...")
    df['regime'] = df.apply(classify_regime, axis=1)
    
    # Group consecutive same-regime bars
    df['regime_change'] = (df['regime'] != df['regime'].shift(1)).cumsum()
    
    # Summarize periods
    periods = []
    for regime_id, group in df.groupby('regime_change'):
        regime = group['regime'].iloc[0]
        start_date = group.index[0]
        end_date = group.index[-1]
        num_bars = len(group)
        
        # Skip short periods and UNKNOWN
        if regime == 'UNKNOWN' or num_bars < min_period_days * 78:  # ~78 bars per day (5min)
            continue
        
        # Calculate statistics
        start_price = group['close'].iloc[0]
        end_price = group['close'].iloc[-1]
        total_return = (end_price - start_price) / start_price
        volatility = group['close'].pct_change().std() * np.sqrt(252)
        
        periods.append({
            'regime': regime,
            'start_date': start_date,
            'end_date': end_date,
            'num_bars': num_bars,
            'num_days': num_bars / 78,
            'total_return': total_return,
            'annualized_volatility': volatility,
            'start_price': start_price,
            'end_price': end_price,
        })
    
    periods_df = pd.DataFrame(periods)
    
    return periods_df


def analyze_regime_balance(periods_df: pd.DataFrame) -> Dict:
    """
    Analyze balance of regimes in dataset.
    
    Returns dict with regime statistics.
    """
    total_bars = periods_df['num_bars'].sum()
    
    regime_stats = {}
    for regime in ['BULL', 'BEAR', 'VOLATILE', 'RANGING', 'MIXED']:
        regime_periods = periods_df[periods_df['regime'] == regime]
        num_periods = len(regime_periods)
        num_bars = regime_periods['num_bars'].sum()
        pct_bars = num_bars / total_bars * 100 if total_bars > 0 else 0
        
        regime_stats[regime] = {
            'num_periods': num_periods,
            'num_bars': num_bars,
            'pct_bars': pct_bars,
            'avg_period_days': regime_periods['num_days'].mean() if num_periods > 0 else 0,
        }
    
    return regime_stats


def recommend_training_periods(
    periods_df: pd.DataFrame,
    target_bull_pct: float = 0.50,
    target_bear_pct: float = 0.25,
    target_volatile_pct: float = 0.15,
    target_ranging_pct: float = 0.10,
) -> List[Dict]:
    """
    Recommend which periods to include for balanced training.
    
    Args:
        periods_df: Regime periods DataFrame
        target_*_pct: Target percentage for each regime
    
    Returns:
        List of recommended periods with justification
    """
    total_bars = periods_df['num_bars'].sum()
    
    recommendations = []
    
    # Select periods by regime to meet targets
    for regime, target_pct in [
        ('BULL', target_bull_pct),
        ('BEAR', target_bear_pct),
        ('VOLATILE', target_volatile_pct),
        ('RANGING', target_ranging_pct),
    ]:
        regime_periods = periods_df[periods_df['regime'] == regime].copy()
        regime_periods = regime_periods.sort_values('num_bars', ascending=False)
        
        target_bars = int(total_bars * target_pct)
        selected_bars = 0
        
        for _, period in regime_periods.iterrows():
            if selected_bars >= target_bars:
                break
            
            recommendations.append({
                'regime': regime,
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'num_days': period['num_days'],
                'reason': f"Include to meet {regime} target ({target_pct*100:.0f}%)",
            })
            
            selected_bars += period['num_bars']
    
    return recommendations


def plot_regime_timeline(bars: pd.DataFrame, periods_df: pd.DataFrame, output_path: Path):
    """
    Create visualization of regime timeline.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Plot 1: Price with regime colors
    ax1.plot(bars.index, bars['close'], color='black', linewidth=0.5, alpha=0.5)
    
    regime_colors = {
        'BULL': 'green',
        'BEAR': 'red',
        'VOLATILE': 'orange',
        'RANGING': 'blue',
        'MIXED': 'gray',
    }
    
    for _, period in periods_df.iterrows():
        ax1.axvspan(
            period['start_date'],
            period['end_date'],
            alpha=0.2,
            color=regime_colors.get(period['regime'], 'gray'),
            label=period['regime'] if period['regime'] not in [p.get_label() for p in ax1.patches] else ""
        )
    
    ax1.set_ylabel('Price')
    ax1.set_title('Market Regimes Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime distribution bar chart
    regime_stats = periods_df.groupby('regime')['num_bars'].sum()
    regime_stats = regime_stats / regime_stats.sum() * 100
    
    colors = [regime_colors.get(regime, 'gray') for regime in regime_stats.index]
    ax2.bar(regime_stats.index, regime_stats.values, color=colors, alpha=0.7)
    ax2.set_ylabel('Percentage of Bars (%)')
    ax2.set_xlabel('Regime')
    ax2.set_title('Regime Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved regime visualization to: {output_path}")


def main():
    logger.info("="*80)
    logger.info("MARKET REGIME IDENTIFICATION")
    logger.info("="*80)
    
    # Load data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\nLoading data from: {data_path}")
    
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    
    # Filter to analysis period
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    bars = bars[(bars.index >= start_date) & (bars.index <= end_date)]
    
    logger.info(f"Analysis period: {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")
    
    # Identify regime periods
    periods_df = identify_regime_periods(bars, min_period_days=10)
    
    logger.info(f"\nIdentified {len(periods_df)} regime periods:")
    logger.info("\n" + periods_df.to_string())
    
    # Analyze balance
    logger.info("\n" + "="*80)
    logger.info("REGIME BALANCE ANALYSIS")
    logger.info("="*80)
    
    regime_stats = analyze_regime_balance(periods_df)
    
    for regime, stats in regime_stats.items():
        logger.info(f"\n{regime}:")
        logger.info(f"  Periods: {stats['num_periods']}")
        logger.info(f"  Bars: {stats['num_bars']:,} ({stats['pct_bars']:.1f}%)")
        logger.info(f"  Avg period: {stats['avg_period_days']:.1f} days")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("TRAINING DATA RECOMMENDATIONS")
    logger.info("="*80)
    
    recommendations = recommend_training_periods(periods_df)
    
    logger.info(f"\nRecommended {len(recommendations)} periods for balanced training:")
    for rec in recommendations:
        logger.info(f"\n{rec['regime']} ({rec['start_date'].date()} to {rec['end_date'].date()}):")
        logger.info(f"  Duration: {rec['num_days']:.0f} days")
        logger.info(f"  Reason: {rec['reason']}")
    
    # Create visualization
    output_dir = Path("ml_intraday_v3/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "market_regimes_2024_2025.png"
    plot_regime_timeline(bars, periods_df, plot_path)
    
    # Save results
    periods_path = output_dir / "regime_periods_2024_2025.csv"
    periods_df.to_csv(periods_path, index=False)
    logger.info(f"\nSaved regime periods to: {periods_path}")
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    
    # Check balance
    bull_pct = regime_stats['BULL']['pct_bars']
    bear_pct = regime_stats['BEAR']['pct_bars']
    
    logger.info(f"\nCurrent distribution:")
    logger.info(f"  BULL: {bull_pct:.1f}%")
    logger.info(f"  BEAR: {bear_pct:.1f}%")
    logger.info(f"  VOLATILE: {regime_stats['VOLATILE']['pct_bars']:.1f}%")
    logger.info(f"  RANGING: {regime_stats['RANGING']['pct_bars']:.1f}%")
    
    if bull_pct > 60:
        logger.warning("\n⚠️  WARNING: Training data is heavily BULL-biased!")
        logger.warning("   This will create models that struggle with SHORT predictions.")
        logger.warning("   Recommendation: Include more BEAR/VOLATILE periods or use undersampling.")
    elif bear_pct > 60:
        logger.warning("\n⚠️  WARNING: Training data is heavily BEAR-biased!")
        logger.warning("   This will create models that struggle with LONG predictions.")
        logger.warning("   Recommendation: Include more BULL periods or use undersampling.")
    else:
        logger.info("\n✅ Data appears reasonably balanced across regimes.")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
