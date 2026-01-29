#!/usr/bin/env python3
"""
Comprehensive Model Capability Validation

Tests whether models can predict both LONG and SHORT trades by:
1. Testing with synthetic features (bullish/bearish)
2. Testing with actual market data
3. Analyzing training data distribution
4. Validating prediction pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_synthetic_features(predictor: LiveModelPredictor, model_name: str) -> Dict:
    """
    Test model predictions with synthetic features.
    
    Returns dict with test results.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {model_name} with Synthetic Features")
    logger.info(f"{'='*80}")
    
    results = {
        'model': model_name,
        'has_side_feature': predictor.has_side_feature,
        'has_dual_model': predictor.has_dual_model,
        'tests': []
    }
    
    # Create neutral baseline features
    neutral_features = pd.Series(0.0, index=predictor.feature_columns)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Neutral',
            'features': neutral_features.copy(),
        },
        {
            'name': 'Strong Bullish',
            'features': neutral_features.copy(),
            'adjustments': {
                'log_return_1': 0.01,
                'log_return_2': 0.015,
                'log_return_6': 0.025,
                'ema_spread': 0.01,
                'trend_strength': 0.05,
            }
        },
        {
            'name': 'Strong Bearish',
            'features': neutral_features.copy(),
            'adjustments': {
                'log_return_1': -0.01,
                'log_return_2': -0.015,
                'log_return_6': -0.025,
                'ema_spread': -0.01,
                'trend_strength': -0.05,
            }
        },
        {
            'name': 'Moderate Bullish',
            'features': neutral_features.copy(),
            'adjustments': {
                'log_return_1': 0.005,
                'ema_spread': 0.003,
            }
        },
        {
            'name': 'Moderate Bearish',
            'features': neutral_features.copy(),
            'adjustments': {
                'log_return_1': -0.005,
                'ema_spread': -0.003,
            }
        },
    ]
    
    for scenario in scenarios:
        features = scenario['features']
        
        # Apply adjustments
        if 'adjustments' in scenario:
            for feat, val in scenario['adjustments'].items():
                if feat in features.index:
                    features[feat] = val
        
        # Get prediction
        pred = predictor.predict(features, use_meta=False, side=None)
        
        test_result = {
            'scenario': scenario['name'],
            'side': pred.get('side', 'MISSING'),
            'score_ev': pred.get('score_ev', 0.0),
            'score_ev_long': pred.get('score_ev_long', 'N/A'),
            'score_ev_short': pred.get('score_ev_short', 'N/A'),
            'p_target': pred.get('p_target', 'N/A'),
            'p_stop': pred.get('p_stop', 'N/A'),
        }
        
        results['tests'].append(test_result)
        
        # Log results
        logger.info(f"\n--- {scenario['name']} ---")
        logger.info(f"   Predicted Side: {test_result['side']} (1=LONG, -1=SHORT, 0=skip)")
        logger.info(f"   Score EV: {test_result['score_ev']:.4f}")
        
        if test_result['score_ev_long'] != 'N/A':
            logger.info(f"   EV LONG:  {test_result['score_ev_long']:.4f}")
            logger.info(f"   EV SHORT: {test_result['score_ev_short']:.4f}")
        
        if test_result['p_target'] != 'N/A':
            logger.info(f"   P(target): {test_result['p_target']:.4f}")
            logger.info(f"   P(stop):   {test_result['p_stop']:.4f}")
    
    # Analysis
    logger.info(f"\n{'='*80}")
    logger.info(f"Analysis for {model_name}")
    logger.info(f"{'='*80}")
    
    # Count predictions by side
    sides = [t['side'] for t in results['tests'] if t['side'] != 'MISSING']
    long_count = sum(1 for s in sides if s == 1)
    short_count = sum(1 for s in sides if s == -1)
    skip_count = sum(1 for s in sides if s == 0)
    
    logger.info(f"Prediction Distribution:")
    logger.info(f"   LONG:  {long_count}/{len(sides)} ({long_count/len(sides)*100:.1f}%)")
    logger.info(f"   SHORT: {short_count}/{len(sides)} ({short_count/len(sides)*100:.1f}%)")
    logger.info(f"   SKIP:  {skip_count}/{len(sides)} ({skip_count/len(sides)*100:.1f}%)")
    
    # Check for structural bias
    ev_shorts = [t['score_ev_short'] for t in results['tests'] if t['score_ev_short'] != 'N/A']
    
    if ev_shorts:
        avg_ev_short = np.mean(ev_shorts)
        max_ev_short = np.max(ev_shorts)
        min_ev_short = np.min(ev_shorts)
        
        logger.info(f"\nSHORT EV Statistics:")
        logger.info(f"   Mean:   {avg_ev_short:.4f}")
        logger.info(f"   Max:    {max_ev_short:.4f}")
        logger.info(f"   Min:    {min_ev_short:.4f}")
        
        if max_ev_short < 0:
            logger.warning(f"   ⚠️  STRUCTURAL BIAS DETECTED: All SHORT EVs are negative!")
            results['has_short_bias'] = True
        else:
            logger.info(f"   ✅ Model CAN predict profitable SHORT trades")
            results['has_short_bias'] = False
    
    # Check if model responds to bearish features
    bearish_tests = [t for t in results['tests'] if 'Bearish' in t['scenario']]
    if bearish_tests:
        bearish_shorts = sum(1 for t in bearish_tests if t['side'] == -1)
        logger.info(f"\nBearish Scenario Response:")
        logger.info(f"   Predicted SHORT: {bearish_shorts}/{len(bearish_tests)} bearish scenarios")
        
        if bearish_shorts == 0:
            logger.warning(f"   ⚠️  Model does NOT respond to bearish features!")
            results['responds_to_bearish'] = False
        else:
            logger.info(f"   ✅ Model responds to bearish features")
            results['responds_to_bearish'] = True
    
    return results


def test_actual_market_data(predictor: LiveModelPredictor, model_name: str, bars: pd.DataFrame) -> Dict:
    """
    Test model predictions on actual market data.
    
    Returns dict with statistics.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {model_name} on Actual Market Data")
    logger.info(f"{'='*80}")
    
    # Import feature generator
    from ml_intraday_v3.live_trading.feature_generator import LiveFeatureGenerator
    
    config_dir = Path("ml_intraday_v3/configs")
    
    feature_gen = LiveFeatureGenerator(
        feature_columns=predictor.feature_columns,
        bar_size='5m',
        features_config_path=config_dir / 'features.yaml',
    )
    
    # Sample 100 random bars (after warmup)
    warmup = 50
    sample_size = min(100, len(bars) - warmup)
    sample_indices = np.random.choice(
        range(warmup, len(bars)),
        size=sample_size,
        replace=False
    )
    sample_indices = sorted(sample_indices)
    
    logger.info(f"Testing on {sample_size} random bars from market data")
    
    predictions = []
    
    for i, idx in enumerate(sample_indices):
        # Get bar window
        bar_window = bars.iloc[max(0, idx - warmup):idx + 1]
        
        if len(bar_window) < 30:
            continue
        
        # Generate features
        features = feature_gen.generate_features(bar_window)
        
        if features.empty:
            continue
        
        # Get prediction
        try:
            pred = predictor.predict(features, use_meta=False, side=None)
            predictions.append({
                'timestamp': bar_window.index[-1],
                'close': bar_window.iloc[-1]['close'],
                'side': pred.get('side', 0),
                'score_ev': pred.get('score_ev', 0.0),
                'score_ev_long': pred.get('score_ev_long', 0.0),
                'score_ev_short': pred.get('score_ev_short', 0.0),
            })
        except Exception as e:
            logger.warning(f"Prediction failed for bar {idx}: {e}")
            continue
    
    if not predictions:
        logger.error("No valid predictions generated!")
        return {'model': model_name, 'error': 'no_predictions'}
    
    df = pd.DataFrame(predictions)
    
    # Statistics
    total = len(df)
    long_count = (df['side'] == 1).sum()
    short_count = (df['side'] == -1).sum()
    skip_count = (df['side'] == 0).sum()
    
    logger.info(f"\nPrediction Distribution on Real Data:")
    logger.info(f"   Total predictions: {total}")
    logger.info(f"   LONG:  {long_count} ({long_count/total*100:.1f}%)")
    logger.info(f"   SHORT: {short_count} ({short_count/total*100:.1f}%)")
    logger.info(f"   SKIP:  {skip_count} ({skip_count/total*100:.1f}%)")
    
    # EV statistics
    logger.info(f"\nEV Score Statistics:")
    logger.info(f"   LONG  - Mean: {df['score_ev_long'].mean():.4f}, Std: {df['score_ev_long'].std():.4f}")
    logger.info(f"   SHORT - Mean: {df['score_ev_short'].mean():.4f}, Std: {df['score_ev_short'].std():.4f}")
    
    # Check for structural issues
    if short_count == 0:
        logger.warning(f"   ⚠️  Model NEVER predicts SHORT on real data!")
    
    if df['score_ev_short'].max() < 0:
        logger.warning(f"   ⚠️  SHORT EV is ALWAYS negative (max={df['score_ev_short'].max():.4f})")
    
    results = {
        'model': model_name,
        'total_predictions': total,
        'long_count': long_count,
        'short_count': short_count,
        'skip_count': skip_count,
        'long_pct': long_count / total * 100,
        'short_pct': short_count / total * 100,
        'ev_long_mean': df['score_ev_long'].mean(),
        'ev_short_mean': df['score_ev_short'].mean(),
        'ev_long_std': df['score_ev_long'].std(),
        'ev_short_std': df['score_ev_short'].std(),
        'predictions': df,
    }
    
    return results


def main():
    logger.info("="*80)
    logger.info("MODEL CAPABILITY VALIDATION")
    logger.info("="*80)
    
    models_dir = Path("ml_intraday_v3/models/saved")
    
    models = [
        ('retrained_clean', models_dir / "model_bundle_retrained_clean.pkl"),
        ('OLD_BASELINE', models_dir / "model_bundle_OLD_BASELINE.pkl"),
    ]
    
    # Load market data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    if data_path.exists():
        logger.info(f"\nLoading market data: {data_path}")
        bars = pd.read_hdf(data_path, key='bars_5min')
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.set_index('timestamp').sort_index()
        
        # Use Dec 2025 for testing
        test_start = pd.Timestamp('2025-12-01', tz='UTC')
        test_end = pd.Timestamp('2025-12-18 23:59:59', tz='UTC')
        bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]
        
        logger.info(f"Test data: {len(bars_test)} bars ({bars_test.index[0].date()} to {bars_test.index[-1].date()})")
    else:
        logger.warning(f"Market data not found: {data_path}")
        bars_test = None
    
    all_results = []
    
    for model_name, model_path in models:
        if not model_path.exists():
            logger.warning(f"\nModel not found: {model_path}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading {model_name}")
        logger.info(f"{'='*80}")
        
        # Load predictor
        predictor = LiveModelPredictor(model_path)
        
        # Test 1: Synthetic features
        synthetic_results = test_synthetic_features(predictor, model_name)
        
        # Test 2: Actual market data
        if bars_test is not None:
            market_results = test_actual_market_data(predictor, model_name, bars_test)
        else:
            market_results = None
        
        all_results.append({
            'model': model_name,
            'synthetic': synthetic_results,
            'market': market_results,
        })
    
    # Final comparison
    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*80}")
    
    for result in all_results:
        model_name = result['model']
        logger.info(f"\n{model_name}:")
        
        synth = result['synthetic']
        logger.info(f"   Synthetic Tests:")
        logger.info(f"      Can predict SHORT: {'✅ YES' if not synth.get('has_short_bias', True) else '❌ NO'}")
        logger.info(f"      Responds to bearish: {'✅ YES' if synth.get('responds_to_bearish', False) else '❌ NO'}")
        
        if result['market']:
            market = result['market']
            logger.info(f"   Real Market Tests:")
            logger.info(f"      LONG predictions: {market['long_pct']:.1f}%")
            logger.info(f"      SHORT predictions: {market['short_pct']:.1f}%")
            logger.info(f"      Avg EV SHORT: {market['ev_short_mean']:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*80}")
    
    for result in all_results:
        model_name = result['model']
        synth = result['synthetic']
        market = result.get('market')
        
        logger.info(f"\n{model_name}:")
        
        if synth.get('has_short_bias', True):
            logger.warning(f"   ❌ NOT READY: Model has structural LONG bias")
            logger.info(f"      → Retrain with balanced LONG/SHORT data")
        elif market and market['short_pct'] < 10:
            logger.warning(f"   ⚠️  CAUTION: Rarely predicts SHORT on real data ({market['short_pct']:.1f}%)")
            logger.info(f"      → Check thresholds or market regime")
        else:
            logger.info(f"   ✅ Model appears capable of bidirectional trading")
            logger.info(f"      → Proceed with full backtest validation")
    
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
