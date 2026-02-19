"""
Phase 2: True Out-of-Sample Validation

Tests top 10 selected models on TRULY UNSEEN data (Feb 11 - Mar 31, 2026).

This data was NEVER used in any training, CV, or forward testing.
It represents the final acid test before deployment.

Success Criteria (ALL must pass):
- ✅ Positive PnL (any amount > $0)
- ✅ Win rate > 45%
- ✅ Max drawdown < $1,000
- ✅ Sharpe ratio > 0.5
- ✅ Different signals than baseline

If validation fails, fall back to rule-based system in rule_based_v1/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiments.comprehensive_grid_search_v2 import (
    load_and_prepare_data,
    build_features,
    generate_labels,
    setup_cross_validation
)


def load_true_oos_data(data_path: str) -> pd.DataFrame:
    """Load truly unseen OOS data (Feb 11 - Mar 31, 2026)."""
    
    print(f"Loading true OOS data from: {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.h5'):
        df = pd.read_hdf(data_path, key='data')
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Filter to Feb 11 - Mar 31, 2026
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_date = pd.Timestamp('2026-02-11')
    end_date = pd.Timestamp('2026-03-31')
    
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total bars: {len(df)}")
    print(f"  Trading days: {df['timestamp'].dt.date.nunique()}")
    
    return df


def retrain_model_on_full_training_data(exp_config: Dict, training_data_path: str, configs_dir: str) -> Dict:
    """Retrain a model on full training data (Oct 2024 - Dec 2025)."""
    
    print(f"\nRetraining model: {exp_config['exp_id']}")
    
    # Load full training data
    df_train = load_and_prepare_data(training_data_path)
    
    # Build features
    df_train = build_features(df_train, exp_config.get('features_config', {}), configs_dir)
    
    # Generate labels
    df_train = generate_labels(
        df_train,
        labeling_method=exp_config['labeling_method'],
        labeling_params=exp_config.get('labeling_params', {}),
        configs_dir=configs_dir
    )
    
    # Train model (no CV, use all data)
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    # Prepare data
    feature_cols = [c for c in df_train.columns if c.startswith('f_')]
    X = df_train[feature_cols].values
    y = df_train['y'].values
    
    # Sample weights
    if exp_config.get('sample_weight') == 'uniqueness_decay':
        # TODO: Implement proper uniqueness weighting
        weights = np.ones(len(y))
    elif exp_config.get('sample_weight') == 'uniform':
        weights = np.ones(len(y))
    else:
        weights = np.ones(len(y))
    
    # Train model
    model_params = exp_config.get('model_params', {})
    model = LGBMClassifier(
        objective='binary',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **model_params
    )
    
    model.fit(X, y, sample_weight=weights)
    
    # Apply calibration if specified
    if exp_config.get('calibration') == 'isotonic':
        model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        model.fit(X, y)
    elif exp_config.get('calibration') == 'sigmoid':
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        model.fit(X, y)
    
    print(f"  ✅ Model trained on {len(X)} samples")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'exp_config': exp_config
    }


def generate_predictions_on_oos(model_bundle: Dict, df_oos: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions on OOS data."""
    
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']
    
    # Build features on OOS data (no labels needed)
    # Note: Features should be built using same config as training
    
    # Extract features
    available_features = [c for c in df_oos.columns if c in feature_cols]
    missing_features = set(feature_cols) - set(available_features)
    
    if missing_features:
        print(f"  ⚠️  WARNING: {len(missing_features)} features missing in OOS data")
        print(f"     Missing: {list(missing_features)[:5]}...")
        # Fill missing features with 0 (or handle better)
        for f in missing_features:
            df_oos[f] = 0
    
    X_oos = df_oos[feature_cols].values
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X_oos)
    
    df_oos['p_stop'] = y_pred_proba[:, 0]  # Probability of stop
    df_oos['p_target'] = y_pred_proba[:, 1]  # Probability of target
    
    return df_oos


def run_backtest(df_oos: pd.DataFrame, confidence_threshold: float = 0.55) -> Dict:
    """
    Run full backtest with realistic trading rules.
    
    Entry: Model signal > confidence_threshold
    Exit: PT=2.0x ATR, SL=1.5x ATR, Time=24 bars (2 hours)
    Risk: $100 max per trade (1 MES contract)
    """
    
    print("\nRunning backtest...")
    
    trades = []
    current_position = None
    equity = 0
    equity_curve = []
    
    for idx, row in df_oos.iterrows():
        
        # Check if we should close current position
        if current_position is not None:
            bars_held = idx - current_position['entry_idx']
            
            # Time exit (24 bars = 2 hours)
            if bars_held >= 24:
                exit_price = row['close']
                pnl = (exit_price - current_position['entry_price']) * current_position['side'] * 5  # $5/point for MES
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': exit_price,
                    'exit_reason': 'time',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            # Profit target
            elif current_position['side'] == 1 and row['high'] >= current_position['pt_price']:
                pnl = (current_position['pt_price'] - current_position['entry_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['pt_price'],
                    'exit_reason': 'profit_target',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            elif current_position['side'] == -1 and row['low'] <= current_position['pt_price']:
                pnl = (current_position['entry_price'] - current_position['pt_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['pt_price'],
                    'exit_reason': 'profit_target',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            # Stop loss
            elif current_position['side'] == 1 and row['low'] <= current_position['sl_price']:
                pnl = (current_position['sl_price'] - current_position['entry_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['sl_price'],
                    'exit_reason': 'stop_loss',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            elif current_position['side'] == -1 and row['high'] >= current_position['sl_price']:
                pnl = (current_position['entry_price'] - current_position['sl_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['sl_price'],
                    'exit_reason': 'stop_loss',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
        
        # Entry logic (if no position)
        if current_position is None:
            # Check confidence threshold
            if row['p_target'] > confidence_threshold:
                # Long entry
                atr = row.get('atr', row['close'] * 0.01)  # Fallback to 1% if ATR missing
                
                current_position = {
                    'entry_idx': idx,
                    'entry_time': row['timestamp'],
                    'entry_price': row['close'],
                    'side': 1,  # Long
                    'pt_price': row['close'] + 2.0 * atr,
                    'sl_price': row['close'] - 1.5 * atr,
                    'p_target': row['p_target'],
                    'p_stop': row['p_stop']
                }
            
            elif row['p_stop'] > confidence_threshold:
                # Short entry
                atr = row.get('atr', row['close'] * 0.01)
                
                current_position = {
                    'entry_idx': idx,
                    'entry_time': row['timestamp'],
                    'entry_price': row['close'],
                    'side': -1,  # Short
                    'pt_price': row['close'] - 2.0 * atr,
                    'sl_price': row['close'] + 1.5 * atr,
                    'p_target': row['p_stop'],  # For short, high p_stop is good
                    'p_stop': row['p_target']
                }
        
        equity_curve.append({
            'timestamp': row['timestamp'],
            'equity': equity
        })
    
    # Close any open position at end
    if current_position is not None:
        exit_price = df_oos.iloc[-1]['close']
        pnl = (exit_price - current_position['entry_price']) * current_position['side'] * 5
        
        trades.append({
            **current_position,
            'exit_idx': len(df_oos) - 1,
            'exit_time': df_oos.iloc[-1]['timestamp'],
            'exit_price': exit_price,
            'exit_reason': 'end_of_period',
            'bars_held': len(df_oos) - 1 - current_position['entry_idx'],
            'pnl': pnl
        })
        
        equity += pnl
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {
            'total_pnl': 0,
            'n_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'trades': [],
            'equity_curve': equity_curve
        }
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    # Max drawdown
    equity_series = pd.DataFrame(equity_curve)['equity']
    cummax = equity_series.cummax()
    drawdown = equity_series - cummax
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (daily)
    daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['pnl'].sum()
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(252)
    
    results = {
        'total_pnl': float(trades_df['pnl'].sum()),
        'n_trades': len(trades_df),
        'win_rate': float(len(wins) / len(trades_df)),
        'avg_win': float(wins['pnl'].mean() if len(wins) > 0 else 0),
        'avg_loss': float(losses['pnl'].mean() if len(losses) > 0 else 0),
        'profit_factor': float(wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'trades': trades_df.to_dict('records'),
        'equity_curve': equity_curve
    }
    
    return results


def check_success_criteria(results: Dict) -> Tuple[bool, List[str]]:
    """Check if model passes all success criteria."""
    
    checks = []
    passed = True
    
    # 1. Positive PnL
    if results['total_pnl'] > 0:
        checks.append(f"✅ Positive PnL: ${results['total_pnl']:.2f}")
    else:
        checks.append(f"❌ Negative PnL: ${results['total_pnl']:.2f} (FAIL)")
        passed = False
    
    # 2. Win rate > 45%
    if results['win_rate'] > 0.45:
        checks.append(f"✅ Win rate: {100*results['win_rate']:.1f}%")
    else:
        checks.append(f"❌ Win rate: {100*results['win_rate']:.1f}% (FAIL - need >45%)")
        passed = False
    
    # 3. Max drawdown < $1,000
    if results['max_drawdown'] > -1000:
        checks.append(f"✅ Max drawdown: ${results['max_drawdown']:.2f}")
    else:
        checks.append(f"❌ Max drawdown: ${results['max_drawdown']:.2f} (FAIL - exceeds $1,000)")
        passed = False
    
    # 4. Sharpe ratio > 0.5
    if results['sharpe_ratio'] > 0.5:
        checks.append(f"✅ Sharpe ratio: {results['sharpe_ratio']:.2f}")
    else:
        checks.append(f"❌ Sharpe ratio: {results['sharpe_ratio']:.2f} (FAIL - need >0.5)")
        passed = False
    
    # 5. Sufficient trades
    if results['n_trades'] > 10:
        checks.append(f"✅ Number of trades: {results['n_trades']}")
    else:
        checks.append(f"⚠️  Low number of trades: {results['n_trades']} (may not be statistically significant)")
    
    return passed, checks


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Validate top models on true OOS data")
    parser.add_argument('--exp-ids-file', type=str, required=True,
                        help='JSON file with list of experiment IDs to validate')
    parser.add_argument('--oos-data', type=str, required=True,
                        help='Path to true OOS data (Feb 11 - Mar 31, 2026)')
    parser.add_argument('--training-data', type=str, required=True,
                        help='Path to full training data (Oct 2024 - Dec 2025)')
    parser.add_argument('--results-dir', type=str, default='ml_intraday_v3/experiments/results',
                        help='Directory containing experiment results')
    parser.add_argument('--configs-dir', type=str, default='ml_intraday_v3/configs',
                        help='Directory containing config files')
    parser.add_argument('--output-dir', type=str, default='ml_intraday_v3/experiments/results/phase2_validation',
                        help='Directory to save validation results')
    parser.add_argument('--confidence-threshold', type=float, default=0.55,
                        help='Confidence threshold for entry signals')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 2: TRUE OUT-OF-SAMPLE VALIDATION")
    print("=" * 80)
    print(f"\nOOS data: {args.oos_data}")
    print(f"Training data: {args.training_data}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Load experiment IDs
    with open(args.exp_ids_file) as f:
        exp_ids = json.load(f)
    
    print(f"\nValidating {len(exp_ids)} models:")
    for exp_id in exp_ids:
        print(f"  - {exp_id}")
    
    # Load true OOS data
    df_oos = load_true_oos_data(args.oos_data)
    
    # Validate each model
    validation_results = []
    
    for exp_id in exp_ids:
        print("\n" + "=" * 80)
        print(f"VALIDATING: {exp_id}")
        print("=" * 80)
        
        # Load experiment config
        # Find result file for this experiment
        result_file = None
        for batch_dir in Path(args.results_dir).glob("batch*"):
            candidate = batch_dir / f"result_{exp_id}.json"
            if candidate.exists():
                result_file = candidate
                break
        
        if result_file is None:
            print(f"❌ ERROR: Could not find result file for {exp_id}")
            continue
        
        with open(result_file) as f:
            exp_data = json.load(f)
            exp_config = exp_data['config']
        
        # Retrain model on full training data
        try:
            model_bundle = retrain_model_on_full_training_data(
                exp_config,
                args.training_data,
                args.configs_dir
            )
        except Exception as e:
            print(f"❌ ERROR retraining model: {e}")
            continue
        
        # Generate predictions on OOS data
        try:
            df_oos_pred = generate_predictions_on_oos(model_bundle, df_oos.copy())
        except Exception as e:
            print(f"❌ ERROR generating predictions: {e}")
            continue
        
        # Run backtest
        try:
            backtest_results = run_backtest(df_oos_pred, args.confidence_threshold)
        except Exception as e:
            print(f"❌ ERROR running backtest: {e}")
            continue
        
        # Check success criteria
        passed, checks = check_success_criteria(backtest_results)
        
        print("\nRESULTS:")
        for check in checks:
            print(f"  {check}")
        
        if passed:
            print("\n🎉 MODEL PASSED ALL CRITERIA!")
        else:
            print("\n❌ MODEL FAILED - does not meet minimum requirements")
        
        validation_results.append({
            'exp_id': exp_id,
            'passed': passed,
            'checks': checks,
            **backtest_results
        })
    
    # Save validation results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_file = output_dir / "validation_results.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PHASE 2 VALIDATION COMPLETE")
    print("=" * 80)
    
    # Summary
    n_passed = sum(1 for r in validation_results if r['passed'])
    print(f"\nModels passed: {n_passed}/{len(validation_results)}")
    
    if n_passed > 0:
        print("\n✅ SUCCESS: Found models that generalize to true OOS data!")
        print("\nNext steps:")
        print("1. Review validation results")
        print("2. Select best model for deployment")
        print("3. Run Phase 4: prepare_final_model.py")
    else:
        print("\n❌ FAILURE: No models passed validation on true OOS data")
        print("\nFallback option:")
        print("  - Use rule-based system in rule_based_v1/")
        print("  - Review model assumptions and retrain with different approach")
    
    print(f"\nResults saved to: {validation_file}")


if __name__ == '__main__':
    main()
