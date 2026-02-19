"""
Phase 4: Final Model Preparation

Packages the selected model for deployment with all necessary artifacts:
- Trained model file (.pkl)
- Feature columns list
- Configuration
- Performance report
- Validation results

Creates a production-ready model bundle for live trading.
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.comprehensive_grid_search_v2 import (
    load_and_prepare_data,
    build_features,
    generate_labels
)


def select_final_model(validation_results_file: str) -> str:
    """Select the best model from Phase 2 validation results."""
    
    print("=" * 80)
    print("SELECTING FINAL MODEL")
    print("=" * 80)
    
    with open(validation_results_file) as f:
        validation_results = json.load(f)
    
    # Filter to models that passed
    passed_models = [r for r in validation_results if r['passed']]
    
    if len(passed_models) == 0:
        raise ValueError("No models passed validation! Cannot proceed.")
    
    print(f"\n{len(passed_models)} models passed validation")
    
    # Rank by composite score: 40% PnL + 25% Sharpe + 15% Win Rate + 10% AUC + 10% Max DD
    for model in passed_models:
        # Normalize metrics to 0-1
        pnl_norm = model['total_pnl'] / max(r['total_pnl'] for r in passed_models)
        sharpe_norm = model['sharpe_ratio'] / max(r['sharpe_ratio'] for r in passed_models)
        wr_norm = model['win_rate'] / max(r['win_rate'] for r in passed_models)
        # Max DD is negative, so invert (less negative = better)
        dd_norm = 1.0 - (abs(model['max_drawdown']) / max(abs(r['max_drawdown']) for r in passed_models))
        
        model['deployment_score'] = (
            0.40 * pnl_norm +
            0.25 * sharpe_norm +
            0.15 * wr_norm +
            0.10 * dd_norm
        )
    
    # Sort by deployment score
    passed_models.sort(key=lambda x: x['deployment_score'], reverse=True)
    
    # Select best model
    best_model = passed_models[0]
    
    print(f"\n✅ Selected model: {best_model['exp_id']}")
    print(f"   Deployment Score: {best_model['deployment_score']:.3f}")
    print(f"   PnL: ${best_model['total_pnl']:.2f}")
    print(f"   Sharpe: {best_model['sharpe_ratio']:.2f}")
    print(f"   Win Rate: {100*best_model['win_rate']:.1f}%")
    print(f"   Max DD: ${best_model['max_drawdown']:.2f}")
    print(f"   Trades: {best_model['n_trades']}")
    
    if len(passed_models) > 1:
        print(f"\nOther candidates:")
        for i, model in enumerate(passed_models[1:4], start=2):  # Show top 4 total
            print(f"   {i}. {model['exp_id']} (Score: {model['deployment_score']:.3f}, PnL: ${model['total_pnl']:.2f})")
    
    return best_model['exp_id']


def train_final_model(exp_id: str, results_dir: str, training_data_path: str, configs_dir: str) -> Dict:
    """Train final model on full training dataset."""
    
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    # Load experiment config
    result_file = None
    for batch_dir in Path(results_dir).glob("batch*"):
        candidate = batch_dir / f"result_{exp_id}.json"
        if candidate.exists():
            result_file = candidate
            break
    
    if result_file is None:
        raise FileNotFoundError(f"Could not find result file for {exp_id}")
    
    with open(result_file) as f:
        exp_data = json.load(f)
        exp_config = exp_data['config']
        cv_results = exp_data.get('cv_results', {})
    
    print(f"\nConfiguration:")
    print(f"  Labeling: {exp_config['labeling_method']}")
    print(f"  CV: {exp_config['cv_method']}")
    print(f"  Sample Weight: {exp_config['sample_weight']}")
    print(f"  Calibration: {exp_config.get('calibration', 'none')}")
    print(f"  Feature Set: {exp_config.get('feature_set_name', 'baseline')}")
    
    # Load full training data
    print(f"\nLoading training data: {training_data_path}")
    df_train = load_and_prepare_data(training_data_path)
    
    # Build features
    print("Building features...")
    df_train = build_features(df_train, exp_config.get('features_config', {}), configs_dir)
    
    # Generate labels
    print("Generating labels...")
    df_train = generate_labels(
        df_train,
        labeling_method=exp_config['labeling_method'],
        labeling_params=exp_config.get('labeling_params', {}),
        configs_dir=configs_dir
    )
    
    # Prepare training data
    feature_cols = [c for c in df_train.columns if c.startswith('f_')]
    X = df_train[feature_cols].values
    y = df_train['y'].values
    
    print(f"\nTraining data:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Sample weights
    if exp_config.get('sample_weight') == 'uniqueness_decay':
        # TODO: Implement proper uniqueness weighting
        weights = np.ones(len(y))
    elif exp_config.get('sample_weight') == 'uniform':
        weights = np.ones(len(y))
    else:
        weights = np.ones(len(y))
    
    # Train model
    print("\nTraining LightGBM model...")
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    model_params = exp_config.get('model_params', {})
    model = LGBMClassifier(
        objective='binary',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **model_params
    )
    
    model.fit(X, y, sample_weight=weights)
    print("  ✅ Model trained")
    
    # Apply calibration if specified
    calibration_method = exp_config.get('calibration', 'none')
    if calibration_method in ['isotonic', 'sigmoid']:
        print(f"\nApplying {calibration_method} calibration...")
        model = CalibratedClassifierCV(model, method=calibration_method, cv='prefit')
        model.fit(X, y)
        print("  ✅ Calibration applied")
    
    # Create model bundle
    model_bundle = {
        'model': model,
        'feature_columns': feature_cols,
        'exp_id': exp_id,
        'exp_config': exp_config,
        'cv_results': cv_results,
        'training_metadata': {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'label_distribution': np.bincount(y).tolist(),
            'trained_date': datetime.now().isoformat(),
            'training_data_path': training_data_path,
        }
    }
    
    return model_bundle


def generate_performance_report(
    model_bundle: Dict,
    validation_results_file: str,
    output_dir: Path
) -> Dict:
    """Generate comprehensive performance report."""
    
    print("\n" + "=" * 80)
    print("GENERATING PERFORMANCE REPORT")
    print("=" * 80)
    
    exp_id = model_bundle['exp_id']
    cv_results = model_bundle['cv_results']
    
    # Load validation results
    with open(validation_results_file) as f:
        validation_results = json.load(f)
    
    val_result = next((r for r in validation_results if r['exp_id'] == exp_id), None)
    if val_result is None:
        raise ValueError(f"Could not find validation results for {exp_id}")
    
    # Compile report
    report = {
        'model_id': exp_id,
        'generation_date': datetime.now().isoformat(),
        
        # Configuration
        'configuration': {
            'labeling_method': model_bundle['exp_config']['labeling_method'],
            'cv_method': model_bundle['exp_config']['cv_method'],
            'sample_weight': model_bundle['exp_config']['sample_weight'],
            'calibration': model_bundle['exp_config'].get('calibration', 'none'),
            'feature_set': model_bundle['exp_config'].get('feature_set_name', 'baseline'),
            'model_params': model_bundle['exp_config'].get('model_params', {}),
        },
        
        # Cross-validation performance
        'cv_performance': {
            'median_test_auc': cv_results.get('median_test_auc', 0),
            'median_train_auc': cv_results.get('median_train_auc', 0),
            'std_test_auc': cv_results.get('std_test_auc', 0),
            'median_brier': cv_results.get('median_brier', 1.0),
            'median_pr_auc': cv_results.get('median_pr_auc', 0),
        },
        
        # Signal quality
        'signal_quality': {
            'pct_signals_055': cv_results.get('median_pct_signals_above_055', 0),
            'pct_signals_060': cv_results.get('median_pct_signals_above_060', 0),
            'est_trades_per_day': cv_results.get('median_est_trades_per_day', 0),
        },
        
        # True OOS validation
        'oos_validation': {
            'total_pnl': val_result['total_pnl'],
            'n_trades': val_result['n_trades'],
            'win_rate': val_result['win_rate'],
            'avg_win': val_result['avg_win'],
            'avg_loss': val_result['avg_loss'],
            'profit_factor': val_result['profit_factor'],
            'sharpe_ratio': val_result['sharpe_ratio'],
            'max_drawdown': val_result['max_drawdown'],
            'passed': val_result['passed'],
        },
        
        # Training metadata
        'training_metadata': model_bundle['training_metadata'],
        
        # Feature information
        'features': {
            'n_features': len(model_bundle['feature_columns']),
            'feature_columns': model_bundle['feature_columns'],
        }
    }
    
    # Save report
    report_file = output_dir / f"{exp_id}_performance_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Performance report saved to: {report_file}")
    
    # Also save human-readable version
    readme_file = output_dir / f"{exp_id}_README.md"
    with open(readme_file, 'w') as f:
        f.write(f"# Model Performance Report: {exp_id}\n\n")
        f.write(f"**Generated**: {report['generation_date']}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Labeling**: {report['configuration']['labeling_method']}\n")
        f.write(f"- **CV Method**: {report['configuration']['cv_method']}\n")
        f.write(f"- **Sample Weight**: {report['configuration']['sample_weight']}\n")
        f.write(f"- **Calibration**: {report['configuration']['calibration']}\n")
        f.write(f"- **Feature Set**: {report['configuration']['feature_set']}\n\n")
        
        f.write("## Cross-Validation Performance\n\n")
        f.write(f"- **Median Test AUC**: {report['cv_performance']['median_test_auc']:.3f}\n")
        f.write(f"- **Std Test AUC**: {report['cv_performance']['std_test_auc']:.3f}\n")
        f.write(f"- **Median Brier**: {report['cv_performance']['median_brier']:.3f}\n\n")
        
        f.write("## Signal Quality\n\n")
        f.write(f"- **Signals >0.55**: {100*report['signal_quality']['pct_signals_055']:.1f}%\n")
        f.write(f"- **Est. Trades/Day**: {report['signal_quality']['est_trades_per_day']:.1f}\n\n")
        
        f.write("## True Out-of-Sample Validation (Feb 11 - Mar 31, 2026)\n\n")
        f.write(f"- **Total PnL**: ${report['oos_validation']['total_pnl']:.2f}\n")
        f.write(f"- **Win Rate**: {100*report['oos_validation']['win_rate']:.1f}%\n")
        f.write(f"- **Sharpe Ratio**: {report['oos_validation']['sharpe_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown**: ${report['oos_validation']['max_drawdown']:.2f}\n")
        f.write(f"- **Profit Factor**: {report['oos_validation']['profit_factor']:.2f}\n")
        f.write(f"- **Number of Trades**: {report['oos_validation']['n_trades']}\n")
        f.write(f"- **Passed Validation**: {'✅ YES' if report['oos_validation']['passed'] else '❌ NO'}\n\n")
        
        f.write("## Training Information\n\n")
        f.write(f"- **Training Samples**: {report['training_metadata']['n_samples']:,}\n")
        f.write(f"- **Features**: {report['training_metadata']['n_features']}\n")
        f.write(f"- **Trained Date**: {report['training_metadata']['trained_date']}\n\n")
        
        f.write("## Deployment Checklist\n\n")
        f.write("- [ ] Model loads correctly in live_trading/model_predictor.py\n")
        f.write("- [ ] Features match between training and live\n")
        f.write("- [ ] Prediction outputs are calibrated probabilities\n")
        f.write("- [ ] Confidence threshold (0.55) is appropriate\n")
        f.write("- [ ] Risk management parameters set\n")
        f.write("- [ ] Topstep rules enforced\n")
        f.write("- [ ] Paper trading completed successfully (1-2 weeks)\n")
    
    print(f"✅ README saved to: {readme_file}")
    
    return report


def save_model_bundle(model_bundle: Dict, output_dir: Path):
    """Save complete model bundle for deployment."""
    
    print("\n" + "=" * 80)
    print("SAVING MODEL BUNDLE")
    print("=" * 80)
    
    exp_id = model_bundle['exp_id']
    
    # Save model
    model_file = output_dir / f"{exp_id}_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"\n✅ Model bundle saved to: {model_file}")
    print(f"   Size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save feature columns separately for easy access
    features_file = output_dir / f"{exp_id}_features.json"
    with open(features_file, 'w') as f:
        json.dump(model_bundle['feature_columns'], f, indent=2)
    
    print(f"✅ Feature columns saved to: {features_file}")
    
    # Save config separately
    config_file = output_dir / f"{exp_id}_config.json"
    with open(config_file, 'w') as f:
        json.dump(model_bundle['exp_config'], f, indent=2)
    
    print(f"✅ Configuration saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Prepare final model for deployment")
    parser.add_argument('--validation-results', type=str, required=True,
                        help='Path to Phase 2 validation results JSON')
    parser.add_argument('--training-data', type=str, required=True,
                        help='Path to full training data (Oct 2024 - Dec 2025)')
    parser.add_argument('--results-dir', type=str, default='ml_intraday_v3/experiments/results',
                        help='Directory containing experiment results')
    parser.add_argument('--configs-dir', type=str, default='ml_intraday_v3/configs',
                        help='Directory containing config files')
    parser.add_argument('--output-dir', type=str, default='ml_intraday_v3/models/final',
                        help='Directory to save final model artifacts')
    parser.add_argument('--model-id', type=str, default=None,
                        help='Specific model ID to prepare (if not selecting from validation)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 4: FINAL MODEL PREPARATION")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select final model (or use specified model)
    if args.model_id:
        exp_id = args.model_id
        print(f"\nUsing specified model: {exp_id}")
    else:
        exp_id = select_final_model(args.validation_results)
    
    # Train final model
    model_bundle = train_final_model(
        exp_id,
        args.results_dir,
        args.training_data,
        args.configs_dir
    )
    
    # Generate performance report
    report = generate_performance_report(
        model_bundle,
        args.validation_results,
        output_dir
    )
    
    # Save model bundle
    save_model_bundle(model_bundle, output_dir)
    
    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)
    
    print("\n✅ Final model prepared for deployment!")
    print(f"\nModel artifacts saved to: {output_dir}/")
    print(f"\nNext steps:")
    print("1. Review performance report and README")
    print("2. Integrate model with live_trading/model_predictor.py")
    print("3. Run integration tests")
    print("4. Begin Phase 5: Paper trading (1-2 weeks)")
    print("5. Monitor performance and adjust if needed")
    print("6. Phase 6: Live deployment on Topstep combine")
    
    print(f"\n🎯 Model ready for paper trading: {exp_id}")


if __name__ == '__main__':
    main()
