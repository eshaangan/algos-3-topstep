#!/usr/bin/env python3
"""
Diagnose January 2026 Model Failure

Purpose: Understand WHY the model failed on Jan 2026 vs succeeding on 2024-2025

Analysis:
1. Feature distribution shifts (2024-2025 vs Jan 2026)
2. Label distribution shifts (target rates)
3. Volatility regime changes
4. Model probability calibration
5. Prediction confidence analysis
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compare_distributions(df_train, df_test, feature_cols):
    """Compare feature distributions between train and test."""

    results = []

    for feature in feature_cols:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue

        train_vals = df_train[feature].dropna()
        test_vals = df_test[feature].dropna()

        if len(train_vals) == 0 or len(test_vals) == 0:
            continue

        # KS test for distribution difference
        ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)

        # Mean shift
        train_mean = train_vals.mean()
        test_mean = test_vals.mean()
        mean_shift_pct = 100 * (test_mean - train_mean) / (abs(train_mean) + 1e-9)

        # Std shift
        train_std = train_vals.std()
        test_std = test_vals.std()
        std_shift_pct = 100 * (test_std - train_std) / (train_std + 1e-9)

        results.append({
            'feature': feature,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'mean_shift_pct': mean_shift_pct,
            'train_std': train_std,
            'test_std': test_std,
            'std_shift_pct': std_shift_pct,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'significant_shift': ks_pvalue < 0.05
        })

    return pd.DataFrame(results).sort_values('ks_statistic', ascending=False)


def analyze_prediction_confidence(model, scaler, X_test, y_test, feature_cols):
    """Analyze model confidence on test predictions."""

    # Scale features
    X_test_clean = X_test[feature_cols].dropna()
    y_test_clean = y_test[X_test_clean.index]

    X_scaled = scaler.transform(X_test_clean)

    # Get probabilities
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Confidence (distance from 0.5)
    confidence = np.abs(y_proba - 0.5)

    results = pd.DataFrame({
        'y_true': y_test_clean.values,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confidence': confidence,
        'correct': y_test_clean.values == y_pred
    })

    return results


def main():
    """Run diagnostic analysis."""

    project_root = Path(__file__).parent.parent.parent

    # Paths
    features_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "features.parquet"
    events_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "events.parquet"
    weights_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "weights.parquet"

    diagnostics_dir = project_root / "ml_intraday_v3" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")

    # Load data
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)
    weights = pd.read_parquet(weights_path)

    # Merge
    events_with_weights = events.merge(weights[['event_id', 'w_final']], on='event_id')
    events_with_weights['t0'] = pd.to_datetime(events_with_weights['t0'])
    events_with_weights = events_with_weights.set_index('t0').sort_index()

    data = features.join(events_with_weights[['y', 'w_final', 'side', 't1']], how='inner')

    # Filter vertical barriers
    data = data[data['y'] != 0].copy()

    # Core features
    core_features = [
        'log_return_1', 'log_return_4', 'atr_14', 'vol_20', 'vol_regime',
        'ema_13', 'ema_21', 'ema_spread', 'minute_of_day_sin', 'minute_of_day_cos'
    ]
    feature_cols = [f for f in core_features if f in data.columns]

    # Split periods
    train_2024_2025 = data[(data.index >= '2024-01-01') & (data.index < '2026-01-01')]
    test_jan_2026 = data[(data.index >= '2026-01-01') & (data.index < '2026-02-01')]

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET OVERVIEW")
    logger.info(f"{'='*60}")
    logger.info(f"Train (2024-2025): {len(train_2024_2025):,} events")
    logger.info(f"Test (Jan 2026): {len(test_jan_2026):,} events")

    # ===== ANALYSIS 1: Feature Distribution Shifts =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 1: Feature Distribution Shifts")
    logger.info(f"{'='*60}")

    dist_shifts = compare_distributions(train_2024_2025, test_jan_2026, feature_cols)

    logger.info("\nTop 5 features with largest distribution shifts:")
    logger.info(dist_shifts[['feature', 'mean_shift_pct', 'std_shift_pct', 'ks_statistic', 'significant_shift']].head(10).to_string(index=False))

    # Save full results
    dist_shifts.to_csv(diagnostics_dir / "jan2026_feature_distribution_shifts.csv", index=False)

    # ===== ANALYSIS 2: Label Distribution =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 2: Label Distribution Shifts")
    logger.info(f"{'='*60}")

    train_target_rate = (train_2024_2025['y'] == 1).sum() / len(train_2024_2025)
    test_target_rate = (test_jan_2026['y'] == 1).sum() / len(test_jan_2026)

    logger.info(f"Train (2024-2025) target rate: {train_target_rate:.1%}")
    logger.info(f"Test (Jan 2026) target rate: {test_target_rate:.1%}")
    logger.info(f"Shift: {100*(test_target_rate - train_target_rate):.1f} percentage points")

    # ===== ANALYSIS 3: Volatility Regime =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 3: Volatility Regime Shifts")
    logger.info(f"{'='*60}")

    if 'vol_20' in train_2024_2025.columns and 'vol_20' in test_jan_2026.columns:
        train_vol_mean = train_2024_2025['vol_20'].mean()
        test_vol_mean = test_jan_2026['vol_20'].mean()
        vol_shift = 100 * (test_vol_mean - train_vol_mean) / train_vol_mean

        logger.info(f"Train volatility (vol_20): {train_vol_mean:.6f}")
        logger.info(f"Test volatility (vol_20): {test_vol_mean:.6f}")
        logger.info(f"Volatility shift: {vol_shift:+.1f}%")

    # ===== ANALYSIS 4: Retrain and Test Model =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 4: Model Performance on Jan 2026")
    logger.info(f"{'='*60}")

    # Prepare training data
    X_train = train_2024_2025[feature_cols]
    y_train = train_2024_2025['y']
    w_train = train_2024_2025['w_final']

    # Filter NaN
    train_valid = ~X_train.isna().any(axis=1)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    w_train = w_train[train_valid]

    # Prepare test data
    X_test = test_jan_2026[feature_cols]
    y_test = test_jan_2026['y']

    test_valid = ~X_test.isna().any(axis=1)
    X_test = X_test[test_valid]
    y_test = y_test[test_valid]

    if len(X_test) == 0:
        logger.error("No valid test samples in Jan 2026!")
        return

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = np.nan

    logger.info(f"Jan 2026 Accuracy: {acc:.1%}")
    logger.info(f"Jan 2026 AUC: {auc:.3f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Stop (-1)', 'Target (+1)']))

    # ===== ANALYSIS 5: Prediction Confidence =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 5: Prediction Confidence Analysis")
    logger.info(f"{'='*60}")

    conf_results = analyze_prediction_confidence(model, scaler, test_jan_2026, y_test, feature_cols)

    # Accuracy by confidence bins
    conf_results['conf_bin'] = pd.cut(conf_results['confidence'], bins=[0, 0.1, 0.2, 0.3, 0.5], labels=['Low', 'Medium', 'High', 'Very High'])

    logger.info("\nAccuracy by Confidence Level:")
    for conf_level in ['Low', 'Medium', 'High', 'Very High']:
        subset = conf_results[conf_results['conf_bin'] == conf_level]
        if len(subset) > 0:
            subset_acc = subset['correct'].mean()
            logger.info(f"  {conf_level:12s}: {subset_acc:.1%} ({len(subset)} predictions)")

    # Save confidence analysis
    conf_results.to_csv(diagnostics_dir / "jan2026_prediction_confidence.csv", index=False)

    # ===== ANALYSIS 6: Feature Importance Stability =====
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS 6: Feature Importance (Coefficients)")
    logger.info(f"{'='*60}")

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0],
        'abs_coef': np.abs(model.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    logger.info("\nTop 5 Most Important Features:")
    logger.info(feature_importance[['feature', 'coefficient']].head().to_string(index=False))

    # ===== SUMMARY =====
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*60}")

    # Count significant shifts
    n_significant_shifts = dist_shifts['significant_shift'].sum()
    pct_shifted = 100 * n_significant_shifts / len(dist_shifts)

    logger.info(f"\n1. FEATURE DISTRIBUTION:")
    logger.info(f"   - {n_significant_shifts}/{len(dist_shifts)} features ({pct_shifted:.1f}%) have significant distribution shifts")
    logger.info(f"   - Largest shift: {dist_shifts.iloc[0]['feature']} (KS={dist_shifts.iloc[0]['ks_statistic']:.3f})")

    logger.info(f"\n2. LABEL DISTRIBUTION:")
    logger.info(f"   - Target rate shifted {100*(test_target_rate - train_target_rate):+.1f} percentage points")

    logger.info(f"\n3. VOLATILITY:")
    if 'vol_20' in train_2024_2025.columns:
        logger.info(f"   - Volatility shifted {vol_shift:+.1f}%")

    logger.info(f"\n4. MODEL PERFORMANCE:")
    logger.info(f"   - Jan 2026 accuracy: {acc:.1%} (vs 80.3% in walk-forward)")
    logger.info(f"   - Jan 2026 AUC: {auc:.3f} (vs 0.76 in walk-forward)")

    logger.info(f"\n5. PREDICTION CONFIDENCE:")
    low_conf = conf_results[conf_results['conf_bin'] == 'Low']
    high_conf = conf_results[conf_results['conf_bin'].isin(['High', 'Very High'])]
    if len(low_conf) > 0:
        logger.info(f"   - Low confidence predictions: {len(low_conf)} ({100*len(low_conf)/len(conf_results):.1f}%)")
    if len(high_conf) > 0:
        logger.info(f"   - High confidence predictions: {len(high_conf)} ({100*len(high_conf)/len(conf_results):.1f}%)")
        logger.info(f"   - High confidence accuracy: {high_conf['correct'].mean():.1%}")

    logger.info(f"\n{'='*60}")
    logger.info("ARTIFACTS SAVED:")
    logger.info(f"{'='*60}")
    logger.info(f"  - {diagnostics_dir / 'jan2026_feature_distribution_shifts.csv'}")
    logger.info(f"  - {diagnostics_dir / 'jan2026_prediction_confidence.csv'}")


if __name__ == "__main__":
    main()
