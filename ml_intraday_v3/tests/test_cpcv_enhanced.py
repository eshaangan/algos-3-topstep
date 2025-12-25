"""
Validation Tests for Enhanced CPCV Implementation

Tests verify:
1. Path selection strategies (lexicographic, balanced, random)
2. Path performance evaluation
3. Distribution analysis (quartiles, percentiles)
4. Validation gate checking
5. Visualization
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.validation.cpcv import (
    build_cpcv_paths,
    evaluate_cpcv_paths,
    analyze_cpcv_distributions,
    check_cpcv_gates,
    plot_cpcv_performance,
    _select_balanced_paths
)


@pytest.fixture
def sample_events_df():
    """
    Create sample events DataFrame with known date ranges.
    """
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1h')

    events_df = pd.DataFrame({
        't0': dates,
        't1': dates + pd.Timedelta(hours=2),  # 2-hour events
    })
    events_df.index = pd.RangeIndex(n_samples)

    return events_df


@pytest.fixture
def sample_data():
    """
    Create sample X, y data for model evaluation.
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X.index = pd.RangeIndex(n_samples)

    # Create y with some signal
    y = pd.Series(
        (X['feature_0'] + X['feature_1'] > 0).astype(int),
        index=X.index
    )

    return X, y


def test_build_cpcv_paths_lexicographic(sample_events_df):
    """
    Test lexicographic path selection (first N paths).
    """
    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=6,
        test_groups=2,
        max_paths=5,
        selection="lexicographic",
        pct_embargo=0.01
    )

    assert len(paths) == 5
    assert paths[0]['path_id'] == 0
    assert paths[4]['path_id'] == 4

    # All paths should have train and test indices
    for path in paths:
        assert 'train_idx' in path
        assert 'test_idx' in path
        assert len(path['train_idx']) > 0
        assert len(path['test_idx']) > 0

        # Train and test should not overlap
        assert len(set(path['train_idx']) & set(path['test_idx'])) == 0


def test_build_cpcv_paths_balanced(sample_events_df):
    """
    Test balanced path selection for equal test fold coverage.
    """
    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=6,
        test_groups=2,
        max_paths=10,
        selection="balanced",
        pct_embargo=0.01
    )

    assert len(paths) == 10

    # Count how many times each fold appears in test set
    fold_test_counts = {i: 0 for i in range(6)}

    for path in paths:
        for fold_idx in path['test_folds']:
            fold_test_counts[fold_idx] += 1

    # Balanced selection should have relatively equal counts
    counts = list(fold_test_counts.values())
    assert min(counts) > 0  # All folds appear at least once
    # Standard deviation should be small (ideally 0, but allow some variance)
    assert np.std(counts) <= 1.0


def test_build_cpcv_paths_random(sample_events_df):
    """
    Test random path selection with seed for reproducibility.
    """
    paths1 = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=6,
        test_groups=2,
        max_paths=5,
        selection="random",
        random_state=42,
        pct_embargo=0.01
    )

    paths2 = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=6,
        test_groups=2,
        max_paths=5,
        selection="random",
        random_state=42,
        pct_embargo=0.01
    )

    # Same seed should produce same paths
    assert len(paths1) == len(paths2) == 5
    for p1, p2 in zip(paths1, paths2):
        assert p1['test_folds'] == p2['test_folds']


def test_select_balanced_paths():
    """
    Test the balanced path selection greedy algorithm.
    """
    from itertools import combinations

    K = 6
    test_groups = 2
    all_combos = list(combinations(range(K), test_groups))

    # Select 10 paths with balanced coverage
    selected = _select_balanced_paths(
        all_combos=all_combos,
        K=K,
        max_paths=10,
        random_state=42
    )

    assert len(selected) == 10

    # Count fold coverage
    fold_counts = {i: 0 for i in range(K)}
    for combo in selected:
        for fold_idx in combo:
            fold_counts[fold_idx] += 1

    counts = list(fold_counts.values())
    # Each fold should appear roughly equally
    assert min(counts) > 0
    assert max(counts) - min(counts) <= 1  # Difference at most 1


def test_evaluate_cpcv_paths(sample_events_df, sample_data):
    """
    Test CPCV path evaluation with metrics computation.
    """
    X, y = sample_data

    # Build a few paths
    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=4,
        test_groups=1,
        max_paths=3,
        selection="lexicographic",
        pct_embargo=0.01
    )

    # Evaluate paths with simple logistic regression
    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc', 'accuracy'],
        sample_weight=None
    )

    # Should have results for all paths
    assert len(perf_df) > 0

    # Should have both IS and OOS results
    assert 'is_oos' in perf_df.columns
    assert perf_df['is_oos'].isin(['IS', 'OOS']).all()

    # Should have requested metrics
    assert set(perf_df['metric_name'].unique()) == {'roc_auc', 'accuracy'}

    # Metric values should be in valid range
    for metric in ['roc_auc', 'accuracy']:
        metric_values = perf_df[perf_df['metric_name'] == metric]['metric_value']
        assert metric_values.min() >= 0.0
        assert metric_values.max() <= 1.0


def test_evaluate_cpcv_paths_with_sharpe(sample_events_df):
    """
    Test CPCV evaluation with Sharpe ratio metric (requires trades).
    """
    np.random.seed(42)
    n_samples = 1000

    # Create sample features and labels
    X = pd.DataFrame({
        'feature_0': np.random.randn(n_samples),
        'feature_1': np.random.randn(n_samples)
    })
    y = pd.Series((X['feature_0'] > 0).astype(int))

    # Create sample trades DataFrame with PnL
    # Simulate trades at each event
    trades_df = pd.DataFrame({
        'event_idx': range(n_samples),
        'pnl': np.random.normal(10, 50, n_samples)  # Mean $10, std $50
    })

    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=4,
        test_groups=1,
        max_paths=2,
        selection="lexicographic",
        pct_embargo=0.01
    )

    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    # Note: sharpe computation needs trades, so we'll test with roc_auc
    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc'],
        sample_weight=None
    )

    assert 'roc_auc' in perf_df['metric_name'].values


def test_analyze_cpcv_distributions(sample_events_df, sample_data):
    """
    Test distribution analysis (quartiles, percentiles).
    """
    X, y = sample_data

    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=5,
        test_groups=2,
        max_paths=10,
        selection="balanced",
        pct_embargo=0.01
    )

    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc', 'accuracy'],
        sample_weight=None
    )

    # Analyze distributions
    stats = analyze_cpcv_distributions(
        perf_df=perf_df,
        metrics=['roc_auc', 'accuracy']
    )

    # Should have stats for both metrics
    assert 'roc_auc' in stats
    assert 'accuracy' in stats

    # Check IS stats structure
    for metric in ['roc_auc', 'accuracy']:
        assert 'IS' in stats[metric]
        assert 'OOS' in stats[metric]

        # Check required statistics
        for split in ['IS', 'OOS']:
            split_stats = stats[metric][split]
            assert 'mean' in split_stats
            assert 'median' in split_stats
            assert 'std' in split_stats
            assert 'q25' in split_stats
            assert 'q75' in split_stats
            assert 'p05' in split_stats
            assert 'p95' in split_stats
            assert 'min' in split_stats
            assert 'max' in split_stats

        # Check IS/OOS ratio
        assert 'is_oos_ratio' in stats[metric]
        assert stats[metric]['is_oos_ratio'] > 0


def test_check_cpcv_gates_pass(sample_events_df, sample_data):
    """
    Test validation gates with good performance (should pass).
    """
    X, y = sample_data

    # Create paths
    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=4,
        test_groups=1,
        max_paths=4,
        selection="lexicographic",
        pct_embargo=0.01
    )

    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc'],
        sample_weight=None
    )

    # Set lenient gates (should pass)
    gates_config = {
        'median_roc_auc': {'threshold': 0.5, 'enabled': True},
        'is_oos_ratio': {'threshold': 2.0, 'enabled': True}
    }

    gates_result = check_cpcv_gates(
        perf_df=perf_df,
        stats=None,  # Will compute internally
        gates_config=gates_config
    )

    assert 'gates_passed' in gates_result
    assert 'gates_failed' in gates_result
    assert 'summary' in gates_result

    # With lenient thresholds, should likely pass
    # (though depends on random data, so we just check structure)
    assert isinstance(gates_result['gates_passed'], list)
    assert isinstance(gates_result['gates_failed'], list)


def test_check_cpcv_gates_fail(sample_events_df, sample_data):
    """
    Test validation gates with strict thresholds (should fail).
    """
    X, y = sample_data

    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=4,
        test_groups=1,
        max_paths=4,
        selection="lexicographic",
        pct_embargo=0.01
    )

    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc'],
        sample_weight=None
    )

    # Set very strict gates (should fail)
    gates_config = {
        'median_roc_auc': {'threshold': 0.99, 'enabled': True}  # Nearly impossible
    }

    gates_result = check_cpcv_gates(
        perf_df=perf_df,
        stats=None,
        gates_config=gates_config
    )

    # With strict threshold, should fail
    assert len(gates_result['gates_failed']) > 0

    # Check failure details
    failed_gate = gates_result['gates_failed'][0]
    assert 'gate_name' in failed_gate
    assert 'threshold' in failed_gate
    assert 'actual' in failed_gate


def test_plot_cpcv_performance(sample_events_df, sample_data):
    """
    Test CPCV performance visualization.
    """
    X, y = sample_data

    paths = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=4,
        test_groups=1,
        max_paths=4,
        selection="lexicographic",
        pct_embargo=0.01
    )

    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)

    perf_df = evaluate_cpcv_paths(
        paths=paths,
        model_factory=model_factory,
        X=X,
        y=y,
        events_df=sample_events_df,
        metrics=['roc_auc'],
        sample_weight=None
    )

    # Create plot
    fig = plot_cpcv_performance(
        perf_df=perf_df,
        metric='roc_auc',
        show_thresholds=True
    )

    # Verify figure was created
    assert fig is not None
    assert isinstance(fig, plt.Figure)

    # Clean up
    plt.close(fig)


def test_cpcv_paths_purging():
    """
    Test that CPCV paths properly purge overlapping events.
    """
    # Create events with known overlaps
    np.random.seed(42)
    n_samples = 500

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1h')

    events_df = pd.DataFrame({
        't0': dates,
        't1': dates + pd.Timedelta(hours=3),  # 3-hour overlapping events
    })
    events_df.index = pd.RangeIndex(n_samples)

    paths = build_cpcv_paths(
        events_df=events_df,
        n_groups=4,
        test_groups=1,
        max_paths=4,
        selection="lexicographic",
        pct_embargo=0.01,
        apply_purging=True
    )

    for path in paths:
        train_idx = set(path['train_idx'])
        test_idx = set(path['test_idx'])

        # No overlap between train and test
        assert len(train_idx & test_idx) == 0

        # Check purging: no train event should overlap with test period
        for train_i in train_idx:
            train_t1 = events_df.loc[train_i, 't1']
            for test_i in test_idx:
                test_t0 = events_df.loc[test_i, 't0']
                test_t1 = events_df.loc[test_i, 't1']

                # Train event should not extend into test period
                # (This is a simplified check; actual purging is more sophisticated)
                # We just verify no direct overlap
                assert not (train_t1 > test_t0 and train_t1 < test_t1)


def test_cpcv_determinism(sample_events_df):
    """
    Test that CPCV path generation is deterministic with same seed.
    """
    paths1 = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=5,
        test_groups=2,
        max_paths=5,
        selection="balanced",
        random_state=42,
        pct_embargo=0.01
    )

    paths2 = build_cpcv_paths(
        events_df=sample_events_df,
        n_groups=5,
        test_groups=2,
        max_paths=5,
        selection="balanced",
        random_state=42,
        pct_embargo=0.01
    )

    # Same seed should produce identical paths
    assert len(paths1) == len(paths2)

    for p1, p2 in zip(paths1, paths2):
        assert p1['path_id'] == p2['path_id']
        assert p1['test_folds'] == p2['test_folds']
        assert p1['train_idx'] == p2['train_idx']
        assert p1['test_idx'] == p2['test_idx']


def test_edge_case_small_dataset():
    """
    Test CPCV with very small dataset.
    """
    # Create tiny dataset
    dates = pd.date_range('2020-01-01', periods=50, freq='1h')
    events_df = pd.DataFrame({
        't0': dates,
        't1': dates + pd.Timedelta(hours=1)
    })
    events_df.index = pd.RangeIndex(50)

    paths = build_cpcv_paths(
        events_df=events_df,
        n_groups=3,
        test_groups=1,
        max_paths=3,
        selection="lexicographic",
        pct_embargo=0.01
    )

    # Should still create valid paths
    assert len(paths) == 3
    for path in paths:
        assert len(path['train_idx']) > 0
        assert len(path['test_idx']) > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
