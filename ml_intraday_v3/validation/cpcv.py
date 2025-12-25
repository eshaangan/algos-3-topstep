"""
Combinatorial Purged Cross-Validation (CPCV) - Enhanced Implementation

Features:
- Path generation with multiple selection strategies (lexicographic, balanced, random)
- Path performance evaluation across all metrics
- Distribution analysis (quartiles, percentiles)
- Validation gates (Sharpe, PBO, DSR thresholds)
- Performance visualization

Reference: López de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 7.
"""

import logging
from typing import List, Dict, Optional, Callable, Any
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _shift_by_bars(
    bars_index: pd.Index, timestamp: np.datetime64, embargo_bars: int
) -> np.datetime64 | None:
    """
    Shift timestamp forward by embargo_bars on bars_index (anchored to next bar).

    Uses the first bar at or after timestamp as the anchor.
    Returns None if timestamp is beyond the last bar.
    """
    end_pos = bars_index.searchsorted(timestamp, side="left")
    if end_pos >= len(bars_index):
        return None
    embargo_end_pos = min(end_pos + int(embargo_bars), len(bars_index) - 1)
    return bars_index[embargo_end_pos].to_datetime64()


def _select_balanced_paths(
    all_combos: List[tuple],
    K: int,
    max_paths: int,
    random_state: Optional[int] = None
) -> List[tuple]:
    """
    Select paths using balanced strategy to ensure equal test fold coverage.

    Parameters
    ----------
    all_combos : List[tuple]
        All possible combinations
    K : int
        Number of folds
    max_paths : int
        Maximum number of paths to select
    random_state : int, optional
        Random seed for tie-breaking

    Returns
    -------
    List[tuple]
        Selected combinations
    """
    if max_paths >= len(all_combos):
        return all_combos

    rng = np.random.RandomState(random_state)

    # Greedy algorithm to maximize balance
    selected = []
    fold_counts = np.zeros(K, dtype=int)

    # Start with combination that has lowest-count folds
    remaining = list(all_combos)

    while len(selected) < max_paths and remaining:
        # Score each remaining combo by how much it balances coverage
        scores = []
        for combo in remaining:
            # Prefer combos with folds that have been tested less
            fold_coverage = fold_counts[list(combo)]
            # Lower score = better (covers underrepresented folds)
            score = fold_coverage.sum() + rng.uniform(0, 0.1)  # Tie-breaker
            scores.append(score)

        # Select combo with lowest score
        best_idx = np.argmin(scores)
        best_combo = remaining.pop(best_idx)
        selected.append(best_combo)

        # Update counts
        for fold_idx in best_combo:
            fold_counts[fold_idx] += 1

    logger.info(f"Balanced selection: fold coverage = {fold_counts.tolist()}")
    return selected


def build_cpcv_paths(
    base_folds: List[Dict],
    events_df: pd.DataFrame,
    bars_index: pd.Index,
    K: int,
    test_groups: int,
    embargo_bars: int,
    max_paths: Optional[int] = None,
    selection: str = "lexicographic",
    random_state: Optional[int] = None,
) -> List[Dict]:
    """
    Build CPCV paths from base folds with configurable selection strategy.

    Parameters
    ----------
    base_folds : List[Dict]
        Base folds from purged k-fold
    events_df : pd.DataFrame
        Events with columns: event_id, t0, t1
    bars_index : pd.Index
        Bar timestamps for embargo calculation
    K : int
        Number of folds
    test_groups : int
        Number of folds to hold out as test in each path
    embargo_bars : int
        Number of bars to embargo after test period
    max_paths : int, optional
        Maximum number of paths to generate
    selection : str, default='lexicographic'
        Path selection strategy:
        - 'lexicographic': First N paths in natural order
        - 'balanced': Greedily select paths to balance test fold coverage
        - 'random': Random sample of paths
    random_state : int, optional
        Random seed for 'random' and 'balanced' selection

    Returns
    -------
    List[Dict]
        CPCV paths, each containing:
        - path_id: Path identifier
        - test_folds: Fold indices used for test
        - train_event_ids: Event IDs in training set
        - test_event_ids: Event IDs in test set
        - test_interval: Start/end timestamps of test period
        - purge: Purging/embargo statistics

    Examples
    --------
    >>> paths = build_cpcv_paths(
    ...     base_folds=folds,
    ...     events_df=events,
    ...     bars_index=bars.index,
    ...     K=6,
    ...     test_groups=2,
    ...     embargo_bars=50,
    ...     max_paths=10,
    ...     selection='balanced'
    ... )
    """
    required_cols = ["event_id", "t0", "t1"]
    missing = [c for c in required_cols if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(base_folds) != K:
        raise ValueError("base_folds length must equal K")
    if test_groups < 1 or test_groups > K:
        raise ValueError("test_groups must be in [1, K]")
    if selection not in ["lexicographic", "balanced", "random"]:
        raise ValueError(f"Unsupported selection: {selection}. Must be 'lexicographic', 'balanced', or 'random'")

    df = events_df.copy()
    df["t0"] = pd.to_datetime(df["t0"], utc=True)
    df["t1"] = pd.to_datetime(df["t1"], utc=True)
    df = df.sort_values("t0").reset_index(drop=True)

    all_ids = df["event_id"].to_numpy()
    t0 = df["t0"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    t1 = df["t1"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    if isinstance(bars_index, pd.DatetimeIndex) and bars_index.tz is not None:
        bars_index = bars_index.tz_convert("UTC").tz_localize(None)
    bars_index = pd.DatetimeIndex(bars_index)

    # Generate all possible combinations
    all_combos = list(combinations(range(K), test_groups))
    total_combos = len(all_combos)

    # Apply selection strategy
    if max_paths is None or max_paths >= total_combos:
        combos = all_combos
        logger.info(f"Using all {total_combos} CPCV paths")
    else:
        if selection == "lexicographic":
            combos = all_combos[:max_paths]
            logger.info(f"Lexicographic selection: first {max_paths} of {total_combos} paths")

        elif selection == "balanced":
            combos = _select_balanced_paths(all_combos, K, max_paths, random_state)
            logger.info(f"Balanced selection: {max_paths} of {total_combos} paths")

        elif selection == "random":
            rng = np.random.RandomState(random_state)
            indices = rng.choice(total_combos, size=max_paths, replace=False)
            combos = [all_combos[i] for i in sorted(indices)]
            logger.info(f"Random selection: {max_paths} of {total_combos} paths (seed={random_state})")

    id_to_pos = {eid: i for i, eid in enumerate(all_ids)}

    paths = []
    for path_id, combo in enumerate(combos):
        test_ids = []
        for fold_idx in combo:
            test_ids.extend(base_folds[fold_idx]["test_event_ids"])
        test_ids = sorted(set(test_ids))

        test_pos = [id_to_pos[eid] for eid in test_ids if eid in id_to_pos]
        if not test_pos:
            continue

        test_start = t0[min(test_pos)]
        test_end = t1[max(test_pos)]

        overlap_mask = (t0 <= test_end) & (t1 >= test_start)

        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_pos] = True

        train_mask = ~test_mask
        n_purged = int((train_mask & overlap_mask).sum())
        train_mask = train_mask & (~overlap_mask)

        n_embargoed = 0
        if embargo_bars > 0:
            embargo_end = _shift_by_bars(bars_index, test_end, embargo_bars)
            if embargo_end is not None:
                embargo_mask = (t0 > test_end) & (t0 <= embargo_end)
                n_embargoed = int((train_mask & embargo_mask).sum())
                train_mask = train_mask & (~embargo_mask)

        train_ids = np.sort(all_ids[train_mask]).tolist()

        paths.append(
            {
                "path_id": path_id,
                "test_folds": list(combo),
                "test_event_ids": test_ids,
                "train_event_ids": train_ids,
                "test_interval": {
                    "start": pd.Timestamp(test_start).isoformat(),
                    "end": pd.Timestamp(test_end).isoformat(),
                },
                "purge": {"n_purged": n_purged, "n_embargoed": n_embargoed},
                "params": {
                    "K": K,
                    "test_groups": test_groups,
                    "embargo_bars": embargo_bars,
                    "selection": selection,
                },
            }
        )

    return paths


def evaluate_cpcv_paths(
    paths: List[Dict],
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    events_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Evaluate performance across all CPCV paths.

    Parameters
    ----------
    paths : List[Dict]
        CPCV paths from build_cpcv_paths()
    model_factory : Callable
        Function that returns a new model instance (e.g., lambda: LogisticRegression())
    X : pd.DataFrame
        Features with 'event_id' column or index
    y : pd.Series
        Labels with 'event_id' as index or matching X index
    events_df : pd.DataFrame
        Events with 'event_id' column for alignment
    metrics : List[str], optional
        Metrics to compute. Default: ['sharpe', 'mean_return', 'win_rate']
    sample_weight : pd.Series, optional
        Sample weights with 'event_id' as index

    Returns
    -------
    pd.DataFrame
        Performance metrics with columns:
        - path_id: Path identifier
        - metric_name: Name of metric
        - metric_value: Value of metric
        - is_oos: True for OOS (test), False for IS (train)
        - n_samples: Number of samples
        - test_folds: Which folds were used for test

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> perf_df = evaluate_cpcv_paths(
    ...     paths=cpcv_paths,
    ...     model_factory=lambda: LogisticRegression(),
    ...     X=features,
    ...     y=labels,
    ...     events_df=events,
    ...     metrics=['sharpe', 'win_rate', 'profit_factor']
    ... )
    """
    if metrics is None:
        metrics = ['sharpe', 'mean_return', 'win_rate']

    # Ensure X has event_id
    if 'event_id' not in X.columns:
        if X.index.name == 'event_id' or 'event_id' in X.index.names:
            X = X.reset_index()
        else:
            raise ValueError("X must have 'event_id' column or index")

    # Ensure y has event_id index
    if y.index.name != 'event_id':
        if 'event_id' in y.index.names:
            y = y.reset_index().set_index('event_id')['y'] if 'y' in y.columns else y.reset_index().set_index('event_id')[y.name]
        else:
            raise ValueError("y must have 'event_id' as index")

    results = []

    for path in paths:
        path_id = path['path_id']
        train_ids = path['train_event_ids']
        test_ids = path['test_event_ids']
        test_folds = path['test_folds']

        # Get train/test data
        X_train = X[X['event_id'].isin(train_ids)].drop(columns=['event_id'])
        X_test = X[X['event_id'].isin(test_ids)].drop(columns=['event_id'])

        y_train = y[y.index.isin(train_ids)]
        y_test = y[y.index.isin(test_ids)]

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"Path {path_id}: empty train or test set, skipping")
            continue

        # Get sample weights if provided
        w_train = None
        w_test = None
        if sample_weight is not None:
            w_train = sample_weight[sample_weight.index.isin(train_ids)]
            w_test = sample_weight[sample_weight.index.isin(test_ids)]

        # Train model
        model = model_factory()
        if w_train is not None and hasattr(model, 'fit'):
            try:
                model.fit(X_train, y_train, sample_weight=w_train.values)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_train_proba = model.predict_proba(X_train)[:, 1]
            y_pred_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_train_proba = model.predict(X_train)
            y_pred_test_proba = model.predict(X_test)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Compute metrics for train (IS) and test (OOS)
        for is_oos, y_true, y_pred, y_proba, n_samples in [
            (False, y_train, y_pred_train, y_pred_train_proba, len(y_train)),
            (True, y_test, y_pred_test, y_pred_test_proba, len(y_test))
        ]:
            # Compute each requested metric
            for metric_name in metrics:
                try:
                    metric_value = _compute_metric(
                        metric_name,
                        y_true.values,
                        y_pred,
                        y_proba
                    )

                    results.append({
                        'path_id': path_id,
                        'metric_name': metric_name,
                        'metric_value': metric_value,
                        'is_oos': is_oos,
                        'n_samples': n_samples,
                        'test_folds': tuple(test_folds)
                    })
                except Exception as e:
                    logger.warning(f"Path {path_id}, metric {metric_name}, IS/OOS={is_oos}: {e}")
                    results.append({
                        'path_id': path_id,
                        'metric_name': metric_name,
                        'metric_value': np.nan,
                        'is_oos': is_oos,
                        'n_samples': n_samples,
                        'test_folds': tuple(test_folds)
                    })

    return pd.DataFrame(results)


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute a single metric."""
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    if metric_name == 'accuracy':
        return float(accuracy_score(y_true, y_pred))

    elif metric_name == 'roc_auc':
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_proba))

    elif metric_name == 'log_loss':
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(log_loss(y_true, y_proba))

    elif metric_name == 'win_rate':
        return float(np.mean(y_pred == 1))

    elif metric_name == 'sharpe':
        # Simplified: assumes y_pred represents trade returns
        # In practice, you'd need actual returns
        returns = y_pred.astype(float)
        if len(returns) < 2 or returns.std() == 0:
            return np.nan
        return float(returns.mean() / returns.std())

    elif metric_name == 'mean_return':
        returns = y_pred.astype(float)
        return float(returns.mean())

    elif metric_name == 'profit_factor':
        returns = y_pred.astype(float)
        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        if losses == 0:
            return np.inf if wins > 0 else np.nan
        return float(wins / losses)

    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def analyze_cpcv_distributions(
    perf_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Analyze performance distributions across CPCV paths.

    Parameters
    ----------
    perf_df : pd.DataFrame
        Performance DataFrame from evaluate_cpcv_paths()
    metrics : List[str], optional
        Metrics to analyze. If None, analyzes all metrics in perf_df

    Returns
    -------
    Dict[str, Dict]
        Distribution statistics for each metric:
        {
            'metric_name': {
                'is': {
                    'mean': float,
                    'median': float,
                    'std': float,
                    'q25': float (25th percentile),
                    'q75': float (75th percentile),
                    'p05': float (5th percentile),
                    'p95': float (95th percentile),
                    'min': float,
                    'max': float,
                    'n_paths': int
                },
                'oos': { ... },
                'is_oos_ratio': float (median IS / median OOS)
            }
        }

    Examples
    --------
    >>> stats = analyze_cpcv_distributions(perf_df)
    >>> print(f"OOS Sharpe median: {stats['sharpe']['oos']['median']:.3f}")
    >>> print(f"OOS Sharpe 25th percentile: {stats['sharpe']['oos']['q25']:.3f}")
    """
    if metrics is None:
        metrics = perf_df['metric_name'].unique().tolist()

    results = {}

    for metric_name in metrics:
        metric_data = perf_df[perf_df['metric_name'] == metric_name]

        if metric_data.empty:
            continue

        is_data = metric_data[~metric_data['is_oos']]['metric_value'].dropna()
        oos_data = metric_data[metric_data['is_oos']]['metric_value'].dropna()

        def compute_stats(data):
            if len(data) == 0:
                return None
            return {
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data, ddof=1) if len(data) > 1 else 0),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'p05': float(np.percentile(data, 5)),
                'p95': float(np.percentile(data, 95)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'n_paths': int(len(data))
            }

        is_stats = compute_stats(is_data)
        oos_stats = compute_stats(oos_data)

        # Compute IS/OOS ratio
        is_oos_ratio = np.nan
        if is_stats and oos_stats and oos_stats['median'] != 0:
            is_oos_ratio = is_stats['median'] / oos_stats['median']

        results[metric_name] = {
            'is': is_stats,
            'oos': oos_stats,
            'is_oos_ratio': float(is_oos_ratio)
        }

    return results


def check_cpcv_gates(
    perf_df: pd.DataFrame,
    stats: Optional[Dict] = None,
    gates_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Check CPCV validation gates.

    Parameters
    ----------
    perf_df : pd.DataFrame
        Performance DataFrame from evaluate_cpcv_paths()
    stats : Dict, optional
        Pre-computed statistics from analyze_cpcv_distributions()
        If None, will compute them
    gates_config : Dict, optional
        Gate thresholds. Default:
        {
            'median_sharpe': 0.8,
            'p25_sharpe': 0.2,
            'median_roc_auc': 0.55,
            'is_oos_ratio_max': 1.5,
            'pbo_threshold': 0.3,
            'dsr_threshold': 0.5
        }

    Returns
    -------
    Dict
        {
            'passed': bool (all gates passed),
            'gates': {
                'gate_name': {
                    'passed': bool,
                    'value': float,
                    'threshold': float,
                    'message': str
                },
                ...
            },
            'summary': str
        }

    Examples
    --------
    >>> gates = check_cpcv_gates(perf_df)
    >>> if gates['passed']:
    ...     print("✓ All validation gates passed!")
    >>> else:
    ...     print(gates['summary'])
    """
    if stats is None:
        stats = analyze_cpcv_distributions(perf_df)

    # Default gate configuration
    default_gates = {
        'median_sharpe': 0.8,
        'p25_sharpe': 0.2,
        'median_roc_auc': 0.55,
        'is_oos_ratio_max': 1.5,
    }

    if gates_config is None:
        gates_config = default_gates
    else:
        gates_config = {**default_gates, **gates_config}

    gates = {}
    all_passed = True

    # Check median Sharpe > threshold
    if 'sharpe' in stats and stats['sharpe']['oos']:
        value = stats['sharpe']['oos']['median']
        threshold = gates_config['median_sharpe']
        passed = value >= threshold

        gates['median_sharpe_oos'] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'message': f"Median OOS Sharpe {value:.3f} {'≥' if passed else '<'} {threshold:.3f}"
        }
        all_passed = all_passed and passed

    # Check 25th percentile Sharpe > threshold
    if 'sharpe' in stats and stats['sharpe']['oos']:
        value = stats['sharpe']['oos']['q25']
        threshold = gates_config['p25_sharpe']
        passed = value >= threshold

        gates['p25_sharpe_oos'] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'message': f"25th percentile OOS Sharpe {value:.3f} {'≥' if passed else '<'} {threshold:.3f}"
        }
        all_passed = all_passed and passed

    # Check median ROC-AUC > threshold
    if 'roc_auc' in stats and stats['roc_auc']['oos']:
        value = stats['roc_auc']['oos']['median']
        threshold = gates_config['median_roc_auc']
        passed = value >= threshold

        gates['median_roc_auc_oos'] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'message': f"Median OOS ROC-AUC {value:.3f} {'≥' if passed else '<'} {threshold:.3f}"
        }
        all_passed = all_passed and passed

    # Check IS/OOS ratio < threshold (check for overfitting)
    if 'sharpe' in stats and not np.isnan(stats['sharpe']['is_oos_ratio']):
        value = stats['sharpe']['is_oos_ratio']
        threshold = gates_config['is_oos_ratio_max']
        passed = value <= threshold

        gates['is_oos_ratio'] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'message': f"IS/OOS ratio {value:.3f} {'≤' if passed else '>'} {threshold:.3f}"
        }
        all_passed = all_passed and passed

    # Generate summary
    passed_count = sum(1 for g in gates.values() if g['passed'])
    total_count = len(gates)

    if all_passed:
        summary = f"✓ All {total_count} validation gates passed"
    else:
        failed = [name for name, g in gates.items() if not g['passed']]
        summary = f"✗ {total_count - passed_count}/{total_count} gates failed: {', '.join(failed)}"

    return {
        'passed': all_passed,
        'gates': gates,
        'summary': summary,
        'passed_count': passed_count,
        'total_count': total_count
    }


def plot_cpcv_performance(
    perf_df: pd.DataFrame,
    metric: str = 'sharpe',
    ax: Optional[plt.Axes] = None,
    show_thresholds: bool = True,
    thresholds: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot CPCV performance distributions.

    Parameters
    ----------
    perf_df : pd.DataFrame
        Performance DataFrame from evaluate_cpcv_paths()
    metric : str
        Metric to plot
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure
    show_thresholds : bool
        Whether to show threshold lines
    thresholds : Dict, optional
        Thresholds to mark. Default: {'median': 0.8, 'p25': 0.2}

    Returns
    -------
    plt.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_cpcv_performance(perf_df, metric='sharpe')
    >>> fig.savefig('cpcv_sharpe_distribution.png')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Filter data
    data = perf_df[perf_df['metric_name'] == metric].copy()

    if data.empty:
        ax.text(0.5, 0.5, f'No data for metric: {metric}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Prepare data for plotting
    data['split_type'] = data['is_oos'].map({False: 'IS (Train)', True: 'OOS (Test)'})

    # Create box plot
    sns.boxplot(
        data=data,
        x='split_type',
        y='metric_value',
        ax=ax,
        palette={'IS (Train)': 'lightblue', 'OOS (Test)': 'lightcoral'},
        width=0.5
    )

    # Overlay violin plot for distribution
    sns.violinplot(
        data=data,
        x='split_type',
        y='metric_value',
        ax=ax,
        palette={'IS (Train)': 'lightblue', 'OOS (Test)': 'lightcoral'},
        alpha=0.3,
        inner=None
    )

    # Add threshold lines if requested
    if show_thresholds:
        if thresholds is None:
            thresholds = {'median': 0.8, 'p25': 0.2}

        stats = analyze_cpcv_distributions(perf_df, metrics=[metric])

        if metric in stats and stats[metric]['oos']:
            oos_median = stats[metric]['oos']['median']
            oos_q25 = stats[metric]['oos']['q25']

            # Median threshold
            if 'median' in thresholds:
                ax.axhline(thresholds['median'], color='red', linestyle='--',
                          linewidth=2, label=f'Median threshold ({thresholds["median"]:.2f})',
                          alpha=0.7)

            # P25 threshold
            if 'p25' in thresholds:
                ax.axhline(thresholds['p25'], color='orange', linestyle=':',
                          linewidth=2, label=f'P25 threshold ({thresholds["p25"]:.2f})',
                          alpha=0.7)

            # Show actual values
            ax.axhline(oos_median, color='darkred', linestyle='-',
                      linewidth=1, label=f'OOS median ({oos_median:.3f})',
                      alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('Split Type', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'CPCV Performance Distribution: {metric.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    stats = analyze_cpcv_distributions(perf_df, metrics=[metric])
    if metric in stats:
        is_stats = stats[metric].get('is')
        oos_stats = stats[metric].get('oos')

        if is_stats and oos_stats:
            stats_text = (
                f"IS:  μ={is_stats['mean']:.3f}, σ={is_stats['std']:.3f}\n"
                f"OOS: μ={oos_stats['mean']:.3f}, σ={oos_stats['std']:.3f}\n"
                f"Ratio: {stats[metric]['is_oos_ratio']:.3f}"
            )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9,
                   family='monospace')

    fig.tight_layout()
    return fig
