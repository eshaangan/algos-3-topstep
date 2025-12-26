"""
Patch for analyze_training function to handle None values in meta-labeling metrics.

Add this code to your notebook to fix the TypeError when acceptance_rate is None.
"""

def analyze_training_fixed(bar_size, cv_kind):
    """Fixed version of analyze_training that handles None values."""
    bs_dir = RUN_DIR / f"bar_size={bar_size}"
    cv_dir = bs_dir / "training" / cv_kind

    # Load summary
    summary = read_json(cv_dir / "summary.json")

    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY ({cv_kind}, bar_size={bar_size})")
    print(f"{'='*70}")

    print(f"\nSplits: {summary['n_splits']}")

    # Aggregate metrics
    metrics_list = []
    for split in summary['metrics_by_split']:
        metrics_list.append(split['metrics'])

    metrics_df = pd.DataFrame(metrics_list)
    print(f"\nMetrics (mean ± std):")
    for col in metrics_df.columns:
        mean = metrics_df[col].mean()
        std = metrics_df[col].std()
        print(f"  {col:30s}: {mean:.4f} ± {std:.4f}")

    # Meta-labeling statistics (if enabled)
    print(f"\n{'='*70}")
    print(f"META-LABELING STATISTICS")
    print(f"{'='*70}")

    has_meta_stats = False
    for split in summary['metrics_by_split']:
        sid = split['split_id']

        # Try to load meta metrics from fold directory
        fold_dir = cv_dir / f"fold_{sid}"
        meta_metrics_path = fold_dir / "meta_metrics.json"

        if meta_metrics_path.exists():
            has_meta_stats = True
            meta_metrics = read_json(meta_metrics_path)

            proposed = meta_metrics.get('n_proposed_trades_test', 0)
            accepted = meta_metrics.get('n_meta_positive_test', 0)
            rate = meta_metrics.get('acceptance_rate')
            reason = meta_metrics.get('reason', 'unknown')

            # Handle None values
            if proposed == 0 or rate is None:
                print(f"  Split {sid}: NO TRADES (reason: {reason})")
            else:
                print(f"  Split {sid}: {accepted:>5,}/{proposed:>6,} accepted ({rate:>5.1%})")

    if not has_meta_stats:
        print("  Meta-labeling not enabled or no meta metrics found")

    return summary


# Usage in notebook:
# Replace this:
#   train_summary = analyze_training("1m", CV_KIND)
# With this:
#   train_summary = analyze_training_fixed("1m", CV_KIND)
