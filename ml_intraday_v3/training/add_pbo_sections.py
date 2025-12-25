#!/usr/bin/env python3
"""
Script to add Enhanced PBO sections to the pipeline notebook.

Adds:
1. Section 4.6.4: Enhanced PBO Analysis (after 4.6.3, before 4.7)
2. Section 5.3: PBO Validation Across Multiple Runs (after 5.2, before Section 6)
"""

import json
from pathlib import Path


def create_pbo_section_464():
    """Create Section 4.6.4: Enhanced PBO Analysis."""

    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.4 Enhanced PBO (Probability of Backtest Overfitting) Analysis\n",
            "\n",
            "**What is PBO?**\n",
            "\n",
            "PBO (Probability of Backtest Overfitting) measures the likelihood that your \"best\" configuration was selected due to overfitting rather than true predictive skill.\n",
            "\n",
            "**How it works:**\n",
            "1. For each CPCV path, select the best trial based on in-sample performance on all other paths\n",
            "2. Measure that trial's out-of-sample rank on the held-out path (lambda value)\n",
            "3. PBO = fraction of paths where lambda < 0.5 (below median)\n",
            "\n",
            "**Interpretation:**\n",
            "- **PBO < 0.3**: Low risk - Configuration appears robust\n",
            "- **PBO 0.3-0.5**: Moderate risk - Monitor carefully\n",
            "- **PBO > 0.5**: High risk - Likely overfitting, consider reducing search space\n",
            "\n",
            "**Reference:** López de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 11.\n"
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Section 4.6.4: Enhanced PBO Analysis\n",
            "\n",
            "from ml_intraday_v3.experiments.trial_tracker import TrialTracker\n",
            "from ml_intraday_v3.experiments.diagnostics import (\n",
            "    compute_pbo_with_confidence,\n",
            "    plot_pbo_distribution,\n",
            "    plot_pbo_with_confidence,\n",
            "    generate_pbo_report\n",
            ")\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"Section 4.6.4: Enhanced PBO (Probability of Backtest Overfitting) Analysis\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "# Check if trials exist\n",
            "trials_path = RUN_DIR / 'trials' / 'trials.json'\n",
            "\n",
            "if trials_path.exists():\n",
            "    print(f\"Loading trials from: {trials_path}\")\n",
            "    \n",
            "    # Load trials\n",
            "    tracker = TrialTracker(RUN_DIR)\n",
            "    \n",
            "    # Display trial summary\n",
            "    summary = tracker.get_summary_stats()\n",
            "    print(f\"\\nTrial Summary:\")\n",
            "    print(f\"  - Total trials tracked: {summary['n_trials']}\")\n",
            "    print(f\"  - CPCV paths: {summary['n_paths']}\")\n",
            "    print(f\"  - Model types: {summary['model_types']}\")\n",
            "    print(f\"  - Date range: {summary['date_range']['first']} to {summary['date_range']['last']}\")\n",
            "    \n",
            "    if summary['n_trials'] < 2:\n",
            "        print(\"\\n⚠️  Warning: Need at least 2 trials to compute PBO.\")\n",
            "        print(\"   Track trials during hyperparameter search using TrialTracker.\")\n",
            "    else:\n",
            "        # Convert to DataFrame\n",
            "        trials_df = tracker.to_dataframe()\n",
            "        print(f\"\\nTrials DataFrame shape: {trials_df.shape}\")\n",
            "        print(f\"Columns: {list(trials_df.columns)}\")\n",
            "        \n",
            "        # Compute PBO with confidence intervals\n",
            "        print(\"\\nComputing PBO with bootstrap confidence intervals...\")\n",
            "        pbo_result = compute_pbo_with_confidence(\n",
            "            trials_df=trials_df,\n",
            "            metric_name='roc_auc',\n",
            "            higher_is_better=True,\n",
            "            n_bootstrap=1000,\n",
            "            confidence_level=0.95,\n",
            "            random_state=42\n",
            "        )\n",
            "        \n",
            "        if pbo_result['pbo'] is not None:\n",
            "            # Display results\n",
            "            pbo = pbo_result['pbo']\n",
            "            pbo_lower = pbo_result.get('pbo_lower')\n",
            "            pbo_upper = pbo_result.get('pbo_upper')\n",
            "            \n",
            "            print(f\"\\n{'='*60}\")\n",
            "            print(f\"PBO Results:\")\n",
            "            print(f\"{'='*60}\")\n",
            "            print(f\"PBO = {pbo:.3f} ({pbo*100:.1f}%)\")\n",
            "            if pbo_lower is not None and pbo_upper is not None:\n",
            "                print(f\"95% CI: [{pbo_lower:.3f}, {pbo_upper:.3f}]\")\n",
            "            print(f\"\\nLambda Statistics:\")\n",
            "            print(f\"  - Mean: {pbo_result['lambda_mean']:.3f}\")\n",
            "            print(f\"  - Median: {pbo_result['lambda_median']:.3f}\")\n",
            "            print(f\"  - Std: {pbo_result['lambda_std']:.3f}\")\n",
            "            print(f\"\\nTrials: {pbo_result['n_trials']}\")\n",
            "            print(f\"CPCV Paths: {pbo_result['n_paths']}\")\n",
            "            \n",
            "            # Interpretation\n",
            "            if pbo > 0.5:\n",
            "                risk = \"🔴 HIGH RISK\"\n",
            "                interpretation = \"Likely overfitting - Consider reducing search space or increasing sample size\"\n",
            "            elif pbo > 0.3:\n",
            "                risk = \"🟠 MODERATE RISK\"\n",
            "                interpretation = \"Some overfitting risk - Monitor performance carefully\"\n",
            "            else:\n",
            "                risk = \"🟢 LOW RISK\"\n",
            "                interpretation = \"Configuration appears robust\"\n",
            "            \n",
            "            print(f\"\\nRisk Level: {risk}\")\n",
            "            print(f\"Interpretation: {interpretation}\")\n",
            "            print(f\"{'='*60}\\n\")\n",
            "            \n",
            "            # Create visualizations\n",
            "            print(\"Creating PBO visualizations...\")\n",
            "            \n",
            "            fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
            "            \n",
            "            # Lambda distribution\n",
            "            plot_pbo_distribution(\n",
            "                lambda_values=pbo_result['lambda_values'],\n",
            "                pbo_value=pbo,\n",
            "                ax=axes[0]\n",
            "            )\n",
            "            \n",
            "            # PBO with confidence intervals\n",
            "            plot_pbo_with_confidence(\n",
            "                pbo_result=pbo_result,\n",
            "                ax=axes[1]\n",
            "            )\n",
            "            \n",
            "            plt.tight_layout()\n",
            "            \n",
            "            # Save figure\n",
            "            pbo_fig_path = RUN_DIR / 'pbo_analysis.png'\n",
            "            fig.savefig(pbo_fig_path, dpi=150, bbox_inches='tight')\n",
            "            print(f\"Saved PBO visualization to: {pbo_fig_path}\")\n",
            "            \n",
            "            plt.show()\n",
            "            \n",
            "            # Generate markdown report\n",
            "            print(\"\\nGenerating PBO report...\")\n",
            "            report_path = RUN_DIR / 'pbo_report.md'\n",
            "            report = generate_pbo_report(\n",
            "                pbo_result=pbo_result,\n",
            "                save_path=str(report_path)\n",
            "            )\n",
            "            print(f\"Saved PBO report to: {report_path}\")\n",
            "            \n",
            "            # Display report in notebook\n",
            "            print(\"\\n\" + \"=\"*80)\n",
            "            print(\"PBO Report Preview:\")\n",
            "            print(\"=\"*80)\n",
            "            display(Markdown(report))\n",
            "            \n",
            "        else:\n",
            "            reason = pbo_result.get('reason', 'unknown')\n",
            "            print(f\"\\n⚠️  Cannot compute PBO: {reason}\")\n",
            "            print(\"   Ensure trials have IS/OOS metrics for all CPCV paths.\")\n",
            "    \n",
            "else:\n",
            "    print(f\"\\n⚠️  No trials found at: {trials_path}\")\n",
            "    print(\"\\nTo use PBO analysis, track trials during training:\")\n",
            "    print(\"\"\"\\n",
            "from ml_intraday_v3.experiments.trial_tracker import TrialTracker\n",
            "\n",
            "tracker = TrialTracker(RUN_DIR)\n",
            "\n",
            "# During hyperparameter search:\n",
            "for config in all_configs:\n",
            "    trial_id = tracker.log_trial(\n",
            "        config=config,\n",
            "        model_type='logreg',\n",
            "        hyperparameters=config['model']['params']\n",
            "    )\n",
            "    \n",
            "    # After CPCV evaluation:\n",
            "    for path_id, (is_metric, oos_metric) in cpcv_results.items():\n",
            "        tracker.update_path_metrics(trial_id, path_id, is_metric, oos_metric)\n",
            "\n",
            "tracker.save()\n",
            "    \"\"\")\n",
            "    print(\"\\nSee ml_intraday_v3/experiments/PBO_ENHANCED_README.md for details.\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"Section 4.6.4 Complete\")\n",
            "print(\"=\"*80 + \"\\n\")\n"
        ]
    }

    return [markdown_cell, code_cell]


def create_pbo_section_53():
    """Create Section 5.3: PBO Validation Across Multiple Runs."""

    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.3 PBO Validation Across Multiple Runs\n",
            "\n",
            "This section demonstrates how to compare PBO across different hyperparameter searches or model variants, and provides guidelines for stopping hyperparameter search based on PBO thresholds.\n",
            "\n",
            "**Key Questions:**\n",
            "1. How does PBO change with number of trials (selection bias)?\n",
            "2. When should we stop hyperparameter search?\n",
            "3. How do different model variants compare in terms of overfitting risk?\n",
            "\n",
            "**Guidelines:**\n",
            "- **Stop if PBO > 0.5**: High overfitting risk, reduce search space\n",
            "- **Stop if PBO increases**: More trials without improvement suggest overfitting\n",
            "- **Compare across runs**: Lower PBO is better (less selection bias)\n"
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Section 5.3: PBO Validation Across Multiple Runs\n",
            "\n",
            "from ml_intraday_v3.experiments.trial_tracker import TrialTracker\n",
            "from ml_intraday_v3.experiments.diagnostics import compute_pbo_enhanced\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"Section 5.3: PBO Validation Across Multiple Runs\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "trials_path = RUN_DIR / 'trials' / 'trials.json'\n",
            "\n",
            "if trials_path.exists():\n",
            "    # Load trials\n",
            "    tracker = TrialTracker(RUN_DIR)\n",
            "    trials_df = tracker.to_dataframe()\n",
            "    \n",
            "    n_trials = len(trials_df)\n",
            "    \n",
            "    if n_trials >= 5:\n",
            "        print(\"Analyzing how PBO changes with number of trials...\\n\")\n",
            "        \n",
            "        # Compute PBO for increasing number of trials\n",
            "        trial_counts = []\n",
            "        pbo_values = []\n",
            "        \n",
            "        # Start with at least 2 trials, then increase\n",
            "        trial_range = range(2, n_trials + 1)\n",
            "        \n",
            "        for n in trial_range:\n",
            "            subset_df = trials_df.iloc[:n].copy()\n",
            "            result = compute_pbo_enhanced(\n",
            "                subset_df,\n",
            "                metric_name='roc_auc',\n",
            "                higher_is_better=True\n",
            "            )\n",
            "            \n",
            "            if result['pbo'] is not None:\n",
            "                trial_counts.append(n)\n",
            "                pbo_values.append(result['pbo'])\n",
            "        \n",
            "        if len(pbo_values) > 0:\n",
            "            # Plot PBO vs number of trials\n",
            "            fig, ax = plt.subplots(figsize=(12, 6))\n",
            "            \n",
            "            ax.plot(\n",
            "                trial_counts,\n",
            "                pbo_values,\n",
            "                marker='o',\n",
            "                linewidth=2,\n",
            "                markersize=8,\n",
            "                color='steelblue',\n",
            "                label='PBO vs Trials'\n",
            "            )\n",
            "            \n",
            "            # Reference lines\n",
            "            ax.axhline(\n",
            "                0.5,\n",
            "                color='red',\n",
            "                linestyle='--',\n",
            "                linewidth=2,\n",
            "                alpha=0.7,\n",
            "                label='High Risk Threshold (0.5)'\n",
            "            )\n",
            "            ax.axhline(\n",
            "                0.3,\n",
            "                color='orange',\n",
            "                linestyle='--',\n",
            "                linewidth=2,\n",
            "                alpha=0.7,\n",
            "                label='Moderate Risk Threshold (0.3)'\n",
            "            )\n",
            "            \n",
            "            ax.set_xlabel('Number of Trials', fontsize=12)\n",
            "            ax.set_ylabel('PBO (Probability of Backtest Overfitting)', fontsize=12)\n",
            "            ax.set_title(\n",
            "                'PBO vs Number of Trials (Selection Bias Demonstration)',\n",
            "                fontsize=14,\n",
            "                fontweight='bold'\n",
            "            )\n",
            "            ax.legend(loc='best', fontsize=10)\n",
            "            ax.grid(True, alpha=0.3)\n",
            "            ax.set_ylim(0, 1)\n",
            "            \n",
            "            plt.tight_layout()\n",
            "            \n",
            "            # Save figure\n",
            "            pbo_trend_path = RUN_DIR / 'pbo_vs_trials.png'\n",
            "            fig.savefig(pbo_trend_path, dpi=150, bbox_inches='tight')\n",
            "            print(f\"Saved PBO trend plot to: {pbo_trend_path}\")\n",
            "            \n",
            "            plt.show()\n",
            "            \n",
            "            # Analysis\n",
            "            print(f\"\\n{'='*60}\")\n",
            "            print(\"PBO Trend Analysis:\")\n",
            "            print(f\"{'='*60}\")\n",
            "            print(f\"Starting PBO (n={trial_counts[0]}): {pbo_values[0]:.3f}\")\n",
            "            print(f\"Final PBO (n={trial_counts[-1]}): {pbo_values[-1]:.3f}\")\n",
            "            print(f\"Change: {pbo_values[-1] - pbo_values[0]:+.3f}\")\n",
            "            \n",
            "            # Check if PBO increased\n",
            "            if pbo_values[-1] > pbo_values[0]:\n",
            "                print(\"\\n⚠️  Warning: PBO increased with more trials (selection bias)\")\n",
            "                print(\"   Consider stopping hyperparameter search.\")\n",
            "            else:\n",
            "                print(\"\\n✓ PBO did not increase significantly with more trials.\")\n",
            "            \n",
            "            # Stopping criteria\n",
            "            print(f\"\\n{'='*60}\")\n",
            "            print(\"Stopping Criteria Recommendations:\")\n",
            "            print(f\"{'='*60}\")\n",
            "            \n",
            "            final_pbo = pbo_values[-1]\n",
            "            if final_pbo > 0.5:\n",
            "                print(\"🔴 STOP: PBO > 0.5 (high overfitting risk)\")\n",
            "                print(\"   Actions:\")\n",
            "                print(\"   1. Reduce hyperparameter search space\")\n",
            "                print(\"   2. Increase training sample size\")\n",
            "                print(\"   3. Use simpler models\")\n",
            "                print(\"   4. Consider ensemble methods instead of single best config\")\n",
            "            elif final_pbo > 0.3:\n",
            "                print(\"🟠 CAUTION: PBO > 0.3 (moderate risk)\")\n",
            "                print(\"   Actions:\")\n",
            "                print(\"   1. Validate on additional out-of-sample data\")\n",
            "                print(\"   2. Monitor performance closely after deployment\")\n",
            "                print(\"   3. Be conservative with position sizing initially\")\n",
            "            else:\n",
            "                print(\"🟢 PROCEED: PBO < 0.3 (low risk)\")\n",
            "                print(\"   Configuration appears robust, but continue monitoring.\")\n",
            "            \n",
            "            print(f\"{'='*60}\\n\")\n",
            "            \n",
            "            # Create summary table\n",
            "            summary_df = pd.DataFrame({\n",
            "                'n_trials': trial_counts,\n",
            "                'pbo': pbo_values,\n",
            "                'risk_level': [\n",
            "                    'High' if pbo > 0.5 else 'Moderate' if pbo > 0.3 else 'Low'\n",
            "                    for pbo in pbo_values\n",
            "                ]\n",
            "            })\n",
            "            \n",
            "            print(\"\\nPBO Summary Table:\")\n",
            "            print(summary_df.to_string(index=False))\n",
            "            \n",
            "            # Save summary\n",
            "            summary_path = RUN_DIR / 'pbo_summary.csv'\n",
            "            summary_df.to_csv(summary_path, index=False)\n",
            "            print(f\"\\nSaved summary to: {summary_path}\")\n",
            "        \n",
            "        else:\n",
            "            print(\"Could not compute PBO for trial subsets.\")\n",
            "    \n",
            "    else:\n",
            "        print(f\"Need at least 5 trials for trend analysis (have {n_trials}).\")\n",
            "        print(\"This section demonstrates how PBO changes with number of trials.\")\n",
            "    \n",
            "    # Compare model types if multiple exist\n",
            "    print(\"\\n\" + \"=\"*80)\n",
            "    print(\"Model Type Comparison:\")\n",
            "    print(\"=\"*80 + \"\\n\")\n",
            "    \n",
            "    if 'model_type' in trials_df.columns:\n",
            "        model_types = trials_df['model_type'].unique()\n",
            "        \n",
            "        if len(model_types) > 1:\n",
            "            print(f\"Found {len(model_types)} model types: {list(model_types)}\\n\")\n",
            "            \n",
            "            model_pbo_comparison = []\n",
            "            \n",
            "            for model_type in model_types:\n",
            "                model_df = trials_df[trials_df['model_type'] == model_type].copy()\n",
            "                \n",
            "                if len(model_df) >= 2:\n",
            "                    result = compute_pbo_enhanced(\n",
            "                        model_df,\n",
            "                        metric_name='roc_auc',\n",
            "                        higher_is_better=True\n",
            "                    )\n",
            "                    \n",
            "                    if result['pbo'] is not None:\n",
            "                        model_pbo_comparison.append({\n",
            "                            'model_type': model_type,\n",
            "                            'n_trials': len(model_df),\n",
            "                            'pbo': result['pbo'],\n",
            "                            'lambda_mean': result['lambda_mean']\n",
            "                        })\n",
            "            \n",
            "            if model_pbo_comparison:\n",
            "                comparison_df = pd.DataFrame(model_pbo_comparison)\n",
            "                comparison_df = comparison_df.sort_values('pbo')\n",
            "                \n",
            "                print(\"Model Type PBO Comparison:\")\n",
            "                print(comparison_df.to_string(index=False))\n",
            "                print(f\"\\nBest model type (lowest PBO): {comparison_df.iloc[0]['model_type']}\")\n",
            "        else:\n",
            "            print(f\"Only one model type found: {model_types[0]}\")\n",
            "    else:\n",
            "        print(\"Model type information not available in trials.\")\n",
            "\n",
            "else:\n",
            "    print(f\"No trials found at: {trials_path}\")\n",
            "    print(\"Run Section 4.6.4 first and ensure trials are tracked during training.\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"Section 5.3 Complete\")\n",
            "print(\"=\"*80 + \"\\n\")\n"
        ]
    }

    return [markdown_cell, code_cell]


def main():
    """Add PBO sections to the pipeline notebook."""

    nb_path = Path('/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb')

    print(f"Loading notebook: {nb_path}")
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Find insertion points
    insert_464_after = None
    insert_53_after = None

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '### 4.6.3' in source:
                # Find last code cell after 4.6.3
                for j in range(i + 1, len(nb['cells'])):
                    if nb['cells'][j]['cell_type'] == 'code':
                        insert_464_after = j
                    elif nb['cells'][j]['cell_type'] == 'markdown' and '###' in ''.join(nb['cells'][j]['source']):
                        break
            elif '### 5.2 Equity Curve Uncertainty' in source:
                # Find last code cell after 5.2
                for j in range(i + 1, len(nb['cells'])):
                    if nb['cells'][j]['cell_type'] == 'code':
                        insert_53_after = j
                    elif nb['cells'][j]['cell_type'] == 'markdown' and '##' in ''.join(nb['cells'][j]['source']):
                        break

    if insert_464_after is None:
        print("ERROR: Could not find insertion point for Section 4.6.4")
        return

    if insert_53_after is None:
        print("ERROR: Could not find insertion point for Section 5.3")
        return

    print(f"Inserting Section 4.6.4 after cell {insert_464_after}")
    print(f"Inserting Section 5.3 after cell {insert_53_after}")

    # Create new sections
    section_464 = create_pbo_section_464()
    section_53 = create_pbo_section_53()

    # Insert sections (insert in reverse order to maintain indices)
    # First insert 5.3 (higher index)
    for cell in reversed(section_53):
        nb['cells'].insert(insert_53_after + 1, cell)

    # Then insert 4.6.4 (lower index)
    for cell in reversed(section_464):
        nb['cells'].insert(insert_464_after + 1, cell)

    # Save modified notebook
    backup_path = nb_path.with_suffix('.ipynb.backup')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Saving modified notebook: {nb_path}")
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print("\n✓ Successfully added PBO sections to notebook!")
    print("\nAdded sections:")
    print("  - Section 4.6.4: Enhanced PBO Analysis (after 4.6.3, before 4.7)")
    print("  - Section 5.3: PBO Validation Across Multiple Runs (after 5.2, before Section 6)")
    print("\nBackup saved to:", backup_path)


if __name__ == '__main__':
    main()
