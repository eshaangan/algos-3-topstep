#!/usr/bin/env python3
"""
Fix notebook to use 1m bar size instead of 5m.
"""

import json
import sys
from pathlib import Path

def fix_notebook_bar_size(notebook_path):
    """Update notebook to use 1m instead of 5m."""

    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    changes_made = 0

    # Iterate through all cells
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])

            # Process each line
            new_source = []
            for line in source:
                original_line = line

                # Replace common patterns
                replacements = [
                    ('BAR_SIZE_5M_DIR = "bar_size=5m"', 'BAR_SIZE_1M_DIR = "bar_size=1m"'),
                    ('BAR_SIZE_5M_DIR', 'BAR_SIZE_1M_DIR'),
                    ('bar_size=5m', 'bar_size=1m'),
                    ('analyze_data_quality("5m")', 'analyze_data_quality("1m")'),
                    ('analyze_features("5m")', 'analyze_features("1m")'),
                    ('analyze_labels("5m")', 'analyze_labels("1m")'),
                    ('analyze_cv("5m")', 'analyze_cv("1m")'),
                    ('analyze_training("5m"', 'analyze_training("1m"'),
                    ('analyze_rare_events("5m")', 'analyze_rare_events("1m")'),
                    ('analyze_backtest("5m")', 'analyze_backtest("1m")'),
                    ('analyze_pbo("5m")', 'analyze_pbo("1m")'),
                    ('"5m"', '"1m"'),  # Generic string replacement
                ]

                for old, new in replacements:
                    if old in line:
                        line = line.replace(old, new)
                        if line != original_line:
                            changes_made += 1
                            print(f"Changed: {original_line.strip()[:60]}... → {line.strip()[:60]}...")

                new_source.append(line)

            cell['source'] = new_source

    # Write back
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Writing updated notebook: {notebook_path}")
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n✓ Done! Made {changes_made} changes.")
    print(f"✓ Backup saved to: {backup_path}")

    return changes_made

if __name__ == "__main__":
    notebook_path = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb")

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    changes = fix_notebook_bar_size(notebook_path)

    if changes > 0:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Updated {changes} references from 5m to 1m")
        print("Backup created with .backup extension")
        print("\nRestart your Jupyter kernel and re-run the notebook!")
        print("="*70)
    else:
        print("No changes needed - notebook already uses 1m")
