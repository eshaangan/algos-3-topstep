#!/usr/bin/env python3
"""
Data Corruption Diagnostic Tool

Analyzes raw CSV data to identify and quantify corruption issues:
- Spread contracts (symbols with "-") vs outright contracts
- Negative prices (impossible for ES/MES futures)
- Invalid low prices (< 100, should be 4000+)
- Other data quality issues

Generates a comprehensive JSON report with statistics and sample rows.

Usage:
    python ml_intraday_v3/data/diagnose_corruption.py <csv_path>

Example:
    python ml_intraday_v3/data/diagnose_corruption.py \
        "data/raw/GLBX-20251220-LWFB9HCEL5/glbx-mdp3-20100606-20251219.ohlcv-1m.csv"
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def diagnose_corruption(csv_path: Path) -> Dict[str, Any]:
    """
    Analyze raw CSV for data corruption issues.

    Args:
        csv_path: Path to raw CSV file

    Returns:
        Dictionary with diagnostic statistics
    """
    print(f"\n{'='*70}")
    print(f"DATA CORRUPTION DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"File: {csv_path}")
    print(f"Size: {csv_path.stat().st_size / (1024**2):.1f} MB")

    # Load CSV
    print(f"\nLoading CSV...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Initialize report
    report = {
        "file_path": str(csv_path),
        "file_size_mb": csv_path.stat().st_size / (1024**2),
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # Check for symbol column
    symbol_col = None
    for col in ['symbol', 'Symbol', 'SYMBOL', 'contract', 'Contract']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        print(f"\n⚠️  WARNING: No symbol column found. Cannot identify spread contracts.")
        report["has_symbol_column"] = False
    else:
        print(f"\n✓ Found symbol column: '{symbol_col}'")
        report["has_symbol_column"] = True
        report["symbol_column"] = symbol_col

        # Analyze spread vs outright contracts
        print(f"\n{'='*70}")
        print(f"SPREAD CONTRACT ANALYSIS")
        print(f"{'='*70}")

        # Identify spread contracts (contain "-")
        is_spread = df[symbol_col].astype(str).str.contains('-', na=False)
        n_spread = is_spread.sum()
        n_outright = (~is_spread).sum()
        pct_spread = (n_spread / len(df)) * 100

        print(f"Spread contracts (contain '-'):  {n_spread:>10,} ({pct_spread:>5.1f}%)")
        print(f"Outright contracts:              {n_outright:>10,} ({100-pct_spread:>5.1f}%)")

        report["spread_contracts"] = {
            "count": int(n_spread),
            "percentage": float(pct_spread),
            "sample_symbols": list(df[is_spread][symbol_col].unique()[:10]),
        }
        report["outright_contracts"] = {
            "count": int(n_outright),
            "percentage": float(100 - pct_spread),
            "sample_symbols": list(df[~is_spread][symbol_col].unique()[:10]),
        }

        if n_spread > 0:
            print(f"\n⚠️  CORRUPTION DETECTED: {n_spread:,} spread contract rows")
            print(f"   Sample spread symbols:")
            for sym in df[is_spread][symbol_col].unique()[:5]:
                print(f"     - {sym}")

    # Analyze price data
    print(f"\n{'='*70}")
    print(f"PRICE DATA ANALYSIS")
    print(f"{'='*70}")

    price_cols = []
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            price_cols.append(col)

    if not price_cols:
        print(f"⚠️  WARNING: No OHLC columns found")
        report["has_price_data"] = False
    else:
        print(f"✓ Found price columns: {price_cols}")
        report["has_price_data"] = True

        # Analyze each price column
        price_stats = {}
        for col in price_cols:
            prices = df[col]

            n_negative = (prices < 0).sum()
            n_low = ((prices >= 0) & (prices < 100)).sum()
            n_valid = (prices >= 100).sum()

            stats = {
                "min": float(prices.min()),
                "max": float(prices.max()),
                "mean": float(prices.mean()),
                "negative_count": int(n_negative),
                "negative_pct": float((n_negative / len(df)) * 100),
                "low_price_count": int(n_low),
                "low_price_pct": float((n_low / len(df)) * 100),
                "valid_count": int(n_valid),
                "valid_pct": float((n_valid / len(df)) * 100),
            }
            price_stats[col] = stats

            print(f"\n{col.upper()}:")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Negative prices:      {n_negative:>10,} ({stats['negative_pct']:>5.1f}%)")
            print(f"  Low prices (0-100):   {n_low:>10,} ({stats['low_price_pct']:>5.1f}%)")
            print(f"  Valid prices (≥100):  {n_valid:>10,} ({stats['valid_pct']:>5.1f}%)")

        report["price_analysis"] = price_stats

        # Find worst corrupted rows
        any_negative = df[price_cols].lt(0).any(axis=1)
        any_low = df[price_cols].ge(0).any(axis=1) & df[price_cols].lt(100).any(axis=1)

        if any_negative.sum() > 0:
            print(f"\n⚠️  CORRUPTION: {any_negative.sum():,} rows with negative prices")
            sample_negative = df[any_negative].head(5)
            print(f"\nSample corrupted rows (negative prices):")
            print(sample_negative[price_cols + ([symbol_col] if symbol_col else [])])

            report["sample_negative_price_rows"] = sample_negative.to_dict('records')[:5]

        if any_low.sum() > 0:
            print(f"\n⚠️  CORRUPTION: {any_low.sum():,} rows with invalid low prices (0-100)")
            sample_low = df[any_low].head(5)
            print(f"\nSample corrupted rows (low prices):")
            print(sample_low[price_cols + ([symbol_col] if symbol_col else [])])

            report["sample_low_price_rows"] = sample_low.to_dict('records')[:5]

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    issues = []
    if report.get("spread_contracts", {}).get("count", 0) > 0:
        issues.append(f"✗ {report['spread_contracts']['count']:,} spread contract rows")

    if report.get("has_price_data"):
        for col, stats in report.get("price_analysis", {}).items():
            if stats["negative_count"] > 0:
                issues.append(f"✗ {stats['negative_count']:,} negative {col} prices")
            if stats["low_price_count"] > 0:
                issues.append(f"✗ {stats['low_price_count']:,} invalid low {col} prices")

    if issues:
        print(f"\n🚨 CORRUPTION DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        report["corruption_detected"] = True
        report["issues"] = issues
    else:
        print(f"\n✅ No corruption detected - data appears clean")
        report["corruption_detected"] = False
        report["issues"] = []

    return report


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        print(f"\nExample:")
        print(f'  {sys.argv[0]} "data/raw/GLBX-20251220-LWFB9HCEL5/glbx-mdp3-20100606-20251219.ohlcv-1m.csv"')
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)

    # Run diagnostics
    report = diagnose_corruption(csv_path)

    # Write report
    output_path = csv_path.parent / "corruption_report.json"
    print(f"\n{'='*70}")
    print(f"Writing report to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✓ Report saved")
    print(f"\nTo view report:")
    print(f"  cat {output_path}")
    print(f"  # or")
    print(f"  python -m json.tool {output_path}")
    print(f"{'='*70}\n")

    # Exit with error code if corruption detected
    if report.get("corruption_detected", False):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
