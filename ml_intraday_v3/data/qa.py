"""
Quality assurance module.

Runs QA checks on OHLCV data and generates JSON reports.
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QAViolationError(Exception):
    """Raised when QA checks fail in fail-fast mode."""

    pass


@dataclass
class QAReport:
    """
    QA report container.

    Attributes:
        passed: Whether all checks passed
        total_bars: Total number of bars
        total_checks: Number of checks performed
        passed_checks: Number of passed checks
        failed_checks: List of failed check names
        checks: Dict mapping check name to check result details
        thresholds: Dict of threshold values used
    """

    passed: bool
    total_bars: int
    total_checks: int
    passed_checks: int
    failed_checks: List[str]
    checks: Dict[str, Dict]
    thresholds: Dict[str, any]

    def to_json(self, path: Path) -> None:
        """Write report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and ensure JSON-serializable types
        data = asdict(self)

        # Convert numpy types to Python native types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            else:
                return obj

        data = convert_types(data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Wrote QA report to {path}")

    @staticmethod
    def from_json(path: Path) -> "QAReport":
        """Load report from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return QAReport(**data)


def run_qa_checks(
    df: pd.DataFrame,
    checks: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, any]] = None,
    qa_fail_fast: bool = True,
) -> QAReport:
    """
    Run QA checks on OHLCV DataFrame.

    Available checks:
    - monotonic_index: Timestamps strictly increasing
    - no_duplicates: No duplicate timestamps
    - missing_bar_pct: Missing bars per day (if is_synthetic column exists)
    - ohlc_validity: Low <= min(O,C) <= max(O,C) <= High
    - volume_sanity: Volume >= 0
    - price_range: All OHLC prices within [min_valid_price, max_valid_price]

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        checks: List of check names to run (None = run all)
        thresholds: Dict of threshold values:
            - max_missing_bar_pct_per_day: Max % missing bars per day
            - max_ohlc_violations: Max number of OHLC violations
            - max_duplicate_timestamps: Max number of duplicate timestamps
            - min_valid_price: Minimum valid price for OHLC (default: 100.0)
            - max_valid_price: Maximum valid price for OHLC (default: 10000.0)
        qa_fail_fast: If True, raise QAViolationError on failures (DEFAULT)
            If False, log warnings and continue (WARNING MODE - not recommended)

    Returns:
        QAReport with check results

    Raises:
        ValueError: If index is not DatetimeIndex
        QAViolationError: If qa_fail_fast=True and any check fails
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Default checks
    if checks is None:
        checks = [
            "monotonic_index",
            "no_duplicates",
            "no_duplicates_after_merge",
            "missing_bar_pct",
            "ohlc_validity",
            "volume_sanity",
            "price_range",
        ]

    # Default thresholds
    if thresholds is None:
        thresholds = {
            "max_missing_bar_pct_per_day": 10.0,
            "max_ohlc_violations": 0,
            "max_duplicate_timestamps": 0,
            "min_valid_price": 100.0,
            "max_valid_price": 10000.0,
        }

    logger.info(f"Running {len(checks)} QA checks on {len(df)} bars")
    if qa_fail_fast:
        logger.info("QA fail-fast mode ENABLED - will halt on violations")
    else:
        logger.warning("QA fail-fast mode DISABLED - violations will be logged as warnings")

    check_results = {}
    failed_checks = []

    # 1. Monotonic index check
    if "monotonic_index" in checks:
        is_monotonic = df.index.is_monotonic_increasing
        check_results["monotonic_index"] = {
            "passed": is_monotonic,
            "message": "Index is monotonic increasing"
            if is_monotonic
            else "Index is NOT monotonic increasing",
        }
        if not is_monotonic:
            failed_checks.append("monotonic_index")

    # 2. No duplicates check
    if "no_duplicates" in checks:
        n_duplicates = df.index.duplicated().sum()
        threshold = thresholds.get("max_duplicate_timestamps", 0)
        no_dups = n_duplicates <= threshold
        check_results["no_duplicates"] = {
            "passed": no_dups,
            "n_duplicates": int(n_duplicates),
            "threshold": threshold,
            "message": f"Found {n_duplicates} duplicate timestamps (threshold: {threshold})",
        }
        if not no_dups:
            failed_checks.append("no_duplicates")

    # 2b. No duplicates after merge check
    if "no_duplicates_after_merge" in checks:
        n_duplicates = df.index.duplicated().sum()
        threshold = thresholds.get("max_duplicate_timestamps", 0)
        no_dups = n_duplicates <= threshold
        check_results["no_duplicates_after_merge"] = {
            "passed": no_dups,
            "n_duplicates": int(n_duplicates),
            "threshold": threshold,
            "message": f"Found {n_duplicates} duplicate timestamps after merge (threshold: {threshold})",
        }
        if not no_dups:
            failed_checks.append("no_duplicates_after_merge")

    # 3. Missing bar percentage check
    if "missing_bar_pct" in checks:
        if "is_synthetic" in df.columns:
            # Compute per-day missing %
            df_temp = df.copy()
            df_temp["_date"] = df_temp.index.date

            per_day_pct = {}
            max_pct_day = 0.0
            worst_date = None

            for date, group in df_temp.groupby("_date"):
                n_synthetic = group["is_synthetic"].sum()
                total = len(group)
                pct = (n_synthetic / total * 100) if total > 0 else 0.0
                per_day_pct[str(date)] = round(pct, 2)

                if pct > max_pct_day:
                    max_pct_day = pct
                    worst_date = str(date)

            # Check against threshold
            threshold = thresholds.get("max_missing_bar_pct_per_day", 10.0)
            passed = max_pct_day <= threshold

            check_results["missing_bar_pct"] = {
                "passed": passed,
                "max_pct_per_day": round(max_pct_day, 2),
                "worst_date": worst_date,
                "threshold": threshold,
                "per_day_pct": per_day_pct,
                "message": f"Max missing % per day: {max_pct_day:.2f}% on {worst_date} (threshold: {threshold}%)",
            }

            if not passed:
                failed_checks.append("missing_bar_pct")
        else:
            check_results["missing_bar_pct"] = {
                "passed": True,
                "message": "No is_synthetic column found - skipping check",
            }

    # 4. OHLC validity check
    if "ohlc_validity" in checks:
        required_cols = ["open", "high", "low", "close"]
        if all(col in df.columns for col in required_cols):
            # Check: low <= min(open, close) and max(open, close) <= high
            # Allow small floating point tolerance
            tol = 1e-6

            low_violation = df["low"] > df[["open", "close"]].min(axis=1) + tol
            high_violation = df["high"] < df[["open", "close"]].max(axis=1) - tol

            n_low_violations = low_violation.sum()
            n_high_violations = high_violation.sum()
            n_total_violations = (low_violation | high_violation).sum()

            threshold = thresholds.get("max_ohlc_violations", 0)
            passed = n_total_violations <= threshold

            check_results["ohlc_validity"] = {
                "passed": passed,
                "n_low_violations": int(n_low_violations),
                "n_high_violations": int(n_high_violations),
                "n_total_violations": int(n_total_violations),
                "threshold": threshold,
                "message": f"Found {n_total_violations} OHLC violations (threshold: {threshold})",
            }

            if not passed:
                failed_checks.append("ohlc_validity")
        else:
            check_results["ohlc_validity"] = {
                "passed": True,
                "message": f"Missing OHLC columns - skipping check",
            }

    # 5. Volume sanity check
    if "volume_sanity" in checks:
        if "volume" in df.columns:
            negative_volume = df["volume"] < 0
            n_negative = negative_volume.sum()
            passed = n_negative == 0

            check_results["volume_sanity"] = {
                "passed": passed,
                "n_negative_volume": int(n_negative),
                "message": f"Found {n_negative} bars with negative volume",
            }

            if not passed:
                failed_checks.append("volume_sanity")
        else:
            check_results["volume_sanity"] = {
                "passed": True,
                "message": "No volume column found - skipping check",
            }

    # Check: price_range - validate OHLC within valid range
    if "price_range" in checks:
        min_valid_price = thresholds.get("min_valid_price", 100.0)
        max_valid_price = thresholds.get("max_valid_price", 10000.0)

        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if price_cols:
            violations = pd.Series(False, index=df.index)
            for col in price_cols:
                violations |= (df[col] < min_valid_price) | (df[col] > max_valid_price)

            n_violations = violations.sum()
            passed = n_violations == 0

            if passed:
                check_results["price_range"] = {
                    "passed": True,
                    "message": f"All {len(price_cols)} price columns within range [{min_valid_price}, {max_valid_price}]",
                }
            else:
                check_results["price_range"] = {
                    "passed": False,
                    "message": (
                        f"Found {n_violations} bars with prices outside valid range "
                        f"[{min_valid_price}, {max_valid_price}]"
                    ),
                    "n_violations": int(n_violations),
                    "min_valid_price": min_valid_price,
                    "max_valid_price": max_valid_price,
                }
                failed_checks.append("price_range")
        else:
            check_results["price_range"] = {
                "passed": True,
                "message": "No price columns found - skipping check",
            }

    # Build report
    total_checks = len(checks)
    passed_checks = total_checks - len(failed_checks)
    overall_passed = len(failed_checks) == 0

    report = QAReport(
        passed=overall_passed,
        total_bars=len(df),
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        checks=check_results,
        thresholds=thresholds,
    )

    if overall_passed:
        logger.info(f"All {total_checks} QA checks PASSED")
    else:
        error_msg = (
            f"QA checks FAILED: {len(failed_checks)}/{total_checks} checks failed\n"
            f"Failed checks: {failed_checks}\n"
        )

        # Add details for each failed check
        for check_name in failed_checks:
            check_result = check_results.get(check_name, {})
            error_msg += f"  - {check_name}: {check_result.get('message', 'No details')}\n"

        if qa_fail_fast:
            logger.error(error_msg)
            logger.error("QA fail-fast mode enabled - halting pipeline")
            raise QAViolationError(error_msg)
        else:
            logger.warning(error_msg)
            logger.warning("QA fail-fast mode disabled - continuing with warnings")

    return report
