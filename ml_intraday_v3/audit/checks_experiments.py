"""
Experiment artifact audits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def check_experiment_leaderboards(exp_root: Path) -> dict:
    if not exp_root.exists():
        return {"status": "SKIP", "reason": "no_experiments"}

    leaderboard_files = list(exp_root.glob("leaderboard_bar_size=*.parquet"))
    if not leaderboard_files:
        return {"status": "SKIP", "reason": "no_leaderboards"}

    missing_dirs = 0
    total_rows = 0
    for lb in leaderboard_files:
        df = pd.read_parquet(lb)
        total_rows += len(df)
        if "exp_id" not in df.columns:
            return {"status": "FAIL", "reason": "missing_exp_id_column"}
        for exp_id in df["exp_id"].dropna().unique():
            exp_dir = exp_root / f"exp_{exp_id}"
            if not exp_dir.exists():
                missing_dirs += 1

    status = "PASS" if missing_dirs == 0 else "FAIL"
    return {
        "status": status,
        "missing_exp_dirs": missing_dirs,
        "leaderboard_rows": total_rows,
    }


def check_experiment_diagnostics(exp_root: Path) -> dict:
    if not exp_root.exists():
        return {"status": "SKIP", "reason": "no_experiments"}

    exp_dirs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    if not exp_dirs:
        return {"status": "SKIP", "reason": "no_experiment_dirs"}

    issues = 0
    details = []
    for exp_dir in exp_dirs:
        config_path = exp_dir / "config_snapshot.json"
        results_path = exp_dir / "results.parquet"
        pbo_path = exp_dir / "pbo.json"

        if not config_path.exists() or not results_path.exists():
            issues += 1
            details.append({"exp_dir": exp_dir.name, "reason": "missing_inputs"})
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        selection_metric = (
            config.get("grid_config", {})
            .get("diagnostics", {})
            .get("selection_metric", "total_pnl_usd")
        )
        cv_kind = config.get("cv_kind")

        results = pd.read_parquet(results_path)
        if selection_metric not in results.columns:
            issues += 1
            details.append(
                {
                    "exp_dir": exp_dir.name,
                    "reason": "selection_metric_missing",
                    "metric": selection_metric,
                }
            )

        if pbo_path.exists():
            with open(pbo_path, "r") as f:
                pbo = json.load(f)
            not_applicable = pbo.get("not_applicable") or pbo.get("disabled")
            if cv_kind != "cpcv" and not not_applicable:
                issues += 1
                details.append(
                    {
                        "exp_dir": exp_dir.name,
                        "reason": "pbo_should_be_not_applicable",
                        "cv_kind": cv_kind,
                    }
                )
            if cv_kind == "cpcv":
                bar_sizes = pbo.get("bar_sizes", {})
                for bar_size, payload in bar_sizes.items():
                    if "definitions" not in payload:
                        issues += 1
                        details.append(
                            {
                                "exp_dir": exp_dir.name,
                                "reason": "pbo_definitions_missing",
                                "bar_size": bar_size,
                            }
                        )

    status = "PASS" if issues == 0 else "FAIL"
    return {"status": status, "issues": issues, "details": details}
