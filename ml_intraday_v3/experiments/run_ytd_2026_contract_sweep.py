#!/usr/bin/env python3
"""
Backtest 2026 YTD for multiple contract sizes with train-once + proportional risk.

Trains each promotion window once, then only re-runs the simulator per contract size.
OOS window: `_ytd_2026_contract_sweep_config.yaml`.
See README_SCALING_AND_CONTRACTS.md for risk policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from ml_intraday_v3.experiments.run_standalone_topstep_candidate import run_candidate_contract_variants


def _parse_contract_counts(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --contracts")
    out: list[int] = []
    for p in parts:
        n = int(p)
        if n <= 0:
            raise ValueError(f"contract count must be positive, got {n}")
        out.append(n)
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="YTD 2026 contract sweep (dual-meta MES, train once per window)")
    parser.add_argument(
        "--contracts",
        default="1,2,3,4,5",
        help="Comma-separated contract counts (e.g. 1,2,3,4,5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results root (default: ml_intraday_v3/experiments/results/ytd_2026_contract_sweep)",
    )
    parser.add_argument(
        "--no-scale-risk",
        action="store_true",
        help="Keep dollar limits fixed at risk.yaml values (not recommended for cross-size comparison)",
    )
    args = parser.parse_args()

    out_root = args.output_dir or (
        PROJECT_ROOT / "ml_intraday_v3/experiments/results/ytd_2026_contract_sweep"
    )
    out_root = out_root if out_root.is_absolute() else PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    contract_counts = _parse_contract_counts(args.contracts)

    acceptance = PROJECT_ROOT / "ml_intraday_v3/experiments/_ytd_2026_contract_sweep_config.yaml"
    backtest_cfg = PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml"

    summaries = run_candidate_contract_variants(
        data_path=PROJECT_ROOT / "data/processed/mes_bars_databento_rth.h5",
        hdf_key="bars_5min",
        training_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml",
        labeling_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/labeling.yaml",
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        execution_spec_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec.yaml",
        backtest_cfg_path=backtest_cfg,
        risk_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml",
        acceptance_cfg_path=acceptance,
        output_dir=out_root,
        contract_counts=contract_counts,
        scale_risk_with_contracts=not args.no_scale_risk,
    )

    aggregate = [{"contracts": n, "promotion_summary": s} for n, s in sorted(summaries.items())]
    out_json = out_root / "aggregate_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"Wrote {out_json} and contract_variants_aggregate.json")


if __name__ == "__main__":
    main()
