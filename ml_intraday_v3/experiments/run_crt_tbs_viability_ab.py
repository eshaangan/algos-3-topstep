#!/usr/bin/env python3
"""Run baseline vs CRT/TBS feature configs on standalone_viability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.experiments.run_standalone_topstep_candidate import run_candidate


def _pick_summary(summary: dict) -> dict:
    return {
        "passed": summary.get("passed"),
        "passed_windows": summary.get("passed_windows"),
        "total_windows": summary.get("total_windows"),
        "pass_ratio": summary.get("pass_ratio"),
        "overall_total_pnl_usd": summary.get("overall_total_pnl_usd"),
        "overall_total_trades": summary.get("overall_total_trades"),
        "overall_profit_factor": summary.get("overall_profit_factor"),
        "overall_max_drawdown_usd": summary.get("overall_max_drawdown_usd"),
        "overall_failures": summary.get("overall_failures"),
    }


def main() -> None:
    out_root = PROJECT_ROOT / "ml_intraday_v3/experiments/results/crt_tbs_viability_ab"
    out_root.mkdir(parents=True, exist_ok=True)

    common = dict(
        data_path=PROJECT_ROOT / "data/processed/mes_bars_databento_rth.h5",
        hdf_key="bars_5min",
        training_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml",
        labeling_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/labeling.yaml",
        execution_spec_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec.yaml",
        backtest_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml",
        risk_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml",
        acceptance_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/standalone_viability.yaml",
    )

    baseline_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/features_structure_context.yaml",
        output_dir=out_root / "baseline_structure_context",
    )
    crt_tbs_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/features_crt_tbs.yaml",
        output_dir=out_root / "crt_tbs_features",
    )

    merged = {
        "acceptance_config": "ml_intraday_v3/configs/standalone_viability.yaml",
        "baseline_features": "ml_intraday_v3/configs/features_structure_context.yaml",
        "crt_tbs_features": "ml_intraday_v3/configs/features_crt_tbs.yaml",
        "promotion_criteria": (
            "CRT/TBS is not promoted automatically. Require equal or better pass_ratio, "
            "overall_total_pnl_usd, drawdown behavior, and per-window stability before "
            "using it in live features."
        ),
        "baseline": _pick_summary(baseline_summary),
        "crt_tbs": _pick_summary(crt_tbs_summary),
    }

    out_json = out_root / "crt_tbs_viability_ab_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
