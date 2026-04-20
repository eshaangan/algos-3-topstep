#!/usr/bin/env python3
"""
Run baseline vs HMM feature configs on standalone_viability; write merged summary.

Requires: pip install hmmlearn>=0.3.0 (for the HMM arm).
"""

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
        "overall_failures": summary.get("overall_failures"),
    }


def main() -> None:
    out_root = PROJECT_ROOT / "ml_intraday_v3/experiments/results/hmm_viability_ab"
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

    baseline_dir = out_root / "baseline"
    hmm_dir = out_root / "hmm_features"

    baseline_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        output_dir=baseline_dir,
    )
    hmm_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/features_hmm_experiment.yaml",
        output_dir=hmm_dir,
    )

    promotion_criteria = (
        "HMM is not promoted automatically. Compare overall_total_pnl_usd, pass_ratio, "
        "and per-window gate_result; require hmm >= baseline on agreed metrics before "
        "switching live features.yaml."
    )

    merged = {
        "acceptance_config": "ml_intraday_v3/configs/standalone_viability.yaml",
        "baseline_features": "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        "hmm_features": "ml_intraday_v3/configs/live_dual_meta_mes_real/features_hmm_experiment.yaml",
        "hmm_regime_stability": {
            "covariance_type": "diag",
            "n_iter": 300,
            "tol": 0.001,
            "note": "Set in features_hmm_experiment.yaml; build_hmm_regime_features passes these to HMMRegimeDetector.",
        },
        "baseline": _pick_summary(baseline_summary),
        "hmm": _pick_summary(hmm_summary),
        "promotion_criteria": promotion_criteria,
    }

    out_json = out_root / "hmm_viability_ab_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
