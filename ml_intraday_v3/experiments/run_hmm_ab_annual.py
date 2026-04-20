#!/usr/bin/env python3
"""HMM feature A/B on the multi-year annual OOS windows (2022-2025_ytd).

Extends the single-window hmm_ab_ytd experiment (results/hmm_ab_ytd_*) to the
full annual horizon defined in experiments/_oos_long_history_annual.yaml.
Writes two result folders; compare them against the strict gate config
experiments/_oos_long_history_annual_strict.yaml to gate promotion.

Requires: pip install hmmlearn>=0.3.0 for the HMM arm.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.experiments.run_standalone_topstep_candidate import run_candidate


def _slim(summary: dict) -> dict:
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
    out_root = PROJECT_ROOT / "ml_intraday_v3/experiments/results/hmm_ab_annual"
    out_root.mkdir(parents=True, exist_ok=True)

    common = dict(
        data_path=PROJECT_ROOT / "data/processed/mes_bars_databento_rth.h5",
        hdf_key="bars_5min",
        training_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml",
        labeling_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/labeling.yaml",
        execution_spec_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec.yaml",
        backtest_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml",
        risk_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml",
        acceptance_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/experiments/_oos_long_history_annual.yaml",
    )

    baseline_dir = out_root / "baseline"
    hmm_dir = out_root / "hmm"

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting baseline arm -> {baseline_dir}")
    baseline_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        output_dir=baseline_dir,
    )
    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Baseline done in {t1 - t0:.0f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Starting HMM arm -> {hmm_dir}")
    hmm_summary = run_candidate(
        **common,
        feature_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/live_dual_meta_mes_real/features_hmm_experiment.yaml",
        output_dir=hmm_dir,
    )
    t2 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] HMM done in {t2 - t1:.0f}s  total {t2 - t0:.0f}s")

    combined = {
        "windows_config": "ml_intraday_v3/experiments/_oos_long_history_annual.yaml",
        "baseline_features": "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        "hmm_features": "ml_intraday_v3/configs/live_dual_meta_mes_real/features_hmm_experiment.yaml",
        "baseline": _slim(baseline_summary),
        "hmm": _slim(hmm_summary),
        "delta_pnl_usd": (hmm_summary.get("overall_total_pnl_usd") or 0)
        - (baseline_summary.get("overall_total_pnl_usd") or 0),
        "seconds_baseline": t1 - t0,
        "seconds_hmm": t2 - t1,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
