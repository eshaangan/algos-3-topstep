#!/usr/bin/env python3
"""
Stressed execution: YTD threshold sweep + standalone_viability (baseline features).

Uses execution_spec_stress.yaml (higher slippage / commission).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.experiments.run_standalone_topstep_candidate import run_candidate


def main() -> None:
    stress_spec = PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec_stress.yaml"
    if not stress_spec.is_file():
        raise SystemExit(f"Missing {stress_spec}")

    # 1) YTD-only threshold sweep (train once, many backtests) under stress fills/costs
    ytd_out = PROJECT_ROOT / "ml_intraday_v3/experiments/results/decision_threshold_sweep_ytd_2026_stress"
    cmd = [
        sys.executable,
        "-m",
        "ml_intraday_v3.experiments.run_decision_threshold_sweep",
        "--acceptance",
        "ml_intraday_v3/experiments/_ytd_2026_contract_sweep_config.yaml",
        "--execution-spec",
        "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec_stress.yaml",
        "--output-dir",
        "ml_intraday_v3/experiments/results/decision_threshold_sweep_ytd_2026_stress",
        "--min-trades",
        "10",
        "--max-drawdown",
        "3000",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    # 2) Full viability promotion under stress (baseline features only)
    via_out = PROJECT_ROOT / "ml_intraday_v3/experiments/results/stress_viability_baseline"
    summary = run_candidate(
        data_path=PROJECT_ROOT / "data/processed/mes_bars_databento_rth.h5",
        hdf_key="bars_5min",
        training_cfg_path=PROJECT_ROOT
        / "ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml",
        labeling_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/labeling.yaml",
        feature_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
        execution_spec_path=stress_spec,
        backtest_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml",
        risk_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml",
        acceptance_cfg_path=PROJECT_ROOT / "ml_intraday_v3/configs/standalone_viability.yaml",
        output_dir=via_out,
    )

    merged = {
        "execution_spec_stress": str(stress_spec.relative_to(PROJECT_ROOT)),
        "ytd_threshold_sweep_dir": str(ytd_out.relative_to(PROJECT_ROOT)),
        "viability_output_dir": str(via_out.relative_to(PROJECT_ROOT)),
        "viability_summary": {
            "passed": summary.get("passed"),
            "passed_windows": summary.get("passed_windows"),
            "total_windows": summary.get("total_windows"),
            "overall_total_pnl_usd": summary.get("overall_total_pnl_usd"),
        },
    }
    out_json = PROJECT_ROOT / "ml_intraday_v3/experiments/results/stress_oos_checks_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
