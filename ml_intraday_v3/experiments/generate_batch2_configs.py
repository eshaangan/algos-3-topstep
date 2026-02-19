#!/usr/bin/env python3
"""Generate Batch 2 configs (CV + feature combinations)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def lhs_indices(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((n_samples, n_dims), dtype=float)
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        m[:, d] = (perm + rng.random(n_samples)) / n_samples
    return m


def pick(values, u: float):
    vals = list(values)
    idx = min(int(np.floor(u * len(vals))), len(vals) - 1)
    return vals[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", default=str(Path(__file__).with_name("batch2_cv_features.yaml")))
    parser.add_argument("--output-dir", default="ml_intraday_v3/experiments/batch2_configs")
    parser.add_argument("--n-configs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.axes, "r") as f:
        cfg = yaml.safe_load(f)

    n = int(args.n_configs or cfg.get("n_configs_default", 200))
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    axes = cfg["axes"]

    u = lhs_indices(n, 7, seed)
    out = []
    for i in range(n):
        cv_method = pick(axes["cv_method"], u[i, 0])
        feature_set = pick(axes["feature_set"], u[i, 1])
        calibration = pick(axes["calibration"], u[i, 2])

        num_leaves = int(pick(axes["model_params"]["num_leaves"], u[i, 3]))
        max_depth = int(pick(axes["model_params"]["max_depth"], u[i, 4]))
        min_data_in_leaf = int(pick(axes["model_params"]["min_data_in_leaf"], u[i, 5]))

        features_cfg = {}
        if feature_set in {"baseline_hmm", "baseline_fracdiff_hmm"}:
            features_cfg["hmm_regime"] = {"enabled": True}
        if feature_set in {"baseline_multiresolution"}:
            features_cfg["multi_resolution"] = {
                "enabled": True,
                "resolutions_min": [15, 30, 60],
                "prefix": "mr",
                "align_method": "ffill",
            }
        if feature_set in {"baseline_fracdiff_hmm"}:
            features_cfg["fractional_diff"] = {
                "enabled": True,
                "d": 0.4,
                "threshold": 0.01,
                "apply_to": ["close", "ema_13", "ema_34", "price_vs_vwap"],
            }

        out.append(
            {
                "exp_id": f"batch2_exp_{i+1:05d}",
                "phase": "batch2",
                "cv_method": cv_method,
                "cv_n_splits": 5,
                "cv_n_test_splits": 2,
                "cv_embargo_pct": 0.01 if cv_method == "cpcv_embargo" else 0.0,
                "cv_purge_pct": 0.02,
                "feature_set_name": feature_set,
                "features_config": features_cfg,
                "sample_weight": "uniqueness",
                "calibration": calibration,
                "model_kind": "lightgbm",
                "model_params": {
                    "n_estimators": 500,
                    "learning_rate": 0.03,
                    "num_leaves": num_leaves,
                    "max_depth": max_depth,
                    "min_child_samples": min_data_in_leaf,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.2,
                },
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in out:
        with open(out_dir / f"{row['exp_id']}.json", "w") as f:
            json.dump(row, f, indent=2)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"n_configs": len(out), "seed": seed, "axes": str(args.axes)}, f, indent=2)

    print(f"Generated {len(out)} configs in {out_dir}")


if __name__ == "__main__":
    main()
