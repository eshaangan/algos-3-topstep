#!/usr/bin/env python3
"""Generate Batch 3 configs (meta-labeling architecture sweep)."""

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
    parser.add_argument("--axes", default=str(Path(__file__).with_name("batch3_metalabeling.yaml")))
    parser.add_argument("--output-dir", default="ml_intraday_v3/experiments/batch3_configs")
    parser.add_argument("--n-configs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.axes, "r") as f:
        cfg = yaml.safe_load(f)

    n = int(args.n_configs or cfg.get("n_configs_default", 200))
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    axes = cfg["axes"]

    u = lhs_indices(n, 6, seed)
    out = []
    for i in range(n):
        architecture = pick(axes["architecture"], u[i, 0])
        recall_threshold = float(pick(axes["primary_model"]["recall_threshold"], u[i, 1]))
        t_value_threshold = float(pick(axes["primary_model"]["t_value_threshold"], u[i, 2]))
        sec_leaves = int(pick(axes["secondary_model"]["num_leaves"], u[i, 3]))
        sec_depth = int(pick(axes["secondary_model"]["max_depth"], u[i, 4]))
        final_threshold = float(pick(axes["final_threshold"], u[i, 5]))

        out.append(
            {
                "exp_id": f"batch3_exp_{i+1:05d}",
                "phase": "batch3",
                "architecture": architecture,
                "labeling_method": "trend_scanning",
                "labeling_params": {
                    "max_lookahead": 25,
                    "min_t_value": t_value_threshold,
                },
                "primary_recall_threshold": recall_threshold,
                "model_kind": "lightgbm",
                "model_params": {
                    "n_estimators": 500,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "min_child_samples": 80,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.2,
                },
                "secondary_model_params": {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": sec_leaves,
                    "max_depth": sec_depth,
                    "min_child_samples": 50,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                },
                "final_threshold": final_threshold,
                "sample_weight": "uniqueness_decay",
                "calibration": "isotonic",
                "cv_method": "cpcv",
                "cv_n_splits": 5,
                "cv_n_test_splits": 2,
                "cv_embargo_pct": 0.01,
                "cv_purge_pct": 0.02,
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
