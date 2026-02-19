#!/usr/bin/env python3
"""Generate Batch 1 configs (labeling-focused) with LHS coverage."""

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
    parser.add_argument("--axes", default=str(Path(__file__).with_name("batch1_labeling.yaml")))
    parser.add_argument("--output-dir", default="ml_intraday_v3/experiments/batch1_configs")
    parser.add_argument("--n-configs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.axes, "r") as f:
        cfg = yaml.safe_load(f)

    n = int(args.n_configs or cfg.get("n_configs_default", 200))
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    axes = cfg["axes"]

    u = lhs_indices(n, 8, seed)
    out = []
    for i in range(n):
        labeling_method = pick(axes["labeling_method"], u[i, 0])
        sample_weight = pick(axes["sample_weight"], u[i, 1])
        features = pick(axes["features"], u[i, 2])

        item = {
            "exp_id": f"batch1_exp_{i+1:05d}",
            "phase": "batch1",
            "labeling_method": labeling_method,
            "sample_weight": sample_weight,
            "model_kind": "lightgbm",
            "model_params": {
                "n_estimators": 400,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.2,
            },
            "cv_method": "kfold",
            "cv_n_splits": 5,
            "cv_embargo_bars": 12,
            "features_config": {},
            "feature_set_name": features,
        }

        if features == "baseline_fracdiff":
            item["features_config"]["fractional_diff"] = {
                "enabled": True,
                "d": 0.4,
                "threshold": 0.01,
                "apply_to": ["close", "ema_13", "ema_34", "price_vs_vwap"],
            }

        if labeling_method == "triple_barrier":
            item["labeling_params"] = {
                "pt_mult": pick(axes["triple_barrier"]["pt_mult"], u[i, 3]),
                "sl_mult": pick(axes["triple_barrier"]["sl_mult"], u[i, 4]),
                "time_mult": int(pick(axes["triple_barrier"]["time_mult"], u[i, 5])),
            }
        else:
            item["labeling_params"] = {
                "max_lookahead": int(pick(axes["trend_scanning"]["max_lookahead"], u[i, 6])),
                "min_t_value": float(pick(axes["trend_scanning"]["min_t_value"], u[i, 7])),
            }

        out.append(item)

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
