#!/usr/bin/env python3
"""
Generate experiment JSON files from exhaustive axes config.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def load_axes(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _is_valid_combo(combo: Dict) -> bool:
    # Keep compute bounded and avoid obvious nonsense.
    if combo["calibration"] == "isotonic" and combo["model_kind"] == "mlp":
        return False
    if combo["training_window_months"] <= 2 and combo["cv_n_splits"] >= 8:
        return False
    return True


def generate_configs(cfg: Dict, max_experiments: int, seed: int) -> List[Dict]:
    axes = cfg["axes"]
    combos = []

    product_space = itertools.product(
        axes["session_mode"],
        axes["training_window_months"],
        axes["sample_weight"],
        axes["calibration"],
        axes["balance_method"],
        axes["cv_n_splits"],
        axes["cv_embargo_bars"],
        axes["labeling"],
        axes["model"],
        axes["feature_set"],
    )

    for (
        session_mode,
        training_window_months,
        sample_weight,
        calibration,
        balance_method,
        cv_n_splits,
        cv_embargo_bars,
        labeling,
        model,
        feature_set,
    ) in product_space:
        combo = {
            "session_mode": session_mode,
            "training_window_months": int(training_window_months),
            "sample_weight": sample_weight,
            "calibration": calibration,
            "balance_method": balance_method,
            "cv_n_splits": int(cv_n_splits),
            "cv_embargo_bars": int(cv_embargo_bars),
            "labeling": labeling,
            "model_name": model["name"],
            "model_kind": model["model_kind"],
            "model_params": model["model_params"],
            "feature_set_name": feature_set["name"],
            "feature_set": feature_set["feature_set"],
            "features_config": feature_set.get("features_config", {}),
        }
        if _is_valid_combo(combo):
            combos.append(combo)

    rng = np.random.default_rng(seed)
    if len(combos) > max_experiments:
        idx = rng.choice(len(combos), size=max_experiments, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    output = []
    for i, combo in enumerate(combos, 1):
        output.append(
            {
                "exp_id": f"exhaustive_exp_{i:05d}",
                "phase": "exhaustive",
                **combo,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate exhaustive experiment configs")
    parser.add_argument(
        "--axes-config",
        type=str,
        default=str(Path(__file__).with_name("exhaustive_axes.yaml")),
    )
    parser.add_argument("--output-dir", type=str, default="experiment_configs_exhaustive")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_axes(Path(args.axes_config))
    max_experiments = args.max_experiments or int(cfg["constraints"]["max_experiments_default"])
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    configs = generate_configs(cfg, max_experiments=max_experiments, seed=seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in configs:
        out_file = out_dir / f"{item['exp_id']}.json"
        with open(out_file, "w") as f:
            json.dump(item, f, indent=2)

    manifest = {
        "n_configs": len(configs),
        "axes_config": args.axes_config,
        "output_dir": str(out_dir),
        "max_experiments": max_experiments,
        "seed": seed,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(configs)} configs in {out_dir}")


if __name__ == "__main__":
    main()
