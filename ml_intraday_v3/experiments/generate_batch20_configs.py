"""
Generate Batch 20: Meta-Labeling on Rule-Based Primary Configs
==============================================================
Generates ~100 experiment configs for GCP batch 20:
  - Primary signal: rule_based_v1 (direction + entry timing)
  - Secondary model: ML classifier trained to predict "did this signal work?"
  - Axes: top-5 primary params × 3 model kinds × 3 thresholds × 3 feature windows

Usage:
    python ml_intraday_v3/experiments/generate_batch20_configs.py \
        --oos-results ml_intraday_v3/diagnostics/rule_based_oos_results.json \
        --output-dir ml_intraday_v3/experiments/batch20_metalabel_configs/ \
        --n-configs 100

Output:
    batch20_metalabel_configs/
        batch20_metalabel_001.json
        batch20_metalabel_002.json
        ...
        manifest.json   <- lists all config files
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep axes (plan spec)
# ---------------------------------------------------------------------------
SECONDARY_MODEL_KINDS = ["random_forest", "extra_trees", "lightgbm"]
SECONDARY_THRESHOLDS = [0.40, 0.45, 0.50]
FEATURE_WINDOW_BARS = [10, 20, 40]
CV_METHODS = ["cpcv"]

# Default secondary model params by kind
SECONDARY_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 30,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 30,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 30,
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1,
    },
}

# GCP experiment settings
GCP_DEFAULTS = {
    "training_start": "2024-10-01",
    "training_end": "2025-11-30",
    "oos_start": "2026-01-01",
    "oos_end": "2026-02-14",
    "data_path": "gs://topstep-data/mes_bars_databento_rth.h5",
    "data_key": "bars",
    "n_cpcv_splits": 5,
    "n_cpcv_test_splits": 2,
    "embargo_pct": 0.01,
    "purge_pct": 0.02,
    "point_value": 5.0,
    "commission_per_side": 0.62,
    "slippage_ticks": 1,
}


def load_top_primary_params(oos_results_path: str, top_n: int = 5) -> List[dict]:
    """Load top-N primary params from validate_rule_based_oos.py output."""
    with open(oos_results_path) as f:
        data = json.load(f)

    all_results = data.get("all_results", data.get("top5", []))
    top_configs = all_results[:top_n]

    primary_params_list = []
    for i, r in enumerate(top_configs):
        params = r.get("params", {})
        primary_params_list.append({
            "rank": i + 1,
            "ema_fast": params.get("ema_fast", 13),
            "ema_slow": params.get("ema_slow", 55),
            "pt_atr_mult": params.get("pt_atr_mult", 2.0),
            "sl_atr_mult": params.get("sl_atr_mult", 1.5),
            "n_contracts": params.get("n_contracts", 1),
            "oos_p_pass": r.get("monte_carlo", {}).get("p_pass", 0.0),
            "oos_win_rate": r.get("backtest", {}).get("win_rate", 0.0),
        })

    return primary_params_list


def default_primary_params() -> List[dict]:
    """Default primary params when OOS results aren't available."""
    return [
        {"rank": 1, "ema_fast": 13, "ema_slow": 55, "pt_atr_mult": 2.0, "sl_atr_mult": 1.5, "n_contracts": 1},
        {"rank": 2, "ema_fast": 8,  "ema_slow": 34, "pt_atr_mult": 2.5, "sl_atr_mult": 1.5, "n_contracts": 1},
        {"rank": 3, "ema_fast": 13, "ema_slow": 34, "pt_atr_mult": 1.5, "sl_atr_mult": 1.0, "n_contracts": 1},
        {"rank": 4, "ema_fast": 21, "ema_slow": 55, "pt_atr_mult": 3.0, "sl_atr_mult": 2.0, "n_contracts": 2},
        {"rank": 5, "ema_fast": 8,  "ema_slow": 55, "pt_atr_mult": 2.0, "sl_atr_mult": 1.0, "n_contracts": 1},
    ]


def build_config(
    exp_id: str,
    primary_params: dict,
    secondary_model_kind: str,
    secondary_threshold: float,
    feature_window_bars: int,
    cv_method: str,
    gcp_defaults: dict,
) -> dict:
    """Build a single experiment config dict."""
    return {
        "exp_id": exp_id,
        "batch": "batch20",
        "description": "Meta-labeling: rule_based_v1 primary + ML secondary filter",

        # Primary signal: rule-based system
        "primary_signal": "rule_based_v1",
        "primary_params": {
            "ema_fast": primary_params["ema_fast"],
            "ema_slow": primary_params["ema_slow"],
            "pt_atr_mult": primary_params["pt_atr_mult"],
            "sl_atr_mult": primary_params["sl_atr_mult"],
            "n_contracts": primary_params["n_contracts"],
            "min_spread_atr_ratio": 0.3,
            "slope_lookback": 3,
            "atr_period": 14,
            "time_stop_bars": 24,
            "trailing_activation_atr": 1.0,
            "trailing_distance_atr": 0.75,
            "session_start": "09:35",
            "session_end": "15:45",
            "volume_min_ratio": 1.2,
            "volume_max_ratio": 2.0,
            "bb_period": 20,
            "rsi_period": 14,
        },
        "primary_rank": primary_params.get("rank", 1),

        # Secondary model: meta-labeling classifier
        "secondary_model_kind": secondary_model_kind,
        "secondary_threshold": secondary_threshold,
        "secondary_params": SECONDARY_PARAMS[secondary_model_kind].copy(),
        "feature_window_bars": feature_window_bars,

        # Meta-labeling label construction
        "meta_label_method": "trade_outcome",  # y=1 if primary signal was profitable
        "min_meta_samples": 50,                # Min trades needed to train secondary

        # Cross-validation
        "cv_method": cv_method,
        "n_cpcv_splits": gcp_defaults["n_cpcv_splits"],
        "n_cpcv_test_splits": gcp_defaults["n_cpcv_test_splits"],
        "embargo_pct": gcp_defaults["embargo_pct"],
        "purge_pct": gcp_defaults["purge_pct"],

        # Data
        "training_start": gcp_defaults["training_start"],
        "training_end": gcp_defaults["training_end"],
        "oos_start": gcp_defaults["oos_start"],
        "oos_end": gcp_defaults["oos_end"],
        "data_path": gcp_defaults["data_path"],
        "data_key": gcp_defaults["data_key"],

        # Costs
        "point_value": gcp_defaults["point_value"],
        "commission_per_side": gcp_defaults["commission_per_side"],
        "slippage_ticks": gcp_defaults["slippage_ticks"],

        # Success criteria for this config
        "success_thresholds": {
            "min_meta_auc": 0.52,
            "min_precision_lift_pct": 5.0,
            "min_win_rate_after_filter": 0.48,
        },
    }


def generate_configs(
    primary_params_list: List[dict],
    n_configs: int,
    output_dir: Path,
    gcp_defaults: dict,
) -> List[str]:
    """Generate all config files and return list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all combinations
    combos = list(product(
        range(len(primary_params_list)),  # primary index
        SECONDARY_MODEL_KINDS,
        SECONDARY_THRESHOLDS,
        FEATURE_WINDOW_BARS,
        CV_METHODS,
    ))

    # Cap at n_configs
    combos = combos[:n_configs]

    config_files = []
    for i, (primary_idx, model_kind, threshold, window, cv) in enumerate(combos, 1):
        exp_id = f"batch20_metalabel_{i:03d}"
        primary_params = primary_params_list[primary_idx]

        config = build_config(
            exp_id=exp_id,
            primary_params=primary_params,
            secondary_model_kind=model_kind,
            secondary_threshold=threshold,
            feature_window_bars=window,
            cv_method=cv,
            gcp_defaults=gcp_defaults,
        )

        config_path = output_dir / f"{exp_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        config_files.append(str(config_path))

        if i % 20 == 0:
            logger.info(f"Generated {i}/{len(combos)} configs")

    return config_files


def main():
    parser = argparse.ArgumentParser(description="Generate batch 20 meta-labeling configs")
    parser.add_argument(
        "--oos-results",
        type=str,
        help="Path to rule_based_oos_results.json (to extract top primary params)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_intraday_v3/experiments/batch20_metalabel_configs",
        help="Output directory for config files",
    )
    parser.add_argument("--n-configs", type=int, default=100, help="Max configs to generate")
    parser.add_argument("--top-n-primary", type=int, default=5, help="Top N primary param sets to use")
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://topstep-data/mes_bars_databento_rth.h5",
        help="GCS path to data file (for GCP jobs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load primary params
    if args.oos_results and Path(args.oos_results).exists():
        logger.info(f"Loading top primary params from {args.oos_results}")
        primary_params_list = load_top_primary_params(args.oos_results, top_n=args.top_n_primary)
    else:
        logger.info("Using default primary params (run validate_rule_based_oos.py first for OOS-informed selection)")
        primary_params_list = default_primary_params()

    logger.info(f"Using {len(primary_params_list)} primary param sets")
    for p in primary_params_list:
        logger.info(
            f"  rank={p.get('rank')}: ema={p['ema_fast']}/{p['ema_slow']} "
            f"pt={p['pt_atr_mult']} sl={p['sl_atr_mult']} "
            f"p_pass={p.get('oos_p_pass', 0):.1%}"
        )

    gcp_defaults = {**GCP_DEFAULTS, "data_path": args.data_path}
    output_dir = Path(args.output_dir)

    config_files = generate_configs(
        primary_params_list=primary_params_list,
        n_configs=args.n_configs,
        output_dir=output_dir,
        gcp_defaults=gcp_defaults,
    )

    # Write manifest
    manifest = {
        "batch": "batch20",
        "n_configs": len(config_files),
        "primary_params_used": primary_params_list,
        "sweep_axes": {
            "secondary_model_kinds": SECONDARY_MODEL_KINDS,
            "secondary_thresholds": SECONDARY_THRESHOLDS,
            "feature_window_bars": FEATURE_WINDOW_BARS,
            "cv_methods": CV_METHODS,
        },
        "config_files": config_files,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(config_files)} configs in {output_dir}/")
    print(f"Manifest written to {manifest_path}")
    print("\nTo upload to GCS:")
    print(f"  gsutil -m cp {output_dir}/*.json gs://topstep-experiments/batch20/")
    print("\nTo launch GCP batch:")
    print(f"  python ml_intraday_v3/experiments/run_metalabel_rulebased.py --config-dir {output_dir}/")


if __name__ == "__main__":
    main()
