#!/usr/bin/env python3
"""
Create a dual-side model bundle compatible with LiveModelPredictor.
"""

import argparse
import pickle
from pathlib import Path

from ml_intraday_v3.ensemble import DualSideModel


def _load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a dual-side model bundle for live trading"
    )
    parser.add_argument(
        "--long-bundle",
        required=True,
        help="Path to long-side bundle.pkl",
    )
    parser.add_argument(
        "--short-bundle",
        required=True,
        help="Path to short-side bundle.pkl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for model_bundle_dual.pkl",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.03,
        help="Primary score threshold for live trading",
    )

    args = parser.parse_args()

    long_path = Path(args.long_bundle)
    short_path = Path(args.short_bundle)
    out_path = Path(args.output)

    if not long_path.exists():
        raise FileNotFoundError(f"Missing long bundle: {long_path}")
    if not short_path.exists():
        raise FileNotFoundError(f"Missing short bundle: {short_path}")

    long_bundle = _load_bundle(long_path)
    short_bundle = _load_bundle(short_path)

    long_model = long_bundle.get("model")
    short_model = short_bundle.get("model")
    long_features = long_bundle.get("feature_columns")
    short_features = short_bundle.get("feature_columns")

    if long_model is None or short_model is None:
        raise ValueError("Both bundles must include a 'model' key")
    if long_features != short_features:
        raise ValueError("Long/short feature columns do not match")

    dual_model = DualSideModel(long_model, short_model)

    bundle = {
        "primary_model": dual_model,
        "primary_preprocessor": long_bundle.get("preprocessor"),
        "primary_feature_columns": long_features,
        "thresholds": {"primary_threshold": float(args.primary_threshold)},
        "meta_model": None,
        "meta_preprocessor": None,
        "meta_feature_columns": None,
        "model_type": "LGBMDualSide",
        "n_base_models": 2,
        "source_bundles": {
            "long": str(long_path),
            "short": str(short_path),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved dual-side bundle: {out_path}")


if __name__ == "__main__":
    main()
