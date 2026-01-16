#!/usr/bin/env python3
"""
Create an ensemble (optionally dual-side) model bundle compatible with LiveModelPredictor.

This is intended as a practical, variance-reduction wrapper for live trading.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

try:
    # When run as a module: python -m ml_intraday_v3.create_ensemble_model_bundle
    from .ensemble import DualSideModel, LGBMPreprocessedEnsemble
except ImportError:  # pragma: no cover
    # When run as a script from within the package directory.
    from ensemble import DualSideModel, LGBMPreprocessedEnsemble


def _load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_fold_bundles(training_dir: Path) -> list[dict]:
    training_dir = Path(training_dir)
    bundle_paths = sorted(training_dir.glob("fold_*/bundle.pkl"))
    if not bundle_paths:
        raise FileNotFoundError(f"No fold_*/bundle.pkl found under: {training_dir}")
    return [_load_bundle(p) for p in bundle_paths]


def _validate_compat(bundles: list[dict]) -> tuple[list, list, list[dict]]:
    models = []
    preprocessors = []
    feature_columns = None
    for b in bundles:
        model = b.get("model")
        cols = b.get("feature_columns")
        prep = b.get("preprocessor")
        if model is None:
            raise ValueError("Bundle missing 'model'")
        if cols is None:
            raise ValueError("Bundle missing 'feature_columns'")
        if prep is None:
            raise ValueError("Bundle missing 'preprocessor'")
        if feature_columns is None:
            feature_columns = list(cols)
        elif list(cols) != feature_columns:
            raise ValueError("Fold feature columns do not match")
        models.append(model)
        preprocessors.append(prep)
    return models, feature_columns, preprocessors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an ensemble model bundle for live trading"
    )
    parser.add_argument(
        "--long-training-dir",
        required=True,
        help="Path to training directory containing fold_*/bundle.pkl (long side)",
    )
    parser.add_argument(
        "--short-training-dir",
        default=None,
        help="Optional path to training directory containing fold_*/bundle.pkl (short side)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for model_bundle_ensemble.pkl",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.08,
        help="Primary score threshold for live trading",
    )

    args = parser.parse_args()

    long_dir = Path(args.long_training_dir)
    short_dir = Path(args.short_training_dir) if args.short_training_dir else None
    out_path = Path(args.output)

    long_bundles = _load_fold_bundles(long_dir)
    long_models, feature_columns, long_preprocessors = _validate_compat(long_bundles)
    long_ensemble = LGBMPreprocessedEnsemble(long_models, long_preprocessors)

    model_type = "LGBMEnsemble"
    primary_model = long_ensemble
    n_base_models = len(long_models)
    sources = {"long_training_dir": str(long_dir)}

    if short_dir is not None:
        short_bundles = _load_fold_bundles(short_dir)
        short_models, short_cols, short_preprocessors = _validate_compat(short_bundles)
        if short_cols != feature_columns:
            raise ValueError("Long/short feature columns do not match")
        short_ensemble = LGBMPreprocessedEnsemble(short_models, short_preprocessors)
        primary_model = DualSideModel(long_ensemble, short_ensemble)
        model_type = "LGBMDualSideEnsemble"
        n_base_models = len(long_models) + len(short_models)
        sources["short_training_dir"] = str(short_dir)

    # Passthrough preprocessor: the ensemble applies fold-specific preprocessing itself.
    passthrough_preprocessor = {
        "impute": "none",
        "scaler": "none",
        "medians": [0.0] * len(feature_columns),
        "means": [0.0] * len(feature_columns),
        "stds": [1.0] * len(feature_columns),
    }

    bundle = {
        "primary_model": primary_model,
        "primary_preprocessor": passthrough_preprocessor,
        "primary_feature_columns": feature_columns,
        "thresholds": {"primary_threshold": float(args.primary_threshold)},
        "meta_model": None,
        "meta_preprocessor": None,
        "meta_feature_columns": None,
        "model_type": model_type,
        "n_base_models": int(n_base_models),
        "source": sources,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved ensemble bundle: {out_path}")


if __name__ == "__main__":
    main()
