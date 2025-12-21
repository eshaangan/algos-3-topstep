"""
Inference helpers for tiny MLP models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datetime import time as dt_time
from features.engineer import add_features
from models.nn_model import TinyMLP

LABEL_VERSION_ALIGNED = "aligned_fixed_horizon_v1"

REQUIRED_NN_CONFIG_KEYS = [
    "horizon_bars",
    "threshold_ticks",
    "feature_lookback",
    "score_quantile",
    "score_threshold",
    "max_trades_per_day",
    "min_bars_between_trades",
    "selection_mode",
    "day_percentile_floor",
    "global_floor_score",
    "session_mode",
    "deadline_time",
    "execution_mode",
    "exit_price_mode",
    "max_hold_bars",
    "tick_size",
    "tick_value",
    "bar_minutes",
    "stop_loss_ticks",
    "target_multiplier",
    "label_version",
    "label_entry_price_col",
    "label_exit_price_col",
    "model_hidden_dims",
]


def validate_nn_config(nn_cfg: Dict[str, object]) -> None:
    missing = [k for k in REQUIRED_NN_CONFIG_KEYS if k not in nn_cfg]
    if missing:
        raise ValueError(f"config.json missing nn_config keys: {missing}")
    if nn_cfg.get("score_threshold") is None:
        raise ValueError("Missing score_threshold in config.json")

    label_version = str(nn_cfg.get("label_version"))
    if label_version != LABEL_VERSION_ALIGNED:
        raise ValueError(
            f"Unsupported label_version={label_version!r}; expected {LABEL_VERSION_ALIGNED}. Retrain the model."
        )
    if str(nn_cfg.get("label_entry_price_col")) != "open":
        raise ValueError("label_entry_price_col must be 'open' for aligned labels.")
    if str(nn_cfg.get("label_exit_price_col")) != "close":
        raise ValueError("label_exit_price_col must be 'close' for aligned labels.")

    if str(nn_cfg.get("execution_mode")) != "time_exit":
        raise ValueError("execution_mode must be 'time_exit' for aligned labels.")
    if str(nn_cfg.get("exit_price_mode")) != "bar_close":
        raise ValueError("exit_price_mode must be 'bar_close' for aligned labels.")
    if int(nn_cfg.get("max_hold_bars")) != int(nn_cfg.get("horizon_bars")):
        raise ValueError("max_hold_bars must equal horizon_bars for aligned labels.")

    hidden_dims = nn_cfg.get("model_hidden_dims")
    if not isinstance(hidden_dims, Sequence) or len(hidden_dims) != 2:
        raise ValueError("model_hidden_dims must be a sequence of length 2.")

    selection_mode = str(nn_cfg.get("selection_mode", "global_threshold")).lower()
    if selection_mode not in {"global_threshold", "day_adaptive_topn"}:
        raise ValueError(f"Unsupported selection_mode={selection_mode!r}")
    day_floor = nn_cfg.get("day_percentile_floor")
    if day_floor is None or not (0.0 <= float(day_floor) <= 1.0):
        raise ValueError("day_percentile_floor must be between 0 and 1.")
    if nn_cfg.get("global_floor_score") is None:
        raise ValueError("global_floor_score is required in nn_config.")
    cat_stop = nn_cfg.get("catastrophic_stop_ticks")
    if cat_stop is not None and int(cat_stop) <= 0:
        raise ValueError("catastrophic_stop_ticks must be > 0 when provided.")


def artifact_compatibility_issues(config: Dict[str, object], *, strict_versions: bool = True) -> List[str]:
    issues: List[str] = []
    nn_cfg = config.get("nn_config", {})
    try:
        validate_nn_config(nn_cfg)
    except Exception as exc:
        issues.append(str(exc))

    versions = config.get("versions", {})
    required_versions = ["python", "numpy", "pandas", "sklearn", "torch"]
    missing_versions = [k for k in required_versions if k not in versions]
    if missing_versions:
        issues.append(f"Missing versions in config.json: {missing_versions}")
    elif strict_versions:
        import platform
        import numpy as np
        import pandas as pd
        import sklearn
        import torch

        current = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        }
        mismatched = [k for k in required_versions if str(versions.get(k)) != str(current.get(k))]
        if mismatched:
            issues.append(f"Version mismatch for: {mismatched}")

    return issues


@dataclass
class NNBundle:
    model_dir: Path
    feature_cols: List[str]
    scaler: StandardScaler
    long_model: TinyMLP
    short_model: Optional[TinyMLP]
    long_calibrator: Optional[LogisticRegression]
    short_calibrator: Optional[LogisticRegression]
    config: Dict[str, object]
    device: torch.device


def _apply_calibrator(calibrator: Optional[LogisticRegression], logits: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return 1.0 / (1.0 + np.exp(-logits))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def load_nn_bundle(model_dir: str, *, fold: int = 0, device: Optional[str] = None) -> NNBundle:
    base = Path(model_dir)
    fold_dir = base / f"fold_{fold}"
    model_path = fold_dir if fold_dir.exists() else base

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("model_type") != "tiny_mlp":
        raise ValueError(f"Unsupported model_type: {config.get('model_type')}")

    bar_minutes = config.get("bar_minutes")
    if bar_minutes is not None and int(bar_minutes) != 5:
        raise ValueError(f"Expected 5-minute bars, got bar_minutes={bar_minutes}")

    feature_cols = config.get("feature_cols")
    if not feature_cols:
        raise ValueError("config.json missing feature_cols")

    nn_cfg = config.get("nn_config", {})
    issues = artifact_compatibility_issues(config, strict_versions=True)
    if issues:
        raise ValueError(f"Artifact incompatible, retrain required: {issues}")

    scaler_path = model_path / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.pkl in {model_path}")
    scaler = joblib.load(scaler_path)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    hidden_dims = tuple(int(x) for x in nn_cfg.get("model_hidden_dims", (32, 16)))

    long_model_path = model_path / "model_long.pt"
    if not long_model_path.exists():
        raise FileNotFoundError(f"Missing model_long.pt in {model_path}")
    long_model = TinyMLP(len(feature_cols), hidden_dims=hidden_dims)
    long_model.load_state_dict(torch.load(long_model_path, map_location=dev))
    long_model.to(dev).eval()

    short_model_path = model_path / "model_short.pt"
    short_model = None
    if short_model_path.exists():
        short_model = TinyMLP(len(feature_cols), hidden_dims=hidden_dims)
        short_model.load_state_dict(torch.load(short_model_path, map_location=dev))
        short_model.to(dev).eval()

    long_cal_path = model_path / "calibrator_long.pkl"
    short_cal_path = model_path / "calibrator_short.pkl"
    long_cal = joblib.load(long_cal_path) if long_cal_path.exists() else None
    short_cal = joblib.load(short_cal_path) if short_cal_path.exists() else None

    if isinstance(nn_cfg.get("deadline_time"), str):
        try:
            nn_cfg["deadline_time"] = dt_time.fromisoformat(nn_cfg["deadline_time"])
        except ValueError as exc:
            raise ValueError("Invalid deadline_time in config.json") from exc
    if nn_cfg.get("catastrophic_stop_ticks") is None:
        nn_cfg["catastrophic_stop_ticks"] = int(nn_cfg.get("threshold_ticks", 0)) * 4

    return NNBundle(
        model_dir=model_path,
        feature_cols=feature_cols,
        scaler=scaler,
        long_model=long_model,
        short_model=short_model,
        long_calibrator=long_cal,
        short_calibrator=short_cal,
        config=config,
        device=dev,
    )


def predict_scores_for_bars(bars_df: pd.DataFrame, bundle: NNBundle) -> pd.DataFrame:
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index(drop=True)
    if "idx" not in features_df.columns:
        features_df.insert(0, "idx", np.arange(len(features_df)))

    valid_mask = features_df[bundle.feature_cols].notna().all(axis=1)
    valid_idx = features_df.loc[valid_mask, "idx"].values
    X = features_df.loc[valid_mask, bundle.feature_cols].values
    X = bundle.scaler.transform(X).astype(np.float32)

    X_tensor = torch.from_numpy(X).to(bundle.device)
    with torch.no_grad():
        long_logits = bundle.long_model(X_tensor).cpu().numpy()
        short_logits = None
        if bundle.short_model is not None:
            short_logits = bundle.short_model(X_tensor).cpu().numpy()

    long_prob = _apply_calibrator(bundle.long_calibrator, long_logits)
    if short_logits is None:
        short_prob = np.zeros_like(long_prob)
    else:
        short_prob = _apply_calibrator(bundle.short_calibrator, short_logits)
    score = np.maximum(long_prob, short_prob)
    direction = np.where(long_prob >= short_prob, "long", "short")

    prob_df = pd.DataFrame(
        {
            "idx": valid_idx,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "score": score,
            "direction": direction,
        }
    )

    full = pd.DataFrame({"idx": features_df["idx"].values})
    full = full.merge(prob_df, on="idx", how="left")
    return full


def predict_latest(
    bars_df: pd.DataFrame, bundle: NNBundle
) -> Optional[Dict[str, float]]:
    if bars_df.empty:
        return None

    features_df = add_features(bars_df, verbose=False)
    last_row = features_df.iloc[-1]
    if last_row[bundle.feature_cols].isna().any():
        return None

    X = last_row[bundle.feature_cols].values.reshape(1, -1)
    X = bundle.scaler.transform(X).astype(np.float32)
    X_tensor = torch.from_numpy(X).to(bundle.device)
    with torch.no_grad():
        long_logit = bundle.long_model(X_tensor).cpu().numpy()[0]
        short_logit = None
        if bundle.short_model is not None:
            short_logit = bundle.short_model(X_tensor).cpu().numpy()[0]

    long_prob = float(_apply_calibrator(bundle.long_calibrator, np.array([long_logit]))[0])
    if short_logit is None:
        short_prob = 0.0
    else:
        short_prob = float(_apply_calibrator(bundle.short_calibrator, np.array([short_logit]))[0])
    score = max(long_prob, short_prob)
    direction = "long" if long_prob >= short_prob else "short"

    return {
        "long_prob": long_prob,
        "short_prob": short_prob,
        "score": score,
        "direction": direction,
    }
