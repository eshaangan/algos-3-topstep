"""
Build event-level training datasets from V3 artifacts.
"""

import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_feature_columns(bar_dir: Path) -> List[str]:
    schema_path = bar_dir / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path, "r") as f:
            schema = json.load(f)
        return list(schema.get("feature_columns", []))
    return []


def build_event_dataset(
    run_dir: Path | str,
    bar_size: str,
    training_config: dict | None = None,
) -> pd.DataFrame:
    """
    Build event-level dataset by joining features at event t0.

    Returns DataFrame with columns:
    event_id, t0, y, w_final, <feature columns in deterministic order>
    """
    run_dir = Path(run_dir)
    bar_dir = run_dir / f"bar_size={bar_size}"

    events_path = bar_dir / "events.parquet"
    features_path = bar_dir / "features.parquet"
    weights_path = bar_dir / "weights.parquet"

    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {events_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"features.parquet not found: {features_path}")

    events_df = pd.read_parquet(events_path)
    required_cols = ["event_id", "t0", "y"]
    missing = [c for c in required_cols if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in events: {missing}")

    events_df = events_df.copy()
    events_df["t0"] = pd.to_datetime(events_df["t0"])
    events_df = events_df.sort_values("event_id").reset_index(drop=True)

    if events_df["y"].isna().any():
        before = len(events_df)
        events_df = events_df[events_df["y"].notna()].reset_index(drop=True)
        logger.warning(
            f"Dropped {before - len(events_df)} events with NaN target"
        )

    features_df = pd.read_parquet(features_path)
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)

    t0 = events_df["t0"].to_list()
    idx = features_df.index.get_indexer(t0)
    missing_mask = idx < 0
    if missing_mask.any():
        raise ValueError(
            f"{missing_mask.sum()} event t0 timestamps not found in features index"
        )

    features_at_t0 = features_df.reindex(t0)

    if "usable_for_training" in features_at_t0.columns:
        usable = features_at_t0["usable_for_training"].fillna(False).astype(bool)
        if (~usable).any():
            logger.info(
                f"Filtering {(~usable).sum()} events where usable_for_training is False"
            )
        usable_mask = usable.to_numpy()
        events_df = events_df[usable_mask].reset_index(drop=True)
        features_at_t0 = features_at_t0.loc[usable_mask].reset_index(
            drop=True
        )

    feature_cols = _load_feature_columns(bar_dir)
    if not feature_cols:
        feature_cols = list(features_df.columns)

    cfg = training_config or {}
    feat_cfg = cfg.get("features", {})
    use_columns = feat_cfg.get("use_columns", "all")
    drop_meta = feat_cfg.get(
        "drop_meta_columns", ["is_synthetic", "usable_for_training"]
    )

    if use_columns != "all":
        if not isinstance(use_columns, list):
            raise ValueError("features.use_columns must be 'all' or list")
        feature_cols = [c for c in use_columns]

    feature_cols = [c for c in feature_cols if c not in drop_meta]
    missing_feats = [c for c in feature_cols if c not in features_at_t0.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns at t0: {missing_feats}")

    features_sel = features_at_t0[feature_cols].reset_index(drop=True)

    weight_cfg = cfg.get("sample_weight", {})
    weight_enabled = bool(weight_cfg.get("enabled", False))
    weight_col = weight_cfg.get("column", "w_final")

    if weight_enabled:
        if not weights_path.exists():
            raise FileNotFoundError(f"weights.parquet not found: {weights_path}")
        weights_df = pd.read_parquet(weights_path)
        if weight_col not in weights_df.columns:
            raise ValueError(f"Missing weight column: {weight_col}")
        weights_df = weights_df[["event_id", weight_col]]
        merged = events_df.merge(weights_df, on="event_id", how="left")
        if merged[weight_col].isna().any():
            raise ValueError("Missing sample weights after merge")
        w_final = merged[weight_col].to_numpy()
    else:
        w_final = np.ones(len(events_df), dtype=float)

    dataset = pd.DataFrame(
        {
            "event_id": events_df["event_id"].to_numpy(),
            "t0": events_df["t0"].to_numpy(),
            "y": events_df["y"].to_numpy(),
            "w_final": w_final,
        }
    )

    dataset = pd.concat([dataset, features_sel], axis=1)
    return dataset


def build_meta_dataset(
    primary_preds_df: pd.DataFrame,
    base_event_dataset_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, list]:
    """
    Build meta-labeling dataset from primary predictions and event outcomes.

    Returns:
        meta_df: DataFrame with columns event_id, y (meta label), w_final, p_primary,
                 p_primary_logit (optional), and feature columns
        feature_cols: ordered list of feature columns used for meta model
    """
    meta_cfg = config.get("meta", {})
    if not meta_cfg.get("enabled", False):
        return pd.DataFrame(), []

    threshold_primary = float(meta_cfg.get("threshold_primary", 0.5))
    features_cfg = meta_cfg.get("features", {})
    include_primary_prob = bool(features_cfg.get("include_primary_prob", True))
    include_primary_logit = bool(features_cfg.get("include_primary_logit", True))
    include_original = bool(features_cfg.get("include_original_features", True))

    preds = primary_preds_df[["event_id", "y_prob"]].copy()
    preds["proposed_trade"] = preds["y_prob"] >= threshold_primary
    preds = preds[preds["proposed_trade"]].reset_index(drop=True)
    if preds.empty:
        return pd.DataFrame(), []

    base_df = base_event_dataset_df.merge(preds, on="event_id", how="inner")

    events_cols = ["event_id", "y"]
    if "ret_net" in events_df.columns:
        events_cols.append("ret_net")
    events_sub = events_df[events_cols].copy()
    merged = base_df.merge(events_sub, on="event_id", how="left", suffixes=("", "_evt"))

    target_kind = meta_cfg.get("target", {}).get("kind", "y_positive")
    if target_kind == "y_positive":
        meta_y = (merged["y_evt"] == 1).astype(int)
    elif target_kind == "ret_net_positive":
        if "ret_net" not in merged.columns:
            raise ValueError("ret_net not available for meta target")
        meta_y = (merged["ret_net"] > 0).astype(int)
    elif target_kind == "y_and_ret_net":
        if "ret_net" not in merged.columns:
            raise ValueError("ret_net not available for meta target")
        meta_y = ((merged["y_evt"] == 1) & (merged["ret_net"] > 0)).astype(int)
    else:
        raise ValueError(f"Unsupported meta target kind: {target_kind}")

    base_feature_cols = [
        c
        for c in base_event_dataset_df.columns
        if c not in ["event_id", "t0", "y", "w_final"]
    ]

    feature_cols = []
    if include_original:
        feature_cols.extend(base_feature_cols)
    if include_primary_prob:
        feature_cols.append("p_primary")
    if include_primary_logit:
        feature_cols.append("p_primary_logit")

    merged["p_primary"] = merged["y_prob"].astype(float)
    eps = 1e-6
    p_clip = merged["p_primary"].clip(eps, 1.0 - eps)
    merged["p_primary_logit"] = np.log(p_clip / (1.0 - p_clip))

    meta_df = pd.DataFrame(
        {
            "event_id": merged["event_id"].to_numpy(),
            "y": meta_y.to_numpy(),
            "w_final": merged["w_final"].to_numpy(),
            "p_primary": merged["p_primary"].to_numpy(),
            "p_primary_logit": merged["p_primary_logit"].to_numpy(),
        }
    )

    if include_original:
        meta_df = pd.concat([meta_df, merged[base_feature_cols]], axis=1)

    base_cols = ["event_id", "y", "w_final", "p_primary", "p_primary_logit"]
    ordered_cols = base_cols + [c for c in feature_cols if c not in base_cols]
    meta_df = meta_df[ordered_cols]
    return meta_df, feature_cols
