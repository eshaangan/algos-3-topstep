"""
Alignment checks for events, features, and weights.
"""

from __future__ import annotations

import pandas as pd


def _coerce_to_index_tz(ts: pd.Series, index: pd.Index) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(index, "tz", None) is None:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
    else:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(index.tz)
        else:
            ts = ts.dt.tz_convert(index.tz)
    return ts


def check_events_on_grid(
    bars_index: pd.Index, events_df: pd.DataFrame
) -> dict:
    if events_df.empty:
        return {
            "status": "SKIP",
            "reason": "empty_events",
            "offgrid_t0": 0,
            "offgrid_t1": 0,
        }

    t0 = _coerce_to_index_tz(events_df["t0"], bars_index)
    t1 = _coerce_to_index_tz(events_df["t1"], bars_index)
    t0_pos = bars_index.get_indexer(t0)
    t1_pos = bars_index.get_indexer(t1)
    off_t0 = int((t0_pos < 0).sum())
    off_t1 = int((t1_pos < 0).sum())

    status = "PASS" if off_t0 == 0 and off_t1 == 0 else "FAIL"
    return {
        "status": status,
        "offgrid_t0": off_t0,
        "offgrid_t1": off_t1,
        "n_events": int(len(events_df)),
    }


def check_features_index(
    bars_index: pd.Index, features_df: pd.DataFrame
) -> dict:
    if features_df.empty:
        return {
            "status": "SKIP",
            "reason": "empty_features",
        }

    feat_index = features_df.index
    if not isinstance(feat_index, pd.DatetimeIndex):
        if "timestamp" in features_df.columns:
            feat_index = pd.to_datetime(features_df["timestamp"])
        else:
            return {
                "status": "FAIL",
                "reason": "features_index_not_datetime",
            }

    feat_index = _coerce_to_index_tz(
        pd.Series(feat_index, index=feat_index), bars_index
    )
    feat_index = pd.DatetimeIndex(feat_index)

    if len(feat_index) != len(bars_index):
        return {
            "status": "FAIL",
            "reason": "length_mismatch",
            "features_len": int(len(feat_index)),
            "bars_len": int(len(bars_index)),
        }

    if not feat_index.equals(bars_index):
        mismatch = int((feat_index != bars_index).sum())
        return {
            "status": "FAIL",
            "reason": "index_mismatch",
            "mismatch_count": mismatch,
        }

    return {"status": "PASS", "features_len": int(len(feat_index))}


def check_weights_event_ids(
    events_df: pd.DataFrame, weights_df: pd.DataFrame
) -> dict:
    if weights_df.empty:
        return {"status": "SKIP", "reason": "empty_weights"}
    if "event_id" not in weights_df.columns:
        return {"status": "FAIL", "reason": "missing_event_id"}

    usable_events = events_df
    if "y" in events_df.columns:
        usable_events = events_df[events_df["y"].notna()]

    events_ids = set(usable_events["event_id"].tolist())
    weights_ids = set(weights_df["event_id"].tolist())

    missing = events_ids - weights_ids
    extra = weights_ids - events_ids
    status = "PASS" if not missing and not extra else "FAIL"

    return {
        "status": status,
        "n_events": int(len(events_ids)),
        "n_weights": int(len(weights_ids)),
        "missing_in_weights": int(len(missing)),
        "extra_in_weights": int(len(extra)),
    }
