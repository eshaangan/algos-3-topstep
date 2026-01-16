"""
Helpers for combining long/short model predictions into a single decision stream.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _score_ev_from_cols(df: pd.DataFrame, suffix: str) -> pd.Series:
    score_col = f"score_ev_{suffix}"
    if score_col in df.columns:
        return df[score_col].astype(float)
    p_target = f"p_target_{suffix}"
    p_stop = f"p_stop_{suffix}"
    if p_target in df.columns and p_stop in df.columns:
        return df[p_target].astype(float) - df[p_stop].astype(float)
    raise ValueError(
        f"Missing score_ev or p_target/p_stop for {suffix} predictions"
    )


def _pick_side_values(
    merged: pd.DataFrame,
    predicted_side: pd.Series,
    col_base: str,
) -> np.ndarray:
    col_long = f"{col_base}_long"
    col_short = f"{col_base}_short"
    if col_long not in merged.columns or col_short not in merged.columns:
        return np.full(len(merged), np.nan, dtype=float)

    out = np.full(len(merged), np.nan, dtype=float)
    long_mask = predicted_side == 1
    short_mask = predicted_side == -1
    out[long_mask] = merged.loc[long_mask, col_long].astype(float)
    out[short_mask] = merged.loc[short_mask, col_short].astype(float)
    return out


def _attach_event_keys(
    preds: pd.DataFrame,
    events_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    key_cols = ["event_id", "t0", "t1", "horizon_bars", "pt_mult", "sl_mult"]
    if not set(key_cols).issubset(events_df.columns):
        missing = sorted(set(key_cols) - set(events_df.columns))
        raise ValueError(f"events_df is missing required columns: {missing}")
    event_ids = preds["event_id"].unique()
    events_subset = events_df.loc[
        events_df["event_id"].isin(event_ids), key_cols
    ]
    merged = preds.merge(events_subset, on="event_id", how="left")
    missing_keys = merged["t0"].isna()
    if missing_keys.any():
        logger.warning(
            "Missing event metadata for %d %s predictions; dual merge may drop rows",
            int(missing_keys.sum()),
            label,
        )
    return merged


def combine_dual_primary_predictions(
    long_preds: pd.DataFrame,
    short_preds: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    long_with_keys = _attach_event_keys(long_preds, events_df, "long")
    short_with_keys = _attach_event_keys(short_preds, events_df, "short")

    long_with_keys = long_with_keys.rename(columns={"event_id": "event_id_long"})
    short_with_keys = short_with_keys.rename(columns={"event_id": "event_id_short"})

    join_keys = ["t0", "t1", "horizon_bars", "pt_mult", "sl_mult"]
    merged = long_with_keys.merge(
        short_with_keys,
        on=join_keys,
        how="outer",
        suffixes=("_long", "_short"),
    )
    if merged.empty:
        raise ValueError("No overlapping long/short events after key join")

    score_long = _score_ev_from_cols(merged, "long").fillna(-np.inf)
    score_short = _score_ev_from_cols(merged, "short").fillna(-np.inf)

    choose_long = score_long >= score_short
    choose_short = score_short > score_long
    pos_long = score_long > 0
    pos_short = score_short > 0

    predicted_side = np.where(
        choose_long & pos_long,
        1,
        np.where(choose_short & pos_short, -1, 0),
    ).astype(int)

    event_id_long = merged.get("event_id_long")
    event_id_short = merged.get("event_id_short")

    keep_long = event_id_long.notna()
    keep_short = event_id_short.notna()

    long_rows = merged.loc[keep_long].copy()
    short_rows = merged.loc[keep_short].copy()

    long_rows["event_id"] = event_id_long.loc[keep_long].astype(int)
    short_rows["event_id"] = event_id_short.loc[keep_short].astype(int)

    long_rows["predicted_side"] = np.where(
        predicted_side[keep_long.to_numpy()] == 1, 1, 0
    ).astype(int)
    short_rows["predicted_side"] = np.where(
        predicted_side[keep_short.to_numpy()] == -1, -1, 0
    ).astype(int)

    long_rows["score_ev"] = np.where(
        long_rows["predicted_side"] == 1, score_long[keep_long.to_numpy()], 0.0
    )
    short_rows["score_ev"] = np.where(
        short_rows["predicted_side"] == -1, score_short[keep_short.to_numpy()], 0.0
    )

    long_rows["score_ev_long"] = score_long[keep_long.to_numpy()]
    long_rows["score_ev_short"] = score_short[keep_long.to_numpy()]
    short_rows["score_ev_long"] = score_long[keep_short.to_numpy()]
    short_rows["score_ev_short"] = score_short[keep_short.to_numpy()]

    for col in ["p_target", "p_stop", "p_vertical"]:
        long_rows[col] = _pick_side_values(long_rows, long_rows["predicted_side"], col)
        short_rows[col] = _pick_side_values(
            short_rows, short_rows["predicted_side"], col
        )

    out = pd.concat(
        [
            long_rows[["event_id", "score_ev", "score_ev_long", "score_ev_short", "predicted_side", "p_target", "p_stop", "p_vertical"]],
            short_rows[["event_id", "score_ev", "score_ev_long", "score_ev_short", "predicted_side", "p_target", "p_stop", "p_vertical"]],
        ],
        ignore_index=True,
    )

    return out


def combine_dual_meta_predictions(
    long_meta: pd.DataFrame,
    short_meta: pd.DataFrame,
    combined_primary: pd.DataFrame,
) -> pd.DataFrame:
    base = combined_primary[["event_id", "predicted_side"]].copy()
    merged = base.merge(
        long_meta[["event_id", "p_meta"]],
        on="event_id",
        how="left",
    ).rename(columns={"p_meta": "p_meta_long"})
    merged = merged.merge(
        short_meta[["event_id", "p_meta"]],
        on="event_id",
        how="left",
    ).rename(columns={"p_meta": "p_meta_short"})

    predicted_side = merged["predicted_side"].to_numpy()
    p_meta = np.full(len(merged), np.nan, dtype=float)
    long_mask = predicted_side == 1
    short_mask = predicted_side == -1
    p_meta[long_mask] = merged.loc[long_mask, "p_meta_long"].astype(float)
    p_meta[short_mask] = merged.loc[short_mask, "p_meta_short"].astype(float)

    out = pd.DataFrame(
        {
            "event_id": merged["event_id"].to_numpy(),
            "p_meta": p_meta,
            "predicted_side": predicted_side,
            "p_meta_long": merged["p_meta_long"].to_numpy(),
            "p_meta_short": merged["p_meta_short"].to_numpy(),
        }
    )

    return out
