"""
Event generation for triple-barrier labeling.

Creates events for each feasible bar under the configured event policy.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_atr(bars_df: pd.DataFrame, atr_period: int) -> pd.Series:
    """Compute ATR using the same true-range definition as features."""
    df = bars_df.sort_index()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=atr_period, adjust=False).mean()


def _get_fill_offsets(fill_price: str) -> Dict[str, int]:
    """
    Determine entry and barrier-start offsets relative to t0.

    - next_bar_open: entry on t0+1 open; barriers start at t0+1
    - next_bar_close: entry on t0+1 close; barriers start at t0+2
    """
    if fill_price == "next_bar_open":
        return {"entry_offset": 1, "barrier_start_offset": 1, "entry_col": "open"}
    if fill_price == "next_bar_close":
        return {"entry_offset": 1, "barrier_start_offset": 2, "entry_col": "close"}
    raise ValueError(f"Unsupported fill_price: {fill_price}")


def generate_events(
    bars_df: pd.DataFrame,
    bar_size: str,
    labeling_config: dict,
    execution_spec: dict,
) -> pd.DataFrame:
    """
    Generate events for triple-barrier labeling.

    Feasibility rules:
    - close at t0 must be non-NaN
    - do not start events on synthetic bars by default (if is_synthetic exists)
    - must have enough future bars to reach vertical barrier t1
    - entry fill price at t0+1 must be non-NaN
    """
    df = bars_df.sort_index()

    required_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    primary_cfg = labeling_config.get("primary_labeling", {})
    event_policy = primary_cfg.get("event_policy", "every_bar")
    if event_policy != "every_bar":
        raise NotImplementedError(f"event_policy not supported: {event_policy}")

    tb_cfg = primary_cfg.get("triple_barrier", {})
    pt_multipliers = tb_cfg.get("pt_multipliers", [1.0])
    sl_multipliers = tb_cfg.get("sl_multipliers", [1.0])
    horizon_map = tb_cfg.get("horizon_bars", {})
    if bar_size not in horizon_map:
        raise ValueError(f"Missing horizon_bars for bar_size={bar_size}")
    horizon_bars_list = horizon_map[bar_size]

    # Validate against execution spec holding constraints
    max_holding = (
        execution_spec.get("holding_constraints", {})
        .get("max_holding_bars", {})
        .get(bar_size)
    )
    if max_holding is not None and max(horizon_bars_list) > max_holding:
        raise ValueError(
            f"horizon_bars max {max(horizon_bars_list)} exceeds execution_spec "
            f"max_holding_bars {max_holding} for {bar_size}"
        )

    atr_period = tb_cfg.get("volatility_params", {}).get("atr_period", 14)
    sigma = _compute_atr(df, atr_period)

    fill_price = (
        execution_spec.get("fill_model", {}).get("fill_price", "next_bar_open")
    )
    offsets = _get_fill_offsets(fill_price)
    entry_offset = offsets["entry_offset"]
    barrier_start_offset = offsets["barrier_start_offset"]
    entry_col = offsets["entry_col"]

    close_ok = df["close"].notna()
    sigma_ok = sigma.notna()
    base_ok = close_ok & sigma_ok

    if "is_synthetic" in df.columns:
        synth = df["is_synthetic"].astype(bool)
        base_ok = base_ok & (~synth)

    entry_price_ok = df[entry_col].shift(-entry_offset).notna()
    base_ok = base_ok & entry_price_ok

    if "is_synthetic" in df.columns:
        entry_synth = df["is_synthetic"].shift(
            -entry_offset, fill_value=False
        ).astype(bool)
        base_ok = base_ok & (~entry_synth)

    n = len(df)
    t0_idx_all = np.arange(n)
    events = []

    for horizon_bars in horizon_bars_list:
        t1_idx = t0_idx_all + barrier_start_offset + horizon_bars - 1
        feasible = base_ok & (t1_idx < n)

        if not feasible.any():
            continue

        t0_idx = t0_idx_all[feasible.to_numpy()]
        t1_idx = t1_idx[feasible.to_numpy()]

        base_df = pd.DataFrame(
            {
                "t0": df.index[t0_idx],
                "t1": df.index[t1_idx],
                "bar_size": bar_size,
                "side": 0,
                "sigma": sigma.iloc[t0_idx].to_numpy(),
                "horizon_bars": horizon_bars,
            }
        )

        for pt_mult in pt_multipliers:
            for sl_mult in sl_multipliers:
                df_events = base_df.copy()
                df_events["pt_mult"] = pt_mult
                df_events["sl_mult"] = sl_mult
                events.append(df_events)

    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "t0",
                "t1",
                "bar_size",
                "side",
                "sigma",
                "pt_mult",
                "sl_mult",
                "horizon_bars",
            ]
        )

    events_df = pd.concat(events, axis=0, ignore_index=True)
    events_df.insert(0, "event_id", np.arange(len(events_df), dtype=int))

    return events_df
