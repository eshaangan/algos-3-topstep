"""
Triple-barrier labeling logic.

Implements first-touch determination with execution-aligned costs.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.instrument import InstrumentSpec

logger = logging.getLogger(__name__)


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


def _decide_ohlc_touch(
    open_p: float, high_p: float, low_p: float, upper: float, lower: float
) -> str:
    """
    Deterministic within-bar path: open -> high -> low -> close.
    Returns "target", "stop", or "" if no barrier hit.
    """
    if open_p >= upper:
        return "target"
    if open_p <= lower:
        return "stop"
    if high_p >= upper:
        return "target"
    if low_p <= lower:
        return "stop"
    return ""


def _decide_ohlc_touch_side(
    open_p: float,
    high_p: float,
    low_p: float,
    target_level: float,
    stop_level: float,
    side: int,
) -> str:
    """
    Deterministic within-bar path: open -> high -> low -> close.

    `target_level` is the profit-taking barrier for the given `side`.
    `stop_level` is the stop-loss barrier for the given `side`.
    """
    if side >= 0:
        # Long: target above, stop below
        if open_p >= target_level:
            return "target"
        if open_p <= stop_level:
            return "stop"
        if high_p >= target_level:
            return "target"
        if low_p <= stop_level:
            return "stop"
        return ""

    # Short: target below, stop above
    if open_p <= target_level:
        return "target"
    if open_p >= stop_level:
        return "stop"
    if high_p >= stop_level:
        return "stop"
    if low_p <= target_level:
        return "target"
    return ""


def _compute_cost_points(
    execution_spec: dict, bar_size: str, instrument_spec: InstrumentSpec
) -> Tuple[float, float]:
    """Return total round-trip cost in points and per-side cost in points."""
    costs = execution_spec.get("costs", {})
    slippage_ticks = costs.get("slippage_ticks", {}).get(bar_size, 0.0)
    commission = float(costs.get("commission_per_contract", 0.0))

    point_value = instrument_spec.contract_multiplier_usd_per_point
    commission_points = commission / point_value if point_value else 0.0
    slippage_points = (
        float(slippage_ticks) * instrument_spec.tick_size_points
    )
    per_side = commission_points + slippage_points
    round_trip = 2.0 * per_side
    return round_trip, per_side


def apply_triplebarrier(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    bar_size: str,
    labeling_config: dict,
    execution_spec: dict,
    instrument_spec: InstrumentSpec,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to events.

    Labels:
      +1: profit target hit first
      -1: stop loss hit first
       0: vertical barrier hit (or no touch)
    """
    df = bars_df.sort_index()

    required_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    events = events_df.copy().reset_index(drop=True)
    if events.empty:
        return events

    index = df.index
    t0_idx = index.get_indexer(events["t0"])
    t1_idx = index.get_indexer(events["t1"])

    if (t0_idx < 0).any() or (t1_idx < 0).any():
        raise ValueError("Some event t0/t1 timestamps not found in bars index")

    fill_price = (
        execution_spec.get("fill_model", {}).get("fill_price", "next_bar_open")
    )
    touch_ordering = (
        execution_spec.get("fill_model", {}).get("touch_ordering", "ohlc_path")
    )
    offsets = _get_fill_offsets(fill_price)
    entry_offset = offsets["entry_offset"]
    barrier_start_offset = offsets["barrier_start_offset"]
    entry_col = offsets["entry_col"]

    entry_idx = t0_idx + entry_offset
    barrier_start_idx = t0_idx + barrier_start_offset

    n = len(df)
    entry_prices = np.full(len(events), np.nan)
    entry_times = np.full(len(events), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_entry = entry_idx < n
    if valid_entry.any():
        entry_series = df[entry_col].to_numpy()
        entry_prices[valid_entry] = entry_series[entry_idx[valid_entry]]
        entry_times[valid_entry] = index[entry_idx[valid_entry]].to_numpy()

    sigma = (
        events["sigma"].to_numpy()
        if "sigma" in events.columns
        else None
    )

    tb_cfg = labeling_config.get("primary_labeling", {}).get("triple_barrier", {})
    label_encoding = tb_cfg.get(
        "label_encoding", {"target_first": 1, "stop_first": -1, "vertical": 0}
    )
    account_for_costs = bool(tb_cfg.get("account_for_costs", False))

    total_cost_points, _ = _compute_cost_points(
        execution_spec, bar_size, instrument_spec
    )
    cost_buffer = total_cost_points if account_for_costs else 0.0

    open_arr = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()

    y = np.full(len(events), np.nan)
    t_touch = np.full(len(events), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_price = np.full(len(events), np.nan)
    exit_reason = np.array([""] * len(events), dtype=object)
    ret_gross = np.full(len(events), np.nan)
    ret_net = np.full(len(events), np.nan)

    for i in range(len(events)):
        if not valid_entry[i]:
            continue
        if np.isnan(entry_prices[i]):
            continue
        if sigma is None or np.isnan(sigma[i]):
            continue
        if t1_idx[i] >= n or t1_idx[i] < 0:
            continue

        side_val = 1
        if "side" in events.columns and pd.notna(events.loc[i, "side"]):
            try:
                side_val = int(events.loc[i, "side"])
            except Exception:
                side_val = 1
        if side_val == 0:
            side_val = 1
        side_val = 1 if side_val > 0 else -1

        start_idx = barrier_start_idx[i]
        end_idx = t1_idx[i]
        if start_idx > end_idx:
            start_idx = end_idx + 1

        p0 = float(entry_prices[i])
        pt_points = float(events.loc[i, "pt_mult"]) * sigma[i] + cost_buffer
        sl_points = float(events.loc[i, "sl_mult"]) * sigma[i] + cost_buffer

        if side_val > 0:
            target_level = p0 + pt_points
            stop_level = p0 - sl_points
        else:
            target_level = p0 - pt_points
            stop_level = p0 + sl_points

        touch = ""
        touch_idx = -1

        for j in range(start_idx, end_idx + 1):
            high = high_arr[j]
            low = low_arr[j]
            if np.isnan(high) or np.isnan(low):
                continue

            if side_val > 0:
                hit_target = high >= target_level
                hit_stop = low <= stop_level
            else:
                hit_target = low <= target_level
                hit_stop = high >= stop_level

            if not hit_target and not hit_stop:
                continue

            if hit_target and hit_stop:
                if touch_ordering == "stop_first":
                    touch = "stop"
                elif touch_ordering == "target_first":
                    touch = "target"
                elif touch_ordering == "ohlc_path":
                    touch = _decide_ohlc_touch_side(
                        open_arr[j],
                        high,
                        low,
                        target_level,
                        stop_level,
                        side_val,
                    )
                else:
                    raise ValueError(
                        f"Unsupported touch_ordering: {touch_ordering}"
                    )
            elif hit_target:
                touch = "target"
            else:
                touch = "stop"

            if touch:
                touch_idx = j
                break

        if touch_idx >= 0:
            if touch == "target":
                exit_reason[i] = "target"
                exit_price[i] = target_level
                y[i] = label_encoding.get("target_first", 1)
            else:
                exit_reason[i] = "stop"
                exit_price[i] = stop_level
                y[i] = label_encoding.get("stop_first", -1)
            t_touch[i] = index[touch_idx].to_numpy()
        else:
            exit_reason[i] = "vertical"
            exit_price[i] = close_arr[end_idx]
            y[i] = label_encoding.get("vertical", 0)
            t_touch[i] = index[end_idx].to_numpy()

        ret_gross[i] = side_val * (exit_price[i] - p0)
        if account_for_costs:
            ret_net[i] = ret_gross[i] - total_cost_points
        else:
            ret_net[i] = ret_gross[i]

    events["entry_time"] = entry_times
    events["entry_price"] = entry_prices
    events["t_touch"] = t_touch
    events["exit_price"] = exit_price
    events["exit_reason"] = exit_reason
    events["y"] = y
    events["ret_gross"] = ret_gross
    events["ret_net"] = ret_net

    valid_mask = ~np.isnan(events["y"].to_numpy())
    events = events[valid_mask].reset_index(drop=True)

    ordered_cols = [
        "event_id",
        "t0",
        "t1",
        "bar_size",
        "side",
        "sigma",
        "pt_mult",
        "sl_mult",
        "horizon_bars",
        "entry_time",
        "entry_price",
        "t_touch",
        "exit_price",
        "exit_reason",
        "y",
        "ret_gross",
        "ret_net",
    ]
    remaining_cols = [c for c in events.columns if c not in ordered_cols]
    events = events[ordered_cols + remaining_cols]

    return events
