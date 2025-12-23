"""
Entry/exit and cost computation consistent with labeling assumptions.
"""

from dataclasses import dataclass
import pandas as pd

TICK_SIZE_POINTS = 0.25


@dataclass
class FillResult:
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: pd.Timestamp
    exit_px: float
    exit_reason: str


def _get_fill_offsets(fill_price: str) -> dict:
    if fill_price == "next_bar_open":
        return {"entry_offset": 1, "entry_col": "open"}
    if fill_price == "next_bar_close":
        return {"entry_offset": 1, "entry_col": "close"}
    raise ValueError(f"Unsupported fill_price: {fill_price}")


def compute_cost_points(execution_spec: dict, risk_cfg: dict, bar_size: str) -> float:
    costs = execution_spec.get("costs", {})
    slippage_ticks = float(costs.get("slippage_ticks", {}).get(bar_size, 0.0))
    commission = float(costs.get("commission_per_contract", 0.0))
    contract_multiplier = float(
        risk_cfg.get("topstep", {}).get("contract_multiplier", 1.0)
    )
    commission_points = commission / contract_multiplier if contract_multiplier else 0.0
    slippage_points = slippage_ticks * TICK_SIZE_POINTS
    return 2.0 * (commission_points + slippage_points)


def get_entry_exit(
    event_row: pd.Series,
    bars_df: pd.DataFrame,
    execution_spec: dict,
) -> FillResult:
    """
    Determine entry/exit from events fields; fallback to bars if missing.
    """
    fill_price = execution_spec.get("fill_model", {}).get(
        "fill_price", "next_bar_open"
    )
    offsets = _get_fill_offsets(fill_price)

    if "entry_time" in event_row and pd.notna(event_row["entry_time"]):
        entry_ts = pd.Timestamp(event_row["entry_time"])
        entry_px = float(event_row["entry_price"])
    else:
        t0 = pd.Timestamp(event_row["t0"])
        idx = bars_df.index.searchsorted(t0, side="left")
        entry_idx = min(idx + offsets["entry_offset"], len(bars_df) - 1)
        entry_ts = bars_df.index[entry_idx]
        entry_px = float(bars_df.iloc[entry_idx][offsets["entry_col"]])

    if "t_touch" in event_row and pd.notna(event_row["t_touch"]):
        exit_ts = pd.Timestamp(event_row["t_touch"])
    else:
        exit_ts = pd.Timestamp(event_row["t1"])

    if "exit_price" in event_row and pd.notna(event_row["exit_price"]):
        exit_px = float(event_row["exit_price"])
        exit_reason = "event_exit"
    else:
        exit_px = float(bars_df.loc[exit_ts]["close"])
        exit_reason = "event_exit"

    return FillResult(
        entry_ts=entry_ts,
        entry_px=entry_px,
        exit_ts=exit_ts,
        exit_px=exit_px,
        exit_reason=exit_reason,
    )


def apply_forced_flatten(
    fill: FillResult,
    bars_df: pd.DataFrame,
    flatten_time_chicago: str | None,
) -> FillResult:
    if not flatten_time_chicago:
        return fill

    idx = bars_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    chicago = idx.tz_convert("America/Chicago")

    entry_chi = fill.entry_ts.tz_localize("UTC") if fill.entry_ts.tzinfo is None else fill.entry_ts
    entry_chi = entry_chi.tz_convert("America/Chicago")
    exit_chi = fill.exit_ts.tz_localize("UTC") if fill.exit_ts.tzinfo is None else fill.exit_ts
    exit_chi = exit_chi.tz_convert("America/Chicago")

    flatten_time = pd.Timestamp(
        f"{entry_chi.date()} {flatten_time_chicago}", tz="America/Chicago"
    )
    if exit_chi <= flatten_time:
        return fill

    flatten_pos = chicago.searchsorted(flatten_time, side="left")
    if flatten_pos >= len(idx):
        return fill
    flatten_ts = idx[flatten_pos]
    exit_px = float(bars_df.iloc[flatten_pos]["close"])
    return FillResult(
        entry_ts=fill.entry_ts,
        entry_px=fill.entry_px,
        exit_ts=flatten_ts,
        exit_px=exit_px,
        exit_reason="forced_flatten",
    )
