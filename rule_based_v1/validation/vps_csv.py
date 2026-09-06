"""Load VPS strategy-lab trade CSVs into the validation harness trade schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ET = "America/New_York"

_NON_TRADE = frozenset({"nofill", "no_fill", "skip", "skip_wide", "cancelled"})

_OUTCOME_MAP = {
    "tp": "profit_target",
    "sl": "stop_loss",
    "be": "break_even",
}

_DIR_MAP = {"long": 1, "short": -1}


def load_vps_trades(path: Path | str) -> pd.DataFrame:
    """Convert a VPS book CSV into harness trades (entry_time, pnl, ...).

    VPS schema: date,time,direction,entry,stop,target,exit,[rr,]pnl_pts,pnl_usd,outcome
    Filters NON_TRADE outcomes (nofill, skip, cancelled, etc.).
    """
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "entry_time", "pnl", "direction", "entry_price",
                "exit_price", "exit_reason",
            ]
        )

    outcome = df["outcome"].astype(str).str.strip().str.lower()
    df = df.loc[~outcome.isin(_NON_TRADE)].copy()
    outcome = outcome.loc[df.index]

    entry_naive = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y-%m-%d %H:%M",
    )
    entry_et = entry_naive.dt.tz_localize(ET)
    entry_utc = entry_et.dt.tz_convert("UTC")

    direction = (
        df["direction"].astype(str).str.strip().str.lower().map(_DIR_MAP).astype("Int64")
    )
    if direction.isna().any():
        bad = df.loc[direction.isna(), "direction"].unique().tolist()
        raise ValueError(f"Unknown direction values in {path}: {bad}")

    exit_reason = outcome.map(lambda o: _OUTCOME_MAP.get(o, o))

    out = pd.DataFrame({
        "entry_time": entry_utc,
        "pnl": df["pnl_usd"].astype(float),
        "direction": direction.astype(int),
        "entry_price": df["entry"].astype(float),
        "exit_price": df["exit"].astype(float),
        "exit_reason": exit_reason.to_numpy(),
        # carried so harness.execution_realism() can audit the bracket; without
        # these the realism check reports "unverifiable" and the gate fails.
        "stop_price": df["stop"].astype(float),
        "target_price": df["target"].astype(float),
    })
    return out.reset_index(drop=True)


def slice_holdout(
    trades: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Filter trades to entry_time calendar dates in [start, end] inclusive."""
    if trades is None or len(trades) == 0:
        return trades.iloc[0:0].copy() if trades is not None else pd.DataFrame()
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    dates = pd.to_datetime(trades["entry_time"]).dt.tz_convert(ET).dt.date
    mask = (dates >= start_d) & (dates <= end_d)
    return trades.loc[mask].reset_index(drop=True)
