"""
Risk gate audits using trades and equity curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_trades(backtest_dir: Path) -> pd.DataFrame:
    trade_files = list(backtest_dir.glob("*/trades.parquet"))
    if not trade_files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in trade_files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_equity(backtest_dir: Path) -> pd.DataFrame:
    eq_files = list(backtest_dir.glob("*/equity.parquet"))
    if not eq_files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in eq_files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _find_backtest_dir(bar_dir: Path) -> Path | None:
    base = bar_dir / "backtests"
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    for name in ["purged_kfold", "cpcv"]:
        candidate = base / name
        if candidate.exists():
            return candidate
    return subdirs[0]


def _assign_session_date(
    ts: pd.Series, reset_time: str, reset_tz: str
) -> pd.Series:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    dt = dt.dt.tz_convert(reset_tz)
    reset_clock = pd.to_datetime(reset_time).time()
    session_date = dt.dt.date
    needs_prev = dt.dt.time < reset_clock
    session_date = pd.to_datetime(session_date) - pd.to_timedelta(
        needs_prev.astype(int), unit="D"
    )
    return session_date


def check_daily_loss_limit(bar_dir: Path, risk_cfg: dict) -> dict:
    backtest_dir = _find_backtest_dir(bar_dir)
    if backtest_dir is None:
        return {"status": "SKIP", "reason": "no_backtests"}

    trades_df = _load_trades(backtest_dir)
    if trades_df.empty:
        return {"status": "SKIP", "reason": "no_trades"}

    executed = trades_df[trades_df.get("executed", False)].copy()
    if executed.empty:
        return {"status": "SKIP", "reason": "no_executed_trades"}

    daily_cfg = risk_cfg.get("daily_loss_limit", {})
    if not daily_cfg.get("enabled", False):
        return {"status": "SKIP", "reason": "daily_loss_disabled"}

    max_loss = float(daily_cfg.get("max_daily_loss", 0.0))
    reset_time = daily_cfg.get("reset_time", "17:00")
    reset_tz = daily_cfg.get("reset_timezone", "America/Chicago")

    if "exit_ts" not in executed.columns:
        return {"status": "SKIP", "reason": "missing_exit_ts"}

    session_date = _assign_session_date(
        executed["exit_ts"], reset_time, reset_tz
    )
    pnl = executed["pnl_usd"].astype(float)
    daily_pnl = pnl.groupby(session_date).sum()

    violations = int((daily_pnl < -max_loss).sum()) if max_loss > 0 else 0
    status = "PASS" if violations == 0 else "FAIL"

    return {
        "status": status,
        "violations": violations,
        "max_daily_loss": max_loss,
    }


def check_trailing_drawdown(bar_dir: Path, risk_cfg: dict) -> dict:
    backtest_dir = _find_backtest_dir(bar_dir)
    if backtest_dir is None:
        return {"status": "SKIP", "reason": "no_backtests"}

    eq_df = _load_equity(backtest_dir)
    if eq_df.empty:
        return {"status": "SKIP", "reason": "no_equity"}

    dd_cfg = risk_cfg.get("trailing_drawdown", {})
    if not dd_cfg.get("enabled", False):
        return {"status": "SKIP", "reason": "drawdown_disabled"}

    max_dd = float(dd_cfg.get("max_drawdown", 0.0))
    if max_dd <= 0:
        return {"status": "SKIP", "reason": "max_drawdown_not_set"}

    if "equity" not in eq_df.columns:
        return {"status": "SKIP", "reason": "missing_equity_column"}

    equity = eq_df["equity"].astype(float).to_numpy()
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    violations = int((drawdown > max_dd).sum())
    status = "PASS" if violations == 0 else "FAIL"

    return {
        "status": status,
        "violations": violations,
        "max_drawdown": max_dd,
    }


def check_forced_flatten(bar_dir: Path, backtest_cfg: dict) -> dict:
    backtest_dir = _find_backtest_dir(bar_dir)
    if backtest_dir is None:
        return {"status": "SKIP", "reason": "no_backtests"}

    trades_df = _load_trades(backtest_dir)
    if trades_df.empty:
        return {"status": "SKIP", "reason": "no_trades"}

    executed = trades_df[trades_df.get("executed", False)].copy()
    if executed.empty:
        return {"status": "SKIP", "reason": "no_executed_trades"}

    flatten_time = backtest_cfg.get("session", {}).get(
        "flatten_time_chicago", None
    )
    if not flatten_time:
        return {"status": "SKIP", "reason": "flatten_time_not_set"}

    entry_ts = pd.to_datetime(executed["entry_ts"], utc=True, errors="coerce")
    exit_ts = pd.to_datetime(executed["exit_ts"], utc=True, errors="coerce")

    entry_chi = entry_ts.dt.tz_convert("America/Chicago")
    exit_chi = exit_ts.dt.tz_convert("America/Chicago")
    flatten_ts = pd.to_datetime(
        entry_chi.dt.date.astype(str) + " " + flatten_time,
        utc=False,
    ).dt.tz_localize("America/Chicago")

    violation_mask = exit_chi > flatten_ts
    if "exit_reason" in executed.columns:
        violation_mask &= executed["exit_reason"] != "forced_flatten"

    violations = int(violation_mask.sum())
    status = "PASS" if violations == 0 else "FAIL"

    return {"status": status, "violations": violations}
