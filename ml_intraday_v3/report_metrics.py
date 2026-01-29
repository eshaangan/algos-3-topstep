#!/usr/bin/env python3
"""
Report all metrics from the latest replay run (trades + metrics CSVs in logs/).

Usage:
  python ml_intraday_v3/report_metrics.py [--trades PATH] [--metrics PATH]
  If paths omitted, uses most recent trades_*.csv and metrics_*.csv in logs/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _find_latest(base: Path, pattern: str) -> Path | None:
    files = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def report(trades_path: Path, metrics_path: Path | None) -> None:
    trades = pd.read_csv(trades_path)
    if trades.empty:
        print("No trades in file.")
        return

    pnl_col = next((c for c in ["pnl", "pnl_usd", "realized_pnl"] if c in trades.columns), None)
    if not pnl_col:
        print("No P&L column in trades.")
        return

    dir_col = "direction" if "direction" in trades.columns else ("side" if "side" in trades.columns else None)
    long_val = "LONG" if dir_col == "direction" else 1
    short_val = "SHORT" if dir_col == "direction" else -1

    # --- From trades ---
    n = len(trades)
    pnl = trades[pnl_col].astype(float)
    winners = (pnl > 0).sum()
    losers = (pnl < 0).sum()
    total_pnl = pnl.sum()
    gross_profit = pnl[pnl > 0].sum() if winners else 0.0
    gross_loss = abs(pnl[pnl < 0].sum()) if losers else 0.0
    avg_win = pnl[pnl > 0].mean() if winners else 0.0
    avg_loss = pnl[pnl < 0].mean() if losers else 0.0
    avg_trade = pnl.mean()
    best_trade = pnl.max()
    worst_trade = pnl.min()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    std_pnl = pnl.std()
    sharpe_trade = (avg_trade / std_pnl * (n**0.5)) if std_pnl and std_pnl > 0 else 0.0

    long_trades = int((trades[dir_col] == long_val).sum()) if dir_col else 0
    short_trades = int((trades[dir_col] == short_val).sum()) if dir_col else 0
    long_pct = 100.0 * long_trades / n if dir_col and n else 0.0
    short_pct = 100.0 * short_trades / n if dir_col and n else 0.0

    exit_col = "exit_reason" if "exit_reason" in trades.columns else None
    if exit_col:
        exit_counts = trades[exit_col].value_counts()
        target_count = int(exit_counts.get("target", 0))
        stop_count = int(exit_counts.get("stop", 0))
        other_exits = int((trades[exit_col].notna() & ~trades[exit_col].isin(["target", "stop"])).sum())
    else:
        target_count = stop_count = other_exits = 0

    # --- From metrics (last row) ---
    md = {}
    if metrics_path and metrics_path.exists():
        m = pd.read_csv(metrics_path)
        if not m.empty:
            r = m.iloc[-1]
            md = {
                "max_drawdown": r.get("max_drawdown"),
                "max_drawdown_pct": r.get("max_drawdown_pct"),
                "peak_equity": r.get("peak_equity"),
                "current_equity": r.get("current_equity"),
                "return_pct": r.get("return_pct"),
                "signals_generated": int(r.get("signals_generated", 0)),
                "signals_executed": int(r.get("signals_executed", 0)),
                "signals_rejected": int(r.get("signals_rejected", 0)),
                "execution_rate_pct": float(r.get("execution_rate", 0)),
                "max_winning_streak": int(r.get("max_winning_streak", 0)),
                "max_losing_streak": int(r.get("max_losing_streak", 0)),
                "current_streak": int(r.get("current_streak", 0)),
            }

    # --- Print ---
    print()
    print("=" * 70)
    print("METRICS REPORT")
    print("=" * 70)
    print(f"  Trades file: {trades_path.name}")
    if metrics_path:
        print(f"  Metrics file: {metrics_path.name if metrics_path.exists() else 'N/A'}")
    print()
    print("--- TRADES ---")
    print(f"  total_trades        {n}")
    if dir_col:
        print(f"  long_trades         {long_trades} ({long_pct:.1f}%)")
        print(f"  short_trades        {short_trades} ({short_pct:.1f}%)")
    print(f"  winners             {winners}")
    print(f"  losers              {losers}")
    print(f"  win_rate            {100.0 * winners / n:.1f}%")
    if exit_col:
        print(f"  exits_target        {target_count}")
        print(f"  exits_stop          {stop_count}")
        if other_exits:
            print(f"  exits_other         {other_exits}")
    print()
    print("--- P&L ---")
    print(f"  total_pnl           ${total_pnl:,.2f}")
    print(f"  gross_profit        ${gross_profit:,.2f}")
    print(f"  gross_loss          ${gross_loss:,.2f}")
    print(f"  avg_win             ${avg_win:,.2f}")
    print(f"  avg_loss            ${avg_loss:,.2f}")
    print(f"  avg_trade           ${avg_trade:,.2f}")
    print(f"  best_trade          ${best_trade:,.2f}")
    print(f"  worst_trade         ${worst_trade:,.2f}")
    print(f"  profit_factor       {profit_factor:.2f}" if profit_factor != float("inf") else "  profit_factor       INF")
    print(f"  sharpe_trade        {sharpe_trade:.2f}")
    print()
    if md:
        print("--- RISK / SESSION (from metrics) ---")
        if md.get("max_drawdown") is not None:
            print(f"  max_drawdown        ${md['max_drawdown']:,.2f}")
        if md.get("max_drawdown_pct") is not None:
            print(f"  max_drawdown_pct    {md['max_drawdown_pct']:.2f}%")
        if md.get("peak_equity") is not None:
            print(f"  peak_equity         ${md['peak_equity']:,.2f}")
        if md.get("current_equity") is not None:
            print(f"  current_equity      ${md['current_equity']:,.2f}")
        if md.get("return_pct") is not None:
            print(f"  return_pct          {md['return_pct']:.2f}%")
        if md.get("max_winning_streak") is not None:
            print(f"  max_winning_streak  {md['max_winning_streak']}")
        if md.get("max_losing_streak") is not None:
            print(f"  max_losing_streak   {md['max_losing_streak']}")
        if md.get("current_streak") is not None:
            print(f"  current_streak      {md['current_streak']}")
        print()
        print("--- SIGNALS ---")
        if md.get("signals_generated") is not None:
            print(f"  signals_generated   {md['signals_generated']}")
        if md.get("signals_executed") is not None:
            print(f"  signals_executed    {md['signals_executed']}")
        if md.get("signals_rejected") is not None:
            print(f"  signals_rejected    {md['signals_rejected']}")
        if md.get("execution_rate_pct") is not None:
            print(f"  execution_rate      {md['execution_rate_pct']:.2f}%")
    print("=" * 70)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Report metrics from replay trades and metrics CSVs.")
    ap.add_argument("--trades", type=Path, help="Path to trades CSV")
    ap.add_argument("--metrics", type=Path, help="Path to metrics CSV (optional)")
    args = ap.parse_args()

    logs = project_root / "logs"
    if not logs.exists():
        print("logs/ not found")
        return 1

    trades_path = args.trades or _find_latest(logs, "trades_*.csv")
    if not trades_path or not trades_path.exists():
        print("No trades_*.csv found in logs/")
        return 1

    # Infer metrics from trades timestamp if not given
    metrics_path = args.metrics
    if not metrics_path:
        # trades_20260127_160400 -> metrics_20260127_160400
        stem = trades_path.stem  # trades_20260127_160400
        if stem.startswith("trades_"):
            metrics_path = logs / stem.replace("trades_", "metrics_", 1) + ".csv"
        if not (metrics_path and metrics_path.exists()):
            metrics_path = _find_latest(logs, "metrics_*.csv")

    report(trades_path, metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
