"""
Monte Carlo pass-rate simulation for Topstep combine rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.risk_presets import RISK_PRESET_NAME, get_risk_preset


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Trades missing columns: {missing}")


def load_trades_csv(path: str) -> pd.DataFrame:
    trades = pd.read_csv(path)
    _require_columns(trades, ["pnl", "entry_time"])
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time"]).copy()
    return trades


def _trade_count_distribution(trades: pd.DataFrame) -> np.ndarray:
    trades["trade_day"] = trades["entry_time"].dt.date
    counts = trades.groupby("trade_day")["pnl"].count().values
    return counts.astype(int)


def simulate_combine(
    trades: pd.DataFrame,
    *,
    starting_balance: float,
    profit_target: float,
    daily_loss_limit: float,
    trailing_drawdown: float,
    runs: int = 10_000,
    seed: int = 42,
    max_days: int = 252,
    consistency_limit: Optional[float] = None,
) -> Dict[str, object]:
    """
    Simulate Topstep combine pass/fail using empirical trade distribution.

    Rules:
      - profit_target: pass if equity - starting_balance >= target
      - daily_loss_limit: fail if daily PnL <= -limit
      - trailing_drawdown: fail if peak_equity - equity >= limit
    """
    _require_columns(trades, ["pnl", "entry_time"])
    if trades.empty:
        raise ValueError("No trades provided for simulation.")

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time"]).copy()
    if trades.empty:
        raise ValueError("No valid entry_time values after parsing.")

    pnl_values = trades["pnl"].astype(float).values
    trade_counts = _trade_count_distribution(trades)
    if len(trade_counts) == 0:
        raise ValueError("Trade count distribution is empty.")

    rng = np.random.default_rng(seed)

    pass_flags = []
    days_to_pass = []
    fail_reasons: Dict[str, int] = {"daily_loss": 0, "trailing_drawdown": 0, "consistency_limit": 0, "max_days": 0}
    max_drawdowns = []

    for _ in range(runs):
        equity = starting_balance
        peak = equity
        max_dd = 0.0
        passed = False
        reason = None

        for day in range(1, max_days + 1):
            trades_today = int(rng.choice(trade_counts))
            daily_pnl = 0.0

            for _ in range(trades_today):
                trade_pnl = float(rng.choice(pnl_values))
                equity += trade_pnl
                daily_pnl += trade_pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)

                if equity - starting_balance >= profit_target:
                    passed = True
                    days_to_pass.append(day)
                    break
                if trailing_drawdown > 0 and peak - equity >= trailing_drawdown:
                    reason = "trailing_drawdown"
                    break
                if daily_loss_limit > 0 and daily_pnl <= -daily_loss_limit:
                    reason = "daily_loss"
                    break

            if consistency_limit is not None and daily_pnl > consistency_limit:
                reason = "consistency_limit"

            if passed or reason:
                break

        if not passed:
            reason = reason or "max_days"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        pass_flags.append(passed)
        max_drawdowns.append(max_dd)

    pass_rate = float(np.mean(pass_flags))

    def _percentiles(values: list[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=float)
        return {
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }

    return {
        "runs": runs,
        "pass_rate": pass_rate,
        "days_to_pass": _percentiles(days_to_pass),
        "fail_reasons": fail_reasons,
        "max_drawdown": _percentiles(max_drawdowns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo Topstep combine pass-rate simulation")
    parser.add_argument("--trades-csv", required=True, help="Backtest trades CSV (must include pnl and entry_time)")
    parser.add_argument("--preset", default=RISK_PRESET_NAME, help="Risk preset name (DEFAULT or TOPSTEP_50K)")
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=252)
    parser.add_argument("--profit-target", type=float, default=None)
    parser.add_argument("--daily-loss", type=float, default=None)
    parser.add_argument("--trailing-dd", type=float, default=None)
    parser.add_argument("--consistency-limit", type=float, default=None)
    args = parser.parse_args()

    preset = get_risk_preset(args.preset)
    risk_cfg = preset.risk_config

    profit_target = args.profit_target if args.profit_target is not None else preset.profit_target
    daily_loss = args.daily_loss if args.daily_loss is not None else risk_cfg.max_daily_loss
    trailing_dd = args.trailing_dd if args.trailing_dd is not None else risk_cfg.trailing_drawdown
    consistency = args.consistency_limit if args.consistency_limit is not None else preset.consistency_limit

    trades = load_trades_csv(args.trades_csv)
    result = simulate_combine(
        trades,
        starting_balance=risk_cfg.starting_balance,
        profit_target=profit_target,
        daily_loss_limit=daily_loss,
        trailing_drawdown=trailing_dd,
        runs=args.runs,
        seed=args.seed,
        max_days=args.max_days,
        consistency_limit=consistency,
    )

    payload = {
        "preset": preset.name,
        "risk_config": asdict(risk_cfg),
        "profit_target": profit_target,
        "daily_loss_limit": daily_loss,
        "trailing_drawdown": trailing_dd,
        "consistency_limit": consistency,
        "result": result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
