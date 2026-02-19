"""
Topstep 50k Combine Simulator
==============================
Simulates whether a trade/daily-PnL sequence passes Topstep combine rules.
Supports both deterministic simulation and Monte Carlo bootstrapping.

Topstep 50k Rules:
  - Account size: $50,000
  - Profit target: $3,000 (in 5+ trading days)
  - Max trailing drawdown: $2,000 (trails high-water mark, starts at $48,000 floor)
  - Max daily loss: $1,000
  - Consistency: no single day > 30% of total cumulative profit
  - Min trading days: 5 days before you can hit the profit target
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class CombineResult:
    passed: bool
    days_to_pass: Optional[int]         # None if failed
    fail_reason: Optional[str]          # None if passed
    peak_account: float
    min_account: float                  # lowest account value seen
    max_drawdown_reached: float         # max drawdown from peak (positive number)
    daily_loss_breaches: int
    consistency_violations: int
    final_profit: float
    trading_days: int
    timeline_df: pd.DataFrame           # day-by-day account/floor/breach tracking


class TopstepCombineSimulator:
    """
    Simulates the Topstep 50k Combine given daily or trade-level PnL.

    Parameters
    ----------
    account_size : float
        Starting account balance. Default 50_000.
    profit_target : float
        Profit needed to pass the combine. Default 3_000.
    max_trailing_drawdown : float
        Maximum allowed drawdown from the trailing high-water mark. Default 2_000.
    max_daily_loss : float
        Maximum loss allowed in a single day. Default 1_000.
    consistency_pct : float
        Maximum fraction of total profit any single day can contribute. Default 0.30.
    min_trading_days : int
        Minimum number of trading days required before passing. Default 5.
    commission_per_rt : float
        Round-trip commission per contract (already included if using actual PnL).
    """

    def __init__(
        self,
        account_size: float = 50_000,
        profit_target: float = 3_000,
        max_trailing_drawdown: float = 2_000,
        max_daily_loss: float = 1_000,
        consistency_pct: float = 0.30,
        min_trading_days: int = 5,
        commission_per_rt: float = 0.0,
    ):
        self.account_size = account_size
        self.profit_target = profit_target
        self.max_trailing_drawdown = max_trailing_drawdown
        self.max_daily_loss = max_daily_loss
        self.consistency_pct = consistency_pct
        self.min_trading_days = min_trading_days
        self.commission_per_rt = commission_per_rt

        # Initial trailing floor = account_size - max_trailing_drawdown
        self._initial_floor = account_size - max_trailing_drawdown

    def simulate(self, daily_pnl: pd.Series) -> CombineResult:
        """
        Run a deterministic combine simulation from a series of daily PnL values.

        Parameters
        ----------
        daily_pnl : pd.Series
            Daily net PnL (positive = profit). Index should be date-like for the
            timeline DataFrame, but plain integers also work.

        Returns
        -------
        CombineResult
        """
        account = self.account_size
        peak_account = self.account_size
        trailing_floor = self._initial_floor

        daily_loss_breaches = 0
        consistency_violations = 0
        cumulative_profit = 0.0
        trading_days = 0
        passed = False
        fail_reason = None
        days_to_pass = None

        records = []

        for day_idx, pnl in enumerate(daily_pnl):
            # --- Daily loss limit check ---
            if pnl < -self.max_daily_loss:
                daily_loss_breaches += 1
                fail_reason = (
                    f"Daily loss limit breached on day {day_idx+1}: "
                    f"${pnl:,.0f} < -${self.max_daily_loss:,.0f}"
                )
                account += pnl  # still apply the loss for record keeping
                records.append({
                    "day": day_idx + 1,
                    "pnl": pnl,
                    "account": account,
                    "peak": peak_account,
                    "floor": trailing_floor,
                    "drawdown": peak_account - account,
                    "daily_breach": True,
                    "trailing_breach": False,
                    "consistency_violation": False,
                    "status": "FAIL_DAILY_LOSS",
                })
                break

            account += pnl
            cumulative_profit += pnl
            trading_days += 1

            # --- Update trailing high-water mark ---
            if account > peak_account:
                peak_account = account
                # Floor trails the peak upward only
                new_floor = peak_account - self.max_trailing_drawdown
                trailing_floor = max(trailing_floor, new_floor)

            drawdown = peak_account - account
            trailing_breach = account < trailing_floor

            # --- Trailing drawdown breach ---
            if trailing_breach:
                fail_reason = (
                    f"Trailing drawdown breached on day {day_idx+1}: "
                    f"account ${account:,.0f} < floor ${trailing_floor:,.0f}"
                )
                records.append({
                    "day": day_idx + 1,
                    "pnl": pnl,
                    "account": account,
                    "peak": peak_account,
                    "floor": trailing_floor,
                    "drawdown": drawdown,
                    "daily_breach": False,
                    "trailing_breach": True,
                    "consistency_violation": False,
                    "status": "FAIL_TRAILING_DD",
                })
                break

            # --- Consistency check ---
            consistency_violation = False
            if cumulative_profit > 0 and pnl > 0:
                if pnl / cumulative_profit > self.consistency_pct:
                    consistency_violations += 1
                    consistency_violation = True

            # --- Pass check ---
            profit_achieved = account - self.account_size
            status = "RUNNING"
            if (
                profit_achieved >= self.profit_target
                and trading_days >= self.min_trading_days
                and not consistency_violation
            ):
                passed = True
                days_to_pass = trading_days
                status = "PASS"
                records.append({
                    "day": day_idx + 1,
                    "pnl": pnl,
                    "account": account,
                    "peak": peak_account,
                    "floor": trailing_floor,
                    "drawdown": drawdown,
                    "daily_breach": False,
                    "trailing_breach": False,
                    "consistency_violation": consistency_violation,
                    "status": status,
                })
                break

            records.append({
                "day": day_idx + 1,
                "pnl": pnl,
                "account": account,
                "peak": peak_account,
                "floor": trailing_floor,
                "drawdown": drawdown,
                "daily_breach": False,
                "trailing_breach": False,
                "consistency_violation": consistency_violation,
                "status": status,
            })

        if not passed and fail_reason is None:
            fail_reason = "Ran out of trading days without hitting profit target"

        timeline_df = pd.DataFrame(records)
        max_drawdown_reached = timeline_df["drawdown"].max() if not timeline_df.empty else 0.0
        min_account = timeline_df["account"].min() if not timeline_df.empty else account

        return CombineResult(
            passed=passed,
            days_to_pass=days_to_pass,
            fail_reason=fail_reason if not passed else None,
            peak_account=peak_account,
            min_account=float(min_account),
            max_drawdown_reached=float(max_drawdown_reached),
            daily_loss_breaches=daily_loss_breaches,
            consistency_violations=consistency_violations,
            final_profit=account - self.account_size,
            trading_days=trading_days,
            timeline_df=timeline_df,
        )

    def monte_carlo(
        self,
        trade_pnl_list: List[float],
        n_paths: int = 10_000,
        trades_per_day_range: tuple = (2, 8),
        max_days: int = 60,
        seed: int = 42,
    ) -> dict:
        """
        Bootstrap trade sequences to estimate P(pass) and median days to pass.

        Parameters
        ----------
        trade_pnl_list : list of float
            Individual trade PnL values (net of commissions).
        n_paths : int
            Number of Monte Carlo paths. Default 10,000.
        trades_per_day_range : tuple (min, max)
            Uniformly sample number of trades per day from this range.
        max_days : int
            Maximum number of simulated trading days per path. Default 60.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            p_pass            - probability of passing
            median_days       - median days to pass (among passes)
            p95_days          - 95th percentile of days to pass (among passes)
            p_fail_daily      - fraction of paths failing daily loss limit
            p_fail_trailing   - fraction of paths failing trailing drawdown
            p_consistency     - fraction of paths with consistency violations
            mean_final_profit - mean final profit across all paths
            p95_max_drawdown  - 95th percentile of max drawdown seen
            n_paths           - number of paths simulated
            paths_passed      - number of paths that passed
        """
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        if not trade_pnl_list:
            return {"error": "Empty trade_pnl_list", "p_pass": 0.0}

        trade_arr = np.array(trade_pnl_list, dtype=float)

        results_passed = []
        results_days = []
        results_max_dd = []
        results_fail_daily = 0
        results_fail_trailing = 0
        results_consistency = 0
        results_final_profit = []

        for _ in range(n_paths):
            # Build daily PnL by sampling trades
            daily_pnls = []
            for _ in range(max_days):
                n_trades = rng.randint(*trades_per_day_range)
                indices = np_rng.integers(0, len(trade_arr), size=n_trades)
                day_pnl = float(trade_arr[indices].sum())
                daily_pnls.append(day_pnl)

            result = self.simulate(pd.Series(daily_pnls))
            results_passed.append(result.passed)
            results_max_dd.append(result.max_drawdown_reached)
            results_final_profit.append(result.final_profit)

            if result.passed:
                results_days.append(result.days_to_pass)
            if result.daily_loss_breaches > 0:
                results_fail_daily += 1
            if "trailing" in (result.fail_reason or "").lower():
                results_fail_trailing += 1
            if result.consistency_violations > 0:
                results_consistency += 1

        p_pass = sum(results_passed) / n_paths
        paths_passed = sum(results_passed)

        return {
            "p_pass": round(p_pass, 4),
            "median_days": int(np.median(results_days)) if results_days else None,
            "p25_days": int(np.percentile(results_days, 25)) if results_days else None,
            "p75_days": int(np.percentile(results_days, 75)) if results_days else None,
            "p95_days": int(np.percentile(results_days, 95)) if results_days else None,
            "p_fail_daily": round(results_fail_daily / n_paths, 4),
            "p_fail_trailing": round(results_fail_trailing / n_paths, 4),
            "p_consistency_issues": round(results_consistency / n_paths, 4),
            "mean_final_profit": round(float(np.mean(results_final_profit)), 2),
            "p95_max_drawdown": round(float(np.percentile(results_max_dd, 95)), 2),
            "n_paths": n_paths,
            "paths_passed": paths_passed,
        }


def simulate_from_trades(
    trades_df: pd.DataFrame,
    simulator: TopstepCombineSimulator,
    date_col: str = "exit_time",
    pnl_col: str = "pnl",
) -> CombineResult:
    """
    Convenience function: aggregate trade-level PnL to daily and simulate.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must have `date_col` (datetime) and `pnl_col` (float) columns.
    simulator : TopstepCombineSimulator
    date_col : str
        Column with trade exit timestamps.
    pnl_col : str
        Column with per-trade net PnL.

    Returns
    -------
    CombineResult
    """
    df = trades_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date
    daily = df.groupby("date")[pnl_col].sum()
    return simulator.simulate(daily)


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = TopstepCombineSimulator()

    # Scenario 1: Steadily profitable
    good_days = pd.Series([250, 200, 300, 150, 350, 200, 250, 400, 300, 200])
    r1 = sim.simulate(good_days)
    print(f"Scenario 1 (steady profit): passed={r1.passed}, days={r1.days_to_pass}, "
          f"profit=${r1.final_profit:.0f}, max_dd=${r1.max_drawdown_reached:.0f}")

    # Scenario 2: Bad day early
    bad_early = pd.Series([-1200, 300, 400, 350, 500])
    r2 = sim.simulate(bad_early)
    print(f"Scenario 2 (bad day early): passed={r2.passed}, fail={r2.fail_reason}")

    # Scenario 3: Trailing drawdown breach
    dd_breach = pd.Series([500, 300, -800, -600, -400, -300])
    r3 = sim.simulate(dd_breach)
    print(f"Scenario 3 (drawdown breach): passed={r3.passed}, fail={r3.fail_reason}")

    # Monte Carlo
    import numpy as np
    np.random.seed(42)
    sample_trades = list(np.random.normal(loc=50, scale=150, size=500))
    mc = sim.monte_carlo(sample_trades, n_paths=5000, seed=42)
    print(f"\nMonte Carlo (5000 paths, mean=$50/trade, std=$150):")
    print(f"  P(pass)        = {mc['p_pass']:.1%}")
    print(f"  Median days    = {mc['median_days']}")
    print(f"  P(fail_daily)  = {mc['p_fail_daily']:.1%}")
    print(f"  P(fail_trail)  = {mc['p_fail_trailing']:.1%}")
    print(f"  P95 max_dd     = ${mc['p95_max_drawdown']:.0f}")
