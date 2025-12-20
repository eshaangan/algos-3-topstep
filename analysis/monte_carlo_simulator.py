"""
Monte Carlo simulator for backtest analysis.

Bootstraps trade outcomes to generate distribution of possible equity curves,
drawdowns, and performance metrics.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for trade outcomes."""

    def __init__(self, trades: List[Dict[str, Any]], initial_equity: float, seed: int = 42):
        """
        Initialize simulator with historical trades.
        
        Args:
            trades: List of trade dictionaries, each with at least a 'pnl' key
            initial_equity: Starting equity for simulations
            seed: Random seed for reproducibility
        """
        self.trades = trades
        self.initial_equity = initial_equity
        self.rng = np.random.default_rng(seed)
        
        # Extract P&L values
        self.pnl_values = np.array([t.get("pnl", 0.0) for t in trades])
        self.n_trades = len(self.pnl_values)

    def run(self, runs: int = 500, sample_size: int | None = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            runs: Number of simulation trials
            sample_size: Number of trades to sample per trial (defaults to n_trades)
            
        Returns:
            Dictionary with summary statistics and raw results
        """
        if self.n_trades == 0:
            return {
                "runs": 0,
                "sample_size": 0,
                "summary": {
                    "ending_equity": {"p05": self.initial_equity, "p50": self.initial_equity, "p95": self.initial_equity},
                    "max_drawdown": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                    "win_rate": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                    "profit_factor": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                    "net_pnl": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                }
            }
        
        sample_size = sample_size or self.n_trades
        
        ending_equities = []
        max_drawdowns = []
        win_rates = []
        profit_factors = []
        net_pnls = []
        
        for _ in range(runs):
            # Sample trades with replacement
            sampled_pnl = self.rng.choice(self.pnl_values, size=sample_size, replace=True)
            
            # Build equity curve
            equity_curve = self.initial_equity + np.cumsum(sampled_pnl)
            
            # Calculate metrics
            ending_equity = equity_curve[-1]
            ending_equities.append(ending_equity)
            
            # Max drawdown
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = peaks - equity_curve
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            max_drawdowns.append(max_dd)
            
            # Win rate
            wins = (sampled_pnl > 0).sum()
            win_rate = wins / len(sampled_pnl) if len(sampled_pnl) > 0 else 0.0
            win_rates.append(win_rate)
            
            # Profit factor
            gross_win = sampled_pnl[sampled_pnl > 0].sum()
            gross_loss = abs(sampled_pnl[sampled_pnl < 0].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0)
            profit_factors.append(pf if pf != float('inf') else 1000.0)  # Cap inf for stats
            
            # Net P&L
            net_pnl = float(sampled_pnl.sum())
            net_pnls.append(net_pnl)
        
        # Calculate percentiles
        def percentiles(values: List[float]) -> Dict[str, float]:
            arr = np.array(values)
            return {
                "p05": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        
        return {
            "runs": runs,
            "sample_size": sample_size,
            "summary": {
                "ending_equity": percentiles(ending_equities),
                "max_drawdown": percentiles(max_drawdowns),
                "win_rate": percentiles(win_rates),
                "profit_factor": percentiles(profit_factors),
                "net_pnl": percentiles(net_pnls),
            },
            "raw_results": {
                "ending_equities": ending_equities,
                "max_drawdowns": max_drawdowns,
                "win_rates": win_rates,
                "profit_factors": profit_factors,
                "net_pnls": net_pnls,
            }
        }

