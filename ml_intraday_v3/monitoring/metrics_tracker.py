"""
Real-time performance metrics tracker.

Tracks and stores key trading metrics for monitoring and analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks real-time trading performance metrics.

    Stores metrics in memory and periodically saves to disk.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.metrics_history: List[Dict] = []
        self.trade_history: List[Dict] = []

        # Current session metrics
        self.session_start = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_win = 0.0
        self.max_loss = 0.0
        self.current_streak = 0
        self.max_winning_streak = 0
        self.max_losing_streak = 0

        # Risk metrics
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0

        # Position tracking
        self.open_positions = 0
        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_rejected = 0

        logger.info(f"MetricsTracker initialized: {output_dir}")

    def set_starting_equity(self, equity: float):
        """Set starting equity for the session."""
        self.starting_equity = equity
        self.current_equity = equity
        self.peak_equity = equity

    def update_equity(self, equity: float, daily_pnl: float):
        """
        Update current equity and risk metrics.

        Args:
            equity: Current account equity
            daily_pnl: Current daily P&L
        """
        self.current_equity = equity
        self.daily_pnl = daily_pnl

        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = self.peak_equity - equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def record_signal(self, executed: bool):
        """
        Record a signal generation.

        Args:
            executed: Whether signal was executed or rejected
        """
        self.signals_generated += 1
        if executed:
            self.signals_executed += 1
        else:
            self.signals_rejected += 1

    def record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        contracts: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
    ):
        """
        Record a completed trade.

        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            direction: LONG or SHORT
            contracts: Number of contracts
            entry_price: Entry price
            exit_price: Exit price
            pnl: Trade P&L in USD
            exit_reason: Reason for exit (stop/target/flatten)
        """
        # Update counts
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
            self.max_win = max(self.max_win, pnl)

            # Update streak
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_winning_streak = max(self.max_winning_streak, self.current_streak)

        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
            self.max_loss = min(self.max_loss, pnl)

            # Update streak
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_losing_streak = max(self.max_losing_streak, abs(self.current_streak))

        self.total_pnl += pnl

        # Store trade
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'duration_minutes': (exit_time - entry_time).total_seconds() / 60,
        }
        self.trade_history.append(trade)

        logger.debug(f"Trade recorded: {direction} {contracts} @ {entry_price:.2f}, PnL=${pnl:.2f}")

    def update_positions(self, open_positions: int):
        """Update count of open positions."""
        self.open_positions = open_positions

    def snapshot(self) -> Dict:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary of current metrics
        """
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        avg_win = (self.gross_profit / self.winning_trades) if self.winning_trades > 0 else 0.0
        avg_loss = (self.gross_loss / self.losing_trades) if self.losing_trades > 0 else 0.0
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else float('inf')
        avg_trade = (self.total_pnl / self.total_trades) if self.total_trades > 0 else 0.0

        return_pct = ((self.current_equity - self.starting_equity) / self.starting_equity * 100) if self.starting_equity > 0 else 0.0
        dd_pct = (self.max_drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0.0

        snapshot = {
            'timestamp': datetime.now(),
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60,

            # Trade stats
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,

            # P&L
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'profit_factor': profit_factor,
            'max_win': self.max_win,
            'max_loss': self.max_loss,

            # Equity
            'starting_equity': self.starting_equity,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'return_pct': return_pct,

            # Risk
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': dd_pct,

            # Streaks
            'current_streak': self.current_streak,
            'max_winning_streak': self.max_winning_streak,
            'max_losing_streak': self.max_losing_streak,

            # Positions
            'open_positions': self.open_positions,

            # Signals
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'signals_rejected': self.signals_rejected,
            'execution_rate': (self.signals_executed / self.signals_generated * 100) if self.signals_generated > 0 else 0.0,
        }

        return snapshot

    def save_snapshot(self):
        """Save current metrics snapshot to disk."""
        snapshot = self.snapshot()

        # Append to metrics history
        self.metrics_history.append(snapshot)

        # Save to CSV
        metrics_path = self.output_dir / f"metrics_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.metrics_history).to_csv(metrics_path, index=False)

        logger.debug(f"Metrics snapshot saved: {metrics_path}")

    def save_trades(self):
        """Save trade history to disk."""
        if not self.trade_history:
            return

        trades_path = self.output_dir / f"trades_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.trade_history).to_csv(trades_path, index=False)

        logger.info(f"Trade history saved: {trades_path}")

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for display.

        Returns:
            Dictionary of summary stats
        """
        snapshot = self.snapshot()

        return {
            'Total Trades': snapshot['total_trades'],
            'Win Rate': f"{snapshot['win_rate']:.1f}%",
            'Total P&L': f"${snapshot['total_pnl']:,.2f}",
            'Avg Trade': f"${snapshot['avg_trade']:,.2f}",
            'Profit Factor': f"{snapshot['profit_factor']:.2f}" if snapshot['profit_factor'] != float('inf') else 'INF',
            'Max Drawdown': f"${snapshot['max_drawdown']:,.2f} ({snapshot['max_drawdown_pct']:.1f}%)",
            'Current Equity': f"${snapshot['current_equity']:,.2f}",
            'Return': f"{snapshot['return_pct']:+.2f}%",
            'Open Positions': snapshot['open_positions'],
            'Execution Rate': f"{snapshot['execution_rate']:.1f}%",
        }
