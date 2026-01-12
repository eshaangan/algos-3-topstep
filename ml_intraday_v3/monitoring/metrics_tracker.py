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

        # Signal tracking
        self.signal_history: List[Dict] = []

        # Kelly sizing tracking
        self.kelly_sizing_log: List[Dict] = []

        logger.info(f"MetricsTracker initialized: {output_dir}")

    def load_trades_from_csv(self, csv_path: str) -> int:
        """
        Load trade history from a previous session's CSV file.

        This enables Kelly Criterion to continue from previous session's statistics
        rather than resetting to learning phase on restart.

        Args:
            csv_path: Path to trades CSV file from previous session

        Returns:
            Number of trades loaded

        Side effects:
            - Populates self.trade_history
            - Recalculates all metrics (total_trades, win_rate, pnl, etc.)
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            logger.warning(f"Trade history file not found: {csv_path}")
            return 0

        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning(f"Trade history file is empty: {csv_path}")
                return 0

            # Verify this is a completed trades file (not a signal log)
            required_columns = ['entry_time', 'exit_time', 'pnl']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(
                    f"Skipping {csv_path.name}: wrong format (missing columns: {missing_columns}). "
                    f"This appears to be a signal log, not a completed trades file."
                )
                return 0

            # Filter out trades with missing P&L (incomplete trades)
            df = df.dropna(subset=['pnl'])
            
            if df.empty:
                logger.warning(f"Trade history file has no completed trades: {csv_path}")
                return 0

            # Convert timestamp strings back to datetime objects
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'])

            # Convert to list of dicts
            trades = df.to_dict('records')
            
            # Populate trade history
            self.trade_history = trades

            # Recalculate all metrics from loaded trades
            self._recalculate_metrics_from_history()

            logger.info(f"✓ Loaded {len(trades)} trades from {csv_path}")
            logger.info(f"  Total P&L: ${self.total_pnl:.2f}, Win Rate: {self.winning_trades}/{self.total_trades} ({self.winning_trades/self.total_trades*100:.1f}%)")

            return len(trades)

        except Exception as e:
            logger.error(f"Error loading trade history from {csv_path}: {e}")
            return 0

    def _recalculate_metrics_from_history(self):
        """
        Recalculate all metrics from self.trade_history.

        Called after loading trades from CSV to restore session state.
        """
        # Reset metrics
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

        # Recalculate from history
        for trade in self.trade_history:
            pnl = trade.get('pnl', 0.0)
            
            self.total_trades += 1
            self.total_pnl += pnl

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

        logger.debug(f"Metrics recalculated from {len(self.trade_history)} trades")

    def find_latest_trade_file(self, log_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Find the most recent COMPLETED TRADES CSV file in the logs directory.

        Filters out signal log files (which also have trades_*.csv pattern but wrong format).
        Only returns files with the correct format: entry_time, exit_time, pnl columns.

        Args:
            log_dir: Directory to search (defaults to self.output_dir)

        Returns:
            Path to most recent trades file, or None if not found
        """
        search_dir = Path(log_dir) if log_dir else self.output_dir
        
        if not search_dir.exists():
            logger.warning(f"Log directory not found: {search_dir}")
            return None

        # Find all trades_*.csv files
        trade_files = list(search_dir.glob("trades_*.csv"))
        
        if not trade_files:
            logger.info(f"No previous trade history files found in {search_dir}")
            return None

        # Sort by modification time (most recent first)
        trade_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Find first file with correct format (completed trades, not signal log)
        for trade_file in trade_files:
            try:
                # Quick check: read just the header
                df = pd.read_csv(trade_file, nrows=0)
                required_columns = ['entry_time', 'exit_time', 'pnl']
                
                if all(col in df.columns for col in required_columns):
                    logger.info(f"Found latest trade file: {trade_file.name}")
                    return trade_file
                else:
                    logger.debug(f"Skipping {trade_file.name}: wrong format (signal log)")
                    
            except Exception as e:
                logger.debug(f"Error checking {trade_file.name}: {e}")
                continue
        
        logger.warning(f"No valid completed trades files found in {search_dir}")
        return None

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

    def record_signal(
        self,
        executed: bool,
        score: float,
        timestamp: Optional[pd.Timestamp] = None,
        direction: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """
        Record a signal generation.

        Args:
            executed: Whether signal was executed or rejected
            score: Prediction score (score_ev)
            timestamp: Signal timestamp
            direction: Signal direction (LONG/SHORT) if executed
            reason: Rejection reason if not executed
        """
        self.signals_generated += 1
        if executed:
            self.signals_executed += 1
        else:
            self.signals_rejected += 1

        # Log to signal history (for CSV export)
        signal_record = {
            'timestamp': timestamp or datetime.now(),
            'score': score,
            'executed': executed,
            'direction': direction,
            'reason': reason,
        }
        
        # Add to signal log if it exists
        if not hasattr(self, 'signal_history'):
            self.signal_history = []
        self.signal_history.append(signal_record)

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

    def save_signal_log(self):
        """Save signal history to disk."""
        if not self.signal_history:
            return

        signal_path = self.output_dir / f"signals_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.signal_history).to_csv(signal_path, index=False)
        logger.debug(f"Signal log saved: {signal_path}")

    def record_kelly_decision(self, sizing_decision: Dict):
        """Record Kelly sizing decision for post-analysis."""
        self.kelly_sizing_log.append(sizing_decision)

    def save_kelly_log(self):
        """Save Kelly sizing log to CSV."""
        if not self.kelly_sizing_log:
            return

        kelly_path = self.output_dir / f"kelly_sizing_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.kelly_sizing_log).to_csv(kelly_path, index=False)
        logger.debug(f"Kelly sizing log saved: {kelly_path}")

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
