"""
Terminal-based live trading dashboard.

Displays real-time performance metrics in the terminal.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TerminalDashboard:
    """
    Terminal-based dashboard for live trading metrics.

    Displays key metrics in a formatted terminal output.
    """

    def __init__(self):
        """Initialize dashboard."""
        self.last_update = None
        logger.info("TerminalDashboard initialized")

    def render(self, metrics: Dict, alerts_summary: Optional[str] = None):
        """
        Render dashboard to terminal.

        Args:
            metrics: Dictionary of current metrics
            alerts_summary: Optional alerts summary string
        """
        self.last_update = datetime.now()

        # Clear screen (simple version - works on most terminals)
        print("\033[2J\033[H")  # ANSI escape codes

        # Header
        print("=" * 80)
        print(f"{'LIVE TRADING DASHBOARD':^80}")
        print(f"{'Last Update: ' + self.last_update.strftime('%Y-%m-%d %H:%M:%S'):^80}")
        print("=" * 80)
        print()

        # Account Status
        print("┌─ ACCOUNT STATUS " + "─" * 62 + "┐")
        print(f"│ Equity: ${metrics.get('current_equity', 0):>16,.2f}   │   Starting: ${metrics.get('starting_equity', 0):>16,.2f}   │")

        return_pct = metrics.get('return_pct', 0)
        return_color = self._get_color(return_pct)
        print(f"│ Return: {return_color}{return_pct:>+15.2f}%\033[0m   │   Peak: ${metrics.get('peak_equity', 0):>20,.2f}   │")

        daily_pnl = metrics.get('daily_pnl', 0)
        daily_color = self._get_color(daily_pnl)
        print(f"│ Daily P&L: {daily_color}${daily_pnl:>14,.2f}\033[0m   │   Open Positions: {metrics.get('open_positions', 0):>16}   │")
        if metrics.get('broker_balance') is not None:
            bb = metrics['broker_balance']
            rp = metrics.get('broker_realized_pnl', 0.0)
            up = metrics.get('broker_open_pnl', 0.0)
            rp_c = self._get_color(rp)
            up_c = self._get_color(up)
            print(f"│ Topstep Balance: ${bb:>12,.2f}   │   RP&L: {rp_c}${rp:>12,.2f}\033[0m   │   UP&L: {up_c}${up:>12,.2f}\033[0m   │")
        print("└" + "─" * 78 + "┘")
        print()

        # Trading Performance
        print("┌─ TRADING PERFORMANCE " + "─" * 56 + "┐")

        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        total_pnl = metrics.get('total_pnl', 0)
        pnl_color = self._get_color(total_pnl)

        print(f"│ Total Trades: {total_trades:>10}   │   Winning: {metrics.get('winning_trades', 0):>12}   │   Losing: {metrics.get('losing_trades', 0):>12}   │")
        print(f"│ Win Rate: {win_rate:>14.1f}%   │   Profit Factor: {metrics.get('profit_factor', 0):>10.2f}   │")
        print(f"│ Total P&L: {pnl_color}${total_pnl:>13,.2f}\033[0m   │   Avg Trade: ${metrics.get('avg_trade', 0):>15,.2f}   │")
        print(f"│ Max Win: ${metrics.get('max_win', 0):>15,.2f}   │   Max Loss: ${metrics.get('max_loss', 0):>16,.2f}   │")
        print("└" + "─" * 78 + "┘")
        print()

        # Risk Metrics
        print("┌─ RISK METRICS " + "─" * 63 + "┐")

        max_dd = metrics.get('max_drawdown', 0)
        max_dd_pct = metrics.get('max_drawdown_pct', 0)
        dd_color = self._get_warning_color(max_dd, 2000)  # Warning if > $2000

        print(f"│ Max Drawdown: {dd_color}${max_dd:>11,.2f} ({max_dd_pct:>5.1f}%)\033[0m   │   Daily Loss Limit: $2,000   │")

        daily_limit_pct = (abs(daily_pnl) / 2000 * 100) if daily_pnl < 0 else 0
        daily_limit_color = self._get_warning_color(abs(daily_pnl), 2000)

        print(f"│ Daily Limit Used: {daily_limit_color}{daily_limit_pct:>9.1f}%\033[0m   │   DD Limit: $2,500             │")

        streak = metrics.get('current_streak', 0)
        streak_symbol = "W" if streak > 0 else "L" if streak < 0 else "-"
        print(f"│ Current Streak: {abs(streak):>7} {streak_symbol}   │   Max Win Streak: {metrics.get('max_winning_streak', 0):>10}   │")
        print("└" + "─" * 78 + "┘")
        print()

        # Signals
        print("┌─ SIGNAL STATISTICS " + "─" * 58 + "┐")

        signals_gen = metrics.get('signals_generated', 0)
        signals_exec = metrics.get('signals_executed', 0)
        signals_rej = metrics.get('signals_rejected', 0)
        exec_rate = metrics.get('execution_rate', 0)

        print(f"│ Signals Generated: {signals_gen:>8}   │   Executed: {signals_exec:>13}   │   Rejected: {signals_rej:>11}   │")
        print(f"│ Execution Rate: {exec_rate:>11.1f}%   │")
        print("└" + "─" * 78 + "┘")
        print()

        # Alerts Summary
        if alerts_summary:
            print("┌─ ALERTS " + "─" * 69 + "┐")
            print(f"│ {alerts_summary:<76} │")
            print("└" + "─" * 78 + "┘")
            print()

        # Session Info
        session_duration = metrics.get('session_duration_minutes', 0)
        hours = int(session_duration // 60)
        minutes = int(session_duration % 60)
        print(f"Session Duration: {hours:02d}h {minutes:02d}m")
        print()
        print("Press Ctrl+C to stop trading")
        print("=" * 80)

    def _get_color(self, value: float) -> str:
        """Get color code based on value sign."""
        if value > 0:
            return "\033[32m"  # Green
        elif value < 0:
            return "\033[31m"  # Red
        else:
            return "\033[37m"  # White

    def _get_warning_color(self, value: float, threshold: float) -> str:
        """Get warning color based on threshold."""
        pct = (value / threshold * 100) if threshold > 0 else 0

        if pct >= 90:
            return "\033[31m"  # Red (critical)
        elif pct >= 75:
            return "\033[33m"  # Yellow (warning)
        else:
            return "\033[32m"  # Green (OK)

    def display_startup_status(self, checks: Dict[str, bool]):
        """
        Display startup check status.

        Args:
            checks: Dictionary of check names and results
        """
        print()
        print("=" * 80)
        print("STARTUP CHECKS")
        print("=" * 80)

        for check_name, passed in checks.items():
            status = "\033[32m✓ PASS\033[0m" if passed else "\033[31m✗ FAIL\033[0m"
            print(f"  {check_name}: {status}")

        all_passed = all(checks.values())
        print()
        if all_passed:
            print("\033[32m✓ All startup checks passed\033[0m")
        else:
            print("\033[31m✗ Some startup checks failed\033[0m")
        print("=" * 80)
        print()

    def display_summary(self, metrics: Dict):
        """
        Display end-of-session summary.

        Args:
            metrics: Final metrics dictionary
        """
        print()
        print("=" * 80)
        print(f"{'SESSION SUMMARY':^80}")
        print("=" * 80)
        print()

        total_pnl = metrics.get('total_pnl', 0)
        pnl_color = self._get_color(total_pnl)

        print(f"Total Trades:      {metrics.get('total_trades', 0):>10}")
        print(f"Winning Trades:    {metrics.get('winning_trades', 0):>10}")
        print(f"Losing Trades:     {metrics.get('losing_trades', 0):>10}")
        print(f"Win Rate:          {metrics.get('win_rate', 0):>9.1f}%")
        print()
        print(f"Total P&L:         {pnl_color}${total_pnl:>13,.2f}\033[0m")
        print(f"Gross Profit:      ${metrics.get('gross_profit', 0):>13,.2f}")
        print(f"Gross Loss:        ${metrics.get('gross_loss', 0):>13,.2f}")
        print(f"Profit Factor:     {metrics.get('profit_factor', 0):>13.2f}")
        print(f"Average Trade:     ${metrics.get('avg_trade', 0):>13,.2f}")
        print()
        print(f"Max Win:           ${metrics.get('max_win', 0):>13,.2f}")
        print(f"Max Loss:          ${metrics.get('max_loss', 0):>13,.2f}")
        print(f"Max Drawdown:      ${metrics.get('max_drawdown', 0):>13,.2f} ({metrics.get('max_drawdown_pct', 0):.1f}%)")
        print()
        print(f"Final Equity:      ${metrics.get('current_equity', 0):>13,.2f}")
        print(f"Return:            {metrics.get('return_pct', 0):>+12.2f}%")
        print()
        print("=" * 80)
