"""
Alert system for live trading events.

Sends notifications for important events like trades, risk breaches, errors.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertManager:
    """
    Manages alerts for trading events.

    Can send alerts via:
    - Logging (always enabled)
    - File (always enabled)
    - Email (optional, future)
    - Slack (optional, future)
    - SMS (optional, future)
    """

    def __init__(self, output_dir: Path, enable_sound: bool = True):
        """
        Initialize alert manager.

        Args:
            output_dir: Directory to save alert logs
            enable_sound: Whether to play sound on critical alerts (macOS/Linux)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.enable_sound = enable_sound
        self.alert_log_path = self.output_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Alert counts
        self.info_count = 0
        self.warning_count = 0
        self.critical_count = 0

        logger.info(f"AlertManager initialized: {self.alert_log_path}")

    def alert(self, level: AlertLevel, title: str, message: str, data: Optional[dict] = None):
        """
        Send an alert.

        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            data: Optional additional data
        """
        # Update counts
        if level == AlertLevel.INFO:
            self.info_count += 1
        elif level == AlertLevel.WARNING:
            self.warning_count += 1
        elif level == AlertLevel.CRITICAL:
            self.critical_count += 1

        # Format alert
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_msg = f"[{level.value}] {title}: {message}"

        # Log to standard logger
        if level == AlertLevel.INFO:
            logger.info(alert_msg)
        elif level == AlertLevel.WARNING:
            logger.warning(alert_msg)
        elif level == AlertLevel.CRITICAL:
            logger.critical(alert_msg)

        # Write to alert log file
        with open(self.alert_log_path, 'a') as f:
            f.write(f"{timestamp} {alert_msg}\n")
            if data:
                f.write(f"  Data: {data}\n")

        # Play sound for critical alerts (macOS)
        if level == AlertLevel.CRITICAL and self.enable_sound:
            try:
                import subprocess
                subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'], check=False)
            except:
                pass  # Ignore errors if sound can't play

    def trade_executed(self, direction: str, contracts: int, price: float, prediction_score: float):
        """Alert for trade execution."""
        self.alert(
            AlertLevel.INFO,
            "Trade Executed",
            f"{direction} {contracts} @ {price:.2f}",
            data={'direction': direction, 'contracts': contracts, 'price': price, 'score': prediction_score}
        )

    def trade_closed(self, direction: str, pnl: float, exit_reason: str):
        """Alert for trade closure."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        self.alert(
            level,
            "Trade Closed",
            f"{direction} closed, PnL=${pnl:,.2f} ({exit_reason})",
            data={'pnl': pnl, 'exit_reason': exit_reason}
        )

    def risk_breach(self, breach_type: str, current_value: float, limit_value: float):
        """Alert for risk limit breach."""
        self.alert(
            AlertLevel.CRITICAL,
            "RISK BREACH",
            f"{breach_type}: ${current_value:,.2f} exceeds limit ${limit_value:,.2f}",
            data={'type': breach_type, 'current': current_value, 'limit': limit_value}
        )

    def daily_loss_warning(self, daily_pnl: float, daily_limit: float):
        """Alert for approaching daily loss limit."""
        pct_used = abs(daily_pnl) / daily_limit * 100
        self.alert(
            AlertLevel.WARNING,
            "Daily Loss Warning",
            f"Daily P&L ${daily_pnl:,.2f} is {pct_used:.1f}% of limit (${daily_limit:,.2f})",
            data={'daily_pnl': daily_pnl, 'daily_limit': daily_limit, 'pct_used': pct_used}
        )

    def drawdown_warning(self, drawdown: float, dd_limit: float):
        """Alert for approaching drawdown limit."""
        pct_used = drawdown / dd_limit * 100
        self.alert(
            AlertLevel.WARNING,
            "Drawdown Warning",
            f"Drawdown ${drawdown:,.2f} is {pct_used:.1f}% of limit (${dd_limit:,.2f})",
            data={'drawdown': drawdown, 'limit': dd_limit, 'pct_used': pct_used}
        )

    def connection_lost(self, connection_type: str):
        """Alert for lost connection."""
        self.alert(
            AlertLevel.CRITICAL,
            "Connection Lost",
            f"{connection_type} connection lost",
            data={'connection': connection_type}
        )

    def connection_restored(self, connection_type: str):
        """Alert for restored connection."""
        self.alert(
            AlertLevel.INFO,
            "Connection Restored",
            f"{connection_type} connection restored",
            data={'connection': connection_type}
        )

    def error_occurred(self, error_type: str, error_message: str):
        """Alert for errors."""
        self.alert(
            AlertLevel.CRITICAL,
            "Error Occurred",
            f"{error_type}: {error_message}",
            data={'type': error_type, 'message': error_message}
        )

    def session_started(self, account_id: str, starting_equity: float):
        """Alert for trading session start."""
        self.alert(
            AlertLevel.INFO,
            "Trading Session Started",
            f"Account {account_id}, Starting Equity: ${starting_equity:,.2f}",
            data={'account_id': account_id, 'starting_equity': starting_equity}
        )

    def session_ended(self, total_trades: int, total_pnl: float, final_equity: float):
        """Alert for trading session end."""
        self.alert(
            AlertLevel.INFO,
            "Trading Session Ended",
            f"Trades: {total_trades}, P&L: ${total_pnl:,.2f}, Final Equity: ${final_equity:,.2f}",
            data={'total_trades': total_trades, 'total_pnl': total_pnl, 'final_equity': final_equity}
        )

    def performance_milestone(self, milestone: str, value: float):
        """Alert for performance milestones."""
        self.alert(
            AlertLevel.INFO,
            "Performance Milestone",
            f"{milestone}: ${value:,.2f}",
            data={'milestone': milestone, 'value': value}
        )

    def get_summary(self) -> str:
        """Get alert summary."""
        total = self.info_count + self.warning_count + self.critical_count
        return (
            f"Alerts: {total} total "
            f"(INFO: {self.info_count}, WARNING: {self.warning_count}, CRITICAL: {self.critical_count})"
        )
