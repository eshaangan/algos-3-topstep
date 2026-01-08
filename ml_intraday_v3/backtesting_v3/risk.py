"""
Risk gate enforcement for offline backtest.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _session_day(timestamp: pd.Timestamp, reset_time: str, tz: str) -> pd.Timestamp:
    ts = timestamp
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_local = ts.tz_convert(tz)
    reset_ts = pd.Timestamp(f"{ts_local.date()} {reset_time}", tz=tz)
    if ts_local < reset_ts:
        reset_ts = reset_ts - pd.Timedelta(days=1)
    return reset_ts.date()


class RiskManager:
    def __init__(self, risk_cfg: dict):
        self.risk_cfg = risk_cfg
        top = risk_cfg.get("topstep", {})
        self.starting_balance = float(top.get("starting_balance", 0.0))
        self.equity = self.starting_balance
        self.hwm = self.starting_balance

        daily_cfg = risk_cfg.get("daily_loss_limit", {})
        self.daily_enabled = bool(daily_cfg.get("enabled", False))
        self.max_daily_loss = float(daily_cfg.get("max_daily_loss", 0.0))
        self.reset_time = daily_cfg.get("reset_time", "17:00")
        self.reset_tz = daily_cfg.get("reset_timezone", "America/Chicago")
        self.daily_breach_action = daily_cfg.get("breach_action", "halt_trading")

        dd_cfg = risk_cfg.get("trailing_drawdown", {})
        self.dd_enabled = bool(dd_cfg.get("enabled", False))
        self.max_drawdown = float(dd_cfg.get("max_drawdown", 0.0))
        self.dd_pnl_calc = dd_cfg.get(
            "pnl_calculation", "realized_and_unrealized"
        )
        self.hwm_update_policy = dd_cfg.get("hwm_update_policy", "end_of_day")
        self.dd_breach_action = dd_cfg.get("breach_action", "halt_trading")

        intraday = risk_cfg.get("intraday_controls", {})
        self.max_trades_per_day = int(intraday.get("max_trades_per_day", 10**9))
        self.min_seconds_between_trades = int(
            intraday.get("min_seconds_between_trades", 0)
        )
        self.max_consecutive_losses = int(
            intraday.get("max_consecutive_losses", 10**9)
        )

        self.current_day = None
        self.daily_pnl = 0.0
        self.halted_today = False
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_exit_ts = None

    @staticmethod
    def _normalize_ts(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None:
            return None
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _maybe_reset_day(self, timestamp: pd.Timestamp):
        timestamp = self._normalize_ts(timestamp)
        if timestamp is None:
            return
        day = _session_day(timestamp, self.reset_time, self.reset_tz)
        if self.current_day != day:
            if self.hwm_update_policy == "end_of_day":
                self.hwm = max(self.hwm, self.equity)
            self.current_day = day
            self.daily_pnl = 0.0
            self.halted_today = False
            self.trades_today = 0
            self.consecutive_losses = 0

    def can_trade(self, entry_ts: pd.Timestamp) -> tuple[bool, str]:
        entry_ts = self._normalize_ts(entry_ts)
        if entry_ts is None:
            return False, "invalid_timestamp"
        self._maybe_reset_day(entry_ts)
        if self.halted_today:
            return False, "halted"
        if self.trades_today >= self.max_trades_per_day:
            return False, "max_trades"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, "consecutive_losses"
        # Only check min_seconds_between_trades if it's explicitly set (> 0)
        # This allows concurrent positions when min_seconds_between_trades = 0
        if self.min_seconds_between_trades > 0 and self.last_trade_exit_ts is not None:
            last_exit = self._normalize_ts(self.last_trade_exit_ts)
            delta = (entry_ts - last_exit).total_seconds()
            if delta < self.min_seconds_between_trades:
                return False, "min_time"
        if self.daily_enabled and self.daily_pnl <= -self.max_daily_loss:
            self.halted_today = True
            return False, "risk_daily_loss"
        if self.dd_enabled and (self.hwm - self.equity) >= self.max_drawdown:
            self.halted_today = True
            return False, "risk_drawdown"
        return True, ""

    def check_breach(
        self, timestamp: pd.Timestamp, equity_unrealized: float
    ) -> tuple[bool, str]:
        """
        Check for daily loss or drawdown breach using unrealized equity.
        """
        timestamp = self._normalize_ts(timestamp)
        if timestamp is None:
            return False, ""
        self._maybe_reset_day(timestamp)

        if self.daily_enabled:
            if self.risk_cfg.get("daily_loss_limit", {}).get(
                "pnl_calculation", "realized_and_unrealized"
            ) == "realized_and_unrealized":
                daily_unrealized = self.daily_pnl + (equity_unrealized - self.equity)
            else:
                daily_unrealized = self.daily_pnl
            if daily_unrealized <= -self.max_daily_loss:
                self.halted_today = True
                return True, "risk_daily_loss"

        if self.dd_enabled:
            if self.dd_pnl_calc == "realized_and_unrealized":
                dd_equity = equity_unrealized
            else:
                dd_equity = self.equity
            if (self.hwm - dd_equity) >= self.max_drawdown:
                self.halted_today = True
                return True, "risk_drawdown"

        return False, ""

    def record_trade(self, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, pnl_usd: float):
        entry_ts = self._normalize_ts(entry_ts)
        exit_ts = self._normalize_ts(exit_ts)
        if entry_ts is None or exit_ts is None:
            return
        self._maybe_reset_day(exit_ts)

        # Enforce equity floor at zero (prevent liquidation beyond capital)
        new_equity = self.equity + pnl_usd

        if new_equity < 0:
            # Cap loss at available equity
            actual_pnl = -self.equity
            self.equity = 0.0
            logger.warning(
                f"Equity floor enforced at {exit_ts}: "
                f"requested PnL ${pnl_usd:.2f} capped to ${actual_pnl:.2f}. "
                f"ACCOUNT LIQUIDATED."
            )
            self.halted_today = True  # Stop trading when liquidated
            # Still record the capped PnL for daily tracking
            self.daily_pnl += actual_pnl
        else:
            self.equity = new_equity
            self.daily_pnl += pnl_usd

        self.trades_today += 1
        self.last_trade_exit_ts = exit_ts

        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.hwm_update_policy == "real_time":
            self.hwm = max(self.hwm, self.equity)

        # Halt trading if liquidated
        if self.equity <= 0:
            logger.error(f"Account liquidated at {exit_ts}. Trading halted.")
            self.halted_today = True

        if self.daily_enabled and self.daily_pnl <= -self.max_daily_loss:
            self.halted_today = True
        if self.dd_enabled and (self.hwm - self.equity) >= self.max_drawdown:
            self.halted_today = True
