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
        try:
            self.profit_target_usd = float(top.get("profit_target_usd", 0) or 0.0)
        except (TypeError, ValueError):
            self.profit_target_usd = 0.0
        # Combine-style: once cumulative profit reaches target, block new entries (backtest + live RiskManager).
        self.profit_target_met = False

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

        forced = risk_cfg.get("forced_flatten", {}) or {}
        self.flatten_at_loss_threshold = float(
            forced.get("flatten_at_loss_threshold", 0.0) or 0.0
        )
        self.flatten_at_drawdown_threshold = float(
            forced.get("flatten_at_drawdown_threshold", 0.0) or 0.0
        )

        intraday = risk_cfg.get("intraday_controls", {})
        self.max_trades_per_day = int(intraday.get("max_trades_per_day", 10**9))
        self.min_seconds_between_trades = int(
            intraday.get("min_seconds_between_trades", 0)
        )
        self.max_consecutive_losses = int(
            intraday.get("max_consecutive_losses", 10**9)
        )
        self.daily_profit_lock_usd = float(intraday.get("daily_profit_lock_usd", 0.0) or 0.0)
        self.halt_minutes = int(intraday.get("halt_minutes", 0) or 0)

        self.current_day = None
        self.daily_pnl = 0.0
        self.halted_today = False
        self.cooldown_until_ts = None
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_exit_ts = None

    def apply_broker_snapshot(
        self,
        *,
        equity: float,
        daily_pnl: float,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """
        Overwrite equity and session daily P&L from the broker API (TopstepX).

        Use this in live mode so risk gates and dashboards match the combine account,
        including manual trades outside the bot.
        """
        if timestamp is not None:
            self._maybe_reset_day(timestamp)
        self.equity = float(equity)
        self.daily_pnl = float(daily_pnl)
        self.hwm = max(self.hwm, self.equity)
        self._refresh_profit_target()

    def _refresh_profit_target(self) -> None:
        if self.profit_target_usd <= 0:
            return
        if (self.equity - self.starting_balance) >= self.profit_target_usd:
            self.profit_target_met = True

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
            self.cooldown_until_ts = None
            self.trades_today = 0
            self.consecutive_losses = 0

    def _maybe_clear_cooldown(self, now_ts: pd.Timestamp) -> None:
        if self.cooldown_until_ts is None:
            return
        now_ts = self._normalize_ts(now_ts)
        cooldown_until = self._normalize_ts(self.cooldown_until_ts)
        if cooldown_until is None:
            self.cooldown_until_ts = None
            return
        if now_ts >= cooldown_until:
            self.cooldown_until_ts = None

    def _start_cooldown(self, entry_ts: pd.Timestamp) -> None:
        if not self.halt_minutes:
            return
        entry_ts = self._normalize_ts(entry_ts)
        if entry_ts is None:
            return
        self.cooldown_until_ts = entry_ts + pd.Timedelta(minutes=self.halt_minutes)

    def can_trade(self, entry_ts: pd.Timestamp) -> tuple[bool, str]:
        entry_ts = self._normalize_ts(entry_ts)
        if entry_ts is None:
            return False, "invalid_timestamp"
        self._maybe_reset_day(entry_ts)
        self._maybe_clear_cooldown(entry_ts)
        self._refresh_profit_target()
        if self.profit_target_met:
            return False, "profit_target"
        if self.cooldown_until_ts is not None:
            return False, "cooldown"
        if self.halted_today:
            return False, "halted"
        if self.daily_profit_lock_usd and self.daily_pnl >= self.daily_profit_lock_usd:
            self.halted_today = True
            return False, "daily_profit_lock"

        # Pre-breach circuit breakers: once we are deep in the red, stop opening
        # new trades for the session to avoid spiraling beyond Topstep limits.
        if (
            self.daily_enabled
            and self.flatten_at_loss_threshold
            and 0 < self.flatten_at_loss_threshold < 1.0
            and self.max_daily_loss > 0
            and self.daily_pnl <= -self.max_daily_loss * self.flatten_at_loss_threshold
        ):
            if self.halt_minutes:
                self._start_cooldown(entry_ts)
                return False, "risk_daily_loss_soft"
            self.halted_today = True
            return False, "risk_daily_loss_soft"

        if (
            self.dd_enabled
            and self.flatten_at_drawdown_threshold
            and 0 < self.flatten_at_drawdown_threshold < 1.0
            and self.max_drawdown > 0
            and (self.hwm - self.equity)
            >= self.max_drawdown * self.flatten_at_drawdown_threshold
        ):
            if self.halt_minutes:
                self._start_cooldown(entry_ts)
                return False, "risk_drawdown_soft"
            self.halted_today = True
            return False, "risk_drawdown_soft"

        if self.trades_today >= self.max_trades_per_day:
            return False, "max_trades"
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.halt_minutes:
                # Cooldown, then allow trading to resume (reset streak breaker).
                self._start_cooldown(entry_ts)
                self.consecutive_losses = 0
                return False, "consecutive_losses"
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
        self,
        timestamp: pd.Timestamp,
        equity_unrealized: float,
        equity_best: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check for daily loss or drawdown breach using unrealized equity.
        """
        timestamp = self._normalize_ts(timestamp)
        if timestamp is None:
            return False, ""
        self._maybe_reset_day(timestamp)

        # Real-time HWM updates make trailing DD conservative (Topstep-like).
        # If both best- and worst-case equity are available (e.g., intrabar high/low),
        # update HWM with best-case and check breaches against worst-case.
        if self.hwm_update_policy == "real_time":
            try:
                update_val = (
                    float(equity_best)
                    if equity_best is not None
                    else float(equity_unrealized)
                )
                self.hwm = max(self.hwm, update_val)
            except Exception:
                pass

        if self.daily_enabled:
            if self.risk_cfg.get("daily_loss_limit", {}).get(
                "pnl_calculation", "realized_and_unrealized"
            ) == "realized_and_unrealized":
                daily_unrealized = self.daily_pnl + (equity_unrealized - self.equity)
            else:
                daily_unrealized = self.daily_pnl

            # Pre-breach flattening to reduce overshoot risk on coarser bars.
            if (
                self.flatten_at_loss_threshold
                and 0 < self.flatten_at_loss_threshold < 1.0
                and self.max_daily_loss > 0
                and daily_unrealized <= -self.max_daily_loss * self.flatten_at_loss_threshold
            ):
                return True, "risk_daily_loss_soft"

            if daily_unrealized <= -self.max_daily_loss:
                self.halted_today = True
                return True, "risk_daily_loss"

        if self.dd_enabled:
            if self.dd_pnl_calc == "realized_and_unrealized":
                dd_equity = equity_unrealized
            else:
                dd_equity = self.equity

            dd = self.hwm - dd_equity
            if (
                self.flatten_at_drawdown_threshold
                and 0 < self.flatten_at_drawdown_threshold < 1.0
                and self.max_drawdown > 0
                and dd >= self.max_drawdown * self.flatten_at_drawdown_threshold
            ):
                return True, "risk_drawdown_soft"
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
        if (
            self.daily_enabled
            and self.flatten_at_loss_threshold
            and 0 < self.flatten_at_loss_threshold < 1.0
            and self.max_daily_loss > 0
            and self.daily_pnl <= -self.max_daily_loss * self.flatten_at_loss_threshold
        ):
            self.halted_today = True
        if self.dd_enabled and (self.hwm - self.equity) >= self.max_drawdown:
            self.halted_today = True
        if (
            self.dd_enabled
            and self.flatten_at_drawdown_threshold
            and 0 < self.flatten_at_drawdown_threshold < 1.0
            and self.max_drawdown > 0
            and (self.hwm - self.equity)
            >= self.max_drawdown * self.flatten_at_drawdown_threshold
        ):
            self.halted_today = True
        if self.daily_profit_lock_usd and self.daily_pnl >= self.daily_profit_lock_usd:
            self.halted_today = True
        self._refresh_profit_target()
