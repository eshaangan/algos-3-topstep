"""
Live risk manager for Topstep circuit breakers.

Uses risk.yaml -> risk_management for intraday gating, sizing, and thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Risk config not found: {self.config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f) or {}

        rm_cfg = self.config.get("risk_management", {}) or {}
        top_cfg = self.config.get("topstep", {}) or {}
        daily_cfg = self.config.get("daily_loss_limit", {}) or {}

        self.enabled = bool(rm_cfg.get("enabled", False))

        self.daily_loss_limit_usd = float(rm_cfg.get("daily_loss_limit_usd", 0.0))
        self.daily_loss_warning_usd = float(rm_cfg.get("daily_loss_warning_usd", 0.0))
        self.daily_loss_critical_usd = float(rm_cfg.get("daily_loss_critical_usd", 0.0))

        self.base_position_size = int(rm_cfg.get("base_position_size", 1))
        self.reduce_size_on_loss = bool(rm_cfg.get("reduce_size_on_loss", False))
        self.size_after_warning = float(rm_cfg.get("size_after_warning", 1.0))
        self.size_after_critical = float(rm_cfg.get("size_after_critical", 0.0))

        self.max_drawdown_from_hwm = float(rm_cfg.get("max_drawdown_from_hwm", 0.0))
        self.drawdown_warning = float(rm_cfg.get("drawdown_warning", 0.0))

        self.avoid_first_hour = bool(rm_cfg.get("avoid_first_hour", False))
        self.avoid_last_hour = bool(rm_cfg.get("avoid_last_hour", False))

        self.threshold_increase_on_loss = float(rm_cfg.get("threshold_increase_on_loss", 0.0))
        self.daily_profit_lock_usd = float(rm_cfg.get("daily_profit_lock_usd", 0.0) or 0.0)
        self.halt_cooldown_minutes = int(rm_cfg.get("halt_cooldown_minutes", 0) or 0)
        self.halt_until_utc: datetime | None = None
        self.halt_reason: str | None = None

        sigma_cfg = rm_cfg.get("sigma_position_sizing", {}) or {}
        self.sigma_sizing_enabled = bool(sigma_cfg.get("enabled", False))
        self.sigma_thresholds = sigma_cfg.get("sigma_thresholds", []) or []

        self.starting_balance = float(top_cfg.get("starting_balance", 0.0)) or 50_000.0
        self.equity = self.starting_balance
        self.high_water_mark = self.starting_balance

        self.daily_pnl = 0.0
        self.current_session_date = None

        self.reset_time = self._parse_time(daily_cfg.get("reset_time", "17:00"))
        self.reset_tz = ZoneInfo(daily_cfg.get("reset_timezone", "America/Chicago"))
        self.trade_tz = ZoneInfo("America/New_York")

        if self.enabled:
            logger.info(
                "RiskManager enabled: daily_limit=$%.0f drawdown=$%.0f",
                self.daily_loss_limit_usd,
                self.max_drawdown_from_hwm,
            )

    @staticmethod
    def _parse_time(value: str) -> dt_time:
        try:
            return dt_time.fromisoformat(value)
        except ValueError:
            logger.warning("Invalid reset_time '%s', defaulting to 17:00", value)
            return dt_time(17, 0)

    def _session_date(self, now: datetime) -> datetime.date:
        now_local = now.astimezone(self.reset_tz)
        reset_dt = datetime.combine(now_local.date(), self.reset_time, tzinfo=self.reset_tz)
        if now_local < reset_dt:
            reset_dt -= timedelta(days=1)
        return reset_dt.date()

    def _maybe_reset_daily(self, now: datetime) -> None:
        session_date = self._session_date(now)
        if self.current_session_date != session_date:
            self.current_session_date = session_date
            self.daily_pnl = 0.0
            self.high_water_mark = max(self.high_water_mark, self.equity)
            self.halt_until_utc = None
            self.halt_reason = None
            logger.info("RiskManager daily reset: session_date=%s", session_date)

    def _maybe_clear_halt(self, now: datetime) -> None:
        if self.halt_until_utc is None:
            return
        if now >= self.halt_until_utc:
            self.halt_until_utc = None
            self.halt_reason = None

    def _start_halt(self, now: datetime, reason: str) -> None:
        if not self.halt_cooldown_minutes:
            return
        if self.halt_until_utc is not None and now < self.halt_until_utc:
            return
        self.halt_reason = reason
        self.halt_until_utc = now + timedelta(minutes=self.halt_cooldown_minutes)

    @staticmethod
    def _within_window(now_time: dt_time, start: dt_time, end: dt_time) -> bool:
        if start <= end:
            return start <= now_time < end
        return now_time >= start or now_time < end

    def _check_time_filters(self, now: datetime) -> tuple[bool, str]:
        if not (self.avoid_first_hour or self.avoid_last_hour):
            return False, ""

        now_et = now.astimezone(self.trade_tz).time()

        if self.avoid_first_hour and self._within_window(now_et, dt_time(17, 0), dt_time(18, 0)):
            return True, "avoid_first_hour"

        if self.avoid_last_hour and self._within_window(now_et, dt_time(15, 0), dt_time(16, 0)):
            return True, "avoid_last_hour"

        return False, ""

    def sync_equity(self, equity: float) -> None:
        if equity is None:
            return
        self.equity = float(equity)
        self.high_water_mark = max(self.high_water_mark, self.equity)

    def sync_state(self, equity: float | None, daily_pnl: float | None) -> None:
        if not self.enabled:
            return
        now = datetime.now(tz=ZoneInfo("UTC"))
        self._maybe_reset_daily(now)
        if equity is not None:
            self.equity = float(equity)
            self.high_water_mark = max(self.high_water_mark, self.equity)
        if daily_pnl is not None:
            self.daily_pnl = float(daily_pnl)

    def update_pnl(self, trade_pnl: float) -> None:
        now = datetime.now(tz=ZoneInfo("UTC"))
        self._maybe_reset_daily(now)

        self.daily_pnl += float(trade_pnl)
        self.equity += float(trade_pnl)
        self.high_water_mark = max(self.high_water_mark, self.equity)

    def check_trading_allowed(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "risk_management_disabled"

        now = datetime.now(tz=ZoneInfo("UTC"))
        self._maybe_reset_daily(now)
        self._maybe_clear_halt(now)
        if self.halt_until_utc is not None:
            reason = self.halt_reason or "cooldown"
            return False, f"cooldown_{reason}"

        blocked, reason = self._check_time_filters(now)
        if blocked:
            return False, reason

        drawdown = self.high_water_mark - self.equity
        if self.max_drawdown_from_hwm and drawdown >= self.max_drawdown_from_hwm:
            return False, "trailing_drawdown_limit"

        if self.daily_loss_limit_usd and self.daily_pnl <= -self.daily_loss_limit_usd:
            return False, "daily_loss_limit"

        if self.daily_profit_lock_usd and self.daily_pnl >= self.daily_profit_lock_usd:
            return False, "daily_profit_lock"

        if self.daily_loss_critical_usd and self.daily_pnl <= -self.daily_loss_critical_usd:
            if self.halt_cooldown_minutes:
                self._start_halt(now, "daily_loss_critical")
                return False, "cooldown_daily_loss_critical"
            return False, "daily_loss_critical"

        if self.drawdown_warning and drawdown >= self.drawdown_warning:
            logger.warning("RiskManager drawdown warning: %.2f", drawdown)

        return True, "ok"

    def get_position_size(self, base_size: int, sigma: float | None = None) -> int:
        if not self.enabled:
            return int(base_size)

        base = int(base_size)

        # Sigma-based sizing (optional).
        sized = base
        if self.sigma_sizing_enabled:
            if sigma is None:
                # Fail safe: if sigma isn't available, do not size up.
                sized = min(sized, 1)
            else:
                try:
                    sigma_val = float(sigma)
                except Exception:
                    sigma_val = None
                if sigma_val is not None:
                    for rule in self.sigma_thresholds:
                        try:
                            max_sigma = float(rule.get("max_sigma"))
                            contracts = int(rule.get("contracts"))
                        except Exception:
                            continue
                        if sigma_val <= max_sigma:
                            sized = contracts
                            break

        # Loss-based throttles apply after sigma sizing.
        if self.reduce_size_on_loss:
            if self.daily_loss_critical_usd and self.daily_pnl <= -self.daily_loss_critical_usd:
                sized = max(0, int(round(sized * self.size_after_critical)))

            if self.daily_loss_warning_usd and self.daily_pnl <= -self.daily_loss_warning_usd:
                scaled = int(round(sized * self.size_after_warning))
                if scaled <= 0 and self.size_after_warning > 0 and sized > 0:
                    sized = 1
                else:
                    sized = max(0, scaled)

        return int(max(0, sized))

    def get_threshold_adjustment(self) -> float:
        if not self.enabled:
            return 0.0
        if self.daily_pnl < 0:
            return float(self.threshold_increase_on_loss)
        return 0.0

    def reset_daily(self) -> None:
        now = datetime.now(tz=ZoneInfo("UTC"))
        self.current_session_date = self._session_date(now)
        self.daily_pnl = 0.0
        self.high_water_mark = max(self.high_water_mark, self.equity)
