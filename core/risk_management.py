"""
All TopstepX risk enforcement logic lives in this module.

Responsibilities:
- Enforce daily loss limits and trailing drawdown.
- Restrict trading hours and ensure the account is flat before session end.
- Translate strategy signals into position sizes that honour dollar risk limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from math import floor
from typing import List, Optional
from zoneinfo import ZoneInfo

from config import RiskConfig, SessionConfig
from models import RiskState, Signal, SignalAction, Trade

LOGGER = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Result of evaluating a signal."""

    allowed: bool
    reason: str = ""
    contracts: int = 0


class RiskManager:
    """Stateful risk engine that mirrors TopstepX prop-firm guardrails."""

    def __init__(self, risk_cfg: RiskConfig, session_cfg: SessionConfig) -> None:
        self.cfg = risk_cfg
        self.session = session_cfg
        self.state = RiskState(
            trading_locked=False,
            realized_daily_pnl=0.0,
            trailing_peak=risk_cfg.starting_balance,
            cumulative_pnl=0.0,
            open_positions=0,
        )
        self.daily_locked = False
        self.permanently_locked = False
        self.current_session_day: Optional[datetime.date] = None
        self.session_start_equity: Optional[float] = self.cfg.starting_balance
        # Live-mode state (used when syncing from a real account).
        self.use_live_state: bool = getattr(self.cfg, "use_live_account_state", False)
        self.live_equity: Optional[float] = None
        self.live_open_pnl: float = 0.0
        self.trades_today: int = 0
        self._session_timezone = ZoneInfo(self.session.timezone)
        self._research_allow_resets = getattr(self.cfg, "research_allow_risk_resets", False)
        # Adaptive sizing state - track recent trade results (True=win, False=loss)
        self.recent_trade_results: List[bool] = []
        # Profit target tracking
        self._profit_target_logged: bool = False
    def _current_equity(self) -> float:
        if self.use_live_state and self.live_equity is not None:
            return self.live_equity
        return self.cfg.starting_balance + self.state.cumulative_pnl

    def _reject(self, reason: str, rule: str, signal: Optional[Signal] = None) -> RiskAssessment:
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Risk rejection rule=%s reason=%s timestamp=%s daily_pnl=%.2f trailing_peak=%.2f equity=%.2f trades_today=%d",
                rule,
                reason,
                getattr(signal, "timestamp", None),
                self._effective_daily_pnl(),
                self.state.trailing_peak,
                self._current_equity(),
                self.trades_today,
            )
        return RiskAssessment(False, reason)

    # --------------------------------------------------------------------- utils
    def _update_session_day(self, timestamp: datetime) -> None:
        """Reset daily counters when a new session starts."""

        if self.current_session_day != timestamp.date():
            self.current_session_day = timestamp.date()
            self.state.realized_daily_pnl = 0.0
            self.daily_locked = False
            # Reset per-day trade counter so max_trades_per_day applies per session.
            self.trades_today = 0
            # Reset profit target logged flag for new session
            self._profit_target_logged = False
            if self.use_live_state and self.live_equity is not None:
                self.session_start_equity = self.live_equity
            else:
                self.session_start_equity = self.cfg.starting_balance

    def _within_trading_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within the allowed session window."""

        session_start = _parse_time(self.session.session_start)
        session_end = _parse_time(self.session.session_end)
        flat_guard = minutes_to_time(
            time_to_minutes(session_end) - self.session.flat_buffer_minutes
        )
        now_time = timestamp.time()
        return session_start <= now_time < flat_guard

    def sync_from_live(self, account_equity: float, open_pnl: float, realized_pnl: Optional[float] = None) -> None:
        """
        Update internal state from a live ProjectX/TopstepX account snapshot.

        Parameters
        ----------
        account_equity:
            Current account equity including open PnL, as reported by the broker.
        open_pnl:
            Current unrealised PnL on all open positions.
        """

        if not self.use_live_state:
            return
        now_local = datetime.now(self._session_timezone)
        self._update_session_day(now_local)

        self.live_equity = float(account_equity)
        self.live_open_pnl = float(open_pnl)
        # Keep trailing peak aligned with the best observed live equity.
        self.state.trailing_peak = max(self.state.trailing_peak, self.live_equity)
        self.state.cumulative_pnl = self.live_equity - self.cfg.starting_balance

        start_equity = self.session_start_equity or self.cfg.starting_balance
        realized_today = (self.live_equity - start_equity) - self.live_open_pnl
        self.state.realized_daily_pnl = realized_today

    def _effective_daily_pnl(self) -> float:
        """
        Daily PnL used for limit checks.

        In live mode we approximate this as realised + open PnL; otherwise we
        fall back to the realised-only measure used in backtests.
        """

        if self.use_live_state and self.live_equity is not None:
            return self.state.realized_daily_pnl + self.live_open_pnl
        return self.state.realized_daily_pnl

    def record_trade_result(self, trade: Trade) -> None:
        """
        Record whether a trade was profitable for adaptive sizing.

        Parameters
        ----------
        trade:
            The completed trade to record (win if pnl > 0, loss otherwise).
        """

        is_win = trade.pnl > 0
        self.recent_trade_results.append(is_win)

        # Keep only the most recent N trades
        lookback = getattr(self.cfg, "recent_trades_lookback", 10)
        if len(self.recent_trade_results) > lookback:
            self.recent_trade_results = self.recent_trade_results[-lookback:]

        if LOGGER.isEnabledFor(logging.DEBUG):
            wins = sum(self.recent_trade_results)
            losses = len(self.recent_trade_results) - wins
            LOGGER.debug(
                "Trade result recorded: %s (PnL=%.2f). Recent record: %d wins, %d losses",
                "WIN" if is_win else "LOSS",
                trade.pnl,
                wins,
                losses,
            )

    def _calculate_adaptive_multiplier(self) -> float:
        """
        Calculate position size multiplier based on recent trade results.

        Returns a multiplier between 0.0 and max configured value based on
        current winning/losing streaks.
        """

        if not getattr(self.cfg, "adaptive_sizing_enabled", False):
            return 1.0

        if not self.recent_trade_results:
            return 1.0

        # Calculate current streak from the end of the list
        current_streak = 0
        is_winning_streak = self.recent_trade_results[-1]

        for result in reversed(self.recent_trade_results):
            if result == is_winning_streak:
                current_streak += 1
            else:
                break

        # Apply losing streak multiplier
        losing_threshold = getattr(self.cfg, "losing_streak_threshold", 3)
        if not is_winning_streak and current_streak >= losing_threshold:
            multiplier = getattr(self.cfg, "size_multiplier_on_losing_streak", 0.5)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Adaptive sizing: losing streak of %d (threshold=%d), multiplier=%.2f",
                    current_streak,
                    losing_threshold,
                    multiplier,
                )
            return max(0.0, multiplier)

        # Apply winning streak multiplier (if enabled)
        winning_threshold = getattr(self.cfg, "winning_streak_threshold", 0)
        if is_winning_streak and winning_threshold > 0 and current_streak >= winning_threshold:
            multiplier = getattr(self.cfg, "size_multiplier_on_winning_streak", 1.0)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Adaptive sizing: winning streak of %d (threshold=%d), multiplier=%.2f",
                    current_streak,
                    winning_threshold,
                    multiplier,
                )
            return multiplier

        return 1.0

    # ---------------------------------------------------------------- assessments
    def assess_signal(self, signal: Signal) -> RiskAssessment:
        """Return whether a strategy signal can be acted upon."""

        self._update_session_day(signal.timestamp)
        if self.permanently_locked:
            return self._reject("Trailing drawdown breached.", "TRAILING_LOCK", signal)
        if self.daily_locked:
            return self._reject("Daily loss limit breached.", "DAILY_LOCK", signal)
        if not self._within_trading_hours(signal.timestamp):
            return self._reject("Outside trading session.", "SESSION_HOURS", signal)
        # CRITICAL FIX: Check total contracts in use, not just position count
        # open_positions now tracks TOTAL CONTRACTS, not number of positions
        if self.state.open_positions >= self.cfg.max_contracts:
            return self._reject(
                f"Max contracts in use: {self.state.open_positions}/{self.cfg.max_contracts}",
                "CONTRACT_LIMIT",
                signal
            )
        if self.state.trading_locked:
            return self._reject("Trading locked.", "MANUAL_LOCK", signal)

        # Optional hard cap on trades per day.
        max_trades = getattr(self.cfg, "max_trades_per_day", 0)
        if max_trades and self.trades_today >= max_trades:
            return self._reject("Max trades per day reached.", "TRADES_PER_DAY", signal)

        # Determine effective per-trade risk budget with optional strategy-specific overrides.
        risk_budget = self.cfg.fixed_risk_per_trade
        strategy_mode = signal.metadata.get("strategy_mode")
        scalping_override = getattr(self.cfg, "scalping_risk_per_trade", None)
        if strategy_mode == "scalping_v1" and scalping_override:
            risk_budget = scalping_override
        if (
            getattr(self.cfg, "drawdown_risk_scale_enabled", False)
            and self.state.cumulative_pnl <= -getattr(self.cfg, "drawdown_risk_scale_dd", 0.0)
        ):
            multiplier = getattr(self.cfg, "drawdown_risk_scale_multiplier", 1.0)
            risk_budget *= max(0.0, multiplier)

        if getattr(self.cfg, "atr_risk_scale_enabled", False):
            atr_ticks = signal.metadata.get("atr_ticks")
            reference = getattr(self.cfg, "atr_reference_ticks", 0.0)
            if atr_ticks and atr_ticks > 0 and reference > 0:
                scale = reference / atr_ticks
                floor_scale = getattr(self.cfg, "atr_scale_floor", 0.1)
                cap_scale = getattr(self.cfg, "atr_scale_cap", 2.0)
                scale = max(floor_scale, min(cap_scale, scale))
                risk_budget *= scale

        entry_price = signal.metadata.get("ideal_entry")
        risk_ticks = signal.metadata.get("risk_ticks")
        if risk_ticks is None:
            if entry_price is None:
                return self._reject("Signal missing ideal entry metadata.", "METADATA", signal)
            risk_ticks = abs(signal.stop_price - entry_price) / self.cfg.tick_size
        if risk_ticks <= 0:
            return self._reject("Invalid stop distance.", "RISK_PARAMS", signal)
        risk_per_contract = risk_ticks * self.cfg.tick_value
        if risk_per_contract <= 0:
            return self._reject("Invalid stop distance.", "RISK_PARAMS", signal)

        # Apply adaptive sizing multiplier based on recent performance
        adaptive_multiplier = self._calculate_adaptive_multiplier()
        adjusted_risk_budget = risk_budget * adaptive_multiplier

        contracts = min(
            self.cfg.max_contracts,
            floor(adjusted_risk_budget / risk_per_contract),
        )

        # Ensure at least 1 contract if the base calculation allows it
        if contracts <= 0 and floor(risk_budget / risk_per_contract) > 0:
            contracts = 1

        if contracts <= 0:
            return self._reject("Risk per contract exceeds allocation.", "RISK_SIZING", signal)

        # CRITICAL FIX: Check if adding these contracts would exceed max_contracts
        if self.state.open_positions + contracts > self.cfg.max_contracts:
            contracts = max(0, self.cfg.max_contracts - self.state.open_positions)
            if contracts <= 0:
                return self._reject(
                    f"Adding {contracts} would exceed max contracts ({self.state.open_positions} in use)",
                    "CONTRACT_LIMIT",
                    signal
                )

        # Ensure taking this trade cannot breach the daily loss limit if stop is hit.
        daily_pnl = self._effective_daily_pnl()
        projected_daily_pnl = daily_pnl - (contracts * risk_per_contract)
        if projected_daily_pnl <= -self.cfg.max_daily_loss:
            return self._reject(
                "Trade would breach daily loss limit if stopped out.",
                "DAILY_LOSS_LIMIT",
                signal,
            )
        return RiskAssessment(True, contracts=contracts)

    # ---------------------------------------------------------------- state hooks
    def register_position_open(self, contracts: int) -> None:
        """Track total contracts in use for enforcement of max concurrency."""
        # CRITICAL FIX: Track actual contract count, not just position count
        self.state.open_positions = min(
            self.state.open_positions + contracts,
            self.cfg.max_contracts
        )
        LOGGER.info(
            "Position opened: +%d contracts (total in use: %d/%d)",
            contracts,
            self.state.open_positions,
            self.cfg.max_contracts
        )

    def register_position_close(self, contracts: int) -> None:
        """Reduce contract count when trade closes."""
        # CRITICAL FIX: Subtract actual contract count, not just 1
        self.state.open_positions = max(0, self.state.open_positions - contracts)
        LOGGER.info(
            "Position closed: -%d contracts (total in use: %d/%d)",
            contracts,
            self.state.open_positions,
            self.cfg.max_contracts
        )

    def update_after_trade(self, trade: Trade) -> None:
        """Update realized metrics post trade and lock trading if breaches occur."""

        self._update_session_day(trade.exit_time)
        self.state.realized_daily_pnl += trade.pnl
        self.state.cumulative_pnl += trade.pnl
        # In live mode we prefer the broker-reported equity if available.
        if self.use_live_state and self.live_equity is not None:
            equity = self.live_equity
        else:
            equity = self.cfg.starting_balance + self.state.cumulative_pnl
        self.state.trailing_peak = max(self.state.trailing_peak, equity)
        self.trades_today += 1

        if self._effective_daily_pnl() <= -self.cfg.max_daily_loss:
            self.daily_locked = True
            LOGGER.debug("Daily loss lock engaged (PnL=%.2f).", self._effective_daily_pnl())

        if equity <= self.state.trailing_peak - self.cfg.trailing_drawdown:
            self.permanently_locked = True
            LOGGER.debug(
                "Trailing drawdown lock engaged (equity=%.2f peak=%.2f limit=%.2f).",
                equity,
                self.state.trailing_peak,
                self.cfg.trailing_drawdown,
            )

    def reset_for_research(self) -> None:
        """
        Reset drawdown and daily loss locks for research/backtest scenarios.

        This should never be enabled in live trading; it exists to let
        long-running historical sweeps continue collecting statistics after a
        simulated breach.
        """

        if not self._research_allow_resets:
            raise RuntimeError(
                "Risk reset attempted without enabling risk_cfg.research_allow_risk_resets."
            )
        LOGGER.warning("Risk state reset for research purposes.")
        self.state = RiskState(
            trading_locked=False,
            realized_daily_pnl=0.0,
            trailing_peak=self.cfg.starting_balance,
            cumulative_pnl=0.0,
            open_positions=0,
        )
        self.daily_locked = False
        self.permanently_locked = False
        self.trades_today = 0
        self.state.trailing_peak = self.cfg.starting_balance

    def should_flatten_now(self, timestamp: datetime) -> bool:
        """Determine if positions must be liquidated because session is ending or profit target hit."""

        # Check if daily profit target has been reached
        if self.cfg.daily_profit_target is not None:
            daily_pnl = self._effective_daily_pnl()
            if daily_pnl >= self.cfg.daily_profit_target:
                if not hasattr(self, '_profit_target_logged') or not self._profit_target_logged:
                    LOGGER.info(
                        "Daily profit target of $%.2f reached (current: $%.2f). Flattening all positions.",
                        self.cfg.daily_profit_target,
                        daily_pnl
                    )
                    self._profit_target_logged = True
                return True

        # Check if session is ending
        session_end = _parse_time(self.session.session_end)
        flat_guard = minutes_to_time(
            time_to_minutes(session_end) - self.session.flat_buffer_minutes
        )
        return timestamp.time() >= flat_guard


# --------------------------------------------------------------------------- util
def _parse_time(value: str) -> time:
    """Convert HH:MM string into datetime.time."""

    hour, minute = map(int, value.split(":"))
    return time(hour=hour, minute=minute)


def time_to_minutes(value: time) -> int:
    """Return minutes since midnight."""

    return value.hour * 60 + value.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight back into time while clamping at 0."""

    minutes = max(0, minutes)
    return time(hour=minutes // 60, minute=minutes % 60)
