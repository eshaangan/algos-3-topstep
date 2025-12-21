"""
Live execution engine that routes orders to ProjectX for a TopstepX account.

This module mirrors the intent of `execution.py` but delegates all order
handling to `ProjectXClient` instead of simulating fills from historical bars.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional
from zoneinfo import ZoneInfo

from config import RiskConfig, StrategyConfig
from models import Signal, SignalAction
from projectx_client import BracketInstruction, OrderSnapshot, OrderState, ProjectXClient
from risk_management import RiskManager

LOGGER = logging.getLogger("live_runner")


@dataclass
class LiveExecutionResult:
    """Outcome of attempting to route a live signal."""

    order: Optional[OrderState]
    rejection_reason: Optional[str] = None


class LiveExecutionEngine:
    """
    Live execution abstraction using the ProjectX API.

    This engine does not maintain a full local `Position` object; instead it
    assumes the broker (ProjectX/TopstepX) is the source of truth for open
    positions and order state.
    """

    def __init__(
        self,
        client: ProjectXClient,
        risk_manager: RiskManager,
        risk_cfg: RiskConfig,
        strategy_cfg: StrategyConfig,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.risk_cfg = risk_cfg
        self.strategy_cfg = strategy_cfg
        self._tracked_orders: Dict[str, LiveOrderTracker] = {}

    def handle_signal(self, signal: Signal) -> LiveExecutionResult:
        """
        Assess and, if allowed, route a strategy signal to ProjectX.

        The intent is identical to `ExecutionEngine.process_entry`, but fills
        and lifecycle are broker-managed. Exits (stops/targets) should be
        implemented as server-side brackets where possible.
        """

        # Time guard: block new entries after the daily cutoff.
        now_local = datetime.now(ZoneInfo(self.risk_manager.session.timezone))
        if self._past_cutoff(now_local):
            LOGGER.warning("Live signal blocked: past cutoff time; reason=%s", signal.reason)
            return LiveExecutionResult(order=None, rejection_reason="PAST_CUTOFF")

        # Hard guard: block if broker net position already exceeds cap.
        try:
            positions = self.client.search_open_positions()
            net_contracts = 0
            for pos in positions:
                if pos.get("contract_id") != self.client._contract_id:  # type: ignore[attr-defined]
                    continue
                size = int(pos.get("size", 0))
                # type: 1 assumed long, otherwise treat as short
                net_contracts += size if int(pos.get("type", 1)) == 1 else -size
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Position check failed; proceeding without broker sync: %s", exc)
            net_contracts = 0

        cap = self.risk_cfg.max_contracts
        if abs(net_contracts) >= cap:
            LOGGER.warning(
                "Live signal blocked: broker net=%d exceeds cap=%d; reason=%s",
                net_contracts,
                cap,
                signal.reason,
            )
            return LiveExecutionResult(order=None, rejection_reason="POSITION_CAP")

        assessment = self.risk_manager.assess_signal(signal)
        if not assessment.allowed:
            LOGGER.warning("Live signal rejected by risk engine: %s", assessment.reason)
            return LiveExecutionResult(order=None, rejection_reason=assessment.reason)

        # Route explicit exit signals directly.
        if signal.action == SignalAction.EXIT:
            return self._handle_exit(signal)

        # Map SignalAction to broker side semantics.
        if signal.action == SignalAction.ENTER_LONG:
            side = "BUY"
        elif signal.action == SignalAction.ENTER_SHORT:
            side = "SELL"
        else:
            return LiveExecutionResult(order=None, rejection_reason="Unsupported signal action.")

        LOGGER.info(
            "Pre-trade open positions (tracked): %d / cap %d",
            self.risk_manager.state.open_positions,
            self.risk_cfg.max_contracts,
        )

        # Build brackets (using signed ticks as required by Topstep)
        bracket_stop = self._build_stop_bracket(signal)
        bracket_target = self._build_target_bracket(signal)
        client_order_id = f"EMA-{uuid.uuid4().hex[:12]}"

        # Log intent (prices for reference; brackets drive execution)
        LOGGER.info(
            "Routing live order: %s %s x%d stop_price=%.2f target_price=%.2f stop_ticks=%s target_ticks=%s",
            side,
            signal.symbol,
            assessment.contracts,
            signal.stop_price or 0.0,
            signal.target_price or 0.0,
            bracket_stop.ticks if bracket_stop else None,
            bracket_target.ticks if bracket_target else None,
        )
        # Try without brackets - they might be causing errorCode 9
        contract_id = getattr(self.client, "_contract_id", None) or getattr(self.risk_cfg, "contract_id", None) or "CON.F.US.MES.Z25"

        order = self.client.place_order(
            symbol=signal.symbol,
            side=side,
            quantity=assessment.contracts,
            order_type="MARKET",
            client_order_id=client_order_id,
            contract_id=contract_id,
            stop_loss_bracket=bracket_stop,
            take_profit_bracket=bracket_target,
        )
        self.risk_manager.register_position_open(assessment.contracts)
        self._tracked_orders[order.order_id] = LiveOrderTracker(
            order_id=order.order_id,
            symbol=signal.symbol,
            side=side,
            contracts=assessment.contracts,
            entry_price=signal.metadata.get("ideal_entry", 0.0),
            opened_at=signal.timestamp,
            client_tag=client_order_id,
        )
        return LiveExecutionResult(order=order)

    def reconcile_open_orders(self, open_orders: Optional[list[OrderSnapshot]] = None) -> None:
        """
        Reconcile tracked orders with broker state and unlock risk when fills complete.

        Parameters
        ----------
        open_orders:
            Optional pre-fetched list from /api/Order/searchOpen to avoid duplicate API calls.
        """

        if not self._tracked_orders:
            return
        broker_orders = open_orders or self.client.search_open_orders()
        live_ids = {str(order.order_id) for order in broker_orders}
        closed_ids = [order_id for order_id in self._tracked_orders if order_id not in live_ids]
        for order_id in closed_ids:
            tracker = self._tracked_orders.pop(order_id, None)
            if not tracker:
                continue
            LOGGER.info("Order %s closed at broker; unlocking %d contracts.", order_id, tracker.contracts)
            self.risk_manager.register_position_close(tracker.contracts)
            # RiskManager will pull realized PnL via sync_from_live, so no manual trade needed.

    def _build_stop_bracket(self, signal: Signal) -> Optional[BracketInstruction]:
        """Return a stop-loss bracket instruction if metadata provides risk ticks."""

        risk_ticks = signal.metadata.get("risk_ticks")
        if not risk_ticks:
            # Fallback to strategy stop ticks so we always place a protective stop.
            risk_ticks = self.strategy_cfg.stop_ticks
        if not risk_ticks:
            return None
        stop_ticks = max(1, int(round(risk_ticks)))
        # Topstep expects signed ticks: negative for longs (stop below), positive for shorts (stop above).
        if signal.action == SignalAction.ENTER_LONG:
            stop_ticks = -abs(stop_ticks)
        else:
            stop_ticks = abs(stop_ticks)
        try:
            return BracketInstruction(ticks=stop_ticks, order_type=self.risk_cfg.bracket_stop_type)
        except ValueError as exc:
            LOGGER.warning("Invalid stop bracket config ignored: %s", exc)
            return None

    def _build_target_bracket(self, signal: Signal) -> Optional[BracketInstruction]:
        """Return a take-profit bracket instruction sized to the configured RR multiple."""

        if signal.metadata.get("execution_mode") == "time_exit":
            return None
        risk_ticks = signal.metadata.get("risk_ticks")
        if not risk_ticks:
            risk_ticks = self.strategy_cfg.stop_ticks
        if not risk_ticks:
            return None
        rr = signal.metadata.get("target_rr_multiple", self.strategy_cfg.target_rr_multiple)
        target_ticks = max(1, int(round(risk_ticks * rr)))
        # Take-profit should be opposite sign of stop: positive for longs (target above), negative for shorts (target below).
        if signal.action == SignalAction.ENTER_LONG:
            target_ticks = abs(target_ticks)
        else:
            target_ticks = -abs(target_ticks)
        try:
            return BracketInstruction(ticks=target_ticks, order_type=self.risk_cfg.bracket_target_type)
        except ValueError as exc:
            LOGGER.warning("Invalid target bracket config ignored: %s", exc)
            return None

    def _handle_exit(self, signal: Signal) -> LiveExecutionResult:
        """Flatten any open position for the configured contract."""

        try:
            positions = self.client.search_open_positions()
            net_contracts = 0
            for pos in positions:
                if pos.get("contract_id") != self.client._contract_id:  # type: ignore[attr-defined]
                    continue
                size = int(pos.get("size", 0))
                net_contracts += size if int(pos.get("type", 1)) == 1 else -size
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Exit position check failed: %s", exc)
            return LiveExecutionResult(order=None, rejection_reason="EXIT_POSITION_CHECK_FAILED")

        if net_contracts == 0:
            return LiveExecutionResult(order=None, rejection_reason="NO_OPEN_POSITION")

        side = "SELL" if net_contracts > 0 else "BUY"
        qty = abs(net_contracts)
        client_order_id = f"EXIT-{uuid.uuid4().hex[:12]}"
        contract_id = getattr(self.client, "_contract_id", None) or getattr(self.risk_cfg, "contract_id", None)

        LOGGER.info(
            "Routing exit order: %s %s x%d reason=%s",
            side,
            signal.symbol,
            qty,
            signal.reason,
        )

        order = self.client.place_order(
            symbol=signal.symbol,
            side=side,
            quantity=qty,
            order_type="MARKET",
            client_order_id=client_order_id,
            contract_id=contract_id,
        )
        self._tracked_orders[order.order_id] = LiveOrderTracker(
            order_id=order.order_id,
            symbol=signal.symbol,
            side=side,
            contracts=qty,
            entry_price=signal.metadata.get("ideal_exit", 0.0),
            opened_at=signal.timestamp,
            client_tag=client_order_id,
        )
        return LiveExecutionResult(order=order)

    def _past_cutoff(self, now_local: datetime) -> bool:
        """Return True if we are past the daily entry cutoff/flat time."""
        buffer_min = getattr(self.risk_cfg, "entry_cutoff_buffer_minutes", 0)

        # Prefer explicit flat_by_time from risk config
        cutoff_str = getattr(self.risk_cfg, "flat_by_time", None)
        if cutoff_str:
            cutoff_time = _parse_time(cutoff_str)
            cutoff_time = minutes_to_time(max(0, time_to_minutes(cutoff_time) - buffer_min))
            return now_local.time() >= cutoff_time

        # Fallback: use session end minus flat buffer
        session_end = _parse_time(self.risk_manager.session.session_end)
        sess_buffer = getattr(self.risk_manager.session, "flat_buffer_minutes", 0)
        cutoff_min = max(0, time_to_minutes(session_end) - sess_buffer - buffer_min)
        cutoff_time = minutes_to_time(cutoff_min)
        return now_local.time() >= cutoff_time


def _parse_time(ts: str) -> time:
    hour, minute = map(int, ts.split(":"))
    return time(hour=hour, minute=minute)


def time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def minutes_to_time(m: int) -> time:
    m = max(0, m % (24 * 60))
    return time(hour=m // 60, minute=m % 60)


@dataclass
class LiveOrderTracker:
    """Internal structure used to map local intents to broker order ids."""

    order_id: str
    symbol: str
    side: str
    contracts: int
    entry_price: float
    opened_at: datetime
    client_tag: str

