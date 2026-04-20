"""IVB ORB live runner — subclasses LiveRunner for IVB-specific stop/target logic.

Differences from standard LiveRunner:
  1. Uses IVBORBRule as the primary rule (volume profile + aggression filter).
  2. Stop and target come from rule signal metadata (OR-floor stop, 1x range target)
     rather than ATR multipliers from config.
  3. Disables PrevVWAP, GapUp, GEX, and PSS filters (not relevant to IVB ORB).
  4. max_trades_per_day=3 to match the backtest config.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.runner import LiveRunner
from rules.ivb_orb_rule import IVBORBRule
from rules.time_of_day import TimeOfDayRule
from engine.signal_aggregator import SignalAggregator
from utils.indicators import atr

logger = logging.getLogger(__name__)


class IVBLiveRunner(LiveRunner):
    """Live runner wired for IVB ORB on MES with OR-floor stop and 1x range target."""

    def _build_rules(self) -> None:
        cfg = self.rules_cfg
        ivb_cfg = cfg["ivb_orb"]
        self.skip_monday = ivb_cfg.get("skip_monday", False)

        self._body_ratio_normal: float = ivb_cfg.get("min_body_ratio", 0.35)
        self._body_ratio_high_vix: float = ivb_cfg.get("min_body_ratio_high_vix", 0.25)
        self._vix_high_threshold: float = ivb_cfg.get("vix_high_threshold", 25.0)
        # VIX gate: only trade IVB when prior-day VIX >= this level (0 = disabled)
        # Validated on 7yr MES history: VIX>=20 → Sharpe 1.30, MaxDD $1,617 vs Sharpe 0.15, MaxDD $3,787 ungated
        self._vix_min_to_trade: float = ivb_cfg.get("min_vix_to_trade", 20.0)
        self._session_vix: float | None = None  # prior-day VIX fetched once per session

        primary = IVBORBRule(
            or_minutes=ivb_cfg.get("or_minutes", 40),
            entry_cutoff_time=ivb_cfg.get("entry_cutoff_time", "14:00"),
            min_volume_ratio=ivb_cfg.get("min_volume_ratio", 1.1),
            reload_tolerance_ticks=ivb_cfg.get("reload_tolerance_ticks", 4),
            target_range_mult=ivb_cfg.get("target_range_mult", 1.0),
            tick_size=self.risk_cfg["position"]["tick_size"],
            mode=ivb_cfg.get("mode", "both"),
            min_body_ratio=self._body_ratio_normal,
        )
        self._ivb_primary_rule = primary  # keep reference for per-session body_ratio updates

        tod_cfg = cfg["time_of_day"]
        filters = [TimeOfDayRule(
            session_start=tod_cfg["session_start"],
            session_end=tod_cfg["session_end"],
            lunch_filter_enabled=tod_cfg.get("lunch_filter_enabled", False),
            lunch_start=tod_cfg.get("lunch_start", "12:00"),
            lunch_end=tod_cfg.get("lunch_end", "13:00"),
        )]

        self.aggregator = SignalAggregator(
            primary_rule=primary,
            filter_rules=filters,
            confirmation_rules=[],
            min_confirmations=0,
        )

    def _fetch_session_vix(self) -> None:
        """Fetch prior-day VIX close and store as a trade gate.

        Validated on 7yr MES history: IVB only has edge when prior-day VIX >= 20.
        Below that: Sharpe 0.15, MaxDD $3,787 (ungated).
        At VIX>=20: Sharpe 1.30, MaxDD $1,617 (within combine $2k limit).
        Falls back gracefully — if fetch fails, gate is disabled for the session.
        """
        try:
            import yfinance as yf
            vix_df = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
            if vix_df.empty:
                logger.warning("VIX fetch empty — gate disabled today")
                self._session_vix = None
                return
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0] for c in vix_df.columns]
            closes = vix_df["Close"].squeeze()
            # Prior day's close (no look-ahead: available before session open)
            self._session_vix = float(closes.iloc[-2]) if len(closes) >= 2 else float(closes.iloc[-1])
            logger.info(f"Session VIX (prior close): {self._session_vix:.1f} | min_to_trade={self._vix_min_to_trade}")
        except Exception as exc:
            logger.warning(f"VIX fetch failed ({exc}) — gate disabled today")
            self._session_vix = None

    def _process_orb_signal(self, bar_time, latest_bar, bars_df) -> None:
        """Override: use metadata stop/target from IVBORBRule instead of ATR-based."""
        if self.skip_monday and datetime.now().weekday() == 0:
            logger.info("Skip Monday filter active — no trade today")
            return

        # Fetch VIX once per session (first bar after OR window)
        if not hasattr(self, "_body_ratio_updated_today"):
            self._fetch_session_vix()
            self._body_ratio_updated_today = True

        # VIX gate: skip if prior-day VIX below minimum (no edge in calm markets)
        if self._vix_min_to_trade > 0 and self._session_vix is not None:
            if self._session_vix < self._vix_min_to_trade:
                logger.info(f"VIX gate: {self._session_vix:.1f} < {self._vix_min_to_trade} — no trade today")
                return

        if self.trades_today >= self.max_trades_per_day:
            logger.info(f"Max trades reached: {self.trades_today}/{self.max_trades_per_day}")
            return

        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Risk block: {reason}")
            return

        decision = self.aggregator.evaluate(bars_df)
        logger.info(f"Signal: {decision.summary}")

        if not decision.should_trade:
            return

        meta = decision.primary_signal.metadata
        if "stop_price" not in meta or "target_price" not in meta:
            logger.warning("IVBORBRule signal missing stop_price/target_price metadata — skipping")
            return

        entry_price = latest_bar["close"]
        stop_loss = meta["stop_price"]
        profit_target = meta["target_price"]

        # Sanity check: stop must be on the right side of entry
        if decision.direction == 1 and stop_loss >= entry_price:
            logger.warning(f"IVB LONG stop {stop_loss:.2f} >= entry {entry_price:.2f} — skipping")
            return
        if decision.direction == -1 and stop_loss <= entry_price:
            logger.warning(f"IVB SHORT stop {stop_loss:.2f} <= entry {entry_price:.2f} — skipping")
            return

        direction_str = "LONG" if decision.direction == 1 else "SHORT"
        logger.info(
            f"IVB TRADE: {self.symbol} {direction_str} @ {entry_price:.2f} "
            f"SL={stop_loss:.2f} TP={profit_target:.2f} "
            f"mode={meta.get('mode','?')} OR=[{meta.get('or_low',0):.2f},{meta.get('or_high',0):.2f}]"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Trade logged but not executed")
            self.active_trade = {
                "direction": decision.direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
                "entry_bar_time": bar_time,
            }
            self.trades_today += 1
            return

        success = self._place_order(direction_str, entry_price, stop_loss, profit_target)
        if success:
            self.active_trade = {
                "direction": decision.direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
                "entry_bar_time": bar_time,
            }
            self.trades_today += 1
