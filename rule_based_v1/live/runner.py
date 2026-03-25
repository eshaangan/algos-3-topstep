"""Live trading runner for rule-based system.

Adapted from ml_intraday_v3/live_trading/live_runner.py.
Replaces ML prediction pipeline with rule-based signal generation.
Places orders directly via ProjectXClient with ATR-based bracket stops.

Instrument is driven by configs/risk.yaml  position.instrument field.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.opening_range import OpeningRangeBreakoutRule
from rules.time_of_day import TimeOfDayRule
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager, TradeRecord
from utils.indicators import atr

logger = logging.getLogger(__name__)


class LiveRunner:
    """Live trading runner using rule-based signals.

    Polls for new bars, evaluates rules, and executes trades
    via the TopstepX execution infrastructure.
    """

    def __init__(
        self,
        config_dir: Path,
        dry_run: bool = True,
        contract_id: str | None = None,
        account_id: str | None = None,
    ):
        self.config_dir = Path(config_dir)
        self.dry_run = dry_run
        self.contract_id = contract_id or os.getenv("TOPSTEPX_CONTRACT_ID")
        self.account_id = account_id or os.getenv("TOPSTEPX_ACCOUNT_ID")
        self.running = False
        self.last_bar_time = None

        # Load configs
        self.rules_cfg = self._load_yaml("rules.yaml")
        self.risk_cfg  = self._load_yaml("risk.yaml")
        self.exec_cfg  = self._load_yaml("execution_spec.yaml")

        # Instrument symbol from risk config (e.g. "MNQ", "MES")
        self.symbol = self.risk_cfg["position"]["instrument"]

        # Build trading components
        self._build_rules()
        self._build_risk_manager()

        # Data fetcher and API client (initialised in run())
        self.data_fetcher = None
        self.client = None  # ProjectXClient for direct order placement

        # Session state (reset daily)
        self.trades_today: int = 0
        self.max_trades_per_day: int = self.risk_cfg.get("session", {}).get("max_trades_per_day", 1)
        self.active_trade: dict | None = None  # tracks open position
        self.last_trade_direction: int = 0     # 1=LONG, -1=SHORT
        self.time_stop_bars: int = self.rules_cfg.get("exit_strategy", {}).get("time_stop_bars", 24)
        self.require_prev_vwap: bool = self.rules_cfg.get("opening_range_breakout", {}).get("require_prev_vwap", False)
        self.require_gex_explosive: bool = self.rules_cfg.get("opening_range_breakout", {}).get("require_gex_explosive", False)
        self.session_gex_explosive: bool | None = None  # computed once per day

        # MSITE engine (secondary signal)
        self._init_msite()
        self.msite_active_trade: dict | None = None

        # GIRE engine (gap inventory repair — secondary signal)
        self._init_gire()
        self.gire_active_trade: dict | None = None

    def _init_msite(self) -> None:
        """Initialise the MSITE secondary signal engine from rules.yaml [msite] section."""
        msite_cfg = self.rules_cfg.get("msite", {})
        if not msite_cfg.get("enabled", False):
            self.msite_engine = None
            logger.info("MSITE engine: disabled (rules.yaml msite.enabled=false or missing)")
            return
        try:
            msite_dir = str(Path(__file__).parent)
            if msite_dir not in sys.path:
                sys.path.insert(0, msite_dir)
            from msite_engine import MSITEEngine  # noqa: PLC0415

            self.msite_engine = MSITEEngine(
                n_contracts=msite_cfg.get("n_contracts", 12),
                pt_mult=msite_cfg.get("pt_mult", 2.0),
                sl_frac=msite_cfg.get("sl_frac", 0.55),
                afternoon_cutoff=tuple(msite_cfg.get("afternoon_cutoff", [14, 0])),
            )
            self.msite_time_stop_bars: int = msite_cfg.get("time_stop_bars", 12)
            logger.info(
                f"MSITE engine: enabled (n_contracts={self.msite_engine.n_contracts}, "
                f"pt_mult={self.msite_engine.pt_mult}, sl_frac={self.msite_engine.sl_frac})"
            )
        except Exception as e:
            logger.error(f"MSITE engine: init failed — {e}", exc_info=True)
            self.msite_engine = None

    def _init_gire(self) -> None:
        """Initialise the GIRE secondary signal engine from rules.yaml [gire] section."""
        gire_cfg = self.rules_cfg.get("gire", {})
        if not gire_cfg.get("enabled", False):
            self.gire_engine = None
            logger.info("GIRE engine: disabled (rules.yaml gire.enabled=false or missing)")
            return
        try:
            gire_dir = str(Path(__file__).parent)
            if gire_dir not in sys.path:
                sys.path.insert(0, gire_dir)
            from gire_engine import GIREEngine  # noqa: PLC0415

            self.gire_engine = GIREEngine(
                g_star=gire_cfg.get("g_star", 0.2),
                clv_thresh=gire_cfg.get("clv_thresh", 0.8),
                use_opening_failure=gire_cfg.get("use_opening_failure", True),
                time_stop_hour=gire_cfg.get("time_stop_hour", 12),
                n_contracts=gire_cfg.get("n_contracts", 1),
            )
            self.gire_time_stop_bars: int = gire_cfg.get("time_stop_bars", 48)
            logger.info(
                f"GIRE engine: enabled (g_star={self.gire_engine.g_star}, "
                f"clv_thresh={self.gire_engine.clv_thresh}, "
                f"use_opening_failure={self.gire_engine.use_opening_failure}, "
                f"time_stop_hour={self.gire_engine.time_stop_hour}, "
                f"n_contracts={self.gire_engine.n_contracts})"
            )
        except Exception as e:
            logger.error(f"GIRE engine: init failed — {e}", exc_info=True)
            self.gire_engine = None

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path) as f:
            return yaml.safe_load(f)

    def _build_rules(self):
        """Instantiate ORB as primary rule with TimeOfDay filter."""
        cfg = self.rules_cfg

        orb_cfg = cfg["opening_range_breakout"]
        primary = OpeningRangeBreakoutRule(
            or_end_time=orb_cfg["or_end_time"],
            min_or_bars=orb_cfg.get("min_or_bars", 4),
            min_range_atr=orb_cfg["min_range_atr"],
            entry_cutoff_time=orb_cfg["entry_cutoff_time"],
            atr_period=orb_cfg.get("atr_period", 14),
            use_close_for_signal=orb_cfg.get("use_close_for_signal", True),
            long_only=orb_cfg.get("long_only", False),
        )

        tod_cfg = cfg["time_of_day"]
        filters = [TimeOfDayRule(
            session_start=tod_cfg["session_start"],
            session_end=tod_cfg["session_end"],
            lunch_filter_enabled=tod_cfg.get("lunch_filter_enabled", False),
        )]

        self.aggregator = SignalAggregator(
            primary_rule=primary,
            filter_rules=filters,
            confirmation_rules=[],
            min_confirmations=0,
        )

    def _build_risk_manager(self):
        """Build risk manager from config."""
        cfg = self.risk_cfg
        self.risk_manager = RiskManager(
            contracts=cfg["position"]["contracts"],
            point_value=cfg["position"]["point_value"],
            tick_size=cfg["position"]["tick_size"],
            tick_value=cfg["position"]["tick_value"],
            max_daily_loss=cfg["daily_limits"]["max_daily_loss"],
            per_trade_max_loss=cfg["daily_limits"]["per_trade_max_loss"],
            max_consecutive_losses=cfg["circuit_breaker"]["max_consecutive_losses"],
            cooldown_bars=cfg["circuit_breaker"]["cooldown_bars"],
            flatten_minutes_before_close=cfg["session"]["flatten_minutes_before_close"],
            drawdown_buffer=cfg["drawdown"]["buffer_from_max"],
        )

    def _init_data_fetcher(self):
        """Initialize TopstepX data fetcher and reuse its authenticated client."""
        try:
            ml_v3_path = Path(__file__).parent.parent.parent / "ml_intraday_v3"
            sys.path.insert(0, str(ml_v3_path))
            from live_trading.topstepx_rest_data_fetcher import TopstepXRestDataFetcher

            self.data_fetcher = TopstepXRestDataFetcher(
                contract_id=self.contract_id,
                bar_size_minutes=5,
                lookback_bars=100,
                enable_rth_filter=True,
            )
            self.data_fetcher.initialize_buffer()

            # Reuse the authenticated client from the data fetcher
            self.client = self.data_fetcher.client
            logger.info(f"Data fetcher initialized for {self.symbol} ({self.contract_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to init data fetcher: {e}", exc_info=True)
            return False

    def _is_trading_time(self) -> bool:
        """Check if current time is within trading session."""
        now = datetime.now().astimezone()
        try:
            now_et = now.astimezone(pd.Timestamp.now(tz="US/Eastern").tzinfo)
        except Exception:
            return True  # default to allowing if timezone fails

        t = now_et.time()
        session_start = pd.Timestamp(self.rules_cfg["time_of_day"]["session_start"]).time()
        session_end   = pd.Timestamp(self.rules_cfg["time_of_day"]["session_end"]).time()
        return session_start <= t <= session_end

    def _place_order(self, direction_str: str, entry_price: float,
                     stop_loss: float, profit_target: float) -> bool:
        """Place a bracketed market order via ProjectX REST API."""
        try:
            core_dir = str(Path(__file__).parent.parent)
            if core_dir not in sys.path:
                sys.path.insert(0, core_dir)
            from core.projectx_client import BracketInstruction

            tick_size    = self.risk_cfg["position"]["tick_size"]
            n_contracts  = self.risk_manager.contracts

            # Ticks relative to entry (signed by convention)
            stop_ticks_raw   = int(round((stop_loss    - entry_price) / tick_size))
            target_ticks_raw = int(round((profit_target - entry_price) / tick_size))

            if direction_str == "LONG":
                stop_ticks   = -max(1, abs(stop_ticks_raw))
                target_ticks =  max(1, abs(target_ticks_raw))
                side = "BUY"
            else:
                stop_ticks   =  max(1, abs(stop_ticks_raw))
                target_ticks = -max(1, abs(target_ticks_raw))
                side = "SELL"

            bracket_stop   = BracketInstruction(ticks=stop_ticks,   order_type=4)  # STOP
            bracket_target = BracketInstruction(ticks=target_ticks, order_type=1)  # LIMIT

            logger.info(
                f"Placing {side} {n_contracts}x {self.symbol} @ market | "
                f"stop_ticks={stop_ticks} ({stop_loss:.2f}), "
                f"target_ticks={target_ticks} ({profit_target:.2f})"
            )

            order = self.client.place_order(
                symbol=self.symbol,
                side=side,
                quantity=n_contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
                stop_loss_bracket=bracket_stop,
                take_profit_bracket=bracket_target,
            )
            logger.info(
                f"Order accepted: id={order.order_id}, "
                f"side={order.side}, qty={order.quantity}"
            )
            return True

        except Exception as e:
            logger.error(f"Order placement failed: {e}", exc_info=True)
            return False

    def _check_fill(self):
        """Poll open positions; when position disappears, infer exit and record trade."""
        if self.active_trade is None:
            return
        try:
            positions = self.client.search_open_positions()
            if positions:
                return

            # Position is gone — bracket filled
            trade       = self.active_trade
            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            sl          = trade["stop_loss"]
            tp          = trade["profit_target"]
            last_close  = trade["last_bar_close"]

            if direction == 1:  # LONG
                if last_close >= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close <= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"
            else:  # SHORT
                if last_close <= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close >= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"

            contracts        = self.risk_manager.contracts
            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (exit_price - entry_price) * direction * contracts * point_value
            total_commission = 2 * commission * contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason,
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[FILL] Trade closed via {reason}: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={exit_price})"
            )
            self.active_trade = None

        except Exception as e:
            logger.error(f"_check_fill error: {e}", exc_info=True)

    def _apply_time_stop(self):
        """Cancel open brackets and flatten position after time_stop_bars elapsed."""
        trade = self.active_trade
        direction_label = "LONG" if trade["direction"] == 1 else "SHORT"
        logger.info(
            f"Time stop triggered: {self.time_stop_bars} bars after entry ({direction_label})"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Time stop would cancel brackets and flatten position")
            self.active_trade = None
            return

        try:
            # Check if still in a position (TP/SL may have already closed it)
            positions = self.client.search_open_positions()
            if not positions:
                logger.info("Time stop: position already closed (TP/SL hit), no action needed")
                self.active_trade = None
                return

            logger.info(f"Time stop: {len(positions)} open position(s) found, flattening")

            # Cancel any open bracket orders first
            open_orders = self.client.search_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(str(order.order_id))
                    logger.info(f"Time stop: cancelled bracket order id={order.order_id}")
                except Exception as e:
                    logger.warning(f"Time stop: cancel failed for id={order.order_id}: {e}")

            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            last_close  = trade["last_bar_close"]

            close_side = "SELL" if direction == 1 else "BUY"
            close_order = self.client.place_order(
                symbol=self.symbol,
                side=close_side,
                quantity=self.risk_manager.contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            logger.info(f"Time stop: FLATTEN order placed id={close_order.order_id}")

            contracts        = self.risk_manager.contracts
            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (last_close - entry_price) * direction * contracts * point_value
            total_commission = 2 * commission * contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=last_close,
                pnl=pnl,
                exit_reason="time_stop",
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[TIME STOP] Trade closed: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={last_close})"
            )

        except Exception as e:
            logger.error(f"Time stop flatten failed: {e}", exc_info=True)
        finally:
            self.active_trade = None

    def _compute_prev_vwap_bullish(self, bars_df: pd.DataFrame) -> bool | None:
        """Return True if yesterday's session closed above yesterday's session VWAP.

        Uses the rolling bar buffer already in memory — no extra API calls needed.
        Returns None when there aren't enough prior-day bars to make the determination.
        """
        if bars_df is None or len(bars_df) < 2:
            return None

        today = bars_df.index[-1].date()
        prev_bars = bars_df[bars_df.index.map(lambda t: t.date()) < today]
        if len(prev_bars) == 0:
            return None

        prev_day = prev_bars.index[-1].date()
        day_bars = prev_bars[prev_bars.index.map(lambda t: t.date()) == prev_day]
        if len(day_bars) < 5:
            return None

        typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
        cum_vol = day_bars["volume"].cumsum().replace(0, float("nan"))
        vwap = (typical * day_bars["volume"]).cumsum() / cum_vol

        last_close = float(day_bars["close"].iloc[-1])
        last_vwap = float(vwap.iloc[-1])
        result = last_close > last_vwap
        logger.debug(
            f"PrevVWAP [{prev_day}]: close={last_close:.2f}  vwap={last_vwap:.2f}  "
            f"bullish={result}"
        )
        return result

    def _compute_gex_explosive(self) -> bool:
        """
        GEX proxy: returns True when today's VXN is below its 60-day rolling mean.
        VXN below mean = dealers are net short gamma = explosive breakout regime.
        Uses yfinance — one call per session day. Falls back to True (allow trade) on error.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — GEX filter defaulting to allow trade")
            return True

        try:
            vxn_df = yf.download("^VXN", period="90d", interval="1d", progress=False, auto_adjust=True)
            c = vxn_df["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            c = c.dropna().astype(float)
            if len(c) < 20:
                logger.warning("GEX proxy: insufficient VXN history, defaulting to allow trade")
                return True
            rolling_mean = c.rolling(60, min_periods=20).mean()
            today_vxn  = float(c.iloc[-1])
            mean_60d   = float(rolling_mean.iloc[-1])
            explosive  = today_vxn < mean_60d
            z_score    = (today_vxn - mean_60d) / (c.rolling(60, min_periods=20).std().iloc[-1] or 1.0)
            logger.info(
                f"GEX proxy: VXN={today_vxn:.2f}, 60d_mean={mean_60d:.2f}, "
                f"z={z_score:.2f}, explosive={explosive}"
            )
            return explosive
        except Exception as e:
            logger.warning(f"GEX proxy computation failed ({e}) — defaulting to allow trade")
            return True

    def _process_bar(self, bar_time, latest_bar, bars_df):
        """Process a new bar through the rule engine."""
        self.risk_manager.tick_bar()

        # Active trade monitoring: update bar count and check time stop
        if self.active_trade is not None:
            self.active_trade["bars_since_entry"] += 1
            self.active_trade["last_bar_close"] = latest_bar["close"]
            if self.active_trade["bars_since_entry"] >= self.time_stop_bars:
                self._apply_time_stop()
        else:
            self._process_orb_signal(bar_time, latest_bar, bars_df)

        # MSITE secondary signal engine — runs every bar regardless of ORB outcome
        if self.msite_engine is not None:
            self._process_msite_bar(bar_time, latest_bar, bars_df)

        # GIRE secondary signal engine — runs every bar regardless of ORB/MSITE outcome
        if self.gire_engine is not None:
            self._process_gire_bar(bar_time, latest_bar, bars_df)

    def _process_orb_signal(self, bar_time, latest_bar, bars_df) -> None:
        """Evaluate ORB rule and place order when conditions are met."""
        # Daily trade cap
        if self.trades_today >= self.max_trades_per_day:
            logger.info(f"Max trades reached: {self.trades_today}/{self.max_trades_per_day}")
            return

        # Check risk limits
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Risk block: {reason}")
            return

        # PrevVWAP filter: skip if yesterday closed below session VWAP
        if self.require_prev_vwap:
            pv = self._compute_prev_vwap_bullish(bars_df)
            if pv is False:
                logger.info("PrevVWAP filter: yesterday bearish — no trade today")
                return

        # GEX proxy filter: skip if VXN >= 60d mean (dealers not in explosive regime)
        if self.require_gex_explosive:
            if self.session_gex_explosive is None:
                self.session_gex_explosive = self._compute_gex_explosive()
            if not self.session_gex_explosive:
                logger.info("GEX filter: VXN above 60d mean — non-explosive regime, no trade today")
                return

        # Evaluate rules
        decision = self.aggregator.evaluate(bars_df)
        logger.info(f"Signal: {decision.summary}")

        if not decision.should_trade:
            return

        # ATR for position sizing
        current_atr = atr(bars_df["high"], bars_df["low"], bars_df["close"]).iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            logger.warning("Invalid ATR, skipping trade")
            return

        exit_cfg     = self.rules_cfg["exit_strategy"]
        entry_price  = latest_bar["close"]

        stop_loss = self.risk_manager.compute_stop_price(
            entry_price, decision.direction, current_atr, exit_cfg["stop_loss_atr"]
        )
        profit_target = self.risk_manager.compute_target_price(
            entry_price, decision.direction, current_atr, exit_cfg["profit_target_atr"]
        )

        direction_str = "LONG" if decision.direction == 1 else "SHORT"
        logger.info(
            f"TRADE SIGNAL: {self.symbol} {direction_str} @ {entry_price:.2f}, "
            f"SL={stop_loss:.2f}, TP={profit_target:.2f}, ATR={current_atr:.2f}"
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
            }
            self.trades_today += 1
            self.last_trade_direction = decision.direction
            return

        # Live: place order via ProjectX API
        success = self._place_order(direction_str, entry_price, stop_loss, profit_target)
        if success:
            logger.info(f"[LIVE] Order placed: {self.symbol} {direction_str}")
            self.active_trade = {
                "direction": decision.direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
            }
            self.trades_today += 1
            self.last_trade_direction = decision.direction
        else:
            logger.error(f"[LIVE] Order placement FAILED for {self.symbol} {direction_str}")

    def _process_msite_bar(self, bar_time, latest_bar, bars_df) -> None:
        """Process one bar through the MSITE engine and manage its active trade."""
        # Active MSITE trade monitoring
        if self.msite_active_trade is not None:
            self.msite_active_trade["bars_since_entry"] += 1
            self.msite_active_trade["last_bar_close"] = latest_bar["close"]
            if self.msite_active_trade["bars_since_entry"] >= self.msite_time_stop_bars:
                self._apply_msite_time_stop()
            return

        # Risk gate
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"MSITE risk block: {reason}")
            return

        # Get MSITE signal
        signal = self.msite_engine.update(bar_time, latest_bar)
        if signal is None:
            return

        direction = signal["direction"]
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        profit_target = signal["profit_target"]
        n_contracts = signal["n_contracts"]
        direction_str = "LONG" if direction == 1 else "SHORT"

        logger.info(
            f"MSITE SIGNAL: {self.symbol} {direction_str} @ {entry_price:.2f}, "
            f"SL={stop_loss:.2f}, TP={profit_target:.2f}, "
            f"sigma={signal['sigma_abs']:.2f}, n={n_contracts}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] MSITE trade logged but not executed")
            self.msite_active_trade = {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "n_contracts": n_contracts,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
            }
            return

        # Live: temporarily override contract count for MSITE sizing
        original_contracts = self.risk_manager.contracts
        self.risk_manager.contracts = n_contracts
        try:
            success = self._place_order(direction_str, entry_price, stop_loss, profit_target)
        finally:
            self.risk_manager.contracts = original_contracts

        if success:
            logger.info(f"[LIVE] MSITE order placed: {self.symbol} {direction_str} x{n_contracts}")
            self.msite_active_trade = {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "n_contracts": n_contracts,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
            }
        else:
            logger.error(f"[LIVE] MSITE order placement FAILED for {self.symbol} {direction_str}")

    def _apply_msite_time_stop(self) -> None:
        """Cancel open brackets and flatten MSITE position after time_stop_bars elapsed."""
        trade = self.msite_active_trade
        direction_label = "LONG" if trade["direction"] == 1 else "SHORT"
        logger.info(
            f"MSITE time stop triggered: {self.msite_time_stop_bars} bars after entry ({direction_label})"
        )

        if self.dry_run:
            logger.info("[DRY RUN] MSITE time stop would cancel brackets and flatten position")
            self.msite_active_trade = None
            return

        try:
            positions = self.client.search_open_positions()
            if not positions:
                logger.info("MSITE time stop: position already closed (TP/SL hit), no action needed")
                self.msite_active_trade = None
                return

            logger.info(f"MSITE time stop: {len(positions)} open position(s) found, flattening")

            open_orders = self.client.search_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(str(order.order_id))
                    logger.info(f"MSITE time stop: cancelled bracket order id={order.order_id}")
                except Exception as e:
                    logger.warning(f"MSITE time stop: cancel failed for id={order.order_id}: {e}")

            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            last_close  = trade["last_bar_close"]
            n_contracts = trade["n_contracts"]

            close_side = "SELL" if direction == 1 else "BUY"
            close_order = self.client.place_order(
                symbol=self.symbol,
                side=close_side,
                quantity=n_contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            logger.info(f"MSITE time stop: FLATTEN order placed id={close_order.order_id}")

            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (last_close - entry_price) * direction * n_contracts * point_value
            total_commission = 2 * commission * n_contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=last_close,
                pnl=pnl,
                exit_reason="time_stop",
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[MSITE TIME STOP] Trade closed: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={last_close})"
            )

        except Exception as e:
            logger.error(f"MSITE time stop flatten failed: {e}", exc_info=True)
        finally:
            self.msite_active_trade = None

    def _check_msite_fill(self) -> None:
        """Poll open positions; when MSITE position disappears, infer exit and record trade."""
        if self.msite_active_trade is None:
            return
        try:
            positions = self.client.search_open_positions()
            if positions:
                return

            trade       = self.msite_active_trade
            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            sl          = trade["stop_loss"]
            tp          = trade["profit_target"]
            last_close  = trade["last_bar_close"]
            n_contracts = trade["n_contracts"]

            if direction == 1:
                if last_close >= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close <= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"
            else:
                if last_close <= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close >= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"

            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (exit_price - entry_price) * direction * n_contracts * point_value
            total_commission = 2 * commission * n_contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason,
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[MSITE FILL] Trade closed via {reason}: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={exit_price})"
            )
            self.msite_active_trade = None

        except Exception as e:
            logger.error(f"_check_msite_fill error: {e}", exc_info=True)

    def _process_gire_bar(self, bar_time, latest_bar, bars_df) -> None:
        """Process one bar through the GIRE engine and manage its active trade."""
        # Active GIRE trade monitoring
        if self.gire_active_trade is not None:
            self.gire_active_trade["bars_since_entry"] += 1
            self.gire_active_trade["last_bar_close"] = latest_bar["close"]
            if self.gire_active_trade["bars_since_entry"] >= self.gire_time_stop_bars:
                self._apply_gire_time_stop()
            return

        # Risk gate
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"GIRE risk block: {reason}")
            return

        # Get GIRE signal
        signal = self.gire_engine.update(bar_time, latest_bar, bars_df)
        if signal is None:
            return

        direction = signal["direction"]
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        profit_target = signal["profit_target"]
        n_contracts = signal["n_contracts"]
        direction_str = "LONG" if direction == 1 else "SHORT"

        logger.info(
            f"GIRE SIGNAL: {self.symbol} {direction_str} @ {entry_price:.2f}, "
            f"SL={stop_loss:.2f}, TP={profit_target:.2f}, "
            f"G_t={signal['G_t']:.3f}, CLV_y={signal['CLV_y']:.3f}, "
            f"ATR_d={signal['ATR_d']:.4f}, n={n_contracts}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] GIRE trade logged but not executed")
            self.gire_active_trade = {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "n_contracts": n_contracts,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
            }
            return

        # Live: temporarily override contract count for GIRE sizing
        original_contracts = self.risk_manager.contracts
        self.risk_manager.contracts = n_contracts
        try:
            success = self._place_order(direction_str, entry_price, stop_loss, profit_target)
        finally:
            self.risk_manager.contracts = original_contracts

        if success:
            logger.info(f"[LIVE] GIRE order placed: {self.symbol} {direction_str} x{n_contracts}")
            self.gire_active_trade = {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "n_contracts": n_contracts,
                "bars_since_entry": 0,
                "last_bar_close": entry_price,
            }
        else:
            logger.error(f"[LIVE] GIRE order placement FAILED for {self.symbol} {direction_str}")

    def _apply_gire_time_stop(self) -> None:
        """Cancel open brackets and flatten GIRE position after time_stop_bars elapsed."""
        trade = self.gire_active_trade
        direction_label = "LONG" if trade["direction"] == 1 else "SHORT"
        logger.info(
            f"GIRE time stop triggered: {self.gire_time_stop_bars} bars after entry ({direction_label})"
        )

        if self.dry_run:
            logger.info("[DRY RUN] GIRE time stop would cancel brackets and flatten position")
            self.gire_active_trade = None
            return

        try:
            positions = self.client.search_open_positions()
            if not positions:
                logger.info("GIRE time stop: position already closed (TP/SL hit), no action needed")
                self.gire_active_trade = None
                return

            logger.info(f"GIRE time stop: {len(positions)} open position(s) found, flattening")

            open_orders = self.client.search_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(str(order.order_id))
                    logger.info(f"GIRE time stop: cancelled bracket order id={order.order_id}")
                except Exception as e:
                    logger.warning(f"GIRE time stop: cancel failed for id={order.order_id}: {e}")

            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            last_close  = trade["last_bar_close"]
            n_contracts = trade["n_contracts"]

            close_side = "SELL" if direction == 1 else "BUY"
            close_order = self.client.place_order(
                symbol=self.symbol,
                side=close_side,
                quantity=n_contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            logger.info(f"GIRE time stop: FLATTEN order placed id={close_order.order_id}")

            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (last_close - entry_price) * direction * n_contracts * point_value
            total_commission = 2 * commission * n_contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=last_close,
                pnl=pnl,
                exit_reason="time_stop",
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[GIRE TIME STOP] Trade closed: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={last_close})"
            )

        except Exception as e:
            logger.error(f"GIRE time stop flatten failed: {e}", exc_info=True)
        finally:
            self.gire_active_trade = None

    def _check_gire_fill(self) -> None:
        """Poll open positions; when GIRE position disappears, infer exit and record trade."""
        if self.gire_active_trade is None:
            return
        try:
            positions = self.client.search_open_positions()
            if positions:
                return

            trade       = self.gire_active_trade
            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            sl          = trade["stop_loss"]
            tp          = trade["profit_target"]
            last_close  = trade["last_bar_close"]
            n_contracts = trade["n_contracts"]

            if direction == 1:
                if last_close >= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close <= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"
            else:
                if last_close <= tp:
                    exit_price = tp
                    reason = "profit_target"
                elif last_close >= sl:
                    exit_price = sl
                    reason = "stop_loss"
                else:
                    exit_price = last_close
                    reason = "bracket_fill"

            point_value      = self.risk_cfg["position"]["point_value"]
            commission       = self.risk_cfg["position"].get("commission_per_side", 0.62)
            gross_pnl        = (exit_price - entry_price) * direction * n_contracts * point_value
            total_commission = 2 * commission * n_contracts
            pnl              = gross_pnl - total_commission

            trade_record = TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason,
            )
            self.risk_manager.record_trade(trade_record)
            logger.info(
                f"[GIRE FILL] Trade closed via {reason}: PnL=${pnl:+.2f} "
                f"(entry={entry_price}, exit={exit_price})"
            )
            self.gire_active_trade = None

        except Exception as e:
            logger.error(f"_check_gire_fill error: {e}", exc_info=True)

    def _signal_handler(self, signum, frame):
        logger.warning(f"Signal {signum} received - shutting down")
        self.running = False

    def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info(f"RULE-BASED LIVE TRADING  [{self.symbol}]")
        logger.info("=" * 60)
        logger.info(f"Dry run:  {self.dry_run}")
        logger.info(f"Symbol:   {self.symbol}")
        logger.info(f"Contract: {self.contract_id}")

        if not self._init_data_fetcher():
            logger.error("Failed to initialize - aborting")
            return

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True

        logger.info("Live trading started")
        update_interval = 30  # seconds
        current_date = None

        while self.running:
            try:
                if not self._is_trading_time():
                    logger.debug("Outside trading hours")
                    time.sleep(60)
                    continue

                # New-day state reset
                now_date = datetime.now().date()
                if current_date is not None and now_date != current_date:
                    self.risk_manager.reset_daily()
                    self.trades_today = 0
                    self.active_trade = None
                    self.last_trade_direction = 0
                    self.session_gex_explosive = None  # recompute each day
                    if self.msite_engine is not None:
                        self.msite_engine.reset_day(0.0)
                    self.msite_active_trade = None
                    self.gire_active_trade = None
                    if self.gire_engine is not None:
                        self.gire_engine.reset_day()
                    logger.info("New trading day — risk state reset")
                current_date = now_date

                # Poll for new bar
                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                # Check if active trade bracket filled (live mode only)
                if self.active_trade is not None and not self.dry_run:
                    try:
                        self._check_fill()
                    except Exception as e:
                        logger.error(f"_check_fill loop error: {e}", exc_info=True)

                if self.msite_active_trade is not None and not self.dry_run:
                    try:
                        self._check_msite_fill()
                    except Exception as e:
                        logger.error(f"_check_msite_fill error: {e}", exc_info=True)

                if self.gire_active_trade is not None and not self.dry_run:
                    try:
                        self._check_gire_fill()
                    except Exception as e:
                        logger.error(f"_check_gire_fill error: {e}", exc_info=True)

                latest_bar = self.data_fetcher.get_latest_bar()
                if latest_bar is not None and (
                    self.last_bar_time is None or latest_bar.name > self.last_bar_time
                ):
                    bar_time = latest_bar.name
                    logger.info(
                        f"New bar: {bar_time}, "
                        f"{self.symbol} close={latest_bar['close']:.2f}"
                    )
                    bars_df = self.data_fetcher.get_buffer()
                    self._process_bar(bar_time, latest_bar, bars_df)
                    self.last_bar_time = bar_time

                time.sleep(update_interval)

            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(update_interval)

        logger.info("Live trading stopped")
        stats = self.risk_manager.session_stats
        logger.info(
            f"Session stats: trades_today={self.trades_today}, "
            f"daily_pnl=${stats['daily_pnl']:+.2f}, "
            f"wins={stats['wins']}, losses={stats['losses']}, "
            f"win_rate={stats['win_rate']:.1%}, "
            f"halted={stats['halted']}"
        )
