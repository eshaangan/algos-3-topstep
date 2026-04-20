"""Live runner for MNQ overnight-low liquidity sweep reclaim.

This runner is intentionally separate from the ORB live runner. It keeps ETH
5-minute bars, computes the current session's overnight low/high, and places
fixed stop/target brackets from the sweep extreme and overnight high.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.risk_manager import RiskManager, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class PendingSweep:
    start_i: int
    expire_i: int
    level: float
    opposite: float
    extreme: float


class SweepLiveRunner:
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

        self.rules_cfg = self._load_yaml("rules.yaml")
        self.risk_cfg = self._load_yaml("risk.yaml")
        self.exec_cfg = self._load_yaml("execution_spec.yaml")

        self.symbol = self.risk_cfg["position"]["instrument"]
        sweep = self.rules_cfg["liquidity_sweep_reclaim"]
        self.bar_size_minutes = int(sweep.get("bar_size_minutes", 5))
        self.lookback_bars = int(sweep.get("lookback_bars", 6000))
        self.entry_start = pd.Timestamp(sweep.get("entry_start", "10:30")).time()
        self.entry_end = pd.Timestamp(sweep.get("entry_end", "13:00")).time()
        self.confirm_bars = int(sweep.get("confirm_bars", 1))
        self.min_sweep_ticks = int(sweep.get("min_sweep_ticks", 1))
        self.stop_buffer_ticks = int(sweep.get("stop_buffer_ticks", 1))
        self.min_rr = float(sweep.get("min_rr", 0.8))
        self.time_stop_bars = int(sweep.get("time_stop_bars", 12))
        self.max_trades_per_day = int(sweep.get("max_trades_per_day", 1))
        self.skip_if_account_has_open_position = bool(
            sweep.get("skip_if_account_has_open_position", False)
        )

        if sweep.get("side", "long") != "long" or sweep.get("level", "overnight_low") != "overnight_low":
            raise ValueError("SweepLiveRunner currently supports only overnight_low long reclaim")

        self.data_fetcher = None
        self.client = None
        self.trades_today = 0
        self.active_trade: dict | None = None
        self.pending: PendingSweep | None = None
        self.current_date = None

        self.risk_manager = RiskManager(
            contracts=self.risk_cfg["position"]["contracts"],
            point_value=self.risk_cfg["position"]["point_value"],
            tick_size=self.risk_cfg["position"]["tick_size"],
            tick_value=self.risk_cfg["position"]["tick_value"],
            max_daily_loss=self.risk_cfg["daily_limits"]["max_daily_loss"],
            per_trade_max_loss=self.risk_cfg["daily_limits"]["per_trade_max_loss"],
            max_consecutive_losses=self.risk_cfg["circuit_breaker"]["max_consecutive_losses"],
            cooldown_bars=self.risk_cfg["circuit_breaker"]["cooldown_bars"],
            flatten_minutes_before_close=self.risk_cfg["session"]["flatten_minutes_before_close"],
            drawdown_buffer=self.risk_cfg["drawdown"]["buffer_from_max"],
        )

    def _load_yaml(self, filename: str) -> dict:
        with open(self.config_dir / filename) as f:
            return yaml.safe_load(f)

    def _init_data_fetcher(self) -> bool:
        try:
            ml_v3_path = Path(__file__).parent.parent.parent / "ml_intraday_v3"
            if str(ml_v3_path) not in sys.path:
                sys.path.insert(0, str(ml_v3_path))
            from live_trading.topstepx_rest_data_fetcher import TopstepXRestDataFetcher

            self.data_fetcher = TopstepXRestDataFetcher(
                contract_id=self.contract_id,
                bar_size_minutes=self.bar_size_minutes,
                lookback_bars=self.lookback_bars,
                enable_rth_filter=False,
            )
            self.data_fetcher.initialize_buffer()
            self.client = self.data_fetcher.client
            logger.info("Data fetcher initialized with ETH bars for %s", self.contract_id)
            return True
        except Exception as e:
            logger.error("Failed to initialize data fetcher: %s", e, exc_info=True)
            return False

    def _is_session_time(self) -> bool:
        tod = self.rules_cfg["time_of_day"]
        tz = tod.get("clock_timezone", "US/Eastern")
        now = pd.Timestamp.now(tz=tz).time()
        start = pd.Timestamp(tod["session_start"]).time()
        end = pd.Timestamp(tod["session_end"]).time()
        return start <= now <= end

    def _session_date(self) -> object:
        return pd.Timestamp.now(tz="US/Eastern").date()

    def _overnight_levels(self, bars: pd.DataFrame, session_date: object) -> tuple[float, float] | None:
        if bars.empty:
            return None
        df = bars.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")

        open_ts = pd.Timestamp.combine(session_date, pd.Timestamp("09:30").time()).tz_localize("US/Eastern")
        start_ts = open_ts - pd.Timedelta(hours=15, minutes=30)
        end_ts = open_ts - pd.Timedelta(minutes=1)
        on = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if on.empty:
            return None
        return float(on["low"].min()), float(on["high"].max())

    def _place_order(self, entry_price: float, stop_loss: float, profit_target: float) -> bool:
        try:
            core_dir = str(Path(__file__).parent.parent)
            if core_dir not in sys.path:
                sys.path.insert(0, core_dir)
            from core.projectx_client import BracketInstruction

            tick_size = self.risk_cfg["position"]["tick_size"]
            qty = self.risk_manager.contracts
            stop_ticks = -max(1, abs(int(round((stop_loss - entry_price) / tick_size))))
            target_ticks = max(1, abs(int(round((profit_target - entry_price) / tick_size))))
            bracket_stop = BracketInstruction(ticks=stop_ticks, order_type=4)
            bracket_target = BracketInstruction(ticks=target_ticks, order_type=1)
            logger.info(
                "Placing BUY %sx %s @ market | stop_ticks=%s target_ticks=%s",
                qty,
                self.symbol,
                stop_ticks,
                target_ticks,
            )
            order = self.client.place_order(
                symbol=self.symbol,
                side="BUY",
                quantity=qty,
                order_type="MARKET",
                contract_id=self.contract_id,
                stop_loss_bracket=bracket_stop,
                take_profit_bracket=bracket_target,
            )
            logger.info("Order accepted: id=%s", order.order_id)
            return True
        except Exception as e:
            logger.error("Order placement failed: %s", e, exc_info=True)
            return False

    def _record_exit(self, exit_price: float, reason: str) -> None:
        if self.active_trade is None:
            return
        entry = self.active_trade["entry_price"]
        direction = 1
        contracts = self.risk_manager.contracts
        point_value = self.risk_cfg["position"]["point_value"]
        commission = self.risk_cfg["position"].get("commission_per_side", 0.62)
        pnl = (exit_price - entry) * direction * contracts * point_value - 2 * commission * contracts
        self.risk_manager.record_trade(
            TradeRecord(
                entry_bar=0,
                exit_bar=0,
                direction=direction,
                entry_price=entry,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason,
            )
        )
        logger.info("[EXIT] %s PnL=$%+.2f entry=%.2f exit=%.2f", reason, pnl, entry, exit_price)
        self.active_trade = None

    def _check_fill(self) -> None:
        if self.active_trade is None:
            return
        positions = self.client.search_open_positions()
        if positions:
            return
        last_close = self.active_trade["last_bar_close"]
        stop = self.active_trade["stop_loss"]
        target = self.active_trade["profit_target"]
        if last_close >= target:
            self._record_exit(target, "profit_target")
        elif last_close <= stop:
            self._record_exit(stop, "stop_loss")
        else:
            self._record_exit(last_close, "bracket_fill")

    def _apply_time_stop(self) -> None:
        if self.active_trade is None:
            return
        last_close = self.active_trade["last_bar_close"]
        logger.info("Time stop triggered after %s bars", self.time_stop_bars)
        if self.dry_run:
            self._record_exit(last_close, "time_stop")
            return
        try:
            for order in self.client.search_open_orders():
                try:
                    self.client.cancel_order(str(order.order_id))
                    logger.info("Cancelled bracket order id=%s", order.order_id)
                except Exception as e:
                    logger.warning("Cancel failed for id=%s: %s", order.order_id, e)
            close_order = self.client.place_order(
                symbol=self.symbol,
                side="SELL",
                quantity=self.risk_manager.contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            logger.info("Time stop flatten order placed id=%s", close_order.order_id)
            self._record_exit(last_close, "time_stop")
        except Exception as e:
            logger.error("Time stop flatten failed: %s", e, exc_info=True)

    def _process_bar(self, latest_bar: pd.Series, bars_df: pd.DataFrame) -> None:
        self.risk_manager.tick_bar()
        ts = latest_bar.name
        if ts.tz is None:
            ts_et = ts.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            ts_et = ts.tz_convert("US/Eastern")
        session_date = ts_et.date()
        t = ts_et.time()

        if self.active_trade is not None:
            self.active_trade["bars_since_entry"] += 1
            self.active_trade["last_bar_close"] = float(latest_bar["close"])
            if self.active_trade["bars_since_entry"] >= self.time_stop_bars:
                self._apply_time_stop()
            return

        if self.trades_today >= self.max_trades_per_day:
            return
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info("Risk block: %s", reason)
            return
        if not (self.entry_start <= t <= self.entry_end):
            return
        if self.skip_if_account_has_open_position and not self.dry_run:
            try:
                if self.client.search_open_positions():
                    logger.info("Open account position exists; skipping sweep entry")
                    return
            except Exception as e:
                logger.warning("Could not check open positions: %s", e)

        levels = self._overnight_levels(bars_df, session_date)
        if levels is None:
            logger.debug("No overnight levels available yet")
            return
        overnight_low, overnight_high = levels
        tick = self.risk_cfg["position"]["tick_size"]
        low = float(latest_bar["low"])
        high = float(latest_bar["high"])
        close = float(latest_bar["close"])
        bar_i = len(bars_df) - 1

        if self.pending is not None and bar_i > self.pending.expire_i:
            logger.info("Pending sweep expired without reclaim")
            self.pending = None

        if self.pending is not None:
            self.pending.extreme = min(self.pending.extreme, low)
            if close > self.pending.level:
                entry = close + tick
                stop = self.pending.extreme - self.stop_buffer_ticks * tick
                target = self.pending.opposite
                risk_pts = entry - stop
                reward_pts = target - entry
                risk_usd = risk_pts * self.risk_cfg["position"]["point_value"] * self.risk_manager.contracts
                if risk_pts <= 0 or reward_pts <= 0:
                    logger.info("Invalid reclaim bracket; clearing pending sweep")
                    self.pending = None
                    return
                if reward_pts / risk_pts < self.min_rr:
                    logger.info("Reclaim skipped: RR %.2f < %.2f", reward_pts / risk_pts, self.min_rr)
                    self.pending = None
                    return
                if risk_usd > self.risk_cfg["daily_limits"]["per_trade_max_loss"]:
                    logger.info("Reclaim skipped: risk $%.2f exceeds per-trade limit", risk_usd)
                    self.pending = None
                    return

                logger.info(
                    "SWEEP RECLAIM LONG: entry=%.2f stop=%.2f target=%.2f on_low=%.2f on_high=%.2f",
                    entry,
                    stop,
                    target,
                    overnight_low,
                    overnight_high,
                )
                if self.dry_run or self._place_order(entry, stop, target):
                    self.active_trade = {
                        "direction": 1,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "profit_target": target,
                        "bars_since_entry": 0,
                        "last_bar_close": close,
                    }
                    self.trades_today += 1
                self.pending = None
            return

        if low <= overnight_low - self.min_sweep_ticks * tick:
            self.pending = PendingSweep(
                start_i=bar_i,
                expire_i=bar_i + self.confirm_bars,
                level=overnight_low,
                opposite=overnight_high,
                extreme=low,
            )
            logger.info(
                "Overnight-low sweep detected: low=%.2f overnight_low=%.2f expires_i=%s",
                low,
                overnight_low,
                self.pending.expire_i,
            )

    def _signal_handler(self, signum, frame) -> None:
        logger.warning("Signal %s received - shutting down", signum)
        self.running = False

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("LIQUIDITY SWEEP LIVE SIDECAR [%s]", self.symbol)
        logger.info("=" * 60)
        logger.info("Dry run: %s", self.dry_run)
        logger.info("Contract: %s", self.contract_id)
        logger.info("Config: %s", self.config_dir)

        if not self._init_data_fetcher():
            return

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True
        update_interval = 30

        while self.running:
            try:
                if not self._is_session_time():
                    time.sleep(60)
                    continue
                today = self._session_date()
                if self.current_date is not None and today != self.current_date:
                    self.risk_manager.reset_daily()
                    self.trades_today = 0
                    self.active_trade = None
                    self.pending = None
                    logger.info("New trading day - sweep state reset")
                self.current_date = today

                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                if self.active_trade is not None and not self.dry_run:
                    self._check_fill()

                latest_bar = self.data_fetcher.get_latest_bar()
                if latest_bar is not None and (
                    self.last_bar_time is None or latest_bar.name > self.last_bar_time
                ):
                    logger.info("New bar: %s close=%.2f", latest_bar.name, latest_bar["close"])
                    self._process_bar(latest_bar, self.data_fetcher.get_buffer())
                    self.last_bar_time = latest_bar.name
                time.sleep(update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in sweep loop: %s", e, exc_info=True)
                time.sleep(update_interval)

        logger.info("Sweep live stopped")
