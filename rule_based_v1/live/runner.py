"""Live trading runner for rule-based system.

Adapted from ml_intraday_v3/live_trading/live_runner.py.
Replaces ML prediction pipeline with rule-based signal generation.
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

from rules.ema_trend import EMATrendRule
from rules.time_of_day import TimeOfDayRule
from rules.volume_breakout import VolumeBreakoutRule
from rules.mean_reversion import MeanReversionRule
from rules.rejection_pattern import RejectionPatternRule
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
        self.risk_cfg = self._load_yaml("risk.yaml")
        self.exec_cfg = self._load_yaml("execution_spec.yaml")

        # Build trading components
        self._build_rules()
        self._build_risk_manager()

        # Data fetcher + execution engine (from ml_intraday_v3)
        self.data_fetcher = None
        self.execution_engine = None

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path) as f:
            return yaml.safe_load(f)

    def _build_rules(self):
        """Instantiate all trading rules and aggregator."""
        cfg = self.rules_cfg

        ema_cfg = cfg["ema_trend"]
        primary = EMATrendRule(
            fast_period=ema_cfg["fast_period"],
            slow_period=ema_cfg["slow_period"],
            min_spread_atr_ratio=ema_cfg["min_spread_atr_ratio"],
            slope_lookback=ema_cfg["slope_lookback"],
            atr_period=ema_cfg["atr_period"],
        )

        filters = []
        tod_cfg = cfg["time_of_day"]
        filters.append(TimeOfDayRule(
            session_start=tod_cfg["session_start"],
            session_end=tod_cfg["session_end"],
            lunch_filter_enabled=tod_cfg.get("lunch_filter_enabled", False),
        ))

        vol_cfg = cfg["volume_breakout"]
        filters.append(VolumeBreakoutRule(
            lookback=vol_cfg["lookback"],
            min_ratio=vol_cfg["min_ratio"],
            max_ratio=vol_cfg["max_ratio"],
        ))

        confirmations = []
        mr_cfg = cfg["mean_reversion"]
        confirmations.append(MeanReversionRule(
            bb_period=mr_cfg["bb_period"],
            bb_std=mr_cfg["bb_std"],
            long_bb_threshold=mr_cfg["long_bb_threshold"],
            short_bb_threshold=mr_cfg["short_bb_threshold"],
            rsi_period=mr_cfg["rsi_period"],
            rsi_long_threshold=mr_cfg["rsi_long_threshold"],
            rsi_short_threshold=mr_cfg["rsi_short_threshold"],
        ))

        rej_cfg = cfg["rejection_pattern"]
        confirmations.append(RejectionPatternRule(
            min_wick_body_ratio=rej_cfg["min_wick_body_ratio"],
        ))

        self.aggregator = SignalAggregator(
            primary_rule=primary,
            filter_rules=filters,
            confirmation_rules=confirmations,
            min_confirmations=1,
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
        """Initialize TopstepX data fetcher (reuses ml_intraday_v3 infrastructure)."""
        try:
            # Import from ml_intraday_v3
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
            logger.info("Data fetcher initialized")
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
            return True  # Default to allowing if timezone fails

        t = now_et.time()
        session_start = pd.Timestamp(self.rules_cfg["time_of_day"]["session_start"]).time()
        session_end = pd.Timestamp(self.rules_cfg["time_of_day"]["session_end"]).time()
        return session_start <= t <= session_end

    def _process_bar(self, bar_time, latest_bar, bars_df):
        """Process a new bar through the rule engine."""
        self.risk_manager.tick_bar()

        # Check risk
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Risk block: {reason}")
            return

        # Evaluate rules
        decision = self.aggregator.evaluate(bars_df)
        logger.info(f"Signal: {decision.summary}")

        if not decision.should_trade:
            return

        # Compute ATR for stops
        current_atr = atr(bars_df["high"], bars_df["low"], bars_df["close"]).iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            logger.warning("Invalid ATR, skipping trade")
            return

        exit_cfg = self.rules_cfg["exit_strategy"]
        entry_price = latest_bar["close"]

        stop_loss = self.risk_manager.compute_stop_price(
            entry_price, decision.direction, current_atr, exit_cfg["stop_loss_atr"]
        )
        profit_target = self.risk_manager.compute_target_price(
            entry_price, decision.direction, current_atr, exit_cfg["profit_target_atr"]
        )

        direction_str = "LONG" if decision.direction == 1 else "SHORT"
        logger.info(
            f"TRADE SIGNAL: {direction_str} @ {entry_price:.2f}, "
            f"SL={stop_loss:.2f}, TP={profit_target:.2f}, ATR={current_atr:.2f}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Trade logged but not executed")
            return

        # Execute via TopstepX (if available)
        if self.execution_engine:
            try:
                success, exec_reason = self.execution_engine.execute_signal(
                    timestamp=bar_time,
                    direction=direction_str,
                    prediction={"confidence": decision.confidence},
                    bars_df=bars_df,
                    contracts=self.risk_manager.contracts,
                )
                if success:
                    logger.info(f"Order executed: {direction_str}")
                else:
                    logger.warning(f"Order rejected: {exec_reason}")
            except Exception as e:
                logger.error(f"Execution error: {e}", exc_info=True)

    def _signal_handler(self, signum, frame):
        logger.warning(f"Signal {signum} received - shutting down")
        self.running = False

    def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("RULE-BASED LIVE TRADING")
        logger.info("=" * 60)
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Contract: {self.contract_id}")

        if not self._init_data_fetcher():
            logger.error("Failed to initialize - aborting")
            return

        signal.signal(signal.SIGINT, self._signal_handler)
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

                # New day reset
                now_date = datetime.now().date()
                if current_date is not None and now_date != current_date:
                    self.risk_manager.reset_daily()
                    logger.info("New trading day - risk state reset")
                current_date = now_date

                # Poll for new bar
                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                latest_bar = self.data_fetcher.get_latest_bar()
                if latest_bar is not None and (
                    self.last_bar_time is None or latest_bar.name > self.last_bar_time
                ):
                    bar_time = latest_bar.name
                    logger.info(f"New bar: {bar_time}, close={latest_bar['close']:.2f}")

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
        logger.info(f"Session stats: {stats}")
