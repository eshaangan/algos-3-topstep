"""
Main live trading runner for ml_intraday_v3 strategy.

Orchestrates data fetching, feature generation, prediction, and execution.
"""

import importlib.util
import json
import logging
import os
import sys
import time
import signal
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def setup_dual_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging to both console and file.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    # Create log directory
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"live_trading_{timestamp}.log"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler (terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (save to file)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: terminal + file ({log_file})")

    return logger


# Will be initialized in LiveTradingRunner.__init__
logger = logging.getLogger(__name__)

# Add project root and ml_intraday_v3 to path
ml_v3_dir = Path(__file__).resolve().parents[1]
project_root = ml_v3_dir.parent
sys.path.insert(0, str(ml_v3_dir))
sys.path.insert(0, str(project_root))

# Use TopstepX REST API data fetcher (proven polling approach)
from live_trading.topstepx_rest_data_fetcher import TopstepXRestDataFetcher
from live_trading.feature_generator import LiveFeatureGenerator
from live_trading.model_predictor import LiveModelPredictor
from live_trading.execution_engine import LiveExecutionEngine
from monitoring.metrics_tracker import MetricsTracker
from monitoring.alerts import AlertManager, AlertLevel
from monitoring.dashboard import TerminalDashboard
try:
    from core.projectx_client import ProjectXClient
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    core_path = project_root / "core" / "projectx_client.py"
    spec = importlib.util.spec_from_file_location("projectx_client", core_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["projectx_client"] = module
    spec.loader.exec_module(module)
    ProjectXClient = module.ProjectXClient


def _compute_max_staleness_minutes(bar_size_minutes: int, grace_minutes: int = 2) -> float:
    """
    Maximum allowed staleness (minutes) for completed bars.

    Bar timestamps represent bar start time; a just-completed bar can appear
    close to bar_size minutes after its timestamp, and the next bar may not
    arrive until nearly 2 * bar_size. Add a small grace buffer.
    """
    return (2 * bar_size_minutes) + grace_minutes


def _ensure_contract_matches_expected(contract_id: str, expected_symbol: str, contracts: list[dict]) -> None:
    """
    Hard-fail if the resolved contract_id is not present in the search results
    for the expected symbol (e.g., MES). This prevents silent ES trades.
    """
    expected_symbol = expected_symbol.upper()
    ids = [c.get("id") for c in contracts]
    if contract_id not in ids:
        sample = contracts[:3]
        raise RuntimeError(
            f"Resolved contract_id {contract_id} not found in {expected_symbol} search results (sample={sample})"
        )


class LiveTradingRunner:
    """
    Main orchestrator for live trading.

    Runs the full trading loop:
    1. Fetch new bars
    2. Generate features
    3. Generate predictions
    4. Execute trades
    5. Monitor positions
    """

    def __init__(
        self,
        config_dir: Path,
        model_bundle_path: Optional[Path] = None,
        dry_run: bool = False,
        skip_confirmation: bool = False,
        log_level: str = "INFO",
        contract_id_override: Optional[str] = None,
        account_id_override: Optional[str] = None,
    ):
        """
        Initialize live trading runner.

        Args:
            config_dir: Path to configs directory
            model_bundle_path: Path to model bundle (if None, auto-detect latest)
            dry_run: If True, simulate trades without executing
            skip_confirmation: If True, skip manual confirmation prompt
        """
        self.config_dir = Path(config_dir)
        self.skip_confirmation = skip_confirmation
        self.dry_run = dry_run
        self.running = False

        # Setup dual logging (terminal + file)
        log_dir = Path("logs")
        global logger
        logger = setup_dual_logging(log_dir, log_level=log_level.upper())

        logger.info("=" * 80)
        logger.info("LIVE TRADING RUNNER - ML INTRADAY V3")
        logger.info("=" * 80)

        # Load configurations
        logger.info("Loading configurations...")
        self.live_cfg = self._load_config("live_trading.yaml")
        self.risk_cfg = self._load_config("risk.yaml")
        self.execution_spec = self._load_config("execution_spec.yaml")
        self.backtest_cfg = self._load_config("backtest.yaml")

        # Override dry_run if specified in config
        if self.live_cfg['trading'].get('dry_run', False):
            self.dry_run = True
            logger.warning("DRY RUN MODE enabled in config")

        # Load label schema
        # runs directory is at project root, not in ml_intraday_v3/
        runs_dir = Path(__file__).resolve().parents[2] / "runs"
        if model_bundle_path is None:
            # Auto-detect latest model
            bar_size = self.live_cfg['trading']['bar_size']
            model_bundle_path = LiveModelPredictor.find_latest_model(runs_dir, bar_size)
            if model_bundle_path is None:
                raise FileNotFoundError(f"No model bundle found in {runs_dir}")

        # Load label schema from model run directory
        run_dir = model_bundle_path.parents[3]  # Go up from window_*/walkforward/bar_size=1m
        bar_size = self.live_cfg['trading']['bar_size']
        label_schema_path = run_dir / f"bar_size={bar_size}" / "label_schema.json"
        with open(label_schema_path) as f:
            self.label_schema = json.load(f)

        # Initialize components
        logger.info("Initializing components...")

        # Data fetcher (TopstepX REST API polling - proven approach)
        # Extract bar size as minutes (e.g., "5m" -> 5)
        bar_size_str = self.live_cfg['trading']['bar_size']
        self.bar_size_minutes = int(bar_size_str.replace('m', ''))

        resolved_contract_id = contract_id_override or self.live_cfg["topstep"]["contract_id"]
        account_env_var = self.live_cfg["topstep"].get("account_id_env_var", "TOPSTEPX_ACCOUNT_ID")
        resolved_account_id = account_id_override or os.getenv(account_env_var)
        self.resolved_contract_id = resolved_contract_id
        self.resolved_account_id = resolved_account_id

        self.data_fetcher = TopstepXRestDataFetcher(
            contract_id=resolved_contract_id,
            bar_size_minutes=self.bar_size_minutes,
            lookback_bars=self.live_cfg['data']['lookback_bars'],
            enable_rth_filter=self.live_cfg['data'].get('enable_rth_filter', True),  # Filter to RTH only
        )

        # Model predictor
        logger.info(f"Loading model: {model_bundle_path}")
        self.predictor = LiveModelPredictor(model_bundle_path)
        model_info = self.predictor.get_model_info()
        logger.info(f"Model: {model_info['model_type']}, "
                   f"features={model_info['n_features']}, "
                   f"threshold={model_info['primary_threshold']}")

        # Feature generator (reuse offline features.yaml for parity)
        self.feature_generator = LiveFeatureGenerator(
            feature_columns=self.predictor.feature_columns,
            bar_size=bar_size_str,
            features_config_path=self.config_dir / "features.yaml",
        )

        # Execution engine
        self.execution_engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=self.dry_run,
            contract_id=resolved_contract_id,
            account_id=resolved_account_id,
            order_type=self.live_cfg["topstep"].get("order", {}).get("type", "MARKET"),
        )

        # State tracking
        self.last_bar_time = None
        self.signals_generated = 0
        self.trades_executed = 0

        # Initialize Kelly sizer if enabled
        self.kelly_sizer = None
        if self.live_cfg.get('kelly_sizing', {}).get('enabled', False):
            from live_trading.kelly_sizer import KellySizer
            self.kelly_sizer = KellySizer(self.live_cfg['kelly_sizing'])
            logger.info(f"Kelly sizing enabled: fraction={self.kelly_sizer.config['kelly_fraction']}")
        else:
            logger.info("Kelly sizing disabled - using fixed contracts")

        # Initialize monitoring
        logs_dir = Path("logs")
        self.metrics_tracker = MetricsTracker(logs_dir)
        
        # Load previous session's trade history for Kelly persistence
        # This allows Kelly to continue from previous statistics on restart
        if self.kelly_sizer and self.kelly_sizer.config.get('enabled', False):
            latest_trade_file = self.metrics_tracker.find_latest_trade_file(logs_dir)
            if latest_trade_file:
                trades_loaded = self.metrics_tracker.load_trades_from_csv(latest_trade_file)
                if trades_loaded > 0:
                    logger.info(f"Kelly persistence: Loaded {trades_loaded} trades from previous session")
                    logger.info(f"Kelly will continue from trade #{trades_loaded + 1}")
                else:
                    logger.info("Kelly will start fresh (no previous trades loaded)")
            else:
                logger.info("Kelly will start fresh (no previous trade files found)")
        
        self.alert_manager = AlertManager(logs_dir, enable_sound=True)
        self.dashboard = TerminalDashboard()

        # Set starting equity
        self.metrics_tracker.set_starting_equity(self.execution_engine.get_equity())

        # Update frequency for dashboard (seconds)
        self.dashboard_update_interval = 10  # Update every 10 seconds

        logger.info("Initialization complete")

    def _load_config(self, filename: str) -> dict:
        """Load YAML config file."""
        path = self.config_dir / filename
        with open(path) as f:
            return yaml.safe_load(f)

    def startup_checks(self) -> bool:
        """
        Run startup health checks.

        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STARTUP CHECKS")
        logger.info("=" * 80)

        checks = {
            'data_connection': False,
            'api_connection': False,
            'model_loaded': False,
            'risk_config_valid': False,
        }

        # Check data connection (REST API)
        logger.info("Testing data connection...")
        try:
            # Connect to TopstepX API
            self.data_fetcher.connect()
            # Initialize buffer with historical bars
            self.data_fetcher.initialize_buffer()
            buffer = self.data_fetcher.get_buffer()
            checks['data_connection'] = not buffer.empty
            if checks['data_connection']:
                # Treat the last warmup bar as already processed so we only act on the
                # next *new* bar after startup.
                self.last_bar_time = buffer.index[-1]
                logger.info(
                    f"Warmup buffer loaded; will wait for the next bar after {self.last_bar_time} before trading"
                )
        except Exception as e:
            logger.error(f"Data connection failed: {e}")

        # Check API connection
        logger.info("Testing API connection...")
        try:
            checks['api_connection'] = self.execution_engine.check_api_connection()
        except Exception as e:
            logger.error(f"API connection failed: {e}")

        # Check model loaded
        checks['model_loaded'] = self.predictor.model is not None

        # Check risk config
        checks['risk_config_valid'] = (
            self.risk_cfg.get('topstep', {}).get('starting_balance', 0) > 0
        )

        # Display results
        logger.info("")
        logger.info("Startup Check Results:")
        for check, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {check}: {status}")

        all_passed = all(checks.values())

        if all_passed:
            logger.info("")
            logger.info("✓ All startup checks passed")
        else:
            logger.error("")
            logger.error("✗ Some startup checks failed")

        logger.info("=" * 80)

        return all_passed

    def _check_buffer_health(self) -> bool:
        """
        Check if data buffer is healthy.

        Returns:
            True if buffer is healthy, False otherwise
        """
        bars = self.data_fetcher.get_buffer()

        if len(bars) < 100:
            logger.warning(f"Buffer only has {len(bars)} bars (need 100)")
            return False

        # Check for recent data
        last_bar_time = bars.index[-1]
        now = pd.Timestamp.now(tz='America/Chicago')
        staleness = (now - last_bar_time).total_seconds() / 60

        # Allow up to ~2 bars of staleness + small grace (timestamps are bar start)
        max_staleness = _compute_max_staleness_minutes(self.bar_size_minutes)
        if staleness > max_staleness:
            logger.warning(f"Data is stale: {staleness:.1f} minutes old (threshold: {max_staleness}m)")
            return False

        # Check for gaps
        time_diffs = bars.index.to_series().diff().dt.total_seconds() / 60
        # Gap if difference is significantly larger than bar size (e.g. 1.5x)
        gap_threshold = self.bar_size_minutes * 1.5
        gaps = time_diffs[time_diffs > gap_threshold]

        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} gaps in buffer (threshold: {gap_threshold}m)")

        return True

    def _validate_contract_identity(self):
        """
        Ensure the resolved contract_id maps to the expected symbol (e.g., MES).
        Hard-fails if there is a mismatch to avoid ES trades.
        """
        expected_symbol = self.live_cfg["data"].get("symbol", "").upper()
        if not expected_symbol:
            logger.warning("No expected symbol configured; skipping contract validation")
            return

        live_flag = self.live_cfg["trading"].get("environment", "paper").lower() == "live"
        try:
            client = ProjectXClient(
                contract_id=self.resolved_contract_id,
                account_id=self.resolved_account_id,
            )
            contracts = client.search_contracts(search_text=expected_symbol, live=live_flag)
        except Exception as e:
            logger.error(f"Contract validation failed: {e}")
            raise

        _ensure_contract_matches_expected(self.resolved_contract_id, expected_symbol, contracts)
        logger.info(f"Contract sanity check passed: {self.resolved_contract_id} is {expected_symbol}")

    def run(self):
        """
        Main trading loop.

        Runs continuously, fetching data, generating signals, and executing trades.
        """
        # Run startup checks
        if not self.startup_checks():
            logger.error("Startup checks failed - aborting")
            return

        # Hard-fail if the resolved contract is not the expected symbol (e.g., MES)
        self._validate_contract_identity()

        # Require manual confirmation
        if self.live_cfg['startup'].get('require_manual_confirmation', True):
            logger.info("")
            if not self.skip_confirmation:
                logger.info("=" * 80)
                logger.info("MANUAL CONFIRMATION REQUIRED")
                logger.info("=" * 80)
                logger.info(f"Environment: {self.live_cfg['trading']['environment']}")
                logger.info(f"Dry Run: {self.dry_run}")
                logger.info(f"Account: {self.resolved_account_id or os.getenv('TOPSTEPX_ACCOUNT_ID')}")
                logger.info("")

                response = input("Start trading? (type 'yes' to confirm): ")
                if response.lower() != 'yes':
                    logger.warning("Trading cancelled by user")
                    return
            else:
                logger.info("Skipping confirmation (--no-confirm flag set)")
                logger.info(f"Environment: {self.live_cfg['trading']['environment']}")
                logger.info(f"Dry Run: {self.dry_run}")
                logger.info(f"Account: {self.resolved_account_id or os.getenv('TOPSTEPX_ACCOUNT_ID')}")

        # Buffer already initialized in startup checks
        logger.info("TopstepX REST API ready for polling")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        logger.info("")
        logger.info("=" * 80)
        logger.info("LIVE TRADING STARTED")
        logger.info("=" * 80)

        # Send session start alert
        self.alert_manager.session_started(
            account_id=self.resolved_account_id or os.getenv('TOPSTEPX_ACCOUNT_ID', 'unknown'),
            starting_equity=self.execution_engine.get_equity()
        )

        # Main loop
        update_interval = self.live_cfg['data']['update_frequency_seconds']
        last_dashboard_update = time.time()

        while self.running:
            try:
                # Check if we're in trading session
                if not self._is_trading_time():
                    logger.debug("Outside trading hours - sleeping")
                    time.sleep(60)
                    continue

                # Poll for a new completed bar and update buffer (REST fetcher path)
                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                # Check buffer health every iteration
                if not self._check_buffer_health():
                    logger.error("Buffer health check failed, skipping this iteration")
                    time.sleep(60)
                    continue

                # Get latest completed bar from buffer
                latest_bar = self.data_fetcher.get_latest_bar()

                if latest_bar is not None and (self.last_bar_time is None or latest_bar.name > self.last_bar_time):
                    # New bar available
                    bar_time = latest_bar.name
                    logger.info(f"New bar: {bar_time}, close={latest_bar['close']:.2f}")

                    # Note: Buffer is automatically updated by TopstepX data fetcher

                    # Update positions first (check for exits)
                    closed = self.execution_engine.update_positions(
                        current_time=bar_time,
                        current_bar=latest_bar,
                    )
                    if closed:
                        logger.info(f"Closed {len(closed)} positions")

                        # Record closed trades in metrics
                        for pos in closed:
                            self.metrics_tracker.record_trade(
                                entry_time=pos['entry_ts'],
                                exit_time=pos['exit_ts'],
                                direction=pos['direction'],
                                contracts=pos['contracts'],
                                entry_price=pos['entry_price'],
                                exit_price=pos['exit_price'],
                                pnl=pos['pnl_usd'],
                                exit_reason=pos['exit_reason'],
                            )

                            # Send alert
                            self.alert_manager.trade_closed(
                                direction=pos['direction'],
                                pnl=pos['pnl_usd'],
                                exit_reason=pos['exit_reason']
                            )

                    # Generate signal
                    self._process_bar(bar_time, latest_bar)

                    self.last_bar_time = bar_time

                # Update metrics from execution engine
                status = self.execution_engine.get_status()
                logger.debug(
                    "Broker status: equity=%.2f daily_pnl=%.2f open_positions=%s",
                    status['equity'],
                    status['daily_pnl'],
                    status['open_positions'],
                )
                self.metrics_tracker.update_equity(status['equity'], status['daily_pnl'])
                self.metrics_tracker.update_positions(status['open_positions'])

                # Check for risk warnings
                if status['daily_pnl'] < 0 and abs(status['daily_pnl']) > 1500:  # 75% of $2000 limit
                    self.alert_manager.daily_loss_warning(status['daily_pnl'], 2000)

                if status['drawdown'] > 1875:  # 75% of $2500 limit
                    self.alert_manager.drawdown_warning(status['drawdown'], 2500)

                # Update dashboard periodically
                current_time = time.time()
                if current_time - last_dashboard_update >= self.dashboard_update_interval:
                    metrics = self.metrics_tracker.snapshot()
                    alerts_summary = self.alert_manager.get_summary()
                    self.dashboard.render(metrics, alerts_summary)
                    last_dashboard_update = current_time

                    # Save metrics snapshot
                    self.metrics_tracker.save_snapshot()
                    self.metrics_tracker.save_signal_log()
                    if self.kelly_sizer:
                        self.metrics_tracker.save_kelly_log()

                # Sleep until next update
                time.sleep(update_interval)

            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt - shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(update_interval)

        # Shutdown
        self._shutdown()

    def _process_bar(self, bar_time: pd.Timestamp, latest_bar: pd.Series):
        """
        Process a new bar: generate features, predict, and potentially execute.

        Args:
            bar_time: Bar timestamp
            latest_bar: Latest bar data
        """
        # Get rolling buffer
        bars_df = self.data_fetcher.get_buffer()

        # Generate features
        features = self.feature_generator.generate_features(bars_df)

        # Check feature quality
        quality = self.feature_generator.check_feature_quality(features)
        if not quality['healthy']:
            logger.warning(f"Feature quality issues: {quality}")
            if self.live_cfg['health']['check_feature_quality']:
                logger.warning("Skipping trade due to feature quality issues")
                return

        # Generate prediction
        use_meta = self.live_cfg['signals'].get('use_meta_model', False)
        prediction = self.predictor.predict(features, use_meta=use_meta)

        score = prediction.get('score_ev', 0.0)
        logger.info(
            f"Signal generated: score={score:.3f}, "
            f"p_target={prediction.get('p_target', 0.0):.3f}, "
            f"p_stop={prediction.get('p_stop', 0.0):.3f}"
        )

        self.signals_generated += 1

        # Check if we should trade
        should_trade, reason = self.predictor.should_trade(
            prediction=prediction,
            primary_threshold=self.live_cfg['signals']['primary_threshold'],
            meta_threshold=self.live_cfg['signals'].get('meta_threshold'),
            require_meta_approval=self.live_cfg['signals'].get('require_meta_approval', False),
        )

        if not should_trade:
            logger.info(f"✗ Signal rejected: score={score:.3f}, reason={reason}")
            
            # Record rejected signal
            self.metrics_tracker.record_signal(
                executed=False,
                score=score,
                timestamp=bar_time,
                direction=None,
                reason=reason,
            )
            return

        # Determine direction (for now, always LONG based on positive score)
        # TODO: Add regime detection or direction logic
        direction = "LONG" if prediction['score_ev'] > 0 else "SHORT"

        # Get trade history for Kelly calculation
        trade_history = self.metrics_tracker.trade_history

        # Execute trade (Kelly sizer will override contracts if enabled)
        contracts = self.live_cfg['positions']['contracts_per_trade']
        success, exec_reason = self.execution_engine.execute_signal(
            timestamp=bar_time,
            direction=direction,
            prediction=prediction,
            bars_df=bars_df,
            contracts=contracts,
            kelly_sizer=self.kelly_sizer,  # NEW
            trade_history=trade_history,   # NEW
        )

        if success:
            self.trades_executed += 1
            logger.info(f"✓ Trade executed: {direction} {contracts} contracts, score={score:.3f}")

            # Record signal execution
            self.metrics_tracker.record_signal(
                executed=True,
                score=score,
                timestamp=bar_time,
                direction=direction,
                reason="executed",
            )

            # Send alert
            self.alert_manager.trade_executed(
                direction=direction,
                contracts=contracts,
                price=bars_df.iloc[-1]['close'],
                prediction_score=score
            )
        else:
            logger.warning(f"✗ Trade rejected by execution engine: score={score:.3f}, reason={exec_reason}")

            # Record signal rejection
            self.metrics_tracker.record_signal(
                executed=False,
                score=score,
                timestamp=bar_time,
                direction=direction,
                reason=exec_reason,
            )

    def _is_trading_time(self) -> bool:
        """
        Check if current time (America/Chicago) is within any configured session,
        supporting sessions that span midnight (e.g., ETH 17:00-16:00 next day).
        """
        tz = ZoneInfo("America/Chicago")
        now_ct = datetime.now(tz)
        buffer_minutes = self.live_cfg['session']['no_entry_before_close_minutes']
        buffer_delta = timedelta(minutes=buffer_minutes)

        sessions = self.live_cfg['session']['sessions']
        for session in sessions:
            start_time = datetime.strptime(session['start_time'], "%H:%M").time()
            end_time = datetime.strptime(session['end_time'], "%H:%M").time()

            start_dt = datetime.combine(now_ct.date(), start_time, tzinfo=tz)
            end_dt = datetime.combine(now_ct.date(), end_time, tzinfo=tz)

            # Handle sessions that wrap past midnight (end < start)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            # Check both today's window and the prior day's window (for overnight coverage)
            intervals = [
                (start_dt, end_dt),
                (start_dt - timedelta(days=1), end_dt - timedelta(days=1)),
            ]

            for window_start, window_end in intervals:
                if window_start <= now_ct <= window_end:
                    if now_ct >= window_end - buffer_delta:
                        logger.debug("Within no-entry window before close")
                        return False
                    return True

        return False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum} - shutting down")
        self.running = False

    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 80)

        # Flatten all open positions
        if self.execution_engine.open_positions:
            logger.warning("Flattening open positions...")
            bars_df = self.data_fetcher.get_buffer()
            if not bars_df.empty:
                current_bar = bars_df.iloc[-1]
                self.execution_engine.flatten_all_positions(
                    current_time=pd.Timestamp.now(),
                    current_price=current_bar['close'],
                    reason="shutdown",
                )

        # Display final stats
        status = self.execution_engine.get_status()
        final_metrics = self.metrics_tracker.snapshot()

        # Send session end alert
        self.alert_manager.session_ended(
            total_trades=final_metrics['total_trades'],
            total_pnl=final_metrics['total_pnl'],
            final_equity=status['equity']
        )

        # Display summary dashboard
        self.dashboard.display_summary(final_metrics)

        logger.info("")
        logger.info("Final Status:")
        logger.info(f"  Signals generated: {self.signals_generated}")
        logger.info(f"  Trades executed: {self.trades_executed}")
        logger.info(f"  Final equity: ${status['equity']:,.2f}")
        logger.info(f"  Daily P&L: ${status['daily_pnl']:,.2f}")
        logger.info(f"  Drawdown: ${status['drawdown']:,.2f}")
        logger.info(f"  Trades today: {status['trades_today']}")
        logger.info(f"  {self.alert_manager.get_summary()}")

        # Save metrics and trades
        self.metrics_tracker.save_trades()
        self.metrics_tracker.save_snapshot()

        # Save trade log
        if self.execution_engine.trade_log:
            log_path = Path("logs") / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            log_path.parent.mkdir(exist_ok=True)
            pd.DataFrame(self.execution_engine.trade_log).to_csv(log_path, index=False)
            logger.info(f"Trade log saved: {log_path}")

        # Disconnect from market data
        logger.info("Disconnecting from market data...")
        self.data_fetcher.disconnect()

        logger.info("")
        logger.info("Shutdown complete")
        logger.info("=" * 80)

        # Flush and close log handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                logger.info(f"Session log saved: {handler.baseFilename}")
                handler.close()


def main():
    """Entry point for live trading."""
    import argparse

    # Calculate default config dir relative to this file
    # This file is in ml_intraday_v3/live_trading/live_runner.py
    # We want ml_intraday_v3/configs
    default_config_dir = Path(__file__).resolve().parents[1] / "configs"

    parser = argparse.ArgumentParser(description="Live trading runner for ml_intraday_v3")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Path to configs directory",
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=None,
        help="Path to model bundle (if not specified, auto-detect latest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (simulate trades without executing)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip manual confirmation prompt",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--contract-id",
        type=str,
        default=None,
        help="Override contract id (defaults to live_trading.yaml)",
    )
    parser.add_argument(
        "--account-id",
        type=str,
        default=None,
        help="Override account id (defaults to environment)",
    )

    args = parser.parse_args()

    # Create runner
    runner = LiveTradingRunner(
        config_dir=args.config_dir,
        model_bundle_path=args.model_bundle,
        dry_run=args.dry_run,
        skip_confirmation=args.no_confirm,
        log_level=args.log_level,
        contract_id_override=args.contract_id,
        account_id_override=args.account_id,
    )

    # Start trading
    runner.run()


if __name__ == "__main__":
    main()
