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
from live_trading.risk_manager import RiskManager as LiveRiskManager
from live_trading.execution_engine import LiveExecutionEngine
from live_trading.event_detector import LiveEventDetector
from monitoring.metrics_tracker import MetricsTracker
from monitoring.alerts import AlertManager, AlertLevel
from monitoring.dashboard import TerminalDashboard
# Phase 2a Filter Integrations
from filters.confidence_filter import apply_confidence_filter
from filters.regime_filter import RegimeDetector
from monitoring.adaptive_circuit_breaker import AdaptiveCircuitBreaker
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
        config_name: str = "live_trading.yaml",
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
        self.live_cfg = self._load_config(config_name)
        self.risk_cfg = self._load_config("risk.yaml")
        self.execution_spec = self._load_config("execution_spec.yaml")
        self.backtest_cfg = self._load_config("backtest.yaml")
        self.risk_cfg_source_path = self.config_dir / "risk.yaml"

        # Allow selecting a different risk config file via live_trading.yaml.
        # This keeps defaults backward compatible while enabling Topstep-specific variants.
        risk_cfg_path = (self.live_cfg.get("risk", {}) or {}).get("risk_config_path")
        if risk_cfg_path:
            # Mirror _load_config_any_path resolution logic so the LiveRiskManager reads the same file.
            candidate = Path(risk_cfg_path)
            if not candidate.is_absolute():
                candidate_dir = self.config_dir / risk_cfg_path
                if candidate_dir.exists():
                    candidate = candidate_dir
                else:
                    candidate_root = project_root / risk_cfg_path
                    if candidate_root.exists():
                        candidate = candidate_root
            self.risk_cfg_source_path = candidate
            self.risk_cfg = self._load_config_any_path(risk_cfg_path)
            logger.info(f"Loaded risk config override: {risk_cfg_path}")

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
        # Robustly find run_dir by looking for label_schema.json
        bar_size = self.live_cfg['trading']['bar_size']
        current_path = model_bundle_path.parent
        run_dir = None
        
        # Search up to 5 levels up
        for _ in range(5):
            candidate = current_path / f"bar_size={bar_size}" / "label_schema.json"
            if candidate.exists():
                run_dir = current_path
                break
            # Also check if we are already in the bar_size dir
            if (current_path / "label_schema.json").exists():
                # If we found it directly, we need to deduce run_dir. 
                # run_dir is expected to be the parent of bar_size=...
                # If we are in bar_size=..., then parent is run_dir
                run_dir = current_path.parent
                break
            
            if current_path == current_path.parent: # Root reached
                break
            current_path = current_path.parent
            
        if run_dir is None:
            logger.warning(f"Could not locate label_schema.json relative to {model_bundle_path}. Defaulting run_dir to project root.")
            run_dir = Path(".")

        # Store run_dir and bar_size for later use (volatility filter, regime adjustment)
        self.run_dir = run_dir
        self.bar_size = bar_size

        label_schema_path = run_dir / f"bar_size={bar_size}" / "label_schema.json"
        if not label_schema_path.exists():
             # Fallback: try finding it anywhere in the run_dir or just assume it's where we found it
             # Attempt to locate it one last time if the loop above found it differently
             pass

        if label_schema_path.exists():
            with open(label_schema_path) as f:
                self.label_schema = json.load(f)
        else:
             logger.warning(f"Label schema not found at {label_schema_path}. Using empty schema.")
             self.label_schema = {}

        # Initialize components
        logger.info("Initializing components...")

        # Live risk manager (Topstep circuit breakers)
        self.use_risk_manager = self.live_cfg.get('risk', {}).get('use_risk_manager', False)
        self.live_risk_manager = None
        if self.use_risk_manager:
            self.live_risk_manager = LiveRiskManager(self.risk_cfg_source_path)
            logger.info("Live risk manager enabled")
        else:
            logger.info("Live risk manager disabled")

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

        # Event detector (CUSUM filter to match training event generation)
        event_cfg = self.live_cfg['signals'].get('event_filter', {})
        self.event_detector = None
        if event_cfg.get('enabled', True):  # Default: enabled
            atr_period = event_cfg.get('atr_period', 14)
            cusum_mult = event_cfg.get('cusum_threshold_atr_mult', 0.8)

            # Load adaptive min threshold from training data analysis
            min_cusum_threshold = event_cfg.get('min_cusum_threshold', None)
            if min_cusum_threshold == "adaptive":
                # Load from training run's atr_analysis.json
                atr_analysis_path = self.run_dir / f"bar_size={self.bar_size}" / "atr_analysis.json"
                if atr_analysis_path.exists():
                    with open(atr_analysis_path) as f:
                        atr_analysis = json.load(f)
                    min_cusum_threshold = atr_analysis['recommendations']['min_cusum_threshold']
                    logger.info(f"Loaded adaptive min_cusum_threshold: {min_cusum_threshold:.2f} from training data")
                else:
                    logger.warning(f"Adaptive threshold requested but {atr_analysis_path} not found, using 6.0")
                    min_cusum_threshold = 6.0

            self.event_detector = LiveEventDetector(
                atr_period=atr_period,
                cusum_threshold_atr_mult=cusum_mult,
                min_cusum_threshold=min_cusum_threshold,
            )
            logger.info(f"Event filter enabled: CUSUM with atr_period={atr_period}, mult={cusum_mult}")
        else:
            logger.warning("Event filter DISABLED - predicting on every bar (train/test mismatch!)")

        # Execution engine
        self.execution_engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=self.dry_run,
            contract_id=resolved_contract_id,
            account_id=resolved_account_id,
            config=self.live_cfg,  # Pass live trading config for direction_change settings
        )
        if self.live_risk_manager:
            self.live_risk_manager.sync_equity(self.execution_engine.get_equity())

        # Phase 2a: Initialize filters
        logger.info("Initializing Phase 2a safety filters...")
        
        # Confidence filter configuration
        self.confidence_cfg = self.execution_spec['filters'].get('confidence', {})
        self.confidence_enabled = self.confidence_cfg.get('enabled', False)
        self.confidence_threshold = self.confidence_cfg.get('min_probability_distance', 0.55)
        if self.confidence_enabled:
            logger.info(f"✓ Confidence filter enabled: threshold={self.confidence_threshold:.2f}")
        
        # Adaptive circuit breaker configuration  
        self.circuit_breaker_cfg = self.live_cfg.get('circuit_breaker', {})
        self.circuit_breaker_enabled = self.circuit_breaker_cfg.get('enabled', False)
        self.circuit_breaker = None
        if self.circuit_breaker_enabled:
            self.circuit_breaker = AdaptiveCircuitBreaker(
                consecutive_losses_limit=self.circuit_breaker_cfg.get('consecutive_losses', 3),
                cooling_off_minutes=self.circuit_breaker_cfg.get('cooling_off_minutes', 30),
                temp_confidence_boost=self.circuit_breaker_cfg.get('temp_confidence_boost', 0.10),
                temp_position_reduction=self.circuit_breaker_cfg.get('temp_position_reduction', 0.5),
                daily_loss_limit=self.circuit_breaker_cfg.get('daily_loss_limit', -500.0),
                base_confidence_threshold=self.confidence_threshold
            )
            logger.info(f"✓ Adaptive circuit breaker enabled: daily_loss_limit=${self.circuit_breaker_cfg.get('daily_loss_limit', -500.0):.0f}")
        
        # Regime detector configuration (will be initialized in run() after loading training data)
        self.regime_detector_cfg = self.live_cfg.get('regime_detector', {})
        self.regime_detector_enabled = self.regime_detector_cfg.get('enabled', False)
        self.regime_detector = None
        self.regime_safe = True  # Assume safe until proven otherwise
        if self.regime_detector_enabled:
            logger.info(f"✓ Regime detector will be initialized after loading training data")
        
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

    def _load_config_any_path(self, path_str: str) -> dict:
        """
        Load a YAML config from an absolute path or a project-relative path.

        Priority for relative paths:
          1) path relative to this config_dir
          2) path relative to project root
          3) path as provided
        """
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate_dir = self.config_dir / path_str
            if candidate_dir.exists():
                candidate = candidate_dir
            else:
                candidate_root = project_root / path_str
                if candidate_root.exists():
                    candidate = candidate_root
        if not candidate.exists():
            raise FileNotFoundError(f"Config not found: {path_str} (resolved={candidate})")
        with open(candidate) as f:
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
        
        # Phase 2a: Initialize regime detector with training data
        if self.regime_detector_enabled:
            logger.info("Initializing regime detector with training data...")
            try:
                # Get buffer for feature calculation
                bars_df = self.data_fetcher.get_buffer()
                
                # Generate features on historical data
                historical_features = self.feature_generator.generate_features(bars_df)
                
                # Initialize regime detector
                self.regime_detector = RegimeDetector(
                    feature_cols=self.predictor.feature_columns,
                    reference_window_days=self.regime_detector_cfg.get('reference_window_days', 90),
                    current_window_bars=self.regime_detector_cfg.get('current_window_bars', 100),
                    max_shifted_features_pct=self.regime_detector_cfg.get('max_shifted_pct', 0.30)
                )
                
                # Fit on historical features (buffer contains last N days)
                self.regime_detector.fit(historical_features)
                logger.info("✅ Regime detector initialized and fitted on training data")
            except Exception as e:
                logger.error(f"Failed to initialize regime detector: {e}", exc_info=True)
                logger.warning("Continuing without regime detection")
                self.regime_detector_enabled = False

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
                            if self.live_risk_manager:
                                self.live_risk_manager.update_pnl(pos['pnl_usd'])

                            # Send alert
                            self.alert_manager.trade_closed(
                                direction=pos['direction'],
                                pnl=pos['pnl_usd'],
                                exit_reason=pos['exit_reason']
                            )
                            
                            # Phase 2a: Check adaptive circuit breaker after each trade
                            if self.circuit_breaker_enabled and self.circuit_breaker is not None:
                                # Get current status for daily P&L
                                status = self.execution_engine.get_status()
                                daily_pnl = status['daily_pnl']
                                
                                # Create trade result dict
                                trade_result = {
                                    'pnl': pos['pnl_usd'],
                                    'timestamp': pos['exit_ts'],
                                    'symbol': 'MES'
                                }
                                
                                # Check and adapt
                                action = self.circuit_breaker.check_and_adapt(
                                    trade_result=trade_result,
                                    daily_pnl=daily_pnl,
                                    current_time=pd.Timestamp.now()
                                )
                                
                                if action == 'stop_today':
                                    logger.critical("🚨 CIRCUIT BREAKER: Stopping trading for today")
                                    self.running = False
                                    # Send critical alert
                                    self.alert_manager.alert(
                                        level=AlertLevel.CRITICAL,
                                        message=f"Circuit breaker stopped trading: Daily P&L ${daily_pnl:.2f}",
                                        category="circuit_breaker"
                                    )
                                    return
                                elif action == 'cooling_off':
                                    logger.warning("⚠️ Circuit breaker: Entering cooling-off period")
                                    cb_status = self.circuit_breaker.get_status()
                                    logger.warning(f"   Will resume at {cb_status['cooling_off_until']}")
                                    logger.warning(f"   New threshold: {cb_status['current_threshold']:.2f}")
                                elif action == 'adapted':
                                    cb_status = self.circuit_breaker.get_status()
                                    logger.info(f"📊 Circuit breaker adapted:")
                                    logger.info(f"   Threshold: {cb_status['current_threshold']:.2f}")
                                    logger.info(f"   Position multiplier: {cb_status['current_position_multiplier']:.2f}")

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
                if self.live_risk_manager:
                    self.live_risk_manager.sync_state(status.get("equity"), status.get("daily_pnl"))
                self.metrics_tracker.update_equity(status['equity'], status['daily_pnl'])
                self.metrics_tracker.update_positions(status['open_positions'])

                # Hard circuit-breaker: if we are at a Topstep hard limit and still have exposure, flatten.
                if self.live_risk_manager and status.get("open_positions", 0):
                    allowed, reason = self.live_risk_manager.check_trading_allowed()
                    if not allowed and reason in {"daily_loss_limit", "trailing_drawdown_limit"}:
                        try:
                            current_price = float(latest_bar["close"]) if latest_bar is not None else None
                        except Exception:
                            current_price = None
                        if current_price is not None:
                            self.execution_engine.flatten_all_positions(
                                current_time=pd.Timestamp.now(tz="UTC"),
                                current_price=current_price,
                                reason=f"risk_{reason}",
                            )

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

    def _compute_atr_for_filter(self, bars_df: pd.DataFrame, period: int = 14) -> float:
        """
        Compute ATR for volatility filtering (matches event detector logic).

        Args:
            bars_df: DataFrame with OHLC data
            period: ATR lookback period

        Returns:
            Current ATR value (or NaN if insufficient data)
        """
        import numpy as np

        if len(bars_df) < period + 1:
            return np.nan

        df = bars_df.tail(period + 1).copy()
        prev_close = df["close"].shift(1)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean().iloc[-1]

        return float(atr)

    def _process_bar(self, bar_time: pd.Timestamp, latest_bar: pd.Series):
        """
        Process a new bar: generate features, predict, and potentially execute.

        Args:
            bar_time: Bar timestamp
            latest_bar: Latest bar data
        """
        # Get rolling buffer
        bars_df = self.data_fetcher.get_buffer()
        
        # Phase 2a: Check regime detector FIRST (before generating features)
        # Check every 100 bars to avoid excessive computation
        if self.regime_detector_enabled and self.regime_detector is not None:
            bar_count = len(bars_df)
            if bar_count % 100 == 0:  # Periodic check
                try:
                    # Get recent features for regime check
                    current_features = self.feature_generator.generate_features(bars_df)
                    
                    is_safe, shift_pct, shifted = self.regime_detector.detect_shift(current_features)
                    
                    if not is_safe:
                        self.regime_safe = False
                        logger.warning(f"⚠️ REGIME SHIFT DETECTED: {shift_pct:.1%} of features shifted")
                        logger.warning(f"   PAUSING TRADING until regime stabilizes")
                        logger.warning(f"   Top shifted features: {[f['feature'] for f in shifted[:5]]}")
                        
                        # Record regime shift event
                        self.metrics_tracker.record_signal(
                            executed=False,
                            score=0.0,
                            timestamp=bar_time,
                            direction=None,
                            reason=f"regime_shift_{shift_pct:.1%}",
                        )
                        return
                    else:
                        if not self.regime_safe:
                            logger.info(f"✅ Regime stabilized: {shift_pct:.1%} features shifted (below {self.regime_detector.max_shifted_features_pct:.1%} threshold)")
                        self.regime_safe = True
                except Exception as e:
                    logger.error(f"Regime detector error: {e}", exc_info=True)
        
        # Skip if regime unsafe
        if self.regime_detector_enabled and not self.regime_safe:
            logger.debug("Skipping bar due to regime shift")
            return

        # Volatility filter: check ATR before CUSUM event detection
        vol_filter_cfg = self.live_cfg['signals'].get('volatility_filter', {})
        if vol_filter_cfg.get('enabled', False):
            import numpy as np

            # Compute current ATR
            atr = self._compute_atr_for_filter(bars_df)
            min_atr = vol_filter_cfg.get('min_atr', 0.0)

            # Load adaptive min_atr if configured
            if min_atr == "adaptive":
                atr_analysis_path = self.run_dir / f"bar_size={self.bar_size}" / "atr_analysis.json"
                if atr_analysis_path.exists():
                    with open(atr_analysis_path) as f:
                        atr_analysis = json.load(f)
                    min_atr = atr_analysis['recommendations']['min_atr_balanced']
                else:
                    min_atr = 10.0  # Fallback

            if np.isfinite(atr) and atr < min_atr:
                logger.info(f"✗ Volatility filter: ATR={atr:.2f} < min_atr={min_atr:.2f}, skipping bar")
                self.metrics_tracker.record_signal(
                    executed=False,
                    score=0.0,
                    timestamp=bar_time,
                    direction=None,
                    reason=f"volatility_too_low (atr={atr:.2f})",
                )
                return

        # Event detection: only predict on CUSUM events (matching training)
        if self.event_detector is not None:
            is_event, event_info = self.event_detector.is_event(
                bars_df=bars_df,
                current_bar_close=float(latest_bar['close'])
            )

            if not is_event:
                logger.debug(
                    f"Not a CUSUM event: s_pos={event_info.get('s_pos', 0):.2f}, "
                    f"s_neg={event_info.get('s_neg', 0):.2f}, "
                    f"threshold={event_info.get('threshold', 0):.2f}"
                )
                return  # Skip prediction on non-events

            # Log event details
            logger.info(
                f"✓ CUSUM event detected ({event_info['event_type']}): "
                f"threshold={event_info['threshold']:.2f}, "
                f"price_diff={event_info.get('price_diff', 0):.2f}"
            )

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
        
        # Phase 2a: Apply confidence filter (QUICK WIN #1)
        if self.confidence_enabled:
            # Get prediction probability
            prediction_probability = prediction.get('p_target', 0.5)
            
            # Determine direction from prediction
            if 'side' in prediction and prediction['side'] != 0:
                predicted_side = prediction['side']
            else:
                predicted_side = 1 if score > 0 else -1
            
            # Check confidence threshold
            # With bidirectional models (side feature), p_target is from the chosen
            # side's perspective. Both LONG and SHORT require P(target) > threshold.
            # Without bidirectional, p_target is from LONG perspective:
            #   LONG: P(target) > threshold
            #   SHORT: P(target) < (1 - threshold), i.e., P(stop) > threshold
            is_confident = False
            if self.predictor.has_side_feature:
                # Bidirectional: p_target is from chosen side's perspective
                is_confident = prediction_probability > self.confidence_threshold
            elif predicted_side == 1:  # LONG (non-bidirectional)
                is_confident = prediction_probability > self.confidence_threshold
            else:  # SHORT (non-bidirectional, LONG perspective probabilities)
                is_confident = prediction_probability < (1 - self.confidence_threshold)
            
            if not is_confident:
                logger.info(
                    f"✗ Confidence filter rejected signal: "
                    f"side={'LONG' if predicted_side == 1 else 'SHORT'}, "
                    f"P={prediction_probability:.3f}, "
                    f"threshold={self.confidence_threshold:.2f}"
                )
                
                # Record filtered signal
                self.metrics_tracker.record_signal(
                    executed=False,
                    score=score,
                    timestamp=bar_time,
                    direction="LONG" if predicted_side == 1 else "SHORT",
                    reason=f"confidence_filter_P={prediction_probability:.3f}",
                )
                return

        # Risk manager gating before trade decision
        base_primary_threshold = self.live_cfg['signals']['primary_threshold']
        base_primary_threshold_long = self.live_cfg['signals'].get(
            'primary_threshold_long',
            base_primary_threshold,
        )
        base_primary_threshold_short = self.live_cfg['signals'].get(
            'primary_threshold_short',
            base_primary_threshold,
        )
        primary_threshold = base_primary_threshold
        primary_threshold_long = base_primary_threshold_long
        primary_threshold_short = base_primary_threshold_short
        
        # Phase 2a: Apply circuit breaker threshold adjustment
        if self.circuit_breaker_enabled and self.circuit_breaker is not None:
            # Check if in cooling-off period
            if self.circuit_breaker.is_in_cooling_off():
                logger.info("⏸️ Circuit breaker: In cooling-off period, skipping signal")
                self.metrics_tracker.record_signal(
                    executed=False,
                    score=score,
                    timestamp=bar_time,
                    direction=None,
                    reason="circuit_breaker_cooling_off",
                )
                return
            
            # Apply threshold adjustment if adapted
            cb_threshold = self.circuit_breaker.get_current_threshold()
            if cb_threshold > self.confidence_threshold:
                logger.info(f"📊 Circuit breaker raised threshold: {self.confidence_threshold:.2f} → {cb_threshold:.2f}")
                # Override confidence threshold with circuit breaker's adaptive threshold
                # This affects the primary threshold used for trading decisions
                threshold_boost = cb_threshold - self.confidence_threshold
                primary_threshold += threshold_boost
                primary_threshold_long += threshold_boost
                primary_threshold_short += threshold_boost

        # Regime-aware threshold adjustment (low volatility = higher bar)
        regime_cfg = self.live_cfg['signals'].get('regime_adjustment', {})
        if regime_cfg.get('enabled', False):
            import numpy as np

            # Check ATR against median (from training data)
            atr_current = features.get('atr_14', np.nan)
            if np.isfinite(atr_current):
                # Load median ATR from training analysis
                atr_analysis_path = self.run_dir / f"bar_size={self.bar_size}" / "atr_analysis.json"
                if atr_analysis_path.exists():
                    with open(atr_analysis_path) as f:
                        atr_analysis = json.load(f)
                    atr_median = atr_analysis['atr_stats']['median']

                    # If current ATR is < 70% of median, boost threshold
                    if atr_current < 0.7 * atr_median:
                        boost = regime_cfg.get('low_vol_threshold_boost', 0.05)
                        primary_threshold += boost
                        primary_threshold_long += boost
                        primary_threshold_short += boost
                        logger.info(
                            f"Low volatility adjustment: ATR={atr_current:.2f} < 70% median, "
                            f"boosting threshold by +{boost:.3f} -> {primary_threshold:.3f}"
                        )

        if self.live_risk_manager:
            allowed, risk_reason = self.live_risk_manager.check_trading_allowed()
            if not allowed:
                logger.info(f"✗ Risk manager blocked trade: {risk_reason}")
                self.metrics_tracker.record_signal(
                    executed=False,
                    score=score,
                    timestamp=bar_time,
                    direction=None,
                    reason=f"risk_manager_{risk_reason}",
                )
                return

            threshold_adjustment = self.live_risk_manager.get_threshold_adjustment()
            if threshold_adjustment > 0:
                primary_threshold += threshold_adjustment
                primary_threshold_long += threshold_adjustment
                primary_threshold_short += threshold_adjustment
                logger.info(
                    "Risk manager threshold adjustment: +%.3f -> %.3f",
                    threshold_adjustment,
                    primary_threshold,
                )

        # FIX 5: Cap threshold to prevent stacking beyond achievable range
        max_thresh = self.live_cfg['signals'].get('max_primary_threshold', 0.15)
        primary_threshold = min(primary_threshold, max_thresh)
        primary_threshold_long = min(primary_threshold_long, max_thresh)
        primary_threshold_short = min(primary_threshold_short, max_thresh)

        # Check if we should trade
        should_trade, reason = self.predictor.should_trade(
            prediction=prediction,
            primary_threshold=primary_threshold,
            primary_threshold_long=primary_threshold_long,
            primary_threshold_short=primary_threshold_short,
            meta_threshold=self.live_cfg['signals'].get('meta_threshold'),
            require_meta_approval=self.live_cfg['signals'].get('require_meta_approval', False),
            allowed_directions=self.live_cfg.get('signals', {}).get('allowed_directions'),
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

        # Determine direction from bidirectional model or fallback to score-based
        if 'side' in prediction and prediction['side'] != 0:
            # Use model's predicted side (1=LONG, -1=SHORT)
            direction = "LONG" if prediction['side'] == 1 else "SHORT"
            logger.info(f"  Bidirectional choice: {direction} (EV_long={prediction.get('score_ev_long', 'N/A'):.3f}, EV_short={prediction.get('score_ev_short', 'N/A'):.3f})")
        elif 'side' in prediction and prediction['side'] == 0:
            # Model says skip (neither side has positive EV)
            logger.info(f"✗ Bidirectional model recommends SKIP (both sides have negative EV)")
            return
        else:
            # Fallback for non-bidirectional models
            direction = "LONG" if prediction['score_ev'] > 0 else "SHORT"

        # Get trade history for Kelly calculation
        trade_history = self.metrics_tracker.trade_history

        # Execute trade (Kelly sizer will override contracts if enabled)
        contracts = self.live_cfg['positions']['contracts_per_trade']
        contracts_cap = None
        
        # Phase 2a: Apply circuit breaker position size adjustment
        if self.circuit_breaker_enabled and self.circuit_breaker is not None:
            position_multiplier = self.circuit_breaker.get_current_position_multiplier()
            if position_multiplier < 1.0:
                original_contracts = contracts
                contracts = max(1, int(contracts * position_multiplier))
                logger.info(f"📊 Circuit breaker reduced position size: {original_contracts} → {contracts} contracts ({position_multiplier:.0%})")
        
        if self.live_risk_manager:
            sigma_val = None
            try:
                if isinstance(features, pd.Series) and 'sigma' in features.index:
                    sigma_val = float(features.get('sigma'))
            except Exception:
                sigma_val = None
            contracts = self.live_risk_manager.get_position_size(contracts, sigma=sigma_val)
            if contracts <= 0:
                logger.info("✗ Risk manager blocked trade: position_size=0")
                self.metrics_tracker.record_signal(
                    executed=False,
                    score=score,
                    timestamp=bar_time,
                    direction=direction,
                    reason="risk_manager_position_size_zero",
                )
                return
            contracts_cap = contracts
        success, exec_reason = self.execution_engine.execute_signal(
            timestamp=bar_time,
            direction=direction,
            prediction=prediction,
            bars_df=bars_df,
            contracts=contracts,
            kelly_sizer=self.kelly_sizer,  # NEW
            trade_history=trade_history,   # NEW
            contracts_cap=contracts_cap,
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
        "--config-name",
        type=str,
        default="live_trading.yaml",
        help="Config filename within config-dir (default: live_trading.yaml)",
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
        config_name=args.config_name,
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
