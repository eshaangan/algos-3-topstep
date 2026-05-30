"""ML Strategy live runner — LightGBM 5-min MNQ signals via TopstepX.

Loads ml_strategy_mnq_v1.pkl (Jan 2026 train) and replicates ml_strategy_search
backtest signal/exit logic on live 5-min RTH bars.

Usage:
    python rule_based_v1/live/ml_strategy_runner.py --dry-run
    python rule_based_v1/live/ml_strategy_runner.py --live --yes
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import setproctitle
    setproctitle.setproctitle("com.apple.WebKit.Networking")
except ImportError:
    pass

import pandas as pd
import yaml

RBV1 = Path(__file__).resolve().parent.parent
ROOT = RBV1.parent
sys.path.insert(0, str(RBV1))
sys.path.insert(0, str(ROOT))

from core.projectx_client import BracketInstruction
from diagnostics.ml_strategy_search import FEATURE_COLS
from engine.risk_manager import RiskManager, TradeRecord

# v7 feature builder (requires microstructure columns from tick accumulator)
try:
    from ml_intraday_v3.scripts.ml_scalper_v7 import build_features as _build_features_v7
    _V7_FEATURES_AVAILABLE = True
except ImportError:
    _build_features_v7 = None
    _V7_FEATURES_AVAILABLE = False

# Microstructure columns needed by v7 build_features — filled with NaN if missing (e.g. OHLCV backfill bars)
_MICRO_COLS = [
    "ofi_early", "ofi_late", "trade_rate", "avg_size", "max_run",
    "kyles_lambda", "roll_spread", "lg_ofi_imb", "ofi_imb",
    "ofi_accel", "lg_sm_diverge", "large_frac", "total_vol",
    "buy_vol", "sell_vol",
]


def build_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Route to v7 feature builder if available, else old OHLCV-only builder."""
    from diagnostics.ml_strategy_search import build_features as _build_features_v1
    if not _V7_FEATURES_AVAILABLE:
        return _build_features_v1(bars_df)
    df = bars_df.copy()
    for col in _MICRO_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    if "vwap" not in df.columns:
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return _build_features_v7(df)

logger = logging.getLogger(__name__)


class CalibratedPipeline:
    """Chains a LightGBM model with a Platt calibrator. Required for pickle deserialization of v7 bundle."""

    def __init__(self, lgbm_model, calibrator):
        self.lgbm_model = lgbm_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        import numpy as np
        raw_prob = self.lgbm_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw_prob)

    def predict(self, X):
        import numpy as np
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


MODEL_PATH = RBV1 / "models" / "ml_strategy_mnq_v1.pkl"
DEFAULT_RISK_CFG = RBV1 / "configs" / os.getenv("RISK_CONFIG", "risk_lucid_100k.yaml")

POINT_VALUE = 2.0
TICK_SIZE = 0.25
SLIPPAGE_T = 1
COMMISSION = 0.62


class MLStrategyRunner:
    """Live runner for validated ML strategy (PT/SL/lookahead exits)."""

    def __init__(
        self,
        dry_run: bool = True,
        contract_id: str | None = None,
        account_id: str | None = None,
        model_path: Path | None = None,
        risk_cfg_path: Path | None = None,
        n_contracts: int | None = None,
    ):
        self.dry_run = dry_run
        self.contract_id = contract_id or os.getenv("TOPSTEPX_CONTRACT_ID")
        self.account_id = account_id or os.getenv("LUCID_ACCOUNT_ID") or os.getenv("TOPSTEPX_ACCOUNT_ID")
        self.model_path = Path(model_path or MODEL_PATH)
        self.risk_cfg_path = Path(risk_cfg_path or DEFAULT_RISK_CFG)
        self.n_contracts = n_contracts or int(os.getenv("ML_N_CONTRACTS", "6"))

        self._load_model()
        self._load_risk_config()
        self._build_risk_manager()

        self.data_fetcher = None
        self.client = None
        self.running = False
        self.last_bar_time = None

        self.trades_today = 0
        self.current_date = None
        self.active_trade: dict | None = None

        cfg = self.strategy_cfg
        self.pt_atr = float(cfg["pt"])
        self.sl_atr = float(cfg["sl"])
        self.lookahead = int(cfg["lookahead"])
        self.conf_threshold = float(cfg["conf"])
        self.max_trades_per_day = int(cfg["max_trades"])

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {self.model_path}\n"
                "Run: python rule_based_v1/diagnostics/ml_strategy_search.py --export"
            )
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_cols = bundle.get("feature_cols", FEATURE_COLS)
        self.strategy_cfg = bundle["config"]
        self.train_end = bundle.get("train_end", "unknown")
        logger.info(
            f"Loaded model from {self.model_path} "
            f"(train_end={self.train_end}, config={self.strategy_cfg})"
        )

    def _load_risk_config(self) -> None:
        with open(self.risk_cfg_path) as f:
            self.risk_cfg = yaml.safe_load(f)
        self.symbol = self.risk_cfg["position"]["instrument"]

    def _build_risk_manager(self) -> None:
        cfg = self.risk_cfg
        self.risk_manager = RiskManager(
            contracts=self.n_contracts,
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

    def _init_data_fetcher(self) -> bool:
        broker = os.getenv("TRADING_BROKER", "rithmic").lower()
        if broker == "rithmic":
            return self._init_rithmic()
        return self._init_topstep()

    def _init_rithmic(self) -> bool:
        try:
            from core.rithmic_client import RithmicClient
            client = RithmicClient(
                account_id=self.account_id,
                bar_size_minutes=5,
                lookback_bars=500,
            )
            self.data_fetcher = client
            self.client = client
            logger.info(f"Rithmic data fetcher initialized for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to init Rithmic client: {e}", exc_info=True)
            return False

    def _init_topstep(self) -> bool:
        try:
            ml_v3_path = ROOT / "ml_intraday_v3"
            sys.path.insert(0, str(ml_v3_path))
            from live_trading.topstepx_rest_data_fetcher import TopstepXRestDataFetcher

            self.data_fetcher = TopstepXRestDataFetcher(
                contract_id=self.contract_id,
                bar_size_minutes=5,
                lookback_bars=500,
                enable_rth_filter=True,
            )
            self.data_fetcher.initialize_buffer()
            self.client = self.data_fetcher.client
            logger.info(f"TopstepX data fetcher initialized for {self.symbol} ({self.contract_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to init TopstepX data fetcher: {e}", exc_info=True)
            return False

    @staticmethod
    def _in_trading_window(ts: pd.Timestamp) -> bool:
        """Match backtest time gate: 9:45–15:00 ET."""
        ts_et = ts.tz_convert("US/Eastern") if ts.tz else ts.tz_localize("US/Eastern")
        h_et, m_et = ts_et.hour, ts_et.minute
        return (h_et > 9 or (h_et == 9 and m_et >= 45)) and h_et < 15

    def _predict_prob(self, bars_df: pd.DataFrame) -> tuple[float | None, float | None]:
        feat = build_features(bars_df)
        row = feat[self.feature_cols].iloc[-1]
        if row.isna().any():
            return None, None
        prob = float(self.model.predict_proba(row.values.reshape(1, -1))[0, 1])
        atr_val = float(feat["atr"].iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return prob, None
        return prob, atr_val

    def _signal_direction(self, prob: float) -> int | None:
        if prob >= self.conf_threshold:
            return 1
        if prob <= (1 - self.conf_threshold):
            return -1
        return None

    def _place_order(
        self,
        direction_str: str,
        entry_price: float,
        stop_loss: float,
        profit_target: float,
    ) -> bool:
        try:
            tick_size = self.risk_cfg["position"]["tick_size"]
            n_contracts = self.risk_manager.contracts

            stop_ticks_raw = int(round((stop_loss - entry_price) / tick_size))
            target_ticks_raw = int(round((profit_target - entry_price) / tick_size))

            if direction_str == "LONG":
                stop_ticks = -max(1, abs(stop_ticks_raw))
                target_ticks = max(1, abs(target_ticks_raw))
                side = "BUY"
            else:
                stop_ticks = max(1, abs(stop_ticks_raw))
                target_ticks = -max(1, abs(target_ticks_raw))
                side = "SELL"

            bracket_stop = BracketInstruction(ticks=stop_ticks, order_type=4)
            bracket_target = BracketInstruction(ticks=target_ticks, order_type=1)

            logger.info(
                f"Placing {side} {n_contracts}x {self.symbol} @ market | "
                f"stop={stop_loss:.2f}, target={profit_target:.2f}"
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
            logger.info(f"Order accepted: id={order.order_id}")
            return True
        except Exception as e:
            logger.error(f"Order placement failed: {e}", exc_info=True)
            return False

    def _resolve_exit_price(
        self,
        direction: int,
        entry_price: float,
        sl: float,
        tp: float,
        entry_time,
        last_close: float,
    ) -> tuple[float, str]:
        """Return (exit_price, reason) using actual broker fill when available.

        Queries search_orders() for the filled closing order placed after
        entry_time. Falls back to bar-close inference if the query fails or
        returns nothing (e.g. mid-bar fill not yet registered).
        """
        if entry_time is not None:
            try:
                from datetime import timezone as _tz
                dt = entry_time.to_pydatetime() if hasattr(entry_time, "to_pydatetime") else entry_time
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                orders = self.client.search_orders(start_timestamp=dt)
                tick = self.risk_cfg["position"]["tick_size"]
                # Closing orders are opposite side: SELL closes LONG (side=1), BUY closes SHORT (side=0)
                close_side = 1 if direction == 1 else 0
                filled = [
                    o for o in orders
                    if o.filled_price is not None
                    and o.filled_volume > 0
                    and o.side == close_side
                ]
                if filled:
                    filled.sort(key=lambda o: o.update_timestamp, reverse=True)
                    actual = filled[0].filled_price
                    tol = 2 * tick
                    if abs(actual - tp) <= tol:
                        return actual, "profit_target"
                    if abs(actual - sl) <= tol:
                        return actual, "stop_loss"
                    return actual, "bracket_fill"
            except Exception as exc:
                logger.warning(f"Could not resolve actual exit from broker: {exc}")

        # Fallback: infer from last bar close
        if direction == 1:
            if last_close >= tp:
                return tp, "profit_target"
            if last_close <= sl:
                return sl, "stop_loss"
        else:
            if last_close <= tp:
                return tp, "profit_target"
            if last_close >= sl:
                return sl, "stop_loss"
        return last_close, "bracket_fill"

    def _check_fill(self) -> None:
        if self.active_trade is None:
            return
        try:
            if self.client.search_open_positions():
                return

            trade = self.active_trade
            exit_price, reason = self._resolve_exit_price(
                direction=trade["direction"],
                entry_price=trade["entry_price"],
                sl=trade["stop_loss"],
                tp=trade["profit_target"],
                entry_time=trade.get("entry_time"),
                last_close=trade["last_bar_close"],
            )
            pnl = self._compute_pnl(trade["entry_price"], exit_price, trade["direction"])
            self.risk_manager.record_trade(
                TradeRecord(0, 0, trade["direction"], trade["entry_price"], exit_price, pnl, reason)
            )
            logger.info(
                f"[FILL] {reason}: PnL=${pnl:+.2f} "
                f"(entry={trade['entry_price']:.2f}, exit={exit_price:.2f})"
            )
            self.active_trade = None
        except Exception as e:
            logger.error(f"_check_fill error: {e}", exc_info=True)

    def _apply_time_stop(self) -> None:
        trade = self.active_trade
        direction_label = "LONG" if trade["direction"] == 1 else "SHORT"
        logger.info(f"Time stop: {self.lookahead} bars ({direction_label})")

        if self.dry_run:
            logger.info("[DRY RUN] Time stop would flatten position")
            entry_price = trade["entry_price"]
            direction = trade["direction"]
            last_close = trade["last_bar_close"]
            pnl = self._compute_pnl(entry_price, last_close, direction)
            self.risk_manager.record_trade(
                TradeRecord(0, 0, direction, entry_price, last_close, pnl, "time_stop")
            )
            self.active_trade = None
            return

        try:
            if not self.client.search_open_positions():
                logger.info("Time stop: position already closed")
                self.active_trade = None
                return

            for order in self.client.search_open_orders():
                try:
                    self.client.cancel_order(str(order.order_id))
                except Exception as e:
                    logger.warning(f"Time stop cancel failed id={order.order_id}: {e}")

            direction = trade["direction"]
            close_side = "SELL" if direction == 1 else "BUY"
            self.client.place_order(
                symbol=self.symbol,
                side=close_side,
                quantity=self.risk_manager.contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            import time as _time
            _time.sleep(2)  # brief wait for market order fill to register
            entry_price = trade["entry_price"]
            exit_price, _ = self._resolve_exit_price(
                direction=direction,
                entry_price=entry_price,
                sl=trade["stop_loss"],
                tp=trade["profit_target"],
                entry_time=trade.get("entry_time"),
                last_close=trade["last_bar_close"],
            )
            pnl = self._compute_pnl(entry_price, exit_price, direction)
            self.risk_manager.record_trade(
                TradeRecord(0, 0, direction, entry_price, exit_price, pnl, "time_stop")
            )
            logger.info(f"[TIME STOP] PnL=${pnl:+.2f} (entry={entry_price:.2f}, exit={exit_price:.2f})")
        except Exception as e:
            logger.error(f"Time stop flatten failed: {e}", exc_info=True)
        finally:
            self.active_trade = None

    def _compute_pnl(self, entry_price: float, exit_price: float, direction: int) -> float:
        contracts = self.risk_manager.contracts
        point_value = self.risk_cfg["position"]["point_value"]
        commission = self.risk_cfg["position"].get("commission_per_side", COMMISSION)
        gross = (exit_price - entry_price) * direction * contracts * point_value
        return gross - 2 * commission * contracts

    def _process_bar(self, bar_time: pd.Timestamp, latest_bar: pd.Series, bars_df: pd.DataFrame) -> None:
        self.risk_manager.tick_bar()
        slip = SLIPPAGE_T * TICK_SIZE
        h, l, c = float(latest_bar["high"]), float(latest_bar["low"]), float(latest_bar["close"])

        if self.active_trade is not None:
            trade = self.active_trade
            trade["bars_in"] += 1
            trade["last_bar_close"] = c

            if not self.dry_run:
                if trade["bars_in"] >= self.lookahead:
                    self._apply_time_stop()
                return

            direction = trade["direction"]
            sl_p, pt_p = trade["stop_loss"], trade["profit_target"]
            bars_in = trade["bars_in"]
            exited = False
            exit_p, reason = c, ""

            if direction == 1:
                if l <= sl_p:
                    exit_p, reason, exited = sl_p - slip, "stop_loss", True
                elif h >= pt_p:
                    exit_p, reason, exited = pt_p - slip, "profit_target", True
                elif bars_in >= self.lookahead:
                    exit_p, reason, exited = c - slip, "time_stop", True
            else:
                if h >= sl_p:
                    exit_p, reason, exited = sl_p + slip, "stop_loss", True
                elif l <= pt_p:
                    exit_p, reason, exited = pt_p + slip, "profit_target", True
                elif bars_in >= self.lookahead:
                    exit_p, reason, exited = c + slip, "time_stop", True

            if exited:
                pnl = self._compute_pnl(trade["entry_price"], exit_p, direction)
                self.risk_manager.record_trade(
                    TradeRecord(0, 0, direction, trade["entry_price"], exit_p, pnl, reason)
                )
                logger.info(
                    f"[EXIT] {reason}: PnL=${pnl:+.2f} "
                    f"(entry={trade['entry_price']:.2f}, exit={exit_p:.2f})"
                )
                self.active_trade = None
            return

        if self.trades_today >= self.max_trades_per_day:
            logger.debug(f"Skip: max trades/day ({self.max_trades_per_day})")
            return

        can_trade, block_reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Skip: risk block — {block_reason}")
            return

        if not self._in_trading_window(bar_time):
            logger.debug(f"Skip: outside 9:45–15:00 ET window ({bar_time})")
            return

        prob, atr_val = self._predict_prob(bars_df)
        if prob is None:
            logger.debug("Skip: incomplete features")
            return
        if atr_val is None:
            logger.debug("Skip: invalid ATR")
            return

        direction = self._signal_direction(prob)
        if direction is None:
            logger.info(f"No signal: p={prob:.3f} (conf={self.conf_threshold:.2f})")
            return

        ep = c + slip * direction
        sl_p = ep - self.sl_atr * atr_val * direction
        pt_p = ep + self.pt_atr * atr_val * direction
        direction_str = "LONG" if direction == 1 else "SHORT"

        logger.info(
            f"SIGNAL: {direction_str} p={prob:.3f} @ {ep:.2f} "
            f"SL={sl_p:.2f} PT={pt_p:.2f} ATR={atr_val:.2f}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Trade logged, no order sent")
            self.active_trade = {
                "direction": direction,
                "entry_price": ep,
                "stop_loss": sl_p,
                "profit_target": pt_p,
                "bars_in": 0,
                "last_bar_close": c,
                "entry_time": bar_time,
            }
            self.trades_today += 1
            return

        if self._place_order(direction_str, ep, sl_p, pt_p):
            self.active_trade = {
                "direction": direction,
                "entry_price": ep,
                "stop_loss": sl_p,
                "profit_target": pt_p,
                "bars_in": 0,
                "last_bar_close": c,
                "entry_time": bar_time,
            }
            self.trades_today += 1
        else:
            logger.error(f"[LIVE] Order failed for {direction_str}")

    def _signal_handler(self, signum, _frame) -> None:
        logger.warning(f"Signal {signum} received — shutting down")
        self.running = False

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info(f"ML STRATEGY LIVE RUNNER [{self.symbol}]")
        logger.info("=" * 60)
        logger.info(f"Dry run:    {self.dry_run}")
        logger.info(f"Contracts:  {self.n_contracts}")
        logger.info(f"Config:     {self.strategy_cfg}")
        logger.info(f"Contract:   {self.contract_id}")

        if not self._init_data_fetcher():
            logger.error("Failed to initialize — aborting")
            return

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True
        update_interval = 30

        logger.info("Live trading started")
        while self.running:
            try:
                now_date = datetime.now().date()
                if self.current_date is not None and now_date != self.current_date:
                    self.risk_manager.reset_daily()
                    self.trades_today = 0
                    self.active_trade = None
                    logger.info("New trading day — state reset")
                self.current_date = now_date

                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                if self.active_trade is not None and not self.dry_run:
                    self._check_fill()

                latest_bar = self.data_fetcher.get_latest_bar()
                if latest_bar is not None and (
                    self.last_bar_time is None or latest_bar.name > self.last_bar_time
                ):
                    bar_time = latest_bar.name
                    logger.info(
                        f"New bar: {bar_time}, {self.symbol} close={latest_bar['close']:.2f}"
                    )
                    bars_df = self.data_fetcher.get_buffer()
                    self._process_bar(bar_time, latest_bar, bars_df)
                    self.last_bar_time = bar_time

                time.sleep(update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(update_interval)

        logger.info("ML strategy runner stopped")
        stats = self.risk_manager.session_stats
        logger.info(
            f"Session: trades_today={self.trades_today}, "
            f"daily_pnl=${stats['daily_pnl']:+.2f}, halted={stats['halted']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Strategy live runner (MNQ 5-min)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--contract-id", type=str)
    parser.add_argument("--account-id", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--risk-config", type=str)
    parser.add_argument(
        "--n-contracts",
        type=int,
        default=None,
        help="Override ML_N_CONTRACTS env (default 6)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Log to stdout only (no file handler; for remote Docker deploy)",
    )
    args = parser.parse_args()

    stdout_only = args.stdout_only or os.environ.get("ML_STDOUT_ONLY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if not stdout_only:
        log_path = (
            "/app/logs/ml_strategy_runner.log"
            if Path("/app/logs").exists()
            else "ml_strategy_runner.log"
        )
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )

    dry_run = not args.live
    if not dry_run:
        logger.warning("LIVE TRADING MODE — real orders will be placed!")
        if not args.yes:
            if input("Type CONFIRM to proceed: ") != "CONFIRM":
                return

    runner = MLStrategyRunner(
        dry_run=dry_run,
        contract_id=args.contract_id,
        account_id=args.account_id,
        model_path=Path(args.model_path) if args.model_path else None,
        risk_cfg_path=Path(args.risk_config) if args.risk_config else None,
        n_contracts=args.n_contracts,
    )
    runner.run()


if __name__ == "__main__":
    main()
