"""ML Strategy live runner — LightGBM 5-min MNQ signals via Rithmic.

Architecture (post-incident rewrite, 2026-06-02):
  - Startup: verifies account is flat; halts if not.
  - Entry: plain MARKET order; fill confirmed via fill-history query.
  - Stop: separate STOP_MARKET placed after fill; software OHLC stop runs in
    parallel as the primary fallback (exchange stop is belt-and-suspenders).
  - Exit detection: OHLC bar check (primary) + position poll (secondary).
  - State: every active_trade = None is preceded by record_trade. No exceptions.
  - PnL accounting: all 7 ghost-trade paths from the incident audit are closed.

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
from datetime import datetime, timezone
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

from core.projectx_client import BracketInstruction  # noqa: F401 (kept for API compat)
from diagnostics.ml_strategy_search import FEATURE_COLS
from engine.risk_manager import RiskManager, TradeRecord

try:
    from ml_intraday_v3.scripts.ml_scalper_v7 import build_features as _build_features_v7
    _V7_FEATURES_AVAILABLE = True
except ImportError:
    _build_features_v7 = None
    _V7_FEATURES_AVAILABLE = False

_MICRO_COLS = [
    "ofi_early", "ofi_late", "trade_rate", "avg_size", "max_run",
    "kyles_lambda", "roll_spread", "lg_ofi_imb", "ofi_imb",
    "ofi_accel", "lg_sm_diverge", "large_frac", "total_vol",
    "buy_vol", "sell_vol",
]


def build_features(bars_df: pd.DataFrame) -> pd.DataFrame:
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
    """Pickle-compatible wrapper for LightGBM + Platt calibrator."""

    def __init__(self, lgbm_model, calibrator):
        self.lgbm = lgbm_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        import numpy as np
        raw_prob = self.lgbm.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw_prob)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


MODEL_PATH = RBV1 / "models" / "ml_strategy_mnq_v1.pkl"
DEFAULT_RISK_CFG = RBV1 / "configs" / os.getenv("RISK_CONFIG", "risk_lucid_100k.yaml")

POINT_VALUE = 2.0
TICK_SIZE   = 0.25
SLIPPAGE_T  = 1          # ticks of assumed slippage on entry/exit
COMMISSION  = 0.62       # per side per contract

# Maximum safe contracts — override from risk yaml if present
_MAX_SAFE_CONTRACTS = 3

# Phase-1 fill confirmation: how long to poll fill-history before giving up
_FILL_CONFIRM_TIMEOUT_S = 90
_FILL_POLL_INTERVAL_S   = 5   # seconds between fill-history polls in phase 1

# Phase-2 close detection: consecutive clean-empty position polls before declaring closed
_CLOSE_CONFIRM_POLLS = 3


class MLStrategyRunner:

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

        self._load_model()
        self._load_risk_config()

        # Contract count: env var > argument > default of 1 (NOT 6).
        env_n = os.getenv("ML_N_CONTRACTS")
        if n_contracts is not None:
            self.n_contracts = int(n_contracts)
        elif env_n is not None:
            self.n_contracts = int(env_n)
        else:
            self.n_contracts = 1
            logger.warning("ML_N_CONTRACTS not set — defaulting to 1 (safe). Pass --n-contracts or set env var to change.")

        if self.n_contracts > _MAX_SAFE_CONTRACTS:
            raise ValueError(
                f"n_contracts={self.n_contracts} exceeds _MAX_SAFE_CONTRACTS={_MAX_SAFE_CONTRACTS}. "
                "Edit _MAX_SAFE_CONTRACTS in runner if you intend to trade more."
            )

        self._build_risk_manager()

        self.data_fetcher = None
        self.client       = None
        self.running      = False
        self.last_bar_time = None
        self.trades_today  = 0
        self.current_date  = None

        # active_trade dict keys (all must be set together):
        #   direction       int           1=LONG, -1=SHORT
        #   entry_price     float         theoretical entry (c ± slippage)
        #   actual_entry    float|None    actual fill price once confirmed
        #   stop_loss       float         stop price
        #   profit_target   float         target price
        #   bars_in         int           bars elapsed since entry
        #   last_bar_close  float         close of last processed bar
        #   entry_time      pd.Timestamp  bar time of signal
        #   order_id        str           Rithmic basket_id of entry order
        #   fill_confirmed  bool          True once fill history confirms entry
        #   fill_poll_start float         monotonic time when fill polling began
        #   stop_order_id   str|None      Rithmic basket_id of stop order
        #   consecutive_no_pos int        for phase-2 close confirmation
        self.active_trade: dict | None = None

        cfg = self.strategy_cfg
        self.pt_atr   = float(cfg["pt"])
        self.sl_atr   = float(cfg["sl"])
        self.lookahead = int(cfg["lookahead"])
        self.conf_threshold = float(cfg["conf"])
        self.max_trades_per_day = int(cfg["max_trades"])

    # ─────────────────────────────────────────── initialisation ──────────────

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {self.model_path}")
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model       = bundle["model"]
        self.feature_cols = bundle.get("feature_cols", FEATURE_COLS)
        self.strategy_cfg = bundle["config"]
        self.train_end    = bundle.get("train_end", "unknown")
        logger.info("Loaded model from %s (train_end=%s, config=%s)",
                    self.model_path, self.train_end, self.strategy_cfg)

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
            self.client       = client
            logger.info("Rithmic client ready for %s", self.symbol)
            return True
        except Exception as e:
            logger.error("Failed to init Rithmic client: %s", e, exc_info=True)
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
            logger.info("TopstepX client ready for %s", self.symbol)
            return True
        except Exception as e:
            logger.error("Failed to init TopstepX: %s", e, exc_info=True)
            return False

    # ─────────────────────────────────────────── safety checks ───────────────

    def _startup_position_check(self) -> bool:
        """
        Verify the account is flat before entering the main loop.
        Returns True if safe to trade, False if pre-existing positions were found.
        If the position query itself fails, treats it as UNSAFE (returns False).
        """
        if self.dry_run:
            return True
        logger.info("Startup safety check: querying open positions…")
        try:
            positions = self.client.search_open_positions()
        except Exception as exc:
            logger.critical(
                "Startup position query failed (%s). Treating as UNSAFE — will not trade. "
                "Resolve manually and restart.", exc
            )
            return False

        if not positions:
            logger.info("Account is flat. Safe to start trading.")
            return True

        logger.critical(
            "STARTUP ABORT: account has %d open position(s) before trading started: %s. "
            "Close them manually and restart. Will not enter any trades.",
            len(positions), positions,
        )
        # Block trading for the day but keep running (so logs remain accessible)
        self.trades_today = self.max_trades_per_day
        return False

    # ─────────────────────────────────────────── utilities ───────────────────

    @staticmethod
    def _in_trading_window(ts: pd.Timestamp) -> bool:
        ts_et = ts.tz_convert("US/Eastern") if ts.tz else ts.tz_localize("US/Eastern")
        h, m = ts_et.hour, ts_et.minute
        return (h > 9 or (h == 9 and m >= 45)) and h < 15

    def _predict_prob(self, bars_df: pd.DataFrame) -> tuple[float | None, float | None]:
        feat = build_features(bars_df)
        row  = feat[self.feature_cols].iloc[-1]
        if row.isna().any():
            return None, None
        prob    = float(self.model.predict_proba(row.values.reshape(1, -1))[0, 1])
        atr_val = float(feat["atr"].iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return prob, None
        return prob, atr_val

    def _signal_direction(self, prob: float) -> int | None:
        return 1 if prob >= self.conf_threshold else None

    def _compute_pnl(self, entry: float, exit_price: float, direction: int) -> float:
        point_value = self.risk_cfg["position"]["point_value"]
        commission  = self.risk_cfg["position"].get("commission_per_side", COMMISSION)
        gross = (exit_price - entry) * direction * self.n_contracts * point_value
        return gross - 2 * commission * self.n_contracts

    # ─────────────────────────────────────────── order placement ─────────────

    def _place_entry_order(self, direction_str: str, entry_price: float,
                           stop_loss: float, profit_target: float) -> str | None:
        """
        Submit a plain MARKET entry. Returns order_id on success, None on failure.
        Does NOT set active_trade — caller is responsible.
        """
        side = "BUY" if direction_str == "LONG" else "SELL"
        try:
            logger.info("Placing %s %dx %s @ market | sl=%.2f pt=%.2f",
                        side, self.n_contracts, self.symbol, stop_loss, profit_target)
            order = self.client.place_order(
                symbol=self.symbol,
                side=side,
                quantity=self.n_contracts,
                order_type="MARKET",
                contract_id=self.contract_id,
            )
            logger.info("Entry order accepted: id=%s", order.order_id)
            return order.order_id
        except Exception as e:
            logger.error("Entry order failed: %s", e, exc_info=True)
            return None

    def _place_stop_order(self, stop_price: float, direction: int) -> str | None:
        """
        Submit a STOP_MARKET protective order.
        Returns order_id on success, None on failure.
        Logs a warning if the returned id looks like a local fallback UUID (not a real basket_id).
        """
        close_side = "SELL" if direction == 1 else "BUY"
        for attempt in range(3):
            try:
                stop_id = self.client.place_stop_order(
                    stop_price=stop_price,
                    quantity=self.n_contracts,
                    side=close_side,
                )
                # Warn if we got back a local UUID (Rithmic didn't assign a basket_id)
                if len(stop_id) == 36 and stop_id.count("-") == 4:
                    logger.warning(
                        "Stop order basket_id looks like a local UUID (%s) — "
                        "Rithmic may not have accepted the stop order. "
                        "Software stop will serve as primary protection.", stop_id
                    )
                else:
                    logger.info("Stop order placed: %s @ %.2f", stop_id, stop_price)
                return stop_id
            except Exception as exc:
                logger.error("place_stop_order attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(3)
        logger.error(
            "All stop placement attempts failed. "
            "SOFTWARE STOP IS NOW SOLE PROTECTION — monitor closely."
        )
        return None

    def _cancel_order(self, order_id: str | None, label: str = "order") -> None:
        if not order_id:
            return
        try:
            self.client.cancel_order(order_id)
            logger.info("%s cancelled: %s", label, order_id)
        except Exception as e:
            logger.warning("Could not cancel %s %s: %s", label, order_id, e)

    def _cancel_stop_order(self, stop_id: str | None) -> None:
        self._cancel_order(stop_id, "stop")

    def _send_market_close(self, direction: int) -> bool:
        """Send a MARKET order to close the position. Returns True on acceptance."""
        close_side = "SELL" if direction == 1 else "BUY"
        for attempt in range(3):
            try:
                self.client.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    quantity=self.n_contracts,
                    order_type="MARKET",
                    contract_id=self.contract_id,
                )
                logger.info("Market close sent (%s %dx)", close_side, self.n_contracts)
                return True
            except Exception as e:
                logger.error("Market close attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2)
        logger.critical("ALL MARKET CLOSE ATTEMPTS FAILED. Position may still be open.")
        return False

    # ─────────────────────────────────────────── fill confirmation ────────────

    def _confirm_fill_via_history(self, order_time) -> float | None:
        """
        Query fill history for an entry order placed at order_time.
        Returns actual fill price if found, None otherwise.
        """
        try:
            from datetime import timezone as _tz
            dt = order_time.to_pydatetime() if hasattr(order_time, "to_pydatetime") else order_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            orders = self.client.search_orders(start_timestamp=dt)
            direction = self.active_trade["direction"] if self.active_trade else 1
            entry_side = 0 if direction == 1 else 1   # 0=BUY fills LONG, 1=SELL fills SHORT
            filled = [
                o for o in orders
                if o.filled_price is not None
                and o.filled_volume > 0
                and o.side == entry_side
            ]
            if filled:
                filled.sort(key=lambda o: o.update_timestamp, reverse=True)
                return filled[0].filled_price
        except Exception as exc:
            logger.debug("Fill history query failed: %s", exc)
        return None

    def _confirm_fill_via_position(self) -> bool:
        """Secondary fill check via position query."""
        try:
            return bool(self.client.search_open_positions())
        except Exception as exc:
            logger.debug("Position query failed: %s", exc)
            return False

    # ─────────────────────────────────────────── exit price resolution ────────

    def _resolve_exit_price(self, direction: int, entry: float,
                             sl: float, tp: float, entry_time,
                             last_close: float) -> tuple[float, str]:
        """
        Determine the actual exit price and reason.
        Tries fill history first; falls back to bar-close inference.
        """
        if entry_time is not None:
            try:
                from datetime import timezone as _tz
                dt = entry_time.to_pydatetime() if hasattr(entry_time, "to_pydatetime") else entry_time
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                orders = self.client.search_orders(start_timestamp=dt)
                tick = self.risk_cfg["position"]["tick_size"]
                close_side = 1 if direction == 1 else 0   # 1=SELL closes LONG
                filled = [
                    o for o in orders
                    if o.filled_price is not None
                    and o.filled_volume > 0
                    and o.side == close_side
                ]
                if filled:
                    filled.sort(key=lambda o: o.update_timestamp, reverse=True)
                    actual = filled[0].filled_price
                    tol = 4 * tick   # wider tolerance for stop-market slippage
                    if abs(actual - tp) <= tol:
                        return actual, "profit_target"
                    if abs(actual - sl) <= tol:
                        return actual, "stop_loss"
                    return actual, "market_close"
            except Exception as exc:
                logger.warning("Could not resolve exit from fill history: %s", exc)

        # Fallback: infer from bar close
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
        return last_close, "market_close"

    # ─────────────────────────────────────────── trade lifecycle ─────────────

    def _record_and_clear(self, exit_price: float, reason: str) -> None:
        """
        Record the trade PnL in the risk manager and clear active_trade.
        This is the ONLY place active_trade is set to None during a trade.
        All exit paths must go through here.
        """
        trade = self.active_trade
        if trade is None:
            return
        entry = trade.get("actual_entry") or trade["entry_price"]
        direction = trade["direction"]
        pnl = self._compute_pnl(entry, exit_price, direction)
        self.risk_manager.record_trade(
            TradeRecord(0, 0, direction, entry, exit_price, pnl, reason)
        )
        logger.info("[%s] PnL=$%+.2f (entry=%.2f, exit=%.2f)",
                    reason.upper(), pnl, entry, exit_price)
        self.active_trade = None

    def _exit_trade(self, reason: str, last_close: float) -> None:
        """
        Cancel stop, send market close, then record.
        If market close fails, does NOT clear active_trade — position stays monitored.
        """
        trade = self.active_trade
        if trade is None:
            return

        # Cancel both protective orders first (best-effort)
        self._cancel_order(trade.get("stop_order_id"),  "stop")
        self._cancel_order(trade.get("limit_order_id"), "limit")
        trade["stop_order_id"]  = None
        trade["limit_order_id"] = None

        # Verify position still exists before sending close
        try:
            has_pos = bool(self.client.search_open_positions())
        except Exception:
            has_pos = True   # assume open if query fails

        if not has_pos:
            # Position already gone (stop hit before we got here)
            # Determine exit price from fill history
            exit_price, actual_reason = self._resolve_exit_price(
                direction=trade["direction"],
                entry=trade.get("actual_entry") or trade["entry_price"],
                sl=trade["stop_loss"],
                tp=trade["profit_target"],
                entry_time=trade.get("entry_time"),
                last_close=last_close,
            )
            logger.info("Position already closed (stop hit); recording as %s", actual_reason)
            self._record_and_clear(exit_price, actual_reason)
            return

        close_ok = self._send_market_close(trade["direction"])
        if not close_ok:
            # Close failed — keep active_trade so we keep monitoring
            logger.critical("Market close failed. Keeping trade active for retry next bar.")
            return

        time.sleep(2)  # brief wait for fill to register
        exit_price, actual_reason = self._resolve_exit_price(
            direction=trade["direction"],
            entry=trade.get("actual_entry") or trade["entry_price"],
            sl=trade["stop_loss"],
            tp=trade["profit_target"],
            entry_time=trade.get("entry_time"),
            last_close=last_close,
        )
        self._record_and_clear(exit_price, reason if reason != "auto" else actual_reason)

    # ─────────────────────────────────────────── phase 1: fill confirmation ──

    def _run_fill_confirmation(self) -> None:
        """
        Phase 1: poll fill history (and position as secondary) until fill is confirmed
        or timeout expires.  Called from the main loop every ~10s after entry.

        On fill confirmation:
          - Sets trade["fill_confirmed"] = True
          - Updates trade["actual_entry"] with real fill price
          - Places protective stop (best-effort; logs warning if it fails)
        On timeout:
          - Abandons trade (sets active_trade = None WITHOUT recording PnL, since
            the order may not have filled).  Logs CRITICAL.
        """
        trade = self.active_trade
        if trade is None or trade.get("fill_confirmed"):
            return

        elapsed = time.monotonic() - trade["fill_poll_start"]
        if elapsed > _FILL_CONFIRM_TIMEOUT_S:
            logger.critical(
                "Fill confirmation timeout (%ds). Entry order %s may not have filled. "
                "Abandoning trade. Check broker manually.",
                _FILL_CONFIRM_TIMEOUT_S, trade.get("order_id"),
            )
            # Do NOT record PnL — we don't know if there's a position
            self.active_trade = None
            return

        # Try fill history first (gives actual fill price)
        actual_fill = self._confirm_fill_via_history(trade["entry_time"])
        if actual_fill is None:
            # Try position query as secondary
            if not self._confirm_fill_via_position():
                logger.debug("Phase 1: fill not yet confirmed (%.0fs elapsed)…", elapsed)
                return
            # Position exists but no fill history yet; use theoretical entry
            actual_fill = trade["entry_price"]
            logger.debug("Phase 1: fill confirmed via position (no fill history yet)")
        else:
            logger.info("Phase 1: fill confirmed via history @ %.2f", actual_fill)

        # Fill confirmed — update entry price and set stop
        trade["actual_entry"]   = actual_fill
        trade["fill_confirmed"] = True
        trade["consecutive_no_pos"] = 0

        # Adjust stop/target to actual fill if materially different
        diff = actual_fill - trade["entry_price"]
        if abs(diff) > TICK_SIZE:
            trade["stop_loss"]     += diff
            trade["profit_target"] += diff
            logger.info("Adjusted sl/pt by %.2f to match actual fill", diff)

        # Place exchange stop (primary protection)
        stop_id = self._place_stop_order(trade["stop_loss"], trade["direction"])
        trade["stop_order_id"] = stop_id
        if stop_id is None:
            logger.warning("No exchange stop placed. Software stop is sole protection.")

        # Place exchange limit for profit target — gives exchange-quality fill
        # rather than waiting for bar close and exiting at market.
        # Tracked separately; whichever fills first, we cancel the other.
        limit_side = "SELL" if trade["direction"] == 1 else "BUY"
        try:
            limit_id = self.client.place_limit_order(
                limit_price=trade["profit_target"],
                quantity=self.n_contracts,
                side=limit_side,
            )
            trade["limit_order_id"] = limit_id
            logger.info("Profit-target limit placed: %s @ %.2f", limit_id, trade["profit_target"])
        except Exception as exc:
            logger.warning("Could not place profit-target limit: %s — software target check active", exc)
            trade["limit_order_id"] = None

    # ─────────────────────────────────────────── phase 2: close detection ────

    def _check_position_closed(self) -> None:
        """
        Phase 2: called from main loop every poll interval while fill_confirmed.
        Requires _CLOSE_CONFIRM_POLLS consecutive clean-empty results before
        declaring the position closed.
        Exceptions do NOT count toward the counter (query errors ≠ closed position).
        """
        trade = self.active_trade
        if trade is None or not trade.get("fill_confirmed"):
            return

        try:
            has_position = bool(self.client.search_open_positions())
            query_ok = True
        except Exception as exc:
            logger.debug("Position query error (not counting as absent): %s", exc)
            query_ok = False

        if not query_ok or has_position:
            trade["consecutive_no_pos"] = 0
            return

        trade["consecutive_no_pos"] = trade.get("consecutive_no_pos", 0) + 1
        logger.debug("Phase 2: position absent (%d/%d)", trade["consecutive_no_pos"], _CLOSE_CONFIRM_POLLS)

        if trade["consecutive_no_pos"] < _CLOSE_CONFIRM_POLLS:
            return

        # Position confirmed closed — cancel the surviving protective order
        self._cancel_order(trade.get("stop_order_id"),  "stop")
        self._cancel_order(trade.get("limit_order_id"), "limit")
        trade["stop_order_id"]  = None
        trade["limit_order_id"] = None

        exit_price, reason = self._resolve_exit_price(
            direction=trade["direction"],
            entry=trade.get("actual_entry") or trade["entry_price"],
            sl=trade["stop_loss"],
            tp=trade["profit_target"],
            entry_time=trade.get("entry_time"),
            last_close=trade["last_bar_close"],
        )
        logger.info("Phase 2: position gone for %d polls — recording close", _CLOSE_CONFIRM_POLLS)
        self._record_and_clear(exit_price, reason)

    # ─────────────────────────────────────────── bar processing ──────────────

    def _process_bar(self, bar_time: pd.Timestamp, latest_bar: pd.Series,
                     bars_df: pd.DataFrame) -> None:
        self.risk_manager.tick_bar()
        slip = SLIPPAGE_T * TICK_SIZE
        h = float(latest_bar["high"])
        l = float(latest_bar["low"])
        c = float(latest_bar["close"])

        if self.active_trade is not None:
            trade = self.active_trade
            trade["bars_in"]       += 1
            trade["last_bar_close"] = c

            if self.dry_run:
                self._process_bar_dry_run(trade, h, l, c, slip)
                return

            # ── Live mode ──────────────────────────────────────────────────

            if not trade.get("fill_confirmed"):
                # Still waiting for fill confirmation — just keep polling
                return

            direction = trade["direction"]
            sl_p = trade["stop_loss"]
            pt_p = trade["profit_target"]

            # ① Software stop-loss check (primary protection; mirrors dry-run logic).
            #    This catches cases where the exchange stop was never placed or failed
            #    to trigger (common on sim accounts).
            stop_hit   = (direction == 1 and l <= sl_p) or (direction == -1 and h >= sl_p)
            target_hit = (direction == 1 and h >= pt_p) or (direction == -1 and l <= pt_p)

            if stop_hit and not target_hit:
                logger.info("Software stop triggered: bar low=%.2f, sl=%.2f", l, sl_p)
                self._exit_trade(reason="stop_loss", last_close=c)
                return

            if target_hit:
                logger.info("Profit target hit: bar high=%.2f, pt=%.2f", h, pt_p)
                self._exit_trade(reason="profit_target", last_close=c)
                return

            if stop_hit and target_hit:
                # Both hit in same bar — use stop (conservative)
                logger.info("Both stop and target hit in same bar — exiting at stop (conservative)")
                self._exit_trade(reason="stop_loss", last_close=c)
                return

            if trade["bars_in"] >= self.lookahead:
                logger.info("Time stop: %d bars elapsed", trade["bars_in"])
                self._exit_trade(reason="time_stop", last_close=c)
                return

            return

        # ── No active trade — evaluate signal ─────────────────────────────

        if self.trades_today >= self.max_trades_per_day:
            logger.debug("Skip: max trades/day (%d)", self.max_trades_per_day)
            return

        can_trade, block_reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info("Skip: risk block — %s", block_reason)
            return

        if not self._in_trading_window(bar_time):
            logger.debug("Skip: outside 9:45–15:00 ET window (%s)", bar_time)
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
            logger.info("No signal: p=%.3f (conf=%.3f)", prob, self.conf_threshold)
            return

        ep    = c + slip * direction
        sl_p  = ep - self.sl_atr  * atr_val * direction
        pt_p  = ep + self.pt_atr  * atr_val * direction
        dir_s = "LONG" if direction == 1 else "SHORT"
        logger.info("SIGNAL: %s p=%.3f @ %.2f  SL=%.2f  PT=%.2f  ATR=%.2f",
                    dir_s, prob, ep, sl_p, pt_p, atr_val)

        if self.dry_run:
            self.active_trade = {
                "direction": direction, "entry_price": ep, "actual_entry": ep,
                "stop_loss": sl_p, "profit_target": pt_p, "bars_in": 0,
                "last_bar_close": c, "entry_time": bar_time,
                "order_id": None, "fill_confirmed": True,
                "fill_poll_start": 0.0, "stop_order_id": None,
                "consecutive_no_pos": 0,
            }
            self.trades_today += 1
            logger.info("[DRY RUN] Trade logged, no order sent")
            return

        order_id = self._place_entry_order(dir_s, ep, sl_p, pt_p)
        if order_id is None:
            logger.error("Order placement failed — no trade entered")
            return

        self.active_trade = {
            "direction": direction,
            "entry_price": ep,
            "actual_entry": None,      # filled in by _run_fill_confirmation
            "stop_loss": sl_p,
            "profit_target": pt_p,
            "bars_in": 0,
            "last_bar_close": c,
            "entry_time": bar_time,
            "order_id": order_id,
            "fill_confirmed": False,
            "fill_poll_start": time.monotonic(),
            "stop_order_id": None,
            "limit_order_id": None,
            "consecutive_no_pos": 0,
        }
        self.trades_today += 1

    def _process_bar_dry_run(self, trade: dict, h: float, l: float,
                              c: float, slip: float) -> None:
        direction = trade["direction"]
        sl_p, pt_p = trade["stop_loss"], trade["profit_target"]
        bars_in    = trade["bars_in"]
        exited     = False
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
            self._record_and_clear(exit_p, reason)

    # ─────────────────────────────────────────── main loop ───────────────────

    def _signal_handler(self, signum, _frame) -> None:
        logger.warning("Signal %d received — shutting down", signum)
        self.running = False

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("ML STRATEGY LIVE RUNNER [%s]", self.symbol)
        logger.info("=" * 60)
        logger.info("Dry run:    %s", self.dry_run)
        logger.info("Contracts:  %d", self.n_contracts)
        logger.info("Config:     %s", self.strategy_cfg)

        if not self._init_data_fetcher():
            logger.error("Failed to initialize data fetcher — aborting")
            return

        # ── Startup safety check ──────────────────────────────────────────
        self._startup_position_check()

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True

        logger.info("Live trading started")
        while self.running:
            try:
                # Day rollover
                now_date = datetime.now().date()
                if self.current_date is not None and now_date != self.current_date:
                    if self.active_trade is not None and not self.dry_run:
                        logger.warning("Day rollover with active trade — cancelling stop and recording time_stop")
                        self._cancel_stop_order(self.active_trade.get("stop_order_id"))
                        exit_price, reason = self._resolve_exit_price(
                            direction=self.active_trade["direction"],
                            entry=self.active_trade.get("actual_entry") or self.active_trade["entry_price"],
                            sl=self.active_trade["stop_loss"],
                            tp=self.active_trade["profit_target"],
                            entry_time=self.active_trade.get("entry_time"),
                            last_close=self.active_trade["last_bar_close"],
                        )
                        self._record_and_clear(exit_price, "day_rollover_close")
                    self.risk_manager.reset_daily()
                    self.trades_today = 0
                    logger.info("New trading day — state reset")
                self.current_date = now_date

                # Ingest latest bar
                new_bar = self.data_fetcher.fetch_latest_bar()
                if new_bar is not None:
                    self.data_fetcher.update_buffer(new_bar)

                # ── Trade management (every poll) ─────────────────────────
                if self.active_trade is not None and not self.dry_run:
                    if not self.active_trade.get("fill_confirmed"):
                        self._run_fill_confirmation()
                    else:
                        self._check_position_closed()

                # ── Bar processing (only on new bar) ─────────────────────
                latest_bar = self.data_fetcher.get_latest_bar()
                if latest_bar is not None and (
                    self.last_bar_time is None or latest_bar.name > self.last_bar_time
                ):
                    bar_time = latest_bar.name
                    logger.info("New bar: %s, %s close=%.2f", bar_time, self.symbol, latest_bar["close"])
                    bars_df = self.data_fetcher.get_buffer()
                    self._process_bar(bar_time, latest_bar, bars_df)
                    self.last_bar_time = bar_time

                # Adaptive poll interval: tight while waiting for fill, relaxed otherwise
                if self.active_trade is not None and not self.active_trade.get("fill_confirmed"):
                    time.sleep(_FILL_POLL_INTERVAL_S)
                elif self.active_trade is not None:
                    time.sleep(10)
                else:
                    time.sleep(30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(10)

        logger.info("ML strategy runner stopped")
        stats = self.risk_manager.session_stats
        logger.info("Session: trades=%d, daily_pnl=$%+.2f, halted=%s",
                    self.trades_today, stats["daily_pnl"], stats["halted"])


# ─────────────────────────────────────────────────────────────────── CLI ─────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML Strategy live runner (MNQ 5-min)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live",    action="store_true")
    parser.add_argument("--yes",     action="store_true")
    parser.add_argument("--contract-id",  type=str)
    parser.add_argument("--account-id",   type=str)
    parser.add_argument("--model-path",   type=str)
    parser.add_argument("--risk-config",  type=str)
    parser.add_argument("--n-contracts",  type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--stdout-only",  action="store_true")
    args = parser.parse_args()

    stdout_only = args.stdout_only or os.environ.get("ML_STDOUT_ONLY", "").lower() in ("1", "true", "yes")
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
    for _noisy in ("websockets", "websockets.client", "asyncio", "rithmic"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

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
