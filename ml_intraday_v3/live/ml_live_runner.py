"""
ML Scalper Live Runner — TopstepX SignalR streams + LightGBM prediction.

Data sources (all via TopstepX, no external subscription needed):
  GatewayTrade  → trade ticks with aggressor side → OFI, Kyle's lambda, large-trade metrics
  GatewayDepth  → L2 DOM order book             → bid/ask imbalance, spread

Model: ml_scalper_v3 (AUC=0.559 LONG, OOS Sharpe=5.6 with filters)
Filters: skip 9:30-10:59 ET, skip 1-2pm ET (dead zone), skip Thursday

Usage (local test / dry-run):
    python ml_intraday_v3/live/ml_live_runner.py --dry-run

Usage (live on GCP VM via Docker):
    python ml_intraday_v3/live/ml_live_runner.py --live --yes
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
# ROOT (/app) must come before ml_intraday_v3 so core.projectx_client
# resolves to /app/core and not the shadow core/ inside ml_intraday_v3.
sys.path.insert(0, str(ROOT / "ml_intraday_v3"))
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH  = ROOT / "ml_intraday_v3" / "models" / "ml_scalper_v3_long.pkl"
BAR_MINUTES = 5
LARGE_TRADE_THRESH = 5      # contracts — matches training
ATR_PERIOD  = 10
FEATURE_COLS = [
    "ofi_imb", "lg_ofi_imb", "ofi_accel",
    "lg_sm_diverge", "kyles_lambda", "roll_spread",
    "large_frac", "trade_rate_z", "avg_size_z", "max_run_z",
    "ofi_early_n", "ofi_late_n",
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body",
    "vol_z", "range_pos",
    "vwap_dev",
    "hour_sin", "hour_cos", "dow",
    "is_open_30", "is_close_30",
    "kyles_lambda_3", "lg_ofi_imb_3", "trade_rate_z_3",
]

LONG_THRESHOLD     = 0.248   # p95 of probability distribution — matches backtest CFG
MAX_TRADES_PER_DAY = 6
# 133 pts × $2/pt × 15 contracts = $3,990 ≈ $4k daily stop
# For practice account ($151k balance) this is ~2.6% max daily loss — reasonable.
# Override via ML_N_CONTRACTS env var or --n-contracts CLI flag.
N_CONTRACTS        = int(os.getenv("ML_N_CONTRACTS", "10"))   # ramp to 15 after 20 validated trades
MAX_DAILY_LOSS_PTS = 133     # pts — scales with N_CONTRACTS automatically
PT_ATR_MULT        = 2.0
SL_ATR_MULT        = 1.0
HORIZON_BARS       = 12
COOLDOWN_BARS      = 3
MIN_ATR_PTS        = 5.0


# ---------------------------------------------------------------------------
# Microstructure feature helpers (mirrors build_microstructure_features.py)
# ---------------------------------------------------------------------------
def compute_bar_features(ticks: list[dict]) -> dict | None:
    """Compute microstructure features from a list of trade tick dicts.

    Each tick: {"ts_ns": int, "price": float, "size": float, "side": str}
      side: "B"=buy aggressor, "A"=sell aggressor
    """
    if not ticks:
        return None

    price = np.array([t["price"] for t in ticks], dtype=np.float64)
    size  = np.array([t["size"]  for t in ticks], dtype=np.float64)
    side  = np.array([t["side"]  for t in ticks])
    ts_ns = np.array([t["ts_ns"] for t in ticks], dtype=np.float64)
    n     = len(ticks)

    buy_mask  = side == "B"
    sell_mask = side == "A"
    buy_vol   = size[buy_mask].sum()
    sell_vol  = size[sell_mask].sum()
    total_vol = size.sum()
    ofi       = buy_vol - sell_vol

    large_mask = size >= LARGE_TRADE_THRESH
    small_mask = ~large_mask
    lg_buy  = size[buy_mask  & large_mask].sum()
    lg_sell = size[sell_mask & large_mask].sum()
    sm_buy  = size[buy_mask  & small_mask].sum()
    sm_sell = size[sell_mask & small_mask].sum()
    lg_vol  = size[large_mask].sum()
    sm_vol  = size[small_mask].sum()
    lg_ofi  = lg_buy  - lg_sell
    sm_ofi  = sm_buy  - sm_sell

    p_open  = price[0]
    p_close = price[-1]
    bar_ret = p_close - p_open

    denom = abs(ofi) if abs(ofi) > 0 else np.nan
    kyles_lambda = abs(bar_ret) / denom if denom else np.nan

    dp = np.diff(price)
    roll_spread = np.nan
    if len(dp) >= 2:
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        roll_spread = 2 * np.sqrt(max(-cov, 0))

    trade_rate = n / (BAR_MINUTES * 60)
    avg_size   = size.mean()
    max_size   = size.max()
    size_std   = size.std() if n > 1 else 0.0
    large_frac = lg_vol / total_vol if total_vol > 0 else 0.0

    # Sub-bar OFI (first/last 10% of bar)
    if ts_ns[-1] > ts_ns[0]:
        dur = ts_ns[-1] - ts_ns[0]
        early_cut = ts_ns[0] + dur * 0.1
        late_start = ts_ns[0] + dur * 0.9
        early = ts_ns <= early_cut
        late  = ts_ns >= late_start
        def _ofi(m):
            return size[m & buy_mask].sum() - size[m & sell_mask].sum()
        ofi_early = _ofi(early)
        ofi_late  = _ofi(late)
        ofi_accel = ofi_late - ofi_early
    else:
        ofi_early = ofi_late = ofi_accel = 0.0

    max_run = 1; cur_run = 1
    for i in range(1, n):
        if side[i] == side[i-1] and side[i] in ("B", "A"):
            cur_run += 1; max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    lg_sm_diverge = float((lg_ofi > 0) != (sm_ofi > 0)) if (lg_vol > 0 and sm_vol > 0) else 0.0
    dv   = (price * size).sum()
    vwap = dv / total_vol if total_vol > 0 else p_close
    ofi_imb    = ofi / total_vol if total_vol > 0 else 0.0
    lg_ofi_imb = lg_ofi / lg_vol if lg_vol > 0 else 0.0

    return {
        "buy_vol": buy_vol, "sell_vol": sell_vol, "total_vol": total_vol,
        "large_vol": lg_vol, "large_frac": large_frac,
        "ofi": ofi, "ofi_imb": ofi_imb,
        "lg_ofi": lg_ofi, "lg_ofi_imb": lg_ofi_imb, "sm_ofi": sm_ofi,
        "ofi_accel": ofi_accel, "ofi_early": ofi_early, "ofi_late": ofi_late,
        "lg_sm_diverge": lg_sm_diverge,
        "kyles_lambda": kyles_lambda, "roll_spread": roll_spread,
        "bar_ret": bar_ret, "n_trades": n, "trade_rate": trade_rate,
        "avg_size": avg_size, "max_size": max_size, "size_std": size_std,
        "max_run": float(max_run),
        "open": p_open, "high": price.max(), "low": price.min(),
        "close": p_close, "vwap": vwap,
    }


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def build_ml_features(bar_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full ML feature set from a DataFrame of completed bars."""
    out  = bar_df.copy()
    c    = out["close"]
    atr  = compute_atr(out)
    out["atr"] = atr
    tv   = out["total_vol"].replace(0, np.nan)

    out["ofi_early_n"] = out["ofi_early"] / tv.fillna(1)
    out["ofi_late_n"]  = out["ofi_late"]  / tv.fillna(1)

    tr_ma = out["trade_rate"].rolling(20).mean()
    tr_sd = out["trade_rate"].rolling(20).std().replace(0, np.nan)
    out["trade_rate_z"] = (out["trade_rate"] - tr_ma) / tr_sd

    sz_ma = out["avg_size"].rolling(20).mean()
    sz_sd = out["avg_size"].rolling(20).std().replace(0, np.nan)
    out["avg_size_z"] = (out["avg_size"] - sz_ma) / sz_sd

    run_ma = out["max_run"].rolling(20).mean()
    run_sd = out["max_run"].rolling(20).std().replace(0, np.nan)
    out["max_run_z"] = (out["max_run"] - run_ma) / run_sd

    out["kyles_lambda"] = out["kyles_lambda"].clip(0, 2.0)
    out["kyles_lambda_3"]  = out["kyles_lambda"].rolling(3).mean()
    out["lg_ofi_imb_3"]    = out["lg_ofi_imb"].rolling(3).mean()
    out["trade_rate_z_3"]  = out["trade_rate_z"].rolling(3).mean()

    for n in [1, 3, 6, 12]:
        out[f"ret_{n}"] = np.log(c / c.shift(n))

    out["rsi_5"]  = rsi(c, 5)
    out["rsi_14"] = rsi(c, 14)

    ema9  = c.ewm(span=9,  min_periods=9).mean()
    ema21 = c.ewm(span=21, min_periods=21).mean()
    out["ema9_ratio"]    = (c / ema9  - 1) * 100
    out["ema21_ratio"]   = (c / ema21 - 1) * 100
    out["ema9_21_cross"] = (ema9 / ema21 - 1) * 100

    out["norm_range"] = (out["high"] - out["low"]) / atr.replace(0, np.nan)
    out["norm_body"]  = (c - out["open"]) / atr.replace(0, np.nan)

    vol_ma = tv.rolling(20).mean()
    vol_sd = tv.rolling(20).std().replace(0, np.nan)
    out["vol_z"] = (tv - vol_ma) / vol_sd

    high20 = out["high"].rolling(20).max()
    low20  = out["low"].rolling(20).min()
    out["range_pos"] = (c - low20) / (high20 - low20).replace(0, np.nan)

    out["vwap_dev"] = (c - out["vwap"]) / atr.replace(0, np.nan)

    h_et = (out.index.hour - 5) % 24
    out["hour_sin"]    = np.sin(2 * np.pi * h_et / 24)
    out["hour_cos"]    = np.cos(2 * np.pi * h_et / 24)
    out["dow"]         = out.index.dayofweek.astype(float)
    mins = (h_et - 9) * 60 + out.index.minute - 30
    out["is_open_30"]  = (mins <= 30).astype(float)
    out["is_close_30"] = (mins >= 360).astype(float)

    return out


# ---------------------------------------------------------------------------
# Trade + DOM stream listener
# ---------------------------------------------------------------------------
class MarketStreamListener:
    """Single SignalR connection subscribing to GatewayTrade + GatewayDepth."""

    DOM_TYPE_ASK      = 1
    DOM_TYPE_BID      = 2
    DOM_TYPE_BEST_ASK = 3
    DOM_TYPE_BEST_BID = 4

    def __init__(self, contract_id: str, hub_url: str, token: str, top_dom_levels: int = 5):
        self.contract_id   = contract_id
        self.hub_url       = hub_url
        self.token         = token
        self.top_dom_levels = top_dom_levels

        # Trade tick buffer: bar_ts (floor to 5min) → list of ticks
        self._ticks: dict[int, list] = defaultdict(list)
        self._tick_lock = threading.Lock()

        # L2 DOM state
        self._bids: dict[float, int] = defaultdict(int)
        self._asks: dict[float, int] = defaultdict(int)
        self._dom_lock  = threading.Lock()
        self._dom_ofi   = 0.0
        self._best_bid  = 0.0
        self._best_ask  = 0.0

        self._connected   = False
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MarketStream")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_dom_ofi(self) -> float:
        return self._dom_ofi

    def get_spread(self) -> float:
        return max(0.0, self._best_ask - self._best_bid)

    def pop_bar_ticks(self, bar_ts_ns: int) -> list:
        """Return and clear ticks for a completed bar timestamp."""
        with self._tick_lock:
            return self._ticks.pop(bar_ts_ns, [])

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect()
            except Exception as e:
                logger.warning(f"MarketStream error: {e}")
            finally:
                self._connected = False
            if not self._stop_event.is_set():
                logger.info("MarketStream reconnecting in 5s...")
                time.sleep(5)

    def _connect(self) -> None:
        try:
            from signalrcore.hub_connection_builder import HubConnectionBuilder
        except ImportError:
            raise ImportError("pip install signalrcore")

        def _on_open():
            # Send subscriptions only after the SignalR handshake completes,
            # not immediately after conn.start() — avoids SSL BAD_LENGTH race.
            try:
                conn.send("SubscribeContractTrades", [self.contract_id])
                logger.info(f"Subscribed to GatewayTrade for {self.contract_id}")
            except Exception as e:
                logger.error(f"GatewayTrade subscription failed: {e}")
                return
            try:
                conn.send("SubscribeContractMarketDepth", [self.contract_id])
                logger.info("Subscribed to GatewayDepth (L2) — optional")
            except Exception as e:
                logger.info(f"GatewayDepth skipped (L2 not required): {e}")
            self._set_connected(True)

        conn = (
            HubConnectionBuilder()
            .with_url(self.hub_url, options={"headers": {"Authorization": f"Bearer {self.token}"}})
            .build()
        )
        conn.on("GatewayTrade",  self._on_trade)
        conn.on("GatewayDepth",  self._on_depth)
        conn.on_open(_on_open)
        conn.on_close(lambda: self._set_connected(False))
        conn.start()

        while not self._stop_event.is_set():
            time.sleep(1)
        conn.stop()

    def _set_connected(self, val: bool) -> None:
        self._connected = val
        if not val:
            self._dom_ofi = 0.0

    def _on_trade(self, args: list) -> None:
        """Handle GatewayTrade — type 0=Buy, 1=Sell aggressor."""
        data = args[1] if len(args) > 1 else (args[0] if args else {})
        try:
            price  = float(data.get("price", 0))
            size   = float(data.get("volume", 0))
            t_type = int(data.get("type", -1))
            ts_str = data.get("timestamp", "")
            if price <= 0 or size <= 0:
                return
            side = "B" if t_type == 0 else ("A" if t_type == 1 else "N")
            # Snap to 5-min bar (nanoseconds)
            ts_ns  = int(pd.Timestamp(ts_str, tz="UTC").value) if ts_str else time.time_ns()
            bar_ts = (ts_ns // (BAR_MINUTES * 60 * 10**9)) * (BAR_MINUTES * 60 * 10**9)
            with self._tick_lock:
                self._ticks[bar_ts].append({"ts_ns": ts_ns, "price": price, "size": size, "side": side})
        except Exception as e:
            logger.debug(f"Trade parse error: {e}")

    def _on_depth(self, args: list) -> None:
        # GatewayDepth: args = [contract_id, [entry, ...]] or [entry, ...]
        raw = args[1] if len(args) > 1 else args[0] if args else []
        entries = raw if isinstance(raw, list) else [raw]
        for data in entries:
            if not isinstance(data, dict):
                continue
            try:
                dom_type = int(data.get("type", 0))
                price    = float(data.get("price", 0))
                volume   = int(data.get("volume", 0) or data.get("currentVolume", 0))
                if price <= 0:
                    continue
                with self._dom_lock:
                    if dom_type in (self.DOM_TYPE_BID, self.DOM_TYPE_BEST_BID):
                        if dom_type == self.DOM_TYPE_BEST_BID:
                            self._best_bid = price
                        if volume == 0:
                            self._bids.pop(price, None)
                        else:
                            self._bids[price] = volume
                    elif dom_type in (self.DOM_TYPE_ASK, self.DOM_TYPE_BEST_ASK):
                        if dom_type == self.DOM_TYPE_BEST_ASK:
                            self._best_ask = price
                        if volume == 0:
                            self._asks.pop(price, None)
                        else:
                            self._asks[price] = volume
                    self._dom_ofi = self._compute_dom_ofi()
            except Exception as e:
                logger.debug(f"Depth parse error: {e}")

    def _compute_dom_ofi(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        top_bids = sorted(self._bids.keys(), reverse=True)[:self.top_dom_levels]
        top_asks = sorted(self._asks.keys())[:self.top_dom_levels]
        bid_vol  = sum(self._bids[p] for p in top_bids)
        ask_vol  = sum(self._asks[p] for p in top_asks)
        total    = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
class MLLiveRunner:
    def __init__(self, dry_run: bool = True, contract_id: str | None = None,
                 account_id: str | None = None):
        self.dry_run     = dry_run
        self.contract_id = contract_id or os.getenv("TOPSTEPX_CONTRACT_ID")
        self.account_id  = account_id  or os.getenv("TOPSTEPX_ACCOUNT_ID")

        # Load model
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded")

        # Bar history for rolling features (keep last 50 bars)
        self._bar_history: deque = deque(maxlen=50)
        self._bar_lock = threading.Lock()

        # Risk state
        self._trades_today   = 0
        self._daily_loss_pts = 0.0
        self._current_date   = None
        self._in_trade       = False
        self._active_trade   = None
        self._cooldown_bars  = 0
        self._bars_since_entry = 0

        self._running = False
        self._stream: MarketStreamListener | None = None
        self._client = None

    def run(self) -> None:
        self._running = True

        # Set up ProjectX client for order execution
        from core.projectx_client import ProjectXClient
        self._client = ProjectXClient(
            contract_id=self.contract_id,
            account_id=self.account_id,
        )
        token    = self._client._token
        base_url = os.getenv("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")
        # SignalR hub: swap REST base for WebSocket hub path
        hub_url  = base_url.rstrip("/").replace("api.topstepx.com", "rtc.topstepx.com") + "/hubs/market"
        logger.info(f"Hub URL: {hub_url}")

        self._stream = MarketStreamListener(
            contract_id=self.contract_id,
            hub_url=hub_url,
            token=token,
        )
        self._stream.start()

        # Graceful shutdown on SIGTERM
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        signal.signal(signal.SIGINT,  lambda s, f: self.stop())

        logger.info(f"ML live runner started — {'DRY RUN' if self.dry_run else 'LIVE'}")

        # Wait for connection
        for _ in range(30):
            if self._stream.is_connected:
                break
            time.sleep(1)
        if not self._stream.is_connected:
            logger.warning("Stream not connected after 30s — continuing anyway")

        # Main bar timer loop — fires at each 5-min boundary
        while self._running:
            now_utc = datetime.now(timezone.utc)
            # Time to next 5-min bar close (with 2s buffer for late ticks)
            seconds_into_period = (now_utc.minute % BAR_MINUTES) * 60 + now_utc.second
            wait = (BAR_MINUTES * 60 - seconds_into_period) + 2
            time.sleep(wait)

            if not self._running:
                break

            # The bar that just closed
            bar_close = datetime.now(timezone.utc)
            bar_ts_m  = bar_close.replace(second=0, microsecond=0)
            bar_ts_m  = bar_ts_m.replace(minute=(bar_ts_m.minute // BAR_MINUTES) * BAR_MINUTES)
            bar_ts_ns = int(bar_ts_m.timestamp() * 1e9) - (BAR_MINUTES * 60 * 10**9)

            self._process_bar_close(bar_ts_ns, bar_ts_m)

        self._stream.stop()
        logger.info("ML runner stopped")

    def stop(self) -> None:
        self._running = False

    def _process_bar_close(self, bar_ts_ns: int, bar_time: datetime) -> None:
        ticks = self._stream.pop_bar_ticks(bar_ts_ns)
        if not ticks:
            logger.debug(f"No ticks for bar {bar_time}")
            return

        feats = compute_bar_features(ticks)
        if feats is None:
            return

        feats["ts"] = pd.Timestamp(bar_time, tz="UTC")

        with self._bar_lock:
            self._bar_history.append(feats)
            if len(self._bar_history) < 22:  # need 20+ bars for rolling features
                logger.debug(f"Warming up: {len(self._bar_history)} bars")
                return
            bar_df = pd.DataFrame(list(self._bar_history))
            bar_df = bar_df.set_index("ts")

        # Build ML features
        try:
            feat_df = build_ml_features(bar_df)
        except Exception as e:
            logger.error(f"Feature build error: {e}")
            return

        latest = feat_df.iloc[-1]
        atr_val = latest.get("atr", np.nan)

        # Daily reset
        now_date = bar_time.date()
        if now_date != self._current_date:
            self._trades_today   = 0
            self._daily_loss_pts = 0.0
            self._current_date   = now_date
            self._cooldown_bars  = 0
            logger.info(f"New day: {now_date} — risk state reset")

        # Check fill on active trade
        if self._in_trade and not self.dry_run:
            self._check_fill(latest)

        # Cooldown
        if self._cooldown_bars > 0:
            self._cooldown_bars -= 1
            return

        # Risk gates
        if self._in_trade:
            return
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return
        if self._daily_loss_pts <= -MAX_DAILY_LOSS_PTS:
            return
        if np.isnan(atr_val) or atr_val < MIN_ATR_PTS:
            return

        # Time filter: skip 9:30-10:59 ET (UTC hour 14-15)
        h_et = (bar_time.hour - 5) % 24
        if h_et < 11:
            logger.debug(f"Skipping {bar_time} — first hour ET")
            return

        # Time filter: skip 1-2pm ET (OOS WR=25%, no edge)
        if h_et == 13:
            logger.debug(f"Skipping {bar_time} — 1-2pm ET dead zone")
            return

        # Day filter: skip Thursday
        if bar_time.weekday() == 3:
            logger.debug(f"Skipping {bar_time} — Thursday")
            return

        # Session: only RTH 9:30-15:55 ET
        if not (9 <= h_et < 16):
            return

        # Predict
        row = feat_df.iloc[[-1]][FEATURE_COLS]
        if row.isna().any(axis=1).iloc[0]:
            logger.debug("Skipping — NaN features")
            return

        p_long = self.model.predict_proba(row.astype(np.float32))[0, 1]

        dom_ofi = self._stream.get_dom_ofi()
        logger.info(
            f"Bar {bar_time}  p_long={p_long:.4f}  threshold={LONG_THRESHOLD:.3f}  "
            f"dom_ofi={dom_ofi:+.3f}  atr={atr_val:.1f}  trades_today={self._trades_today}"
        )

        if p_long >= LONG_THRESHOLD:
            self._enter_long(latest, atr_val, bar_time)

    def _enter_long(self, bar: pd.Series, atr: float, bar_time: datetime) -> None:
        entry     = float(bar["close"])
        pt        = round(entry + PT_ATR_MULT * atr, 2)
        sl        = round(entry - SL_ATR_MULT * atr, 2)
        tick_size = 0.25  # MNQ

        logger.info(
            f"LONG signal → entry={entry:.2f}  PT={pt:.2f}  SL={sl:.2f}  "
            f"ATR={atr:.1f}  {'DRY RUN' if self.dry_run else 'LIVE ORDER'}"
        )

        if not self.dry_run:
            from core.projectx_client import BracketInstruction
            pt_ticks = round((pt - entry) / tick_size)
            sl_ticks = round((entry - sl)  / tick_size)
            brackets = dict(
                stop_loss_bracket=BracketInstruction(ticks=-sl_ticks),
                take_profit_bracket=BracketInstruction(ticks=pt_ticks),
                account_id=int(self.account_id),
                contract_id=self.contract_id,
            )

            filled = self._place_with_limit_fallback(entry, pt_ticks, sl_ticks, brackets)
            if not filled:
                return

        self._in_trade  = True
        self._active_trade = {
            "entry": entry, "atr": atr, "pt": pt, "sl": sl,
            "entry_time": bar_time, "bar_n": 0,
        }
        self._trades_today  += 1
        self._bars_since_entry = 0

    def _place_with_limit_fallback(self, entry: float, pt_ticks: int,
                                   sl_ticks: int, brackets: dict) -> bool:
        """Try a limit order at close price; fall back to market after 10s."""
        from core.projectx_client import BracketInstruction
        LIMIT_TIMEOUT = 10  # seconds

        # 1. Try limit at close price — avoids paying the bid-ask spread
        try:
            order = self._client.place_order(
                symbol="", side="BUY", quantity=N_CONTRACTS,
                order_type="LIMIT",
                take_profit=entry,  # sets limitPrice on the order body
                **brackets,
            )
            order_id = order.order_id
            logger.info(f"Limit order placed id={order_id} at {entry:.2f} "
                        f"(timeout {LIMIT_TIMEOUT}s)")
        except Exception as e:
            logger.error(f"Limit order placement failed: {e}")
            return False

        # 2. Poll for fill — position appears when limit fills
        deadline = time.time() + LIMIT_TIMEOUT
        while time.time() < deadline:
            time.sleep(1)
            try:
                if self._client.search_open_positions():
                    logger.info("Limit order filled — saved spread vs market")
                    return True
                # Check order still exists; if gone without position → rejected
                open_orders = self._client.search_open_orders()
                if not any(str(o.order_id) == str(order_id) for o in open_orders):
                    logger.warning("Limit order disappeared without fill — skipping")
                    return False
            except Exception as e:
                logger.warning(f"Fill check error: {e}")

        # 3. Limit timed out — cancel and fall back to market
        try:
            self._client.cancel_order(order_id)
            logger.info("Limit timed out — cancelled, falling back to market")
        except Exception as e:
            logger.warning(f"Cancel failed (may have raced to fill): {e}")
            # If cancel failed, check once more for a position
            try:
                if self._client.search_open_positions():
                    logger.info("Filled during cancel — keeping position")
                    return True
            except Exception:
                pass
            return False

        # 4. Market fallback
        try:
            order = self._client.place_order(
                symbol="", side="BUY", quantity=N_CONTRACTS,
                order_type="MARKET",
                **brackets,
            )
            logger.info(f"Market fallback placed id={order.order_id}")
            return True
        except Exception as e:
            logger.error(f"Market fallback failed: {e}")
            return False

    def _check_fill(self, latest: pd.Series) -> None:
        """Poll open positions to detect fill/exit."""
        try:
            positions = self._client.search_open_positions()
        except Exception as e:
            logger.error(f"Position check error: {e}")
            return

        if not positions:
            # Position closed — record PnL
            if self._active_trade:
                close_price = float(latest.get("close", self._active_trade["entry"]))
                pnl_pts = close_price - self._active_trade["entry"]
                self._daily_loss_pts += pnl_pts
                logger.info(f"Trade closed  PnL_pts={pnl_pts:+.2f}  daily_loss_pts={self._daily_loss_pts:.2f}")
            self._in_trade     = False
            self._active_trade = None
            self._cooldown_bars = COOLDOWN_BARS
        else:
            self._bars_since_entry += 1
            if self._bars_since_entry >= HORIZON_BARS and not self.dry_run:
                # Time stop — close manually
                logger.info("Time stop reached — closing position")
                try:
                    self._client.place_order(
                        contract_id=self.contract_id,
                        account_id=self.account_id,
                        side=1,  # Ask = sell
                        size=N_CONTRACTS,
                    )
                except Exception as e:
                    logger.error(f"Time stop close error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ML Scalper Live Runner")
    parser.add_argument("--dry-run",     action="store_true", default=True)
    parser.add_argument("--live",        action="store_true")
    parser.add_argument("--yes",         action="store_true")
    parser.add_argument("--contract-id", type=str)
    parser.add_argument("--account-id",  type=str)
    parser.add_argument("--n-contracts", type=int, default=None,
                        help="Number of MNQ contracts per trade (default: ML_N_CONTRACTS env or 15)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.n_contracts is not None:
        global N_CONTRACTS
        N_CONTRACTS = args.n_contracts
        logger.info(f"N_CONTRACTS overridden to {N_CONTRACTS}")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/app/logs/ml_runner.log" if Path("/app/logs").exists()
                                else "ml_runner.log"),
        ],
    )

    dry_run = not args.live
    if not dry_run:
        logger.warning("LIVE TRADING MODE — real orders will be placed!")
        if not args.yes:
            if input("Type CONFIRM to proceed: ") != "CONFIRM":
                return

    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.error("Run ml_intraday_v3/scripts/ml_scalper_v3.py first to train and save the model.")
        sys.exit(1)

    runner = MLLiveRunner(
        dry_run=dry_run,
        contract_id=args.contract_id,
        account_id=args.account_id,
    )
    runner.run()


if __name__ == "__main__":
    main()
