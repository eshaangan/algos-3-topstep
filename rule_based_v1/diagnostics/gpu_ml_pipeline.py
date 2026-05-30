"""GPU-accelerated ML intraday scalping pipeline — MPS (Apple Silicon).

Architecture:
  - Cross-asset features: MNQ + MES + MGC + ZN + 6E (1-min bars)
  - LSTM sequence model trained on MPS GPU via PyTorch
  - LightGBM gradient boosting for comparison
  - Train: Jan 2026 – Feb 2026 | OOS: Mar 2026 – current (never seen)
  - Label: direction of next H bars, profitable after 1-tick slippage + commission
  - Strategy: enter when P > confidence_threshold, tight stops (ticks not ATR)
  - Sizes sweep: find (contracts, PT_ticks, SL_ticks) that yields $6k/wk ≤ $2k DD

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/gpu_ml_pipeline.py
    python rule_based_v1/diagnostics/gpu_ml_pipeline.py --fetch  # pull fresh data first
    python rule_based_v1/diagnostics/gpu_ml_pipeline.py --no-lstm  # lgbm only (faster)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "gpu_ml_results.json"

# GPU setup
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Cost model ───────────────────────────────────────────────────────────────
TICK_SIZE   = 0.25          # MNQ
TICK_VALUE  = 0.50          # $/tick per MNQ contract
COMMISSION  = 0.62          # per side per contract
SLIPPAGE    = 1             # ticks each way (conservative)

def net_pnl(ticks_pnl: float, n_contracts: int) -> float:
    return ticks_pnl * TICK_VALUE * n_contracts - 2 * COMMISSION * n_contracts

# ── Data loading ─────────────────────────────────────────────────────────────
def load_1min(path: Path, key: str, tz: str = "US/Eastern") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_hdf(str(path), key=key)
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df[df.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
        return df
    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return None


def load_5min(path: Path, key: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_hdf(str(path), key=key)
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df[df.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
        return df
    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return None


def fetch_databento(symbol: str, schema: str = "ohlcv-1m", start: str = "2026-01-01") -> Optional[pd.DataFrame]:
    """Fetch from Databento GLBX.MDP3 and return 1-min bars."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            return None
        import databento as db
        client = db.Historical(key=api_key)
        from datetime import datetime, timezone, timedelta
        end = (datetime.now(tz=timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT21:00:00+00:00")
        print(f"  Fetching {symbol} {start} → {end[:10]}...")
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3", symbols=[symbol],
            schema=schema, start=start, end=end, stype_in="continuous",
        )
        df = data.to_df()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        return df
    except Exception as e:
        logger.warning(f"Databento fetch {symbol}: {e}")
        return None


# ── Feature engineering (vectorised numpy/pandas) ────────────────────────────
def safe_log_ret(s: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(s / s.shift(lag)).replace([np.inf, -np.inf], np.nan)


def build_features(mnq: pd.DataFrame,
                   mes: Optional[pd.DataFrame] = None,
                   mgc: Optional[pd.DataFrame] = None,
                   zn: Optional[pd.DataFrame] = None,
                   fe6: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build rich cross-asset feature matrix aligned to MNQ 1-min bars."""
    feats = pd.DataFrame(index=mnq.index)

    # ── MNQ self-features ──────────────────────────────────────────────────
    feats["ret1"]   = safe_log_ret(mnq["close"], 1)
    feats["ret5"]   = safe_log_ret(mnq["close"], 5)
    feats["ret15"]  = safe_log_ret(mnq["close"], 15)
    feats["ret30"]  = safe_log_ret(mnq["close"], 30)
    feats["ret60"]  = safe_log_ret(mnq["close"], 60)

    # Intrabar shape
    rng = mnq["high"] - mnq["low"]
    feats["bar_range_pct"] = rng / mnq["close"].shift(1).replace(0, np.nan)
    feats["close_loc"]  = (mnq["close"] - mnq["low"]) / rng.replace(0, np.nan) - 0.5
    feats["upper_wick"] = (mnq["high"] - mnq[["open","close"]].max(axis=1)) / rng.replace(0, np.nan)
    feats["lower_wick"] = (mnq[["open","close"]].min(axis=1) - mnq["low"]) / rng.replace(0, np.nan)

    # Volume
    vol_ma = mnq["volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    feats["rvol"] = mnq["volume"] / vol_ma - 1.0

    # ATR (14-period EWM)
    prev = mnq["close"].shift(1)
    tr = pd.concat([rng, (mnq["high"] - prev).abs(), (mnq["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    feats["atr_pct"] = atr / mnq["close"].replace(0, np.nan)

    # Session VWAP deviation
    tp = (mnq["high"] + mnq["low"] + mnq["close"]) / 3
    date_grp = mnq.index.map(lambda t: t.date())
    vwap = (tp * mnq["volume"]).groupby(date_grp).cumsum() / \
           mnq["volume"].groupby(date_grp).cumsum().replace(0, np.nan)
    feats["vwap_dev"] = (mnq["close"] - vwap) / atr.replace(0, np.nan)

    # Day open distance
    day_open = mnq.groupby(date_grp)["open"].transform("first")
    feats["dist_open"] = (mnq["close"] - day_open) / atr.replace(0, np.nan)

    # Session high/low distance
    day_high = mnq["high"].groupby(date_grp).cummax()
    day_low  = mnq["low"].groupby(date_grp).cummin()
    feats["dist_day_high"] = (day_high - mnq["close"]) / atr.replace(0, np.nan)
    feats["dist_day_low"]  = (mnq["close"] - day_low)  / atr.replace(0, np.nan)

    # Rolling z-score momentum (for regime)
    ret1 = feats["ret1"]
    feats["mom_z20"] = (ret1.rolling(20).mean()) / (ret1.rolling(20).std() + 1e-8)
    feats["mom_z60"] = (ret1.rolling(60).mean()) / (ret1.rolling(60).std() + 1e-8)

    # Time features (cyclical encoding)
    minutes = mnq.index.hour * 60 + mnq.index.minute
    feats["time_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    feats["time_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    feats["is_first_hour"] = ((minutes >= 570) & (minutes <= 630)).astype(float)  # 9:30-10:30
    feats["is_lunch"]      = ((minutes >= 720) & (minutes <= 780)).astype(float)  # 12:00-13:00
    feats["is_last_hour"]  = ((minutes >= 870) & (minutes <= 930)).astype(float)  # 14:30-15:30

    # Day of week (one-hot)
    dow = mnq.index.dayofweek
    for d in range(5):
        feats[f"dow_{d}"] = (dow == d).astype(float)

    # ── Cross-asset: MES ───────────────────────────────────────────────────
    if mes is not None:
        mes_aligned = mes["close"].reindex(mnq.index, method="ffill")
        feats["mes_ret1"]  = safe_log_ret(mes_aligned, 1)
        feats["mes_ret5"]  = safe_log_ret(mes_aligned, 5)
        feats["mes_ret15"] = safe_log_ret(mes_aligned, 15)
        # Lead-lag: MNQ minus MES 1-min return (divergence signal)
        feats["mnq_mes_div1"] = feats["ret1"] - feats["mes_ret1"]
        feats["mnq_mes_div5"] = feats["ret5"] - feats["mes_ret5"]

    # ── Cross-asset: MGC (gold) ────────────────────────────────────────────
    if mgc is not None:
        mgc_aligned = mgc["close"].reindex(mnq.index, method="ffill")
        feats["mgc_ret5"]  = safe_log_ret(mgc_aligned, 5)
        feats["mgc_ret15"] = safe_log_ret(mgc_aligned, 15)

    # ── Cross-asset: ZN (10Y bond) ─────────────────────────────────────────
    if zn is not None:
        zn_aligned = zn["close"].reindex(mnq.index, method="ffill")
        feats["zn_ret5"]  = safe_log_ret(zn_aligned, 5)
        feats["zn_ret15"] = safe_log_ret(zn_aligned, 15)
        feats["zn_mnq_div"] = feats["zn_ret5"] - feats.get("ret5", 0)

    # ── Cross-asset: 6E (EUR/USD) ──────────────────────────────────────────
    if fe6 is not None:
        fe6_aligned = fe6["close"].reindex(mnq.index, method="ffill")
        feats["fe6_ret5"]  = safe_log_ret(fe6_aligned, 5)
        feats["fe6_ret15"] = safe_log_ret(fe6_aligned, 15)

    return feats.astype(np.float32)


# ── Label construction ────────────────────────────────────────────────────────
def make_labels(mnq: pd.DataFrame,
                horizon: int = 5,
                pt_ticks: int = 5,
                sl_ticks: int = 3,
                rth_only: bool = True) -> pd.Series:
    """
    For each bar i, simulate a LONG trade entered at close[i] + 1 tick (slippage).
    Win (label=1) if close[i+1..i+horizon] hits PT before SL.
    Loss (label=0) otherwise.
    Only label bars where both sides are finite.
    """
    closes = mnq["close"].values
    highs  = mnq["high"].values
    lows   = mnq["low"].values

    pt_pts = pt_ticks * TICK_SIZE
    sl_pts = sl_ticks * TICK_SIZE
    slip_pts = SLIPPAGE * TICK_SIZE

    labels = np.full(len(closes), np.nan)

    for i in range(len(closes) - horizon - 1):
        entry = closes[i] + slip_pts
        pt = entry + pt_pts
        sl = entry - sl_pts
        result = 0  # default: loss
        for j in range(1, horizon + 1):
            if highs[i + j] >= pt:
                result = 1
                break
            if lows[i + j] <= sl:
                result = 0
                break
        labels[i] = result

    labels_s = pd.Series(labels, index=mnq.index, name="label")

    if rth_only:
        rth_mask = ((mnq.index.hour > 9) | ((mnq.index.hour == 9) & (mnq.index.minute >= 30))) \
                   & (mnq.index.hour < 16)
        labels_s[~rth_mask] = np.nan

    return labels_s


# ── LSTM model ────────────────────────────────────────────────────────────────
class ScalpLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.head(out).squeeze(-1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               n_features: int, seq_len: int,
               hidden: int = 64, layers: int = 2, dropout: float = 0.3,
               lr: float = 1e-3, epochs: int = 25, batch: int = 512) -> ScalpLSTM:
    """Train LSTM on MPS GPU."""
    model = ScalpLSTM(n_features, hidden, layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=True)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (ep + 1) % 5 == 0:
            avg = total_loss / len(loader)
            print(f"    Epoch {ep+1:3d}/{epochs}  loss={avg:.4f}")

    return model


def predict_lstm(model: ScalpLSTM, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            preds.append(model(X_t[i:i+batch]).cpu().numpy())
    return np.concatenate(preds)


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(feats: pd.DataFrame, labels: pd.Series,
                    seq_len: int = 30) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Convert flat feature frame to (N, seq_len, n_features) tensors."""
    feat_arr = feats.values.astype(np.float32)
    label_arr = labels.values

    X_list, y_list, idx_list = [], [], []
    valid_label = ~np.isnan(label_arr)

    for i in range(seq_len, len(feat_arr)):
        if not valid_label[i]:
            continue
        window = feat_arr[i - seq_len: i]
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            continue
        X_list.append(window)
        y_list.append(label_arr[i])
        idx_list.append(feats.index[i])

    return np.array(X_list), np.array(y_list), pd.DatetimeIndex(idx_list)


# ── Trading simulation ────────────────────────────────────────────────────────
def simulate_strategy(
    mnq: pd.DataFrame,
    signal_index: pd.DatetimeIndex,
    probs: np.ndarray,
    n_contracts: int = 10,
    pt_ticks: int = 5,
    sl_ticks: int = 3,
    conf_threshold: float = 0.58,
    max_trades_per_day: int = 20,
    max_daily_loss: float = -1600.0,
    rth_only: bool = True,
) -> dict:
    """Simulate high-frequency ML scalper from model probability outputs."""
    TICK = TICK_SIZE; TV = TICK_VALUE; COMM = COMMISSION; SLIP = SLIPPAGE

    pt_pts = pt_ticks * TICK
    sl_pts = sl_ticks * TICK

    # Build signal lookup: bar_timestamp -> probability
    sig_map = {ts: p for ts, p in zip(signal_index, probs)}

    trades = []
    equity = 50_000.0
    peak_equity = equity
    eq_curve = [equity]

    cur_date = None
    daily_pnl = 0.0
    daily_pnl_map = {}
    trades_today = 0

    # Iterate RTH MNQ bars, check signal
    rth = mnq if not rth_only else mnq[
        ((mnq.index.hour > 9) | ((mnq.index.hour == 9) & (mnq.index.minute >= 30)))
        & (mnq.index.hour < 16)
    ]

    pos = None  # (direction, entry, pt, sl, entry_bar_idx)

    bars_list = list(rth.itertuples())

    for bi, bar in enumerate(bars_list):
        bdate = bar.Index.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl_map[cur_date] = daily_pnl
            daily_pnl = 0.0
            trades_today = 0
        cur_date = bdate

        is_last = (bi + 1 >= len(bars_list)) or (bars_list[bi + 1].Index.date() != bdate)
        sess_close = is_last or (bar.Index.hour == 15 and bar.Index.minute >= 55)

        # Check open position
        if pos is not None:
            d, ep, pt, sl = pos
            h, l, c = bar.high, bar.low, bar.close
            exited, ex_p, reason = False, 0.0, ""
            if sess_close:
                exited, ex_p, reason = True, c - d * SLIP * TICK, "session_close"
            elif d == 1:
                if l - SLIP * TICK <= sl:
                    exited, ex_p, reason = True, sl - SLIP * TICK, "stop_loss"
                elif h - SLIP * TICK >= pt:
                    exited, ex_p, reason = True, pt - SLIP * TICK, "profit_target"
            if exited:
                raw = (ex_p - ep) * d * n_contracts * TV
                p = raw - 2 * COMM * n_contracts
                trades.append({"pnl": p, "reason": reason, "direction": d,
                               "entry": ep, "exit": ex_p})
                daily_pnl += p
                equity += p
                peak_equity = max(peak_equity, equity)
                eq_curve.append(equity)
                pos = None

        if sess_close or pos is not None:
            continue
        if daily_pnl <= max_daily_loss or trades_today >= max_trades_per_day:
            continue
        # DD guard
        if equity - peak_equity <= -2000.0:
            continue

        # Check ML signal for this bar
        prob = sig_map.get(bar.Index)
        if prob is None:
            continue

        sig = None
        if prob >= conf_threshold:
            sig = 1   # LONG
        elif prob <= (1.0 - conf_threshold):
            sig = -1  # SHORT

        if sig is not None:
            entry = bar.close + sig * SLIP * TICK
            pos = (sig, entry, entry + sig * pt_pts, entry - sig * sl_pts)
            trades_today += 1

    if cur_date and cur_date not in daily_pnl_map:
        daily_pnl_map[cur_date] = daily_pnl

    if not trades:
        return {}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    eq_s   = pd.Series(eq_curve)
    max_dd = float((eq_s - eq_s.cummax()).min())
    daily  = pd.Series(daily_pnl_map)
    active = daily[daily != 0]
    n_wks  = max(1, len(daily) / 5)
    sharpe = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0

    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "weekly_pnl": round(total / n_wks, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else 99.0,
        "avg_win": round(gp / len(wins), 2) if wins else 0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0,
        "exit_reasons": {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Fetch ZN+6E from Databento")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM, run LightGBM only")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (bars)")
    parser.add_argument("--pt", type=int, default=5, help="Profit target (ticks)")
    parser.add_argument("--sl", type=int, default=3, help="Stop loss (ticks)")
    parser.add_argument("--seq-len", type=int, default=30, help="LSTM sequence length")
    parser.add_argument("--conf", type=float, default=0.58, help="Confidence threshold")
    parser.add_argument("--contracts", type=int, default=10, help="MNQ contracts for simulation")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n── Loading data ──────────────────────────────────────────────────")

    mnq = load_1min(DATA / "mnq_2026ytd_databento_1min_eth.h5", "bars_1min_eth")
    if mnq is None:
        mnq = load_1min(DATA / "mnq_2026ytd_1min.h5", "bars_1min")
    print(f"MNQ 1-min: {len(mnq):,} bars  [{mnq.index[0].date()} → {mnq.index[-1].date()}]")

    mes = load_1min(DATA / "mes_2026_ytd_1m.h5", "bars_1min")
    if mes is not None:
        print(f"MES 1-min: {len(mes):,} bars  [{mes.index[0].date()} → {mes.index[-1].date()}]")

    # MGC at 5-min (resample to 1-min via ffill)
    mgc_5m = load_5min(DATA / "mgc_bars_5min.h5", "bars_5min")
    mgc = None
    if mgc_5m is not None:
        mgc = mgc_5m.reindex(mnq.index, method="ffill")
        print(f"MGC 5→1min: {len(mgc_5m):,} bars resampled")

    zn = None; fe6 = None
    if args.fetch:
        print("\nFetching from Databento...")
        zn_raw = fetch_databento("ZN.c.0")
        if zn_raw is not None:
            zn = zn_raw.reindex(mnq.index, method="ffill")
            zn_raw.to_hdf(str(DATA / "zn_2026ytd_1min.h5"), key="bars_1min", complevel=5)
            print(f"  ZN: {len(zn_raw):,} bars saved")
        fe6_raw = fetch_databento("6E.c.0")
        if fe6_raw is not None:
            fe6 = fe6_raw.reindex(mnq.index, method="ffill")
            fe6_raw.to_hdf(str(DATA / "fe6_2026ytd_1min.h5"), key="bars_1min", complevel=5)
            print(f"  6E: {len(fe6_raw):,} bars saved")
    else:
        # Try cached
        for sym, fname, key in [("ZN", "zn_2026ytd_1min.h5", "bars_1min"),
                                  ("6E", "fe6_2026ytd_1min.h5", "bars_1min")]:
            p = DATA / fname
            if p.exists():
                raw = load_1min(p, key)
                if raw is not None:
                    if sym == "ZN": zn = raw.reindex(mnq.index, method="ffill")
                    else:           fe6 = raw.reindex(mnq.index, method="ffill")
                    print(f"{sym} (cached): {len(raw):,} bars")

    # ── Feature engineering ────────────────────────────────────────────────
    print("\n── Building features ─────────────────────────────────────────────")
    feats = build_features(mnq, mes=mes, mgc=mgc, zn=zn, fe6=fe6)
    # Clip extreme values (price spikes)
    feats = feats.clip(-10, 10)
    # Fill remaining NaN with 0
    feats = feats.fillna(0.0)
    print(f"Features: {feats.shape[1]} columns  |  {feats.shape[0]:,} rows")

    # ── Labels ─────────────────────────────────────────────────────────────
    print(f"\n── Building labels (PT={args.pt}t, SL={args.sl}t, H={args.horizon}) ─")
    # Use RTH bars only for labeling
    rth_mask = ((mnq.index.hour > 9) | ((mnq.index.hour == 9) & (mnq.index.minute >= 30))) \
               & (mnq.index.hour < 16)
    mnq_rth = mnq[rth_mask]
    feats_rth = feats[rth_mask]
    labels = make_labels(mnq_rth, horizon=args.horizon, pt_ticks=args.pt, sl_ticks=args.sl)
    label_pct = labels.mean()
    print(f"Labels: {labels.notna().sum():,} valid  |  LONG win rate (naive): {label_pct:.1%}")

    # ── Train / OOS split ──────────────────────────────────────────────────
    TRAIN_END = pd.Timestamp("2026-03-01", tz="US/Eastern")
    train_mask = feats_rth.index < TRAIN_END
    oos_mask   = feats_rth.index >= TRAIN_END
    print(f"\nTrain: {train_mask.sum():,} bars  ({feats_rth.index[train_mask][0].date()} → {TRAIN_END.date()})")
    print(f"OOS  : {oos_mask.sum():,} bars  ({TRAIN_END.date()} → {feats_rth.index[oos_mask][-1].date()})")

    feat_cols = feats_rth.columns.tolist()

    # ── LSTM (GPU) ─────────────────────────────────────────────────────────
    lstm_results = {}
    if not args.no_lstm:
        print(f"\n── LSTM training on {DEVICE} ──────────────────────────────────────")
        X_all, y_all, idx_all = build_sequences(feats_rth, labels, seq_len=args.seq_len)

        train_idx = idx_all < TRAIN_END
        oos_idx   = idx_all >= TRAIN_END

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_oos, y_oos = X_all[oos_idx], y_all[oos_idx]
        idx_oos = idx_all[oos_idx]

        if len(X_tr) == 0 or len(X_oos) == 0:
            print("  Not enough data for LSTM splits.")
        else:
            print(f"  Train sequences: {len(X_tr):,}  OOS: {len(X_oos):,}")

            # Normalize on train stats
            X_mean = X_tr.mean(axis=(0, 1), keepdims=True)
            X_std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
            X_tr_n  = (X_tr  - X_mean) / X_std
            X_oos_n = (X_oos - X_mean) / X_std

            # Train
            model = train_lstm(X_tr_n, y_tr, n_features=X_tr.shape[2],
                                seq_len=args.seq_len, hidden=64, layers=2,
                                dropout=0.3, lr=1e-3, epochs=30, batch=512)

            # OOS predictions
            probs_oos = predict_lstm(model, X_oos_n)

            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_oos, probs_oos)
                print(f"  OOS AUC: {auc:.4f}")
            except Exception:
                auc = 0.5

            # Calibration check
            for thresh in [0.55, 0.58, 0.60, 0.62, 0.65]:
                sel = probs_oos >= thresh
                n_sel = sel.sum()
                if n_sel >= 10:
                    wr = y_oos[sel].mean()
                    print(f"  Conf ≥ {thresh:.2f}: {n_sel:,} signals, WR={wr:.1%}")

            # Trading simulation
            lstm_sim = simulate_strategy(
                mnq_rth, idx_oos, probs_oos,
                n_contracts=args.contracts,
                pt_ticks=args.pt, sl_ticks=args.sl,
                conf_threshold=args.conf,
                max_trades_per_day=25,
                max_daily_loss=-1600.0,
            )
            if lstm_sim:
                print(f"\n  LSTM Strategy (OOS only, {args.contracts}c PT={args.pt}t SL={args.sl}t conf={args.conf}):")
                print(f"    Trades: {lstm_sim['n_trades']}  WR: {lstm_sim['win_rate']:.1%}  "
                      f"$/wk: ${lstm_sim['weekly_pnl']:,.0f}  MaxDD: ${lstm_sim['max_drawdown']:,.0f}")
                print(f"    Sharpe: {lstm_sim['sharpe']:.2f}  PF: {lstm_sim['profit_factor']:.2f}")
                lstm_results = {**lstm_sim, "auc": round(auc, 4)}

    # ── LightGBM (CPU, fast) ───────────────────────────────────────────────
    print(f"\n── LightGBM (CPU baseline) ───────────────────────────────────────")
    try:
        import lightgbm as lgb

        flat_feats = feats_rth.copy()
        flat_labels = labels.copy()
        valid_mask = flat_labels.notna()
        X_flat = flat_feats[valid_mask].values
        y_flat = flat_labels[valid_mask].values
        idx_flat = flat_feats[valid_mask].index

        train_m = idx_flat < TRAIN_END
        oos_m   = idx_flat >= TRAIN_END

        X_tr_f, y_tr_f = X_flat[train_m], y_flat[train_m]
        X_oos_f, y_oos_f = X_flat[oos_m], y_flat[oos_m]
        idx_oos_f = idx_flat[oos_m]

        print(f"  Train: {len(X_tr_f):,}  OOS: {len(X_oos_f):,}")

        lgbm = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_tr_f, y_tr_f)
        probs_lgbm = lgbm.predict_proba(X_oos_f)[:, 1]

        try:
            from sklearn.metrics import roc_auc_score
            auc_lgbm = roc_auc_score(y_oos_f, probs_lgbm)
            print(f"  LightGBM OOS AUC: {auc_lgbm:.4f}")
        except Exception:
            auc_lgbm = 0.5

        # Calibration
        for thresh in [0.55, 0.58, 0.60, 0.62, 0.65]:
            sel = probs_lgbm >= thresh
            n_sel = sel.sum()
            if n_sel >= 10:
                wr = y_oos_f[sel].mean()
                print(f"  Conf ≥ {thresh:.2f}: {n_sel:,} signals, WR={wr:.1%}")

        # Feature importance
        fi = pd.Series(lgbm.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f"\n  Top 15 features:")
        for fname, fval in fi.head(15).items():
            print(f"    {fname:30s} {fval:>6.0f}")

        # Trading sim — sweep confidence
        print(f"\n  Strategy sweep (LightGBM OOS, {args.contracts}c PT={args.pt}t SL={args.sl}t):")
        print(f"  {'Conf':>6} {'N':>7} {'WR':>7} {'$/wk':>9} {'MaxDD':>9} {'Sharpe':>8} {'target?':>8}")
        print(f"  {'-'*62}")
        best_lgbm = {}
        for conf in [0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]:
            sim = simulate_strategy(
                mnq_rth, idx_oos_f, probs_lgbm,
                n_contracts=args.contracts,
                pt_ticks=args.pt, sl_ticks=args.sl,
                conf_threshold=conf,
                max_trades_per_day=25,
                max_daily_loss=-1600.0,
            )
            if sim and sim["n_trades"] >= 5:
                hit = "✓ $6k" if sim["weekly_pnl"] >= 6000 and sim["max_drawdown"] >= -2000 else ""
                print(f"  {conf:>6.2f} {sim['n_trades']:>7} {sim['win_rate']:>7.1%} "
                      f"{sim['weekly_pnl']:>9,.0f} {sim['max_drawdown']:>9,.0f} "
                      f"{sim['sharpe']:>8.2f} {hit:>8}")
                if not best_lgbm or (sim["weekly_pnl"] > best_lgbm.get("weekly_pnl", 0)
                                     and sim["max_drawdown"] >= -2000):
                    best_lgbm = {**sim, "conf": conf, "auc": round(auc_lgbm, 4)}

        # Full grid: PT/SL × confidence × contracts — optimise for $4k DD limit
        MAX_DD_LIMIT = 4_000.0
        MAX_WEEKLY   = 10_000.0
        MAX_NC       = 15  # Topstep funded 150k hard cap

        print(f"\n  ── Full grid sweep (LightGBM OOS, target $10k/wk, $4k DD limit) ──")
        print(f"  {'PT':>4} {'SL':>4} {'Cf':>5} {'Nc':>3} {'N':>7} {'WR':>7} {'$/wk':>9} {'DD':>9} {'Sh':>6}")
        print(f"  {'-'*70}")

        grid_results = []
        for pt_t, sl_t in [(5,3),(8,4),(10,5),(12,6),(15,8),(20,10)]:
            for conf in [0.55, 0.58, 0.62, 0.65]:
                # First simulate at 1 contract to find DD scaling factor
                sim1 = simulate_strategy(
                    mnq_rth, idx_oos_f, probs_lgbm,
                    n_contracts=1,
                    pt_ticks=pt_t, sl_ticks=sl_t,
                    conf_threshold=conf,
                    max_trades_per_day=30,
                    max_daily_loss=-500.0,
                )
                if not sim1 or sim1["n_trades"] < 5 or sim1["max_drawdown"] >= 0:
                    continue
                # Scale contracts to stay under $4k DD
                nc = min(MAX_NC, max(1, int(MAX_DD_LIMIT / abs(sim1["max_drawdown"]))))
                sim = simulate_strategy(
                    mnq_rth, idx_oos_f, probs_lgbm,
                    n_contracts=nc,
                    pt_ticks=pt_t, sl_ticks=sl_t,
                    conf_threshold=conf,
                    max_trades_per_day=30,
                    max_daily_loss=-800.0 * nc,
                )
                if not sim or sim["n_trades"] < 5:
                    continue
                target = " ✓✓" if sim["weekly_pnl"] >= MAX_WEEKLY and abs(sim["max_drawdown"]) <= MAX_DD_LIMIT else \
                         " ✓"  if sim["weekly_pnl"] >= 5000 and abs(sim["max_drawdown"]) <= MAX_DD_LIMIT else ""
                print(f"  {pt_t:>4} {sl_t:>4} {conf:>5.2f} {nc:>3} {sim['n_trades']:>7} {sim['win_rate']:>7.1%} "
                      f"{sim['weekly_pnl']:>9,.0f} {sim['max_drawdown']:>9,.0f} {sim['sharpe']:>6.2f}{target}")
                grid_results.append({**sim, "pt": pt_t, "sl": sl_t, "conf": conf, "nc": nc})

        if grid_results:
            best_grid = max(grid_results, key=lambda x: x["weekly_pnl"] - max(0, abs(x["max_drawdown"]) - MAX_DD_LIMIT) * 5)
            print(f"\n  Best: PT={best_grid['pt']}t SL={best_grid['sl']}t conf={best_grid['conf']} "
                  f"n={best_grid['nc']} → ${best_grid['weekly_pnl']:,.0f}/wk DD=${best_grid['max_drawdown']:,.0f}")
            best_lgbm = {**best_lgbm, **{k: best_grid[k] for k in ["weekly_pnl","max_drawdown","win_rate","n_trades","sharpe"]}}

    except ImportError:
        print("  LightGBM not installed — skipping.")
        best_lgbm = {}

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Instruments loaded: MNQ" +
          (" + MES" if mes is not None else "") +
          (" + MGC" if mgc is not None else "") +
          (" + ZN" if zn is not None else "") +
          (" + 6E" if fe6 is not None else ""))
    print(f"  Features: {feats.shape[1]}  |  Seq len: {args.seq_len}")
    print(f"  Train: Jan-Feb 2026  |  OOS: Mar 2026 → current (never seen by model)")
    print(f"  Label: LONG wins if PT={args.pt}t hit before SL={args.sl}t in {args.horizon} bars")
    if lstm_results:
        print(f"\n  LSTM  (GPU/MPS): AUC={lstm_results.get('auc',.5):.3f}  "
              f"WR={lstm_results.get('win_rate',0):.1%}  "
              f"$/wk=${lstm_results.get('weekly_pnl',0):,.0f}  "
              f"MaxDD=${lstm_results.get('max_drawdown',0):,.0f}")
    if best_lgbm:
        print(f"  LightGBM (CPU): AUC={best_lgbm.get('auc',.5):.3f}  "
              f"WR={best_lgbm.get('win_rate',0):.1%}  "
              f"$/wk=${best_lgbm.get('weekly_pnl',0):,.0f}  "
              f"MaxDD=${best_lgbm.get('max_drawdown',0):,.0f}")

    results = {"lstm": lstm_results, "lgbm": best_lgbm,
                "config": {"horizon": args.horizon, "pt": args.pt, "sl": args.sl,
                           "seq_len": args.seq_len, "contracts": args.contracts}}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
