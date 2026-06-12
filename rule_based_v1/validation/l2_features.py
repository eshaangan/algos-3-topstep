"""Order-book (L2) feature extraction → bars ready for the validation harness.

Reads the raw parquet the recorder writes (data/l2_raw/book_*.parquet,
trade_*.parquet — schema defined in data_collection/record_l2.py), computes
book-imbalance / microprice / depth features per snapshot, and aggregates to bars
at a chosen frequency. Output columns match what harness.simulate expects
(open/high/low/close/atr) plus the L2 feature columns the research layer signals on.

Why these features: trade flow alone is exhausted (no predictive edge at 1-5 min,
confirmed). The order book carries *different* information — resting liquidity,
queue sizes, microprice tilt — that may predict minute-scale moves large enough to
clear the ~$2.24 round-trip cost. That is the hypothesis this pipeline tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Aggressor codes from Rithmic (recorder stores the raw int)
_AGG_BUY = 1
_AGG_SELL = 2


def load_l2_raw(raw_dir: str | Path, kind: str, date: str | None = None) -> pd.DataFrame:
    """Load and time-order the recorder's chunked parquet for one kind ('book'|'trade').

    `date` filters by the YYYYMMDD stamp in the filename (e.g. '20260612').
    Returns empty DataFrame if nothing matches.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(f"{kind}_*.parquet"))
    if date:
        files = [f for f in files if f"_{date}_" in f.name]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("recv_ns").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["recv_ns"], unit="ns", utc=True)
    return df


def snapshot_features(book: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """Per-snapshot order-book features. Expects bid_px/sz/ord_0..9 + ask_* columns."""
    if book.empty:
        return pd.DataFrame()
    bpx0, apx0 = book["bid_px_0"], book["ask_px_0"]
    bsz0, asz0 = book["bid_sz_0"].astype(float), book["ask_sz_0"].astype(float)
    den1 = (bsz0 + asz0).replace(0, np.nan)

    out = pd.DataFrame(index=book.index)
    out["ts"] = book["ts"]
    out["mid"] = (bpx0 + apx0) / 2
    out["spread"] = apx0 - bpx0
    # microprice: mid pulled toward the side with LESS size (weight by opposite size)
    out["microprice"] = (bpx0 * asz0 + apx0 * bsz0) / den1
    out["micro_tilt"] = out["microprice"] - out["mid"]            # >0 ⇒ upward pressure
    out["l1_imb"] = (bsz0 - asz0) / den1                          # top-of-book imbalance

    # multi-level depth imbalance
    bcols = [f"bid_sz_{i}" for i in range(n_levels)]
    acols = [f"ask_sz_{i}" for i in range(n_levels)]
    bid_depth = book[bcols].astype(float).sum(axis=1)
    ask_depth = book[acols].astype(float).sum(axis=1)
    dend = (bid_depth + ask_depth).replace(0, np.nan)
    out["depth_imb"] = (bid_depth - ask_depth) / dend
    out["depth_total"] = bid_depth + ask_depth

    # order-count imbalance (queue thinning proxy)
    bocols = [f"bid_ord_{i}" for i in range(n_levels)]
    aocols = [f"ask_ord_{i}" for i in range(n_levels)]
    if all(c in book.columns for c in bocols + aocols):
        bo = book[bocols].astype(float).sum(axis=1)
        ao = book[aocols].astype(float).sum(axis=1)
        out["ord_imb"] = (bo - ao) / (bo + ao).replace(0, np.nan)
    return out


def trade_flow(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per-bar order-flow imbalance from the trade stream."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["buy"] = np.where(t["aggressor"].values == _AGG_BUY, t["size"], 0)
    t["sell"] = np.where(t["aggressor"].values == _AGG_SELL, t["size"], 0)
    g = t.set_index("ts").resample(freq)
    out = pd.DataFrame()
    out["t_vol"] = g["size"].sum()
    out["t_ofi"] = g["buy"].sum() - g["sell"].sum()
    out["t_n"] = g["size"].count()
    out["t_ofi_imb"] = out["t_ofi"] / out["t_vol"].replace(0, np.nan)
    return out


def build_bars(book: pd.DataFrame, trades: pd.DataFrame | None = None,
               freq: str = "1min", n_levels: int = 5, atr_period: int = 14) -> pd.DataFrame:
    """Aggregate snapshots (+ trades) to bars with OHLC(mid) + L2 features + ATR.

    Book-state features (imbalances, tilt, spread) are averaged over the bar; OHLC
    is on the mid price. The result is directly consumable by harness.simulate
    (open/high/low/close/atr) and by research_l2 signal generators.
    """
    snap = snapshot_features(book, n_levels=n_levels)
    if snap.empty:
        return pd.DataFrame()
    snap = snap.set_index("ts").sort_index()
    g = snap.resample(freq)

    bars = pd.DataFrame()
    bars["open"] = g["mid"].first()
    bars["high"] = g["mid"].max()
    bars["low"] = g["mid"].min()
    bars["close"] = g["mid"].last()
    for col in ["l1_imb", "depth_imb", "micro_tilt", "spread", "depth_total"]:
        if col in snap.columns:
            bars[col] = g[col].mean()
    if "ord_imb" in snap.columns:
        bars["ord_imb"] = g["ord_imb"].mean()
    bars["n_snap"] = g["mid"].count()

    if trades is not None and not trades.empty:
        tf = trade_flow(trades, freq)
        bars = bars.join(tf, how="left")

    bars = bars.dropna(subset=["close"])
    # ATR on mid OHLC
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    bars["atr"] = tr.ewm(span=atr_period, min_periods=atr_period).mean()
    return bars


# ──────────────────────────────────────────────── synthetic data (dev/tests) ──

def synthesize_l2(n_snap: int = 20000, seed: int = 0, signal_strength: float = 0.0,
                  start: str = "2026-06-12 14:00", n_levels: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate book + trade frames matching the recorder schema, for testing the
    pipeline before real data exists. If signal_strength>0, depth imbalance leads
    the next mid move (a planted edge) so tests can confirm the pipeline detects it.
    """
    rng = np.random.default_rng(seed)
    ts0 = pd.Timestamp(start, tz="UTC")
    # ~0.2-2s between snapshots so a modest n_snap spans many minutes (enough bars).
    recv = (ts0.value + np.cumsum(rng.integers(200_000_000, 2_000_000_000, n_snap)))
    # Persistent (AR1) depth imbalance so it survives minute-aggregation, and it
    # LEADS the price one step (the planted edge when signal_strength>0).
    imb = np.zeros(n_snap)
    noise = rng.normal(0, 0.5, n_snap)
    for k in range(1, n_snap):
        imb[k] = 0.85 * imb[k - 1] + 0.5 * noise[k]
    imb = np.clip(imb, -1, 1)
    # Planted edge: imbalance LEADS price by ~one bar (≈50 snapshots), so this bar's
    # mean depth_imb predicts NEXT bar's move (a bar-level predictive edge).
    lag = 50
    shifted = np.concatenate([np.zeros(lag), imb[:-lag]])
    steps = rng.normal(0, 0.25, n_snap) + signal_strength * shifted
    mid = 20000 + np.cumsum(steps)
    tick = 0.25
    book = {"recv_ns": recv, "ssboe": recv // 1_000_000_000, "usecs": (recv // 1000) % 1_000_000,
            "update_type": np.full(n_snap, 3)}
    base_b = (np.abs(rng.normal(20, 8, n_snap)) + 1)
    for i in range(n_levels):
        # size split encodes the imbalance at level 0, decaying with depth
        w = max(0.0, 1 - i * 0.15)
        bs = base_b * (1 + w * imb)
        as_ = base_b * (1 - w * imb)
        book[f"bid_px_{i}"] = np.round((mid - tick * (i + 0.5)) / tick) * tick
        book[f"ask_px_{i}"] = np.round((mid + tick * (i + 0.5)) / tick) * tick
        book[f"bid_sz_{i}"] = np.maximum(bs, 1).astype(int)
        book[f"ask_sz_{i}"] = np.maximum(as_, 1).astype(int)
        book[f"bid_ord_{i}"] = np.maximum(bs / 5, 1).astype(int)
        book[f"ask_ord_{i}"] = np.maximum(as_ / 5, 1).astype(int)
    book_df = pd.DataFrame(book)

    n_tr = n_snap // 2
    tidx = np.sort(rng.integers(0, n_snap, n_tr))
    trades = pd.DataFrame({
        "recv_ns": recv[tidx], "ssboe": recv[tidx] // 1_000_000_000,
        "usecs": (recv[tidx] // 1000) % 1_000_000,
        "price": mid[tidx] + rng.choice([-tick, tick], n_tr),
        "size": rng.integers(1, 8, n_tr),
        "aggressor": rng.choice([_AGG_BUY, _AGG_SELL], n_tr),
    })
    return book_df, trades
