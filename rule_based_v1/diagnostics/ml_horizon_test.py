"""
Multi-horizon ML label test: find optimal PT/SL/horizon for LightGBM trading.
Tests three configs and reports AUC, calibrated WR, and net EV at 10 MNQ.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]
MES_CANDIDATES = [
    ROOT / "data" / "processed" / "mes_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mes_2026ytd_5min.h5",
]

POINT_VALUE = 2.0   # MNQ
TICK_SIZE   = 0.25
COMMISSION  = 0.62  # per side
N_CONTRACTS = 10

def load_bars(candidates, key="bars_5min"):
    for p in candidates:
        if p.exists():
            bars = pd.read_hdf(str(p), key=key)
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("US/Eastern")
            else:
                bars.index = bars.index.tz_convert("US/Eastern")
            return bars
    raise FileNotFoundError(f"No data found in {candidates}")

def build_features(mnq: pd.DataFrame, mes: pd.DataFrame = None) -> pd.DataFrame:
    df = mnq.copy()

    # ATR
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # Returns
    for w in [1, 3, 5, 10, 15, 30]:
        df[f"r{w}"] = df["close"].pct_change(w)

    # VWAP deviation
    tp = (df["high"] + df["low"] + df["close"]) / 3
    date_idx = df.index.map(lambda t: t.date())
    vwap = (tp * df["volume"]).groupby(date_idx).cumsum() / \
           df["volume"].groupby(date_idx).cumsum().replace(0, np.nan)
    df["vwap_dev"] = (df["close"] - vwap) / df["atr"].replace(0, np.nan)

    # Relative volume
    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()

    # High-low range / ATR
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"].replace(0, np.nan)

    # Cross-asset MNQ/MES divergence
    if mes is not None:
        mes_r = mes["close"].pct_change(1).reindex(df.index, method="ffill")
        mnq_r = df["close"].pct_change(1)
        df["div1"] = mnq_r - mes_r
        mes_r5 = mes["close"].pct_change(5).reindex(df.index, method="ffill")
        mnq_r5 = df["close"].pct_change(5)
        df["div5"] = mnq_r5 - mes_r5

    # Time features
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute

    # RSI-14
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # MACD signal
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_sig"] = (macd - macd.ewm(span=9, adjust=False).mean()) / df["atr"].replace(0, np.nan)

    return df

def label_bars(df: pd.DataFrame, pt_ticks: int, sl_ticks: int, horizon: int) -> pd.Series:
    """Label each bar: 1 if PT hit before SL within horizon, 0 otherwise."""
    close = df["close"].values
    n = len(close)
    labels = np.full(n, np.nan)

    pt_pts = pt_ticks * TICK_SIZE
    sl_pts = sl_ticks * TICK_SIZE

    high = df["high"].values
    low  = df["low"].values

    for i in range(n - horizon):
        entry = close[i]
        hit_pt = False
        hit_sl = False
        for j in range(i+1, min(i+1+horizon, n)):
            if high[j] >= entry + pt_pts:
                hit_pt = True; break
            if low[j] <= entry - sl_pts:
                hit_sl = True; break
        if hit_pt:
            labels[i] = 1
        elif hit_sl:
            labels[i] = 0
        # else: no label (NaN → excluded)

    return pd.Series(labels, index=df.index)

def run_config(name, pt_ticks, sl_ticks, horizon, mnq, mes):
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}  PT={pt_ticks}t  SL={sl_ticks}t  H={horizon} bars")
    print(f"{'='*60}")

    gross_win  = pt_ticks * TICK_SIZE * POINT_VALUE * N_CONTRACTS
    gross_loss = sl_ticks * TICK_SIZE * POINT_VALUE * N_CONTRACTS
    comm_rt    = 2 * COMMISSION * N_CONTRACTS

    print(f"  Gross win:  ${gross_win:.2f}  |  Gross loss: ${gross_loss:.2f}")
    print(f"  Commission (RT): ${comm_rt:.2f}  |  Net win: ${gross_win - comm_rt:.2f}  |  Net loss: ${gross_loss + comm_rt:.2f}")
    be_wr = (gross_loss + comm_rt) / (gross_win + gross_loss)
    print(f"  Breakeven WR: {be_wr:.1%}")

    # Build features
    df = build_features(mnq, mes)

    # Label
    labels = label_bars(df, pt_ticks, sl_ticks, horizon)
    df["label"] = labels

    # Drop unlabeled rows and RTH only (9:35-15:45)
    df = df.dropna(subset=["label"])
    df = df[(df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 35))]
    df = df[(df.index.hour < 15) | ((df.index.hour == 15) & (df.index.minute <= 45))]

    print(f"  Labels: {len(df)} total | {df['label'].sum():.0f} positive ({df['label'].mean():.1%} rate)")

    if df["label"].mean() < 0.1 or df["label"].mean() > 0.9:
        print("  WARNING: Extreme label imbalance — skipping")
        return

    # Feature columns
    feat_cols = [c for c in df.columns if c not in
                 ["open","high","low","close","volume","label","atr"] and
                 df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # Train/OOS split: train on Jan-Mar, OOS on Apr+
    train_cut = pd.Timestamp("2026-04-01", tz="US/Eastern")
    train = df[df.index < train_cut].dropna(subset=feat_cols)
    oos   = df[df.index >= train_cut].dropna(subset=feat_cols)

    if len(train) < 200 or len(oos) < 50:
        print(f"  Insufficient data: train={len(train)}, oos={len(oos)}")
        return

    X_tr = train[feat_cols].values
    y_tr = train["label"].values.astype(int)
    X_oo = oos[feat_cols].values
    y_oo = oos["label"].values.astype(int)

    print(f"  Train: {len(X_tr)} bars | OOS: {len(X_oo)} bars")

    # LightGBM with calibration
    base = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_oo)[:, 1]
    auc = roc_auc_score(y_oo, proba)
    naive_wr = y_oo.mean()

    print(f"\n  OOS AUC: {auc:.4f}  |  Naive LONG WR: {naive_wr:.1%}")

    # Simulate at confidence thresholds
    net_win  = gross_win  - comm_rt
    net_loss = gross_loss + comm_rt

    print(f"\n  {'Thresh':>7} {'N':>6} {'WR':>7} {'$/trade':>9} {'EV/wk':>10}  note")
    print(f"  {'-'*65}")

    # Approx weeks in OOS period
    oos_days = (oos.index[-1] - oos.index[0]).days
    n_weeks  = max(1, oos_days / 7)
    bars_per_day = 78  # 5-min RTH bars

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = proba >= thresh
        n_t  = mask.sum()
        if n_t < 10:
            continue
        wr   = y_oo[mask].mean()
        ev   = wr * net_win - (1-wr) * net_loss
        # trades per week = (n_t / n_oos_bars) * bars_per_day * 5
        tpw  = (n_t / len(oos)) * bars_per_day * 5
        weekly_ev = ev * tpw
        note = ""
        if wr >= be_wr + 0.05:
            note = "*** VIABLE ***"
        elif wr >= be_wr:
            note = "marginal"
        else:
            note = "below BE"
        print(f"  {thresh:>7.2f} {n_t:>6} {wr:>7.1%} {ev:>9.2f} {weekly_ev:>10.1f}  {note}")

    # Feature importance
    base2 = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_lambda=1.0, n_jobs=-1, verbose=-1,
    )
    base2.fit(X_tr, y_tr)
    imp = pd.Series(base2.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for fname, fval in imp.head(10).items():
        print(f"    {fname:<20} {fval:.0f}")

def main():
    print("Loading MNQ bars...")
    mnq = load_bars(DATA_CANDIDATES)
    print(f"  MNQ: {len(mnq):,} bars [{mnq.index[0].date()} → {mnq.index[-1].date()}]")

    mes = None
    try:
        mes = load_bars(MES_CANDIDATES)
        print(f"  MES: {len(mes):,} bars [{mes.index[0].date()} → {mes.index[-1].date()}]")
    except FileNotFoundError:
        print("  MES not found — using MNQ-only features")

    configs = [
        ("tight-scalp",  5,  3,  5),
        ("medium",       10, 5, 15),
        ("swing",        15, 8, 30),
    ]

    for name, pt, sl, h in configs:
        run_config(name, pt, sl, h, mnq, mes)

    print("\n\nDone.")

if __name__ == "__main__":
    main()
