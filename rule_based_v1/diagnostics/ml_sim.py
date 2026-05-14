"""
ML-gated trading simulation — one position at a time, proper P&L.

Strategy: Long-only MNQ when:
  1. Price is >= entry_dist ATR below session VWAP (mean reversion setup)
  2. Within trading window (10:30–13:30)
  3. ML confidence >= threshold (trained on same features, no leakage)
  4. No excessive move from open (regime gate)

Exit: PT=15t, SL=8t, or time_stop=30 bars (2.5 hours)

Compares:
  A) Naive (no ML gate)
  B) ML-gated at various confidence thresholds

10 MNQ contracts, commission $0.62/side
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
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
N_CONTRACTS  = 10
MAX_DAILY    = -1600.0
MAX_PER_DAY  = 3
TIME_STOP    = 30       # bars
PT_TICKS     = 15
SL_TICKS     = 8
ENTRY_DIST_ATR = 1.0   # min ATR deviation from VWAP to enter
MAX_MOVE_ATR   = 2.5   # max distance from open (regime gate)
TSTART_M     = 630     # 10:30
TEND_M       = 810     # 13:30

GROSS_WIN  = PT_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS
GROSS_LOSS = SL_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS
COMM_RT    = 2 * COMMISSION * N_CONTRACTS
NET_WIN    = GROSS_WIN  - COMM_RT
NET_LOSS   = GROSS_LOSS + COMM_RT


def load_bars():
    for p in DATA_CANDIDATES:
        if p.exists():
            bars = pd.read_hdf(str(p), key="bars_5min")
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("US/Eastern")
            else:
                bars.index = bars.index.tz_convert("US/Eastern")
            return bars
    raise FileNotFoundError("No MNQ data found")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    for w in [1, 3, 5, 10, 15, 30]:
        df[f"r{w}"] = df["close"].pct_change(w)

    tp = (df["high"] + df["low"] + df["close"]) / 3
    date_idx = df.index.map(lambda t: t.date())
    vwap = (tp * df["volume"]).groupby(date_idx).cumsum() / \
           df["volume"].groupby(date_idx).cumsum().replace(0, np.nan)
    df["vwap"] = vwap
    df["vwap_dev"] = (df["close"] - vwap) / df["atr"].replace(0, np.nan)

    day_opens = df.groupby(df.index.map(lambda t: t.date()))["open"].transform("first")
    df["day_open"] = day_opens
    df["move_from_open"] = (df["close"] - day_opens) / df["atr"].replace(0, np.nan)

    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"].replace(0, np.nan)

    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_sig"] = (macd - macd.ewm(span=9, adjust=False).mean()) / df["atr"].replace(0, np.nan)

    df["hour"]   = df.index.hour
    df["minute"] = df.index.minute

    return df


FEAT_COLS = ["atr_pct","r1","r3","r5","r10","r15","r30",
             "vwap_dev","rvol","range_atr","rsi14","macd_sig","hour","minute"]


def label_bars_forward(df: pd.DataFrame) -> np.ndarray:
    """Label: 1 if PT hit before SL within TIME_STOP bars (LONG perspective)."""
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(close)
    labels = np.full(n, np.nan)
    pt_pts = PT_TICKS * TICK_SIZE
    sl_pts = SL_TICKS * TICK_SIZE
    for i in range(n - TIME_STOP):
        entry = close[i] + TICK_SIZE  # assume buy 1 tick above close
        for j in range(i+1, min(i+1+TIME_STOP, n)):
            if high[j] >= entry + pt_pts:
                labels[i] = 1; break
            if low[j]  <= entry - sl_pts:
                labels[i] = 0; break
    return labels


def simulate(df_all: pd.DataFrame, ml_model=None, ml_thresh=None, label="naive") -> dict:
    """Event-driven sim: one position at a time, 10 MNQ contracts."""
    close_a  = df_all["close"].values
    high_a   = df_all["high"].values
    low_a    = df_all["low"].values
    atr_a    = df_all["atr"].values
    vwap_a   = df_all["vwap"].values
    mopen_a  = df_all["move_from_open"].values
    minutes  = (df_all.index.hour * 60 + df_all.index.minute).values
    dates    = np.array([t.toordinal() for t in df_all.index])
    n        = len(df_all)

    proba_a = None
    if ml_model is not None:
        X_all = df_all[FEAT_COLS].values
        proba_a = ml_model.predict_proba(X_all)[:, 1]

    trades = []
    equity = 50_000.0
    equity_curve = [equity]
    cur_date  = dates[30]
    daily_pnl = 0.0
    day_count = 0

    pos_active = False
    pos_entry  = 0.0
    pos_sl     = 0.0
    pos_pt     = 0.0
    pos_stop_bar = 0

    for i in range(30, n):
        d = dates[i]
        if d != cur_date:
            daily_pnl = 0.0
            day_count = 0
            cur_date  = d

        bt_m     = minutes[i]
        is_close = (bt_m >= 955) or (i+1 >= n) or (dates[i+1] != d)

        # ── Exit logic ──
        if pos_active:
            h = high_a[i]; l = low_a[i]; c = close_a[i]
            exited = False; ep = 0.0; reason = ""
            if is_close or i >= pos_stop_bar:
                ep = c - TICK_SIZE
                reason = "time_stop"; exited = True
            else:
                if l <= pos_sl:
                    ep = pos_sl - TICK_SIZE; reason = "stop_loss"; exited = True
                elif h >= pos_pt:
                    ep = pos_pt - TICK_SIZE; reason = "profit_target"; exited = True
            if exited:
                pnl = (ep - pos_entry) * N_CONTRACTS * POINT_VALUE - COMM_RT
                trades.append({"pnl": pnl, "reason": reason, "equity": equity + pnl})
                daily_pnl += pnl
                equity    += pnl
                equity_curve.append(equity)
                pos_active = False

        if is_close:
            continue
        if pos_active or daily_pnl <= MAX_DAILY or day_count >= MAX_PER_DAY:
            continue
        if not (TSTART_M <= bt_m <= TEND_M):
            continue

        atr_v = atr_a[i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue
        vwap_v = vwap_a[i]
        if np.isnan(vwap_v):
            continue

        c     = close_a[i]
        prev_c = close_a[i-1]
        dev   = (c - vwap_v) / atr_v

        # Regime gate
        mv = mopen_a[i]
        if np.isnan(mv):
            continue
        if dev < 0 and mv < -MAX_MOVE_ATR:
            continue

        # Entry: price below VWAP by ENTRY_DIST_ATR, with bullish last bar
        if dev <= -ENTRY_DIST_ATR and c > prev_c:
            # ML gate
            if ml_model is not None and proba_a is not None:
                p = proba_a[i]
                if np.isnan(p) or p < ml_thresh:
                    continue

            entry = c + TICK_SIZE
            pos_active   = True
            pos_entry    = entry
            pos_sl       = entry - SL_TICKS * TICK_SIZE
            pos_pt       = entry + PT_TICKS * TICK_SIZE
            pos_stop_bar = i + TIME_STOP
            day_count   += 1

    if not trades:
        return {"label": label, "n": 0}

    arr     = np.array([t["pnl"] for t in trades])
    eq_arr  = np.array(equity_curve)
    wins    = (arr > 0).sum()
    total   = arr.sum()
    gp      = arr[arr > 0].sum() if wins > 0 else 0
    gl      = abs(arr[arr <= 0].sum())
    max_dd  = float((eq_arr - np.maximum.accumulate(eq_arr)).min())
    n_days  = len(set(dates[30:]))
    n_weeks = max(1, n_days / 5)
    weekly  = total / n_weeks

    return {
        "label":      label,
        "n":          len(arr),
        "wr":         wins / len(arr),
        "total":      total,
        "weekly":     weekly,
        "max_dd":     max_dd,
        "pf":         gp/gl if gl > 0 else 99,
        "sharpe":     float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0,
    }


def print_result(r: dict):
    if r["n"] == 0:
        print(f"  {r['label']:<25} NO TRADES")
        return
    print(f"  {r['label']:<25} n={r['n']:>4}  WR={r['wr']:.1%}  "
          f"$/wk=${r['weekly']:>8,.0f}  DD=${r['max_dd']:>8,.0f}  "
          f"Sharpe={r['sharpe']:>6.2f}  PF={r['pf']:.2f}")


def main():
    print("Loading MNQ bars...")
    mnq = load_bars()
    print(f"  {len(mnq):,} bars [{mnq.index[0].date()} → {mnq.index[-1].date()}]")

    print("Building features...")
    df = add_indicators(mnq)
    df = df.dropna(subset=FEAT_COLS + ["atr","vwap","move_from_open"])

    # OOS split: train on Jan-Mar, evaluate on Apr+
    train_cut = pd.Timestamp("2026-04-01", tz="US/Eastern")
    df_train  = df[df.index < train_cut].copy()
    df_oos    = df[df.index >= train_cut].copy()

    print(f"  Train: {len(df_train)} bars | OOS: {len(df_oos)} bars")

    # ── Build labels for training ──
    print("Labeling training data (may take ~30s)...")
    labels_tr = label_bars_forward(df_train)
    df_train["label"] = labels_tr

    labeled = df_train.dropna(subset=["label"])
    label_rate = labeled["label"].mean()
    print(f"  Training labels: {len(labeled)} labeled | {label_rate:.1%} positive")

    X_tr = labeled[FEAT_COLS].values
    y_tr = labeled["label"].values.astype(int)

    # ── Train LightGBM ──
    print("Training LightGBM...")
    base = lgb.LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=25, reg_lambda=1.0,
        n_jobs=-1, verbose=-1,
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_tr, y_tr)

    # AUC on OOS labels
    print("Computing OOS AUC...")
    labels_oos = label_bars_forward(df_oos)
    df_oos = df_oos.copy()
    df_oos["label"] = labels_oos
    labeled_oos = df_oos.dropna(subset=["label"])
    X_oo = labeled_oos[FEAT_COLS].values
    y_oo = labeled_oos["label"].values.astype(int)
    p_oo = model.predict_proba(X_oo)[:, 1]
    auc  = roc_auc_score(y_oo, p_oo)
    print(f"\n  OOS AUC: {auc:.4f}  (naive WR: {y_oo.mean():.1%})")

    gross_win  = PT_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS
    gross_loss = SL_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS
    be_wr      = (gross_loss + COMM_RT) / (gross_win + gross_loss)
    print(f"  PT=${gross_win:.0f}  SL=${gross_loss:.0f}  Comm=${COMM_RT:.2f}  "
          f"Net win=${NET_WIN:.2f}  Net loss=${NET_LOSS:.2f}  BE WR={be_wr:.1%}")

    # ── Simulate on full OOS ──
    print(f"\n{'='*75}")
    print("OOS SIMULATION RESULTS (Apr 2026 →)")
    print(f"{'='*75}")

    results = []
    r = simulate(df_oos, ml_model=None, label="Naive (no ML gate)")
    results.append(r); print_result(r)

    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        r = simulate(df_oos, ml_model=model, ml_thresh=thresh,
                     label=f"ML conf≥{thresh:.2f}")
        results.append(r); print_result(r)

    print(f"\n{'='*75}")
    best = max([r for r in results if r["n"] >= 10],
               key=lambda x: x["sharpe"] * x["wr"], default=None)
    if best:
        print(f"Best config: {best['label']}")
        print(f"  {best['n']} trades | WR={best['wr']:.1%} | "
              f"${best['weekly']:,.0f}/wk | DD=${best['max_dd']:,.0f} | "
              f"Sharpe={best['sharpe']:.2f}")

    # ── Scale projections ──
    print(f"\nScaled projections (10 MNQ):")
    print(f"  {'Config':<30} {'WR':>7} {'$/wk':>10} {'MaxDD':>10}")
    print(f"  {'-'*65}")
    for r in results:
        if r["n"] < 5:
            continue
        print(f"  {r['label']:<30} {r['wr']:>7.1%} {r['weekly']:>10,.0f} {r['max_dd']:>10,.0f}")


if __name__ == "__main__":
    main()
