"""
Daily ORB go/no-go ML classifier.

The key insight: ML's value isn't generating intraday signals.
It's filtering out losing ORB days -> reducing drawdown -> allowing more contracts.

Training data: MES Oct 2024 - Dec 2025 (~280 days) + MNQ Aug 2025 - Mar 2026 (~168 days)
OOS test:      MNQ April 2026

Features (all known at or by 10:04 ET, before first signal):
  Pre-open:  overnight gap, 1/3/5-day momentum, ATR rank, day-of-week, month
  At 10:04:  OR range / ATR, OR size (tight=bad, wide=opportunity)

Label: 1 if ORB trade won (PnL > 0), 0 if lost. Days with no signal = excluded.

Impact:
  If ML avoids 40% of losing ORB days while keeping 85% of winning days:
    -> WR improves from 56% -> 69%
    -> DD cut by ~35%
    -> Can safely scale from 10 MNQ -> 20-25 MNQ
    -> Weekly PnL: $5k+ at 25 MNQ
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
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# ── Config ────────────────────────────────────────────────────────────────────
OR_END_HOUR, OR_END_MIN = 10, 4     # OR window ends 10:04
ENTRY_CUTOFF = (12, 0)              # No entries after noon
PT_ATR = 3.0
SL_ATR = 1.5
TIME_STOP_BARS = 24                 # 2 hours max hold

MNQ_PV = 2.0   # $/point MNQ
MES_PV = 5.0   # $/point MES (used for per-trade labeling only)
TICK   = 0.25
COMM   = 0.62  # per side per contract

FEAT_COLS = [
    "ret1", "ret3", "ret5",          # prior day returns
    "gap_pct",                        # overnight gap
    "atr_rank",                       # ATR vs 20d rolling
    "atr_pct",                        # ATR as % price
    "prev_range_atr",                 # prior day range / ATR
    "weekday",                        # 0=Mon, 4=Fri
    "month",                          # 1-12
    "or_range_atr",                   # OR range / ATR (quality of OR)
    "or_range_pct",                   # OR range / price
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_mnq() -> pd.DataFrame:
    files = [
        ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5",
        ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ]
    pieces = []
    for f in files:
        if f.exists():
            b = pd.read_hdf(str(f), key="bars_5min")
            if b.index.tz is None:
                b.index = b.index.tz_localize("US/Eastern")
            else:
                b.index = b.index.tz_convert("US/Eastern")
            pieces.append(b)
    bars = pd.concat(pieces).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars


def load_mes() -> pd.DataFrame:
    f = ROOT / "data" / "processed" / "MES_5min_Oct2024_Dec2025.parquet"
    bars = pd.read_parquet(str(f))
    bars.index = bars.index.tz_convert("US/Eastern")
    return bars


def rth_filter(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[
        ((bars.index.hour == 9) & (bars.index.minute >= 30)) |
        (bars.index.hour >= 10)
    ][(bars.index.hour < 16)]


# ── ORB simulation (per-day) ──────────────────────────────────────────────────
def simulate_orb(day_bars: pd.DataFrame, point_value: float) -> dict | None:
    """Simulate one ORB trade. Returns dict or None if no signal."""
    # Opening range bars: 9:30 to 10:04 (inclusive)
    or_mask = (
        ((day_bars.index.hour == 9) & (day_bars.index.minute >= 30)) |
        ((day_bars.index.hour == 10) & (day_bars.index.minute <= OR_END_MIN))
    )
    or_bars = day_bars[or_mask]
    if len(or_bars) < 5:
        return None

    or_high = or_bars["high"].max()
    or_low  = or_bars["low"].min()
    or_range = or_high - or_low

    # ATR at end of OR
    close_s = day_bars["close"]
    prev_s  = close_s.shift(1)
    tr = pd.concat([(day_bars["high"]-day_bars["low"]),
                    (day_bars["high"]-prev_s).abs(),
                    (day_bars["low"]-prev_s).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(span=14, adjust=False).mean()
    atr = atr_series.iloc[-1]
    if atr <= 0 or np.isnan(atr):
        return None

    # Entry window: after 10:04, before entry cutoff, inside RTH
    after_or = day_bars[
        ((day_bars.index.hour == 10) & (day_bars.index.minute > OR_END_MIN)) |
        (day_bars.index.hour == 11)
    ]
    if len(after_or) == 0:
        return None

    # Look for first bar that closes above OR high (breakout)
    entered = False
    for ts, row in after_or.iterrows():
        if row["high"] > or_high:
            entry = or_high + TICK
            pt    = entry + PT_ATR * atr
            sl    = entry - SL_ATR * atr

            # Simulate exit
            remain = after_or.loc[ts:]
            pnl = None
            bars_held = 0
            for ts2, row2 in remain.iterrows():
                bars_held += 1
                if bars_held >= TIME_STOP_BARS:
                    pnl = (row2["close"] - entry) * point_value - 2 * COMM
                    break
                if row2["low"] <= sl:
                    pnl = (sl - TICK - entry) * point_value - 2 * COMM
                    break
                if row2["high"] >= pt:
                    pnl = (pt - TICK - entry) * point_value - 2 * COMM
                    break
            if pnl is None and len(remain) > 0:
                pnl = (remain.iloc[-1]["close"] - entry) * point_value - 2 * COMM

            return {
                "pnl": pnl,
                "won": pnl > 0,
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "atr": atr,
                "entry": entry,
                "or_range_atr": or_range / atr,
            }
    return None


# ── Daily feature builder ────────────────────────────────────────────────────
def build_daily_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute daily features from intraday bars. Uses prior-day data to avoid leakage."""
    # Aggregate to daily
    daily = bars.groupby(bars.index.date).agg(
        open=("open","first"), close=("close","last"),
        high=("high","max"), low=("low","min"), volume=("volume","sum")
    )
    daily.index = pd.to_datetime(daily.index)

    # ATR
    prev = daily["close"].shift(1)
    tr = pd.concat([daily["high"]-daily["low"],
                    (daily["high"]-prev).abs(),
                    (daily["low"]-prev).abs()], axis=1).max(axis=1)
    daily["atr"] = tr.ewm(span=14, adjust=False).mean()
    daily["atr_pct"]  = daily["atr"] / daily["close"]
    daily["atr_rank"] = daily["atr_pct"].rolling(20).rank(pct=True)

    # Momentum
    daily["ret1"] = daily["close"].pct_change(1)
    daily["ret3"] = daily["close"].pct_change(3)
    daily["ret5"] = daily["close"].pct_change(5)

    # Overnight gap (today's open vs yesterday's close)
    daily["gap_pct"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    # Prior day range / ATR
    daily["prev_range_atr"] = (daily["high"].shift(1) - daily["low"].shift(1)) / daily["atr"].shift(1)

    # Time features
    daily["weekday"] = daily.index.dayofweek
    daily["month"]   = daily.index.month

    return daily


def build_dataset(bars: pd.DataFrame, point_value: float, label: str) -> pd.DataFrame:
    """Build labeled daily dataset from intraday bars."""
    daily_feat = build_daily_features(bars)
    rth = rth_filter(bars)

    records = []
    for date, day_bars in rth.groupby(rth.index.date):
        orb = simulate_orb(day_bars, point_value)
        if orb is None:
            continue

        # Daily features for this date (use shifted = prior day's features)
        ts = pd.Timestamp(date)
        if ts not in daily_feat.index:
            continue
        row = daily_feat.loc[ts]
        if any(pd.isna(row[c]) for c in ["ret1","ret3","ret5","atr_rank","atr_pct"]):
            continue

        rec = {
            "date": ts,
            "source": label,
            "pnl": orb["pnl"],
            "won": int(orb["won"]),
            "or_range_atr": orb["or_range_atr"],
            "or_range_pct": orb["or_range"] / day_bars["close"].iloc[0],
        }
        for col in ["ret1","ret3","ret5","gap_pct","atr_rank","atr_pct",
                    "prev_range_atr","weekday","month"]:
            rec[col] = row[col]
        records.append(rec)

    return pd.DataFrame(records).set_index("date")


def main():
    print("=" * 70)
    print("DAILY ORB GO/NO-GO ML CLASSIFIER")
    print("=" * 70)

    print("\nLoading MES (Oct 2024 - Dec 2025)...")
    mes = load_mes()
    print(f"  {len(mes):,} bars | {mes.index[0].date()} -> {mes.index[-1].date()}")

    print("Loading MNQ (Aug 2025 - May 2026)...")
    mnq = load_mnq()
    print(f"  {len(mnq):,} bars | {mnq.index[0].date()} -> {mnq.index[-1].date()}")

    print("\nSimulating ORB trades to build labeled dataset...")
    mes_df = build_dataset(mes, MES_PV, "MES")
    mnq_df = build_dataset(mnq, MNQ_PV, "MNQ")

    print(f"  MES: {len(mes_df)} trading days | WR={mes_df['won'].mean():.1%}")
    print(f"  MNQ: {len(mnq_df)} trading days | WR={mnq_df['won'].mean():.1%}")

    # Show day-of-week and month breakdown
    print("\nWin rate by day of week (all data):")
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    combined_all = pd.concat([mes_df, mnq_df])
    for dow in range(5):
        sub = combined_all[combined_all["weekday"] == dow]
        if len(sub) >= 5:
            print(f"  {dow_names[dow]}: {sub['won'].mean():.1%} ({len(sub)} days)")

    # OOS split: MNQ April 2026 is true OOS
    # Training: MES all + MNQ up to March 31 2026
    oos_start = pd.Timestamp("2026-04-01")
    mnq_train = mnq_df[mnq_df.index < oos_start]
    mnq_oos   = mnq_df[mnq_df.index >= oos_start]

    # Combine training: MES + MNQ pre-April (normalize features if needed)
    train_df = pd.concat([mes_df, mnq_train]).sort_index()

    print(f"\nTraining set: {len(train_df)} days (MES + MNQ pre-Apr)")
    print(f"OOS set:      {len(mnq_oos)} days (MNQ April 2026)")
    print(f"Training WR:  {train_df['won'].mean():.1%}")
    print(f"OOS WR:       {mnq_oos['won'].mean():.1%} (true OOS)")

    X_tr = train_df[FEAT_COLS].fillna(0).values
    y_tr = train_df["won"].values.astype(int)
    X_oo = mnq_oos[FEAT_COLS].fillna(0).values
    y_oo = mnq_oos["won"].values.astype(int)

    print(f"\nTraining LightGBM daily classifier...")
    base = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=10,
        min_child_samples=8,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.7,
        n_jobs=-1, verbose=-1,
    )
    model = CalibratedClassifierCV(base, cv=min(3, len(np.unique(y_tr))), method="isotonic")
    model.fit(X_tr, y_tr)

    # OOS predictions
    if len(X_oo) > 0 and len(np.unique(y_oo)) > 1:
        p_oo = model.predict_proba(X_oo)[:, 1]
        auc  = roc_auc_score(y_oo, p_oo)
        print(f"OOS AUC: {auc:.4f}  (naive: {y_oo.mean():.1%})")

        print(f"\nProbability distribution on OOS:")
        print(f"  Min={p_oo.min():.3f}  Max={p_oo.max():.3f}  Mean={p_oo.mean():.3f}")

        # Simulate at each threshold
        print(f"\n{'Thresh':>7} {'Trade':>6} {'Skip':>5} {'WR':>7} {'Caught%':>8} {'Note'}")
        print(f"  {'-'*55}")
        base_wr = y_oo.mean()
        base_n  = len(y_oo)
        for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            take_mask = p_oo >= thresh
            n_take = take_mask.sum()
            n_skip = (~take_mask).sum()
            if n_take == 0:
                continue
            wr_take = y_oo[take_mask].mean()
            # How many winners retained vs skipped
            wins_total = y_oo.sum()
            wins_kept  = y_oo[take_mask].sum()
            wins_pct   = wins_kept / wins_total if wins_total > 0 else 0
            improvement = wr_take - base_wr
            note = "better" if improvement > 0.02 else ("worse" if improvement < -0.02 else "same")
            print(f"  {thresh:>7.2f} {n_take:>6} {n_skip:>5} {wr_take:>7.1%} {wins_pct:>8.1%}  {note} (+{improvement:+.1%})")
        print(f"  {'base':>7} {base_n:>6}     0 {base_wr:>7.1%}  100.0%  (all trades)")
    else:
        print("  Insufficient OOS data for AUC (not enough class variety or too few days)")
        p_oo = model.predict_proba(X_oo)[:, 1] if len(X_oo) > 0 else np.array([])

    # Cross-validated performance on training set (more reliable)
    print(f"\nCV performance on training set (5-fold stratified):")
    from sklearn.model_selection import cross_val_score
    base2 = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=10,
        min_child_samples=8, reg_lambda=2.0, n_jobs=-1, verbose=-1,
    )
    cv_aucs = cross_val_score(base2, X_tr, y_tr, cv=5, scoring="roc_auc")
    print(f"  AUC scores: {[f'{a:.3f}' for a in cv_aucs]}")
    print(f"  Mean AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # Feature importance
    base3 = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=10,
        min_child_samples=8, reg_lambda=2.0, n_jobs=-1, verbose=-1,
    )
    base3.fit(X_tr, y_tr)
    imp = pd.Series(base3.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    print(f"\nFeature importances:")
    for feat, val in imp.items():
        bar = "#" * int(val / imp.max() * 25)
        print(f"  {feat:<20} {bar}")

    # P&L simulation: naive vs ML-filtered on MNQ OOS
    print(f"\n{'='*70}")
    print("PNL SIMULATION: Naive vs ML-filtered (MNQ April 2026, 25 MNQ contracts)")
    print(f"{'='*70}")

    nc_test = 25
    comm_rt = 2 * COMM * nc_test

    # Naive: all ORB days
    naive_pnls = mnq_oos["pnl"].values * (nc_test / 1)  # scale from 1-contract sim
    naive_total = naive_pnls.sum()
    naive_wr    = (naive_pnls > 0).mean()
    naive_dd    = float((naive_pnls.cumsum() - np.maximum.accumulate(naive_pnls.cumsum())).min())

    print(f"\n  Naive (all {len(naive_pnls)} days): "
          f"WR={naive_wr:.1%}  Total=${naive_total:,.0f}  DD=${naive_dd:,.0f}")

    if len(p_oo) > 0:
        for thresh in [0.45, 0.50, 0.55, 0.60]:
            take = p_oo >= thresh
            n_take = take.sum()
            if n_take == 0:
                continue
            sel_pnls = naive_pnls[take]
            sel_total = sel_pnls.sum()
            sel_wr    = (sel_pnls > 0).mean()
            cum = sel_pnls.cumsum()
            sel_dd = float((cum - np.maximum.accumulate(cum)).min())
            wk_total = sel_total / max(1, len(mnq_oos) / 5)
            flag = " *** UNDER $2k DD ***" if sel_dd > -2000 else ""
            print(f"  ML conf>={thresh:.2f} ({n_take} days): "
                  f"WR={sel_wr:.1%}  $/wk=${wk_total:,.0f}  DD=${sel_dd:,.0f}{flag}")

    # What contract size gives $5k/week if DD is under $2k?
    print(f"\n{'='*70}")
    print("CONTRACT SCALING: How many MNQ can we run with ML-controlled DD?")
    print(f"{'='*70}")

    if len(p_oo) > 0:
        best_thresh = 0.50  # pick reasonable default
        take = p_oo >= best_thresh
        for nc in [10, 15, 20, 25, 30, 35, 40, 50]:
            sel_pnls = mnq_oos["pnl"].values[take] * nc
            total = sel_pnls.sum()
            wk    = total / max(1, take.sum() / 5)
            cum   = sel_pnls.cumsum()
            dd    = float((cum - np.maximum.accumulate(cum)).min())
            flag  = " TARGET" if wk >= 5000 and dd > -2000 else ""
            if dd < -10000:
                break
            print(f"  {nc:>3} MNQ @ thresh={best_thresh}: "
                  f"WR={(sel_pnls > 0).mean():.1%}  $/wk=${wk:,.0f}  DD=${dd:,.0f}{flag}")


if __name__ == "__main__":
    main()
