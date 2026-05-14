"""
Daily ORB regime classifier — predict at market open whether today is a good ORB day.

Features (all known BEFORE 9:30 ET):
  - prior day return (close-to-close)
  - 3-day, 5-day, 10-day momentum
  - overnight gap (today's open vs yesterday's close)
  - ATR level (14-day EWM)
  - ATR percentile (vs 20-day rolling)
  - day of week (Mon=0, Fri=4)
  - month
  - prior day range / ATR

Label: 1 if that day's ORB trade (long only, entry after 10:04, PT=3x ATR, SL=1.5x ATR)
       would have been profitable, 0 otherwise.
       Days with no ORB signal are excluded.

Uses LightGBM with walk-forward cross-validation (no leakage).
Simulates: take ORB only on high-confidence days (conf >= thresh).
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

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
N_CONTRACTS  = 10
OR_END_M     = 64   # 10:04 = 604 minutes from midnight, bar index within day
# OR window: 9:30–10:04 = 35 mins / 5-min bars = 7 bars
PT_ATR       = 3.0
SL_ATR       = 1.5
MAX_BARS_IN  = 24   # time stop = 24 bars after entry


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


def simulate_orb_day(day_bars: pd.DataFrame) -> dict | None:
    """
    Simulate one ORB trade for a single day.
    Returns dict with 'entered', 'pnl', 'won', 'or_high', 'or_low', 'entry_price'
    or None if no signal.
    """
    # RTH bars within session
    rth = day_bars[(day_bars.index.hour >= 9) & (day_bars.index.hour < 16)]
    if len(rth) < 15:
        return None

    # Opening range: 9:30–10:04 (bars where hour=9 or (hour=10 and minute<=4))
    or_bars = rth[(rth.index.hour == 9) |
                  ((rth.index.hour == 10) & (rth.index.minute <= 4))]
    if len(or_bars) < 5:
        return None

    or_high = or_bars["high"].max()
    or_low  = or_bars["low"].min()
    or_range = or_high - or_low

    # ATR at end of OR
    close_s = rth["close"]
    prev_s  = close_s.shift(1)
    hi_s    = rth["high"]
    lo_s    = rth["low"]
    tr = pd.concat([(hi_s - lo_s), (hi_s - prev_s).abs(), (lo_s - prev_s).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(span=14, adjust=False).mean()
    atr_at_entry = atr_series.loc[or_bars.index[-1]]

    if atr_at_entry <= 0 or np.isnan(atr_at_entry):
        return None

    # Entry: first bar after 10:04 that closes above or_high
    entry_window = rth[(rth.index.hour > 10) |
                       ((rth.index.hour == 10) & (rth.index.minute > 4))]
    entry_window = entry_window[entry_window.index <= rth.index[rth.index.hour <= 12][-1]]
    if len(entry_window) == 0:
        return None

    entered = False
    for idx_i, (ts, row) in enumerate(entry_window.iterrows()):
        if row["high"] > or_high:
            # Enter on breakout
            entry_price = or_high + TICK_SIZE
            pt_price    = entry_price + PT_ATR * atr_at_entry
            sl_price    = entry_price - SL_ATR * atr_at_entry
            bar_stop    = idx_i + MAX_BARS_IN

            # Simulate exit
            pnl = None
            remaining = list(entry_window.iloc[idx_i+1:].itertuples())
            for j, bar in enumerate(remaining):
                if j >= MAX_BARS_IN:
                    # Time stop
                    pnl = (bar.close - entry_price) * N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * N_CONTRACTS
                    break
                if bar.low <= sl_price:
                    pnl = (sl_price - TICK_SIZE - entry_price) * N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * N_CONTRACTS
                    break
                if bar.high >= pt_price:
                    pnl = (pt_price - TICK_SIZE - entry_price) * N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * N_CONTRACTS
                    break
            else:
                # Last bar time stop
                if remaining:
                    pnl = (remaining[-1].close - entry_price) * N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * N_CONTRACTS
                else:
                    pnl = 0.0

            return {
                "entered": True,
                "pnl": pnl,
                "won": pnl > 0,
                "entry_price": entry_price,
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "atr": atr_at_entry,
            }

    return None  # No breakout


def build_daily_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build one row per day with PRE-MARKET features only."""
    # Daily OHLCV
    daily = bars.groupby(bars.index.date).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)

    # Daily ATR
    prev = daily["close"].shift(1)
    tr = pd.concat([daily["high"]-daily["low"],
                    (daily["high"]-prev).abs(),
                    (daily["low"]-prev).abs()], axis=1).max(axis=1)
    daily["atr"] = tr.ewm(span=14, adjust=False).mean()
    daily["atr_pct"] = daily["atr"] / daily["close"]

    # ATR percentile vs rolling 20-day
    daily["atr_pct_rank"] = daily["atr_pct"].rolling(20).rank(pct=True)

    # Returns
    daily["ret1"] = daily["close"].pct_change(1)
    daily["ret3"] = daily["close"].pct_change(3)
    daily["ret5"] = daily["close"].pct_change(5)
    daily["ret10"] = daily["close"].pct_change(10)

    # Overnight gap = today's open vs yesterday's close
    daily["gap"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    # Prior day range / ATR
    daily["prev_range_atr"] = (daily["high"].shift(1) - daily["low"].shift(1)) / daily["atr"].shift(1)

    # Weekday, month
    daily["weekday"] = daily.index.dayofweek
    daily["month"]   = daily.index.month

    return daily


def main():
    print("Loading bars...")
    bars = load_bars()
    print(f"  {len(bars):,} bars [{bars.index[0].date()} → {bars.index[-1].date()}]")

    # Build daily features
    print("Building daily features...")
    daily = build_daily_features(bars)

    # Simulate ORB for each day
    print("Simulating ORB trades per day...")
    days = bars.groupby(bars.index.date)
    results = []
    for date, day_bars in days:
        r = simulate_orb_day(day_bars)
        if r is not None and r["entered"]:
            results.append({
                "date": pd.Timestamp(date),
                **r,
            })

    orb_df = pd.DataFrame(results).set_index("date")
    orb_df.index = pd.to_datetime(orb_df.index)

    print(f"\n  Total ORB trades simulated: {len(orb_df)}")
    print(f"  Win rate (all): {orb_df['won'].mean():.1%}")
    print(f"  Total PnL: ${orb_df['pnl'].sum():,.0f}")
    n_weeks_all = len(orb_df) / 3.0   # ~3 trades/week estimate
    print(f"  Est. $/week (all days): ${orb_df['pnl'].sum() / n_weeks_all:,.0f}")

    # Join with daily features (using PRIOR day features — no leakage)
    daily_feat = daily.shift(1)  # shift so today's row has YESTERDAY's data available pre-open
    daily_feat["gap"]     = daily["gap"]      # gap IS known pre-open (today's open already happened)
    daily_feat["weekday"] = daily["weekday"]
    daily_feat["month"]   = daily["month"]

    feat_cols = ["ret1","ret3","ret5","ret10","gap","atr_pct","atr_pct_rank",
                 "prev_range_atr","weekday","month"]

    merged = orb_df.join(daily_feat[feat_cols], how="left").dropna(subset=feat_cols)
    print(f"  After joining features: {len(merged)} days")

    X = merged[feat_cols].values
    y = merged["won"].values.astype(int)

    print(f"\nLabel distribution: {y.mean():.1%} wins")

    # Walk-forward evaluation: train on first 60% of days, test on last 40%
    split_idx = int(len(merged) * 0.6)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_oo, y_oo = X[split_idx:], y[split_idx:]
    oos_df = merged.iloc[split_idx:]

    print(f"  Train: {len(X_tr)} days | OOS: {len(X_oo)} days")

    base = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=10, min_child_samples=10, reg_lambda=2.0,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1,
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_tr, y_tr)

    proba_oo = model.predict_proba(X_oo)[:, 1]
    auc = roc_auc_score(y_oo, proba_oo) if len(np.unique(y_oo)) > 1 else 0.5
    print(f"\n  OOS AUC: {auc:.4f}  |  Naive WR: {y_oo.mean():.1%}")

    # Simulate at confidence thresholds
    gross_win  = PT_ATR * 10 * TICK_SIZE * POINT_VALUE * N_CONTRACTS  # approx
    gross_loss = SL_ATR * 10 * TICK_SIZE * POINT_VALUE * N_CONTRACTS
    comm_rt    = 2 * COMMISSION * N_CONTRACTS
    be_wr      = (gross_loss + comm_rt) / (gross_win + gross_loss)
    print(f"  Approx breakeven WR: {be_wr:.1%}")

    print(f"\n  Simulating with ML regime gate on OOS period:")
    print(f"  {'Thresh':>7} {'Days':>5} {'WR':>7} {'PnL':>9} {'$/wk':>9}  vs_baseline")

    oos_pnl_all = oos_df["pnl"].sum()
    n_weeks_oos = len(oos_df) / 3.0
    baseline_wk = oos_pnl_all / n_weeks_oos

    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = proba_oo >= thresh
        n = mask.sum()
        if n == 0:
            continue
        pnl_sel = oos_df["pnl"].values[mask]
        wr  = (pnl_sel > 0).mean()
        tot = pnl_sel.sum()
        wk  = tot / max(1, n / 3.0)
        diff = wk - baseline_wk
        flag = "+" if diff > 0 else ""
        print(f"  {thresh:>7.2f} {n:>5} {wr:>7.1%} {tot:>9,.0f} {wk:>9,.0f}  {flag}{diff:,.0f}/wk vs baseline")

    print(f"  {'baseline':>7} {len(oos_df):>5} {y_oo.mean():>7.1%} {oos_pnl_all:>9,.0f} {baseline_wk:>9,.0f}")

    # Feature importance
    base2 = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=10, min_child_samples=10, reg_lambda=2.0,
        n_jobs=-1, verbose=-1,
    )
    base2.fit(X_tr, y_tr)
    imp = pd.Series(base2.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Feature importances:")
    for f, v in imp.items():
        bar = "#" * int(v / imp.max() * 30)
        print(f"    {f:<20} {bar}")

    # Day-of-week breakdown
    print(f"\n  Win rate by day of week (all ORB trades):")
    dow_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri"}
    merged2 = merged.copy()
    for dow, name in dow_map.items():
        sub = merged2[merged2["weekday"] == dow]
        if len(sub) >= 3:
            print(f"    {name}: {sub['won'].mean():.1%}  ({len(sub)} trades, "
                  f"${sub['pnl'].sum():,.0f} total)")

    # Monthly breakdown
    print(f"\n  Win rate by month (all ORB trades):")
    for m in sorted(merged2["month"].unique()):
        sub = merged2[merged2["month"] == m]
        mo = pd.Timestamp(f"2026-{int(m):02d}-01").strftime("%b")
        print(f"    {mo}: {sub['won'].mean():.1%}  ({len(sub)} trades, "
              f"${sub['pnl'].sum():,.0f} total)")

    # Best projected outcomes
    print(f"\n{'='*70}")
    print(f"PROJECTED OUTCOMES (10 MNQ, incl. commission):")
    print(f"{'='*70}")
    total_pnl = merged["pnl"].sum()
    total_wks = len(merged) / 3.0
    print(f"  All ORB days:    ${total_pnl:,.0f} total | ${total_pnl/total_wks:,.0f}/wk | WR={merged['won'].mean():.1%}")

    # Best ML threshold
    best_thresh = None
    best_wk = baseline_wk
    for thresh in np.arange(0.40, 0.80, 0.05):
        mask = proba_oo >= thresh
        n = mask.sum()
        if n < 5:
            continue
        pnl_sel = oos_df["pnl"].values[mask]
        wk = pnl_sel.sum() / max(1, n / 3.0)
        if wk > best_wk:
            best_wk = wk
            best_thresh = thresh

    if best_thresh:
        mask = proba_oo >= best_thresh
        sel = oos_df[mask]
        print(f"  ML-filtered:     ${sel['pnl'].sum():,.0f} total | ${best_wk:,.0f}/wk | "
              f"WR={sel['won'].mean():.1%} @ conf≥{best_thresh:.2f} ({mask.sum()} days)")
        print(f"  Days skipped:    {(~mask).sum()} out of {len(oos_df)} OOS days")


if __name__ == "__main__":
    main()
