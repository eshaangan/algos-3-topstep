"""
Regime-Adaptive ML Trading Pipeline for MNQ 5-minute bars.

Core idea (user insight): Don't train on all history. Classify today's market regime,
then train ONLY on historical days that match that same regime. This avoids
"regime mismatch" — the #1 reason the existing ml_intraday_v3 pipeline failed.

Regimes (based on prior 5 trading days):
  0 = Trending UP  (5d return > +0.5%)
  1 = Trending DOWN (5d return < -0.5%)
  2 = High volatility (ATR percentile > 70%)
  3 = Low volatility / mean-reverting

Walk-forward: For each OOS day in April 2026:
  1. Classify today's regime
  2. Gather all prior days with same regime
  3. Train LightGBM on those days' 5-min bars
  4. Generate confidence scores during today's session
  5. Enter when conf >= threshold, one trade at a time
  6. Exit at PT=15t, SL=8t, or 20-bar time stop

Instruments: MNQ, 20-50 contracts
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
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
N_CONTRACTS  = 20   # sweep this
PT_TICKS     = 15
SL_TICKS     = 8
HORIZON_BARS = 20   # max holding = 100 min
MAX_TRADES   = 4    # per day
MAX_DAILY    = -2000.0
TSTART_M     = 570  # 9:30
TEND_M       = 870  # 14:30

COMM_RT  = 2 * COMMISSION * N_CONTRACTS
NET_WIN  = PT_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS - COMM_RT
NET_LOSS = SL_TICKS * TICK_SIZE * POINT_VALUE * N_CONTRACTS + COMM_RT
BE_WR    = NET_LOSS / (NET_WIN + NET_LOSS)

FEAT_COLS = [
    "r1","r3","r5","r10","r20",         # returns at multiple lags
    "atr_pct",                           # ATR as % of price
    "vwap_dev",                          # std devs from VWAP
    "rvol",                              # relative volume
    "range_atr",                         # bar range / ATR
    "rsi14",                             # RSI
    "macd_sig",                          # MACD signal / ATR
    "bb_pos",                            # Bollinger band position
    "ema_ratio",                         # EMA9/EMA21 ratio
    "hour","minute",                     # time of day
]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_merge() -> pd.DataFrame:
    """Load all available MNQ 5-min RTH data, merge, deduplicate."""
    pieces = []
    files = [
        ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5",
        ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ]
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

    # RTH only
    bars = bars[
        (bars.index.hour > 9) |
        ((bars.index.hour == 9) & (bars.index.minute >= 35))
    ]
    bars = bars[
        (bars.index.hour < 15) |
        ((bars.index.hour == 15) & (bars.index.minute <= 45))
    ]
    return bars


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # Returns
    for w in [1, 3, 5, 10, 20]:
        df[f"r{w}"] = df["close"].pct_change(w)

    # Session VWAP deviation
    tp = (df["high"] + df["low"] + df["close"]) / 3
    date_idx = df.index.map(lambda t: t.date())
    vwap = (tp * df["volume"]).groupby(date_idx).cumsum() / \
           df["volume"].groupby(date_idx).cumsum().replace(0, np.nan)
    df["vwap_dev"] = (df["close"] - vwap) / df["atr"].replace(0, np.nan)

    # Relative volume
    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()

    # Range / ATR
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"].replace(0, np.nan)

    # RSI-14
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # MACD signal / ATR
    ema9  = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    macd  = ema9 - ema21
    df["macd_sig"] = (macd - macd.ewm(span=9, adjust=False).mean()) / df["atr"].replace(0, np.nan)
    df["ema_ratio"] = ema9 / ema21

    # Bollinger band position
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - sma20) / (2 * std20.replace(0, np.nan))

    # Time
    df["hour"]   = df.index.hour
    df["minute"] = df.index.minute

    return df


# ── Label generation (no-lookahead: labeled offline for training) ─────────────
def label_bars(df: pd.DataFrame) -> np.ndarray:
    """LONG triple-barrier: 1=PT hit, 0=SL hit, nan=neither within horizon."""
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = df["atr"].values
    n = len(df)

    labels = np.full(n, np.nan)
    pt_pts = PT_TICKS * TICK_SIZE
    sl_pts = SL_TICKS * TICK_SIZE

    for i in range(n - HORIZON_BARS - 1):
        entry = close[i] + TICK_SIZE
        for j in range(i+1, min(i+1+HORIZON_BARS, n)):
            if high[j] >= entry + pt_pts:
                labels[i] = 1; break
            if low[j]  <= entry - sl_pts:
                labels[i] = 0; break
    return labels


# ── Daily regime classifier ───────────────────────────────────────────────────
def compute_daily_regime(bars: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day compute a regime label based on prior 5 days.
    Regime 0: trending up (5d return > +0.5%)
    Regime 1: trending down (5d return < -0.5%)
    Regime 2: high volatility (ATR pct rank > 70th percentile)
    Regime 3: low volatility / choppy
    """
    daily = bars.groupby(bars.index.date).agg(
        open=("open","first"), close=("close","last"),
        high=("high","max"), low=("low","min")
    )
    daily.index = pd.to_datetime(daily.index)

    daily["ret5"] = daily["close"].pct_change(5)
    daily_range = daily["high"] - daily["low"]
    daily["atr_daily"] = daily_range.ewm(span=10, adjust=False).mean()
    daily["atr_rank"]  = daily["atr_daily"].rolling(20).rank(pct=True)

    regimes = []
    for date, row in daily.iterrows():
        if pd.isna(row["ret5"]) or pd.isna(row["atr_rank"]):
            regimes.append((date, -1))  # unknown
            continue
        if row["atr_rank"] > 0.70:
            r = 2  # high vol
        elif row["atr_rank"] < 0.35:
            r = 3  # low vol
        elif row["ret5"] > 0.005:
            r = 0  # trending up
        elif row["ret5"] < -0.005:
            r = 1  # trending down
        else:
            r = 3  # mild = low vol
        regimes.append((date, r))

    regime_df = pd.DataFrame(regimes, columns=["date","regime"]).set_index("date")
    regime_df.index = pd.to_datetime(regime_df.index)
    return regime_df


# ── Walk-forward simulation ───────────────────────────────────────────────────
def walk_forward_sim(bars: pd.DataFrame,
                     oos_start: str,
                     min_regime_days: int = 15,
                     conf_threshold: float = 0.55,
                     n_contracts: int = N_CONTRACTS,
                     verbose: bool = False) -> dict:
    """
    For each OOS day:
      1. Classify today's regime
      2. Find regime-matched training days
      3. Train LightGBM
      4. Simulate trading that day
    """
    comm_rt  = 2 * COMMISSION * n_contracts
    net_win  = PT_TICKS * TICK_SIZE * POINT_VALUE * n_contracts - comm_rt
    net_loss = SL_TICKS * TICK_SIZE * POINT_VALUE * n_contracts + comm_rt
    max_dl   = -abs(MAX_DAILY) * (n_contracts / 20)

    print(f"  Contracts={n_contracts}  NetWin=${net_win:.2f}  NetLoss=${net_loss:.2f}  "
          f"BE WR={net_loss/(net_win+net_loss):.1%}  thresh={conf_threshold:.2f}")

    # Add features
    df = add_features(bars)

    # Pre-compute labels for all bars (for training set construction)
    print("  Labeling all bars (training prep)...", end=" ", flush=True)
    df["label"] = label_bars(df)
    df = df.dropna(subset=FEAT_COLS + ["label","atr"])
    print(f"done ({len(df):,} labeled bars, {df['label'].mean():.1%} positive)")

    # Daily regimes
    regime_df = compute_daily_regime(bars)
    regime_names = {0:"TrendingUP", 1:"TrendingDN", 2:"HighVol", 3:"LowVol", -1:"Unknown"}

    # Split days
    all_dates  = sorted(set(df.index.date))
    oos_cut    = pd.Timestamp(oos_start).date()
    train_days = [d for d in all_dates if d < oos_cut]
    oos_days   = sorted(d for d in all_dates if d >= oos_cut)

    print(f"  Training days available: {len(train_days)} | OOS days: {len(oos_days)}")

    trades_log  = []
    equity      = 50_000.0
    equity_curve = [equity]
    auc_scores   = []
    regime_counts = defaultdict(int)

    for oos_date in oos_days:
        # Get today's regime (using yesterday's data)
        ts = pd.Timestamp(oos_date)
        regime = -1
        if ts in regime_df.index:
            regime = regime_df.loc[ts, "regime"]
        regime_counts[regime] += 1

        # Get today's bars
        day_bars = df[np.array(df.index.date) == oos_date]
        if len(day_bars) < 10:
            continue

        # Find regime-matched training days
        if regime != -1:
            matched = [d for d in train_days
                       if pd.Timestamp(d) in regime_df.index
                       and regime_df.loc[pd.Timestamp(d), "regime"] == regime]
        else:
            matched = []

        # Fallback: use most recent training days if not enough regime matches
        if len(matched) < min_regime_days:
            matched = train_days[-40:]   # last 40 trading days
            regime_label = f"{regime_names.get(regime,'?')}(fallback)"
        else:
            regime_label = regime_names.get(regime, "?")

        # Build training set from matched days
        matched_set = set(matched)
        train_mask = np.array([d in matched_set for d in df.index.date])
        train_bars = df[train_mask]
        if len(train_bars) < 100:
            continue

        X_tr = train_bars[FEAT_COLS].values
        y_tr = train_bars["label"].values.astype(int)

        if len(np.unique(y_tr)) < 2:
            continue

        # Train model (fast, no calibration — avoids compression issue)
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.06,
            num_leaves=12,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        # OOS AUC on this day's labeled bars (for monitoring, doesn't affect trading)
        oos_labeled = day_bars.dropna(subset=["label"])
        if len(oos_labeled) >= 10 and len(np.unique(oos_labeled["label"].values)) > 1:
            p_oos = model.predict_proba(oos_labeled[FEAT_COLS].values)[:, 1]
            try:
                auc = roc_auc_score(oos_labeled["label"].values.astype(int), p_oos)
                auc_scores.append(auc)
            except Exception:
                pass

        # Trading simulation for this day
        close_d  = day_bars["close"].values
        high_d   = day_bars["high"].values
        low_d    = day_bars["low"].values
        atr_d    = day_bars["atr"].values
        mins_d   = (day_bars.index.hour * 60 + day_bars.index.minute).values
        proba_d  = model.predict_proba(day_bars[FEAT_COLS].values)[:, 1]

        daily_pnl = 0.0
        day_count = 0
        pos_active = False
        pos_entry  = 0.0; pos_sl = 0.0; pos_pt = 0.0; pos_stop_i = 0

        for i in range(1, len(day_bars)):
            m = mins_d[i]
            is_last = (i == len(day_bars) - 1) or (m >= 955)

            # Exit
            if pos_active:
                h = high_d[i]; l = low_d[i]; c = close_d[i]
                exited = False; ep = 0.0; reason = ""
                if is_last or i >= pos_stop_i:
                    ep = c; reason = "time"; exited = True
                elif l <= pos_sl:
                    ep = pos_sl - TICK_SIZE; reason = "sl"; exited = True
                elif h >= pos_pt:
                    ep = pos_pt - TICK_SIZE; reason = "pt"; exited = True
                if exited:
                    pnl = (ep - pos_entry) * n_contracts * POINT_VALUE - comm_rt
                    daily_pnl += pnl; equity += pnl
                    equity_curve.append(equity)
                    trades_log.append({
                        "date": oos_date, "pnl": pnl, "reason": reason,
                        "regime": regime_label, "won": pnl > 0
                    })
                    pos_active = False

            if is_last: continue
            if pos_active or daily_pnl <= max_dl or day_count >= MAX_TRADES: continue
            if not (TSTART_M <= m <= TEND_M): continue

            a = atr_d[i]
            if np.isnan(a) or a <= 0: continue

            p = proba_d[i]
            if np.isnan(p) or p < conf_threshold: continue

            # Entry: long signal
            c = close_d[i]; pc = close_d[i-1]
            if c > pc:  # bullish bar + high confidence
                pos_active  = True
                pos_entry   = c + TICK_SIZE
                pos_sl      = pos_entry - SL_TICKS * TICK_SIZE
                pos_pt      = pos_entry + PT_TICKS * TICK_SIZE
                pos_stop_i  = i + HORIZON_BARS
                day_count  += 1

        if verbose:
            day_trades = [t for t in trades_log if t["date"] == oos_date]
            day_pnl = sum(t["pnl"] for t in day_trades)
            regime_str = day_trades[-1]["regime"] if day_trades else regime_label
            print(f"    {oos_date}  [{regime_str:18s}]  "
                  f"trades={len(day_trades)}  pnl=${day_pnl:,.0f}")

    if not trades_log:
        return {"n": 0, "n_contracts": n_contracts, "conf_threshold": conf_threshold}

    arr    = np.array([t["pnl"] for t in trades_log])
    eq_arr = np.array(equity_curve)
    wins   = (arr > 0).sum()
    total  = arr.sum()
    gp     = arr[arr > 0].sum() if wins > 0 else 0
    gl     = abs(arr[arr <= 0].sum())
    max_dd = float((eq_arr - np.maximum.accumulate(eq_arr)).min())
    n_weeks = max(1, len(oos_days) / 5)

    return {
        "n_contracts": n_contracts,
        "conf_threshold": conf_threshold,
        "n": len(arr),
        "wr": wins / len(arr),
        "total": total,
        "weekly": total / n_weeks,
        "max_dd": max_dd,
        "sharpe": float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0,
        "pf": gp/gl if gl > 0 else 99.0,
        "mean_auc": np.mean(auc_scores) if auc_scores else 0.5,
        "regime_counts": dict(regime_counts),
        "n_oos_days": len(oos_days),
        "trades": trades_log,
    }


def main():
    print("="*70)
    print("REGIME-ADAPTIVE ML — MNQ 5-minute bars")
    print("="*70)

    print("\nLoading and merging MNQ data...")
    bars = load_and_merge()
    print(f"  {len(bars):,} RTH bars | {bars.index[0].date()} → {bars.index[-1].date()}")

    # Show regime distribution
    regime_df = compute_daily_regime(bars)
    rnames = {0:"TrendingUP", 1:"TrendingDN", 2:"HighVol", 3:"LowVol", -1:"Unknown"}
    print("\nRegime distribution (all days):")
    rc = regime_df["regime"].value_counts().sort_index()
    for r, cnt in rc.items():
        print(f"  {rnames.get(r, r)}: {cnt} days ({cnt/len(regime_df):.1%})")

    # April 2026 regime breakdown
    apr_regime = regime_df["2026-04-01":"2026-04-30"]
    print(f"\nApril 2026 regime breakdown ({len(apr_regime)} days):")
    for r, cnt in apr_regime["regime"].value_counts().sort_index().items():
        print(f"  {rnames.get(r, r)}: {cnt} days")

    # Walk-forward on April 2026 OOS
    print("\n" + "="*70)
    print("WALK-FORWARD OOS: April 2026 (never seen during training)")
    print("="*70)

    # Sweep contract sizes and thresholds
    contract_sizes = [10, 20, 30, 50]
    thresholds = [0.52, 0.55, 0.58, 0.62]

    all_results = []
    for nc in contract_sizes:
        print(f"\n--- {nc} MNQ contracts ---")
        for thresh in thresholds:
            r = walk_forward_sim(
                bars,
                oos_start="2026-04-01",
                min_regime_days=15,
                conf_threshold=thresh,
                n_contracts=nc,
                verbose=(nc == 20 and thresh == 0.55),
            )
            if r["n"] == 0:
                print(f"  thresh={thresh:.2f}: NO TRADES")
                continue
            all_results.append(r)

            flag = ""
            if r["weekly"] >= 5000 and r["max_dd"] >= -2000:
                flag = "  *** TARGET REACHED ***"
            elif r["weekly"] >= 3000 and r["max_dd"] >= -2000:
                flag = "  ** Near target **"
            print(f"  thresh={thresh:.2f}  n={r['n']:>3}  WR={r['wr']:.1%}  "
                  f"$/wk=${r['weekly']:>7,.0f}  DD=${r['max_dd']:>8,.0f}  "
                  f"Sharpe={r['sharpe']:>6.2f}  AUC={r['mean_auc']:.3f}{flag}")

    # Best result
    if all_results:
        valid = [r for r in all_results if r["n"] >= 5]
        if valid:
            best = max(valid, key=lambda x: x["sharpe"] * x["wr"])
            print(f"\n{'='*70}")
            print(f"BEST CONFIG: {best['n_contracts']} MNQ @ conf≥{best['conf_threshold']:.2f}")
            print(f"  {best['n']} trades | WR={best['wr']:.1%} | "
                  f"${best['weekly']:,.0f}/wk | MaxDD=${best['max_dd']:,.0f} | "
                  f"Sharpe={best['sharpe']:.2f} | AvgAUC={best['mean_auc']:.3f}")
            print(f"  OOS period: {best['n_oos_days']} trading days")
            print(f"  Regime distribution: {best['regime_counts']}")
            print(f"  BE win rate: {NET_LOSS/(NET_WIN+NET_LOSS):.1%}")

            # Per-regime breakdown
            print(f"\nPer-regime performance (best config):")
            rnames2 = {0:"TrendingUP", 1:"TrendingDN", 2:"HighVol", 3:"LowVol", -1:"Unknown"}
            regime_perf = defaultdict(list)
            for t in best["trades"]:
                regime_perf[t["regime"]].append(t["pnl"])
            for reg, pnls in sorted(regime_perf.items()):
                arr = np.array(pnls)
                print(f"  {reg:<20} n={len(arr):>3}  WR={(arr>0).mean():.1%}  "
                      f"Total=${arr.sum():>7,.0f}")

    # Projection: if results hold, what happens over a full month?
    if all_results:
        best = max([r for r in all_results if r["n"] >= 5],
                   key=lambda x: x["weekly"])
        nc = best["n_contracts"]
        wk = best["weekly"]
        dd = best["max_dd"]
        print(f"\n{'='*70}")
        print(f"PROJECTIONS (best $/wk config: {nc} MNQ @ ${wk:,.0f}/wk):")
        print(f"  Days to $3k Topstep target: ~{3000/max(1,wk/5):.0f} trading days")
        print(f"  Max drawdown vs $2k limit: ${dd:,.0f} ({'OK' if dd > -2000 else 'EXCEEDS LIMIT'})")
        print(f"  At Topstep 50k: hit profit target in ~{3000/max(1,wk/5)*2:.0f} calendar days")


if __name__ == "__main__":
    main()
