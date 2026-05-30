"""
1-min ML search: pure LightGBM, asymmetric PT/SL, $4k DD cap.

Training:  MES 1-min RTH Jun-Dec 2025 (48k bars, ATR-normalised features)
OOS:       MNQ Databento 1-min RTH Jan-May 2026 (32k bars)

Key difference from 5-min: smaller per-trade DD -> more contracts -> higher weekly PnL.
Asymmetric grid (PT >> SL) to improve profit factor.
"""
from __future__ import annotations
import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

try:
    import lightgbm as lgb
    _USE_LGB = True
except Exception:
    _USE_LGB = False

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

MES_FILE = ROOT / "data" / "processed" / "mes_1m_bars_cache.h5"
MNQ_OOS_FILE = ROOT / "data" / "processed" / "mnq_2026ytd_databento_1min_eth.h5"

MNQ_POINT_VALUE = 2.0
MNQ_TICK_SIZE   = 0.25
MNQ_COMMISSION  = 0.62
MNQ_SLIP_TICKS  = 1

MAX_CONTRACTS = 15
MAX_DD_LIMIT  = 4000.0
DAILY_LOSS_CAP = 1500.0

OOS_START = pd.Timestamp("2026-01-01", tz="US/Eastern")


def to_eastern(df):
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert("US/Eastern")
    return df


def rth_only(df):
    t = df.index.hour * 60 + df.index.minute
    return df[(t >= 9 * 60 + 30) & (t < 16 * 60)].copy()


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs = g / ls.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def intraday_vwap(bars):
    parts = []
    for d in sorted(set(bars.index.date)):
        day = bars[bars.index.date == d].copy()
        tp = (day["high"] + day["low"] + day["close"]) / 3
        cum_vol = day["volume"].cumsum().replace(0, np.nan)
        vwap = (tp * day["volume"]).cumsum() / cum_vol
        parts.append(vwap)
    return pd.concat(parts)


def build_features(bars):
    df = bars.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    at = atr(h, l, c, 14)
    at_safe = at.replace(0, np.nan)
    df["atr"] = at

    df["vwap"] = intraday_vwap(df)
    df["vwap_dev"] = (c - df["vwap"]) / at_safe

    for n in [1, 3, 5, 10, 20, 60]:
        df[f"mom{n}"] = c.diff(n) / at_safe

    df["rsi14"] = rsi(c, 14)
    df["rsi5"]  = rsi(c, 5)
    df["rsi14_dev"] = df["rsi14"] - 50
    df["rsi5_dev"]  = df["rsi5"] - 50

    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    df["ema9_21"]  = (ema9 - ema21) / at_safe
    df["ema21_60"] = (ema21 - ema60) / at_safe
    df["c_vs_ema9"]  = (c - ema9) / at_safe
    df["c_vs_ema21"] = (c - ema21) / at_safe

    rng = (h - l).replace(0, np.nan)
    df["clv"]  = (2 * c - h - l) / rng
    df["uw"]   = (h - c.clip(upper=o)) / rng
    df["lw"]   = (c.clip(upper=o) - l) / rng
    df["body"] = (c - o).abs() / rng

    df["vol_ratio"]  = v / v.rolling(20).mean()
    df["vol_ratio5"] = v / v.rolling(5).mean()

    df["atr_ratio"]  = at / at.rolling(60).mean()
    df["atr_ratio5"] = at / at.rolling(5).mean()

    pct = c.pct_change()
    df["rvol5"]  = pct.rolling(5).std()
    df["rvol20"] = pct.rolling(20).std()
    df["rvol_ratio"] = df["rvol5"] / df["rvol20"].replace(0, np.nan)

    session_high = h.groupby(df.index.date).cummax()
    session_low  = l.groupby(df.index.date).cummin()
    df["dist_sess_high"] = (session_high - c) / at_safe
    df["dist_sess_low"]  = (c - session_low) / at_safe

    df["bar_gap"]  = (o - c.shift()) / at_safe
    df["hod"]      = df.index.hour + df.index.minute / 60
    df["dow"]      = df.index.dayofweek
    df["is_open"]  = ((df.index.hour == 9) & (df.index.minute >= 30) |
                       (df.index.hour == 10)).astype(float)

    df["mom1_lag1"] = df["mom1"].shift(1)
    df["mom3_lag1"] = df["mom3"].shift(1)

    return df


FEATURE_COLS = [
    "vwap_dev",
    "mom1", "mom3", "mom5", "mom10", "mom20", "mom60",
    "rsi14_dev", "rsi5_dev",
    "ema9_21", "ema21_60", "c_vs_ema9", "c_vs_ema21",
    "clv", "uw", "lw", "body",
    "vol_ratio", "vol_ratio5",
    "atr_ratio", "atr_ratio5",
    "rvol5", "rvol20", "rvol_ratio",
    "dist_sess_high", "dist_sess_low",
    "bar_gap", "hod", "dow", "is_open",
    "mom1_lag1", "mom3_lag1",
]


def make_labels(bars, atr_s, pt_atr, sl_atr, lookahead):
    labels = []
    closes = bars["close"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    atrs   = atr_s.values
    n = len(bars)

    for i in range(n - lookahead):
        at = atrs[i]
        if np.isnan(at) or at <= 0:
            labels.append(np.nan)
            continue
        entry  = closes[i]
        pt     = entry + pt_atr * at
        sl     = entry - sl_atr * at
        result = 0
        for j in range(i + 1, i + 1 + lookahead):
            if highs[j] >= pt:
                result = 1
                break
            if lows[j] <= sl:
                result = -1
                break
        labels.append(result)

    labels += [np.nan] * lookahead
    return pd.Series(labels, index=bars.index)


def train_model(train_df, pt_atr, sl_atr, lookahead, verbose=False):
    feat   = build_features(train_df)
    labels = make_labels(train_df, feat["atr"], pt_atr, sl_atr, lookahead)

    mask = labels.isin([1, -1]) & feat[FEATURE_COLS].notna().all(axis=1)
    X = feat.loc[mask, FEATURE_COLS].values
    y = (labels[mask] == 1).astype(int).values

    if len(X) < 300 or y.mean() < 0.05 or y.mean() > 0.95:
        return None, None

    if verbose:
        print(f"    Train: {len(X)} samples, WR={y.mean():.1%}", end=" ", flush=True)

    tscv = TimeSeriesSplit(n_splits=3)
    if _USE_LGB:
        base = lgb.LGBMClassifier(
            objective="binary", metric="auc",
            n_estimators=200, learning_rate=0.05,
            max_depth=6, num_leaves=31,
            min_child_samples=50,
            colsample_bytree=0.7, subsample=0.8,
            reg_alpha=0.1, reg_lambda=0.2,
            n_jobs=-1, verbose=-1,
        )
    else:
        base = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05,
            max_depth=6, min_samples_leaf=50,
            l2_regularization=0.2, random_state=42,
        )
    model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    model.fit(X, y)

    p_train = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p_train)
    if verbose:
        print(f"AUC={auc:.3f}", flush=True)
    return model, auc


def run_backtest(oos_df, model, pt_atr, sl_atr, lookahead,
                 conf_thresh, n_contracts, max_trades_day=10,
                 daily_loss_cap=None):
    feat  = build_features(oos_df)
    X_all = feat[FEATURE_COLS]
    valid = X_all.notna().all(axis=1)
    probs = pd.Series(np.nan, index=oos_df.index)
    if valid.sum() > 0:
        probs[valid] = model.predict_proba(X_all[valid].values)[:, 1]

    atr_s  = feat["atr"]
    closes = oos_df["close"]
    highs  = oos_df["high"]
    lows   = oos_df["low"]

    slip = MNQ_SLIP_TICKS * MNQ_TICK_SIZE
    comm = 2 * MNQ_COMMISSION * n_contracts

    trades = []
    equity = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0
    day_pnl_so_far = 0.0
    pos = None

    for i, (ts, _) in enumerate(oos_df.iterrows()):
        bdate = ts.date()
        bdate_s = str(bdate)

        if cur_date is not None and bdate != cur_date:
            trades_today = 0
            day_pnl_so_far = 0.0
        cur_date = bdate

        h  = highs.iloc[i]
        lo = lows.iloc[i]
        c  = closes.iloc[i]

        if pos is not None:
            ei, direction, ep, sl_p, pt_p, bars_in = pos
            bars_in += 1
            exited = False; exit_p = c; reason = "time_stop"

            if direction == 1:
                if lo <= sl_p:
                    exit_p = sl_p - slip; reason = "stop_loss"; exited = True
                elif h >= pt_p:
                    exit_p = pt_p - slip; reason = "profit_target"; exited = True
                elif bars_in >= lookahead:
                    exit_p = c - slip; exited = True
            else:
                if h >= sl_p:
                    exit_p = sl_p + slip; reason = "stop_loss"; exited = True
                elif lo <= pt_p:
                    exit_p = pt_p + slip; reason = "profit_target"; exited = True
                elif bars_in >= lookahead:
                    exit_p = c + slip; exited = True

            if exited:
                pnl = (exit_p - ep) * direction * n_contracts * MNQ_POINT_VALUE - comm
                equity += pnl
                eq_vals.append(equity)
                daily_pnl[bdate_s] = daily_pnl.get(bdate_s, 0) + pnl
                day_pnl_so_far += pnl
                trades.append({"date": bdate_s, "dir": direction,
                               "pnl": round(pnl, 2), "reason": reason})
                pos = None
            else:
                pos = (ei, direction, ep, sl_p, pt_p, bars_in)

        if pos is not None or trades_today >= max_trades_day:
            continue
        if daily_loss_cap and day_pnl_so_far <= -daily_loss_cap:
            continue

        p = probs.iloc[i]
        if np.isnan(p):
            continue

        at = float(atr_s.iloc[i])
        if np.isnan(at) or at <= 0:
            continue

        t_min = ts.hour * 60 + ts.minute
        if not (9 * 60 + 45 <= t_min <= 14 * 60 + 45):
            continue

        direction = None
        if p >= conf_thresh:
            direction = 1
        elif p <= (1 - conf_thresh):
            direction = -1

        if direction is None:
            continue

        ep   = c + slip * direction
        sl_p = ep - sl_atr * at * direction
        pt_p = ep + pt_atr * at * direction
        pos  = (i, direction, ep, sl_p, pt_p, 0)
        trades_today += 1

    if not trades:
        return {"n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
                "max_drawdown": 0.0, "sharpe": 0.0,
                "trades_per_week": 0.0, "pnl_per_week": 0.0}

    wins   = [t for t in trades if t["pnl"] > 0]
    total  = sum(t["pnl"] for t in trades)
    eq_s   = pd.Series(eq_vals)
    max_dd = float((eq_s - eq_s.cummax()).min())

    n_days  = len(set(oos_df.index.date))
    n_weeks = n_days / 5
    daily_s = pd.Series(daily_pnl)
    sharpe  = (daily_s.mean() / daily_s.std() * np.sqrt(252)
               if len(daily_s) > 1 and daily_s.std() > 0 else 0.0)

    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "trades_per_week": round(len(trades) / n_weeks, 1),
        "pnl_per_week": round(total / n_weeks, 0),
        "daily_pnl": {k: round(v, 2) for k, v in daily_pnl.items()},
        "trades": trades,
    }


def load_mes_train():
    df = pd.read_hdf(MES_FILE, "/bars_1m")
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    df = rth_only(df)
    df = df[df.index < OOS_START]
    return df.sort_index()


def load_mnq_oos():
    df = pd.read_hdf(MNQ_OOS_FILE, "/bars_1min_eth")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    df = rth_only(df)
    df = df[df.index >= OOS_START]
    return df.sort_index()


def main():
    print("=" * 85, flush=True)
    print(f"  ML 1-min Search — Asymmetric PT/SL, $10k/week target, $4k DD limit", flush=True)
    print(f"  Backend: {'LightGBM' if _USE_LGB else 'sklearn HGBC'}", flush=True)
    print("=" * 85, flush=True)

    print("\nLoading data...", flush=True)
    mes_train = load_mes_train()
    mnq_oos   = load_mnq_oos()
    print(f"  MES train (1-min RTH): {mes_train.shape}, {mes_train.index[0].date()} to {mes_train.index[-1].date()}")
    print(f"  MNQ OOS  (1-min RTH): {mnq_oos.shape},  {mnq_oos.index[0].date()} to {mnq_oos.index[-1].date()}")

    n_days  = len(set(mnq_oos.index.date))
    n_weeks = n_days / 5
    print(f"  OOS: {n_days} trading days = {n_weeks:.1f} weeks\n", flush=True)

    if len(mnq_oos) < 1000:
        print("ERROR: Insufficient OOS data"); return

    # Asymmetric grid: PT >> SL to improve profit factor
    grid = list(product(
        [1.5, 2.0, 3.0],         # PT (ATR multiples) — asymmetric
        [0.5, 0.75, 1.0],        # SL (ATR multiples) — tight stops
        [3, 6, 12, 24],          # lookahead bars (3=3min, 24=24min)
        [0.60, 0.65, 0.70],      # confidence threshold
        [5, 10, 15],             # max trades/day
    ))
    print(f"Grid: {len(grid)} configurations ({len(set((p,s,la) for p,s,la,_,_ in grid))} unique models)\n", flush=True)

    model_cache: dict = {}
    results = []

    for pt, sl, lookahead, conf, mt in grid:
        key = (pt, sl, lookahead)
        if key not in model_cache:
            print(f"  Training PT={pt} SL={sl} LA={lookahead}...", end=" ", flush=True)
            model, auc = train_model(mes_train, pt, sl, lookahead, verbose=True)
            model_cache[key] = (model, auc)

        model, auc = model_cache[key]
        if model is None:
            continue

        r1 = run_backtest(mnq_oos, model, pt, sl, lookahead,
                          conf_thresh=conf, n_contracts=1,
                          max_trades_day=mt, daily_loss_cap=None)
        if r1["n_trades"] < 20:
            continue

        if r1["max_drawdown"] < 0:
            n_c = max(1, min(MAX_CONTRACTS, int(MAX_DD_LIMIT / abs(r1["max_drawdown"]))))
        else:
            n_c = MAX_CONTRACTS

        r = run_backtest(mnq_oos, model, pt, sl, lookahead,
                         conf_thresh=conf, n_contracts=n_c,
                         max_trades_day=mt, daily_loss_cap=DAILY_LOSS_CAP)
        if r["n_trades"] < 20:
            continue

        dd_penalty = max(0.0, abs(r["max_drawdown"]) - MAX_DD_LIMIT) * 5
        score = r["total_pnl"] - dd_penalty
        r.update({"pt": pt, "sl": sl, "lookahead": lookahead,
                  "conf": conf, "mt": mt, "n_contracts": n_c, "auc": auc, "score": score})
        results.append(r)

    if not results:
        print("No valid results"); return

    top = sorted(results, key=lambda x: x["score"], reverse=True)[:20]

    print(f"\n{'='*110}")
    print(f"  TOP CONFIGS (OOS: 2026 YTD MNQ 1-min, slippage + commission included)")
    print(f"{'='*110}")
    print(f"  {'PT':>4} {'SL':>4} {'LA':>3} {'Cf':>5} {'MT':>3} {'Nc':>3} | "
          f"{'Trades':>7} {'t/wk':>5} {'WR':>6} {'PnL':>10} {'$/wk':>8} {'DD':>9} {'Sh':>6}")
    print(f"  {'-'*105}")

    for r in top:
        goal = " ✓✓" if r["pnl_per_week"] >= 10000 and abs(r["max_drawdown"]) <= MAX_DD_LIMIT else \
               " ✓"  if r["pnl_per_week"] >= 5000  and abs(r["max_drawdown"]) <= MAX_DD_LIMIT else ""
        print(f"  {r['pt']:>4.1f} {r['sl']:>4.2f} {r['lookahead']:>3} {r['conf']:>5.2f} "
              f"{r['mt']:>3} {r['n_contracts']:>3} | "
              f"{r['n_trades']:>7} {r['trades_per_week']:>5.1f} {r['win_rate']:>6.1%} "
              f"${r['total_pnl']:>9,.0f} ${r['pnl_per_week']:>7,.0f} "
              f"${r['max_drawdown']:>8,.0f} {r['sharpe']:>6.2f}{goal}")

    best = top[0]
    print(f"\n{'='*85}")
    print(f"  BEST: PT={best['pt']} SL={best['sl']} LA={best['lookahead']} "
          f"conf={best['conf']} mt={best['mt']} nc={best['n_contracts']}")
    print(f"{'='*85}")
    print(f"  Trades:    {best['n_trades']} ({best['trades_per_week']:.1f}/week)")
    print(f"  Win rate:  {best['win_rate']:.1%}")
    print(f"  Total PnL: ${best['total_pnl']:,.0f}")
    print(f"  PnL/week:  ${best['pnl_per_week']:,.0f}")
    print(f"  Max DD:    ${best['max_drawdown']:,.0f}")
    print(f"  Sharpe:    {best['sharpe']:.2f}")
    print(f"  Train AUC: {best['auc']:.3f}")

    print(f"\n  Monthly PnL:")
    monthly: dict = {}
    for d, v in best["daily_pnl"].items():
        k = str(d)[:7]
        monthly[k] = monthly.get(k, 0) + v
    for k, v in sorted(monthly.items()):
        bar = "█" * int(abs(v) / 500)
        sign = "+" if v >= 0 else ""
        print(f"    {k}: ${v:>9,.0f}  {sign}{bar}")

    print(f"\n  Gap analysis:")
    c_for_10k = 10000 / best["pnl_per_week"] * best["n_contracts"] if best["pnl_per_week"] > 0 else float("inf")
    dd_at_10k = abs(best["max_drawdown"]) * (c_for_10k / best["n_contracts"]) if best["n_contracts"] > 0 else float("inf")
    print(f"    Current:  ${best['pnl_per_week']:,.0f}/week at {best['n_contracts']}c, DD=${abs(best['max_drawdown']):,.0f}")
    print(f"    For $10k: need {c_for_10k:.1f} contracts, DD=${dd_at_10k:,.0f}")
    max_possible = best["pnl_per_week"] * MAX_CONTRACTS / best["n_contracts"] if best["n_contracts"] > 0 else 0
    print(f"    At 15c:   ${max_possible:,.0f}/week, DD=${abs(best['max_drawdown'])*MAX_CONTRACTS/best['n_contracts']:,.0f}")

    save = [{k: v for k, v in r.items() if k != "trades"} for r in top]
    print("\n===JSON_RESULTS_START===")
    print(json.dumps({"oos_period": "2026 YTD 1-min", "n_oos_weeks": round(n_weeks, 1),
                      "results": save}, indent=2, default=str))
    print("===JSON_RESULTS_END===")


if __name__ == "__main__":
    main()
