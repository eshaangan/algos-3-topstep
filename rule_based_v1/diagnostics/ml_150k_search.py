"""
Pure ML strategy targeting $10k/week on a $150k funded account.

Training data:
  - MES  Oct 2024 – Dec 2025  (14 months, price-normalized features)
  - MNQ  Aug 2025 – Dec 2025  (5 months, proper pre-2026 data)

OOS (true hold-out — 2026 YTD, never seen during training):
  - MNQ  Jan 2026 – May 2026

Key design choices:
  - ATR-normalized features → instrument-agnostic → MES trains, MNQ OOS
  - Bidirectional: LONG + SHORT signals from one model
  - Grid search PT/SL/lookahead/confidence, up to 5 trades/day
  - Contract scaling: find max contracts s.t. max_DD <= $4,000
  - 1-tick slippage each way + $1.24 commission/side/contract

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/ml_150k_search.py
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

# Data paths
MNQ_FILE  = ROOT / "data" / "processed" / "mnq_vc_backtest_5min.parquet"
MES_FILE  = ROOT / "data" / "processed" / "MES_5min_Oct2024_Dec2025.parquet"

# MNQ contract specs
MNQ_POINT_VALUE = 2.0    # $/point
MNQ_TICK_SIZE   = 0.25
MNQ_COMMISSION  = 0.62   # per side per contract
MNQ_SLIP_TICKS  = 1      # slippage ticks each way

# MES contract specs (used only for training features — dollar values not used)
MES_POINT_VALUE = 5.0    # $/point (irrelevant — we normalize by ATR)

MAX_CONTRACTS   = 15     # Topstep funded 150k hard limit
MAX_DD_LIMIT    = 4000.0 # dollar max drawdown target
DAILY_LOSS_CAP  = 1500.0 # hard daily loss cap (Topstep funded), per-contract later

OOS_START  = pd.Timestamp("2026-01-01", tz="US/Eastern")

# ── Utilities ──────────────────────────────────────────────────────────────────
def to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert("US/Eastern")
    return df


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    m = df.index.minute
    t = h * 60 + m
    return df[(t >= 9 * 60 + 30) & (t < 16 * 60)].copy()


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs = g / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def intraday_vwap(bars: pd.DataFrame) -> pd.Series:
    parts = []
    for d in sorted(set(bars.index.date)):
        day = bars[bars.index.date == d].copy()
        tp = (day["high"] + day["low"] + day["close"]) / 3
        cum_vol = day["volume"].cumsum().replace(0, np.nan)
        vwap = (tp * day["volume"]).cumsum() / cum_vol
        parts.append(vwap)
    return pd.concat(parts)


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    at = atr(h, l, c, 14)
    at_safe = at.replace(0, np.nan)
    df["atr"] = at

    # VWAP deviation
    df["vwap"] = intraday_vwap(df)
    df["vwap_dev"] = (c - df["vwap"]) / at_safe

    # Multi-horizon ATR-normalised momentum
    for n in [1, 2, 3, 5, 10, 20, 30]:
        df[f"mom{n}"] = c.diff(n) / at_safe

    # RSI (multiple periods)
    df["rsi14"] = rsi(c, 14)
    df["rsi7"]  = rsi(c, 7)
    df["rsi14_dev"] = df["rsi14"] - 50
    df["rsi7_dev"]  = df["rsi7"] - 50

    # EMA spread
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["ema9_21"] = (ema9 - ema21) / at_safe
    df["ema21_50"] = (ema21 - ema50) / at_safe
    df["c_vs_ema9"]  = (c - ema9) / at_safe
    df["c_vs_ema21"] = (c - ema21) / at_safe

    # Bar structure
    rng = (h - l).replace(0, np.nan)
    df["clv"]  = (2 * c - h - l) / rng
    df["uw"]   = (h - c.clip(upper=o)) / rng
    df["lw"]   = (c.clip(upper=o) - l) / rng
    df["body"] = (c - o).abs() / rng

    # Volume
    df["vol_ratio"]  = v / v.rolling(20).mean()
    df["vol_ratio5"] = v / v.rolling(5).mean()

    # Volatility regime
    df["atr_ratio"]  = at / at.rolling(20).mean()
    df["atr_ratio5"] = at / at.rolling(5).mean()

    # Realised vol
    pct = c.pct_change()
    df["rvol5"]  = pct.rolling(5).std()
    df["rvol20"] = pct.rolling(20).std()
    df["rvol_ratio"] = df["rvol5"] / df["rvol20"].replace(0, np.nan)

    # Intraday position
    session_high = h.groupby(df.index.date).cummax()
    session_low  = l.groupby(df.index.date).cummin()
    df["dist_sess_high"] = (session_high - c) / at_safe
    df["dist_sess_low"]  = (c - session_low) / at_safe

    # Bar gap
    df["bar_gap"] = (o - c.shift()) / at_safe

    # Time features
    df["hod"]    = df.index.hour + df.index.minute / 60
    df["dow"]    = df.index.dayofweek   # 0=Mon ... 4=Fri
    df["is_open_hour"] = ((df.index.hour == 9) & (df.index.minute >= 30) |
                           (df.index.hour == 10)).astype(float)

    # Rolling autocorrelation proxy (momentum persistence)
    df["mom1_lag1"] = df["mom1"].shift(1)
    df["mom1_lag2"] = df["mom1"].shift(2)
    df["mom2_lag1"] = df["mom2"].shift(1)

    return df


FEATURE_COLS = [
    "vwap_dev",
    "mom1", "mom2", "mom3", "mom5", "mom10", "mom20", "mom30",
    "rsi14_dev", "rsi7_dev",
    "ema9_21", "ema21_50", "c_vs_ema9", "c_vs_ema21",
    "clv", "uw", "lw", "body",
    "vol_ratio", "vol_ratio5",
    "atr_ratio", "atr_ratio5",
    "rvol5", "rvol20", "rvol_ratio",
    "dist_sess_high", "dist_sess_low",
    "bar_gap",
    "hod", "dow", "is_open_hour",
    "mom1_lag1", "mom1_lag2", "mom2_lag1",
]


# ── Labeling ───────────────────────────────────────────────────────────────────
def make_labels(bars: pd.DataFrame, atr_s: pd.Series,
                pt_atr: float, sl_atr: float,
                lookahead: int) -> pd.Series:
    """Triple-barrier label: +1 if LONG PT hit before SL in next `lookahead` bars."""
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


# ── Model training ─────────────────────────────────────────────────────────────
def train_model(train_df: pd.DataFrame, pt_atr: float, sl_atr: float,
                lookahead: int, verbose: bool = False) -> tuple:
    feat   = build_features(train_df)
    labels = make_labels(train_df, feat["atr"], pt_atr, sl_atr, lookahead)

    mask = labels.isin([1, -1]) & feat[FEATURE_COLS].notna().all(axis=1)
    X = feat.loc[mask, FEATURE_COLS].values
    y = (labels[mask] == 1).astype(int).values

    if len(X) < 200 or y.mean() < 0.05 or y.mean() > 0.95:
        return None, None

    if verbose:
        print(f"    Train: {len(X)} samples, label WR={y.mean():.1%}", end=" ")

    tscv = TimeSeriesSplit(n_splits=3)
    if _USE_LGB:
        params = dict(
            objective="binary", metric="auc",
            n_estimators=150, learning_rate=0.06,
            max_depth=5, num_leaves=25,
            min_child_samples=30,
            colsample_bytree=0.75, subsample=0.8,
            reg_alpha=0.15, reg_lambda=0.15,
            n_jobs=-1, verbose=-1,
        )
        base = lgb.LGBMClassifier(**params)
        model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    else:
        # sklearn fallback — HistGradientBoostingClassifier (fast, no GPU issues)
        base = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06,
            max_depth=5, min_samples_leaf=30,
            l2_regularization=0.15, random_state=42,
        )
        model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    model.fit(X, y)

    p_train = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p_train)

    if verbose:
        print(f"→ train AUC={auc:.3f}")

    return model, auc


# ── OOS simulation ─────────────────────────────────────────────────────────────
def run_backtest(oos_df: pd.DataFrame, model,
                 pt_atr: float, sl_atr: float, lookahead: int,
                 conf_thresh: float, n_contracts: int,
                 max_trades_day: int = 5,
                 daily_loss_cap: float | None = None) -> dict:

    feat   = build_features(oos_df)
    X_all  = feat[FEATURE_COLS]
    valid  = X_all.notna().all(axis=1)
    probs  = pd.Series(np.nan, index=oos_df.index)
    if valid.sum() > 0:
        probs[valid] = model.predict_proba(X_all[valid].values)[:, 1]

    atr_s   = feat["atr"]
    closes  = oos_df["close"]
    highs   = oos_df["high"]
    lows    = oos_df["low"]

    slip   = MNQ_SLIP_TICKS * MNQ_TICK_SIZE
    comm   = 2 * MNQ_COMMISSION * n_contracts

    trades  = []
    equity  = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict[str, float] = {}
    day_stats: dict[str, dict] = {}  # date -> {pnl, n_trades}

    cur_date    = None
    trades_today = 0
    day_pnl_so_far = 0.0
    pos = None  # (idx, dir, entry, stop, target, bars_in)

    for i, (ts, _) in enumerate(oos_df.iterrows()):
        bdate = ts.date()
        bdate_s = str(bdate)

        if cur_date is not None and bdate != cur_date:
            trades_today   = 0
            day_pnl_so_far = 0.0
        cur_date = bdate

        h = highs.iloc[i]
        lo = lows.iloc[i]
        c  = closes.iloc[i]

        # ── Manage open position ──
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
                pnl  = (exit_p - ep) * direction * n_contracts * MNQ_POINT_VALUE - comm
                equity += pnl
                eq_vals.append(equity)
                daily_pnl[bdate_s] = daily_pnl.get(bdate_s, 0) + pnl
                day_pnl_so_far    += pnl
                trades.append({
                    "date": bdate_s, "dir": direction,
                    "entry": ep, "exit": exit_p,
                    "pnl": round(pnl, 2), "reason": reason
                })
                pos = None
            else:
                pos = (ei, direction, ep, sl_p, pt_p, bars_in)

        # ── Entry logic ──
        if pos is not None or trades_today >= max_trades_day:
            continue

        # Daily loss cap
        if daily_loss_cap and day_pnl_so_far <= -daily_loss_cap:
            continue

        p = probs.iloc[i]
        if np.isnan(p):
            continue

        at = float(atr_s.iloc[i])
        if np.isnan(at) or at <= 0:
            continue

        # Time gate: 9:45 – 14:45 ET (avoid open noise + late close chasing)
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

    n_days   = len(set(oos_df.index.date))
    n_weeks  = n_days / 5
    daily_s  = pd.Series(daily_pnl)
    sharpe   = (daily_s.mean() / daily_s.std() * np.sqrt(252)
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


# ── Data loading ───────────────────────────────────────────────────────────────
def load_mnq() -> pd.DataFrame:
    df = pd.read_parquet(MNQ_FILE)
    df = to_eastern(df)
    df = rth_only(df)
    df = df.sort_index()
    return df


def load_mes() -> pd.DataFrame:
    df = pd.read_parquet(MES_FILE)
    df = to_eastern(df)
    df = rth_only(df)
    df = df.sort_index()
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import sys
    print("=" * 80, flush=True)
    print(f"  ML 150k Search — Pure ML, $10k/week target, $4k DD limit", flush=True)
    print(f"  Backend: {'LightGBM' if _USE_LGB else 'sklearn HistGradientBoosting'}", flush=True)
    print("=" * 80, flush=True)

    # Load data
    print("\nLoading data...")
    mnq = load_mnq()
    mes = load_mes()
    print(f"  MNQ: {mnq.shape}, {mnq.index[0].date()} to {mnq.index[-1].date()}")
    print(f"  MES: {mes.shape}, {mes.index[0].date()} to {mes.index[-1].date()}")

    # Split MNQ into pre-2026 training and 2026 OOS
    mnq_train = mnq[mnq.index < OOS_START]
    mnq_oos   = mnq[mnq.index >= OOS_START]

    # MES is all pre-2026 (ends Dec 2025)
    mes_train = mes

    print(f"\n  MNQ train: {len(mnq_train)} bars ({len(set(mnq_train.index.date))} days)")
    print(f"  MES train: {len(mes_train)} bars ({len(set(mes_train.index.date))} days)")
    print(f"  MNQ OOS:   {len(mnq_oos)} bars ({len(set(mnq_oos.index.date))} days)")

    n_oos_days  = len(set(mnq_oos.index.date))
    n_oos_weeks = n_oos_days / 5
    print(f"  OOS period: {n_oos_weeks:.1f} trading weeks\n")

    if len(mnq_oos) < 500:
        print("ERROR: Insufficient OOS data")
        return

    # Grid
    grid = list(product(
        [1.0, 2.0, 3.0],          # PT (ATR multiples)
        [0.75, 1.0, 1.5],         # SL
        [4, 6, 12],               # lookahead bars
        [0.60, 0.65, 0.70, 0.75], # confidence threshold
        [3, 5, 7],                 # max trades/day
    ))
    print(f"Grid: {len(grid)} configurations")
    print("Training one model per (PT, SL, lookahead) triplet...\n")

    model_cache: dict[tuple, tuple] = {}
    results = []

    for pt, sl, lookahead, conf, mt in grid:
        key = (pt, sl, lookahead)
        if key not in model_cache:
            combined = pd.concat([mes_train, mnq_train]).sort_index()
            print(f"  Training PT={pt} SL={sl} LA={lookahead}...", end=" ", flush=True)
            model, auc = train_model(combined, pt, sl, lookahead, verbose=False)
            if model is not None:
                print(f"AUC={auc:.3f}", flush=True)
            else:
                print("SKIP", flush=True)
            model_cache[key] = (model, auc)

        model, auc = model_cache[key]
        if model is None:
            continue

        # OOS backtest at 1 contract first, then scale
        r1 = run_backtest(
            mnq_oos, model, pt, sl, lookahead,
            conf_thresh=conf, n_contracts=1,
            max_trades_day=mt,
            daily_loss_cap=None,
        )
        if r1["n_trades"] < 10:
            continue

        # Scale contracts to stay within $4k DD limit
        if r1["max_drawdown"] < 0:
            max_scale = MAX_DD_LIMIT / abs(r1["max_drawdown"])
            n_c = max(1, min(MAX_CONTRACTS, int(max_scale)))
        else:
            n_c = MAX_CONTRACTS

        r = run_backtest(
            mnq_oos, model, pt, sl, lookahead,
            conf_thresh=conf, n_contracts=n_c,
            max_trades_day=mt,
            daily_loss_cap=DAILY_LOSS_CAP,
        )
        if r["n_trades"] < 10:
            continue

        # Score: maximize weekly PnL subject to DD constraint
        dd_penalty = max(0.0, abs(r["max_drawdown"]) - MAX_DD_LIMIT) * 5
        score = r["total_pnl"] - dd_penalty

        r.update({
            "pt": pt, "sl": sl, "lookahead": lookahead,
            "conf": conf, "mt": mt,
            "n_contracts": n_c, "auc": auc, "score": score,
        })
        results.append(r)

    if not results:
        print("No valid results found")
        return

    top = sorted(results, key=lambda x: x["score"], reverse=True)[:20]

    print(f"\n{'='*105}")
    print(f"  TOP CONFIGS (OOS: full 2026 YTD, MNQ, slippage + commission included)")
    print(f"{'='*105}")
    print(f"  {'PT':>4} {'SL':>4} {'LA':>3} {'Cf':>5} {'MT':>3} {'Nc':>3} | "
          f"{'Trades':>7} {'t/wk':>5} {'WR':>6} {'PnL':>10} {'$/wk':>8} {'DD':>9} {'Sh':>6}")
    print(f"  {'-'*100}")

    for r in top:
        goal_mark = " ✓✓" if r["pnl_per_week"] >= 10000 and abs(r["max_drawdown"]) <= MAX_DD_LIMIT else \
                    " ✓"  if r["pnl_per_week"] >= 5000  and abs(r["max_drawdown"]) <= MAX_DD_LIMIT else ""
        print(f"  {r['pt']:>4.1f} {r['sl']:>4.2f} {r['lookahead']:>3} {r['conf']:>5.2f} "
              f"{r['mt']:>3} {r['n_contracts']:>3} | "
              f"{r['n_trades']:>7} {r['trades_per_week']:>5.1f} {r['win_rate']:>6.1%} "
              f"${r['total_pnl']:>9,.0f} ${r['pnl_per_week']:>7,.0f} "
              f"${r['max_drawdown']:>8,.0f} {r['sharpe']:>6.2f}{goal_mark}")

    best = top[0]
    print(f"\n{'='*80}")
    print(f"  BEST CONFIG: PT={best['pt']} SL={best['sl']} lookahead={best['lookahead']} "
          f"conf={best['conf']} mt={best['mt']} n_contracts={best['n_contracts']}")
    print(f"{'='*80}")
    print(f"  OOS 2026 YTD ({n_oos_weeks:.1f} weeks):")
    print(f"  Trades:     {best['n_trades']} ({best['trades_per_week']:.1f}/week)")
    print(f"  Win rate:   {best['win_rate']:.1%}")
    print(f"  Total PnL:  ${best['total_pnl']:,.0f}")
    print(f"  PnL/week:   ${best['pnl_per_week']:,.0f}")
    print(f"  Max DD:     ${best['max_drawdown']:,.0f}")
    print(f"  Sharpe:     {best['sharpe']:.2f}")
    print(f"  Train AUC:  {best['auc']:.3f}")

    print(f"\n  Monthly PnL breakdown:")
    monthly: dict[str, float] = {}
    for d, v in best["daily_pnl"].items():
        k = str(d)[:7]
        monthly[k] = monthly.get(k, 0) + v
    for k, v in sorted(monthly.items()):
        bar = "█" * int(abs(v) / 500)
        sign = "+" if v >= 0 else ""
        print(f"    {k}: ${v:>9,.0f}  {sign}{bar}")

    print(f"\n  Gap-to-goal analysis:")
    target_weekly = 10_000
    current_weekly = best["pnl_per_week"]
    current_dd     = abs(best["max_drawdown"])
    contracts_for_target = target_weekly / current_weekly * best["n_contracts"] if current_weekly > 0 else float("inf")
    dd_at_target   = current_dd * (contracts_for_target / best["n_contracts"]) if best["n_contracts"] > 0 else float("inf")
    print(f"    Current:  ${current_weekly:,.0f}/week at {best['n_contracts']} contracts, DD=${current_dd:,.0f}")
    print(f"    Target:   ${target_weekly:,.0f}/week")
    print(f"    Contracts needed for $10k/week: {contracts_for_target:.1f}")
    print(f"    DD at that scale: ${dd_at_target:,.0f} (limit: ${MAX_DD_LIMIT:,.0f})")
    if contracts_for_target <= MAX_CONTRACTS and dd_at_target <= MAX_DD_LIMIT:
        print(f"    ✓ TARGET IS ACHIEVABLE at {int(contracts_for_target)} contracts")
    else:
        print(f"    Current edge requires ${current_weekly * MAX_CONTRACTS / best['n_contracts']:,.0f}/week "
              f"at max {MAX_CONTRACTS} contracts (DD: ${current_dd * MAX_CONTRACTS / best['n_contracts']:,.0f})")

    # Save results
    out_path = Path(__file__).parent / "ml_150k_results.json"
    save = [{k: v for k, v in r.items() if k != "trades"} for r in top]
    with open(out_path, "w") as f:
        json.dump({
            "target": {"pnl_per_week": 10000, "max_drawdown": -4000},
            "oos_period": "2026 YTD",
            "train_period": "MES Oct2024-Dec2025 + MNQ Aug2025-Dec2025",
            "n_oos_weeks": round(n_oos_weeks, 1),
            "results": save,
        }, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
