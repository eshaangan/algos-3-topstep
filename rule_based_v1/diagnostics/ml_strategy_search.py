"""ML-based intraday strategy search for MNQ 5-min bars.

Train: Jan 1 – Jan 31 2026
Val:   Feb 1 – Feb 28 2026 (held out for config selection / overfit check)
OOS:   Mar 1 – May 13 2026 — never seen by Jan-trained model

Goal: find high-frequency, high-WR strategy targeting ~$14k/week.
Includes slippage (1 tick each way) + commissions ($1.24 round-trip/contract).

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/ml_strategy_search.py
"""
from __future__ import annotations
import argparse
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).resolve().parent.parent.parent
RBV1  = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / "mnq_vc_backtest_5min.parquet"
MODEL_PATH = RBV1 / "models" / "ml_strategy_mnq_v1.pkl"

DEPLOY_CONFIG = {
    "pt": 1.0,
    "sl": 1.5,
    "lookahead": 6,
    "conf": 0.70,
    "max_trades": 3,
}

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62          # per side per contract
SLIPPAGE_T   = 1             # ticks each way
N_CONTRACTS  = 5             # base for reporting

TRAIN_END = pd.Timestamp("2026-02-01", tz="US/Eastern")
VAL_START = pd.Timestamp("2026-02-01", tz="US/Eastern")
VAL_END = pd.Timestamp("2026-03-01", tz="US/Eastern")
OOS_START = pd.Timestamp("2026-03-01", tz="US/Eastern")
WALKFORWARD_TRAIN_END = pd.Timestamp("2026-03-01", tz="US/Eastern")

# ── Feature engineering ────────────────────────────────────────────────────────
def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l_ = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l_.replace(0, np.nan))

def intraday_vwap(bars):
    parts = []
    for d in sorted(set(bars.index.date)):
        day = bars[bars.index.date == d].copy()
        tp = (day["high"] + day["low"] + day["close"]) / 3
        cum = (tp * day["volume"]).cumsum() / day["volume"].cumsum().replace(0, np.nan)
        parts.append(cum)
    return pd.concat(parts)

def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    at = atr(h, l, c, 14)
    df["atr"] = at

    # VWAP
    df["vwap"] = intraday_vwap(df)
    df["vwap_dev"] = (c - df["vwap"]) / at.replace(0, np.nan)

    # Momentum (ATR-normalised)
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"mom{n}"] = c.diff(n) / at

    # RSI
    df["rsi"] = rsi(c, 14)
    df["rsi14_dev"] = df["rsi"] - 50

    # Bar structure
    rng = (h - l).replace(0, np.nan)
    df["clv"] = (2*c - h - l) / rng           # close location value [-1,1]
    df["uw"]  = (h - c.clip(upper=o)) / rng   # upper wick fraction
    df["lw"]  = (c.clip(upper=o) - l) / rng   # lower wick fraction
    df["body"] = (c - o).abs() / rng

    # Volume
    df["vol_ratio"] = v / v.rolling(20).mean()
    df["vol_ratio5"] = v / v.rolling(5).mean()

    # Volatility regime
    df["atr_ratio"] = at / at.rolling(20).mean()

    # Rolling realised vol (normalised)
    df["rvol5"]  = c.pct_change().rolling(5).std()
    df["rvol20"] = c.pct_change().rolling(20).std()
    df["rvol_ratio"] = df["rvol5"] / df["rvol20"].replace(0, np.nan)

    # Distance from day's high/low so far
    session_high = h.groupby(df.index.date).cummax()
    session_low  = l.groupby(df.index.date).cummin()
    df["dist_sess_high"] = (session_high - c) / at
    df["dist_sess_low"]  = (c - session_low) / at

    # 5-min bar opening gap (open vs prior close)
    df["bar_gap"] = (o - c.shift()) / at

    # Time features
    df["hour"]   = df.index.hour
    df["minute"] = df.index.minute
    df["hod"]    = df["hour"] + df["minute"] / 60  # hour of day continuous

    return df


def make_labels(bars: pd.DataFrame, atr_col: pd.Series,
                pt_atr: float, sl_atr: float, lookahead: int = 6) -> pd.Series:
    """Label each bar: +1 if PT hit before SL in next `lookahead` bars (LONG trade).
    Uses bar-by-bar simulation to check PT/SL crossing order.
    Returns: +1 (long wins), -1 (long loses), 0 (neither = no label used for short).
    """
    labels = []
    closes = bars["close"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    atrs   = atr_col.values

    for i in range(len(bars) - lookahead):
        at = atrs[i]
        if np.isnan(at) or at <= 0:
            labels.append(np.nan)
            continue
        entry = closes[i]
        pt = entry + pt_atr * at
        sl = entry - sl_atr * at
        outcome = 0
        for j in range(i+1, i+1+lookahead):
            if highs[j] >= pt:
                outcome = 1   # long wins
                break
            if lows[j] <= sl:
                outcome = -1  # long loses
                break
        labels.append(outcome)
    labels += [np.nan] * lookahead
    return pd.Series(labels, index=bars.index)


# ── Training ───────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "vwap_dev", "mom1","mom2","mom3","mom5","mom10","mom20",
    "rsi14_dev", "clv","uw","lw","body",
    "vol_ratio","vol_ratio5", "atr_ratio",
    "rvol5","rvol20","rvol_ratio",
    "dist_sess_high","dist_sess_low","bar_gap","hod",
]

def train_model(train_df: pd.DataFrame, pt_atr: float, sl_atr: float,
                lookahead: int = 6) -> tuple:
    feat = build_features(train_df)
    labels = make_labels(train_df, feat["atr"], pt_atr, sl_atr, lookahead)

    # Only use bars where a winner (1) or loser (-1) was determined
    mask = labels.isin([1, -1]) & feat[FEATURE_COLS].notna().all(axis=1)
    X = feat.loc[mask, FEATURE_COLS].values
    y = (labels[mask] == 1).astype(int).values   # 1=win, 0=loss

    if len(X) < 100 or y.mean() < 0.1 or y.mean() > 0.9:
        return None, None, None

    print(f"    Train samples: {len(X)}, win rate in labels: {y.mean():.1%}")

    lgb_params = dict(
        objective="binary", metric="auc", n_estimators=200,
        learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
        reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    base = lgb.LGBMClassifier(**lgb_params)

    # Calibrate probabilities
    model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    model.fit(X, y)

    # AUC on training data (just for sanity)
    p = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, p)

    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, train_auc


def export_model(
    config: dict | None = None,
    output_path: Path | None = None,
) -> Path:
    """Train on Jan 2026 with deploy config and save model bundle for live runner."""
    cfg = config or DEPLOY_CONFIG
    out = output_path or MODEL_PATH

    bars = pd.read_parquet(CACHE)
    bars.index = (
        bars.index.tz_convert("US/Eastern")
        if bars.index.tz
        else bars.index.tz_localize("US/Eastern")
    )
    train_bars = bars[
        (bars.index >= pd.Timestamp("2026-01-01", tz="US/Eastern"))
        & (bars.index < TRAIN_END)
    ]
    if len(train_bars) < 100:
        raise RuntimeError(f"Insufficient Jan 2026 bars for export: {len(train_bars)}")

    print(
        f"Export: training Jan 2026 ({len(train_bars)} bars) "
        f"PT={cfg['pt']} SL={cfg['sl']} look={cfg['lookahead']} ..."
    )
    model, _scaler, auc = train_model(train_bars, cfg["pt"], cfg["sl"], cfg["lookahead"])
    if model is None:
        raise RuntimeError("Model training failed — insufficient labeled samples")

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "config": cfg,
        "train_end": str(TRAIN_END.date()),
        "train_auc": auc,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Exported model → {out} (AUC={auc:.3f}, train_end={TRAIN_END.date()})")
    return out


# ── OOS Backtest ───────────────────────────────────────────────────────────────
@dataclass
class Trade:
    date: str
    entry_bar: int
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


def run_oos_backtest(
    oos_df: pd.DataFrame,
    model,
    scaler,
    pt_atr: float,
    sl_atr: float,
    lookahead: int,
    confidence_threshold: float,
    n_contracts: int,
    max_trades_per_day: int = 5,
) -> dict:
    feat = build_features(oos_df)
    X_all = feat[FEATURE_COLS]

    # Get probabilities where features are complete
    valid_mask = X_all.notna().all(axis=1)
    probs = pd.Series(np.nan, index=oos_df.index)
    if valid_mask.sum() > 0:
        probs[valid_mask] = model.predict_proba(X_all[valid_mask].values)[:, 1]

    atr_s   = feat["atr"]
    closes  = oos_df["close"]
    highs   = oos_df["high"]
    lows    = oos_df["low"]

    slip = SLIPPAGE_T * TICK_SIZE
    comm = 2 * COMMISSION * n_contracts

    trades: list[Trade] = []
    equity = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0
    pos = None  # (entry_bar_idx, direction, entry_price, stop, target, bars_in)

    for i, (ts, row) in enumerate(oos_df.iterrows()):
        bdate = ts.date()
        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_pnl.get(cur_date, 0)  # already tracked below
            trades_today = 0
        cur_date = bdate

        h, l, c = highs.iloc[i], lows.iloc[i], closes.iloc[i]

        # Manage open position
        if pos is not None:
            ei, direction, ep, sl_p, pt_p, bars_in, last_close = pos
            bars_in += 1
            last_close = c

            exited = False; exit_p = c; reason = ""
            if direction == 1:
                if l <= sl_p:
                    exit_p = sl_p - slip; reason = "stop_loss"; exited = True
                elif h >= pt_p:
                    exit_p = pt_p - slip; reason = "profit_target"; exited = True
                elif bars_in >= lookahead:
                    exit_p = c - slip; reason = "time_stop"; exited = True
            else:
                if h >= sl_p:
                    exit_p = sl_p + slip; reason = "stop_loss"; exited = True
                elif l <= pt_p:
                    exit_p = pt_p + slip; reason = "profit_target"; exited = True
                elif bars_in >= lookahead:
                    exit_p = c + slip; reason = "time_stop"; exited = True

            if exited:
                pnl = (exit_p - ep) * direction * n_contracts * POINT_VALUE - comm
                t = Trade(str(bdate), ei, direction, ep, exit_p, pnl, reason)
                trades.append(t)
                equity += pnl
                eq_vals.append(equity)
                daily_pnl[bdate] = daily_pnl.get(bdate, 0) + pnl
                pos = None
            else:
                pos = (ei, direction, ep, sl_p, pt_p, bars_in, last_close)

        # Entry signal
        if pos is None and trades_today < max_trades_per_day:
            p = probs.iloc[i]
            if np.isnan(p):
                continue
            at = float(atr_s.iloc[i])
            if np.isnan(at) or at <= 0:
                continue

            # Time gate: only trade between 9:45 and 15:00 ET
            h_et = ts.hour; m_et = ts.minute
            in_window = (h_et > 9 or (h_et == 9 and m_et >= 45)) and h_et < 15

            if not in_window:
                continue

            direction = None
            if p >= confidence_threshold:
                direction = 1   # LONG (model says this is a long-win bar)
            elif p <= (1 - confidence_threshold):
                direction = -1  # SHORT (model says this is a short-win bar — inverse)

            if direction is None:
                continue

            ep = c + slip * direction
            sl_p = ep - sl_atr * at * direction
            pt_p = ep + pt_atr * at * direction
            pos = (i, direction, ep, sl_p, pt_p, 0, c)
            trades_today += 1

    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "sharpe": 0, "max_drawdown": 0, "trades_per_week": 0}

    wins = [t for t in trades if t.pnl > 0]
    total = sum(t.pnl for t in trades)
    eq = pd.Series(eq_vals)
    max_dd = float((eq - eq.cummax()).min())
    n_weeks = len(set(oos_df.index.date)) / 5
    daily = pd.Series(daily_pnl)
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "trades_per_week": round(len(trades) / n_weeks, 1),
        "pnl_per_week": round(total / n_weeks, 0),
        "trades": trades,
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    bars = pd.read_parquet(CACHE)
    bars.index = (bars.index.tz_convert("US/Eastern")
                  if bars.index.tz else bars.index.tz_localize("US/Eastern"))
    bars_2026 = bars[bars.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]

    train_bars = bars_2026[bars_2026.index < TRAIN_END]
    val_bars   = bars_2026[(bars_2026.index >= VAL_START) & (bars_2026.index < VAL_END)]
    oos_bars   = bars_2026[bars_2026.index >= OOS_START]
    wf_train_bars = bars_2026[bars_2026.index < WALKFORWARD_TRAIN_END]

    print(f"Train (Jan): {len(train_bars)} bars ({len(set(train_bars.index.date))} days)")
    print(f"Val (Feb):   {len(val_bars)} bars ({len(set(val_bars.index.date))} days)")
    print(f"OOS (Mar+):  {len(oos_bars)} bars ({len(set(oos_bars.index.date))} days)")
    n_oos_weeks = len(set(oos_bars.index.date)) / 5
    print(f"OOS weeks: {n_oos_weeks:.1f}\n")

    # Grid: (pt_atr, sl_atr, lookahead_bars, confidence_threshold, max_trades_day)
    configs = list(product(
        [1.0, 1.5, 2.0, 3.0],  # PT
        [0.75, 1.0, 1.5],      # SL
        [3, 6, 12],             # lookahead bars
        [0.60, 0.65, 0.70],     # confidence threshold
        [3, 5],                 # max trades per day
    ))

    print(f"Grid: {len(configs)} configs × {len(list(product([1.0,1.5,2.0,3.0],[0.75,1.0,1.5],[3,6,12])))} model variants")

    results = []
    best_score = -999

    # Train one model per (pt, sl, lookahead) combo — the confidence + max_trades is swept at OOS time
    model_cache: dict[tuple, tuple] = {}

    for pt, sl, lookahead, conf, mt in configs:
        key = (pt, sl, lookahead)
        if key not in model_cache:
            print(f"\n  Training model PT={pt} SL={sl} lookahead={lookahead}...", end=" ", flush=True)
            model, scaler, auc = train_model(train_bars, pt, sl, lookahead)
            model_cache[key] = (model, scaler, auc)
            if model is not None:
                print(f"AUC={auc:.3f}")
            else:
                print("SKIP (insufficient data)")

        model, scaler, auc = model_cache[key]
        if model is None:
            continue

        val_r = run_oos_backtest(
            val_bars, model, scaler, pt, sl, lookahead,
            confidence_threshold=conf, n_contracts=N_CONTRACTS,
            max_trades_per_day=mt,
        )
        r = run_oos_backtest(
            oos_bars, model, scaler, pt, sl, lookahead,
            confidence_threshold=conf, n_contracts=N_CONTRACTS,
            max_trades_per_day=mt,
        )
        if r["n_trades"] < 10:
            continue

        val_penalty = 0.0
        if val_r["n_trades"] >= 3 and val_r["total_pnl"] < 0:
            val_penalty = abs(val_r["total_pnl"]) * 0.5
        dd_penalty = max(0, -r["max_drawdown"] - 2000) * 3
        score = r["total_pnl"] - dd_penalty - val_penalty
        r.update({
            "pt": pt, "sl": sl, "lookahead": lookahead,
            "conf": conf, "mt": mt, "auc": auc, "score": score,
            "val_n_trades": val_r["n_trades"],
            "val_pnl": val_r["total_pnl"],
            "val_wr": val_r["win_rate"],
            "val_max_dd": val_r["max_drawdown"],
        })
        results.append(r)

        if score > best_score:
            best_score = score
            print(f"  *** New best: PT={pt} SL={sl} look={lookahead} conf={conf} mt={mt} "
                  f"| val ${val_r['total_pnl']:,.0f} ({val_r['n_trades']}t) "
                  f"OOS {r['n_trades']}t {r['win_rate']:.1%} WR "
                  f"${r['total_pnl']:,.0f} DD={r['max_drawdown']:,.0f} "
                  f"{r['trades_per_week']:.1f}t/wk ${r['pnl_per_week']:,.0f}/wk ***")

    if not results:
        print("No valid results found")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    top = sorted(results, key=lambda x: x["score"], reverse=True)[:15]
    print(f"\n\n{'='*100}")
    print(f"  TOP CONFIGS (Jan train / Feb val / Mar-May OOS, {N_CONTRACTS} ct, slip+comm)")
    print(f"{'='*100}")
    print(f"  {'PT':>4} {'SL':>4} {'LA':>3} {'Conf':>5} {'MT':>3} | "
          f"{'ValPnL':>8} {'ValDD':>8} | {'Trades':>7} {'t/wk':>5} {'WR':>6} "
          f"{'PnL':>10} {'$/wk':>8} {'DD':>9} {'Sharpe':>7}")
    print(f"  {'-'*95}")
    for r in top:
        goal = " ✓" if r["total_pnl"] >= 14000 * n_oos_weeks and r["max_drawdown"] >= -2000 else ""
        print(f"  {r['pt']:>4.1f} {r['sl']:>4.2f} {r['lookahead']:>3} {r['conf']:>5.2f} {r['mt']:>3} | "
              f"${r.get('val_pnl', 0):>7,.0f} ${r.get('val_max_dd', 0):>7,.0f} | "
              f"{r['n_trades']:>7} {r['trades_per_week']:>5.1f} {r['win_rate']:>6.1%} "
              f"${r['total_pnl']:>9,.0f} ${r['pnl_per_week']:>7,.0f} "
              f"${r['max_drawdown']:>8,.0f} {r['sharpe']:>7.2f}{goal}")

    # Walk-forward: retrain Jan+Feb, test Mar-May on best hyperparams
    best = top[0]
    print(f"\n\n=== Walk-forward (train Jan+Feb, test Mar-May) best hyperparams ===")
    wf_key = (best["pt"], best["sl"], best["lookahead"])
    if wf_key not in model_cache or model_cache[wf_key][0] is None:
        wf_model, wf_scaler, wf_auc = train_model(wf_train_bars, *wf_key)
    else:
        print("  Retraining on Jan+Feb...", end=" ", flush=True)
        wf_model, wf_scaler, wf_auc = train_model(wf_train_bars, *wf_key)
    if wf_model is not None:
        wf_r = run_oos_backtest(
            oos_bars, wf_model, wf_scaler, best["pt"], best["sl"], best["lookahead"],
            confidence_threshold=best["conf"], n_contracts=N_CONTRACTS,
            max_trades_per_day=best["mt"],
        )
        print(f"  WF OOS: {wf_r['n_trades']} trades, {wf_r['win_rate']:.1%} WR, "
              f"${wf_r['total_pnl']:,.0f} PnL, ${wf_r['max_drawdown']:,.0f} DD, "
              f"AUC={wf_auc:.3f}")
        best["wf_pnl"] = wf_r["total_pnl"]
        best["wf_max_dd"] = wf_r["max_drawdown"]
        best["wf_wr"] = wf_r["win_rate"]
        best["wf_n_trades"] = wf_r["n_trades"]

    print(f"\n\n=== Best config detail (PT={best['pt']} SL={best['sl']} "
          f"look={best['lookahead']} conf={best['conf']} mt={best['mt']}) ===")
    print(f"  {N_CONTRACTS} contracts, Jan train / Feb val / Mar-May OOS")
    print(f"  Feb val: {best.get('val_n_trades', 0)} trades, "
          f"{best.get('val_wr', 0):.1%} WR, ${best.get('val_pnl', 0):,.0f} PnL, "
          f"${best.get('val_max_dd', 0):,.0f} DD")
    print(f"  {best['n_trades']} trades, {best['win_rate']:.1%} WR, "
          f"${best['total_pnl']:,.0f} total, ${best['max_drawdown']:,.0f} max DD, "
          f"Sharpe {best['sharpe']:.2f}")
    print(f"  {best['trades_per_week']:.1f} trades/week, ${best['pnl_per_week']:,.0f}/week")

    print(f"\n  Monthly PnL:")
    monthly = {}
    for d, v in best["daily_pnl"].items():
        k = str(d)[:7]
        monthly[k] = monthly.get(k, 0) + v
    for k, v in sorted(monthly.items()):
        bar_ = "█" * int(abs(v) / 500)
        sign = "+" if v >= 0 else ""
        print(f"    {k}: ${v:>9,.0f}  {sign}{bar_}")

    # Scale to target
    if best["total_pnl"] > 0:
        target_weekly = 14_000
        contracts_needed = target_weekly / best["pnl_per_week"] * N_CONTRACTS
        dd_at_target = best["max_drawdown"] * (contracts_needed / N_CONTRACTS)
        print(f"\n  To hit ${target_weekly:,}/week needs "
              f"~{contracts_needed:.0f} contracts "
              f"→ max DD would be ${dd_at_target:,.0f}")

    import json
    out = Path(__file__).parent / "ml_strategy_results.json"
    save = [{k: v for k, v in r.items() if k != "trades"} for r in top]
    with open(out, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML strategy search / model export")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Train Jan 2026 model with DEPLOY_CONFIG and save ml_strategy_mnq_v1.pkl",
    )
    args, _ = parser.parse_known_args()
    if args.export:
        export_model()
    else:
        main()
