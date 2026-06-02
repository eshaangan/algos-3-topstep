"""Re-validate the deployed ML Scalper v7 LONG model under different session filters.

Motivation: the live runner was found to gate entries on 9:45-15:00 TRUE ET, while
the v7 backtest gated on h_et=(UTC_hour-5)%24 >= 11 and != 13 and weekday != Thu.
These windows barely overlap, so live was trading hours v7 never validated.

This script reconstructs the v7 LONG-only backtest from the DEPLOYED bundle and
compares metrics across three session filters on the OOS period (after train_end):

  A. v7_original  — h_et=(hour-5)%24; trade if h_et>=11 and !=13 and weekday!=Thu
  B. rth_strict   — 9:45-15:00 TRUE ET (what the live runner currently uses)
  C. rth_v7_intersect — RTH AND v7's hour/day exclusions

For each: n_trades, win_rate, total_pnl, sharpe, max_dd, trades/day, plus a
Monte Carlo P(pass) for a Lucid 100k combine ($6,000 target, $3,000 MLL).

Usage:
    python ml_intraday_v3/diagnostics/revalidate_v7_rth.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rule_based_v1"))

_TZ_ET = ZoneInfo("America/New_York")
_PARQUET = ROOT / "data" / "processed" / "mnq_microstructure_5min.parquet"
_BUNDLE  = ROOT / "rule_based_v1" / "models" / "ml_strategy_mnq_v7.pkl"
_OUT     = ROOT / "ml_intraday_v3" / "results" / "v7_rth_revalidation.json"

POINT_VALUE = 2.0   # MNQ $/point
N_CONTRACTS = 1
COMMISSION  = 0.62  # per side per contract

# Combine parameters (Lucid 100k Flex)
PROFIT_TARGET = 6000.0
MLL           = 3000.0
COMBINE_DAYS  = 30   # Monte Carlo horizon


class CalibratedPipeline:
    """Pickle-compat shim for the deployed v7 bundle."""
    def __init__(self, lgbm_model=None, calibrator=None):
        self.lgbm = lgbm_model
        self.calibrator = calibrator
    def predict_proba(self, X):
        raw = self.lgbm.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)


def _load():
    with open(_BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    micro = pd.read_parquet(_PARQUET)
    micro.index = pd.to_datetime(micro.index, utc=True)
    return bundle, micro


def _session_mask(idx: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """Return boolean array: True where an entry is allowed."""
    if kind == "v7_original":
        h_et = (idx.hour - 5) % 24
        return (h_et >= 11) & (h_et != 13) & (idx.weekday != 3)
    if kind == "rth_strict":
        et = idx.tz_convert(_TZ_ET)
        mins = et.hour * 60 + et.minute
        return (mins >= 9 * 60 + 45) & (mins < 15 * 60)
    if kind == "rth_v7_intersect":
        et = idx.tz_convert(_TZ_ET)
        mins = et.hour * 60 + et.minute
        rth = (mins >= 9 * 60 + 45) & (mins < 15 * 60)
        h_et = (idx.hour - 5) % 24
        v7 = (h_et != 13) & (idx.weekday != 3)
        return rth & v7
    raise ValueError(kind)


def backtest(feat: pd.DataFrame, model, feature_cols, cfg, session_kind: str) -> pd.DataFrame:
    """LONG-only backtest matching the live runner's exit logic."""
    conf      = float(cfg["conf"])
    pt_mult   = float(cfg["pt"])
    sl_mult   = float(cfg["sl"])
    horizon   = int(cfg["lookahead"])
    max_trades = int(cfg.get("max_trades", 5))

    valid = feat[feature_cols].notna().all(axis=1) & feat["atr"].notna()
    df = feat[valid].copy()
    if df.empty:
        return pd.DataFrame()

    X = df[feature_cols].astype(np.float32).values
    p_long = model.predict_proba(X)[:, 1]

    allow = _session_mask(df.index, session_kind)

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    av = df["atr"].values
    idx = df.index

    trades = []
    daily_count: dict[str, int] = {}
    in_trade = False
    ep = ea = 0.0
    eb = 0
    scale = POINT_VALUE * N_CONTRACTS

    for i in range(len(df)):
        ds = str(idx[i].date())
        daily_count.setdefault(ds, 0)

        if in_trade:
            ptp = ep + pt_mult * ea
            slp = ep - sl_mult * ea
            if h[i] >= ptp:
                pts, why = pt_mult * ea, "PT"
            elif l[i] <= slp:
                pts, why = -sl_mult * ea, "SL"
            elif (i - eb) >= horizon:
                pts, why = c[i] - ep, "TIME"
            else:
                continue
            gross = pts * scale
            net   = gross - 2 * COMMISSION * N_CONTRACTS
            trades.append({
                "entry_time": str(idx[eb]), "exit_time": str(idx[i]),
                "pnl_pts": round(pts, 2), "pnl_dollars": round(net, 2),
                "exit_reason": why,
            })
            in_trade = False
            continue

        if daily_count[ds] >= max_trades:
            continue
        if np.isnan(av[i]) or av[i] <= 0:
            continue
        if not allow[i]:
            continue
        if p_long[i] >= conf:
            in_trade = True
            ep, ea, eb = c[i], av[i], i
            daily_count[ds] += 1

    return pd.DataFrame(trades)


def summarize(tdf: pd.DataFrame, label: str) -> dict:
    if tdf is None or tdf.empty:
        return {"label": label, "n_trades": 0}
    tdf = tdf.copy()
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d")
    daily = tdf.groupby("date")["pnl_dollars"].sum()
    cum = tdf["pnl_dollars"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    wins = tdf[tdf["pnl_dollars"] > 0]
    n = len(tdf)
    nd = tdf["date"].nunique()
    return {
        "label": label,
        "n_trades": n,
        "n_days": int(nd),
        "trades_per_day": round(n / max(nd, 1), 2),
        "total_pnl": round(float(tdf["pnl_dollars"].sum()), 2),
        "win_rate": round(len(wins) / n, 4),
        "avg_win": round(float(wins["pnl_dollars"].mean()), 2) if len(wins) else 0,
        "avg_loss": round(float(tdf[tdf["pnl_dollars"] <= 0]["pnl_dollars"].mean()), 2) if (n - len(wins)) else 0,
        "max_drawdown": round(max_dd, 2),
        "sharpe_daily": round(float(daily.mean() / daily.std() * (252 ** 0.5)), 3) if daily.std() > 0 else 0,
        "daily_pnl_mean": round(float(daily.mean()), 2),
        "daily_pnl_std": round(float(daily.std()), 2) if len(daily) > 1 else 0,
    }


def monte_carlo_pass(tdf: pd.DataFrame, n_contracts: int = 1, max_days: int = 500,
                     n_paths: int = 10000, seed: int = 42) -> dict:
    """
    Bootstrap daily PnL to estimate P(reach +$6k before -$3k trailing DD).

    Lucid Flex has no time limit, so we run each path until pass OR bust
    (max_days is a practical cap). The backtest computes PnL at 1 contract;
    n_contracts scales daily PnL linearly (drawdown scales too).
    """
    if tdf is None or tdf.empty:
        return {"p_pass": 0.0, "p_bust_mll": 0.0, "median_days": None}
    tdf = tdf.copy()
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d")
    daily = tdf.groupby("date")["pnl_dollars"].sum().values * n_contracts
    if len(daily) < 2:
        return {"p_pass": 0.0, "p_bust_mll": 0.0, "median_days": None}

    rng = np.random.default_rng(seed)
    passes = 0
    fails  = 0
    days_to_pass = []
    for _ in range(n_paths):
        equity = 0.0
        peak = 0.0
        for d in range(max_days):
            equity += rng.choice(daily)
            peak = max(peak, equity)
            if equity - peak <= -MLL:   # trailing drawdown from peak
                fails += 1
                break
            if equity >= PROFIT_TARGET:
                passes += 1
                days_to_pass.append(d + 1)
                break
    return {
        "p_pass": round(passes / n_paths, 4),
        "p_bust_mll": round(fails / n_paths, 4),
        "median_days": int(np.median(days_to_pass)) if days_to_pass else None,
    }


def main():
    bundle, micro = _load()
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    cfg          = bundle["config"]
    train_end    = pd.Timestamp(bundle["train_end"], tz="UTC")

    from ml_intraday_v3.scripts.ml_scalper_v7 import build_features
    feat = build_features(micro)
    oos = feat[feat.index > train_end].copy()
    print(f"OOS bars: {len(oos)} ({oos.index[0].date()} -> {oos.index[-1].date()})")
    print(f"Model train_auc={bundle.get('train_auc'):.4f}, conf={cfg['conf']:.4f}, "
          f"pt={cfg['pt']}, sl={cfg['sl']}, lookahead={cfg['lookahead']}\n")

    results = {}
    for kind in ("v7_original", "rth_strict", "rth_v7_intersect"):
        tdf = backtest(oos, model, feature_cols, cfg, kind)
        summ = summarize(tdf, kind)
        # Contract-scaling pass probabilities (unlimited-time Flex combine)
        scaling = {}
        for N in (1, 2, 3, 4, 5, 6):
            scaling[N] = monte_carlo_pass(tdf, n_contracts=N)
        summ["mc_by_contracts"] = scaling
        results[kind] = summ
        print(f"=== {kind} ===")
        for k, v in summ.items():
            if k not in ("label", "mc_by_contracts"):
                print(f"  {k}: {v}")
        print(f"  {'N':>2} {'P(pass)':>8} {'P(bust)':>8} {'med_days':>9}")
        for N, mc in scaling.items():
            print(f"  {N:>2} {mc['p_pass']:>8.3f} {mc['p_bust_mll']:>8.3f} {str(mc['median_days']):>9}")
        print()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    main()
