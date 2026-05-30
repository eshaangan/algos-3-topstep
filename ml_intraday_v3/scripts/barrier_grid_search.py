"""
Barrier parameter grid search — find optimal PT/SL/horizon for ml_scalper_v5+.

For each (pt_atr, sl_atr, horizon_bars) combination:
  - Labels every bar in the dataset
  - Reports: base_rate (% positive), EV per trade, break-even threshold, expected Sharpe

The goal: find parameters that maximise base_rate × WR without destroying risk:reward.
More positive labels → distribution shifts right → more bars exceed threshold → more trades.

Usage:
    python ml_intraday_v3/scripts/barrier_grid_search.py

Output:
    ml_intraday_v3/results/barrier_grid_results.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parents[2]
DATA    = BASE / "data" / "processed"
RESULTS = BASE / "ml_intraday_v3" / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE))

# Grid
PT_VALUES  = [1.25, 1.5, 1.75, 2.0, 2.5]
SL_VALUES  = [0.75, 1.0, 1.25]
HOR_VALUES = [8, 12, 16]
ATR_PERIOD = 10


def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def label_bars(df: pd.DataFrame, pt_atr: float, sl_atr: float,
               horizon_bars: int) -> pd.Series:
    """Triple-barrier labeling. Returns +1 (PT hit), -1 (SL hit), 0 (time/excluded)."""
    c   = df["close"].values
    hi  = df["high"].values
    lo  = df["low"].values
    atr = df["atr"].values
    n   = len(df)
    labels = np.full(n, -2, dtype=np.int8)   # -2 = not yet evaluated

    for i in range(n - horizon_bars):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        e  = c[i]
        pt = e + pt_atr * a
        sl = e - sl_atr * a
        labels[i] = 0  # default: time stop
        for j in range(i + 1, i + horizon_bars + 1):
            if hi[j] >= pt:
                labels[i] = 1
                break
            elif lo[j] <= sl:
                labels[i] = -1
                break
    return pd.Series(labels, index=df.index)


def ev_per_trade(labels: pd.Series, pt_atr: float, sl_atr: float,
                 avg_atr: float) -> float:
    """Expected PnL per trade (in ATR units × avg_atr × point_value × n_contracts)."""
    valid  = labels[labels != -2]
    if len(valid) == 0:
        return np.nan
    wr     = (valid == 1).mean()
    return (wr * pt_atr - (1 - wr) * sl_atr) * avg_atr


def backtest_simple(df: pd.DataFrame, labels: pd.Series,
                    pt_atr: float, sl_atr: float,
                    n_contracts: int = 6, point_value: float = 2.0) -> dict:
    """
    Simple backtest assuming we trade ALL labeled bars (no ML filter).
    This gives an upper-bound baseline for Sharpe and WR.
    """
    valid = labels[labels != -2]
    if len(valid) < 5:
        return {}

    atr = df.loc[valid.index, "atr"]
    pnl = valid.map({1: 1.0, -1: -1.0, 0: 0.0})
    # Scale: 1 pt for PT, -SL/PT pt for SL
    pnl = pnl.copy()
    pnl[valid == 1]  = pt_atr * atr[valid == 1] * n_contracts * point_value
    pnl[valid == -1] = -sl_atr * atr[valid == -1] * n_contracts * point_value
    pnl[valid == 0]  = 0.0   # time stops counted as 0 here

    daily_pnl = pnl.groupby(pd.Grouper(freq="D")).sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    cum  = pnl.cumsum()
    dd   = float((cum - cum.cummax()).min())

    return {
        "n_trades":   int(len(valid)),
        "win_rate":   round(float((valid == 1).mean()), 4),
        "total_pnl":  round(float(pnl.sum()), 2),
        "max_dd":     round(dd, 2),
        "sharpe":     round(float(daily_pnl.mean() / daily_pnl.std() * (252 ** 0.5)), 3)
                      if daily_pnl.std() > 0 else 0.0,
    }


def main():
    print("=== Barrier Grid Search ===", flush=True)

    micro_path = DATA / "mnq_microstructure_5min.parquet"
    if not micro_path.exists():
        print(f"ERROR: {micro_path} not found. Run build_microstructure_features.py first.")
        sys.exit(1)

    micro = pd.read_parquet(micro_path)
    micro.index = pd.to_datetime(micro.index, utc=True)
    micro["atr"] = compute_atr(micro, ATR_PERIOD)
    avg_atr = float(micro["atr"].median())

    n_bars  = len(micro)
    date_range = f"{micro.index[0].date()} → {micro.index[-1].date()}"
    print(f"Dataset: {n_bars} bars, {date_range}", flush=True)
    print(f"  Median ATR: {avg_atr:.1f} pts  (${avg_atr * 2:.0f}/pt × 6c)", flush=True)
    print(f"\nGrid: {len(PT_VALUES)}×{len(SL_VALUES)}×{len(HOR_VALUES)} = "
          f"{len(PT_VALUES)*len(SL_VALUES)*len(HOR_VALUES)} combinations\n", flush=True)

    results = []

    for pt in PT_VALUES:
        for sl in SL_VALUES:
            for hor in HOR_VALUES:
                labels   = label_bars(micro, pt, sl, hor)
                valid    = labels[labels != -2]
                base_rate = float((valid == 1).mean()) if len(valid) > 0 else 0.0
                ev        = ev_per_trade(valid, pt, sl, avg_atr)
                # EV-optimal threshold for this barrier combination
                breakeven_thresh = sl / (pt + sl)
                bt = backtest_simple(micro, labels, pt, sl)

                row = {
                    "pt_atr":           pt,
                    "sl_atr":           sl,
                    "horizon_bars":     hor,
                    "base_rate":        round(base_rate, 4),
                    "rr_ratio":         round(pt / sl, 2),
                    "breakeven_thresh": round(breakeven_thresh, 4),
                    "ev_per_trade_$":   round(ev, 2) if not np.isnan(ev) else None,
                    **bt,
                }
                results.append(row)

                print(
                    f"  PT={pt:.2f}×  SL={sl:.2f}×  H={hor:2d}  │  "
                    f"base={base_rate:.1%}  R:R={pt/sl:.1f}  "
                    f"thresh_ev={breakeven_thresh:.3f}  "
                    f"WR={bt.get('win_rate', 0):.1%}  "
                    f"Sharpe={bt.get('sharpe', 0):.2f}",
                    flush=True,
                )

    # Sort by composite score: base_rate × win_rate × (1 - |max_dd|/total_pnl)
    df_r = pd.DataFrame(results)
    df_r = df_r[df_r["n_trades"] > 0]

    # Score: higher is better. Penalise extreme R:R that destroys WR.
    df_r["score"] = (
        df_r["base_rate"] * df_r["win_rate"]
        * df_r["rr_ratio"].clip(upper=3)  # cap R:R contribution
        * (df_r["sharpe"].clip(lower=0) + 1)
    )

    df_r = df_r.sort_values("score", ascending=False)

    print("\n=== TOP 10 COMBINATIONS ===")
    print(df_r[["pt_atr","sl_atr","horizon_bars","base_rate","win_rate",
                "rr_ratio","ev_per_trade_$","sharpe","max_dd","score"]].head(10).to_string(index=False))

    best = df_r.iloc[0].to_dict()
    print(f"\n→ Recommended: PT={best['pt_atr']}×  SL={best['sl_atr']}×  "
          f"H={int(best['horizon_bars'])}  "
          f"(base_rate={best['base_rate']:.1%}, WR={best['win_rate']:.1%}, "
          f"Sharpe={best['sharpe']:.2f})")
    print(f"  Set CFG pt_atr={best['pt_atr']}, sl_atr={best['sl_atr']}, "
          f"horizon_bars={int(best['horizon_bars'])} in ml_scalper_v5.py")

    out = {"results": results, "top_10": df_r.head(10).to_dict(orient="records")}
    out_path = RESULTS / "barrier_grid_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
