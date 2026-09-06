"""Volatility-targeted sizing on weekend_hold_v1 — attacking the buffer, not the signal.

Everything in this session's hunt says the same thing: drift is not the scarce
resource, path is. Long ES for a week already pays t=+2.43; it is unusable because
the drawdown does not fit a $3k trailing buffer. weekend_hold_v1 has the best
premium-per-unit-of-path in the project and is still limited by MAE, not by mean.

So instead of hunting another signal, size the one that works. Moreira & Muir
(2017, JF, "Volatility-Managed Portfolios") show that scaling exposure inversely
to recent realized volatility raises risk-adjusted returns across equity
strategies. That is a published, mechanism-backed rule, not a fitted filter: it
uses no forward information and has one parameter with an economic meaning
(target dollar risk), which is set from the account's buffer rather than tuned
for PnL.

Contracts are integers with a floor of 1, because that is what actually gets
traded. Sizing is computed from a trailing window that ends BEFORE entry.

Usage:
    python3 rule_based_v1/validation/research_vol_target_weekend.py \
        --bars data/processed/mnq_1m_all.parquet --ts-col ts --tz America/Chicago \
        --point-value 2.0 --out runs/voltarget_mnq.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"


def load_px(path: Path, ts_col: str, tz: str | None) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def weekend_legs(px: pd.Series) -> pd.DataFrame:
    """Sun 18:00 ET -> Mon 15:59 ET, per the live spec, in POINTS (unsized)."""
    days = pd.DatetimeIndex(np.unique(px.index.normalize()))
    rows = []
    for sun in days[days.dayofweek == 6]:
        mon = sun + pd.Timedelta(days=1)
        a = px[sun + pd.Timedelta(hours=18): sun + pd.Timedelta(hours=18, minutes=10)]
        b = px[mon + pd.Timedelta(hours=15, minutes=29): mon + pd.Timedelta(hours=15, minutes=59)]
        if a.empty or b.empty:
            continue
        path = px[a.index[0]:b.index[-1]]
        entry = float(a.iloc[0])
        rows.append({"mon": mon, "entry": entry, "points": float(b.iloc[-1]) - entry,
                     "mae_pts": float(path.min() - entry)})
    return pd.DataFrame(rows)


def trailing_vol(px: pd.Series, asof: pd.DatetimeIndex, lookback: int) -> np.ndarray:
    """Stdev of the last `lookback` daily closes' returns, strictly before asof."""
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    daily = rth.groupby(rth.index.normalize()).last()
    ret = daily.diff()
    vol = ret.rolling(lookback, min_periods=lookback).std().shift(1)
    idx = vol.index.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
    ask = asof.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
    pos = np.searchsorted(idx, ask, "right") - 1
    out = np.full(len(ask), np.nan)
    ok = pos >= 0
    out[ok] = vol.to_numpy()[pos[ok]]
    return out


def book(tr: pd.DataFrame, contracts: np.ndarray, point_value: float,
         tick: float, comm: float) -> pd.DataFrame:
    cost = 2 * contracts * (comm + 1.0 * tick * point_value)
    out = tr.copy()
    out["contracts"] = contracts
    out["pnl"] = tr["points"] * point_value * contracts - cost
    out["mae"] = tr["mae_pts"] * point_value * contracts
    return out


def combine_mc(b: pd.DataFrame, target: float, buffer_usd: float,
               n_paths: int = 20000, max_weeks: int = 104,
               seed: int = 7) -> tuple[float, float]:
    """Probability of passing an evaluation, and median weeks to pass.

    Bootstraps weekends with replacement. Each week the account can bust two ways:
    intra-trade, if equity plus that trade's own MAE breaches the floor, or on the
    close. The floor is EOD-trailing (Lucid updates it at 4:45pm), so it ratchets
    on closed equity only -- an adverse excursion can bust you but cannot raise
    the floor. Passing means reaching +target before ever breaching.
    """
    rng = np.random.default_rng(seed)
    pnl = b["pnl"].to_numpy()
    mae = np.minimum(b["mae"].to_numpy(), 0.0)
    draws = rng.integers(0, len(pnl), size=(n_paths, max_weeks))
    eq = np.zeros(n_paths)
    floor = np.full(n_paths, -buffer_usd)
    peak = np.zeros(n_paths)
    done = np.zeros(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    weeks = np.full(n_paths, max_weeks, dtype=int)
    for w in range(max_weeks):
        live = ~done
        if not live.any():
            break
        i = draws[:, w]
        bust_path = live & ((eq + mae[i]) < floor)
        done |= bust_path
        live = ~done
        eq = np.where(live, eq + pnl[i], eq)
        hit = live & (eq >= target)
        passed |= hit
        weeks = np.where(hit & (weeks == max_weeks), w + 1, weeks)
        done |= hit
        live = ~done
        bust_close = live & (eq < floor)
        done |= bust_close
        live = ~done
        peak = np.where(live, np.maximum(peak, eq), peak)
        # LOCK: LucidFlex's EOD-trailing drawdown STOPS trailing once the floor
        # reaches the starting balance (account_rules.yaml::drawdown.
        # locks_at_starting_balance). Equity is relative here, so the starting
        # balance is 0 and the floor may never exceed it. Trailing past 0 -- which
        # every MC in this repo did until 2026-09-05 -- silently understates
        # P(pass), and it understates it MOST at large sizes, because those are
        # the paths that build a big peak.
        floor = np.where(live, np.maximum(floor, np.minimum(peak - buffer_usd, 0.0)), floor)
    med = float(np.median(weeks[passed])) if passed.any() else float("nan")
    return float(passed.mean()), med


def summarise(b: pd.DataFrame, label: str, buffer_usd: float, target: float) -> dict:
    p = b["pnl"]
    t = stats.ttest_1samp(p, 0)
    eq = p.cumsum()
    p_pass, med = combine_mc(b, target, buffer_usd)
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "t_stat": float(t.statistic), "win_rate": float((p > 0).mean()),
            "mean_contracts": float(b["contracts"].mean()),
            "closed_maxdd": float((eq - eq.cummax()).min()),
            "worst_mae": float(b["mae"].min()),
            "p_pass": p_pass, "median_weeks": med}


def show(s: dict) -> None:
    print(f"  {s['label']:<28} ${s['mean']:+7.2f}/wk  t={s['t_stat']:+5.2f}  "
          f"avgC={s['mean_contracts']:.2f}  worstMAE=${s['worst_mae']:>7,.0f}  "
          f"P(pass)={s['p_pass']:6.1%}  median={s['median_weeks']:.0f}wk")


def main() -> None:
    ap = argparse.ArgumentParser(description="vol-targeted sizing on weekend_hold_v1")
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--ts-col", default="et")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--comm", type=float, default=0.62)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--max-contracts", type=int, default=4)
    ap.add_argument("--buffer", type=float, default=3000.0)
    ap.add_argument("--target", type=float, default=3000.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    px = load_px(args.bars, args.ts_col, args.tz)
    tr = weekend_legs(px)
    vol = trailing_vol(px, pd.DatetimeIndex(tr["mon"]), args.lookback)
    tr = tr.assign(vol=vol).dropna(subset=["vol"]).reset_index(drop=True)
    print(f"weekends {len(tr)}  {tr['mon'].min().date()} .. {tr['mon'].max().date()}  "
          f"point_value ${args.point_value}  buffer ${args.buffer:,.0f}\n")

    results = []
    # Baseline: the live spec, flat 2 contracts.
    base = book(tr, np.full(len(tr), 2), args.point_value, args.tick, args.comm)
    s = summarise(base, "flat 2 (live spec)", args.buffer, args.target)
    results.append(s)
    show(s)

    # Vol target: risk_usd is the dollar move a 1-sigma day would produce, so
    # contracts = target / (vol_points * point_value), floored at 1.
    dollar_vol = tr["vol"].to_numpy() * args.point_value
    print(f"  [1-sigma daily move is ${np.median(dollar_vol):,.0f} per contract at the median]")
    # targets as multiples of the median 1-sigma move, so the grid spans roughly
    # 1 to 4 contracts at typical vol while still varying with the regime
    med_vol = float(np.median(dollar_vol))
    grid = [round(med_vol * k) for k in (1.0, 1.5, 2.0, 3.0, 4.0)]
    for target in grid:
        c = np.clip(np.round(target / dollar_vol), 1, args.max_contracts).astype(int)
        b = book(tr, c, args.point_value, args.tick, args.comm)
        s = summarise(b, f"vol-target ${target}/sigma", args.buffer, args.target)
        results.append(s)
        show(s)

    # Reference: flat sizes, to separate "vol targeting helped" from "smaller helped"
    print()
    for n in (1, 3, 4):
        b = book(tr, np.full(len(tr), n), args.point_value, args.tick, args.comm)
        s = summarise(b, f"flat {n} (reference)", args.buffer, args.target)
        results.append(s)
        show(s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
