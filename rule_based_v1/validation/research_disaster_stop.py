"""Does a pre-specified disaster stop unlock size? (prereg: vps_disaster_stop_sizing.yaml)

Track B's finding was that the binding constraint is single-event risk, not the flatten
and not significance. The weekend leg's worst trade is -$3,801 at 2 micros = 127% of the
$3,000 MLL, so 2 micros is an account-kill at EVERY firm surveyed and the book is stuck
at 1 micro and ~$132-232/week against the $375-500 the deadline needs.

The lever is therefore not a new edge but capping the left tail. This asks whether a hard
stop removes disasters WITHOUT removing the premium.

The fill model is deliberately punitive: a stop fills at min(stop, bar OPEN), so a bar
that gaps through the stop fills at the gap, not the stop. Sunday-reopen and Monday gaps
are precisely the risk being hedged; assuming they can be traded through at the stop
price would be assuming away the problem.

Needs OHLC, so it does not reuse research_portfolio_mc's close-only loader.

Usage:
    python3 rule_based_v1/validation/research_disaster_stop.py --out runs/disaster_stop.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tzguard import assert_et_index, check_window_legal, load_firm_rules
from research_firm_choice_mc import mc

ET = "America/New_York"
STOPS = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]   # per micro, USD
PV, TICK, COMM = 2.0, 0.25, 0.62                      # MNQ micro


def load_ohlc(path: Path, ts_col: str, tz: str | None) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    idx = idx.dt.tz_convert(ET)
    keep = ~idx.isna()
    df = pd.DataFrame({c: raw[c].to_numpy(float)[keep] for c in
                       ("open", "high", "low", "close")}, index=idx[keep])
    df["vol"] = raw["vol"].to_numpy(float)[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    assert_et_index(df.index, df["vol"], label=str(path))
    return df


def run_leg(df: pd.DataFrame, t_in: pd.Timestamp, t_out: pd.Timestamp,
            stop_pts: float | None) -> dict | None:
    """Long from t_in to t_out with an optional hard stop, gap-through modelled."""
    a = df.loc[t_in:t_in + pd.Timedelta(minutes=25)]
    b = df.loc[t_out - pd.Timedelta(minutes=30):t_out]
    if a.empty or b.empty or b.index[-1] <= a.index[0]:
        return None
    entry = float(a["open"].iloc[0])
    path = df.loc[a.index[0]:b.index[-1]]
    stopped, exit_pts = False, float(b["close"].iloc[-1]) - entry
    if stop_pts is not None:
        level = entry - stop_pts
        hit = path.index[path["low"] <= level]
        if len(hit):
            bar = path.loc[hit[0]]
            # gap-through: fill at the bar's OPEN if it opened below the stop
            fill = min(level, float(bar["open"]))
            stopped, exit_pts = True, fill - entry
            # The position is CLOSED here, so the path -- and therefore the MAE the
            # account actually experiences -- ends at this bar. Carrying the unstopped
            # path minimum forward would charge the account for an excursion it was
            # never in, and would make every stopped configuration look untradeable.
            path = df.loc[a.index[0]:hit[0]]
    mae_pts = min(float(path["low"].min()) - entry, exit_pts if stopped else 0.0)
    return {"entry": entry, "points": exit_pts, "mae_pts": mae_pts, "stopped": stopped}


def weekend_rows(df: pd.DataFrame, stop_pts: float | None) -> pd.DataFrame:
    days = pd.DatetimeIndex(np.unique(df.index.normalize()))
    out = []
    for sun in days[days.dayofweek == 6]:
        mon = sun + pd.Timedelta(days=1)
        r = run_leg(df, sun + pd.Timedelta(hours=18),
                    mon + pd.Timedelta(hours=15, minutes=59), stop_pts)
        if r:
            out.append({"day": mon, **r})
    return pd.DataFrame(out)


def fomc_rows(df: pd.DataFrame, cal: Path, stop_pts: float | None,
              entry_hour: float = 18.0) -> pd.DataFrame:
    ann = {pd.Timestamp(d).date() for d in pd.read_csv(cal)["announcement_date"]}
    mins = df.index.hour * 60 + df.index.minute
    rth = df[(mins >= 570) & (mins <= 960)]
    g = rth.groupby(rth.index.normalize()).size()
    sessions = pd.DatetimeIndex(g[g >= 60].index)
    out = []
    for i in range(1, len(sessions)):
        d = sessions[i]
        if d.date() not in ann:
            continue
        r = run_leg(df, sessions[i - 1] + pd.Timedelta(hours=entry_hour),
                    d + pd.Timedelta(hours=13, minutes=55), stop_pts)
        if r:
            out.append({"day": d, **r})
    return pd.DataFrame(out)


def sized(tr: pd.DataFrame, nc: int) -> pd.DataFrame:
    base = 2 * nc * (COMM + TICK * PV)
    cost = base + np.where(tr["stopped"], nc * TICK * PV, 0.0)   # extra tick on stop exits
    return pd.DataFrame({"pnl": tr["points"] * PV * nc - cost,
                         "mae": np.minimum(tr["mae_pts"], 0) * PV * nc})


def summarise(tr: pd.DataFrame, nc: int, label: str) -> dict:
    s = sized(tr, nc)
    t = stats.ttest_1samp(s["pnl"], 0)
    return {"label": label, "n": int(len(s)), "mean": float(s["pnl"].mean()),
            "t_stat": float(t.statistic),
            "worst": float(s["pnl"].min()), "worst_mae": float(s["mae"].min()),
            "stop_rate": float(tr["stopped"].mean()),
            "win_rate": float((s["pnl"] > 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description="disaster-stop sizing")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--calendar", type=Path,
                    default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--firm", default="lucid_flex_100k")
    ap.add_argument("--max-weeks", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = load_firm_rules(args.firm)
    target, mll, cons = float(r["profit_target"]), float(r["max_loss_limit"]), r["consistency"]
    ok_wk, _ = check_window_legal("18:00", "15:59", 1, r)
    ok_fo, _ = check_window_legal("18:00", "13:55", 1, r)
    print(f"\n{r['label']}  target=${target:,.0f}  MLL=${mll:,.0f}  "
          f"consistency={f'{cons:.0%}' if cons else 'none'}")
    print(f"legality: weekend={ok_wk}  fomc@18:00={ok_fo}   "
          f"tail budget = 60% of MLL = ${0.6 * mll:,.0f}")

    df = load_ohlc(args.bars, args.ts_col, args.tz)
    legs = {}
    print(f"\n{'=' * 96}\nDoes the stop cap the tail WITHOUT cutting the premium? "
          f"(per-micro figures)\n{'=' * 96}")
    for name, fn in (("weekend", lambda sp: weekend_rows(df, sp)),
                     ("fomc@18:00", lambda sp: fomc_rows(df, args.calendar, sp, 18.0))):
        base = fn(None)
        b = summarise(base, 1, "no stop")
        print(f"\n{name}:  n={b['n']}")
        print(f"  {'stop/micro':>11}{'mean$':>9}{'t':>7}{'worst$':>10}{'stop%':>8}"
              f"{'WR':>7}   {'mean kept':>10}{'tail cut':>10}")
        print(f"  {'none':>11}{b['mean']:>9.2f}{b['t_stat']:>7.2f}{b['worst']:>10,.0f}"
              f"{0.0:>8.0%}{b['win_rate']:>7.1%}{1.0:>10.0%}{0.0:>10.0%}")
        rows = {None: base}
        for sd in STOPS:
            sp = sd / PV                      # dollars per micro -> points
            tr = fn(sp)
            s = summarise(tr, 1, f"${sd:,.0f}")
            rows[sd] = tr
            kept = s["mean"] / b["mean"] if b["mean"] else float("nan")
            tail_cut = 1 - abs(s["worst"]) / abs(b["worst"]) if b["worst"] else float("nan")
            print(f"  {f'${sd:,.0f}':>11}{s['mean']:>9.2f}{s['t_stat']:>7.2f}"
                  f"{s['worst']:>10,.0f}{s['stop_rate']:>8.1%}{s['win_rate']:>7.1%}"
                  f"{kept:>10.0%}{tail_cut:>10.0%}")
        legs[name] = rows

    # ---- MC over (stop, size) pairs, disqualifying any single-event kill -------------
    budget = 0.60 * mll
    print(f"\n{'=' * 96}\nP(pass) within {args.max_weeks} weeks. Sizes whose worst "
          f"historical trade exceeds\n60% of the MLL (${budget:,.0f}) are DISQUALIFIED "
          f"and not shown.\n{'=' * 96}")
    print(f"  {'stop/micro':>11}{'wk':>4}{'fomc':>6}{'$/wk':>9}{'worst wk$':>11}"
          f"{'worst fo$':>11}{'P(pass)':>10}{'median':>8}{'decay x0.50':>13}")
    results, best = [], None
    for sd in [None] + STOPS:
        wk_raw, fo_raw = legs["weekend"][sd], legs["fomc@18:00"][sd]
        for wn in (1, 2, 3, 4, 5, 6):
            for fn_ in (0, 1, 2, 3, 4, 6):
                wk, fo = sized(wk_raw, wn), sized(fo_raw, fn_) if fn_ else None
                w_worst = float(wk["pnl"].min())
                f_worst = float(fo["pnl"].min()) if fn_ else 0.0
                if abs(w_worst) > budget or abs(f_worst) > budget:
                    continue
                fo_df = fo if fn_ else pd.DataFrame({"pnl": [0.0], "mae": [0.0]})
                kw = dict(target=target, mll=mll, consistency=cons,
                          min_days=int(r["min_trading_days"]), daily_loss_limit=None,
                          fomc_per_year=8.0 if fn_ else 0.0, max_weeks=args.max_weeks)
                m = mc(wk, fo_df, **kw)
                d50 = mc(wk, fo_df, **kw, weekend_decay=0.50)["p_pass"]
                per_wk = wk["pnl"].mean() + (fo["pnl"].mean() * 8 / 52 if fn_ else 0.0)
                rec = {"stop_usd_per_micro": sd, "weekend": wn, "fomc": fn_,
                       "per_week": per_wk, "worst_weekend": w_worst,
                       "worst_fomc": f_worst, "p_pass": m["p_pass"],
                       "median_weeks": m["median_weeks"], "p_pass_decay50": d50}
                results.append(rec)
                if m["p_pass"] >= 0.30:
                    flag = "  <== CLEARS 85%" if m["p_pass"] >= 0.85 else ""
                    print(f"  {(f'${sd:,.0f}' if sd else 'none'):>11}{wn:>4}{fn_:>6}"
                          f"{per_wk:>9.0f}{w_worst:>11,.0f}{f_worst:>11,.0f}"
                          f"{m['p_pass']:>10.1%}{m['median_weeks']:>8.0f}{d50:>13.1%}{flag}")
                if m["p_pass"] >= 0.85 and (best is None or
                                            m["median_weeks"] < best["median_weeks"]):
                    best = rec
    print()
    if best:
        print(f"BEST >=85% within {args.max_weeks}wk: stop "
              f"${best['stop_usd_per_micro']:,.0f}/micro, weekend x{best['weekend']} + "
              f"fomc x{best['fomc']} -> {best['p_pass']:.1%} / "
              f"{best['median_weeks']:.0f}wk, ${best['per_week']:.0f}/wk, "
              f"worst weekend ${best['worst_weekend']:,.0f}")
    else:
        print(f"NOTHING clears 85% within {args.max_weeks} weeks, at any stop or size.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
