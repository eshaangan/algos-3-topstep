"""Portfolio evaluation MC: weekend_hold_v1 + fomc_drift_v1 on one account.

No new edge here. This asks the only question that actually decides whether the
project can pass an evaluation with what it already has: given the two validated,
NON-OVERLAPPING windows, what sizing maximises P(pass) against a $3k trailing
buffer, and how much does the answer depend on the regime?

Overlap matters. monday_rth_v1 is deliberately excluded: its window (Mon 09:30 ->
16:00) sits entirely INSIDE weekend_hold_v1 (Sun 18:00 -> Mon 15:59), so trading
both is not diversification, it is the same exposure counted twice. FOMC drift
(14:00 day-before -> 13:55 decision day, usually Tue->Wed) is genuinely disjoint
from the Sunday-Monday window.

The regime split is reported alongside every number, because this session
established that pre-2020 ES weekends paid -$8.80 while post-2020 paid +$65. A
P(pass) bootstrapped only from the good regime is an answer to the wrong question.

Usage:
    python3 rule_based_v1/validation/research_portfolio_mc.py --out runs/portfolio_mc.json
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


def _leg(px: pd.Series, t_in: pd.Timestamp, t_out: pd.Timestamp) -> dict | None:
    a = px[t_in:t_in + pd.Timedelta(minutes=10)]
    b = px[t_out - pd.Timedelta(minutes=30):t_out]
    if a.empty or b.empty or b.index[-1] <= a.index[0]:
        return None
    entry = float(a.iloc[0])
    path = px[a.index[0]:b.index[-1]]
    return {"points": float(b.iloc[-1]) - entry, "mae_pts": float(path.min() - entry)}


def weekend_trades(px: pd.Series) -> pd.DataFrame:
    days = pd.DatetimeIndex(np.unique(px.index.normalize()))
    rows = []
    for sun in days[days.dayofweek == 6]:
        mon = sun + pd.Timedelta(days=1)
        r = _leg(px, sun + pd.Timedelta(hours=18), mon + pd.Timedelta(hours=15, minutes=59))
        if r:
            rows.append({"day": mon, **r})
    return pd.DataFrame(rows)


def fomc_trades(px: pd.Series, calendar: Path, entry_hour: float = 14.0) -> pd.DataFrame:
    ann = pd.DatetimeIndex(pd.read_csv(calendar)["announcement_date"]).tz_localize(ET)
    days = set(np.unique(px.index.normalize()))
    rows = []
    for a in ann:
        prev = a - pd.Timedelta(days=1)
        while prev not in days and (a - prev).days < 5:
            prev -= pd.Timedelta(days=1)
        if prev not in days or a not in days:
            continue
        r = _leg(px, prev + pd.Timedelta(hours=entry_hour), a + pd.Timedelta(hours=13, minutes=55))
        if r:
            rows.append({"day": a, **r})
    return pd.DataFrame(rows)


def sized(tr: pd.DataFrame, contracts: int, pv: float, tick: float, comm: float) -> pd.DataFrame:
    cost = 2 * contracts * (comm + 1.0 * tick * pv)
    return pd.DataFrame({"pnl": tr["points"] * pv * contracts - cost,
                         "mae": np.minimum(tr["mae_pts"], 0) * pv * contracts})


def portfolio_mc(wk: pd.DataFrame, fo: pd.DataFrame, target: float, buffer_usd: float,
                 fomc_per_year: float = 8.0, n_paths: int = 20000,
                 max_weeks: int = 104, seed: int = 11) -> tuple[float, float]:
    """One account, weekly weekend trade plus a FOMC trade in ~8/52 of weeks."""
    rng = np.random.default_rng(seed)
    wp, wm = wk["pnl"].to_numpy(), wk["mae"].to_numpy()
    fp, fm = fo["pnl"].to_numpy(), fo["mae"].to_numpy()
    p_fomc = fomc_per_year / 52.0
    eq = np.zeros(n_paths)
    floor = np.full(n_paths, -buffer_usd)
    peak = np.zeros(n_paths)
    done = np.zeros(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    weeks = np.full(n_paths, max_weeks, dtype=int)
    for w in range(max_weeks):
        if done.all():
            break
        live = ~done
        i = rng.integers(0, len(wp), n_paths)
        has_f = rng.random(n_paths) < p_fomc
        j = rng.integers(0, len(fp), n_paths)
        step_pnl = wp[i] + np.where(has_f, fp[j], 0.0)
        # trades are on different days, so the worst point of the week is the worse
        # single excursion, not their sum
        step_mae = np.minimum(wm[i], np.where(has_f, fm[j], 0.0))
        bust = live & ((eq + step_mae) < floor)
        done |= bust
        live = ~done
        eq = np.where(live, eq + step_pnl, eq)
        hit = live & (eq >= target)
        passed |= hit
        weeks = np.where(hit & (weeks == max_weeks), w + 1, weeks)
        done |= hit
        live = ~done
        done |= live & (eq < floor)
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
    return float(passed.mean()), (float(np.median(weeks[passed])) if passed.any() else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description="portfolio evaluation MC")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--comm", type=float, default=0.62)
    ap.add_argument("--calendar", type=Path, default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--buffer", type=float, default=3000.0)
    ap.add_argument("--target", type=float, default=3000.0)
    ap.add_argument("--fomc-entry-hour", type=float, default=14.0,
                    help="14 = live spec (24h hold); 16 = HPWZ prior-close entry (22h)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    px = load_px(args.bars, args.ts_col, args.tz)
    wk_raw, fo_raw = weekend_trades(px), fomc_trades(px, args.calendar, args.fomc_entry_hour)
    print(f"[fomc entry {args.fomc_entry_hour:g}:00 prior session]")
    for nm, d in (("weekend", wk_raw), ("fomc", fo_raw)):
        t = stats.ttest_1samp(d["points"], 0)
        print(f"{nm:<9} n={len(d):<4} mean={d['points'].mean():+7.2f} pts  t={t.statistic:+5.2f}")
    print()

    rows = []
    print(f"{'weekend':>8}{'fomc':>6}{'$/wk':>9}{'P(pass)':>9}{'median':>8}")
    for wn in (0, 1, 2):
        for fn in (2, 3, 4, 6):
            wk = sized(wk_raw, wn, args.point_value, args.tick, args.comm) if wn else \
                pd.DataFrame({"pnl": [0.0], "mae": [0.0]})
            fo = sized(fo_raw, fn, args.point_value, args.tick, args.comm) if fn else \
                pd.DataFrame({"pnl": [0.0], "mae": [0.0]})
            p, med = portfolio_mc(wk, fo, args.target, args.buffer,
                                  fomc_per_year=8.0 if fn else 0.0)
            per_wk = (wk["pnl"].mean() if wn else 0.0) + (fo["pnl"].mean() * 8 / 52 if fn else 0.0)
            rows.append({"weekend_contracts": wn, "fomc_contracts": fn,
                         "per_week": per_wk, "p_pass": p, "median_weeks": med})
            print(f"{wn:>8}{fn:>6}{per_wk:>9.2f}{p:>9.1%}{med:>8.0f}")

    # DECAY STRESS. Every number above is bootstrapped from post-2020 MNQ. The 15.5y
    # ES tape says the weekend leg paid -$8.80/wk pre-2020 and +$65 after, so the
    # forward mean is the single most uncertain input. Shrink each leg's MEAN toward
    # zero while keeping its dispersion and MAE, and see where the plan breaks.
    print("\nDECAY STRESS -- weekend leg mean shrunk, dispersion and MAE unchanged")
    print(f"{'config':>12}{'x1.00':>9}{'x0.75':>9}{'x0.50':>9}{'x0.25':>9}{'x0.00':>9}")
    stress = []
    for wn, fn in ((1, 2), (1, 4), (1, 6), (2, 2), (0, 4)):
        line, cells = f"{wn}wk+{fn}fomc", []
        for k in (1.0, 0.75, 0.5, 0.25, 0.0):
            wk = sized(wk_raw, wn, args.point_value, args.tick, args.comm) if wn else None
            if wn:
                wk = wk.assign(pnl=(wk["pnl"] - wk["pnl"].mean()) + wk["pnl"].mean() * k)
            if not wn:
                wk = pd.DataFrame({"pnl": [0.0], "mae": [0.0]})
            fo = (sized(fo_raw, fn, args.point_value, args.tick, args.comm) if fn
                  else pd.DataFrame({"pnl": [0.0], "mae": [0.0]}))
            pp, _ = portfolio_mc(wk, fo, args.target, args.buffer,
                                 fomc_per_year=8.0 if fn else 0.0)
            cells.append(pp)
        stress.append({"weekend": wn, "fomc": fn, "p_pass_by_decay": cells})
        print(f"{line:>12}" + "".join(f"{c:>9.1%}" for c in cells))
    rows.append({"decay_stress": stress})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
