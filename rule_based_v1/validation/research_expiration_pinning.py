"""Expiration-day strike pinning in index futures -- Golez & Jackwerth (2012), JFE.

Two questions, in this order, because the second only matters if the first holds:

1. CLUSTERING (the paper's Eq. 1-2). Is the settlement price more likely to land
   within a narrow band of the at-the-money strike on serial expiration Fridays
   than on the +/-10 surrounding sessions? The paper's structural control is that
   QUARTERLY expirations should show none, because there the front future itself
   expires into the cash basket and SP-option hedging cannot drag it.

2. TRADEABILITY. Clustering is a distributional statistic, not a forecast. The
   tradeable implication is: at decision time T, if price sits d points from the
   nearest strike, it should drift toward that strike by settlement. We price
   that trade net of the project's standard cost model.

Pre-registration: rule_based_v1/validation/vps_prereg/vps_expiration_pinning.yaml

Usage:
    python3 rule_based_v1/validation/research_expiration_pinning.py \
        --instrument es --out runs/pinning_es.json
    python3 rule_based_v1/validation/research_expiration_pinning.py \
        --instrument mnq --out runs/pinning_mnq.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"
SERIAL_MONTHS = (1, 2, 4, 5, 7, 8, 10, 11)
QUARTERLY_MONTHS = (3, 6, 9, 12)

INSTRUMENTS = {
    "es": {
        "path": "data/processed/es_bars_2010_2025.h5",
        "kind": "hdf",
        "hdf_key": "bars_5min",
        "timestamp_col": "timestamp",
        "input_timezone": None,          # already tz-aware UTC
        "point_value": 5.0,              # MES economics; ES price series
        "tick_size": 0.25,
        "grids": (5.0, 25.0, 50.0),      # 5.0 is the paper-faithful primary
        "dev_end": "2017-12-31",
    },
    "mnq": {
        "path": "data/processed/mnq_1m_all.parquet",
        "kind": "parquet",
        "hdf_key": None,
        "timestamp_col": "ts",
        "input_timezone": "America/Chicago",   # DST-aware local, verified
        "point_value": 2.0,
        "tick_size": 0.25,
        "grids": (25.0, 50.0, 100.0),
        "dev_end": "2023-12-31",
    },
}

COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1.0


def round_turn_cost(point_value: float, tick_size: float, contracts: int) -> float:
    tick_value = tick_size * point_value
    return 2.0 * contracts * (COMMISSION_PER_SIDE + SLIPPAGE_TICKS * tick_value)


# --------------------------------------------------------------------------- data


def load_rth_bars(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["path"])
    if cfg["kind"] == "hdf":
        raw = pd.read_hdf(path, key=cfg["hdf_key"])
    else:
        raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[cfg["timestamp_col"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["input_timezone"], ambiguous="NaT", nonexistent="NaT")
    out = raw.drop(columns=[cfg["timestamp_col"]]).set_index(idx.dt.tz_convert(ET))
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    mins = out.index.hour * 60 + out.index.minute
    # RTH only. Bars are stamped at their start, so the 15:55 bar is the last one
    # and its close is the 16:00 settlement print.
    return out[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 59)]


def daily_settlement(bars: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    g = bars.groupby(bars.index.normalize())
    day = g.agg(settle=("close", "last"), n=("close", "size"), last_ts=("close", lambda s: s.index[-1]))
    return day[day.n >= min_bars]


def price_at(bars: pd.DataFrame, day: pd.Timestamp, hhmm: str) -> float | None:
    """Last close at or before hh:mm on `day`. None if the session ended earlier."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    target = day + pd.Timedelta(hours=hh, minutes=mm)
    sess = bars.loc[day : day + pd.Timedelta(hours=23, minutes=59)]
    sess = sess[sess.index <= target]
    return float(sess["close"].iloc[-1]) if len(sess) else None


def expiration_days(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Third Friday of each month, mapped to the last session at or before it."""
    rows = []
    sess = pd.DatetimeIndex(sessions)
    for (y, m), _ in pd.Series(1, index=sess).groupby([sess.year, sess.month]):
        fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
        fridays = [d for d in fridays if d.month == m and d.dayofweek == 4]
        if len(fridays) < 3:
            continue
        third = fridays[2]
        pos = sess.searchsorted(third.tz_localize(ET), side="right") - 1
        if pos < 0:
            continue
        actual = sess[pos]
        # Only accept if it is the third Friday itself or the session just before it
        # (holiday-shifted expirations settle on the prior business day).
        if (third.tz_localize(ET) - actual).days > 3:
            continue
        rows.append({"month": m, "year": y, "expiry": actual,
                     "serial": m in SERIAL_MONTHS})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part 1: clustering


def dist_to_strike(price: float, grid: float) -> float:
    """Absolute distance to the nearest strike on a `grid`-point lattice."""
    return float(abs(price - round(price / grid) * grid))


def clustering_table(day: pd.DataFrame, expiries: pd.DataFrame, grid: float,
                     band: float, window: int) -> dict:
    sess = day.index
    d = day.copy()
    d["dist"] = [dist_to_strike(p, grid) for p in d["settle"]]
    d["pin"] = (d["dist"] <= band).astype(int)

    def rates(days: pd.DatetimeIndex) -> tuple[int, float]:
        sub = d.reindex(days).dropna(subset=["pin"])
        return len(sub), float(sub["pin"].mean()) if len(sub) else float("nan")

    out = {"grid": grid, "band": band,
           "uniform_null": min(1.0, 2.0 * band / grid)}
    for label, mask in (("serial", expiries["serial"]), ("quarterly", ~expiries["serial"])):
        exp_days = pd.DatetimeIndex(expiries.loc[mask, "expiry"])
        ctrl = []
        for e in exp_days:
            pos = sess.searchsorted(e)
            lo, hi = max(0, pos - window), min(len(sess), pos + window + 1)
            ctrl.extend(x for x in sess[lo:hi] if x != e)
        ctrl = pd.DatetimeIndex(sorted(set(ctrl))).difference(exp_days)
        n_e, r_e = rates(exp_days)
        n_c, r_c = rates(ctrl)
        # two-proportion z-test, expiration vs surrounding control window
        if n_e and n_c:
            p = (r_e * n_e + r_c * n_c) / (n_e + n_c)
            se = np.sqrt(p * (1 - p) * (1 / n_e + 1 / n_c))
            z = (r_e - r_c) / se if se > 0 else float("nan")
        else:
            z = float("nan")
        out[label] = {"n_exp": n_e, "pin_rate_exp": r_e,
                      "n_ctrl": n_c, "pin_rate_ctrl": r_c,
                      "z_vs_control": float(z),
                      "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))) if z == z else float("nan")}
    return out


# -------------------------------------------------------------- part 2: tradeability


def pin_trades(bars: pd.DataFrame, day: pd.DataFrame, expiries: pd.DataFrame,
               grid: float, decision: str, point_value: float, tick_size: float,
               contracts: int, serial_only: bool, min_dist_frac: float = 0.0) -> pd.DataFrame:
    """Trades toward the nearest strike.

    ``min_dist_frac`` gates on distance as a fraction of the half-interval: at 0.0
    every expiration trades, at 0.5 only those whose price sits at least halfway
    between two strikes. Pinning has nothing to pull when the price already sits
    on a strike, so a real effect should strengthen as this rises.
    """
    cost = round_turn_cost(point_value, tick_size, contracts)
    rows = []
    for r in expiries.itertuples():
        if serial_only and not r.serial:
            continue
        if not serial_only and r.serial:
            continue
        if r.expiry not in day.index:
            continue
        entry = price_at(bars, r.expiry, decision)
        if entry is None:
            continue
        exit_px = float(day.loc[r.expiry, "settle"])
        strike = round(entry / grid) * grid
        signed = strike - entry           # >0 means the magnet sits above
        if signed == 0:
            continue
        if abs(signed) < min_dist_frac * (grid / 2.0):
            continue
        side = 1 if signed > 0 else -1    # trade toward the strike
        pts = side * (exit_px - entry)
        rows.append({"expiry": r.expiry.date(), "serial": r.serial,
                     "entry": entry, "exit": exit_px, "strike": strike,
                     "dist": abs(signed), "side": side, "points": pts,
                     "pnl": pts * point_value * contracts - cost,
                     "gross": pts * point_value * contracts, "cost": cost})
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n": 0}
    pnl = trades["pnl"]
    t = stats.ttest_1samp(pnl, 0) if len(pnl) > 2 else None
    return {"label": label, "n": int(len(pnl)),
            "mean_points": float(trades["points"].mean()),
            "mean_gross": float(trades["gross"].mean()),
            "mean_net": float(pnl.mean()),
            "total_net": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "t_stat": float(t.statistic) if t is not None else float("nan"),
            "p_value": float(t.pvalue) if t is not None else float("nan")}


# ------------------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser(description="Golez-Jackwerth expiration pinning study")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), required=True)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-bars", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = INSTRUMENTS[args.instrument]
    bars = load_rth_bars(cfg)
    day = daily_settlement(bars, args.min_bars)
    expiries = expiration_days(day.index)
    expiries = expiries[expiries["expiry"].isin(day.index)].reset_index(drop=True)

    print(f"{args.instrument.upper()}  sessions={len(day)}  "
          f"{day.index.min().date()} .. {day.index.max().date()}")
    print(f"  expirations: {int(expiries.serial.sum())} serial, "
          f"{int((~expiries.serial).sum())} quarterly\n")

    report: dict = {"instrument": args.instrument, "sessions": int(len(day)),
                    "span": [str(day.index.min().date()), str(day.index.max().date())],
                    "n_serial": int(expiries.serial.sum()),
                    "n_quarterly": int((~expiries.serial).sum()),
                    "clustering": [], "trades": []}

    print("=" * 78)
    print("PART 1 -- CLUSTERING (paper Eq. 1-2): settlement within band of ATM strike")
    print("=" * 78)
    for grid in cfg["grids"]:
        for band in (0.25, 0.375, 0.5):
            scaled = band * (grid / 5.0)        # paper band is 0.375 on a 5-pt grid
            c = clustering_table(day, expiries, grid, scaled, args.window)
            report["clustering"].append(c)
            s, q = c["serial"], c["quarterly"]
            print(f"grid={grid:>5.1f} band=+/-{scaled:>5.2f} (uniform null {c['uniform_null']:.1%})")
            print(f"   SERIAL    exp {s['pin_rate_exp']:6.1%} (n={s['n_exp']:3d}) vs "
                  f"ctrl {s['pin_rate_ctrl']:6.1%} (n={s['n_ctrl']:4d})   z={s['z_vs_control']:+5.2f}  p={s['p_value']:.3f}")
            print(f"   QUARTERLY exp {q['pin_rate_exp']:6.1%} (n={q['n_exp']:3d}) vs "
                  f"ctrl {q['pin_rate_ctrl']:6.1%} (n={q['n_ctrl']:4d})   z={q['z_vs_control']:+5.2f}  p={q['p_value']:.3f}   [placebo: expect flat]")

    print()
    print("=" * 78)
    print(f"PART 2 -- TRADEABILITY: enter toward nearest strike, exit at settlement")
    print(f"           {args.contracts} contracts, cost "
          f"${round_turn_cost(cfg['point_value'], cfg['tick_size'], args.contracts):.2f}/round-turn")
    print("=" * 78)
    dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)
    for grid in cfg["grids"]:
        max_capture = grid / 2.0 * cfg["point_value"] * args.contracts
        print(f"\ngrid={grid:.1f}  (theoretical max capture = half a grid interval "
              f"= ${max_capture:.2f} gross)")
        for decision in ("13:00", "14:00", "15:00"):
            tr = pin_trades(bars, day, expiries, grid, decision, cfg["point_value"],
                            cfg["tick_size"], args.contracts, serial_only=True)
            pl = pin_trades(bars, day, expiries, grid, decision, cfg["point_value"],
                            cfg["tick_size"], args.contracts, serial_only=False)
            s = summarise(tr, f"serial@{decision}")
            p = summarise(pl, f"quarterly@{decision}")
            dev = summarise(tr[pd.to_datetime(tr["expiry"]) <= dev_end.tz_localize(None)]
                            if not tr.empty else tr, f"dev@{decision}")
            val = summarise(tr[pd.to_datetime(tr["expiry"]) > dev_end.tz_localize(None)]
                            if not tr.empty else tr, f"val@{decision}")
            for row in (s, p, dev, val):
                row["grid"] = grid
                report["trades"].append(row)
            if s["n"]:
                print(f"  T={decision}  SERIAL    n={s['n']:3d}  "
                      f"gross=${s['mean_gross']:+7.2f}  net=${s['mean_net']:+7.2f}  "
                      f"t={s['t_stat']:+5.2f}  WR={s['win_rate']:5.1%}  total=${s['total_net']:+9.0f}")
                print(f"           dev n={dev['n']:3d} net=${dev.get('mean_net',0):+7.2f} "
                      f"t={dev.get('t_stat',float('nan')):+5.2f}   |   "
                      f"val n={val['n']:3d} net=${val.get('mean_net',0):+7.2f} "
                      f"t={val.get('t_stat',float('nan')):+5.2f}")
            if p["n"]:
                print(f"           QUARTERLY n={p['n']:3d}  net=${p['mean_net']:+7.2f}  "
                      f"t={p['t_stat']:+5.2f}   [placebo]")

    print()
    print("=" * 78)
    print("PART 3 -- DISTANCE SWEEP: pinning should strengthen the further price")
    print("           sits from the strike it is supposedly drawn to (serial only)")
    print("=" * 78)
    report["distance_sweep"] = []
    for grid in cfg["grids"]:
        print(f"\ngrid={grid:.1f}")
        for decision in ("13:00", "14:00", "15:00"):
            line = [f"  T={decision} "]
            for frac in (0.0, 0.25, 0.5, 0.75):
                tr = pin_trades(bars, day, expiries, grid, decision, cfg["point_value"],
                                cfg["tick_size"], args.contracts, serial_only=True,
                                min_dist_frac=frac)
                r = summarise(tr, f"grid{grid}@{decision}_d{frac}")
                r["grid"] = grid
                r["min_dist_frac"] = frac
                report["distance_sweep"].append(r)
                if r["n"]:
                    line.append(f"d>={frac:.2f}: n={r['n']:3d} net=${r['mean_net']:+7.2f} t={r['t_stat']:+5.2f}  ")
                else:
                    line.append(f"d>={frac:.2f}: n=  0                      ")
            print("".join(line))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
