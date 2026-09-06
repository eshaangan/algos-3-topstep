"""Mega-cap earnings as index-level uncertainty-resolution events on MNQ.

Pre-registration: rule_based_v1/validation/vps_prereg/vps_megacap_earnings.yaml

Same shape as fomc_drift_v1, which is the only sealed-holdout survivor here, but
at 28 events/yr instead of 8. Velocity is the point: time-to-pass, not P(pass), is
the axis where the incumbent plan (weekend x1 + FOMC x2, 96.4% / 15wk) is beatable.

Every number is reported against the same clock window on non-event days, and
broken out per symbol, because an "index-level premium" that is really just NVDA
is one stock's story and does not get promoted.

Usage:
    python3 rule_based_v1/validation/research_megacap_earnings.py \
        --instrument mnq --window prior_close --out runs/megacap_mnq.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"

INSTRUMENTS = {
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts",
            "tz": "America/Chicago", "pv": 2.0, "tick": 0.25},
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "tz": None, "pv": 5.0, "tick": 0.25},
}


def load_px(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth_sessions(px: pd.Series) -> pd.DatetimeIndex:
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    g = rth.groupby(rth.index.normalize()).size()
    return pd.DatetimeIndex(g[g >= 60].index)


def trades(px: pd.Series, sessions: pd.DatetimeIndex, days: pd.DatetimeIndex,
           window: str, pv: float, contracts: int, cost: float) -> pd.DataFrame:
    rows = []
    for d in days:
        pos = sessions.searchsorted(d, "left")
        if pos >= len(sessions) or sessions[pos] != d:
            continue
        if window == "prior_close":
            if pos == 0:
                continue
            t_in = sessions[pos - 1] + pd.Timedelta(hours=16)
        else:                                    # rth_only
            t_in = d + pd.Timedelta(hours=9, minutes=30)
        t_out = d + pd.Timedelta(hours=15, minutes=55)
        a = px[t_in - pd.Timedelta(minutes=20):t_in + pd.Timedelta(minutes=10)]
        b = px[t_out - pd.Timedelta(minutes=30):t_out]
        if a.empty or b.empty or b.index[-1] <= a.index[0]:
            continue
        entry, exit_px = float(a.iloc[-1] if window == "prior_close" else a.iloc[0]), float(b.iloc[-1])
        path = px[a.index[-1 if window == "prior_close" else 0]:b.index[-1]]
        rows.append({"day": d, "entry": entry, "points": exit_px - entry,
                     "pnl": (exit_px - entry) * pv * contracts - cost,
                     "mae": float(path.min() - entry) * pv * contracts})
    return pd.DataFrame(rows)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 10:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    t = stats.ttest_1samp(p, 0)
    bps = (tr["points"] / tr["entry"] * 1e4).mean()
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "bps": float(bps), "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((p > 0).mean()),
            "mae_p50": float(tr["mae"].median()), "mae_p05": float(tr["mae"].quantile(0.05)),
            "mae_worst": float(tr["mae"].min()),
            "eff": float(p.mean() / abs(tr["mae"].quantile(0.05)))
            if tr["mae"].quantile(0.05) < 0 else float("nan")}


def show(s: dict, tag: str = "") -> None:
    if s.get("n", 0) < 10:
        print(f"  {s['label']:<30} n={s.get('n', 0)} (too few)")
        return
    print(f"  {s['label']:<30} n={s['n']:<5} ${s['mean']:+7.2f}  {s['bps']:+6.1f}bps  "
          f"t={s['t_stat']:+5.2f}  WR={s['win_rate']:5.1%}  "
          f"MAEp05 ${s['mae_p05']:>7,.0f}  eff={s['eff']:.3f}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="mega-cap earnings drift on index futures")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="mnq")
    ap.add_argument("--earnings", type=Path, default=Path("data/processed/megacap_earnings.csv"))
    ap.add_argument("--window", choices=("prior_close", "rth_only"), default="prior_close")
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = INSTRUMENTS[args.instrument]
    cost = 2 * args.contracts * (0.62 + 1.0 * cfg["tick"] * cfg["pv"])
    px = load_px(cfg)
    sess = rth_sessions(px)

    ea = pd.read_csv(args.earnings)
    ea["report_date"] = pd.to_datetime(ea["report_date"]).dt.tz_localize(ET)
    ea = ea[(ea["report_date"] >= sess.min()) & (ea["report_date"] <= sess.max())]
    ev_days = pd.DatetimeIndex(sorted(set(ea["report_date"])))
    ev_days = ev_days.intersection(sess)
    ctrl_days = sess.difference(ev_days)

    print(f"{args.instrument.upper()}  window={args.window}  {args.contracts} contracts  cost ${cost:.2f}")
    print(f"  sessions {len(sess)}  {sess.min().date()} .. {sess.max().date()}")
    print(f"  earnings reports {len(ea)} -> {len(ev_days)} distinct event days "
          f"(clustering: {len(ea) / max(len(ev_days), 1):.2f} reports/day)")
    print(f"  control days {len(ctrl_days)}\n")

    ev = trades(px, sess, ev_days, args.window, cfg["pv"], args.contracts, cost)
    ct = trades(px, sess, ctrl_days, args.window, cfg["pv"], args.contracts, cost)
    report = {"instrument": args.instrument, "window": args.window, "rows": []}

    dev_end = pd.Timestamp("2022-12-31", tz=ET)
    print("EVENT vs CONTROL")
    for era, a, b in (("FULL", sess.min(), sess.max()),
                      ("dev 2020-2022", sess.min(), dev_end),
                      ("val 2023-2026", dev_end, sess.max())):
        e = ev[(ev["day"] > a) & (ev["day"] <= b)] if era != "FULL" else ev
        c = ct[(ct["day"] > a) & (ct["day"] <= b)] if era != "FULL" else ct
        se, sc = stat(e, f"{era} EVENT"), stat(c, f"{era} control")
        report["rows"] += [se, sc]
        show(se)
        show(sc)
        if se.get("n", 0) >= 10 and sc.get("n", 0) >= 10:
            w = stats.ttest_ind(e["pnl"], c["pnl"], equal_var=False)
            print(f"  {'-> event - control':<30} ${e['pnl'].mean() - c['pnl'].mean():+7.2f}"
                  f"          t={w.statistic:+5.2f}  p={w.pvalue:.3f}")
        print()

    print("PER SYMBOL (an index-level premium must not be one name)")
    sym_of = ea.groupby("report_date")["symbol"].apply(lambda s: ",".join(sorted(set(s))))
    ev2 = ev.assign(syms=ev["day"].map(sym_of))
    for sym in sorted({s for v in sym_of for s in v.split(",")}):
        sub = ev2[ev2["syms"].str.contains(sym, na=False)]
        s = stat(sub, sym)
        report["rows"].append(s)
        show(s)
    print()
    for sym in sorted({s for v in sym_of for s in v.split(",")}):
        sub = ev2[~ev2["syms"].str.contains(sym, na=False)]
        s = stat(sub, f"ALL EXCEPT {sym}")
        report["rows"].append(s)
        show(s, tag="  <- drop-one" if s.get("n", 0) >= 10 else "")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
