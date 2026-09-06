"""Post-release intraday continuation (prereg: vps_post_release_continuation.yaml).

Track B showed the gap is VELOCITY: at the only size whose worst trade does not breach
the MLL, the validated book makes ~$212/wk against ~$375-500 needed. weekend_hold fires
52x/yr and fomc_drift 8x/yr. Scheduled macro releases fire ~250x/yr and the window is
intraday, so it is legal at every firm including LucidFlex.

This is NOT the pre-announcement drift this project killed. That family traded the
approach TO a release; this trades the reaction AFTER one, entering 15 minutes in, once
the jump is already priced. Excluding the release instant is also what makes it a
"quiet drift" candidate -- release-instant variance is what made mega-cap earnings
unsizeable.

The control is conditioned IDENTICALLY (same clock, same sign-of-first-15min rule) on
non-event days. Without that this measures intraday momentum, not an event effect.

Usage:
    python3 rule_based_v1/validation/research_post_release_continuation.py --out runs/post_release.json
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

ET = "America/New_York"

INSTRUMENTS = {
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts", "vol": "vol",
            "tz": "America/Chicago", "pv": 2.0, "tick": 0.25, "comm": 0.62,
            "dev_end": "2023-12-31"},
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "vol": "volume", "tz": None, "pv": 5.0, "tick": 0.25, "comm": 0.62,
           "dev_end": "2017-12-31"},
}
HORIZONS = [75, 135]          # minutes after entry (entry is R+15)
PRIMARY_H = 75


def load_px(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    idx = idx.dt.tz_convert(ET)
    keep = ~idx.isna()
    px = pd.Series(raw["close"].to_numpy(float)[keep], index=idx[keep]).sort_index()
    vol = pd.Series(raw[cfg["vol"]].to_numpy(float)[keep], index=idx[keep]).sort_index()
    px, vol = px[~px.index.duplicated(keep="last")], vol[~vol.index.duplicated(keep="last")]
    assert_et_index(px.index, vol, label=cfg["path"])
    return px


def rth_sessions(px: pd.Series) -> pd.DatetimeIndex:
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    g = rth.groupby(rth.index.normalize()).size()
    return pd.DatetimeIndex(g[g >= 60].index)


def at(px: pd.Series, t: pd.Timestamp, back: int = 6) -> float | None:
    """Last executable mark at or just before t."""
    s = px[t - pd.Timedelta(minutes=back):t]
    return float(s.iloc[-1]) if len(s) else None


def trade(px: pd.Series, day: pd.Timestamp, rel_min: int, horizon: int,
          pv: float, contracts: int, cost: float) -> dict | None:
    """Sign of R->R+15 sets the side; hold R+15 -> R+15+horizon."""
    r0 = day + pd.Timedelta(minutes=rel_min)
    r1, r2 = r0 + pd.Timedelta(minutes=15), r0 + pd.Timedelta(minutes=15 + horizon)
    p0, p1, p2 = at(px, r0), at(px, r1), at(px, r2)
    if p0 is None or p1 is None or p2 is None or p1 == p0:
        return None
    side = 1.0 if p1 > p0 else -1.0
    gross = (p2 - p1) * side * pv * contracts
    path = px[r1:r2]
    if path.empty:
        return None
    # MAE is the worst excursion in the direction traded
    worst = float(path.min()) if side > 0 else float(path.max())
    return {"day": day, "side": side, "entry": p1,
            "points": (p2 - p1) * side,
            "pnl": gross - cost,                      # short net = -gross - cost
            "mae": (worst - p1) * side * pv * contracts}


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 10:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    q05 = float(tr["mae"].quantile(0.05))
    t = stats.ttest_1samp(p, 0)
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "t_stat": float(t.statistic), "win_rate": float((p > 0).mean()),
            "mae_p05": q05, "mae_worst": float(tr["mae"].min()),
            "eff": float(p.mean() / abs(q05)) if q05 < 0 else float("nan")}


def show(s: dict, ind="    ") -> None:
    if s.get("n", 0) < 10:
        print(f"{ind}{s['label']:<26} n={s.get('n', 0)} (too few)")
        return
    print(f"{ind}{s['label']:<26} n={s['n']:<5} ${s['mean']:+7.2f} t={s['t_stat']:+5.2f} "
          f"WR={s['win_rate']:5.1%} MAE p05/worst ${s['mae_p05']:>7,.0f}/${s['mae_worst']:>7,.0f} "
          f"eff={s['eff']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="post-release intraday continuation")
    ap.add_argument("--fomc", type=Path, default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--gdp", type=Path, default=Path("data/processed/bea_gdp_advance_releases.csv"))
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rules = load_firm_rules("lucid_flex_100k")
    ok, why = check_window_legal("08:45", "10:00", 0, rules)
    print(f"\nlegality (intraday 08:45 -> 10:00 at the strictest firm): "
          f"{'LEGAL' if ok else 'ILLEGAL'} -- {why}")

    fomc_dates = {pd.Timestamp(d).date() for d in
                  pd.read_csv(args.fomc)["announcement_date"]}
    gcol = pd.read_csv(args.gdp)
    gcol = gcol[[c for c in gcol.columns if "date" in c.lower()][0]]
    gdp_dates = {pd.Timestamp(d).date() for d in gcol}
    print(f"calendars: fomc={len(fomc_dates)}  gdp_advance={len(gdp_dates)}")

    out, trials = {}, 0
    for name, cfg in INSTRUMENTS.items():
        print(f"\n{'=' * 104}\n{name.upper()}  ({args.contracts} contracts)\n{'=' * 104}")
        px = load_px(cfg)
        sessions = rth_sessions(px)
        dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)
        cost = 2 * args.contracts * (cfg["comm"] + cfg["tick"] * cfg["pv"])

        families = {
            "fomc 14:00":     (14 * 60, [d for d in sessions if d.date() in fomc_dates]),
            "gdp_adv 08:30":  (8 * 60 + 30, [d for d in sessions if d.date() in gdp_dates]),
            "claims 08:30 (Thu)": (8 * 60 + 30, [d for d in sessions if d.dayofweek == 3]),
        }
        for fam, (rel_min, days) in families.items():
            ev_days = set(days)
            # control days: same clock, NOT an event of this family, and not adjacent
            excl = set()
            for d in days:
                pos = sessions.searchsorted(d, "left")
                for k in (-1, 0, 1):
                    if 0 <= pos + k < len(sessions):
                        excl.add(sessions[pos + k])
            for hz in HORIZONS:
                trials += 1
                ev = pd.DataFrame([r for d in days
                                   if (r := trade(px, d, rel_min, hz, cfg["pv"],
                                                  args.contracts, cost))])
                ctl = pd.DataFrame([r for d in sessions if d not in excl
                                    if (r := trade(px, d, rel_min, hz, cfg["pv"],
                                                   args.contracts, cost))])
                if len(ev) < 10:
                    continue
                tag = "PRIMARY" if hz == PRIMARY_H else "shape  "
                print(f"\n  --- {fam}  entry R+15 -> R+{15 + hz}min   [{tag}] ---")
                s_ev, s_ct = stat(ev, "event"), stat(ctl, "control (same clock)")
                show(s_ev); show(s_ct)
                if len(ctl) >= 10:
                    w = stats.ttest_ind(ev["pnl"], ctl["pnl"], equal_var=False)
                    diff = float(ev["pnl"].mean() - ctl["pnl"].mean())
                    print(f"      {'EVENT MINUS CONTROL':<26} ${diff:+7.2f}  "
                          f"t={w.statistic:+5.2f}  p={w.pvalue:.4f}")
                else:
                    diff, w = float("nan"), None
                eras = {}
                for era, m in (("dev", ev["day"] <= dev_end), ("val", ev["day"] > dev_end)):
                    eras[era] = stat(ev[m], f"{era} ({int(m.sum())})")
                    show(eras[era], ind="        ")
                out[f"{name}_{fam}_{hz}"] = {
                    "event": s_ev, "control": s_ct, "diff": diff,
                    "diff_t": float(w.statistic) if w is not None else None,
                    "eras": eras}

    bar = stats.norm.ppf(1 - 0.025 / max(trials, 1))
    print(f"\ntrials counted: {trials}   Bonferroni t bar: {bar:.2f}")
    out["_trials"], out["_bonferroni_t"] = trials, float(bar)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
