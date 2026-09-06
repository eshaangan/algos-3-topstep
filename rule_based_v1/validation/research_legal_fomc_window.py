"""Is there a LEGAL parameterisation of fomc_drift_v1? (prereg: vps_legal_fomc_window.yaml)

The ledger records fomc_drift_v1 as ILLEGAL. That verdict was reached by checking the
14:00 and 16:00 ET prior-session entries, both of which sit BEFORE the mandatory 16:45 ET
flatten and are force-closed. But tzguard.check_window_legal() returns LEGAL for any
entry after 16:45 whose exit precedes the next one -- so 17:00/18:00/19:00/20:00 ET ->
13:55 next day are all tradeable. The family was never dead; two of its settings were.

This evaluates the legal half of the drift as a standalone strategy, with the same
machinery that killed ISM and mega-cap earnings: a same-clock-window control on
non-event session pairs, an era split requiring both halves, and MAE efficiency.

Usage:
    python3 rule_based_v1/validation/research_legal_fomc_window.py --out runs/legal_fomc.json
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
from tzguard import assert_et_index, check_window_legal, load_account_rules

ET = "America/New_York"

INSTRUMENTS = {
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts",
            "tz": "America/Chicago", "pv": 2.0, "tick": 0.25, "comm": 0.62, "vol": "vol",
            "dev_end": "2023-12-31"},
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "tz": None, "pv": 5.0, "tick": 0.25, "comm": 0.62, "vol": "volume",
           "dev_end": "2017-12-31"},
}
LEGAL_HOURS = [17.0, 18.0, 19.0, 20.0]
PRIMARY_HOUR = 18.0


def load_px(cfg: dict) -> tuple[pd.Series, pd.Series]:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    idx = idx.dt.tz_convert(ET)
    keep = ~idx.isna()
    px = pd.Series(raw["close"].to_numpy(float)[keep], index=idx[keep]).sort_index()
    vol = pd.Series(raw[cfg["vol"]].to_numpy(float)[keep], index=idx[keep]).sort_index()
    px, vol = px[~px.index.duplicated(keep="last")], vol[~vol.index.duplicated(keep="last")]
    assert_et_index(px.index, vol, label=cfg["path"])          # trap the tz landmine
    return px, vol


def rth_sessions(px: pd.Series) -> pd.DatetimeIndex:
    """Trade dates with a real RTH session -- the definition of 'previous SESSION'."""
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    g = rth.groupby(rth.index.normalize()).size()
    return pd.DatetimeIndex(g[g >= 60].index)


def hold(px: pd.Series, t_in: pd.Timestamp, t_out: pd.Timestamp,
         pv: float, contracts: int, cost: float) -> dict | None:
    """Long from t_in to t_out, marked at real executable bars, with the path MAE."""
    a = px[t_in:t_in + pd.Timedelta(minutes=25)]        # session opens: :00 bar may not exist
    b = px[t_out - pd.Timedelta(minutes=30):t_out]
    if a.empty or b.empty or b.index[-1] <= a.index[0]:
        return None
    entry, exit_px = float(a.iloc[0]), float(b.iloc[-1])
    path = px[a.index[0]:b.index[-1]]
    return {"entry": entry, "points": exit_px - entry,
            "pnl": (exit_px - entry) * pv * contracts - cost,
            "mae": float(path.min() - entry) * pv * contracts}


def build(px: pd.Series, sessions: pd.DatetimeIndex, event_days: pd.DatetimeIndex,
          hour: float, cfg: dict, contracts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event trades and the same-clock-window control on non-event session pairs."""
    cost = 2 * contracts * (cfg["comm"] + cfg["tick"] * cfg["pv"])
    ev_set = set(event_days)
    # a control day must be clean: neither it nor its prior session is event-adjacent
    contaminated = set()
    for d in event_days:
        pos = sessions.searchsorted(d, "left")
        for k in (-1, 0, 1):
            if 0 <= pos + k < len(sessions):
                contaminated.add(sessions[pos + k])
    ev_rows, ctl_rows = [], []
    for pos in range(1, len(sessions)):
        d, prev = sessions[pos], sessions[pos - 1]
        t_in = prev + pd.Timedelta(hours=hour)
        t_out = d + pd.Timedelta(hours=13, minutes=55)
        r = hold(px, t_in, t_out, cfg["pv"], contracts, cost)
        if r is None:
            continue
        r["day"] = d
        (ev_rows if d in ev_set else (ctl_rows if d not in contaminated else [])).append(r)
    return pd.DataFrame(ev_rows), pd.DataFrame(ctl_rows)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 8:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    q05 = float(tr["mae"].quantile(0.05))
    t = stats.ttest_1samp(p, 0)
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "bps": float((tr["points"] / tr["entry"] * 1e4).mean()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((p > 0).mean()),
            "mae_p50": float(tr["mae"].median()), "mae_p05": q05,
            "mae_worst": float(tr["mae"].min()),
            "eff": float(p.mean() / abs(q05)) if q05 < 0 else float("nan")}


def show(s: dict, indent: str = "  ") -> None:
    if s.get("n", 0) < 8:
        print(f"{indent}{s['label']:<30} n={s.get('n', 0)} (too few)")
        return
    print(f"{indent}{s['label']:<30} n={s['n']:<5} ${s['mean']:+8.2f} {s['bps']:+6.1f}bps "
          f"t={s['t_stat']:+5.2f} WR={s['win_rate']:5.1%} "
          f"MAE p50/p05/worst ${s['mae_p50']:>6,.0f}/${s['mae_p05']:>7,.0f}/${s['mae_worst']:>7,.0f} "
          f"eff={s['eff']:+.3f}")


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser(description="legal (post-flatten) FOMC drift window")
    ap.add_argument("--fomc", type=Path, default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rules = load_account_rules()
    print(f"\naccount={rules['account']}  target=${rules['profit_target']:,.0f}  "
          f"MLL=${rules['max_loss_limit']:,.0f}  flatten={rules['flatten_et']} ET")
    print("\nLEGALITY (tzguard.check_window_legal, entry -> 13:55 next day):")
    for h in [14.0, 16.0] + LEGAL_HOURS:
        e = f"{int(h):02d}:00"
        ok, why = check_window_legal(e, "13:55", 1, rules)
        print(f"  {e} -> 13:55   {'LEGAL  ' if ok else 'ILLEGAL'}  {why}")

    fomc = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(args.fomc)["announcement_date"])).normalize()
    print(f"\nFOMC announcement dates: n={len(fomc)}  "
          f"{fomc.min().date()} -> {fomc.max().date()}")

    out: dict = {"prereg": "vps_legal_fomc_window.yaml", "contracts": args.contracts,
                 "cells": {}}
    for name, cfg in INSTRUMENTS.items():
        print(f"\n{'=' * 100}\n{name.upper()}  ({args.contracts} contracts, "
              f"pv={cfg['pv']})\n{'=' * 100}")
        px, vol = load_px(cfg)
        sessions = rth_sessions(px)
        # sessions are tz-aware ET midnights; the calendar is naive dates. Compare on
        # the naive calendar date, or every event silently fails to match.
        fomc_dates = {d.date() for d in fomc}
        ev_days = pd.DatetimeIndex([d for d in sessions if d.date() in fomc_dates])
        print(f"  sessions={len(sessions)}  FOMC sessions in tape={len(ev_days)}")
        dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)

        for h in LEGAL_HOURS:
            ev, ctl = build(px, sessions, ev_days, h, cfg, args.contracts)
            if len(ev) < 8:
                print(f"\n  entry {int(h):02d}:00 -> too few events ({len(ev)})")
                continue
            tag = "PRIMARY" if h == PRIMARY_HOUR else "shape   "
            print(f"\n  --- entry {int(h):02d}:00 ET prior session -> 13:55 ET "
                  f"decision day   [{tag}] ---")
            s_ev, s_ct = stat(ev, "event"), stat(ctl, "control (non-event)")
            show(s_ev); show(s_ct)
            tstat, pval = welch(ev["pnl"], ctl["pnl"])
            diff = float(ev["pnl"].mean() - ctl["pnl"].mean())
            print(f"    {'EVENT MINUS CONTROL':<30} ${diff:+8.2f}  t={tstat:+5.2f}  p={pval:.4f}")

            eras = {}
            for era, mask in (("dev", ev["day"] <= dev_end), ("val", ev["day"] > dev_end)):
                sub = ev[mask]
                eras[era] = stat(sub, f"{era} ({len(sub)})")
                show(eras[era], indent="      ")
            out["cells"][f"{name}_h{int(h)}"] = {
                "event": s_ev, "control": s_ct, "diff": diff,
                "diff_t": tstat, "diff_p": pval, "eras": eras}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}")
    print(f"\nBonferroni bar for {len(LEGAL_HOURS) * len(INSTRUMENTS)} cells: t >= 2.90")


if __name__ == "__main__":
    main()
