"""Does the pre-decision drift that works for the FOMC also work for the ECB?

WHY THIS TEST, AND WHY THIS ONE FIRST
-------------------------------------
`fomc_drift_v1` is the only strategy in this project that passed dev AND a sealed
holdout. Its documented limitation is velocity, not size: 8 events a year, and
that is what sets the portfolio's time-to-pass. The Track B firm survey concluded
that nothing clears 85% in 16 weeks at any firm, and that single-event risk caps
the book at 1 micro on its highest-velocity leg -- roughly $132-232/wk against
the $375-500/wk the deadline needs.

More events at the SAME per-event risk is therefore the lever that fits the
stated constraint. Size is not available; velocity might be.

The ledger's refined mechanism claim is that the drift attaches to POLICY
DECISIONS, not to data releases -- FOMC decisions pay, FOMC minutes do not
(t=+0.60), CPI is regime-bound, NFP/ISM/GDP are dead. The ECB Governing Council
monetary policy decision is a policy decision by that definition, and it is the
largest such event outside the FOMC. It was listed in the ledger as an untested
extension candidate.

HONEST PRIOR: weaker than the FOMC and quite possibly zero. Hu-Pan-Wang-Zhu's
mechanism is a premium for holding an asset into the resolution of uncertainty
that is MATERIAL TO THAT ASSET. Euro-area monetary policy is material to European
equities and only second-order for a US tech index. If this pays at all it should
pay less than the FOMC, and a null is the expected outcome.

DECLARED BEFORE RUNNING
-----------------------
* PRIMARY, confirmatory: ECB decisions on MNQ, one window, stated below.
* The Bank of England is a pre-declared SECONDARY and is NOT run here: it needs
  its own URL-verified calendar, and running it later off the back of an ECB
  result would be a forking path. The Bank of Japan is excluded outright because
  its announcement time is not fixed, so "exit five minutes before the release"
  is undefined for it.
* Success requires the EVENT MINUS CONTROL difference to reach t >= 2.5, with
  dev and val eras agreeing in sign. The level alone is not evidence: this
  project has already established that the plain overnight window pays
  +$34.51/day t=+2.50 on MNQ, so about two thirds of any raw ECB number would be
  nothing but being long overnight. The pre-registered placebo is what saved the
  HPWZ screen from a false replication, and it is what decides this one.

THE WINDOW
----------
Entry  : 18:00 ET on the last Globex open before the decision. That is after the
         mandatory 16:45 ET flatten, so the position is legal to hold -- the same
         correction that made `fomc_drift_v1` tradeable as a partial window.
Exit   : five minutes before the ECB press release.

Release time is read PER DATE from the calendar in Europe/Berlin local time and
converted to ET, because (a) the ECB moved publication from 13:45 to 14:15 CET on
2022-07-21, and (b) Frankfurt and New York change to and from daylight saving on
different dates, so any fixed ET hour is wrong for several meetings a year.

CONTROL
-------
The same clock window on non-event days, MATCHED ON WEEKDAY. ECB decisions are
almost all Thursdays; an unmatched control would pull in Mondays, which carry the
weekend premium this project has separately measured, and would bias the
comparison. FOMC days are excluded from the control so the incumbent edge cannot
leak into the baseline.

Usage:
    python3 rule_based_v1/validation/research_global_cb_drift.py \
        --out runs/ecb_drift.json
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
    """1-minute closes on an ET index, deduplicated and sorted."""
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def globex_opens(px: pd.Series, open_hour: int = 18,
                 window_min: int = 10) -> pd.DatetimeIndex:
    """First traded bar at or just after the Globex reopen, one per session.

    NOT an exact `hour == 18 and minute == 0` match. This tape stamps 1-minute
    bars at their CLOSE, so the first bar of a session that reopens at 18:00 ET
    is 18:01, and there is frequently no 18:00 bar at all. Matching exactly finds
    5 opens in six years instead of ~1,600. That failure mode is recorded in the
    project ledger and is easy to walk into twice.

    Searching a short window after the reopen and taking the first bar per
    calendar day also makes weekends and holidays resolve themselves: the entry
    is simply the most recent reopen that exists in the data before the exit.
    """
    mins = px.index.hour * 60 + px.index.minute
    lo = open_hour * 60
    sub = px.index[(mins >= lo) & (mins <= lo + window_min)]   # sorted, tz-aware
    if len(sub) == 0:
        return pd.DatetimeIndex([])
    first_of_day = ~pd.Index(sub.date).duplicated(keep="first")
    return sub[first_of_day]


def exit_time_et(day: pd.Timestamp, release_local: str, tz: str,
                 lead_minutes: int) -> pd.Timestamp:
    """Release instant minus `lead_minutes`, resolved in the issuer's local tz.

    `day` is a tz-naive calendar date. The release is localised in the issuing
    central bank's timezone and only then converted to ET, so the US/Europe
    daylight-saving offset is handled per date rather than assumed.
    """
    hh, mm = (int(x) for x in release_local.split(":"))
    local = pd.Timestamp(day.date()).replace(hour=hh, minute=mm)
    local = local.tz_localize(tz)
    return local.tz_convert(ET) - pd.Timedelta(minutes=lead_minutes)


def drift_trades(px: pd.Series, opens: pd.DatetimeIndex,
                 events: pd.DataFrame, pv: float, contracts: int,
                 cost: float, lead_minutes: int) -> pd.DataFrame:
    """Long from the last Globex open before the release to release minus lead."""
    rows = []
    for _, ev in events.iterrows():
        day = pd.Timestamp(ev["date"])
        t_out = exit_time_et(day, ev["release_local"], ev["tz"], lead_minutes)

        pos = opens.searchsorted(t_out, "left") - 1
        if pos < 0:
            continue
        t_in = opens[pos]
        # A hold longer than ~30h means we skipped a session (holiday gap); the
        # window is meant to be a single overnight, so drop it rather than
        # silently turn it into a multi-day carry.
        if (t_out - t_in) > pd.Timedelta(hours=30):
            continue

        a = px[t_in:t_in + pd.Timedelta(minutes=10)]
        b = px[t_out - pd.Timedelta(minutes=30):t_out]
        if a.empty or b.empty or b.index[-1] <= a.index[-1]:
            continue
        entry, exit_px = float(a.iloc[0]), float(b.iloc[-1])
        path = px[a.index[0]:b.index[-1]]
        rows.append({
            "day": day, "weekday": day.weekday(), "entry": entry,
            "hours": (t_out - t_in).total_seconds() / 3600.0,
            "points": exit_px - entry,
            "pnl": (exit_px - entry) * pv * contracts - cost,
            "mae": float(path.min() - entry) * pv * contracts,
        })
    return pd.DataFrame(rows)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 8:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    t = stats.ttest_1samp(p, 0)
    q05 = tr["mae"].quantile(0.05)
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "bps": float((tr["points"] / tr["entry"] * 1e4).mean()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((p > 0).mean()),
            "mae_p50": float(tr["mae"].median()), "mae_p05": float(q05),
            "mae_worst": float(tr["mae"].min()),
            "eff": float(p.mean() / abs(q05)) if q05 < 0 else float("nan")}


def show(s: dict) -> None:
    if s.get("n", 0) < 8:
        print(f"  {s['label']:<30} n={s.get('n', 0)} (too few)")
        return
    print(f"  {s['label']:<30} n={s['n']:<5} ${s['mean']:+8.2f}  {s['bps']:+6.1f}bps  "
          f"t={s['t_stat']:+5.2f}  WR={s['win_rate']:5.1%}  "
          f"MAE p50/p05 ${s['mae_p50']:>6,.0f}/${s['mae_p05']:>7,.0f}  "
          f"eff={s['eff']:+.3f}")


def weekday_matched_control(ctrl: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Keep only control days whose weekday appears among the event days.

    ECB decisions are overwhelmingly Thursdays. Comparing them to an all-weekday
    baseline would put the Monday weekend premium in the control and bias the
    difference downward, and putting it in the treatment would bias it upward.
    Matching removes the question.
    """
    wd = set(events["weekday"].unique())
    return ctrl[ctrl["weekday"].isin(wd)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="ECB pre-decision drift on a US index future")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="mnq")
    ap.add_argument("--calendar", type=Path,
                    default=Path("data/processed/ecb_decisions.csv"))
    ap.add_argument("--fomc", type=Path,
                    default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--lead-minutes", type=int, default=5)
    ap.add_argument("--dev-end", default="2022-12-31")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = INSTRUMENTS[a.instrument]
    cost = 2 * a.contracts * (0.62 + 1.0 * cfg["tick"] * cfg["pv"])
    px = load_px(cfg)
    opens = globex_opens(px)

    cal = pd.read_csv(a.calendar)
    events = pd.DataFrame({
        "date": pd.to_datetime(cal["decision_date"]),
        "release_local": cal["release_time_cet"],
        "tz": cal["tz"],
    })
    events = events[(events["date"] >= px.index.min().tz_localize(None)) &
                    (events["date"] <= px.index.max().tz_localize(None))]

    # Control days: every calendar day in range that is not an ECB decision and
    # not an FOMC announcement, run through the SAME window definition.
    fomc = set(pd.to_datetime(pd.read_csv(a.fomc)["announcement_date"]).dt.date)
    ecb_days = set(events["date"].dt.date)
    all_days = pd.DatetimeIndex(sorted({t.tz_localize(None).normalize()
                                        for t in opens}))
    ctrl_days = [d for d in all_days
                 if d.date() not in ecb_days and d.date() not in fomc]

    print(f"{a.instrument.upper()}  ECB pre-decision drift, {a.contracts} contracts, "
          f"cost ${cost:.2f}")
    print(f"  tape {px.index.min().date()} .. {px.index.max().date()}   "
          f"18:00 ET opens {len(opens)}")
    print(f"  ECB decisions in range {len(events)}   control candidate days "
          f"{len(ctrl_days)}\n")

    ev_tr = drift_trades(px, opens, events, cfg["pv"], a.contracts, cost,
                         a.lead_minutes)

    # The control uses each event's own release clock, cycled across control
    # days, so treatment and control share the same exit-time distribution.
    rel = events[["release_local", "tz"]].reset_index(drop=True)
    ctrl_ev = pd.DataFrame({
        "date": ctrl_days,
        "release_local": [rel.loc[i % len(rel), "release_local"]
                          for i in range(len(ctrl_days))],
        "tz": [rel.loc[i % len(rel), "tz"] for i in range(len(ctrl_days))],
    })
    ct_tr = drift_trades(px, opens, ctrl_ev, cfg["pv"], a.contracts, cost,
                         a.lead_minutes)
    ct_tr = weekday_matched_control(ct_tr, ev_tr)

    report = {"instrument": a.instrument, "contracts": a.contracts,
              "cost": cost, "rows": []}

    print("EVENT vs CONTROL")
    for name, tr in (("ECB decision", ev_tr), ("control (wd-matched)", ct_tr)):
        s = stat(tr, name)
        report["rows"].append(s)
        show(s)

    print(f"\n  event weekdays: "
          f"{ev_tr['weekday'].value_counts().sort_index().to_dict()}  (0=Mon)")
    print(f"  median hold {ev_tr['hours'].median():.1f}h")

    if len(ev_tr) >= 8 and len(ct_tr) >= 8:
        w = stats.ttest_ind(ev_tr["pnl"], ct_tr["pnl"], equal_var=False)
        diff = float(ev_tr["pnl"].mean() - ct_tr["pnl"].mean())
        print(f"\n  ECB MINUS CONTROL: ${diff:+8.2f}  t={w.statistic:+5.2f}  "
              f"p={w.pvalue:.4f}   [bar: t >= 2.5]")
        report["minus_control"] = {"diff": diff, "t": float(w.statistic),
                                   "p": float(w.pvalue)}

    cut = pd.Timestamp(a.dev_end)
    print(f"\nERA SPLIT (dev <= {a.dev_end} / val after)")
    for era, sub_e, sub_c in (
            ("dev", ev_tr[ev_tr["day"] <= cut], ct_tr[ct_tr["day"] <= cut]),
            ("val", ev_tr[ev_tr["day"] > cut], ct_tr[ct_tr["day"] > cut])):
        s = stat(sub_e, f"ECB {era}")
        report["rows"].append(s)
        show(s)
        if len(sub_e) >= 8 and len(sub_c) >= 8:
            w = stats.ttest_ind(sub_e["pnl"], sub_c["pnl"], equal_var=False)
            print(f"      minus control: "
                  f"${sub_e['pnl'].mean() - sub_c['pnl'].mean():+8.2f}  "
                  f"t={w.statistic:+5.2f}  p={w.pvalue:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str))
    ev_tr.to_csv(a.out.with_suffix(".trades.csv"), index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
