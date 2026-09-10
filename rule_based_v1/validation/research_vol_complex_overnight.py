"""Can the Cboe vol complex tell us which NIGHTS to skip?

THE TARGET, AND WHY IT IS THE TAIL AND NOT THE MEAN
---------------------------------------------------
`vps_disaster_stop_sizing_VERDICT.md` quantified the gap precisely: the book needs
~$520/wk at a tail-legal size and earns $162/wk, and "the gap is a factor of ~3.2
in edge quality, not a tuning problem". Size is capped because a single overnight
disaster must stay inside ~60% of the MLL.

That makes the highest-value signal NOT one that raises the mean. It is one that
identifies the nights whose LEFT TAIL is fat, so they can be skipped. Cutting the
worst-case per night permits more contracts, and $/wk scales with contracts. A
signal that improves the mean and the tail equally is worthless here, because size
is set by the tail alone.

This is exactly why the ledger's realized-vol conditioning failed
(`project_edge_screens_sep2026`: "High vol is worse"): realized vol predicts
realized vol in BOTH directions, so it removes upside and downside together and
leaves the ratio unchanged.

PRIMARY HYPOTHESIS (declared before running)
--------------------------------------------
> H1. **SKEW** is the one member of the complex that prices the LEFT tail
> specifically -- it is computed from out-of-the-money puts relative to ATM. If it
> carries information, high-SKEW nights should have a worse downside tail WITHOUT
> a proportionally better mean, so skipping them raises mean/|MAE p05|.

The differentiator from the dead realized-vol test is asymmetry: SKEW is a
directional tail measure, realized vol is not. If SKEW behaves symmetrically too,
the whole family is closed and that is a clean result.

SECONDARY, exploratory, reported but not promotable on their own:
    S1 VIX9D/VIX   (near-term slope; is tonight priced above the 30-day?)
    S2 VIX3M/VIX   (term slope: contango vs backwardation)
    S3 VVIX/VIX    (vol-of-vol relative to vol)

TRIAL COUNT: 4 signals x 2 tails (skip-high / skip-low) = **8 cells**,
Bonferroni alpha = 0.05/8 = 0.00625. Declared here, all reported.

⚠️ DISCLOSURE: the VIX3M/VIX slope was already spent once, as weekend covariate H2
in `PREREG_free_data_weekend_covariates.md`, where it was falsified (sign flipped
between halves). It is carried here as exploratory ONLY, on a different target
(nightly, n~3,900 vs n=172) and can never be promoted from this run.

THE WINDOW
----------
Long from 18:00 ET (after the mandatory 16:45 flatten, so the hold is legal) to
09:30 ET the next session. That is the close-to-open window where the equity
premium is documented to accrue (Lou-Polk-Skouras; Cliff-Cooper-Gulen), truncated
at the front by the flatten rule.

CAUSALITY
---------
The signal is the vol index CLOSE on the entry day. Cboe indices settle at 16:15
ET, the entry is 18:00 ET, so the value is known before the trade. Terciles are
EXPANDING-window, so a night is ranked only against history that preceded it.

Usage:
    python3 rule_based_v1/validation/research_vol_complex_overnight.py \
        --instrument es --out runs/vol_complex_overnight_es.json
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
            "tz": "America/Chicago", "pv": 2.0, "rt_cost": 2.06},
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "tz": None, "pv": 5.0, "rt_cost": 2.49},      # MES economics
}

SIGNALS = ("skew", "vix9d_vix", "vix3m_vix", "vvix_vix")
PRIMARY = "skew"


def load_px(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_vol_complex(free_dir: Path) -> pd.DataFrame:
    def col(name: str, field: str) -> pd.Series:
        d = pd.read_parquet(free_dir / f"cboe_{name}.parquet")
        s = d[field] if field in d.columns else d.iloc[:, 0]
        s.index = pd.to_datetime(d.index)
        return s.astype(float)

    vix = col("vix", "close")
    out = pd.DataFrame({
        "vix": vix,
        "skew": col("skew", "skew"),
        "vix9d": col("vix9d", "close"),
        "vix3m": col("vix3m", "close"),
        "vvix": col("vvix", "vvix"),
    })
    out["vix9d_vix"] = out["vix9d"] / out["vix"]
    out["vix3m_vix"] = out["vix3m"] / out["vix"]
    out["vvix_vix"] = out["vvix"] / out["vix"]
    return out


def overnight_trades(px: pd.Series, pv: float, cost: float,
                     entry_h: int = 18, exit_h: float = 9.5) -> pd.DataFrame:
    """Long 18:00 ET -> 09:30 ET next session, one row per night.

    Entry is the FIRST bar at or after 18:00 ET (bars are stamped at their close
    on some feeds, so an exact 18:00 match is unreliable). Exit is the LAST bar at
    or before 09:30 ET. Nights whose span exceeds 30h are dropped as holiday gaps
    rather than silently becoming multi-day carries.
    """
    mins = px.index.hour * 60 + px.index.minute
    ent_mask = (mins >= entry_h * 60) & (mins <= entry_h * 60 + 10)
    ent_idx = px.index[ent_mask]
    ent_idx = ent_idx[~pd.Index(ent_idx.date).duplicated(keep="first")]

    exit_min = int(exit_h * 60)
    ex_mask = (mins >= exit_min - 10) & (mins <= exit_min)
    ex_all = px.index[ex_mask]
    ex_all = ex_all[~pd.Index(ex_all.date).duplicated(keep="last")]

    rows = []
    vals = px
    for t_in in ent_idx:
        pos = ex_all.searchsorted(t_in, "right")
        if pos >= len(ex_all):
            continue
        t_out = ex_all[pos]
        span = (t_out - t_in).total_seconds() / 3600.0
        if span > 30 or span < 2:
            continue
        entry, exit_px = float(vals.loc[t_in]), float(vals.loc[t_out])
        path = vals.loc[t_in:t_out]
        rows.append({
            "signal_date": pd.Timestamp(t_in.date()),
            "hours": span,
            "points": exit_px - entry,
            "pnl": (exit_px - entry) * pv - cost,
            "mae": float(path.min() - entry) * pv,
        })
    return pd.DataFrame(rows)


def expanding_tercile(x: pd.Series, min_obs: int = 250) -> pd.Series:
    """Which expanding-window tercile each value falls in: 0 low, 1 mid, 2 high.

    Strictly causal: the cut points for observation i come from observations
    before i only. NaN until `min_obs` history exists.
    """
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < min_obs or not np.isfinite(v[i]):
            continue
        hist = v[:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_obs:
            continue
        lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
        out[i] = 0 if v[i] <= lo else (2 if v[i] > hi else 1)
    return pd.Series(out, index=x.index)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 30:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    t = stats.ttest_1samp(p, 0)
    q05 = tr["mae"].quantile(0.05)
    return {
        "label": label, "n": int(len(p)), "mean": float(p.mean()),
        "t_stat": float(t.statistic), "p_value": float(t.pvalue),
        "win_rate": float((p > 0).mean()),
        "mae_p05": float(q05), "mae_worst": float(tr["mae"].min()),
        "eff": float(p.mean() / abs(q05)) if q05 < 0 else float("nan"),
    }


def show(s: dict) -> None:
    if s.get("n", 0) < 30:
        print(f"  {s['label']:<28} n={s.get('n', 0)} (too few)")
        return
    print(f"  {s['label']:<28} n={s['n']:<5} ${s['mean']:+7.2f} t={s['t_stat']:+5.2f} "
          f"WR={s['win_rate']:5.1%}  MAEp05 ${s['mae_p05']:>7,.0f} "
          f"worst ${s['mae_worst']:>8,.0f}  eff={s['eff']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="es")
    ap.add_argument("--free-dir", type=Path, default=Path("data/processed/free"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = INSTRUMENTS[a.instrument]
    px = load_px(cfg)
    tr = overnight_trades(px, cfg["pv"], cfg["rt_cost"])
    vol = load_vol_complex(a.free_dir)

    vol_flat = vol.reset_index()
    vol_flat = vol_flat.rename(columns={vol_flat.columns[0]: "signal_date"})
    sd = pd.to_datetime(vol_flat["signal_date"])
    if sd.dt.tz is not None:
        sd = sd.dt.tz_localize(None)
    vol_flat["signal_date"] = sd.dt.normalize()
    tr["signal_date"] = pd.to_datetime(tr["signal_date"]).dt.normalize()
    tr = tr.merge(vol_flat, on="signal_date", how="left")
    tr = tr.sort_values("signal_date").reset_index(drop=True)

    print(f"{a.instrument.upper()} overnight 18:00 ET -> 09:30 ET, 1 micro, "
          f"cost ${cfg['rt_cost']:.2f}")
    print(f"  nights {len(tr)}  {tr.signal_date.min().date()} .. "
          f"{tr.signal_date.max().date()}  median hold {tr.hours.median():.1f}h\n")

    report = {"instrument": a.instrument, "rows": []}
    base = stat(tr, "UNCONDITIONAL")
    report["rows"].append(base)
    print("BASELINE"); show(base)

    print(f"\nSIGNAL BUCKETS (expanding terciles, causal). "
          f"PRIMARY = {PRIMARY}; Bonferroni alpha = 0.00625")
    for sig in SIGNALS:
        sub = tr[tr[sig].notna()].copy()
        if len(sub) < 400:
            print(f"\n  {sig}: only {len(sub)} nights with data, skipped")
            continue
        sub["terc"] = expanding_tercile(sub[sig]).to_numpy()
        sub = sub[sub["terc"].notna()]
        tag = "PRIMARY" if sig == PRIMARY else "exploratory"
        print(f"\n  --- {sig}  ({tag}, n={len(sub)}) ---")
        for k, name in ((0, "low"), (1, "mid"), (2, "high")):
            s = stat(sub[sub["terc"] == k], f"{sig} {name}")
            report["rows"].append(s)
            show(s)
        for skip, name in ((2, "SKIP high (keep low+mid)"),
                           (0, "SKIP low  (keep mid+high)")):
            s = stat(sub[sub["terc"] != skip], f"{sig} {name}")
            report["rows"].append(s)
            show(s)
            if np.isfinite(s.get("eff", np.nan)) and np.isfinite(base["eff"]):
                print(f"      -> eff vs unconditional: "
                      f"{s['eff']:+.4f} vs {base['eff']:+.4f} "
                      f"({'BETTER' if s['eff'] > base['eff'] else 'worse'})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str))
    tr.to_csv(a.out.with_suffix(".nights.csv"), index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
