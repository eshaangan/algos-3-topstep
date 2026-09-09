#!/usr/bin/env python3
"""
Execute the pre-registered free-data covariate tests on `weekend_hold_v1`.

Protocol is fixed in `ml_intraday_v3/PREREG_free_data_weekend_covariates.md` and
must not be varied after seeing results. Three covariates, Bonferroni alpha of
0.05/4 = 0.0125 (three pre-registered tests plus one already spent on realized
volatility), and a three-part decision rule whose binding condition is that the
covariate must still work with 2020 and 2026 removed.

Usage:
    python -m ml_intraday_v3.diagnostics.test_weekend_covariates \
        --out ml_intraday_v3/results/edge_screens/weekend_covariates.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, "rule_based_v1/validation")

from ml_intraday_v3.analysis.alpha_beta import ols_hac            # noqa: E402
from ml_intraday_v3.data.free_sources import (                    # noqa: E402
    fetch_vol_complex, cot_positioning, fetch_coinbase_hourly,
)

ET = "America/New_York"
ALPHA = 0.05 / 4          # pre-registered Bonferroni threshold


def build_weekend_trades(bars: Path) -> pd.DataFrame:
    """The canonical weekend book: Sun 18:00 -> Mon 15:59 ET, 1 MNQ micro, net."""
    from research_portfolio_mc import load_px, weekend_trades, sized
    px = load_px(bars, "ts", "America/Chicago")
    wk = weekend_trades(px)
    tr = sized(wk, 1, 2.0, 0.25, 0.62)
    tr["monday"] = pd.to_datetime(wk["day"].values)
    tr["sunday"] = tr["monday"] - pd.Timedelta(days=1)
    tr["year"] = tr["monday"].dt.year
    return tr


def _expanding_z(s: pd.Series, min_obs: int = 30) -> pd.Series:
    """Z-score against history only -- never includes the current observation."""
    m = s.shift(1).expanding(min_obs).mean()
    v = s.shift(1).expanding(min_obs).std(ddof=0)
    return (s - m) / v


def covariate_crypto(tr: pd.DataFrame) -> pd.Series:
    """
    H1: BTC log return across the futures-closure window,
    Friday 17:00 ET -> Sunday 17:00 ET (one hour before the 18:00 entry).
    """
    btc = fetch_coinbase_hourly("BTC-USD", start="2019-12-01")
    px = btc["close"]
    px.index = px.index.tz_convert(ET)
    out = []
    for sun in tr["sunday"]:
        t_end = pd.Timestamp(sun).tz_localize(ET) + pd.Timedelta(hours=17)
        t_start = t_end - pd.Timedelta(hours=48)          # Friday 17:00 ET
        a = px[t_start - pd.Timedelta(hours=2):t_start + pd.Timedelta(hours=2)]
        b = px[t_end - pd.Timedelta(hours=2):t_end]
        out.append(np.log(float(b.iloc[-1]) / float(a.iloc[0]))
                   if len(a) and len(b) else np.nan)
    return pd.Series(out, index=tr.index, name="crypto_closure_ret")


def covariate_term_structure(tr: pd.DataFrame) -> pd.Series:
    """H2: VIX3M / VIX at the Friday close preceding entry."""
    v = fetch_vol_complex()["ts_slope_3m"].dropna()
    v.index = v.index.tz_convert(ET).normalize()
    out = []
    for sun in tr["sunday"]:
        cutoff = pd.Timestamp(sun).tz_localize(ET).normalize()
        prior = v[v.index < cutoff]
        out.append(float(prior.iloc[-1]) if len(prior) else np.nan)
    return pd.Series(out, index=tr.index, name="vix_term_slope")


def covariate_cot(tr: pd.DataFrame) -> pd.Series:
    """
    H3: COT non-commercial net %OI, 52w z-score, using the most recent report
    RELEASED before entry. Report date is Tuesday; release is Friday 15:30 ET,
    so a >= 3 day lag is required before a report may be used.
    """
    c = cot_positioning("E-MINI S&P 500")["noncomm_net_pct_oi_z"].dropna()
    c.index = c.index.tz_convert(ET).normalize()
    out = []
    for sun in tr["sunday"]:
        cutoff = pd.Timestamp(sun).tz_localize(ET).normalize() - pd.Timedelta(days=3)
        prior = c[c.index <= cutoff]
        out.append(float(prior.iloc[-1]) if len(prior) else np.nan)
    return pd.Series(out, index=tr.index, name="cot_noncomm_z")


def run_one(tr: pd.DataFrame, x: pd.Series, label: str) -> dict:
    """Pre-registered test for one covariate."""
    d = pd.DataFrame({"pnl": tr["pnl"], "x": x, "year": tr["year"]}).dropna()
    z = _expanding_z(d["x"]).replace([np.inf, -np.inf], np.nan)
    d = d.assign(z=z).dropna()
    n = len(d)
    if n < 40:
        return {"covariate": label, "error": f"only {n} usable observations"}

    def fit(frame):
        if len(frame) < 20:
            return None
        f = ols_hac(frame["pnl"].to_numpy(float),
                    np.column_stack([np.ones(len(frame)), frame["z"].to_numpy(float)]),
                    ["a", "b"])
        return f.get("b")

    full = fit(d)
    half = len(d) // 2
    h1, h2 = fit(d.iloc[:half]), fit(d.iloc[half:])
    ex = fit(d[(d.year > 2020) & (d.year < 2026)])

    # Secondary: expanding-window terciles.
    lo_hi = {}
    q_lo = d["z"].shift(1).expanding(30).quantile(1 / 3)
    q_hi = d["z"].shift(1).expanding(30).quantile(2 / 3)
    bucket = pd.Series(np.where(d["z"] > q_hi, "top",
                       np.where(d["z"] < q_lo, "bottom", "mid")), index=d.index)
    for b in ("bottom", "mid", "top"):
        s = d["pnl"][bucket == b]
        lo_hi[b] = {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None}

    cond1 = full is not None and full["p"] < ALPHA
    cond2 = (h1 is not None and h2 is not None
             and np.sign(h1["coef"]) == np.sign(h2["coef"]))
    cond3 = ex is not None and ex["p"] < 0.05
    verdict = "GO_CANDIDATE" if (cond1 and cond2 and cond3) else "NO_GO"

    return {
        "covariate": label, "n": n, "verdict": verdict,
        "full": full, "first_half": h1, "second_half": h2, "excl_2020_2026": ex,
        "conditions": {"p_lt_bonferroni": bool(cond1),
                       "sign_stable_across_halves": bool(cond2),
                       "survives_excl_2020_2026": bool(cond3)},
        "terciles": lo_hi,
    }


def fmt(r: dict) -> str:
    if "error" in r:
        return f"\n{r['covariate']}: SKIPPED — {r['error']}"
    L = [f"\n{r['covariate']}  (n={r['n']})", "-" * 58]
    for k in ("full", "first_half", "second_half", "excl_2020_2026"):
        v = r[k]
        L.append(f"  {k:<18} " + ("n/a" if v is None else
                 f"b={v['coef']:>9.2f}  t={v['t']:>6.2f}  p={v['p']:.4f}"))
    t = r["terciles"]
    L.append("  terciles (mean P/L): " + "  ".join(
        f"{b}={t[b]['mean']:.1f}(n={t[b]['n']})" for b in ("bottom", "mid", "top")
        if t[b]["mean"] is not None))
    c = r["conditions"]
    L.append(f"  conditions: p<{ALPHA:.4f}={c['p_lt_bonferroni']}  "
             f"sign_stable={c['sign_stable_across_halves']}  "
             f"ex2020/26={c['survives_excl_2020_2026']}")
    L.append(f"  VERDICT: {r['verdict']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=Path,
                    default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    tr = build_weekend_trades(args.bars)
    print(f"weekend trades: {len(tr)}  "
          f"{tr['monday'].min().date()} -> {tr['monday'].max().date()}  "
          f"mean=${tr['pnl'].mean():.2f}/wk")
    print(f"\nPRE-REGISTERED: 3 covariates, Bonferroni alpha = {ALPHA:.4f}")
    print("Decision rule requires all three conditions (see PREREG doc).")

    results = []
    for fn, label in ((covariate_crypto, "H1 crypto_closure_ret"),
                      (covariate_term_structure, "H2 vix_term_slope"),
                      (covariate_cot, "H3 cot_noncomm_z")):
        try:
            r = run_one(tr, fn(tr), label)
        except Exception as exc:                       # noqa: BLE001
            r = {"covariate": label, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(fmt(r))

    gos = [r for r in results if r.get("verdict") == "GO_CANDIDATE"]
    print(f"\n{'=' * 58}\nGO candidates: {len(gos)} of 3"
          + (f" -> {[r['covariate'] for r in gos]}" if gos else ""))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"prereg": "ml_intraday_v3/PREREG_free_data_weekend_covariates.md",
                   "alpha": ALPHA, "n_trades": int(len(tr)), "results": results},
                  open(args.out, "w"), indent=2, default=str)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
