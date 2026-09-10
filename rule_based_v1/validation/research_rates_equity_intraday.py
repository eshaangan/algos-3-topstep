"""Does an intraday move in Treasury futures predict MNQ over the next 1-2 hours?

WHY THIS FAMILY, AND WHY THIS HORIZON
-------------------------------------
`runs/feasible_region.json` showed the only arithmetically open zone for this
account is **60-120 minutes traded continuously**, needing an annualised Sharpe of
about 2.0. Everything shorter is eaten by the $2.06 round trip; everything longer,
including the overnight book, is excluded outright. The ledger has spent ~4
configurations in that zone against ~80 elsewhere.

Within the zone, cross-asset CONDITIONING is the one branch still open:
`VERDICT_mnq_mes_spread.md` closed hedged *execution* on cost grounds, but reading
another instrument to gate an outright MNQ trade adds no cost at all.

MECHANISM (declared before running)
-----------------------------------
NQ is the longest-duration equity index -- growth and technology cash flows sit
far out, so its present value is the most rate-sensitive of the majors. A move in
ZN (10-year note) is a move in the discount rate. If equities do not fully reprice
that within the minute, a ZN move over the last L minutes should predict MNQ over
the next H minutes, with a POSITIVE sign: ZN up = yields down = NQ up.

HONEST PRIOR: weak. Index futures are efficient and this link is arbitraged in
seconds, not hours. The reason to run it anyway is that it is the last open branch
in the only open zone, and it is cheap.

THE CONTROL THAT DECIDES IT
---------------------------
ZN could simply be proxying risk-on/risk-off, which is already in MNQ's own recent
move. So the confirmatory statistic is NOT the raw correlation of ZN with forward
MNQ. It is the PARTIAL correlation of ZN with forward MNQ, controlling for MNQ's
own return over the same lookback. If ZN adds nothing beyond MNQ's own momentum,
the family is closed.

GRID, declared in full: L in {15, 30, 60} x H in {60, 120} = **6 cells**.
Bonferroni alpha = 0.05/6 = 0.0083. Dev 2020-2023, val 2024-2026.

Observations are sampled every H minutes (non-overlapping) and never straddle a
session, so the reported t-statistics are not autocorrelation-inflated.

ALIGNMENT: verified by lag scan on 2024 RTH -- MNQ and ZN peak at lag 0
(corr +0.0318), so both tapes are close-stamped and no shift is needed. This check
is mandatory here after the MNQ/ES bar-label bug found on 2026-09-09, where the
two tapes were off by one minute and the naive correlation read 0.014.

Usage:
    python3 rule_based_v1/validation/research_rates_equity_intraday.py \
        --out runs/rates_equity_intraday.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"
LOOKBACKS = (15, 30, 60)
HORIZONS = (60, 120)


def load_mnq(path: str) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = (pd.to_datetime(raw["ts"])
           .dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
           .dt.tz_convert(ET))
    s = pd.Series(raw["close"].to_numpy(float), index=idx)
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_zn(path: str) -> pd.Series:
    s = pd.read_parquet(path)["close"]
    s.index = pd.to_datetime(s.index).tz_convert(ET)
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_es(path: str) -> pd.Series:
    """ES front-month ETH, shifted +1 min onto MNQ's close-stamped clock.

    MNQ stamps bars at their CLOSE and ES at their OPEN, so an ES bar labelled T
    covers the same minute as the MNQ bar labelled T+1. Without this the measured
    1-minute correlation is 0.014 instead of 0.941. See VERDICT_mnq_mes_spread.md.
    """
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw["et"]) + pd.Timedelta(minutes=1)
    s = pd.Series(raw["close"].to_numpy(float), index=idx)
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth(df: pd.DataFrame) -> pd.DataFrame:
    m = df.index.hour * 60 + df.index.minute
    return df[(m >= 9 * 60 + 30) & (m <= 16 * 60)]


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Correlation of x with y after removing z from both, plus its p-value."""
    def resid(a: np.ndarray) -> np.ndarray:
        b = np.polyfit(z, a, 1)
        return a - (b[0] * z + b[1])
    rx, ry = resid(x), resid(y)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def build(mnq: pd.Series, zn: pd.Series, look: int, hor: int) -> pd.DataFrame:
    """Non-overlapping samples: ZN and MNQ lookback returns, MNQ forward return."""
    both = rth(pd.DataFrame({"nq": mnq, "zn": zn}).dropna())
    rows = []
    for _, g in both.groupby(both.index.normalize()):
        v_nq = g["nq"].to_numpy(float)
        v_zn = g["zn"].to_numpy(float)
        n = len(v_nq)
        # start after `look` bars exist, step by `hor` so samples never overlap
        for i in range(look, n - hor, hor):
            if v_nq[i - look] <= 0 or v_zn[i - look] <= 0:
                continue
            rows.append({
                "ts": g.index[i],
                "zn_look": v_zn[i] / v_zn[i - look] - 1.0,
                "nq_look": v_nq[i] / v_nq[i - look] - 1.0,
                "nq_fwd": v_nq[i + hor] / v_nq[i] - 1.0,
                "nq_fwd_pts": v_nq[i + hor] - v_nq[i],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnq", default="data/processed/mnq_1m_all.parquet")
    ap.add_argument("--zn", default="data/processed/zn_1min.parquet")
    ap.add_argument("--other", choices=("zn", "es"), default="zn",
                    help="second instrument used as the conditioning signal")
    ap.add_argument("--es", default="data/processed/es_1min_eth_frontmonth.parquet")
    ap.add_argument("--dev-end", default="2023-12-31")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    mnq = load_mnq(a.mnq)
    zn = load_es(a.es) if a.other == "es" else load_zn(a.zn)
    print(f"conditioning instrument: {a.other.upper()}")
    print(f"MNQ {mnq.index.min().date()}..{mnq.index.max().date()}   "
          f"ZN {zn.index.min().date()}..{zn.index.max().date()}")
    print(f"grid: {len(LOOKBACKS) * len(HORIZONS)} cells, "
          f"Bonferroni alpha = {0.05 / (len(LOOKBACKS) * len(HORIZONS)):.4f}\n")

    print(f"{'look':>5} {'hor':>5} {'n':>7} {'corr(ZN,fwd)':>13} {'p':>8} "
          f"{'PARTIAL|nq_own':>15} {'p':>8} {'corr(nq_own,fwd)':>17} {'dev/val sign':>13}")

    cut = pd.Timestamp(a.dev_end, tz=ET)
    rows = []
    for look in LOOKBACKS:
        for hor in HORIZONS:
            d = build(mnq, zn, look, hor)
            if len(d) < 200:
                continue
            r_raw, p_raw = stats.pearsonr(d["zn_look"], d["nq_fwd"])
            r_par, p_par = partial_corr(d["zn_look"].to_numpy(),
                                        d["nq_fwd"].to_numpy(),
                                        d["nq_look"].to_numpy())
            r_own, _ = stats.pearsonr(d["nq_look"], d["nq_fwd"])

            dev, val = d[d.ts <= cut], d[d.ts > cut]
            sd = sv = np.nan
            if len(dev) > 100 and len(val) > 100:
                sd = stats.pearsonr(dev["zn_look"], dev["nq_fwd"])[0]
                sv = stats.pearsonr(val["zn_look"], val["nq_fwd"])[0]
            agree = ("agree" if np.isfinite(sd) and np.isfinite(sv)
                     and np.sign(sd) == np.sign(sv) else "DISAGREE")

            print(f"{look:>5} {hor:>5} {len(d):>7} {r_raw:>13.4f} {p_raw:>8.3f} "
                  f"{r_par:>15.4f} {p_par:>8.3f} {r_own:>17.4f} {agree:>13}")
            rows.append({"lookback": look, "horizon": hor, "n": len(d),
                         "corr_zn_fwd": r_raw, "p_raw": p_raw,
                         "partial_corr": r_par, "p_partial": p_par,
                         "corr_own_fwd": r_own,
                         "dev_corr": sd, "val_corr": sv, "sign": agree})

    print("\nCONFIRMATORY statistic is PARTIAL|nq_own. A raw correlation that "
          "vanishes\nunder the control means ZN is only proxying MNQ's own move.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
