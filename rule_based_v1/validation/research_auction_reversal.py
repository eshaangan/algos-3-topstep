"""MOC mechanism screen: does closing-auction pressure leave a reversal fingerprint?

A PRE-PURCHASE SCREEN, not an edge hunt. Closing-auction imbalance data is not freely
available (NYSE's free tool is 3 months; TAQ and Databento both cost money and the
Databento account is locked). Before paying for it, test the mechanism's falsifiable
prediction on data we ALREADY own.

The claim behind the MOC family (Bogousslavsky & Muravyev) is that forced auction flow
pushes price into the 16:00 cash close and that this pressure is TEMPORARY -- it unwinds
once the auction clears. Index futures trade past 16:00, so if that mechanism operates in
ES/MNQ it must leave a fingerprint: the 15:50->16:00 move should partially REVERSE in
16:00->16:15, i.e. negative correlation between the two returns.

If the fingerprint is absent, no imbalance feed can rescue the family and the purchase is
not worth making. If present, that is real evidence the data is worth buying.

The placebo is essential: adjacent same-length windows earlier in the day, where no
auction occurs. Intraday returns mean-revert a little everywhere at this horizon, so a
negative correlation at 16:00 means nothing unless it EXCEEDS the placebo.

Usage:
    python3 rule_based_v1/validation/research_auction_reversal.py --out runs/auction_reversal.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tzguard import assert_et_index

ET = "America/New_York"
INST = {
    "es":  {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
            "vol": "volume", "tz": None, "pv": 5.0, "dev_end": "2017-12-31"},
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts", "vol": "vol",
            "tz": "America/Chicago", "pv": 2.0, "dev_end": "2023-12-31"},
}
# (label, push_start, push_end, unwind_end) in ET minutes. First is the auction; rest placebo.
WINDOWS = [
    ("AUCTION 15:50->16:00 / 16:00->16:15", 950, 960, 975),
    ("placebo 14:50->15:00 / 15:00->15:15", 890, 900, 915),
    ("placebo 13:50->14:00 / 14:00->14:15", 830, 840, 855),
    ("placebo 11:50->12:00 / 12:00->12:15", 710, 720, 735),
]


def load(cfg):
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    idx = idx.dt.tz_convert(ET); keep = ~idx.isna()
    px = pd.Series(raw["close"].to_numpy(float)[keep], index=idx[keep]).sort_index()
    vol = pd.Series(raw[cfg["vol"]].to_numpy(float)[keep], index=idx[keep]).sort_index()
    px, vol = px[~px.index.duplicated(keep="last")], vol[~vol.index.duplicated(keep="last")]
    assert_et_index(px.index, vol, label=cfg["path"])
    return px


def pair_returns(px, a, b, c):
    """push = a->b, unwind = b->c, marked at the last real bar at or before each edge."""
    m = px.index.hour * 60 + px.index.minute
    day = px.index.normalize()
    f = pd.DataFrame({"px": px.to_numpy(), "d": day, "m": m})
    def mark(t):
        s = f[(f.m <= t) & (f.m >= t - 6)]
        return s.groupby("d")["px"].last()
    A, B, C = mark(a), mark(b), mark(c)
    j = pd.concat({"a": A, "b": B, "c": C}, axis=1).dropna()
    return pd.DataFrame({"push": j.b - j.a, "unwind": j.c - j.b}, index=j.index)


def main():
    ap = argparse.ArgumentParser(description="closing-auction reversal fingerprint")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for name, cfg in INST.items():
        print(f"\n{'='*104}\n{name.upper()}   (correlation of push vs unwind; the MOC "
              f"mechanism predicts NEGATIVE, and stronger than placebo)\n{'='*104}")
        px = load(cfg)
        dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)
        print(f"  {'window':<38}{'n':>6}{'corr':>8}{'t':>8}{'push mean':>11}"
              f"{'unwind mean':>13}{'dev corr':>10}{'val corr':>10}")
        rows = []
        for lbl, s, e, u in WINDOWS:
            r = pair_returns(px, s, e, u)
            if len(r) < 100: continue
            c, p = stats.pearsonr(r["push"], r["unwind"])
            t = c * np.sqrt((len(r) - 2) / max(1e-12, 1 - c * c))
            dev, val = r.index <= dev_end, r.index > dev_end
            dc = stats.pearsonr(r.push[dev], r.unwind[dev])[0] if dev.sum() > 50 else np.nan
            vc = stats.pearsonr(r.push[val], r.unwind[val])[0] if val.sum() > 50 else np.nan
            pm, um = r["push"].mean(), r["unwind"].mean()
            print(f"  {lbl:<38}{len(r):>6}{c:>8.3f}{t:>8.2f}{pm:>11.3f}"
                  f"{um:>13.3f}{dc:>10.3f}{vc:>10.3f}")
            rows.append({"window": lbl, "n": int(len(r)), "corr": float(c), "t": float(t),
                         "p": float(p), "push_mean": float(r["push"].mean()),
                         "unwind_mean": float(r["unwind"].mean()),
                         "dev_corr": float(dc), "val_corr": float(vc)})
        auc = rows[0]; pl = [x["corr"] for x in rows[1:]]
        print(f"\n  auction corr {auc['corr']:+.3f}  vs placebo mean {np.mean(pl):+.3f}  "
              f"-> excess {auc['corr'] - np.mean(pl):+.3f}")
        out[name] = {"rows": rows, "placebo_mean": float(np.mean(pl)),
                     "excess": float(auc["corr"] - np.mean(pl))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
