"""Intraday periodicity: HKS/Bogousslavsky day-lagged, time-matched autocorrelation.

Prereg: vps_intraday_periodicity.yaml.

NOT a re-run of the 540-cell clock map. That map tested UNCONDITIONAL cell means. This
tests a LAGGED effect at a one-day lag matched by time of day -- a bucket can have a zero
mean and still carry strong day-over-day autocorrelation at its own clock time. Heston/
Korajczyk/Sadka (JF 2010) attribute it to institutional order-splitting at consistent
times; Bogousslavsky (JF 2016) derives it from infrequent rebalancing. Both are
forced-flow mechanisms.

Velocity is the point: 13 buckets x ~250 days = ~3,250 opportunities/yr, against the two
existing legs' 52 and 8. The required-edge frontier says only a high-velocity family can
supply the ~$350/week the 16-week deadline needs.

Cost floor is checked BEFORE the signal test, per the arithmetic-first rule.

Usage:
    python3 rule_based_v1/validation/research_intraday_periodicity.py --out runs/periodicity.json
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
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "vol": "volume", "tz": None, "pv": 5.0, "tick": 0.25, "comm": 0.62,
           "dev_end": "2017-12-31"},
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts", "vol": "vol",
            "tz": "America/Chicago", "pv": 2.0, "tick": 0.25, "comm": 0.62,
            "dev_end": "2023-12-31"},
}
BUCKETS = [(570 + 30 * i, 600 + 30 * i) for i in range(13)]   # 09:30-16:00 ET


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


def bucket_panel(px: pd.Series) -> pd.DataFrame:
    """Rows = sessions, cols = bucket index, values = bucket return in points.

    Built by reindexing onto the bucket EDGES so entry and exit are real executable
    marks, not resampled averages.
    """
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 570) & (mins <= 960)]
    day = rth.index.normalize()
    m = pd.Series(rth.index.hour * 60 + rth.index.minute, index=rth.index)
    frame = pd.DataFrame({"px": rth.to_numpy(), "day": day, "m": m.to_numpy()})
    # last mark at or before each edge, per day
    out = {}
    for k, (a, b) in enumerate(BUCKETS):
        op = frame[frame["m"] <= a].groupby("day")["px"].last()
        cl = frame[frame["m"] <= b].groupby("day")["px"].last()
        n_open = frame[frame["m"] <= a].groupby("day")["m"].last()
        n_close = frame[frame["m"] <= b].groupby("day")["m"].last()
        # require the marks to actually sit near the intended edges
        valid = (n_open >= a - 5) & (n_close >= b - 5)
        out[k] = (cl - op).where(valid)
    panel = pd.DataFrame(out).dropna(how="all")
    return panel[panel.notna().sum(axis=1) >= 10]


def evaluate(panel: pd.DataFrame, cfg: dict, contracts: int, rng: np.random.Generator,
             dev_end: pd.Timestamp) -> tuple[list[dict], dict]:
    pv, cost = cfg["pv"], 2 * contracts * (cfg["comm"] + cfg["tick"] * cfg["pv"])
    rows, pooled_cond, pooled_uncond, pooled_shuf = [], [], [], []
    for k in range(len(BUCKETS)):
        r = panel[k].dropna()
        if len(r) < 200:
            continue
        prev = r.shift(1)
        ok = prev.notna() & (prev != 0)
        r_, p_ = r[ok], prev[ok]
        side = np.sign(p_.to_numpy())
        gross = r_.to_numpy() * side * pv * contracts
        cond = gross - cost                       # short net = -gross - cost, via side
        uncond = r_.to_numpy() * pv * contracts - cost
        # shuffled-lag control: a random OTHER day's same-bucket sign
        perm = rng.permutation(len(p_))
        shuf = r_.to_numpy() * np.sign(p_.to_numpy()[perm]) * pv * contracts - cost
        pooled_cond.append(cond); pooled_uncond.append(uncond); pooled_shuf.append(shuf)
        t_c = stats.ttest_1samp(cond, 0)
        d_u = stats.ttest_rel(cond, uncond)
        d_s = stats.ttest_rel(cond, shuf)
        dev = r_.index <= dev_end
        rows.append({
            "bucket": f"{BUCKETS[k][0]//60:02d}:{BUCKETS[k][0]%60:02d}-"
                      f"{BUCKETS[k][1]//60:02d}:{BUCKETS[k][1]%60:02d}",
            "n": int(len(cond)), "mean_abs_move_pts": float(r_.abs().mean()),
            "cond_mean": float(cond.mean()), "cond_t": float(t_c.statistic),
            "uncond_mean": float(uncond.mean()),
            "vs_uncond_t": float(d_u.statistic), "vs_shuffled_t": float(d_s.statistic),
            "dev_mean": float(cond[dev].mean()) if dev.sum() > 20 else float("nan"),
            "val_mean": float(cond[~dev].mean()) if (~dev).sum() > 20 else float("nan"),
        })
    c = np.concatenate(pooled_cond); u = np.concatenate(pooled_uncond)
    s = np.concatenate(pooled_shuf)
    pooled = {"n": int(len(c)), "cond_mean": float(c.mean()),
              "cond_t": float(stats.ttest_1samp(c, 0).statistic),
              "uncond_mean": float(u.mean()),
              "vs_uncond_t": float(stats.ttest_rel(c, u).statistic),
              "vs_shuffled_t": float(stats.ttest_rel(c, s).statistic)}
    return rows, pooled


def main() -> None:
    ap = argparse.ArgumentParser(description="intraday periodicity (HKS / Bogousslavsky)")
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = load_firm_rules("lucid_flex_100k")
    ok, why = check_window_legal("09:30", "16:00", 0, r)
    print(f"\nlegality (intraday RTH at the strictest firm): "
          f"{'LEGAL' if ok else 'ILLEGAL'} -- {why}")

    rng = np.random.default_rng(23)
    out = {}
    for name, cfg in INSTRUMENTS.items():
        print(f"\n{'=' * 108}\n{name.upper()}  ({args.contracts} contracts, "
              f"pv={cfg['pv']}, round-turn cost="
              f"${2 * args.contracts * (cfg['comm'] + cfg['tick'] * cfg['pv']):.2f} = "
              f"{2 * (cfg['comm'] / cfg['pv'] + cfg['tick']):.2f} pts)\n{'=' * 108}")
        px = load_px(cfg)
        panel = bucket_panel(px)
        print(f"  sessions={len(panel)}  buckets={panel.shape[1]}")
        rows, pooled = evaluate(panel, cfg, args.contracts,
                                rng, pd.Timestamp(cfg["dev_end"], tz=ET))
        cost_pts = 2 * (cfg["comm"] / cfg["pv"] + cfg["tick"])
        print(f"\n  COST FLOOR FIRST -- mean |bucket move| vs the {cost_pts:.2f}pt "
              f"round turn:")
        print(f"  {'bucket':>12}{'n':>6}{'|move|pts':>11}{'cost/|move|':>12}"
              f"{'cond $':>9}{'cond t':>8}{'vs uncond':>11}{'vs shuf':>9}"
              f"{'dev $':>9}{'val $':>9}")
        for z in rows:
            print(f"  {z['bucket']:>12}{z['n']:>6}{z['mean_abs_move_pts']:>11.2f}"
                  f"{cost_pts / z['mean_abs_move_pts']:>12.1%}{z['cond_mean']:>9.2f}"
                  f"{z['cond_t']:>8.2f}{z['vs_uncond_t']:>11.2f}{z['vs_shuffled_t']:>9.2f}"
                  f"{z['dev_mean']:>9.2f}{z['val_mean']:>9.2f}")
        print(f"\n  POOLED  n={pooled['n']:,}  conditional ${pooled['cond_mean']:+.3f}/trade "
              f"t={pooled['cond_t']:+.2f}")
        print(f"          unconditional (long-only) ${pooled['uncond_mean']:+.3f}  "
              f"| cond vs uncond t={pooled['vs_uncond_t']:+.2f}  "
              f"cond vs shuffled-lag t={pooled['vs_shuffled_t']:+.2f}")
        best = max(rows, key=lambda z: abs(z["cond_t"])) if rows else None
        if best:
            print(f"          best single bucket: {best['bucket']} t={best['cond_t']:+.2f} "
                  f"vs a 3.09 Bonferroni bar over 26 cells")
        # what would it be worth per week?
        per_wk = pooled["cond_mean"] * len(rows) * 5
        print(f"          if traded every bucket every day: ${per_wk:+,.0f}/week "
              f"at {args.contracts} micros  (gap to close: ~$350/wk)")
        out[name] = {"buckets": rows, "pooled": pooled, "per_week": per_wk}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\ntrials counted: 26   Bonferroni t bar: 3.09\nwrote {args.out}")


if __name__ == "__main__":
    main()
