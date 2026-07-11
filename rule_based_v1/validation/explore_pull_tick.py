"""Tick-level refinement of the ask-pull lead — DEV DAYS ONLY (< 2026-06-27).

The 5s-grid scan flagged: ask size collapses at unchanged price -> mid falls
~3pt over 60s (t=-2.2, n=124; 1 significant cell of 24 = could be chance).
This refines the event at BBO-update granularity and, critically, splits pulls
by MECHANISM using trade prints in the same window:

  cancel-pull   size drop >> aggressive buy volume  -> liquidity withdrawn
                (the hypothesis: MMs stepping away ahead of downside)
  consumed      size drop ~= buy volume             -> the level was EATEN
                (buying pressure; if anything the opposite prediction)

If the 5s effect was real and mechanism-driven, it should CONCENTRATE in the
cancel bucket and be robust across days and thresholds. If it spreads evenly /
flips day to day, it was noise and dies here.

Self-contained (pandas/numpy); runs on the Mac against consolidated day files.

    python explore_pull_tick.py --data-dir /Users/jg/mnq_l2_data --end 20260627
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

FWD_SECS = [15, 30, 60, 120]
RTH = ("13:30", "20:00")            # 9:30-16:00 ET (EDT)
MIN_EVENT_GAP_S = 30                # de-overlap forward windows
MAX_DT_S = 2.0                      # size drop must happen within this window


def load_day(data_dir: str, day: str):
    bf = os.path.join(data_dir, f"bbo_MNQ_{day}.parquet")
    tf = os.path.join(data_dir, f"trade_MNQ_{day}.parquet")
    if not (os.path.exists(bf) and os.path.exists(tf)):
        return None, None
    b = pd.read_parquet(bf).sort_values("recv_ns")
    for side in ("bid", "ask"):
        absent = b[f"{side}_px"].isna()
        for col in (f"{side}_px", f"{side}_sz"):
            b.loc[absent, col] = np.nan
            b[col] = b[col].ffill()
    b = b.dropna(subset=["bid_px", "ask_px"]).reset_index(drop=True)
    t = pd.read_parquet(tf).sort_values("recv_ns").reset_index(drop=True)
    return b, t


def day_drift(b: pd.DataFrame) -> dict:
    """Unconditional mean fwd mid move per horizon (sampled) — regime control.
    If an 'edge' survives subtracting this, it is directional info, not drift."""
    ns = b["recv_ns"].to_numpy()
    mid = (b["bid_px"].to_numpy() + b["ask_px"].to_numpy()) / 2
    ts = pd.to_datetime(ns, unit="ns", utc=True)
    hhmm = pd.Series(ts).dt.strftime("%H:%M").to_numpy()
    base = np.flatnonzero((hhmm >= RTH[0]) & (hhmm < RTH[1]))[::200]
    out = {}
    for s_ in FWD_SECS:
        j = np.clip(np.searchsorted(ns, ns[base] + int(s_ * 1e9)), 0, len(mid) - 1)
        valid = (ns[j] - ns[base]) < (s_ + 30) * 1e9
        out[s_] = float(np.nanmean(np.where(valid, mid[j] - mid[base], np.nan)))
    return out


def day_events(b: pd.DataFrame, t: pd.DataFrame, side: str,
               min_prev_sz: int, drop_frac: float) -> pd.DataFrame:
    """Pull events on one side at tick level, with cancel/consumed attribution."""
    px = b[f"{side}_px"].to_numpy()
    sz = b[f"{side}_sz"].to_numpy(float)
    ns = b["recv_ns"].to_numpy()
    same_px = px[1:] == px[:-1]
    dt_ok = (ns[1:] - ns[:-1]) <= MAX_DT_S * 1e9
    big = sz[:-1] >= min_prev_sz
    dropped = sz[1:] <= drop_frac * sz[:-1]
    ev = np.flatnonzero(same_px & dt_ok & big & dropped) + 1
    if len(ev) == 0:
        return pd.DataFrame()

    # aggressive volume TOWARD this side in the window (buys eat asks, sells eat bids)
    agg_code = 1 if side == "ask" else 2
    tt = t[t["aggressor"] == agg_code]
    t_ns = tt["recv_ns"].to_numpy()
    t_cum = np.concatenate([[0.0], np.cumsum(tt["size"].to_numpy(float))])
    v0 = t_cum[np.searchsorted(t_ns, ns[ev - 1], side="right")]
    v1 = t_cum[np.searchsorted(t_ns, ns[ev], side="right")]
    eaten = v1 - v0
    drop = sz[ev - 1] - sz[ev]
    with np.errstate(divide="ignore", invalid="ignore"):
        cancel_frac = np.clip((drop - eaten) / np.where(drop > 0, drop, np.nan), 0, 1)

    # forward mid moves via searchsorted on the BBO stream
    mid = (b["bid_px"].to_numpy() + b["ask_px"].to_numpy()) / 2
    out = {"ns": ns[ev], "cancel_frac": cancel_frac, "drop": drop}
    for s_ in FWD_SECS:
        j = np.searchsorted(ns, ns[ev] + int(s_ * 1e9), side="left")
        j = np.clip(j, 0, len(mid) - 1)
        valid = (ns[j] - ns[ev]) < (s_ + 30) * 1e9   # gap guard: quote must exist near t+s
        fwd = mid[j] - mid[ev]
        sgn = -1.0 if side == "bid" else 1.0          # toward pulled side = naive sign
        out[f"f{s_}"] = np.where(valid, sgn * fwd, np.nan)
    df = pd.DataFrame(out)

    # RTH filter + de-overlap (keep first event, then none within MIN_EVENT_GAP_S)
    ts = pd.to_datetime(df["ns"], unit="ns", utc=True)
    hhmm = ts.dt.strftime("%H:%M")
    df = df[(hhmm >= RTH[0]) & (hhmm < RTH[1])].reset_index(drop=True)
    keep, last = [], -1e18
    for i, v in enumerate(df["ns"].to_numpy()):
        if v - last >= MIN_EVENT_GAP_S * 1e9:
            keep.append(i)
            last = v
    return df.iloc[keep].reset_index(drop=True)


def report(tag: str, df: pd.DataFrame, daily: dict) -> None:
    if df.empty:
        print(f"{tag:26s} n=0")
        return
    n = len(df)
    cells = []
    for s_ in FWD_SECS:
        x = df[f"f{s_}"].dropna()
        mu = x.mean()
        tt = mu / (x.std() / np.sqrt(len(x)) + 1e-12)
        cells.append(f"{mu:>+6.2f}p(t{tt:>+4.1f})")
    dneg = [d for d, m in daily.items() if np.isfinite(m)]
    consist = np.mean([daily[d] < 0 for d in dneg]) if dneg else np.nan
    print(f"{tag:26s} n={n:>5} " + " ".join(cells) +
          f"  days<0@60s: {consist:.0%} ({len(dneg)}d)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--end", default="20260627")
    args = ap.parse_args()
    days = sorted({os.path.basename(f).split("_")[-1][:8]
                   for f in glob.glob(os.path.join(args.data_dir, "bbo_MNQ_*.parquet"))})
    days = [d for d in days if d < args.end]

    grids = [(5, 0.4), (3, 0.4), (8, 0.3)]           # (min_prev_sz, drop_frac) robustness
    for side in ("ask", "bid"):
        print(f"\n=== {side.upper()}-PULL, tick-level, dev days (fwd move signed toward pulled side) ===")
        for mps, dfr in grids:
            frames, daily_cancel = [], {}
            for day in days:
                b, t = load_day(args.data_dir, day)
                if b is None or len(b) < 100000:      # skip weekends/dead days
                    continue
                e = day_events(b, t, side, mps, dfr)
                if len(e):
                    e["day"] = day
                    drift = day_drift(b)
                    sgn = -1.0 if side == "bid" else 1.0
                    for s_ in FWD_SECS:
                        e[f"f{s_}"] = e[f"f{s_}"] - sgn * drift[s_]
                    frames.append(e)
                    c = e[e["cancel_frac"] >= 0.7]
                    daily_cancel[day] = c["f60"].mean() if len(c) else np.nan
            if not frames:
                continue
            allev = pd.concat(frames, ignore_index=True)
            print(f"-- prev_sz>={mps}, drop to <={dfr:.0%} --")
            report("  ALL pulls", allev, {})
            report("  CANCEL (frac>=0.7)", allev[allev.cancel_frac >= 0.7], daily_cancel)
            report("  CONSUMED (frac<=0.3)", allev[allev.cancel_frac <= 0.3], {})


if __name__ == "__main__":
    main()
