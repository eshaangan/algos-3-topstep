"""
Screen: does depth-scaled order-flow imbalance clear the MNQ taker cost floor?

THE HYPOTHESIS, STATED BEFORE THE RUN
-------------------------------------
Cont-Kukanov-Stoikov give dP = beta * OFI / AD, so the price-impact coefficient
rises as available depth falls. If that holds on MNQ, then:

    H1  `ofi_scaled` has materially higher rank correlation with forward return
        than raw `ofi`, and the gap widens as depth falls.

    H2  In the LOW-DEPTH tail the conditional move is large enough that a taker
        clears the measured cost floor of $2.06/contract, while in the
        high-depth bulk it is not.

H2 is the only tradeable claim. H1 failing means the CKS scaling does not
describe this instrument and H2 should not be interrogated further.

WHY THIS IS NOT THE KILLED SCREEN AGAIN
---------------------------------------
The September 2026 screen reported "OFI dead, IC ~ 0". It measured OFI by
differencing one-second bar endpoints, which on a ~940 event/second instrument
discards almost all of the flow (locked by a unit test: a bid walking up ten
ticks accumulates 100 units of true OFI and 10 by endpoint differencing), and it
never divided by depth. Both are fixed in features/depth_scaled_flow.py. This
screen is therefore a first test of the CKS relation on MNQ, not a re-test.

PROTOCOL (frozen before looking at any result)
----------------------------------------------
* DEV   = the first 60% of available L3 session dates.
* VAL   = the remaining 40%. Looked at ONCE, after a single DEV cell is chosen.
* OOE   = the MBP-1 days (2026-01 -> 2026-07), a different era, quote-side
          features only. Final check, one look.
* The grid below is declared here in full. Cells are counted and reported, and
  the DEV winner is chosen by NET dollars per contract, not by t-statistic.
* Trades are NON-OVERLAPPING: once entered, no new entry until the hold expires.
  Overlapping 1-second entries on a 60-second hold would inflate every t by
  roughly sqrt(60) and is how this kind of screen usually fools itself.
* Costs are charged at the PREVAILING spread of the entry bar, not an average:
  enter long at the ask, exit at the bid, plus commission both sides.

Usage
-----
    python -m ml_intraday_v3.diagnostics.depth_scaled_screen build \
        --depth-dir data/processed/vps_mirror/mnq_l2_data \
        --out data/processed/mbo_features

    python -m ml_intraday_v3.diagnostics.depth_scaled_screen screen \
        --features data/processed/mbo_features --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ml_intraday_v3.features.depth_scaled_flow import (
    COMMISSION_PER_SIDE, POINT_VALUE, add_depth_state, build_bars,
    build_day_from_mbo, replay_bbo_from_mbp1,
)

# ---- the declared grid -----------------------------------------------------
HORIZONS_S = (1, 5, 10, 30, 60, 300)
# depth_pctile ceilings: "thinnest x% of the trailing 30 minutes"
DEPTH_TAILS = (0.10, 0.25, 0.50, 1.00)
# signal thresholds, as a quantile of |signal| within the depth bucket
SIGNAL_QUANTILES = (0.90, 0.95, 0.99)
SIGNALS = ("ofi", "ofi_scaled", "signed_vol", "signed_vol_scaled",
           "queue_imbalance")

RTH_START, RTH_END = "09:30", "15:55"   # ET; trade file only covers 09:00-16:10
TZ = "America/New_York"


# ===========================================================================
# build
# ===========================================================================

def _dates_in(depth_dir: str) -> list[str]:
    out = []
    for f in sorted(os.listdir(depth_dir)):
        m = re.match(r"depth_MNQ_(\d{8})\.parquet$", f)
        if m:
            out.append(m.group(1))
    return out


def build_features(depth_dir: str, out_dir: str, grid_ms: int = 1000,
                   dates: Optional[Iterable[str]] = None) -> list[str]:
    """Build and cache the per-day feature panel for every available L3 day."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for d in (dates or _dates_in(depth_dir)):
        dst = os.path.join(out_dir, f"features_{d}.parquet")
        if os.path.exists(dst):
            written.append(dst)
            continue
        depth_p = os.path.join(depth_dir, f"depth_MNQ_{d}.parquet")
        trade_p = os.path.join(depth_dir, f"trade_MNQ_{d}.parquet")
        if not os.path.exists(trade_p):
            trade_p = None
        bars = build_day_from_mbo(depth_p, trade_p, grid_ms=grid_ms)
        bars["date"] = d
        bars.to_parquet(dst, index=False)
        written.append(dst)
        print(f"[build] {d}: {len(bars)} bars -> {dst}", flush=True)
    return written


def build_ooe_features(ref_dir: str, out_dir: str, grid_ms: int = 1000) -> list[str]:
    """
    Build the panel for the out-of-era set: MNQ reference BBO, 2026-06 -> 2026-07.

    See PREREG appendix A.1 for why this replaced the MBP-1 set and for the two
    limitations that weaken it: these files stop around 10:32 ET, so the window
    is the first hour only, and they carry no trade stream, so the `signed_vol`
    family cannot be checked out of era at all.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for f in sorted(os.listdir(ref_dir)):
        m = re.match(r"bbo_MNQ_(\d{8})\.parquet$", f)
        if not m:
            continue
        d = m.group(1)
        dst = os.path.join(out_dir, f"features_{d}.parquet")
        if os.path.exists(dst):
            written.append(dst)
            continue
        raw = pd.read_parquet(os.path.join(ref_dir, f))
        bars = add_depth_state(
            build_bars(replay_bbo_from_mbp1(raw), grid_ms=grid_ms))
        bars["date"] = d
        bars.to_parquet(dst, index=False)
        written.append(dst)
        print(f"[ooe] {d}: {len(bars)} bars -> {dst}", flush=True)
    return written


# ===========================================================================
# panel assembly
# ===========================================================================

def load_panel(feature_dir: str, dates: Iterable[str]) -> pd.DataFrame:
    frames = []
    for d in dates:
        p = os.path.join(feature_dir, f"features_{d}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no feature files for {list(dates)} in {feature_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["ts_et"] = pd.to_datetime(df["bar_ns"], unit="ns", utc=True).dt.tz_convert(TZ)
    return df.sort_values("bar_ns").reset_index(drop=True)


def add_forward_returns(df: pd.DataFrame, grid_ms: int = 1000) -> pd.DataFrame:
    """Forward mid-to-mid returns in MNQ POINTS, per session date.

    Computed inside each date so a horizon never reaches across the overnight
    gap, which is untradeable here and would dominate every long horizon.
    """
    per = max(1, int(round(1000 / grid_ms)))
    out = []
    for _, g in df.groupby("date", sort=True):
        g = g.copy()
        for h in HORIZONS_S:
            g[f"fwd_{h}s"] = g["mid"].shift(-h * per) - g["mid"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def restrict_to_rth(df: pd.DataFrame) -> pd.DataFrame:
    t = df["ts_et"].dt.strftime("%H:%M")
    return df[(t >= RTH_START) & (t <= RTH_END)].reset_index(drop=True)


# ===========================================================================
# H1: does depth scaling improve the correlation, and does the gap widen thin?
# ===========================================================================

def ic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman IC of each signal against each forward horizon, by depth bucket.

    Reported with a per-DATE breakdown so a single day cannot carry a result:
    `ic` is the mean of the daily ICs and `t_daily` treats the daily ICs as the
    sample. With 1-second overlapping observations, a pooled t is meaningless;
    the daily statistic is the honest one at this sample size.
    """
    rows = []
    for sig in SIGNALS:
        if sig not in df.columns:
            continue
        for tail in DEPTH_TAILS:
            sub = df[df["depth_pctile"] <= tail] if tail < 1.0 else df
            for h in HORIZONS_S:
                col = f"fwd_{h}s"
                daily = []
                for d, g in sub.groupby("date"):
                    x, y = g[sig], g[col]
                    ok = x.notna() & y.notna()
                    if ok.sum() < 200:
                        continue
                    daily.append(stats.spearmanr(x[ok], y[ok]).statistic)
                if len(daily) < 3:
                    continue
                daily = np.array(daily, dtype=float)
                t = stats.ttest_1samp(daily, 0.0)
                rows.append({
                    "signal": sig, "depth_tail": tail, "horizon_s": h,
                    "n_days": len(daily), "n_obs": int(sub[col].notna().sum()),
                    "ic": daily.mean(), "ic_sd": daily.std(ddof=1),
                    "t_daily": t.statistic, "p_daily": t.pvalue,
                    "frac_days_positive": float((daily > 0).mean()),
                })
    return pd.DataFrame(rows)


# ===========================================================================
# H2: the costed, non-overlapping trade simulation
# ===========================================================================

@dataclass
class CellResult:
    signal: str
    depth_tail: float
    signal_q: float
    horizon_s: int
    n_trades: int
    gross_per_contract: float
    net_per_contract: float
    net_total: float
    t_stat: float
    p_value: float
    win_rate: float
    mean_cost: float
    net_over_mae: float
    trades_per_day: float
    threshold: float = float('nan')


@dataclass
class Panel:
    """Contiguous numpy view of the bar panel, prepared once and reused.

    The screen runs 360 cells over the same bars. Doing this per cell in pandas
    (frame copy, hash index rebuild, `.at` in a Python loop) is what made the
    first DEV attempt die: it allocated a fresh 208k-row copy 360 times.
    """
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    depth_pctile: np.ndarray
    date_code: np.ndarray
    signals: dict


def prepare_panel(df: pd.DataFrame) -> Panel:
    d = df.reset_index(drop=True)
    return Panel(
        mid=d["mid"].to_numpy(dtype=float),
        bid=d["best_bid"].to_numpy(dtype=float),
        ask=d["best_ask"].to_numpy(dtype=float),
        depth_pctile=d["depth_pctile"].to_numpy(dtype=float),
        date_code=pd.factorize(d["date"])[0].astype(np.int64),
        signals={s: d[s].to_numpy(dtype=float) for s in SIGNALS if s in d.columns},
    )


def simulate(panel: Panel, signal: str, depth_tail: float, signal_q: float,
             horizon_s: int, grid_ms: int = 1000,
             threshold: Optional[float] = None) -> Optional[CellResult]:
    """
    Non-overlapping taker simulation for one grid cell.

    Entry: |signal| is at or above the threshold AND the bar's trailing depth
    percentile is at or below `depth_tail`. Direction is the sign of the signal.
    Long enters at the ask and exits at the bid, short the reverse, commission
    both sides.

    `threshold` is derived from `signal_q` when omitted. Pass it explicitly to
    carry a DEV-derived absolute threshold onto VAL, which is what the
    pre-registration requires rather than re-fitting the quantile on VAL.

    The exit is read `hold` bars after entry in the FULL panel, so bars excluded
    by the depth condition still count toward elapsed time. Trades that would
    cross a session boundary are dropped rather than held overnight.
    """
    sig = panel.signals.get(signal)
    if sig is None:
        return None
    per = max(1, int(round(1000 / grid_ms)))
    hold = horizon_s * per
    n = len(panel.mid)

    ok = (np.isfinite(sig) & np.isfinite(panel.mid) & np.isfinite(panel.bid)
          & np.isfinite(panel.ask) & np.isfinite(panel.depth_pctile))
    if depth_tail < 1.0:
        ok &= panel.depth_pctile <= depth_tail
    if ok.sum() < 500:
        return None

    absig = np.abs(sig)
    thr = float(np.quantile(absig[ok], signal_q)) if threshold is None \
        else float(threshold)
    cand = np.flatnonzero(ok & (absig >= thr))
    if len(cand) == 0:
        return None

    comm = 2 * COMMISSION_PER_SIDE
    pnl, costs, maes, dates = [], [], [], set()
    last_exit = -1
    for fi in cand:
        if fi <= last_exit:
            continue
        ex = fi + hold
        if ex >= n or panel.date_code[ex] != panel.date_code[fi]:
            continue
        side = 1.0 if sig[fi] > 0 else -1.0
        if side > 0:
            entry_px, exit_px = panel.ask[fi], panel.bid[ex]
        else:
            entry_px, exit_px = panel.bid[fi], panel.ask[ex]
        if not (np.isfinite(entry_px) and np.isfinite(exit_px)):
            continue

        net = side * (exit_px - entry_px) * POINT_VALUE - comm
        path = panel.mid[fi:ex + 1]
        mae = float(np.nanmin(side * (path - panel.mid[fi])) * POINT_VALUE)

        pnl.append(net)
        costs.append((panel.ask[fi] - panel.bid[fi]) * POINT_VALUE + comm)
        maes.append(mae)
        dates.add(panel.date_code[fi])
        last_exit = ex

    if len(pnl) < 30:
        return None
    pnl = np.asarray(pnl)
    t = stats.ttest_1samp(pnl, 0.0)
    worst_mae = abs(np.percentile(maes, 5))
    return CellResult(
        signal=signal, depth_tail=depth_tail, signal_q=signal_q,
        horizon_s=horizon_s, n_trades=len(pnl),
        gross_per_contract=float(pnl.mean() + np.mean(costs)),
        net_per_contract=float(pnl.mean()),
        net_total=float(pnl.sum()),
        t_stat=float(t.statistic), p_value=float(t.pvalue),
        win_rate=float((pnl > 0).mean()),
        mean_cost=float(np.mean(costs)),
        net_over_mae=float(pnl.mean() / worst_mae) if worst_mae else np.nan,
        trades_per_day=len(pnl) / max(1, len(dates)),
        threshold=thr,
    )


def sweep(df: pd.DataFrame, grid_ms: int = 1000) -> pd.DataFrame:
    """Run the full declared grid. Every cell is reported, none are hidden."""
    panel = prepare_panel(df)
    rows = []
    for sig in SIGNALS:
        if sig not in panel.signals:
            continue
        for tail in DEPTH_TAILS:
            for q in SIGNAL_QUANTILES:
                for h in HORIZONS_S:
                    r = simulate(panel, sig, tail, q, h, grid_ms=grid_ms)
                    if r is not None:
                        rows.append(asdict(r))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("net_per_contract", ascending=False)
    return out


# ===========================================================================
# cli
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--depth-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--grid-ms", type=int, default=1000)
    b.add_argument("--dates", nargs="*", default=None)

    o = sub.add_parser("build-ooe")
    o.add_argument("--ref-dir", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--grid-ms", type=int, default=1000)

    s = sub.add_parser("screen")
    s.add_argument("--features", required=True)
    s.add_argument("--split", choices=["dev", "val", "all"], default="dev")
    s.add_argument("--dev-frac", type=float, default=0.6)
    s.add_argument("--grid-ms", type=int, default=1000)
    s.add_argument("--out", default="ml_intraday_v3/runs/depth_scaled_screen")

    a = ap.parse_args()

    if a.cmd == "build":
        build_features(a.depth_dir, a.out, grid_ms=a.grid_ms, dates=a.dates)
        return

    if a.cmd == "build-ooe":
        build_ooe_features(a.ref_dir, a.out, grid_ms=a.grid_ms)
        return

    dates = sorted(re.match(r"features_(\d{8})\.parquet$", f).group(1)
                   for f in os.listdir(a.features)
                   if re.match(r"features_(\d{8})\.parquet$", f))
    cut = int(len(dates) * a.dev_frac)
    use = {"dev": dates[:cut], "val": dates[cut:], "all": dates}[a.split]
    print(f"[screen] split={a.split} dates={use}", flush=True)

    df = restrict_to_rth(add_forward_returns(load_panel(a.features, use),
                                             grid_ms=a.grid_ms))
    print(f"[screen] {len(df)} RTH bars over {df['date'].nunique()} days",
          flush=True)

    os.makedirs(a.out, exist_ok=True)
    ics = ic_table(df)
    ics.to_csv(os.path.join(a.out, f"ic_{a.split}.csv"), index=False)
    cells = sweep(df, grid_ms=a.grid_ms)
    cells.to_csv(os.path.join(a.out, f"cells_{a.split}.csv"), index=False)

    with open(os.path.join(a.out, f"meta_{a.split}.json"), "w") as fh:
        json.dump({"split": a.split, "dates": list(use),
                   "n_cells": int(len(cells)), "grid_ms": a.grid_ms,
                   "cost_floor_per_contract":
                       "measured 2026-09-09: $2.06 = 1.03 MNQ points"}, fh,
                  indent=2)

    print("\n=== H1: IC, raw vs depth-scaled (full sample, all depths) ===")
    print(ics[ics.depth_tail == 1.0]
          .pivot_table(index="horizon_s", columns="signal", values="ic")
          .round(4).to_string())
    print("\n=== H1: IC in the thinnest 10% of book states ===")
    print(ics[ics.depth_tail == 0.10]
          .pivot_table(index="horizon_s", columns="signal", values="ic")
          .round(4).to_string())
    print(f"\n=== H2: top cells of {len(cells)} run ===")
    if len(cells):
        print(cells.head(12).to_string(index=False))
    else:
        print("no cell produced 30+ non-overlapping trades")


if __name__ == "__main__":
    main()
