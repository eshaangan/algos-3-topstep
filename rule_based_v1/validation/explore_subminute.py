"""Sub-minute event-study scan on recorded MNQ ticks — DEV DAYS ONLY (< 2026-06-27).

The 30s/1-min layer is exhausted (four signal families killed). This scans the
horizon we've never touched: 5s-grid events with forward mid moves at 15s-120s.
Self-contained on purpose (pandas/numpy only) so it runs on the remote Mac next
to the consolidated day files.

Event studies:
  A. SWEEPS      one-sided aggressive volume burst (bucket vol>p99, |imb|>0.7)
                 -> continuation or fade over the next 15-120s?
  B. LIQ PULLS   one side's resting size collapses >70% in 5s
                 -> does price move toward the pulled side?
  C. IMB IC      l1_imb at 5s vs forward mid move (does the fade/follow structure
                 that was dead at 30s+ exist inside the minute?)
  D. RUNS        >=N consecutive same-side aggressor trades -> ignition?

Output: aggregate table (n, mean fwd move in pts signed by event direction,
per-event t, and mean-of-daily-means to guard against one crazy day). Costs for
reference: MNQ round trip ~ $2.24 + half-spread 0.25pt -> an exploitable event
needs |move| >> ~0.8 pt at 2 contracts.

    python explore_subminute.py --data-dir /Users/jg/mnq_l2_data --end 20260627
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

GRID = "5s"
FWD = [3, 6, 12, 24]          # forward horizons in 5s steps: 15/30/60/120s
RTH = ("13:30", "20:00")      # 9:30-16:00 ET in UTC (summer)


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
    b = b.dropna(subset=["bid_px", "ask_px"])
    b["ts"] = pd.to_datetime(b["recv_ns"], unit="ns", utc=True)
    t = pd.read_parquet(tf).sort_values("recv_ns")
    t["ts"] = pd.to_datetime(t["recv_ns"], unit="ns", utc=True)
    return b, t


def day_grid(b: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """5s grid: last quote state per bucket + aggressive volume per side."""
    b = b.set_index("ts")
    g = b.resample(GRID).last()[["bid_px", "ask_px", "bid_sz", "ask_sz"]].ffill()
    g["mid"] = (g.bid_px + g.ask_px) / 2
    den = (g.bid_sz + g.ask_sz).replace(0, np.nan)
    g["l1_imb"] = (g.bid_sz - g.ask_sz) / den
    g["n_upd"] = b["bid_px"].resample(GRID).count()          # quote velocity
    t = t.set_index("ts")
    g["buy_v"] = t[t.aggressor == 1]["size"].resample(GRID).sum()
    g["sell_v"] = t[t.aggressor == 2]["size"].resample(GRID).sum()
    g[["buy_v", "sell_v"]] = g[["buy_v", "sell_v"]].fillna(0)
    g["vol"] = g.buy_v + g.sell_v
    g["tfi"] = (g.buy_v - g.sell_v) / g["vol"].replace(0, np.nan)
    # restrict to RTH and drop stale buckets (ffill manufactures constant rows
    # on dead/weekend stretches — they poison ICs and event baselines)
    hhmm = g.index.strftime("%H:%M")
    g = g[(hhmm >= RTH[0]) & (hhmm < RTH[1])]
    g = g[g["n_upd"].fillna(0) > 0]
    return g


def fwd_moves(g: pd.DataFrame) -> dict:
    return {k: g["mid"].shift(-k) - g["mid"] for k in FWD}


def collect(g: pd.DataFrame) -> dict:
    """Per-day event frames: each a DataFrame of signed forward moves."""
    fm = fwd_moves(g)
    out = {}

    # A. sweeps
    thr = g["vol"].quantile(0.97)
    for side, name in ((1, "sweep_buy"), (-1, "sweep_sell")):
        ev = (g["vol"] > thr) & (side * g["tfi"] > 0.6)
        out[name] = pd.DataFrame({f"f{k}": side * fm[k][ev] for k in FWD})

    # B. liquidity pulls (size collapse >70% bucket-over-bucket, price unmoved)
    for side, name, sgn in (("bid", "pull_bid", -1), ("ask", "pull_ask", 1)):
        sz = g[f"{side}_sz"]
        ev = (sz.shift(1) >= 5) & (sz <= 0.4 * sz.shift(1)) & \
             (g[f"{side}_px"] == g[f"{side}_px"].shift(1))
        # signed toward the pulled side: pull bid -> expect down (sgn=-1)
        out[name] = pd.DataFrame({f"f{k}": sgn * fm[k][ev] for k in FWD})

    # D. aggressor runs are approximated at grid level: 3 consecutive buckets
    # of one-sided flow (|tfi|>0.6 same sign, vol>median)
    med = g["vol"].median()
    for side, name in ((1, "run_buy"), (-1, "run_sell")):
        s = (side * g["tfi"] > 0.5) & (g["vol"] > med)
        ev = s & s.shift(1, fill_value=False)
        out[name] = pd.DataFrame({f"f{k}": side * fm[k][ev] for k in FWD})

    # C. imbalance IC (continuous, not events) — return raw pairs
    out["_ic"] = pd.DataFrame({"imb": g["l1_imb"],
                               **{f"f{k}": fm[k] for k in FWD}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--end", default="20260627", help="exclusive dev cutoff day")
    args = ap.parse_args()

    days = sorted({os.path.basename(f).split("_")[-1][:8]
                   for f in glob.glob(os.path.join(args.data_dir, "bbo_MNQ_*.parquet"))})
    days = [d for d in days if d < args.end]
    pooled: dict[str, list] = {}
    daily_means: dict[str, list] = {}
    ics: dict[int, list] = {k: [] for k in FWD}

    for day in days:
        b, t = load_day(args.data_dir, day)
        if b is None or len(b) < 10000:
            continue
        g = day_grid(b, t)
        if len(g) < 1000:   # kills weekends/half-days
            continue
        res = collect(g)
        for name, df in res.items():
            if name == "_ic":
                for k in FWD:
                    m = df["imb"].notna() & df[f"f{k}"].notna()
                    if m.sum() > 300:
                        v_ = float(df.loc[m, "imb"].corr(df.loc[m, f"f{k}"],
                                                                  method="spearman"))
                        if np.isfinite(v_):
                            ics[k].append(v_)
                continue
            df = df.dropna()
            if len(df):
                pooled.setdefault(name, []).append(df)
                daily_means.setdefault(name, []).append(df["f6"].mean())  # 30s ref
        print(f"  {day}: grid={len(g)} done", flush=True)

    print(f"\n=== DEV-only sub-minute events, {len(days)} days, {GRID} grid, RTH ===")
    print(f"{'event':11s} {'n':>6} " + " ".join(f"{'f'+str(k)+'(' + str(k*5) + 's)':>12}" for k in FWD)
          + f" {'t(30s)':>7} {'dayMean30s':>10}")
    for name in sorted(pooled):
        df = pd.concat(pooled[name], ignore_index=True)
        n = len(df)
        cells = []
        for k in FWD:
            mu = df[f"f{k}"].mean()
            tt = mu / (df[f"f{k}"].std() / np.sqrt(n) + 1e-12)
            cells.append(f"{mu:>+7.2f}p(t{tt:>+4.1f})")
        dm = np.mean(daily_means[name])
        print(f"{name:11s} {n:>6} " + " ".join(cells) + f" {dm:>+9.2f}p")
    print("\nIC of l1_imb (mean of daily Spearman):")
    for k in FWD:
        v = ics[k]
        print(f"  fwd {k*5:>3}s: IC={np.mean(v):+.4f} (days={len(v)})" if v else f"  fwd {k*5}s: n/a")
    print("\nCost wall reference: ~0.8pt needed at 2 lots before a strategy is even "
          "worth simulating. Exploratory, dev-only — holdout untouched.")


if __name__ == "__main__":
    main()
