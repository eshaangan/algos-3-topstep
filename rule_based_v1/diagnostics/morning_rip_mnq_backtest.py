"""
Morning Rip backtest on MNQ 1-min data.

Strategy: at 9:30 AM ET (equities open), record opening print.
Place buy-stop at open+offset, sell-stop at open-offset (OCO).
Whichever fills carries TP and SL bracket. One trade per day.

Three resolution modes tested:
  1. bounds_honest  : only count confirmed outcomes (no ambiguity)
     - Win  : TP reached in a bar that does NOT also contain SL
     - Loss : SL reached in a bar that does NOT also contain TP
     - Ambiguous : both TP and SL within same bar's range (skip or coin-flip)
  2. path_model     : when ambiguous, assume direction based on bar momentum
     (if high-open > open-low, up move came first → TP for long / SL for short)
  3. drift          : one-sided straddle — only enter in overnight-gap direction

Variants: entry offset 5/8/10/12 pts, symmetric TP=SL, plus Preston 8/15.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import pytz

TICK_SIZE = 0.25               # MNQ tick = 0.25 pts

ET = pytz.timezone("America/New_York")

# ── Load data — merge two sources ─────────────────────────────────────────────
def load_data():
    frames = []

    # Source 1: Aug 2025 – Apr 2026
    f1 = "data/processed/mnq_aug2025_apr2026_1min_eth.h5"
    df1 = pd.read_hdf(f1, key="/bars_5min_eth")
    df1.index = pd.to_datetime(df1.index, utc=True)
    frames.append(df1)

    # Source 2: Mar 2026 – Jun 2026 (Rithmic pull)
    f2 = "data/processed/mnq_ohlcv_1min_rithmic.h5"
    df2 = pd.read_hdf(f2, key="/bars_5min")
    df2.index = pd.to_datetime(df2.index, utc=True)
    frames.append(df2)

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert(ET)
    return df

# ── Isolate one day's bars starting from 9:30 open bar ────────────────────────
def get_day_bars(df, date):
    """Return bars from 9:30 AM to 4:00 PM ET for a given date."""
    lo = pd.Timestamp(date, tz=ET).replace(hour=9, minute=30)
    hi = pd.Timestamp(date, tz=ET).replace(hour=16, minute=0)
    return df.loc[lo:hi]

# ── Simulate one trade ─────────────────────────────────────────────────────────
def simulate_trade(bars, offset, tp_pts, sl_pts, direction=None):
    """
    bars     : DataFrame from 9:30 onwards
    offset   : entry stop offset from opening print (pts)
    tp_pts   : take-profit distance from entry (pts)
    sl_pts   : stop-loss distance from entry (pts)
    direction: 'long', 'short', or None (straddle — whichever fills first)

    Returns dict with keys: outcome, pts, bars_held, entry_bar, entry_px,
                             direction_taken, whipsaw_ambiguous
    """
    if len(bars) < 2:
        return None

    open_print = bars.iloc[0]["open"]

    long_entry  = open_print + offset
    short_entry = open_print - offset

    # Scan for first fill
    entry_bar_idx = None
    entry_px      = None
    side          = None

    for i, (ts, bar) in enumerate(bars.iterrows()):
        h, l, o = bar["high"], bar["low"], bar["open"]

        # Long fill: bar's high >= long_entry
        long_fills  = (h >= long_entry)  and (direction in (None, "long"))
        # Short fill: bar's low  <= short_entry
        short_fills = (l <= short_entry) and (direction in (None, "short"))

        if long_fills and short_fills:
            # Both fill in the same bar — take direction with larger gap
            if direction == "long":
                side = "long"; entry_px = long_entry; entry_bar_idx = i
                break
            elif direction == "short":
                side = "short"; entry_px = short_entry; entry_bar_idx = i
                break
            else:
                # straddle: pick whichever stop is further from open (bigger gap)
                # Tie-break: bar opened closer to which stop?
                gap_l = abs(o - long_entry)
                gap_s = abs(o - short_entry)
                if gap_l <= gap_s:
                    side = "long";  entry_px = long_entry
                else:
                    side = "short"; entry_px = short_entry
                entry_bar_idx = i
                break
        elif long_fills:
            side = "long";  entry_px = long_entry;  entry_bar_idx = i; break
        elif short_fills:
            side = "short"; entry_px = short_entry; entry_bar_idx = i; break

    if entry_bar_idx is None:
        return {"outcome": "no_fill", "pts": 0, "bars_held": 0,
                "direction_taken": None, "whipsaw_ambiguous": False}

    # Determine TP / SL levels
    if side == "long":
        tp_level = entry_px + tp_pts
        sl_level = entry_px - sl_pts
    else:
        tp_level = entry_px - tp_pts
        sl_level = entry_px + sl_pts

    # Scan post-entry bars for exit (include partial fill on entry bar)
    post_entry = bars.iloc[entry_bar_idx:]

    for j, (ts, bar) in enumerate(post_entry.iterrows()):
        h, l = bar["high"], bar["low"]

        if side == "long":
            tp_hit = (h >= tp_level)
            sl_hit = (l <= sl_level)
        else:
            tp_hit = (l <= tp_level)
            sl_hit = (h >= sl_level)

        if tp_hit and sl_hit:
            # Ambiguous bar — path model: whichever extreme is further from open
            bar_open = bar["open"]
            if side == "long":
                # if high-open > open-low, up move was bigger → TP first
                path_up   = h - bar_open
                path_down = bar_open - l
                outcome = "win" if path_up >= path_down else "loss"
            else:
                path_up   = h - bar_open
                path_down = bar_open - l
                outcome = "win" if path_down >= path_up else "loss"
            pts = tp_pts if outcome == "win" else -sl_pts
            return {"outcome": outcome, "pts": pts, "bars_held": j,
                    "direction_taken": side, "whipsaw_ambiguous": True,
                    "entry_px": entry_px, "open_print": open_print}

        elif tp_hit:
            return {"outcome": "win", "pts": tp_pts, "bars_held": j,
                    "direction_taken": side, "whipsaw_ambiguous": False,
                    "entry_px": entry_px, "open_print": open_print}
        elif sl_hit:
            return {"outcome": "loss", "pts": -sl_pts, "bars_held": j,
                    "direction_taken": side, "whipsaw_ambiguous": False,
                    "entry_px": entry_px, "open_print": open_print}

    # No exit hit by session end
    close_px = post_entry.iloc[-1]["close"]
    if side == "long":
        pts = close_px - entry_px
    else:
        pts = entry_px - close_px
    return {"outcome": "timeout", "pts": pts, "bars_held": len(post_entry) - 1,
            "direction_taken": side, "whipsaw_ambiguous": False,
            "entry_px": entry_px, "open_print": open_print}


def run_backtest(df, variants):
    """
    variants: list of dicts with keys offset, tp, sl, name, drift
    Returns per-variant list of daily trade records.
    """
    trading_days = sorted(set(df.index.date))
    results = {v["name"]: [] for v in variants}

    for date in trading_days:
        day = get_day_bars(df, date)
        if len(day) < 5:
            continue

        # Compute overnight gap direction for drift variant
        # gap = today's 9:30 open vs prior close (last bar before 9:30)
        prior = df[df.index < day.index[0]]
        prior_close = prior.iloc[-1]["close"] if len(prior) else None
        today_open  = day.iloc[0]["open"]
        gap_dir     = "long" if (prior_close and today_open > prior_close) else "short"

        for v in variants:
            direction = gap_dir if v.get("drift") else None
            t = simulate_trade(day, v["offset"], v["tp"], v["sl"], direction=direction)
            if t:
                t["date"]    = str(date)
                t["variant"] = v["name"]
                results[v["name"]].append(t)

    return results


def print_stats(results, mnq_dollar_per_pt=2.0, contracts=2):
    """Print scoreboard."""
    multiplier = mnq_dollar_per_pt * contracts  # $4/pt on 2 MNQ = $8/pt
    # NQ equivalent: multiply by 10 for display (NQ = 10×MNQ)

    print(f"\n{'Variant':<22} {'N':>4} {'NoFill':>6} {'WR%':>6} {'Ambig%':>7} {'Net pts':>8} {'Net $':>8} {'$/day':>7}")
    print("-" * 75)

    rows = []
    for name, trades in results.items():
        filled = [t for t in trades if t["outcome"] != "no_fill"]
        if not filled:
            continue
        n        = len(filled)
        wins     = sum(1 for t in filled if t["outcome"] == "win")
        no_fills = len(trades) - n
        ambig    = sum(1 for t in filled if t["whipsaw_ambiguous"])
        net_pts  = sum(t["pts"] for t in filled)
        net_usd  = net_pts * multiplier
        days     = len(trades)
        rows.append((name, n, no_fills, wins/n*100, ambig/n*100,
                     net_pts, net_usd, net_usd/days))

    rows.sort(key=lambda x: -x[6])
    for r in rows:
        name, n, nf, wr, amb, pts, usd, dpd = r
        print(f"{name:<22} {n:>4} {nf:>6} {wr:>6.1f} {amb:>7.1f} {pts:>8.1f} {usd:>8.0f} {dpd:>7.0f}")

    # Separate confirmed-only stats (drop ambiguous)
    print(f"\n--- Confirmed-only (exclude ambiguous bars) ---")
    print(f"{'Variant':<22} {'N_conf':>7} {'WR%':>6} {'Net pts':>8} {'Net $':>8}")
    print("-" * 55)
    conf_rows = []
    for name, trades in results.items():
        conf = [t for t in trades if t["outcome"] != "no_fill" and not t["whipsaw_ambiguous"]]
        if not conf:
            continue
        wins = sum(1 for t in conf if t["outcome"] == "win")
        pts  = sum(t["pts"] for t in conf)
        conf_rows.append((name, len(conf), wins/len(conf)*100, pts, pts*4))

    conf_rows.sort(key=lambda x: -x[4])
    for r in conf_rows:
        name, n, wr, pts, usd = r
        print(f"{name:<22} {n:>7} {wr:>6.1f} {pts:>8.1f} {usd:>8.0f}")


def main():
    print("Loading MNQ 1-min data (Aug 2025 – Apr 2026)...")
    df = load_data()
    print(f"  {len(df)} bars | {df.index[0].date()} to {df.index[-1].date()}")

    # Filter to RTH (9:30–16:00 ET) for the session, but we need overnight for gaps
    trading_days = sorted(set(df.index.date))
    print(f"  {len(trading_days)} calendar dates with data")

    # Count actual trading days (has 9:30 bar)
    has_930 = 0
    for d in trading_days:
        day = get_day_bars(df, d)
        if len(day) > 0 and day.index[0].hour == 9 and day.index[0].minute == 30:
            has_930 += 1
    print(f"  {has_930} days with 9:30 AM ET open bar\n")

    variants = [
        {"name": "main 10/10",     "offset": 10, "tp": 10, "sl": 10, "drift": False},
        {"name": "tight 8/8",      "offset":  8, "tp":  8, "sl":  8, "drift": False},
        {"name": "tight 6/6",      "offset":  6, "tp":  6, "sl":  6, "drift": False},
        {"name": "scalp 5/10",     "offset":  5, "tp": 10, "sl": 10, "drift": False},
        {"name": "wide 12/12",     "offset": 12, "tp": 12, "sl": 12, "drift": False},
        {"name": "preston 8/15",   "offset":  8, "tp": 15, "sl":  8, "drift": False},
        {"name": "drift 10/10",    "offset": 10, "tp": 10, "sl": 10, "drift": True},
        {"name": "drift 8/8",      "offset":  8, "tp":  8, "sl":  8, "drift": True},
        {"name": "wide 25/25",     "offset": 25, "tp": 25, "sl": 25, "drift": False},
        {"name": "wide 50/50",     "offset": 50, "tp": 50, "sl": 50, "drift": False},
    ]

    print("Running backtest...")
    results = run_backtest(df, variants)

    print_stats(results)

    # First-minute range analysis
    print("\n--- First-minute (9:30 bar) range stats ---")
    ranges = []
    for d in trading_days:
        day = get_day_bars(df, d)
        if len(day) > 0 and day.index[0].hour == 9 and day.index[0].minute == 30:
            r = day.iloc[0]["high"] - day.iloc[0]["low"]
            ranges.append(r)
    if ranges:
        r = np.array(ranges)
        print(f"  n={len(r)}, median={np.median(r):.1f}pts, mean={np.mean(r):.1f}pts, "
              f"p25={np.percentile(r,25):.1f}, p75={np.percentile(r,75):.1f}, "
              f"min={r.min():.1f}, max={r.max():.1f}")

        # % of days where both TP and SL (10/10) are within first bar
        ambig_10 = sum(1 for x in ranges if x >= 20) / len(ranges) * 100
        ambig_8  = sum(1 for x in ranges if x >= 16) / len(ranges) * 100
        ambig_6  = sum(1 for x in ranges if x >=  12) / len(ranges) * 100
        print(f"  % days first bar >= 20pts (10/10 fully ambiguous): {ambig_10:.0f}%")
        print(f"  % days first bar >= 16pts (8/8  fully ambiguous):  {ambig_8:.0f}%")
        print(f"  % days first bar >= 12pts (6/6  fully ambiguous):  {ambig_6:.0f}%")

    # Monthly breakdown for best variant
    best = "drift 8/8"
    trades = results.get(best, [])
    if trades:
        print(f"\n--- Monthly breakdown: {best} ---")
        by_month = defaultdict(list)
        for t in trades:
            mo = t["date"][:7]
            by_month[mo].append(t)
        for mo in sorted(by_month):
            mo_trades = by_month[mo]
            filled = [t for t in mo_trades if t["outcome"] != "no_fill"]
            if not filled: continue
            wins = sum(1 for t in filled if t["outcome"] == "win")
            pts  = sum(t["pts"] for t in filled)
            print(f"  {mo}: n={len(filled)}, WR={wins/len(filled)*100:.0f}%, "
                  f"pts={pts:.1f}, ${pts*4:.0f}")


if __name__ == "__main__":
    main()
