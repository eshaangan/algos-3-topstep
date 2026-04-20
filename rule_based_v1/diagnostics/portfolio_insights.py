"""Portfolio Insights — Deep analytics across ORB, MSITE, and GIRE strategies.

Run from project root:
    python3 rule_based_v1/diagnostics/portfolio_insights.py
"""

from __future__ import annotations

import sys
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(ROOT), str(ROOT / "rule_based_v1"), str(ROOT / "rule_based_v1/diagnostics")]:
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "portfolio_insights_results.json"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("Loading data...")

bars_ytd = pd.read_hdf(ROOT / "data/processed/mnq_2026ytd_5min.h5", "/bars_5min")
bars_long = pd.read_hdf(ROOT / "data/processed/mnq_5min_aug25_mar26.h5", "/bars_5min")

# Ensure tz-aware
for name, bars in [("bars_ytd", bars_ytd), ("bars_long", bars_long)]:
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

# MES 1-min (full ETH, integer index -> set timestamp as index)
with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
    mes1m = s["/bars_1m"].set_index("timestamp")
mes1m.index = pd.to_datetime(mes1m.index, utc=True).tz_convert("US/Eastern")
mes1m = mes1m.sort_index()
mes_aug = mes1m[mes1m.index >= "2025-08-01"]

print(f"  bars_ytd:  {len(bars_ytd):,} bars  {bars_ytd.index[0].date()} -> {bars_ytd.index[-1].date()}")
print(f"  bars_long: {len(bars_long):,} bars  {bars_long.index[0].date()} -> {bars_long.index[-1].date()}")
print(f"  mes_aug:   {len(mes_aug):,} 1-min bars")

# ---------------------------------------------------------------------------
# Run ORB strategy
# ---------------------------------------------------------------------------
print("\nRunning ORB strategy...")
from novel_filter_sweep import run_orb, build_day_meta, enrich_prev_vwap


def run_orb_strategy(bars):
    meta = enrich_prev_vwap(bars, build_day_meta(bars))
    return run_orb(bars, meta, filter_fn=lambda m: m.get("prev_vwap_bullish") is True)


orb_ytd = run_orb_strategy(bars_ytd)
orb_long = run_orb_strategy(bars_long)
print(f"  ORB YTD:  {orb_ytd['n']} trades, PnL=${orb_ytd['pnl']:,.0f}")
print(f"  ORB Long: {orb_long['n']} trades, PnL=${orb_long['pnl']:,.0f}")

# ---------------------------------------------------------------------------
# Run MSITE strategy — MUST call _apply_tf_config(5) first
# ---------------------------------------------------------------------------
print("\nRunning MSITE strategy...")
import msite_backtest as mb

mb._apply_tf_config(5)


def run_msite(bars):
    trades, _ = mb.run_backtest(
        bars,
        allow_release=True,
        allow_exhaustion=False,
        long_only=True,
        afternoon_cutoff=(14, 0),
        pt_mult=2.0,
        sl_frac=0.55,
        time_stop_bars=12,
    )
    return trades


msite_ytd = run_msite(bars_ytd)
msite_long = run_msite(bars_long)
print(f"  MSITE YTD:  {len(msite_ytd)} trades")
print(f"  MSITE Long: {len(msite_long)} trades")

# ---------------------------------------------------------------------------
# Run GIRE strategy (long period only, MES 1-min)
# ---------------------------------------------------------------------------
print("\nRunning GIRE strategy...")
from gire_backtest import build_daily_data, run_backtest as gire_run

gire_daily_data = build_daily_data(mes_aug)
gire_trades = gire_run(
    gire_daily_data,
    g_star=0.2,
    clv_thresh=0.8,
    ey_thresh=0.0,
    st_thresh=0.0,
    use_opening_failure=True,
    time_stop_hour=12,
)
print(f"  GIRE Long: {len(gire_trades)} trades")

# GIRE YTD: inject 1 manual trade
gire_ytd_trades = [{"date": "2026-01-28", "pnl_usd": 321.0}]

# ---------------------------------------------------------------------------
# Scale factors
# ORB:   scale=1.0 (already 3x MNQ $2/pt)
# MSITE: scale=6.0 (2x -> 12x MNQ)
# GIRE:  scale=2.0 (1x MES $5/pt -> 5x MNQ $10/pt)
# ---------------------------------------------------------------------------
ORB_SCALE = 1.0
MSITE_SCALE = 6.0
GIRE_SCALE = 2.0


# ---------------------------------------------------------------------------
# Helper: build daily PnL dict from trade list
# ---------------------------------------------------------------------------
def orb_daily_pnl(trades: list, scale: float = 1.0) -> dict:
    """ORB trades have keys: date (datetime.date), pnl (float)."""
    d = defaultdict(float)
    for t in trades:
        dt = t["date"]
        dt_str = str(dt) if not isinstance(dt, str) else dt
        d[dt_str] += t["pnl"] * scale
    return dict(d)


def msite_daily_pnl(trades: list, scale: float = 1.0) -> dict:
    """MSITE trades have keys: date (string like '2026-01-30'), pnl (float)."""
    d = defaultdict(float)
    for t in trades:
        d[str(t["date"])] += t["pnl"] * scale
    return dict(d)


def gire_daily_pnl(trades: list, scale: float = 1.0) -> dict:
    """GIRE trades have keys: date (string), pnl_usd (float)."""
    d = defaultdict(float)
    for t in trades:
        d[str(t["date"])] += t["pnl_usd"] * scale
    return dict(d)


def merge_daily_pnl(*dicts: dict) -> dict:
    """Sum PnL across strategy daily dicts."""
    merged = defaultdict(float)
    for d in dicts:
        for dt, pnl in d.items():
            merged[dt] += pnl
    return dict(merged)


# Build daily PnL dicts
orb_ytd_daily = orb_daily_pnl(orb_ytd["trades"], ORB_SCALE)
orb_long_daily = orb_daily_pnl(orb_long["trades"], ORB_SCALE)

msite_ytd_daily = msite_daily_pnl(msite_ytd, MSITE_SCALE)
msite_long_daily = msite_daily_pnl(msite_long, MSITE_SCALE)

gire_ytd_daily = gire_daily_pnl(gire_ytd_trades, GIRE_SCALE)
gire_long_daily = gire_daily_pnl(gire_trades, GIRE_SCALE)

# Portfolio daily PnL
port_ytd_daily = merge_daily_pnl(orb_ytd_daily, msite_ytd_daily, gire_ytd_daily)
port_long_daily = merge_daily_pnl(orb_long_daily, msite_long_daily, gire_long_daily)


# ---------------------------------------------------------------------------
# Utility functions for analytics
# ---------------------------------------------------------------------------

def compute_equity_curve(daily_pnl: dict) -> pd.Series:
    s = pd.Series(daily_pnl).sort_index()
    return s.cumsum()


def max_drawdown_series(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def sharpe_annualized(daily_pnl: dict) -> float:
    s = pd.Series(daily_pnl)
    s = s[s != 0]
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(252))


def sortino_annualized(daily_pnl: dict) -> float:
    s = pd.Series(daily_pnl)
    s = s[s != 0]
    downside = s[s < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(s.mean() / downside.std() * np.sqrt(252))


def calmar_ratio(daily_pnl: dict) -> float:
    s = pd.Series(daily_pnl).sort_index()
    total = float(s.sum())
    n_days = len(s)
    annual_return = total * 252 / max(n_days, 1)
    eq = s.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    mdd = float(dd.min())
    if mdd == 0:
        return float("inf")
    return annual_return / abs(mdd)


def win_streaks(daily_pnl: dict):
    s = pd.Series(daily_pnl).sort_index()
    max_win = max_loss = cur_win = cur_loss = 0
    for v in s:
        if v > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        elif v < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
        else:
            cur_win = cur_loss = 0
    return max_win, max_loss


# ---------------------------------------------------------------------------
# Section 1: Strategy Correlation (long period)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 1: STRATEGY CORRELATION (long period, daily PnL)")
print("=" * 72)

strat_names = ["ORB", "MSITE", "GIRE"]
strat_dicts = [orb_long_daily, msite_long_daily, gire_long_daily]
strat_series = [pd.Series(d) for d in strat_dicts]

corr_matrix = {}
pairs = [("ORB", "MSITE", 0, 1), ("ORB", "GIRE", 0, 2), ("MSITE", "GIRE", 1, 2)]
for n1, n2, i1, i2 in pairs:
    s1 = strat_series[i1]
    s2 = strat_series[i2]
    common = s1.index.intersection(s2.index)
    # Only dates where both have non-zero PnL
    common = [d for d in common if s1[d] != 0 and s2[d] != 0]
    if len(common) >= 2:
        v1 = s1[common].values.astype(float)
        v2 = s2[common].values.astype(float)
        corr = float(np.corrcoef(v1, v2)[0, 1])
    else:
        corr = float("nan")
    corr_matrix[f"{n1}_vs_{n2}"] = {"corr": round(corr, 4), "n_common_days": len(common)}

print(f"\n  {'Pair':<20} {'Pearson r':>10}  {'N common days':>14}")
print(f"  {'-'*48}")
for k, v in corr_matrix.items():
    print(f"  {k:<20} {v['corr']:>10.4f}  {v['n_common_days']:>14}")

# ---------------------------------------------------------------------------
# Section 2: Best and Worst 5 Days (long period)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 2: BEST AND WORST 5 PORTFOLIO DAYS (long period)")
print("=" * 72)

all_dates = sorted(set(orb_long_daily) | set(msite_long_daily) | set(gire_long_daily))
day_rows = []
for d in all_dates:
    orb_p = orb_long_daily.get(d, 0.0)
    msite_p = msite_long_daily.get(d, 0.0)
    gire_p = gire_long_daily.get(d, 0.0)
    total = orb_p + msite_p + gire_p
    day_rows.append({"date": d, "orb": orb_p, "msite": msite_p, "gire": gire_p, "total": total})

day_rows.sort(key=lambda x: x["total"])

print(f"\n  {'Date':<12} {'ORB $':>10} {'MSITE $':>10} {'GIRE $':>10} {'Portfolio $':>12}")
print(f"  {'-'*58}")
print("  -- WORST 5 --")
for row in day_rows[:5]:
    print(f"  {row['date']:<12} {row['orb']:>10,.0f} {row['msite']:>10,.0f} {row['gire']:>10,.0f} {row['total']:>12,.0f}")
print("  -- BEST 5 --")
for row in day_rows[-5:][::-1]:
    print(f"  {row['date']:<12} {row['orb']:>10,.0f} {row['msite']:>10,.0f} {row['gire']:>10,.0f} {row['total']:>12,.0f}")

# ---------------------------------------------------------------------------
# Section 3: Drawdown Decomposition (long period, 3 deepest episodes)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 3: DRAWDOWN DECOMPOSITION (long period)")
print("=" * 72)

port_series = pd.Series(port_long_daily).sort_index()
equity_curve = port_series.cumsum()
peak_curve = equity_curve.cummax()
dd_curve = equity_curve - peak_curve

# Find top-3 drawdown episodes (peak-to-trough)
def find_drawdown_episodes(equity: pd.Series, n: int = 3):
    """Find n deepest distinct drawdown episodes."""
    peak_ser = equity.cummax()
    dd_ser = equity - peak_ser
    episodes = []
    visited = set()

    sorted_idx = dd_ser.sort_values().index.tolist()  # most negative first

    for trough_date in sorted_idx:
        if trough_date in visited:
            continue
        trough_val = dd_ser[trough_date]
        if trough_val >= 0:
            break

        # Find peak: last date where equity == peak before trough
        pre = equity[:trough_date]
        peak_val = float(pre.max())
        peak_date = pre[pre == peak_val].index[-1]

        # Mark all dates in this episode as visited
        ep_dates = dd_ser[peak_date:trough_date].index.tolist()
        for d in ep_dates:
            visited.add(d)

        episodes.append({
            "peak_date": str(peak_date),
            "trough_date": str(trough_date),
            "depth": round(float(trough_val), 2),
        })

        if len(episodes) >= n:
            break

    return episodes

dd_episodes = find_drawdown_episodes(equity_curve, n=3)

print(f"\n  {'#':<3} {'Peak Date':<12} {'Trough Date':<13} {'Depth $':>10}  {'Main Contributor'}")
print(f"  {'-'*65}")
for idx, ep in enumerate(dd_episodes, 1):
    # Find which strategy contributed most to losses in this window
    ep_range = slice(ep["peak_date"], ep["trough_date"])
    orb_ep = sum(orb_long_daily.get(d, 0) for d in port_series[ep_range].index)
    msite_ep = sum(msite_long_daily.get(d, 0) for d in port_series[ep_range].index)
    gire_ep = sum(gire_long_daily.get(d, 0) for d in port_series[ep_range].index)
    contributors = {"ORB": orb_ep, "MSITE": msite_ep, "GIRE": gire_ep}
    worst = min(contributors, key=contributors.get)
    print(f"  {idx:<3} {ep['peak_date']:<12} {ep['trough_date']:<13} {ep['depth']:>10,.0f}  "
          f"{worst} (${contributors[worst]:,.0f})")

# ---------------------------------------------------------------------------
# Section 4: Per-Strategy Trade Distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 4: PER-STRATEGY TRADE DISTRIBUTION")
print("=" * 72)


def trade_dist(pnls: list[float], label: str):
    if not pnls:
        print(f"\n  [{label}]: no trades")
        return {}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    wr = len(wins) / len(pnls)
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    print(f"\n  [{label}]")
    print(f"    N={len(pnls)}  WR={wr:.1%}  AvgWin=${avg_win:,.0f}  AvgLoss=${avg_loss:,.0f}"
          f"  W/L ratio={wl_ratio:.2f}")
    print(f"    MaxWin=${max(pnls):,.0f}  MaxLoss=${min(pnls):,.0f}  ProfitFactor={pf:.2f}")
    return {"N": len(pnls), "WR": round(wr, 4), "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2), "wl_ratio": round(wl_ratio, 4),
            "max_win": round(max(pnls), 2), "max_loss": round(min(pnls), 2),
            "profit_factor": round(pf, 4)}


# ORB (long period)
orb_pnls_long = [t["pnl"] * ORB_SCALE for t in orb_long["trades"]]
orb_dist = trade_dist(orb_pnls_long, "ORB (long period)")

# ORB exit reason counts
orb_reasons = defaultdict(int)
for t in orb_long["trades"]:
    orb_reasons[t.get("reason", "unknown")] += 1
print(f"    Exit reasons: {dict(orb_reasons)}")

# MSITE (long period)
msite_pnls_long = [t["pnl"] * MSITE_SCALE for t in msite_long]
msite_dist = trade_dist(msite_pnls_long, "MSITE (long period)")

# MSITE exit reason counts and avg bars
msite_reasons = defaultdict(int)
msite_bars_list = []
for t in msite_long:
    msite_reasons[t.get("reason", "unknown")] += 1
    if "bars_in_trade" in t:
        msite_bars_list.append(t["bars_in_trade"])
avg_bars = np.mean(msite_bars_list) if msite_bars_list else float("nan")
print(f"    Exit reasons: {dict(msite_reasons)}")
print(f"    Avg bars_in_trade: {avg_bars:.1f}" if not np.isnan(avg_bars) else "    Avg bars_in_trade: N/A")

# GIRE (long period)
gire_pnls_long = [t["pnl_usd"] * GIRE_SCALE for t in gire_trades]
gire_dist = trade_dist(gire_pnls_long, "GIRE (long period)")

# ---------------------------------------------------------------------------
# Section 5: Monthly Hit Rate Table (long period)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 5: MONTHLY HIT RATE TABLE (long period)")
print("=" * 72)


def monthly_pnl_map(daily: dict) -> dict:
    m = defaultdict(float)
    for d, p in daily.items():
        m[d[:7]] += p
    return dict(m)


orb_monthly = monthly_pnl_map(orb_long_daily)
msite_monthly = monthly_pnl_map(msite_long_daily)
gire_monthly = monthly_pnl_map(gire_long_daily)
port_monthly = monthly_pnl_map(port_long_daily)

all_months = sorted(set(orb_monthly) | set(msite_monthly) | set(gire_monthly) | set(port_monthly))

print(f"\n  {'Month':<10} {'ORB $':>10} {'MSITE $':>10} {'GIRE $':>10} {'Portfolio $':>12}  Result")
print(f"  {'-'*60}")
for m in all_months:
    orb_m = orb_monthly.get(m, 0.0)
    msite_m = msite_monthly.get(m, 0.0)
    gire_m = gire_monthly.get(m, 0.0)
    port_m = port_monthly.get(m, 0.0)
    result = "PASS" if port_m > 0 else "FAIL"
    print(f"  {m:<10} {orb_m:>10,.0f} {msite_m:>10,.0f} {gire_m:>10,.0f} {port_m:>12,.0f}  {result}")

# ---------------------------------------------------------------------------
# Section 6: Monte Carlo Combine P(pass) — 10,000 paths
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 6: MONTE CARLO COMBINE P(pass)  [10,000 paths]")
print("=" * 72)

# Use long-period ACTIVE trading days' portfolio daily PnL
port_active = {d: v for d, v in port_long_daily.items() if v != 0}
port_sample = np.array(list(port_active.values()), dtype=float)

START_EQUITY = 50_000.0
TRAIL_DD_LIMIT = 2_000.0
DAILY_LOSS_LIMIT = 1_000.0
PROFIT_TARGET = 3_000.0
N_PATHS = 10_000
WINDOW = 60

rng = np.random.default_rng(42)

n_pass = n_bust_dd = n_bust_daily = n_timeout = 0
days_to_pass = []

for _ in range(N_PATHS):
    equity = START_EQUITY
    peak = START_EQUITY
    busted = False
    passed = False
    days_elapsed = 0

    sample = rng.choice(port_sample, size=WINDOW, replace=True)

    for pnl in sample:
        days_elapsed += 1

        # Daily loss limit check
        if pnl < -DAILY_LOSS_LIMIT:
            n_bust_daily += 1
            busted = True
            break

        equity += pnl
        peak = max(peak, equity)

        # Trailing drawdown check
        if equity < (peak - TRAIL_DD_LIMIT):
            n_bust_dd += 1
            busted = True
            break

        # Pass check
        if equity >= START_EQUITY + PROFIT_TARGET:
            n_pass += 1
            days_to_pass.append(days_elapsed)
            passed = True
            break

    if not busted and not passed:
        n_timeout += 1

p_pass = n_pass / N_PATHS
p_bust_dd = n_bust_dd / N_PATHS
p_bust_daily = n_bust_daily / N_PATHS
p_timeout = n_timeout / N_PATHS
median_days = float(np.median(days_to_pass)) if days_to_pass else float("nan")

print(f"\n  Sample size: {len(port_sample)} active trading days")
print(f"  Daily PnL   min=${port_sample.min():,.0f}  max=${port_sample.max():,.0f}"
      f"  mean=${port_sample.mean():,.0f}  std=${port_sample.std():,.0f}")
print(f"\n  P(pass)          = {p_pass:.1%}")
print(f"  P(bust_dd)       = {p_bust_dd:.1%}")
print(f"  P(bust_daily)    = {p_bust_daily:.1%}")
print(f"  P(timeout)       = {p_timeout:.1%}")
print(f"  Median days/pass = {median_days:.1f}" if not np.isnan(median_days) else "  Median days/pass = N/A")

# ---------------------------------------------------------------------------
# Section 7: MSITE Signal Characteristics (long period)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 7: MSITE SIGNAL CHARACTERISTICS (long period)")
print("=" * 72)

m_vals = [t["M_at_entry"] for t in msite_long if "M_at_entry" in t]
b_vals = [t["B_at_entry"] for t in msite_long if "B_at_entry" in t]

if m_vals:
    m_arr = np.array(m_vals)
    print(f"\n  M_at_entry: min={m_arr.min():.4f}  p25={np.percentile(m_arr, 25):.4f}"
          f"  median={np.median(m_arr):.4f}  p75={np.percentile(m_arr, 75):.4f}"
          f"  max={m_arr.max():.4f}")
    m_median = float(np.median(m_arr))
    # WR when M_at_entry > median vs <= median
    high_m = [t for t in msite_long if "M_at_entry" in t and t["M_at_entry"] > m_median]
    low_m = [t for t in msite_long if "M_at_entry" in t and t["M_at_entry"] <= m_median]
    wr_high_m = sum(1 for t in high_m if t["pnl"] > 0) / max(len(high_m), 1)
    wr_low_m = sum(1 for t in low_m if t["pnl"] > 0) / max(len(low_m), 1)
    print(f"  WR when M > median ({m_median:.4f}): {wr_high_m:.1%}  (N={len(high_m)})")
    print(f"  WR when M <= median:                 {wr_low_m:.1%}  (N={len(low_m)})")
else:
    print("\n  M_at_entry: not present in trade dicts")
    m_arr = np.array([])
    m_median = float("nan")
    wr_high_m = wr_low_m = float("nan")

if b_vals:
    b_arr = np.array(b_vals)
    print(f"\n  B_at_entry: min={b_arr.min():.4f}  p25={np.percentile(b_arr, 25):.4f}"
          f"  median={np.median(b_arr):.4f}  p75={np.percentile(b_arr, 75):.4f}"
          f"  max={b_arr.max():.4f}")
    # WR when B_at_entry > 0 vs <= 0
    pos_b = [t for t in msite_long if "B_at_entry" in t and t["B_at_entry"] > 0]
    neg_b = [t for t in msite_long if "B_at_entry" in t and t["B_at_entry"] <= 0]
    wr_pos_b = sum(1 for t in pos_b if t["pnl"] > 0) / max(len(pos_b), 1)
    wr_neg_b = sum(1 for t in neg_b if t["pnl"] > 0) / max(len(neg_b), 1)
    print(f"  WR when B > 0:  {wr_pos_b:.1%}  (N={len(pos_b)})")
    print(f"  WR when B <= 0: {wr_neg_b:.1%}  (N={len(neg_b)})")
else:
    print("\n  B_at_entry: not present in trade dicts")
    b_arr = np.array([])
    wr_pos_b = wr_neg_b = float("nan")

# ---------------------------------------------------------------------------
# Section 8: Portfolio Summary Stats (YTD and Long)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  SECTION 8: PORTFOLIO SUMMARY STATS")
print("=" * 72)


def summary_stats(daily: dict, label: str) -> dict:
    s = pd.Series(daily).sort_index()
    total = float(s.sum())
    eq = s.cumsum()
    mdd = max_drawdown_series(eq)
    sharpe = sharpe_annualized(daily)
    sortino = sortino_annualized(daily)
    calmar = calmar_ratio(daily)
    active = s[s != 0]
    win_day_pct = float((active > 0).mean()) if len(active) > 0 else 0.0
    max_win_streak, max_loss_streak = win_streaks(active.to_dict())

    print(f"\n  [{label}]")
    print(f"    Total PnL:        ${total:,.0f}")
    print(f"    Sharpe (ann):     {sharpe:.3f}")
    print(f"    Sortino (ann):    {sortino:.3f}")
    print(f"    Max Drawdown:     ${mdd:,.0f}")
    print(f"    Calmar Ratio:     {calmar:.3f}")
    print(f"    Win Day %:        {win_day_pct:.1%}  (of active days)")
    print(f"    Longest Win Str:  {max_win_streak}")
    print(f"    Longest Loss Str: {max_loss_streak}")

    return {
        "total_pnl": round(total, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(mdd, 2),
        "calmar": round(calmar, 4),
        "win_day_pct": round(win_day_pct, 4),
        "longest_win_streak": max_win_streak,
        "longest_loss_streak": max_loss_streak,
    }


ytd_stats = summary_stats(port_ytd_daily, "YTD Portfolio")
long_stats = summary_stats(port_long_daily, "Long-Period Portfolio")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)

def _safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


results = {
    "strategy_correlation": {k: {kk: _safe(vv) for kk, vv in v.items()} for k, v in corr_matrix.items()},
    "best_worst_days": {
        "worst_5": [
            {k: (_safe(v) if not isinstance(v, str) else v) for k, v in row.items()}
            for row in day_rows[:5]
        ],
        "best_5": [
            {k: (_safe(v) if not isinstance(v, str) else v) for k, v in row.items()}
            for row in day_rows[-5:][::-1]
        ],
    },
    "drawdown_episodes": dd_episodes,
    "trade_distribution": {
        "orb_long": orb_dist,
        "orb_exit_reasons": dict(orb_reasons),
        "msite_long": msite_dist,
        "msite_exit_reasons": dict(msite_reasons),
        "msite_avg_bars_in_trade": _safe(avg_bars),
        "gire_long": gire_dist,
    },
    "monthly_table": {
        m: {
            "orb": round(orb_monthly.get(m, 0), 2),
            "msite": round(msite_monthly.get(m, 0), 2),
            "gire": round(gire_monthly.get(m, 0), 2),
            "portfolio": round(port_monthly.get(m, 0), 2),
            "result": "PASS" if port_monthly.get(m, 0) > 0 else "FAIL",
        }
        for m in all_months
    },
    "monte_carlo": {
        "n_paths": N_PATHS,
        "sample_days": int(len(port_sample)),
        "p_pass": round(p_pass, 4),
        "p_bust_dd": round(p_bust_dd, 4),
        "p_bust_daily": round(p_bust_daily, 4),
        "p_timeout": round(p_timeout, 4),
        "median_days_to_pass": _safe(median_days),
    },
    "msite_signal_characteristics": {
        "M_at_entry": {
            "min": _safe(float(m_arr.min())) if len(m_arr) else None,
            "p25": _safe(float(np.percentile(m_arr, 25))) if len(m_arr) else None,
            "median": _safe(float(np.median(m_arr))) if len(m_arr) else None,
            "p75": _safe(float(np.percentile(m_arr, 75))) if len(m_arr) else None,
            "max": _safe(float(m_arr.max())) if len(m_arr) else None,
            "wr_above_median": _safe(wr_high_m),
            "wr_below_median": _safe(wr_low_m),
        },
        "B_at_entry": {
            "min": _safe(float(b_arr.min())) if len(b_arr) else None,
            "p25": _safe(float(np.percentile(b_arr, 25))) if len(b_arr) else None,
            "median": _safe(float(np.median(b_arr))) if len(b_arr) else None,
            "p75": _safe(float(np.percentile(b_arr, 75))) if len(b_arr) else None,
            "max": _safe(float(b_arr.max())) if len(b_arr) else None,
            "wr_b_positive": _safe(wr_pos_b),
            "wr_b_nonpositive": _safe(wr_neg_b),
        },
    },
    "portfolio_summary": {
        "ytd": ytd_stats,
        "long": long_stats,
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"  Results saved -> {RESULTS_PATH}")
print("  Done.")
