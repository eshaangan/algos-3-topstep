"""P1-A validation sweep: IVB body_ratio × VIX threshold on long MES history.

Tests three strategies:
  A) Always 0.35 body_ratio (current live config)
  B) Always 0.25 body_ratio (fully relaxed)
  C) VIX-adaptive: 0.35 on low-VIX days, 0.25 on high-VIX days

Runs on the full MES history (mes_bars.h5 + mes_2026_ytd_rth_5m.h5).
Fetches daily VIX closes via yfinance.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/ivb_body_ratio_sweep.py
    python rule_based_v1/diagnostics/ivb_body_ratio_sweep.py --vix-threshold 20
    python rule_based_v1/diagnostics/ivb_body_ratio_sweep.py --no-vix-data  # skip VIX fetch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = RBV1 / "diagnostics" / "ivb_body_ratio_sweep_results.json"

# Live config constants (match vm_ivb_mes/rules.yaml)
OR_MINUTES         = 40
ENTRY_CUTOFF       = "14:00"
MIN_VOLUME_RATIO   = 1.1
RELOAD_TOL_TICKS   = 4
TARGET_RANGE_MULT  = 1.0
MODE               = "both"
SKIP_MONDAY        = True
TIME_STOP_BARS     = 12

POINT_VALUE        = 5.0
TICK_SIZE          = 0.25
COMMISSION         = 0.62
SLIPPAGE_TICKS     = 1
N_CONTRACTS        = 2
MAX_TRADES_PER_DAY = 3
MAX_DAILY_LOSS     = -900.0
PER_TRADE_MAX_LOSS = 500.0
MAX_CONSEC_LOSSES  = 2
COOLDOWN_BARS      = 5
STARTING_EQUITY    = 50_000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mes_full() -> pd.DataFrame:
    parts = []
    for path, key in [
        (ROOT / "data/processed/mes_bars.h5", "bars_5min"),
        (ROOT / "data/processed/mes_2026_ytd_rth_5m.h5", "bars_5min"),
    ]:
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        df = pd.read_hdf(str(path), key=key)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns=str.lower)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        df = df[["open", "high", "low", "close", "volume"]]
        parts.append(df)

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    # RTH only
    rth = ((combined.index.hour > 9) | ((combined.index.hour == 9) & (combined.index.minute >= 30))) \
          & (combined.index.hour < 16)
    return combined[rth]


def fetch_vix_daily(start: str, end: str) -> dict[str, float]:
    """Return {date_str: prior_day_vix_close} for the given range."""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if vix.empty:
            return {}
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        # Flatten multi-level columns if present (newer yfinance versions)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        closes = vix["Close"].squeeze()  # ensure Series
        shifted = closes.shift(1)  # prior close = VIX known before session open
        return {str(d.date()): float(v) for d, v in shifted.items() if not np.isnan(float(v))}
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _vp(bars: pd.DataFrame, bins: int = 24, va: float = 0.70) -> dict:
    lo, hi = float(bars["low"].min()), float(bars["high"].max())
    if hi <= lo:
        return {"poc": hi, "vah": hi, "val": lo}
    step = max(1e-6, (hi - lo) / bins)
    edges = np.arange(lo, hi + step, step)
    hist = np.zeros(len(edges) - 1)
    for _, b in bars.iterrows():
        t = np.where((edges[:-1] <= float(b["high"])) & (edges[1:] >= float(b["low"])))[0]
        if len(t):
            hist[t] += max(float(b["volume"]), 0.0) / len(t)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() <= 0:
        return {"poc": float(centers[len(centers) // 2]), "vah": hi, "val": lo}
    pi = int(hist.argmax())
    sel = {pi}
    tot = hist[pi]
    tgt = hist.sum() * va
    l, r = pi - 1, pi + 1
    while tot < tgt and (l >= 0 or r < len(hist)):
        lv = hist[l] if l >= 0 else -1.0
        rv = hist[r] if r < len(hist) else -1.0
        if rv >= lv:
            sel.add(r); tot += max(rv, 0); r += 1
        else:
            sel.add(l); tot += max(lv, 0); l -= 1
    return {"poc": float(centers[pi]), "vah": float(edges[max(sel) + 1]), "val": float(edges[min(sel)])}


def _agg(bar: pd.Series, avg_vol: float, d: int, min_body_ratio: float, rm: float = 1.0) -> bool:
    if avg_vol <= 0:
        return False
    body = float(bar["close"] - bar["open"]) * d
    rng = max(float(bar["high"] - bar["low"]), 1e-9)
    return (body / rng) >= min_body_ratio and float(bar["volume"]) >= avg_vol * MIN_VOLUME_RATIO * rm


def _slip(p: float, d: int, entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return p + s * d if entry else p - s * d


def _pnl(ep: float, xp: float, d: int, c: int) -> float:
    return (xp - ep) * d * c * POINT_VALUE - 2 * COMMISSION * c


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------
def run_backtest(bars: pd.DataFrame, vix_map: dict, vix_threshold: float,
                 body_ratio_normal: float, body_ratio_high_vix: float) -> dict:
    """Run IVB backtest with VIX-adaptive body_ratio."""
    entry_t = pd.Timestamp(ENTRY_CUTOFF).time()
    ss = pd.Timestamp("09:30").time()
    se = pd.Timestamp("16:00").time()

    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    max_dd = 0.0
    trades: list[dict] = []
    daily_pnl: dict = {}
    cl = 0
    cool = 0

    for date, day in bars.groupby(bars.index.date):
        if SKIP_MONDAY and pd.Timestamp(date).dayofweek == 0:
            daily_pnl[date] = 0.0
            continue

        # Select body_ratio for today
        vix_today = vix_map.get(str(date))
        if vix_today is not None and vix_today >= vix_threshold:
            body_ratio = body_ratio_high_vix
        else:
            body_ratio = body_ratio_normal

        rth = day[(day.index.time >= ss) & (day.index.time < se)]
        if len(rth) < 10:
            daily_pnl[date] = 0.0
            continue
        st = rth.index[0]
        oret = st + pd.Timedelta(minutes=OR_MINUTES)
        orb = rth[rth.index < oret]
        if len(orb) < 4:
            daily_pnl[date] = 0.0
            continue
        orh = float(orb["high"].max())
        orl = float(orb["low"].min())
        orr = orh - orl
        if orr <= TICK_SIZE * 4:
            daily_pnl[date] = 0.0
            continue
        vp = _vp(orb)
        av = float(orb["volume"].mean())
        rt = RELOAD_TOL_TICKS * TICK_SIZE

        pos = None
        tt = 0
        dl = 0.0
        bu = False
        bd = False

        for i, (ts, bar) in enumerate(rth.iterrows()):
            if pos:
                h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                sess = (ts == rth.index[-1]) or ts.time() >= pd.Timestamp("15:55").time()
                ex, xp, rsn = False, 0.0, ""
                if sess:
                    ex, xp, rsn = True, _slip(c, pos[0], False), "session_close"
                elif i >= pos[4]:
                    ex, xp, rsn = True, _slip(c, pos[0], False), "time_stop"
                elif pos[0] == 1:
                    if l <= pos[2]:
                        ex, xp, rsn = True, _slip(pos[2], 1, False), "stop_loss"
                    elif h >= pos[3]:
                        ex, xp, rsn = True, _slip(pos[3], 1, False), "profit_target"
                else:
                    if h >= pos[2]:
                        ex, xp, rsn = True, _slip(pos[2], -1, False), "stop_loss"
                    elif l <= pos[3]:
                        ex, xp, rsn = True, _slip(pos[3], -1, False), "profit_target"
                if ex:
                    p = _pnl(pos[1], xp, pos[0], N_CONTRACTS)
                    equity += p
                    peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak)
                    dl += p
                    trades.append({
                        "date": str(date),
                        "dir": "long" if pos[0] == 1 else "short",
                        "pnl": round(p, 2),
                        "reason": rsn,
                        "body_ratio_used": round(body_ratio, 3),
                        "vix": round(vix_today, 1) if vix_today else None,
                    })
                    cl = cl + 1 if p < 0 else 0
                    if p < 0:
                        cool = COOLDOWN_BARS
                    pos = None

            if ts < oret:
                continue
            if pos or tt >= MAX_TRADES_PER_DAY or ts.time() > entry_t:
                continue
            if dl <= MAX_DAILY_LOSS:
                continue
            if cl >= MAX_CONSEC_LOSSES:
                if cool > 0:
                    cool -= 1
                    continue
                else:
                    cl = 0
            if cool > 0:
                cool -= 1
                continue

            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            if close > orh:
                bu = True
            if close < orl:
                bd = True

            d, em = 0, ""
            if MODE in {"breakout", "both"}:
                if close > orh and _agg(bar, av, 1, body_ratio):
                    d, em = 1, "breakout"
                elif close < orl and _agg(bar, av, -1, body_ratio):
                    d, em = -1, "breakout"
            if d == 0 and MODE in {"reload", "both"}:
                if bu and low <= vp["vah"] + rt and close > vp["vah"] and _agg(bar, av, 1, body_ratio, 0.9):
                    d, em = 1, "reload"
                elif bd and high >= vp["val"] - rt and close < vp["val"] and _agg(bar, av, -1, body_ratio, 0.9):
                    d, em = -1, "reload"
            if d == 0:
                continue

            ep = _slip(close, d, True)
            sl = min(vp["val"], orl) - TICK_SIZE if d == 1 else max(vp["vah"], orh) + TICK_SIZE
            pt = ep + TARGET_RANGE_MULT * orr if d == 1 else ep - TARGET_RANGE_MULT * orr
            risk = abs(ep - sl)
            if risk <= TICK_SIZE or risk * N_CONTRACTS * POINT_VALUE > PER_TRADE_MAX_LOSS:
                continue
            pos = (d, ep, sl, pt, i + TIME_STOP_BARS)
            tt += 1

        daily_pnl[date] = dl

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    daily_s = pd.Series({k: v for k, v in daily_pnl.items() if v != 0}, dtype=float)
    sharpe = float(daily_s.mean() / daily_s.std() * np.sqrt(252)) if len(daily_s) > 1 and daily_s.std() > 0 else 0.0

    from collections import Counter
    reasons = dict(Counter(t["reason"] for t in trades))

    return {
        "num_trades": int(len(trades)),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "total_pnl": round(float(pnls.sum()), 2) if len(pnls) else 0.0,
        "max_drawdown": round(float(max_dd), 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if len(losses) else 0.0,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "active_days": int((daily_s != 0).sum()),
        "exit_reasons": reasons,
        "trades": trades,
    }


def summarize(label: str, r: dict, vix_threshold: float = None, br_normal: float = None, br_high: float = None):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    if vix_threshold:
        print(f"  body_ratio: {br_normal} normal | {br_high} when VIX≥{vix_threshold}")
    print(f"{'─'*60}")
    print(f"  Trades      : {r['num_trades']}  ({r['active_days']} active days)")
    print(f"  Win Rate    : {r['win_rate']:.1%}")
    print(f"  Total PnL   : ${r['total_pnl']:,.2f}")
    print(f"  Avg Win     : ${r['avg_win']:,.2f}")
    print(f"  Avg Loss    : ${r['avg_loss']:,.2f}")
    print(f"  Profit Fac  : {r['profit_factor']:.2f}")
    print(f"  Sharpe      : {r['sharpe']:.2f}")
    print(f"  Max DrawDown: ${r['max_drawdown']:,.2f}")
    print(f"  Exit reasons: {r['exit_reasons']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: all history)")
    parser.add_argument("--vix-threshold", type=float, default=25.0)
    parser.add_argument("--no-vix-data", action="store_true", help="Skip VIX fetch (use body_ratio=0.35 for all)")
    args = parser.parse_args()

    bars = load_mes_full()
    if args.start:
        bars = bars[bars.index >= pd.Timestamp(args.start, tz="US/Eastern")]

    date_min = str(bars.index[0].date())
    date_max = str(bars.index[-1].date())
    print(f"\nLoaded {len(bars):,} RTH 5-min bars  {date_min} → {date_max}")

    # Fetch VIX
    vix_map: dict = {}
    if not args.no_vix_data:
        print("Fetching VIX daily closes...")
        vix_map = fetch_vix_daily(date_min, date_max)
        if vix_map:
            vix_vals = list(vix_map.values())
            print(f"  VIX coverage: {len(vix_map)} days  min={min(vix_vals):.1f}  max={max(vix_vals):.1f}  mean={np.mean(vix_vals):.1f}")
            high_vix_days = sum(1 for v in vix_vals if v >= args.vix_threshold)
            print(f"  High-VIX days (≥{args.vix_threshold}): {high_vix_days} ({high_vix_days/len(vix_vals):.1%})")
        else:
            print("  VIX fetch returned no data — proceeding without VIX-adaptive logic")

    # Strategy A: always 0.35 (current baseline)
    print("\nRunning A: baseline (body_ratio=0.35 always)...")
    r_a = run_backtest(bars, vix_map={}, vix_threshold=999, body_ratio_normal=0.35, body_ratio_high_vix=0.35)

    # Strategy B: always 0.25 (fully relaxed)
    print("Running B: fully relaxed (body_ratio=0.25 always)...")
    r_b = run_backtest(bars, vix_map={}, vix_threshold=999, body_ratio_normal=0.25, body_ratio_high_vix=0.25)

    # Strategy C: VIX-adaptive at the specified threshold
    print(f"Running C: VIX-adaptive (0.35 normal, 0.25 when VIX≥{args.vix_threshold})...")
    r_c = run_backtest(bars, vix_map=vix_map, vix_threshold=args.vix_threshold,
                       body_ratio_normal=0.35, body_ratio_high_vix=0.25)

    # Also test a sweep of thresholds
    print("\nRunning threshold sweep: VIX thresholds 20/25/30 × body ratios...")
    sweep_results = []
    for threshold in [20.0, 25.0, 30.0]:
        for br_high in [0.20, 0.25, 0.30]:
            r = run_backtest(bars, vix_map=vix_map, vix_threshold=threshold,
                             body_ratio_normal=0.35, body_ratio_high_vix=br_high)
            sweep_results.append({
                "vix_threshold": threshold,
                "body_ratio_normal": 0.35,
                "body_ratio_high_vix": br_high,
                **{k: v for k, v in r.items() if k != "trades"},
            })

    summarize("A — Baseline (body_ratio=0.35 always)", r_a)
    summarize("B — Fully Relaxed (body_ratio=0.25 always)", r_b)
    summarize(f"C — VIX-Adaptive (threshold={args.vix_threshold})", r_c,
              args.vix_threshold, 0.35, 0.25)

    print(f"\n{'='*70}")
    print(f"  Threshold sweep  (body_ratio_normal=0.35)")
    print(f"{'='*70}")
    print(f"  {'VIX≥':>6} {'BR_hv':>7} | {'N':>4} {'WR':>6} {'PnL':>9} {'PF':>5} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'-'*65}")
    for s in sorted(sweep_results, key=lambda x: -x["sharpe"]):
        print(
            f"  {s['vix_threshold']:>6.0f} {s['body_ratio_high_vix']:>7.2f} | "
            f"{s['num_trades']:>4} {s['win_rate']:>6.1%} ${s['total_pnl']:>8,.0f} "
            f"{s['profit_factor']:>5.2f} {s['sharpe']:>7.2f} ${s['max_drawdown']:>7,.0f}"
        )

    out = {
        "baseline_0.35": {k: v for k, v in r_a.items() if k != "trades"},
        "relaxed_0.25": {k: v for k, v in r_b.items() if k != "trades"},
        f"adaptive_vix{args.vix_threshold}": {k: v for k, v in r_c.items() if k != "trades"},
        "threshold_sweep": sweep_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
