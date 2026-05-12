"""Volatility Clustering (VC) Filter Backtest for MNQ ORB.

Tests the Oxford Strategies finding: large daily range days tend to be followed
by continued moves in the same direction. Applied as a pre-filter on the live ORB.

Variants tested:
  baseline        - plain ORB (no VC filter)
  vc_wide_only    - only trade after prior day range > rolling median (no direction bias)
  vc_directional  - LONG after wide up day, SHORT after wide down day
  vc_strict       - prior day range > 75th percentile (more selective)
  vc_loose        - prior day range > 25th percentile (less selective)
  vc_dir_longonly - directional filter but accept LONG signals only

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/vc_filter_backtest.py [--fetch] [--start 2026-01-01]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.risk_manager import RiskManager, TradeRecord
from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CACHE_PATH = ROOT / "data" / "processed" / "mnq_vc_backtest_5min.parquet"
RESULTS_PATH = RBV1 / "diagnostics" / "vc_filter_results.json"

# ── Strategy config (mirrors live vm_orb_sidecar) ───────────────────────────
OR_END_TIME      = "10:04"
MIN_OR_BARS      = 7
PT_MULT          = 3.0
SL_MULT          = 1.5
ENTRY_CUTOFF     = "12:00"
MIN_RANGE_ATR    = 0.3
ATR_PERIOD       = 14
TIME_STOP_BARS   = 24
TRAILING_ACT_ATR = 999.0
TRAILING_DIST_ATR= 0.75

POINT_VALUE      = 2.0
TICK_SIZE        = 0.25
TICK_VALUE       = 0.50
COMMISSION       = 0.62
SLIPPAGE_TICKS   = 1

N_CONTRACTS      = 2
MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS   = -950.0
DRAWDOWN_BUFFER  = 1_950.0
PER_TRADE_MAX_LOSS = 1_000.0
COOLDOWN_BARS    = 3
MAX_CONSEC_LOSSES = 10
STARTING_EQUITY  = 50_000.0

# ── Volatility Clustering config ─────────────────────────────────────────────
VC_LOOKBACK = 10   # rolling N-day window for range median/percentile


# ── Data fetch ───────────────────────────────────────────────────────────────
def fetch(start: str = "2025-08-01", end: str | None = None) -> pd.DataFrame:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")
    import databento as db
    if end is None:
        from datetime import timezone, datetime, timedelta
        today = datetime.now(tz=timezone.utc).date()
        cutoff = today - timedelta(days=2)
        end = f"{cutoff}T21:00:00+00:00"

    log.info(f"Fetching MNQ.c.0  {start} → {end} ...")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=["MNQ.c.0"], schema="ohlcv-1m",
        start=start, end=end, stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    log.info(f"Fetched {len(df):,} 1-min bars")

    df5 = df.resample("5min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"])
    df5.index = df5.index.tz_convert("US/Eastern")
    rth = ((df5.index.hour > 9) | ((df5.index.hour == 9) & (df5.index.minute >= 30))) \
          & (df5.index.hour < 16)
    df5 = df5.loc[rth]
    log.info(f"After RTH filter: {len(df5):,} 5-min bars  {df5.index[0].date()} → {df5.index[-1].date()}")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df5.to_parquet(CACHE_PATH)
    log.info(f"Cached → {CACHE_PATH}")
    return df5


# ── Volatility Clustering: daily metadata ────────────────────────────────────
def build_vc_meta(bars: pd.DataFrame, lookback: int = VC_LOOKBACK) -> dict:
    """For each trading date, compute prior-day's range and rolling percentile rank.

    Returns dict[date] -> {
        "prev_range":      float,   prior session high - low
        "prev_direction":  int,     +1 up day, -1 down day, 0 flat
        "range_pct_rank":  float,   [0,1] where 1 = widest day in window
        "wide_day":        bool,    range > rolling median
        "strict_wide":     bool,    range > 75th percentile
        "loose_wide":      bool,    range > 25th percentile
    }
    """
    dates = sorted(set(bars.index.date))
    daily: dict = {}
    for d in dates:
        mask = bars.index.date == d
        day = bars[mask]
        if len(day) == 0:
            continue
        daily[d] = {
            "range": float(day["high"].max() - day["low"].min()),
            "open":  float(day["open"].iloc[0]),
            "close": float(day["close"].iloc[-1]),
        }
        daily[d]["direction"] = (
            1 if daily[d]["close"] > daily[d]["open"] else
            -1 if daily[d]["close"] < daily[d]["open"] else 0
        )

    meta: dict = {}
    for i, d in enumerate(dates):
        if i == 0:
            meta[d] = None
            continue
        prev_d = dates[i - 1]
        prev = daily.get(prev_d)
        if prev is None:
            meta[d] = None
            continue

        # Rolling window of prior N days (not including today)
        window_days = dates[max(0, i - lookback): i]
        window_ranges = [daily[wd]["range"] for wd in window_days if wd in daily]
        if len(window_ranges) < 3:
            meta[d] = None
            continue

        wr = np.array(window_ranges)
        p25, p50, p75 = np.percentile(wr, [25, 50, 75])
        pr = prev["range"]
        pct_rank = float(np.mean(wr < pr))

        meta[d] = {
            "prev_range":     pr,
            "prev_direction": prev["direction"],
            "range_pct_rank": pct_rank,
            "wide_day":       pr > p50,
            "strict_wide":    pr > p75,
            "loose_wide":     pr > p25,
        }
    return meta


# ── Simulation helpers ────────────────────────────────────────────────────────
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float
    trailing_active: bool = False
    trailing_stop: float = 0.0
    peak_favorable: float = 0.0


def _slip(price, direction, is_entry):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry, exit_, direction):
    raw = (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE
    return raw - 2 * COMMISSION * N_CONTRACTS


def _check_exit(pos: Position, bar, idx: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close or idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop" if not sess_close else "session_close"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
        if not pos.trailing_active and h - pos.entry_price >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active:
            if h > pos.peak_favorable:
                pos.peak_favorable = h
                pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
            if l <= pos.trailing_stop:
                return True, _slip(pos.trailing_stop, 1, False), "trailing_stop"
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
        if not pos.trailing_active and pos.entry_price - l >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active:
            if l < pos.peak_favorable:
                pos.peak_favorable = l
                pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
            if h >= pos.trailing_stop:
                return True, _slip(pos.trailing_stop, -1, False), "trailing_stop"
    return False, 0.0, ""


# ── Core backtest engine ──────────────────────────────────────────────────────
def run_backtest(bars: pd.DataFrame, vc_meta: dict, variant: str) -> dict:
    """Run ORB backtest with a specific VC filter variant.

    variant options:
        "baseline"        no filter
        "vc_wide_only"    prior day range > median, any direction
        "vc_directional"  prior day range > median AND signal direction = prior day direction
        "vc_strict"       prior day range > 75th percentile, any direction
        "vc_loose"        prior day range > 25th percentile, any direction
        "vc_dir_longonly" directional + LONG only
    """
    long_only = variant == "vc_dir_longonly"
    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME, min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR, entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD, long_only=long_only,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    rm = RiskManager(
        contracts=N_CONTRACTS, point_value=POINT_VALUE, tick_size=TICK_SIZE,
        tick_value=TICK_VALUE, max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=PER_TRADE_MAX_LOSS,
        max_consecutive_losses=MAX_CONSEC_LOSSES,
        cooldown_bars=COOLDOWN_BARS, drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    atr_s = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos = None
    trades: list[TradeRecord] = []
    equity = STARTING_EQUITY
    eq_vals, eq_times = [STARTING_EQUITY], [bars.index[0]]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bt = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern") if bt.tzinfo else bt
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (
            (bars.index[i+1].tz_convert("US/Eastern") if bars.index[i+1].tzinfo
             else bars.index[i+1]).date() != bdate
        )
        sess_close = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                p = _pnl(pos.entry_price, exit_p, pos.direction)
                tr = TradeRecord(
                    entry_bar=pos.entry_bar_idx, exit_bar=i,
                    direction=pos.direction, entry_price=pos.entry_price,
                    exit_price=exit_p, pnl=p, exit_reason=reason,
                )
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt)
                pos = None

        if pos is None and not sess_close and trades_today < MAX_TRADES_PER_DAY:
            ok, _ = rm.can_trade()
            if not ok:
                continue

            # ── VC filter gate ──────────────────────────────────────────────
            meta = vc_meta.get(bdate)
            if variant != "baseline":
                if meta is None:
                    continue  # not enough history yet

                if variant == "vc_wide_only" and not meta["wide_day"]:
                    continue
                elif variant == "vc_strict" and not meta["strict_wide"]:
                    continue
                elif variant == "vc_loose" and not meta["loose_wide"]:
                    continue
                elif variant in ("vc_directional", "vc_dir_longonly") and not meta["wide_day"]:
                    continue
            # ── Signal evaluation ───────────────────────────────────────────
            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg.evaluate(lookback)
            if not dec.should_trade:
                continue

            # For directional variants, gate signal direction vs. prior day
            if variant in ("vc_directional", "vc_dir_longonly") and meta is not None:
                prev_dir = meta["prev_direction"]
                if prev_dir != 0 and dec.direction != prev_dir:
                    continue

            cur_atr = atr_s.iloc[i]
            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            ep  = _slip(bar["close"], dec.direction, True)
            sl  = rm.compute_stop_price(ep, dec.direction, cur_atr, SL_MULT)
            pt  = rm.compute_target_price(ep, dec.direction, cur_atr, PT_MULT)
            pos = Position(
                direction=dec.direction, entry_price=ep, entry_bar_idx=i,
                stop_loss=sl, profit_target=pt,
                time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr,
            )
            trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total  = sum(t.pnl for t in trades)
    gp     = sum(t.pnl for t in wins)
    gl     = abs(sum(t.pnl for t in losses))

    eq = pd.Series(eq_vals, index=eq_times)
    max_dd = float((eq - eq.cummax()).min()) if len(eq) > 1 else 0.0

    daily = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    n_days = len(set(bars.index.date))
    trade_days = len([v for v in daily_pnl.values() if v != 0])

    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1

    return {
        "variant":       variant,
        "n_trades":      len(trades),
        "n_trading_days": n_days,
        "trades_per_day": round(len(trades) / n_days, 2) if n_days else 0,
        "win_rate":      round(len(wins) / len(trades), 3) if trades else 0,
        "total_pnl":     round(total, 2),
        "avg_win":       round(gp / len(wins), 2) if wins else 0,
        "avg_loss":      round(-gl / len(losses), 2) if losses else 0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 2),
        "exit_reasons":  dict(reasons),
        "daily_pnl":     {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }


# ── Printing ──────────────────────────────────────────────────────────────────
HEADER = ["variant", "trades", "WR%", "PnL", "PF", "Sharpe", "MaxDD", "AvgW", "AvgL"]
FMT    = "{:<22} {:>7} {:>6} {:>9} {:>6} {:>7} {:>9} {:>7} {:>7}"


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  VOLATILITY CLUSTERING FILTER COMPARISON — MNQ ORB")
    print("=" * 80)
    print(FMT.format(*HEADER))
    print("-" * 80)
    for r in results:
        pf  = f"{r['profit_factor']:.2f}" if r["profit_factor"] else "∞"
        print(FMT.format(
            r["variant"],
            r["n_trades"],
            f"{r['win_rate']*100:.1f}%",
            f"${r['total_pnl']:,.0f}",
            pf,
            f"{r['sharpe']:.2f}",
            f"${r['max_drawdown']:,.0f}",
            f"${r['avg_win']:,.0f}",
            f"${r['avg_loss']:,.0f}",
        ))
    print("=" * 80)

    # Delta vs baseline
    base = next((r for r in results if r["variant"] == "baseline"), None)
    if base and len(results) > 1:
        print("\n  DELTA vs BASELINE")
        print(FMT.format(*HEADER))
        print("-" * 80)
        for r in results:
            if r["variant"] == "baseline":
                continue
            dt = r["n_trades"] - base["n_trades"]
            dwr = (r["win_rate"] - base["win_rate"]) * 100
            dpnl = r["total_pnl"] - base["total_pnl"]
            dsh = r["sharpe"] - base["sharpe"]
            ddd = r["max_drawdown"] - base["max_drawdown"]
            print(FMT.format(
                r["variant"],
                f"{dt:+d}",
                f"{dwr:+.1f}pp",
                f"${dpnl:+,.0f}",
                "—",
                f"{dsh:+.2f}",
                f"${ddd:+,.0f}",
                "—", "—",
            ))
        print("=" * 80)


# ── VC regime breakdown ───────────────────────────────────────────────────────
def print_vc_regime_breakdown(bars: pd.DataFrame, vc_meta: dict) -> None:
    """Show how often each regime occurs and ORB base WR by regime."""
    dates = sorted(set(bars.index.date))
    wide, narrow, no_meta = 0, 0, 0
    for d in dates:
        m = vc_meta.get(d)
        if m is None:
            no_meta += 1
        elif m["wide_day"]:
            wide += 1
        else:
            narrow += 1
    total = wide + narrow + no_meta
    print(f"\n  VC Regime Distribution ({total} trading days):")
    print(f"    Wide day (range > median) : {wide:>4}  ({wide/total*100:.0f}%)")
    print(f"    Narrow day                : {narrow:>4}  ({narrow/total*100:.0f}%)")
    print(f"    No prior data             : {no_meta:>4}  ({no_meta/total*100:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Force re-fetch from Databento")
    parser.add_argument("--start", default="2025-08-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None)
    parser.add_argument("--recent-only", action="store_true",
                        help="Also print 2026-only sub-period results")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.fetch or not CACHE_PATH.exists():
        bars = fetch(start=args.start, end=args.end)
    else:
        log.info(f"Loading cached data from {CACHE_PATH}")
        bars = pd.read_parquet(CACHE_PATH)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("US/Eastern")
        else:
            bars.index = bars.index.tz_convert("US/Eastern")

    if args.start:
        bars = bars[bars.index >= pd.Timestamp(args.start, tz="US/Eastern")]
    if args.end:
        bars = bars[bars.index < pd.Timestamp(args.end, tz="US/Eastern")]

    log.info(f"Dataset: {len(bars):,} bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    # ── Build VC metadata ─────────────────────────────────────────────────────
    vc_meta = build_vc_meta(bars, lookback=VC_LOOKBACK)

    # ── Run all variants ──────────────────────────────────────────────────────
    VARIANTS = [
        "baseline",
        "vc_wide_only",
        "vc_directional",
        "vc_strict",
        "vc_loose",
        "vc_dir_longonly",
    ]
    results = []
    for v in VARIANTS:
        log.info(f"Running variant: {v}")
        r = run_backtest(bars, vc_meta, v)
        results.append(r)
        log.info(f"  {v}: {r['n_trades']} trades  WR={r['win_rate']:.1%}  PnL=${r['total_pnl']:,.0f}  Sharpe={r['sharpe']:.2f}")

    print_table(results)
    print_vc_regime_breakdown(bars, vc_meta)

    # ── 2026-only sub-period ──────────────────────────────────────────────────
    if args.recent_only or True:
        bars_2026 = bars[bars.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
        if len(bars_2026) > 100:
            vc_meta_2026 = build_vc_meta(bars_2026, lookback=VC_LOOKBACK)
            log.info("\n  --- 2026 YTD sub-period ---")
            results_2026 = []
            for v in VARIANTS:
                r = run_backtest(bars_2026, vc_meta_2026, v)
                r["variant"] = v + " [2026]"
                results_2026.append(r)
            print("\n  === 2026 YTD ONLY ===")
            print_table(results_2026)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "full_period": {r["variant"]: r for r in results},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
