"""2026 YTD IVB ORB backtest — MES, exact live config from vm_ivb_mes/.

Mirrors the deployed strategy on topstep-ivb-mes-vm:
  - or_minutes=40, vol_ratio=1.1, target=1.0x OR-range, mode=both, skip_monday=True
  - Stop: OR-floor (min(VAL, or_low) - tick for LONG; max(VAH, or_high) + tick for SHORT)
  - Risk: 3 contracts, max_daily_loss=-$900, circuit_breaker=2 consec losses, cooldown=5 bars

Uses cached mes_2026_ytd_rth_5m.h5 (RTH bars). Re-fetches if --fetch flag is passed.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/fetch_backtest_ivb_2026ytd.py
    python rule_based_v1/diagnostics/fetch_backtest_ivb_2026ytd.py --fetch
    python rule_based_v1/diagnostics/fetch_backtest_ivb_2026ytd.py --contracts 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rules.ivb_orb_rule import IVBORBRule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RTH_PATH     = ROOT / "data" / "processed" / "mes_2026_ytd_rth_5m.h5"
RAW_PATH     = ROOT / "data" / "processed" / "mes_2026_ytd_5m.h5"
RESULTS_PATH = RBV1 / "diagnostics" / "2026ytd_ivb_results.json"

# ---------------------------------------------------------------------------
# Live config — mirrors vm_ivb_mes/rules.yaml + risk.yaml exactly
# ---------------------------------------------------------------------------
OR_MINUTES         = 40
ENTRY_CUTOFF       = "14:00"
MIN_VOLUME_RATIO   = 1.1
RELOAD_TOL_TICKS   = 4
TARGET_RANGE_MULT  = 1.0
MODE               = "both"
SKIP_MONDAY        = True
TIME_STOP_BARS     = 18

POINT_VALUE        = 5.0
TICK_SIZE          = 0.25
TICK_VALUE         = 1.25
COMMISSION         = 0.62   # per side per contract
SLIPPAGE_TICKS     = 1

N_CONTRACTS        = 3
MAX_TRADES_PER_DAY = 3
MAX_DAILY_LOSS     = -900.0
PER_TRADE_MAX_LOSS = 500.0
MAX_CONSEC_LOSSES  = 2
COOLDOWN_BARS      = 5
STARTING_EQUITY    = 50_000.0
DRAWDOWN_BUFFER    = 1_800.0


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def fetch_mes(start: str = "2026-01-01") -> pd.DataFrame:
    from datetime import timezone, datetime, timedelta
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")

    import databento as db
    # 2 days ago to stay within free historical window
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=2)).date()
    end = f"{cutoff}T21:00:00+00:00"

    logger.info(f"Fetching MES.c.0  {start} → {end} ...")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MES.c.0"],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info(f"Fetched {len(df):,} 1-min bars  {df.index[0].date()} → {df.index[-1].date()}")

    df5 = df.resample("5min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"])
    df5.index = df5.index.tz_convert("US/Eastern")
    rth = ((df5.index.hour > 9) | ((df5.index.hour == 9) & (df5.index.minute >= 30))) \
          & (df5.index.hour < 16)
    df5_rth = df5.loc[rth]

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df5.to_hdf(str(RAW_PATH), key="bars_5min", mode="w", complevel=5)
    df5_rth.to_hdf(str(RTH_PATH), key="bars_rth_5m", mode="w", complevel=5)
    logger.info(f"Saved RTH → {RTH_PATH}  ({len(df5_rth):,} bars)")
    return df5_rth


def load_rth() -> pd.DataFrame:
    with pd.HDFStore(str(RTH_PATH), "r") as store:
        keys = [k.strip("/") for k in store.keys()]
    key = next((k for k in ("bars_rth_5m", "bars_5min", "bars") if k in keys), keys[0])
    bars = pd.read_hdf(str(RTH_PATH), key=key)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")
    return bars.sort_index()


# ---------------------------------------------------------------------------
# Volume profile (same as IVBORBRule._volume_profile)
# ---------------------------------------------------------------------------
def _volume_profile(bars: pd.DataFrame, bins: int = 24, value_area_pct: float = 0.70) -> dict:
    lo = float(bars["low"].min())
    hi = float(bars["high"].max())
    if hi <= lo:
        return {"poc": hi, "vah": hi, "val": lo}
    step = max(1e-6, (hi - lo) / bins)
    edges = np.arange(lo, hi + step, step)
    if len(edges) < 2:
        edges = np.array([lo, lo + 1e-6])
    hist = np.zeros(len(edges) - 1)
    for _, bar in bars.iterrows():
        touched = np.where((edges[:-1] <= float(bar["high"])) & (edges[1:] >= float(bar["low"])))[0]
        if len(touched):
            hist[touched] += max(float(bar["volume"]), 0.0) / len(touched)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() <= 0:
        return {"poc": float(centers[len(centers) // 2]), "vah": hi, "val": lo}
    poc_idx = int(hist.argmax())
    selected = {poc_idx}
    total, target = hist[poc_idx], hist.sum() * value_area_pct
    left, right = poc_idx - 1, poc_idx + 1
    while total < target and (left >= 0 or right < len(hist)):
        lv = hist[left] if left >= 0 else -1.0
        rv = hist[right] if right < len(hist) else -1.0
        if rv >= lv:
            selected.add(right); total += max(rv, 0); right += 1
        else:
            selected.add(left); total += max(lv, 0); left -= 1
    return {"poc": float(centers[poc_idx]), "vah": float(edges[max(selected)+1]), "val": float(edges[min(selected)])}


def _aggression(bar: pd.Series, avg_volume: float, direction: int, ratio_mult: float = 1.0) -> bool:
    if avg_volume <= 0:
        return False
    body = float(bar["close"] - bar["open"]) * direction
    rng = max(float(bar["high"] - bar["low"]), 1e-9)
    return (body / rng) >= 0.35 and float(bar["volume"]) >= avg_volume * MIN_VOLUME_RATIO * ratio_mult


def _slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry: float, exit_: float, direction: int, contracts: int) -> float:
    return (exit_ - entry) * direction * contracts * POINT_VALUE - 2 * COMMISSION * contracts


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    or_range: float


def run_backtest(bars: pd.DataFrame, contracts: int = N_CONTRACTS) -> tuple:
    or_end_minutes = OR_MINUTES
    entry_cutoff_t = pd.Timestamp(ENTRY_CUTOFF).time()
    session_start_t = pd.Timestamp("09:30").time()
    session_end_t   = pd.Timestamp("16:00").time()
    reload_tol      = RELOAD_TOL_TICKS * TICK_SIZE

    equity = STARTING_EQUITY
    peak_equity = STARTING_EQUITY
    max_dd = 0.0
    trades: list[dict] = []
    daily_pnl: dict = {}
    eq_curve: list[tuple] = [(bars.index[0], equity)]

    consec_losses = 0
    cooldown_remaining = 0

    for date, day_bars in bars.groupby(bars.index.date):
        # Skip Monday
        if SKIP_MONDAY and pd.Timestamp(date).dayofweek == 0:
            daily_pnl[date] = 0.0
            continue

        rth = day_bars[
            (day_bars.index.time >= session_start_t) &
            (day_bars.index.time < session_end_t)
        ]
        if len(rth) < 10:
            daily_pnl[date] = 0.0
            continue

        start_ts  = rth.index[0]
        or_end_ts = start_ts + pd.Timedelta(minutes=or_end_minutes)
        or_bars   = rth[rth.index < or_end_ts]
        if len(or_bars) < 4:
            daily_pnl[date] = 0.0
            continue

        or_high   = float(or_bars["high"].max())
        or_low    = float(or_bars["low"].min())
        or_range  = or_high - or_low
        if or_range <= TICK_SIZE * 4:
            daily_pnl[date] = 0.0
            continue

        vp        = _volume_profile(or_bars)
        avg_vol   = float(or_bars["volume"].mean())
        post      = rth[rth.index >= or_end_ts]

        pos       = None
        trades_today    = 0
        daily_loss_today = 0.0
        broke_up  = False
        broke_down = False

        for i, (ts, bar) in enumerate(rth.iterrows()):
            # Exit check
            if pos is not None:
                h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                is_last = (ts == rth.index[-1])
                sess_close = is_last or (ts.time() >= pd.Timestamp("15:55").time())
                exited, exit_p, reason = False, 0.0, ""

                if sess_close:
                    exited, exit_p, reason = True, _slip(c, pos.direction, False), "session_close"
                elif i >= pos.time_stop_bar:
                    exited, exit_p, reason = True, _slip(c, pos.direction, False), "time_stop"
                elif pos.direction == 1:
                    if l <= pos.stop_loss:
                        exited, exit_p, reason = True, _slip(pos.stop_loss, 1, False), "stop_loss"
                    elif h >= pos.profit_target:
                        exited, exit_p, reason = True, _slip(pos.profit_target, 1, False), "profit_target"
                else:
                    if h >= pos.stop_loss:
                        exited, exit_p, reason = True, _slip(pos.stop_loss, -1, False), "stop_loss"
                    elif l <= pos.profit_target:
                        exited, exit_p, reason = True, _slip(pos.profit_target, -1, False), "profit_target"

                if exited:
                    p = _pnl(pos.entry_price, exit_p, pos.direction, contracts)
                    equity += p
                    peak_equity = max(peak_equity, equity)
                    max_dd = min(max_dd, equity - peak_equity)
                    daily_loss_today += p
                    eq_curve.append((ts, equity))
                    trades.append({
                        "date": str(date),
                        "direction": "long" if pos.direction == 1 else "short",
                        "entry": round(pos.entry_price, 2),
                        "exit": round(exit_p, 2),
                        "pnl": round(p, 2),
                        "reason": reason,
                        "or_range": round(or_range, 2),
                    })
                    if p < 0:
                        consec_losses += 1
                        cooldown_remaining = COOLDOWN_BARS
                    else:
                        consec_losses = 0
                    pos = None

            # Entry check
            if ts < or_end_ts:
                continue
            if pos is not None:
                continue
            if trades_today >= MAX_TRADES_PER_DAY:
                continue
            if ts.time() > entry_cutoff_t:
                continue
            if daily_loss_today <= MAX_DAILY_LOSS:
                continue
            if consec_losses >= MAX_CONSEC_LOSSES:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue
                else:
                    consec_losses = 0  # reset after cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            close = float(bar["close"])
            high  = float(bar["high"])
            low   = float(bar["low"])

            if close > or_high:
                broke_up = True
            if close < or_low:
                broke_down = True

            direction = 0
            entry_mode = ""

            # Breakout entries
            if MODE in {"breakout", "both"}:
                if close > or_high and _aggression(bar, avg_vol, 1):
                    direction, entry_mode = 1, "breakout"
                elif close < or_low and _aggression(bar, avg_vol, -1):
                    direction, entry_mode = -1, "breakout"

            # Reload entries
            if direction == 0 and MODE in {"reload", "both"}:
                if broke_up and low <= vp["vah"] + reload_tol and close > vp["vah"] \
                        and _aggression(bar, avg_vol, 1, 0.9):
                    direction, entry_mode = 1, "reload"
                elif broke_down and high >= vp["val"] - reload_tol and close < vp["val"] \
                        and _aggression(bar, avg_vol, -1, 0.9):
                    direction, entry_mode = -1, "reload"

            if direction == 0:
                continue

            ep = _slip(close, direction, True)

            # OR-floor stop (live config — atr_stop_mult not used in deployed version)
            if direction == 1:
                sl = min(vp["val"], or_low) - TICK_SIZE
                pt = ep + TARGET_RANGE_MULT * or_range
            else:
                sl = max(vp["vah"], or_high) + TICK_SIZE
                pt = ep - TARGET_RANGE_MULT * or_range

            risk = abs(ep - sl)
            if risk <= TICK_SIZE:
                continue
            # Per-trade max loss check
            if risk * contracts * POINT_VALUE > PER_TRADE_MAX_LOSS:
                continue

            pos = Position(direction, ep, i, sl, pt, i + TIME_STOP_BARS, or_range)
            trades_today += 1

        daily_pnl[date] = daily_loss_today

    return trades, eq_curve, daily_pnl, max_dd


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(trades: list, eq_curve: list, daily_pnl: dict, max_dd: float, contracts: int):
    if not trades:
        print("No trades generated.")
        return {}

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = sum(pnls)
    gp     = sum(wins)
    gl     = abs(sum(losses))

    daily = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0

    from collections import Counter
    reasons = Counter(t["reason"] for t in trades)
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    longs  = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    reloads = [t for t in trades if t.get("mode") == "reload"]

    n_days = len(daily_pnl)
    active_days = sum(1 for v in daily_pnl.values() if v != 0)

    print(f"\n{'='*65}")
    print(f"  IVB ORB MES  |  2026 YTD  |  or={OR_MINUTES}min  tgt={TARGET_RANGE_MULT}x  n={contracts}c  mode={MODE}")
    print(f"{'='*65}")
    print(f"  Period       : {min(daily_pnl)} → {max(daily_pnl)}")
    print(f"  Trades       : {len(trades)}  ({len(trades)/n_days:.2f}/day over {n_days} days, {active_days} active)")
    print(f"  Win Rate     : {len(wins)/len(trades):.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL    : ${total:,.2f}")
    print(f"  Avg Win      : ${gp/len(wins):,.2f}" if wins else "  Avg Win      : N/A")
    print(f"  Avg Loss     : ${-gl/len(losses):,.2f}" if losses else "  Avg Loss     : N/A")
    print(f"  Profit Factor: {gp/gl:.2f}" if gl > 0 else "  Profit Factor: ∞")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max Drawdown : ${max_dd:,.2f}")
    print(f"  Long trades  : {len(longs)}  ({sum(1 for t in longs if t['pnl']>0)}/{len(longs)} wins)")
    print(f"  Short trades : {len(shorts)}  ({sum(1 for t in shorts if t['pnl']>0)}/{len(shorts)} wins)" if shorts else "  Short trades : 0")
    print(f"  Reload trades: {len(reloads)}")
    print(f"  Exit reasons : {reason_pct}")

    print(f"\n  {'Date':<12} {'PnL':>8}  {'Cum':>10}")
    print(f"  {'-'*34}")
    cum = 0.0
    for d, v in sorted(daily_pnl.items()):
        if v == 0:
            continue
        cum += v
        bar_char = "+" if v >= 0 else "-"
        print(f"  {str(d):<12} ${v:>7,.0f}  ${cum:>8,.0f}  {bar_char * min(20, int(abs(v)/30))}")

    print(f"\n  {'#':<4} {'Date':<12} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>8}  Reason")
    print(f"  {'-'*60}")
    for i, t in enumerate(trades, 1):
        print(f"  {i:<4} {t['date']:<12} {t['direction'].upper():<6} {t['entry']:>9.2f} {t['exit']:>9.2f} ${t['pnl']:>7,.2f}  {t['reason']}")

    result = {
        "strategy": "IVB ORB MES",
        "config": {
            "or_minutes": OR_MINUTES, "target_range_mult": TARGET_RANGE_MULT,
            "min_volume_ratio": MIN_VOLUME_RATIO, "mode": MODE,
            "skip_monday": SKIP_MONDAY, "contracts": contracts,
            "max_daily_loss": MAX_DAILY_LOSS, "max_consec_losses": MAX_CONSEC_LOSSES,
        },
        "period": f"{min(daily_pnl)} to {max(daily_pnl)}",
        "num_trades": len(trades), "win_rate": round(len(wins)/len(trades), 3),
        "total_pnl": round(total, 2), "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(gp/gl, 3) if gl > 0 else None,
        "exit_reasons": reason_pct,
        "long_trades": len(longs), "short_trades": len(shorts),
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
        "trades": trades,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Re-fetch MES data from Databento")
    parser.add_argument("--contracts", type=int, default=N_CONTRACTS)
    args = parser.parse_args()

    if args.fetch or not RTH_PATH.exists():
        bars = fetch_mes()
    else:
        bars = load_rth()
        logger.info(f"Loaded cached data: {len(bars):,} bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    logger.info(f"Running IVB backtest: {bars.index[0].date()} → {bars.index[-1].date()}, {args.contracts} contracts")
    trades, eq_curve, daily_pnl, max_dd = run_backtest(bars, contracts=args.contracts)
    result = print_results(trades, eq_curve, daily_pnl, abs(max_dd), args.contracts)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
