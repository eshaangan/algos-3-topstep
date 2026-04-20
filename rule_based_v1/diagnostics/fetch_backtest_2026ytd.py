"""Fetch 2026 YTD MNQ data and run full backtest with current live config.

Fetches MNQ.c.0 from 2026-01-01 to today, resamples to 5-min RTH bars,
then runs the deployed config: or_end=10:04, PT=3.0x, SL=1.5x, 3 contracts.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/fetch_backtest_2026ytd.py
    python rule_based_v1/diagnostics/fetch_backtest_2026ytd.py --backtest-only
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

from engine.risk_manager import RiskManager, TradeRecord
from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH    = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "2026ytd_results.json"

# ---------------------------------------------------------------------------
# Live config (mirrors deployed rules.yaml + risk.yaml)
# ---------------------------------------------------------------------------
OR_END_TIME        = "10:04"
MIN_OR_BARS        = 7
PT_MULT            = 3.0
SL_MULT            = 1.5
ENTRY_CUTOFF       = "12:00"
MIN_RANGE_ATR      = 0.3
ATR_PERIOD         = 14
TIME_STOP_BARS     = 24
TRAILING_ACT_ATR   = 999.0
TRAILING_DIST_ATR  = 0.75

POINT_VALUE        = 2.0
TICK_SIZE          = 0.25
TICK_VALUE         = 0.50
COMMISSION         = 0.62   # per side per contract
SLIPPAGE_TICKS     = 1

N_CONTRACTS        = 3
MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS     = -950.0
DRAWDOWN_BUFFER    = 1_950.0
PER_TRADE_MAX_LOSS = 1_000.0
COOLDOWN_BARS      = 3
MAX_CONSEC_LOSSES  = 10
STARTING_EQUITY    = 50_000.0


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch(start: str = "2026-01-01", end: str | None = None) -> pd.DataFrame:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")

    import databento as db
    if end is None:
        from datetime import timezone, datetime, timedelta
        # Use 2 days ago close (21:00 UTC) to stay within free historical window (no live sub needed)
        today = datetime.now(tz=timezone.utc).date()
        cutoff_day = today - timedelta(days=2)
        end = f"{cutoff_day}T21:00:00+00:00"

    logger.info(f"Fetching MNQ.c.0  {start} → {end} ...")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info(f"Fetched {len(df):,} 1-min bars  {df.index[0].date()} → {df.index[-1].date()}")

    # Resample to 5-min
    df5 = df.resample("5min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"])

    # RTH filter: 09:30–16:00 ET
    df5.index = df5.index.tz_convert("US/Eastern")
    rth = ((df5.index.hour > 9) | ((df5.index.hour == 9) & (df5.index.minute >= 30))) \
          & (df5.index.hour < 16)
    df5 = df5.loc[rth]
    logger.info(f"After RTH filter: {len(df5):,} 5-min bars  {df5.index[0]} → {df5.index[-1]}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df5.to_hdf(str(DATA_PATH), key="bars_5min", mode="w", complevel=5)
    logger.info(f"Saved → {DATA_PATH}")
    return df5


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    trailing_active: bool = False
    trailing_stop: float = 0.0
    peak_favorable: float = 0.0
    atr_at_entry: float = 0.0


def _slip(price, direction, is_entry):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry, exit_, direction):
    raw = (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE
    return raw - 2 * COMMISSION * N_CONTRACTS


def _check_exit(pos, bar, idx, sess_close):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        return True, _slip(c, pos.direction, False), "session_close"
    if idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop"

    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if pos.trailing_active and l <= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, 1, False), "trailing_stop"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
        if not pos.trailing_active and h - pos.entry_price >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active, pos.peak_favorable = True, h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and h > pos.peak_favorable:
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if pos.trailing_active and h >= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, -1, False), "trailing_stop"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
        if not pos.trailing_active and pos.entry_price - l >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active, pos.peak_favorable = True, l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and l < pos.peak_favorable:
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
    return False, 0.0, ""


# ---------------------------------------------------------------------------
# PrevVWAP helper
# ---------------------------------------------------------------------------
def _build_daily_meta(bars: pd.DataFrame) -> dict:
    """Pre-compute per-day prev-VWAP signal.

    Returns dict keyed by date: {"prev_vwap_bullish": bool | None}
    """
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp_vol = (typical * bars["volume"]).groupby(bars.index.date).cumsum()
    cum_vol    = bars["volume"].groupby(bars.index.date).cumsum()
    vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

    dates = sorted(set(bars.index.date))
    session_close_above_vwap: dict = {}
    session_last_close: dict = {}

    for d in dates:
        day_mask  = bars.index.date == d
        day_bars  = bars[day_mask]
        day_vwap  = vwap_series[day_mask]
        if len(day_bars) == 0:
            continue
        last_close = float(day_bars["close"].iloc[-1])
        last_vwap  = float(day_vwap.iloc[-1]) if not day_vwap.empty else last_close
        session_close_above_vwap[d] = (last_close > last_vwap)
        session_last_close[d] = last_close

    meta: dict = {}
    for i, d in enumerate(dates):
        prev_vwap_bullish = None
        if i > 0:
            prev_d = dates[i - 1]
            prev_vwap_bullish = session_close_above_vwap.get(prev_d)
        meta[d] = {"prev_vwap_bullish": prev_vwap_bullish}

    return meta


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_backtest(bars: pd.DataFrame, require_prev_vwap: bool = False) -> dict:
    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME, min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR, entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD, long_only=True,
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
    min_bars_needed = agg.required_bars()
    daily_meta = _build_daily_meta(bars) if require_prev_vwap else {}

    pos = None
    trades: list[TradeRecord] = []
    eq_vals, eq_times = [STARTING_EQUITY], [bars.index[0]]
    equity = STARTING_EQUITY
    cur_date = None
    daily_pnl: dict = {}
    trades_today = 0

    for i in range(min_bars_needed, len(bars)):
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
                tr = TradeRecord(entry_bar=pos.entry_bar_idx, exit_bar=i,
                                 direction=pos.direction, entry_price=pos.entry_price,
                                 exit_price=exit_p, pnl=p, exit_reason=reason)
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt)
                pos = None

        if pos is None and not sess_close and trades_today < MAX_TRADES_PER_DAY:
            ok, _ = rm.can_trade()
            if ok:
                if require_prev_vwap:
                    pv = daily_meta.get(bdate, {}).get("prev_vwap_bullish")
                    if pv is False:
                        continue
                lookback = bars.iloc[max(0, i - min_bars_needed + 1): i + 1]
                dec = agg.evaluate(lookback)
                if dec.should_trade:
                    cur_atr = atr_s.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        ep = _slip(bar["close"], dec.direction, True)
                        sl = rm.compute_stop_price(ep, dec.direction, cur_atr, SL_MULT)
                        pt = rm.compute_target_price(ep, dec.direction, cur_atr, PT_MULT)
                        pos = Position(direction=dec.direction, entry_price=ep,
                                       entry_bar_idx=i, stop_loss=sl, profit_target=pt,
                                       time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr)
                        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    return trades, eq_vals, eq_times, daily_pnl


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(trades, eq_vals, eq_times, daily_pnl):
    if not trades:
        print("No trades generated.")
        return {}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    total = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))

    eq = pd.Series(eq_vals, index=eq_times)
    max_dd = (eq - eq.cummax()).min()

    daily = pd.Series(daily_pnl)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    n_days = len(daily_pnl)

    print(f"\n{'='*60}")
    print(f"  2026 YTD  |  Config: or_end={OR_END_TIME}  PT={PT_MULT}x  SL={SL_MULT}x  n={N_CONTRACTS}c")
    print(f"{'='*60}")
    print(f"  Trades       : {len(trades)}  ({len(trades)/n_days:.2f}/day over {n_days} days)")
    print(f"  Win Rate     : {len(wins)/len(trades):.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL    : ${total:,.2f}")
    print(f"  Avg Win      : ${gp/len(wins):,.2f}" if wins else "  Avg Win      : N/A")
    print(f"  Avg Loss     : ${-gl/len(losses):,.2f}" if losses else "  Avg Loss     : N/A")
    print(f"  Profit Factor: {gp/gl:.2f}" if gl > 0 else "  Profit Factor: ∞")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max Drawdown : ${max_dd:,.2f}")
    print(f"  Exit reasons : {reason_pct}")

    print(f"\n  {'Date':<12} {'PnL':>8}  {'Cumulative':>12}")
    print(f"  {'-'*36}")
    cum = 0.0
    for d, v in sorted(daily_pnl.items()):
        cum += v
        bar_char = "+" if v >= 0 else "-"
        print(f"  {str(d):<12} ${v:>7,.0f}  ${cum:>10,.0f}  {bar_char * min(20, int(abs(v)/20))}")

    # Per-trade list
    print(f"\n  {'#':<4} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>8}  Reason")
    print(f"  {'-'*54}")
    for i, t in enumerate(trades, 1):
        d = "LONG" if t.direction == 1 else "SHORT"
        print(f"  {i:<4} {d:<6} {t.entry_price:>9.2f} {t.exit_price:>9.2f} ${t.pnl:>7,.2f}  {t.exit_reason}")

    result = {
        "config": {"or_end": OR_END_TIME, "pt": PT_MULT, "sl": SL_MULT, "contracts": N_CONTRACTS},
        "period": f"2026-01-01 to {pd.Timestamp.now().date()}",
        "num_trades": len(trades), "win_rate": round(len(wins)/len(trades), 3),
        "total_pnl": round(total, 2), "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3), "profit_factor": round(gp/gl, 3) if gl > 0 else None,
        "exit_reasons": reason_pct,
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-only", action="store_true", help="Skip fetch, use cached data")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--prev-vwap", action="store_true",
                        help="Enable PrevVWAP filter (only trade when yesterday close > yesterday VWAP)")
    args = parser.parse_args()

    if args.backtest_only:
        if not DATA_PATH.exists():
            logger.error(f"No cached data at {DATA_PATH}. Run without --backtest-only first.")
            sys.exit(1)
        bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars = fetch(start=args.start)

    logger.info(f"Running backtest on {len(bars):,} bars: {bars.index[0].date()} → {bars.index[-1].date()}")
    if args.prev_vwap:
        logger.info("PrevVWAP filter: ENABLED")
    trades, eq_vals, eq_times, daily_pnl = run_backtest(bars, require_prev_vwap=args.prev_vwap)
    result = print_results(trades, eq_vals, eq_times, daily_pnl)

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
