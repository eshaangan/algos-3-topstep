"""Instrument Frequency Research (IFR) Backtest
===============================================
Runs the same ORB strategy (LONG-only, or_end=10:04, PT=3.0x, SL=1.5x)
across three instruments: MNQ, MES, M2K (Micro Russell 2000).

Parts:
  1. Fetch & backtest MES 2026 YTD (3c, 4c, 5c)
  2. Backtest M2K 2026 YTD from cached or freshly-fetched data (3c, 4c, 5c)
  3. Portfolio correlation analysis: daily PnL MNQ vs MES vs M2K
  4. Independent signal day analysis (diversification check)

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/ifr_backtest.py               # fetch + backtest all
    python rule_based_v1/diagnostics/ifr_backtest.py --backtest-only  # use cached data only

Artifacts:
    rule_based_v1/diagnostics/ifr_results.json
    data/processed/mes_2026ytd_5min.h5      (key='bars_5min')
    data/processed/m2k_2026ytd_5min.h5      (key='bars_5min')

NOTE: The earlier Inventory Failure Reversion (IFR) script has been
superseded by this Instrument Frequency Research script. The prior
IFR results were saved separately in ifr_longterm_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import time as dt_time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
MNQ_PATH     = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
MES_PATH     = ROOT / "data" / "processed" / "mes_2026ytd_5min.h5"
M2K_YTD_PATH = ROOT / "data" / "processed" / "m2k_2026ytd_5min.h5"
M2K_LT_PATH  = ROOT / "data" / "processed" / "m2k_bars_5min.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "ifr_results.json"

# ---------------------------------------------------------------------------
# Shared ORB config (mirrors deployed live MNQ config)
# ---------------------------------------------------------------------------
OR_END_TIME       = "10:04"
MIN_OR_BARS       = 7
PT_MULT           = 3.0
SL_MULT           = 1.5
ENTRY_CUTOFF      = "12:00"
MIN_RANGE_ATR     = 0.3
ATR_PERIOD        = 14
TIME_STOP_BARS    = 24
TRAILING_ACT_ATR  = 999.0    # effectively disabled
TRAILING_DIST_ATR = 0.75

MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS     = -950.0
DRAWDOWN_BUFFER    = 1_950.0
PER_TRADE_MAX_LOSS = 1_000.0
COOLDOWN_BARS      = 3
MAX_CONSEC_LOSSES  = 10
STARTING_EQUITY    = 50_000.0

# ---------------------------------------------------------------------------
# Instrument specs
# ---------------------------------------------------------------------------
INSTRUMENT_SPECS = {
    "MNQ": {
        "symbol":           "MNQ.c.0",
        "point_value":      2.0,
        "tick_size":        0.25,
        "tick_value":       0.50,
        "commission":       0.62,
        "slippage_ticks":   1,
        "test_contracts":   [3],        # reference config (already known optimal)
    },
    "MES": {
        "symbol":           "MES.c.0",
        "point_value":      5.0,        # $5 per point
        "tick_size":        0.25,
        "tick_value":       1.25,
        "commission":       0.62,
        "slippage_ticks":   1,
        "test_contracts":   [3, 4, 5],
    },
    "M2K": {
        "symbol":           "M2K.c.0",
        "point_value":      5.0,        # $5 per point
        "tick_size":        0.10,
        "tick_value":       0.50,
        "commission":       0.62,
        "slippage_ticks":   2,          # slightly wider spread for RTY
        "test_contracts":   [3, 4, 5],
    },
}


# ---------------------------------------------------------------------------
# Data fetch helpers
# ---------------------------------------------------------------------------
def _fetch_databento(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch 1-min OHLCV from Databento and resample to 5-min RTH bars."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set in .env")

    import databento as db
    logger.info(f"Fetching {symbol}  {start} → {end} ...")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info(f"  Raw: {len(df):,} 1-min bars  {df.index[0].date()} → {df.index[-1].date()}")

    # Resample to 5-min
    df5 = df.resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["open"])

    # RTH filter: 09:30–16:00 ET
    df5.index = df5.index.tz_convert("US/Eastern")
    rth = (
        (df5.index.hour > 9) | ((df5.index.hour == 9) & (df5.index.minute >= 30))
    ) & (df5.index.hour < 16)
    df5 = df5.loc[rth]
    logger.info(f"  RTH 5-min: {len(df5):,} bars  {df5.index[0]} → {df5.index[-1]}")
    return df5


def _fetch_databento_eth(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch 1m OHLCV, resample to 5m — no US RTH filter (Globex ETH for Asia-session ORB)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set in .env")

    import databento as db
    logger.info(f"Fetching ETH (unfiltered) {symbol}  {start} → {end} ...")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df5 = df.resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["open"])
    logger.info(f"  ETH 5-min UTC: {len(df5):,} bars  {df5.index[0]} → {df5.index[-1]}")
    return df5


def fetch_and_save(symbol: str, out_path: Path, start: str, end: str) -> pd.DataFrame:
    df5 = _fetch_databento(symbol, start, end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df5.to_hdf(str(out_path), key="bars_5min", mode="w", complevel=5)
    logger.info(f"  Saved → {out_path}")
    return df5


def fetch_and_save_eth(symbol: str, out_path: Path, start: str, end: str) -> pd.DataFrame:
    df5 = _fetch_databento_eth(symbol, start, end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df5.to_hdf(str(out_path), key="bars_5min_eth", mode="w", complevel=5)
    logger.info(f"  Saved → {out_path}")
    return df5


# ---------------------------------------------------------------------------
# Data load helpers
# ---------------------------------------------------------------------------
def _load_bars(path: Path, hdf_key: str = "bars_5min") -> pd.DataFrame:
    logger.info(f"Loading {path.name}  key={hdf_key}")
    try:
        df = pd.read_hdf(str(path), key=hdf_key)
    except KeyError:
        with pd.HDFStore(str(path), "r") as store:
            available = store.keys()
        logger.warning(f"  Key '{hdf_key}' not found; available={available}. Using first.")
        df = pd.read_hdf(str(path), key=available[0])
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def _filter_ytd(df: pd.DataFrame, start: str = "2026-01-01") -> pd.DataFrame:
    cutoff = pd.Timestamp(start, tz="US/Eastern")
    return df[df.index >= cutoff].copy()


# ---------------------------------------------------------------------------
# Position / simulation helpers
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


def _slip(price: float, direction: int, is_entry: bool, tick_size: float, slippage_ticks: int) -> float:
    s = slippage_ticks * tick_size
    return price + s * direction if is_entry else price - s * direction


def _trade_pnl(
    entry: float, exit_: float, direction: int,
    n_contracts: int, point_value: float, commission: float,
) -> float:
    raw = (exit_ - entry) * direction * n_contracts * point_value
    return raw - 2.0 * commission * n_contracts


def _check_exit(
    pos: Position, bar: pd.Series, idx: int, sess_close: bool,
    tick_size: float, slippage_ticks: int,
) -> tuple[bool, float, str]:
    h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])

    if sess_close:
        return True, _slip(c, pos.direction, False, tick_size, slippage_ticks), "session_close"
    if idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False, tick_size, slippage_ticks), "time_stop"

    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False, tick_size, slippage_ticks), "stop_loss"
        if pos.trailing_active and l <= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, 1, False, tick_size, slippage_ticks), "trailing_stop"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False, tick_size, slippage_ticks), "profit_target"
        if not pos.trailing_active and (h - pos.entry_price) >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and h > pos.peak_favorable:
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False, tick_size, slippage_ticks), "stop_loss"
        if pos.trailing_active and h >= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, -1, False, tick_size, slippage_ticks), "trailing_stop"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False, tick_size, slippage_ticks), "profit_target"
        if not pos.trailing_active and (pos.entry_price - l) >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and l < pos.peak_favorable:
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Core backtest engine (instrument-agnostic)
# ---------------------------------------------------------------------------
def run_backtest_instrument(
    bars: pd.DataFrame,
    spec: dict,
    n_contracts: int,
    *,
    or_end_time: str = OR_END_TIME,
    min_or_bars: int = MIN_OR_BARS,
    min_range_atr: float = MIN_RANGE_ATR,
    entry_cutoff: str = ENTRY_CUTOFF,
    pt_mult: float = PT_MULT,
    sl_mult: float = SL_MULT,
    time_stop_bars: int = TIME_STOP_BARS,
    atr_period: int = ATR_PERIOD,
    long_only: bool = True,
    session_start_time: str = "09:30",
    session_timezone: str = "US/Eastern",
    calendar_mode: str = "us_et",
    tokyo_flat_time: str = "15:55",
) -> dict:
    """Run ORB LONG-only backtest for one instrument / contract-size combo."""

    tick_size      = spec["tick_size"]
    tick_value     = spec["tick_value"]
    point_value    = spec["point_value"]
    commission     = spec["commission"]
    slippage_ticks = spec["slippage_ticks"]

    orb = OpeningRangeBreakoutRule(
        session_timezone=session_timezone,
        session_start_time=session_start_time,
        or_end_time=or_end_time,
        min_or_bars=min_or_bars,
        min_range_atr=min_range_atr,
        entry_cutoff_time=entry_cutoff,
        atr_period=atr_period,
        long_only=long_only,
    )
    agg = SignalAggregator(
        primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0
    )
    rm = RiskManager(
        contracts=n_contracts, point_value=point_value,
        tick_size=tick_size, tick_value=tick_value,
        max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=PER_TRADE_MAX_LOSS,
        max_consecutive_losses=MAX_CONSEC_LOSSES,
        cooldown_bars=COOLDOWN_BARS,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    atr_s = atr(bars["high"], bars["low"], bars["close"], atr_period)
    min_bars_needed = agg.required_bars()

    pos = None
    trades: list[TradeRecord] = []
    eq_vals, eq_times = [STARTING_EQUITY], [bars.index[0]]
    equity = STARTING_EQUITY
    cur_date = None
    daily_pnl: dict = {}
    trades_today = 0

    flat_h, flat_m = map(int, tokyo_flat_time.split(":"))
    tokyo_flat_cut = dt_time(flat_h, flat_m)

    for i in range(min_bars_needed, len(bars)):
        bar  = bars.iloc[i]
        bt   = bars.index[i]
        bt_aware = bt if bt.tzinfo is not None else pd.Timestamp(bt).tz_localize("UTC")

        if calendar_mode == "tokyo":
            bt_jp = bt_aware.tz_convert("Asia/Tokyo")
            bdate = bt_jp.date()
            if i + 1 < len(bars):
                nxt = bars.index[i + 1]
                nxt_aware = nxt if nxt.tzinfo is not None else pd.Timestamp(nxt).tz_localize("UTC")
                njp = nxt_aware.tz_convert("Asia/Tokyo")
                is_last = njp.date() != bt_jp.date()
            else:
                is_last = True
            sess_close = is_last or (bt_jp.time() >= tokyo_flat_cut)
        else:
            bt_et = bt_aware.tz_convert("US/Eastern")
            bdate = bt_et.date()
            is_last = (i + 1 >= len(bars)) or (
                (bars.index[i + 1].tz_convert("US/Eastern") if bars.index[i + 1].tzinfo
                 else bars.index[i + 1]).date() != bdate
            )
            sess_close = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        # --- Exit check ---
        if pos is not None:
            exited, exit_p, reason = _check_exit(
                pos, bar, i, sess_close, tick_size, slippage_ticks
            )
            if exited:
                p = _trade_pnl(pos.entry_price, exit_p, pos.direction,
                               n_contracts, point_value, commission)
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

        # --- Entry check ---
        if pos is None and not sess_close and trades_today < MAX_TRADES_PER_DAY:
            ok, _ = rm.can_trade()
            if ok:
                lookback = bars.iloc[max(0, i - min_bars_needed + 1): i + 1]
                dec = agg.evaluate(lookback)
                if dec.should_trade:
                    cur_atr = float(atr_s.iloc[i])
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        ep = _slip(float(bar["close"]), dec.direction, True,
                                   tick_size, slippage_ticks)
                        sl = rm.compute_stop_price(ep, dec.direction, cur_atr, sl_mult)
                        pt = rm.compute_target_price(ep, dec.direction, cur_atr, pt_mult)
                        pos = Position(
                            direction=dec.direction, entry_price=ep, entry_bar_idx=i,
                            stop_loss=sl, profit_target=pt,
                            time_stop_bar=i + time_stop_bars,
                            atr_at_entry=cur_atr,
                        )
                        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    # ---- stats ----
    if not trades:
        return {
            "n_contracts": n_contracts, "num_trades": 0, "win_rate": None,
            "total_pnl": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
            "profit_factor": None, "exit_reasons": {},
            "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
            "monthly_breakdown": {},
        }

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total  = sum(t.pnl for t in trades)
    gp     = sum(t.pnl for t in wins)
    gl     = abs(sum(t.pnl for t in losses))

    eq_series = pd.Series(eq_vals, index=eq_times)
    max_dd = float((eq_series - eq_series.cummax()).min())

    # Sharpe on non-zero daily PnL
    daily_series = pd.Series({k: v for k, v in daily_pnl.items() if v != 0.0})
    sharpe = (
        float(daily_series.mean() / daily_series.std() * np.sqrt(252))
        if len(daily_series) > 1 and daily_series.std() > 0 else 0.0
    )

    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    n_days  = len(daily_pnl)
    n_trade_days = len([v for v in daily_pnl.values() if v != 0.0])

    # Per-trade list for JSON
    trade_list = [
        {
            "entry_bar": t.entry_bar, "exit_bar": t.exit_bar,
            "direction": t.direction, "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4), "pnl": round(t.pnl, 2),
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]

    # Monthly breakdown
    monthly_series = pd.Series(
        {pd.Timestamp(k): v for k, v in daily_pnl.items()}
    ).resample("ME").sum()
    monthly = {str(k.date()): round(v, 2) for k, v in monthly_series.items()}

    return {
        "n_contracts":    n_contracts,
        "num_trades":     len(trades),
        "trade_days":     n_trade_days,
        "total_days":     n_days,
        "trades_per_day": round(len(trades) / max(n_days, 1), 3),
        "win_rate":       round(len(wins) / len(trades), 3),
        "total_pnl":      round(total, 2),
        "avg_win":        round(gp / len(wins), 2)  if wins   else None,
        "avg_loss":       round(-gl / len(losses), 2) if losses else None,
        "max_drawdown":   round(max_dd, 2),
        "sharpe":         round(sharpe, 3),
        "profit_factor":  round(gp / gl, 3) if gl > 0 else None,
        "exit_reasons":   reason_pct,
        "monthly_breakdown": monthly,
        "daily_pnl":      {str(k): round(v, 2) for k, v in daily_pnl.items()},
        "trades":         trade_list,
    }


# ---------------------------------------------------------------------------
# Topstep combine pass/fail check
# ---------------------------------------------------------------------------
def _topstep_check(r: dict) -> dict:
    total_pnl = r.get("total_pnl", 0.0)
    max_dd    = r.get("max_drawdown", 0.0)
    daily_pnl = r.get("daily_pnl", {})

    worst_day  = min(daily_pnl.values(), default=0.0)
    daily_pass = worst_day >= -950.0
    dd_pass    = max_dd >= -2_000.0
    n_days     = r.get("total_days", 57)
    ann_pnl    = total_pnl / max(n_days, 1) * 252

    return {
        "worst_day":           round(worst_day, 2),
        "daily_limit_pass":    daily_pass,
        "max_drawdown":        max_dd,
        "drawdown_limit_pass": dd_pass,
        "annualised_pnl_est":  round(ann_pnl, 2),
        "combine_pass_likely": daily_pass and dd_pass and total_pnl >= 0,
    }


# ---------------------------------------------------------------------------
# Portfolio / correlation analysis
# ---------------------------------------------------------------------------
def _portfolio_analysis(daily_pnl_per_instr: dict[str, dict]) -> dict:
    series = {}
    for instr, dpnl in daily_pnl_per_instr.items():
        series[instr] = pd.Series({pd.Timestamp(k): float(v) for k, v in dpnl.items()})

    df = pd.DataFrame(series).fillna(0.0)

    corr_matrix: dict = {}
    instruments = list(df.columns)
    for a in instruments:
        corr_matrix[a] = {}
        for b in instruments:
            corr_matrix[a][b] = round(float(df[[a, b]].corr().iloc[0, 1]), 3)

    # Pairwise independent signal days
    indep: dict = {}
    for i, a in enumerate(instruments):
        for b in instruments[i + 1:]:
            a_days = set(df[df[a] != 0].index)
            b_days = set(df[df[b] != 0].index)
            both   = a_days & b_days
            only_a = a_days - b_days
            only_b = b_days - a_days
            union  = a_days | b_days
            pair = f"{a}_vs_{b}"
            indep[pair] = {
                "both_trade":        len(both),
                f"only_{a}":         len(only_a),
                f"only_{b}":         len(only_b),
                "overlap_pct":       round(len(both) / max(len(union), 1) * 100, 1),
                "diversification_pct": round(
                    (len(only_a) + len(only_b)) / max(len(union), 1) * 100, 1
                ),
            }

    # Combined portfolio daily PnL (simple sum — equal $ weight)
    combined = df.sum(axis=1)
    combined_total = float(combined.sum())
    combined_sharpe = (
        float(combined.mean() / combined.std() * np.sqrt(252))
        if combined.std() > 0 else 0.0
    )
    combined_dd = float((combined.cumsum() - combined.cumsum().cummax()).min())

    return {
        "instruments_included":    instruments,
        "correlation_matrix":      corr_matrix,
        "pairwise_signal_days":    indep,
        "combined_portfolio": {
            "total_pnl":    round(combined_total, 2),
            "sharpe":       round(combined_sharpe, 3),
            "max_drawdown": round(combined_dd, 2),
        },
    }


# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------
def _print_result(instr: str, r: dict):
    nc = r["n_contracts"]
    print(f"\n{'='*65}")
    print(f"  {instr} {nc}c  |  ORB LONG-only  or_end={OR_END_TIME}  PT={PT_MULT}x  SL={SL_MULT}x")
    print(f"{'='*65}")
    if r["num_trades"] == 0:
        print("  NO TRADES generated.")
        return
    wr = r["win_rate"]
    print(f"  Trades        : {r['num_trades']}  ({r['trades_per_day']:.2f}/day over {r['total_days']} calendar days)")
    print(f"  Trade days    : {r['trade_days']}")
    print(f"  Win Rate      : {wr:.1%}")
    print(f"  Total PnL     : ${r['total_pnl']:,.2f}")
    if r["avg_win"]:
        print(f"  Avg Win       : ${r['avg_win']:,.2f}")
    if r["avg_loss"]:
        print(f"  Avg Loss      : ${r['avg_loss']:,.2f}")
    print(f"  Profit Factor : {r['profit_factor']}")
    print(f"  Sharpe        : {r['sharpe']:.2f}")
    print(f"  Max Drawdown  : ${r['max_drawdown']:,.2f}")
    print(f"  Exit reasons  : {r['exit_reasons']}")

    tc = _topstep_check(r)
    flag = "PASS" if tc["combine_pass_likely"] else "FAIL"
    print(f"\n  Topstep check : worst_day=${tc['worst_day']:,.0f}  "
          f"daily={'OK' if tc['daily_limit_pass'] else 'BREACH'}  "
          f"dd={'OK' if tc['drawdown_limit_pass'] else 'BREACH'}  "
          f"combined={flag}")

    if r.get("monthly_breakdown"):
        print(f"\n  Monthly breakdown:")
        for mo, pnl in sorted(r["monthly_breakdown"].items()):
            bar = "+" * min(30, int(abs(pnl) / 50)) if pnl >= 0 else "-" * min(30, int(abs(pnl) / 50))
            print(f"    {mo}  ${pnl:>8,.0f}  {bar}")


def _print_portfolio(analysis: dict):
    print(f"\n{'='*65}")
    print("  PORTFOLIO ANALYSIS")
    print(f"{'='*65}")

    instrs = analysis.get("instruments_included", [])
    cm = analysis.get("correlation_matrix", {})
    print(f"\n  Pairwise daily-PnL correlations:")
    for i, a in enumerate(instrs):
        for b in instrs[i + 1:]:
            val = cm.get(a, {}).get(b, "N/A")
            interp = ""
            if isinstance(val, float):
                if abs(val) < 0.2:
                    interp = "  LOW (good diversification)"
                elif abs(val) < 0.5:
                    interp = "  MODERATE"
                else:
                    interp = "  HIGH (similar signal days)"
            print(f"    {a} vs {b}: {val:.3f}{interp}")

    print(f"\n  Pairwise signal day overlap:")
    for pair, d in analysis.get("pairwise_signal_days", {}).items():
        print(f"    {pair}:")
        print(f"      both trade: {d['both_trade']} days   overlap: {d['overlap_pct']}%   "
              f"diversification: {d.get('diversification_pct', 'N/A')}%")

    comb = analysis.get("combined_portfolio", {})
    print(f"\n  Combined portfolio (sum of all):")
    print(f"    PnL=${comb.get('total_pnl', 0):,.2f}  "
          f"Sharpe={comb.get('sharpe', 0):.2f}  "
          f"MaxDD=${comb.get('max_drawdown', 0):,.2f}")


def _print_summary(results: dict):
    print(f"\n{'='*75}")
    print(f"  SUMMARY TABLE  — ORB LONG-only  or_end={OR_END_TIME}  PT={PT_MULT}x  SL={SL_MULT}x")
    print(f"{'='*75}")
    print(f"  {'Key':<14}  {'N':>5}  {'WR':>6}  {'PnL':>10}  {'Sharpe':>7}  "
          f"{'MaxDD':>10}  {'Ann.PnL':>10}  Pass?")
    print(f"  {'-'*73}")
    for k, r in results.items():
        if k in ("portfolio_analysis", "config", "run_metadata"):
            continue
        tc = _topstep_check(r)
        wr = f"{r['win_rate']:.1%}" if r.get("win_rate") else "N/A "
        pf = "PASS" if tc["combine_pass_likely"] else "FAIL"
        print(f"  {k:<14}  {r['num_trades']:>5}  {wr:>6}  "
              f"${r['total_pnl']:>9,.0f}  {r['sharpe']:>7.2f}  "
              f"${r['max_drawdown']:>9,.0f}  "
              f"${tc['annualised_pnl_est']:>9,.0f}  {pf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IFR: Multi-instrument ORB backtest")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Skip all fetches; use cached data only")
    parser.add_argument("--no-mes-fetch", action="store_true",
                        help="Skip MES fetch (use cached if available)")
    parser.add_argument("--no-m2k-fetch", action="store_true",
                        help="Skip M2K fetch (use cached if available)")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end",   default="2026-03-22T00:00:00+00:00")
    args = parser.parse_args()

    results: dict = {}
    daily_pnl_per_instr: dict = {}

    # -----------------------------------------------------------------------
    # MNQ — reference benchmark (cached data only)
    # -----------------------------------------------------------------------
    logger.info("─── MNQ reference ───")
    if not MNQ_PATH.exists():
        logger.error(f"MNQ data missing at {MNQ_PATH}. Run fetch_backtest_2026ytd.py first.")
        sys.exit(1)

    mnq_bars = _load_bars(MNQ_PATH, "bars_5min")
    mnq_bars = _filter_ytd(mnq_bars, args.start)
    logger.info(f"MNQ: {len(mnq_bars):,} bars  {mnq_bars.index[0].date()} → {mnq_bars.index[-1].date()}")

    mnq_spec = INSTRUMENT_SPECS["MNQ"]
    mnq_r    = run_backtest_instrument(mnq_bars, mnq_spec, n_contracts=3)
    mnq_r["instrument"] = "MNQ"
    _print_result("MNQ", mnq_r)
    results["MNQ_3c"] = mnq_r
    daily_pnl_per_instr["MNQ"] = mnq_r["daily_pnl"]

    # -----------------------------------------------------------------------
    # MES — fetch or load
    # -----------------------------------------------------------------------
    logger.info("─── MES ───")
    mes_bars = None
    need_fetch_mes = not args.backtest_only and not args.no_mes_fetch

    if need_fetch_mes:
        try:
            mes_bars = fetch_and_save("MES.c.0", MES_PATH, args.start, args.end)
        except Exception as e:
            logger.error(f"MES fetch failed: {e}")

    if mes_bars is None and MES_PATH.exists():
        logger.info("Loading cached MES data...")
        mes_bars = _load_bars(MES_PATH, "bars_5min")
        mes_bars = _filter_ytd(mes_bars, args.start)

    if mes_bars is None:
        logger.warning("No MES data — skipping MES backtest.")
    else:
        logger.info(f"MES: {len(mes_bars):,} bars  {mes_bars.index[0].date()} → {mes_bars.index[-1].date()}")
        mes_spec = INSTRUMENT_SPECS["MES"]
        best_mes: dict | None = None

        for nc in mes_spec["test_contracts"]:
            logger.info(f"Running MES {nc}c ...")
            r = run_backtest_instrument(mes_bars, mes_spec, n_contracts=nc)
            r["instrument"] = "MES"
            _print_result("MES", r)
            key = f"MES_{nc}c"
            results[key] = r

            if best_mes is None or r.get("sharpe", 0) > best_mes.get("sharpe", 0):
                best_mes = r

        if best_mes is not None:
            daily_pnl_per_instr["MES"] = best_mes["daily_pnl"]

    # -----------------------------------------------------------------------
    # M2K — fetch or load
    # -----------------------------------------------------------------------
    logger.info("─── M2K (Micro Russell 2000) ───")
    m2k_bars = None
    need_fetch_m2k = not args.backtest_only and not args.no_m2k_fetch

    if need_fetch_m2k:
        try:
            m2k_bars = fetch_and_save("M2K.c.0", M2K_YTD_PATH, args.start, args.end)
        except Exception as e:
            logger.error(f"M2K fetch failed: {e}")

    if m2k_bars is None:
        for cand_path in [M2K_YTD_PATH, M2K_LT_PATH]:
            if cand_path.exists():
                logger.info(f"Loading cached M2K from {cand_path.name} ...")
                m2k_bars = _load_bars(cand_path, "bars_5min")
                m2k_bars = _filter_ytd(m2k_bars, args.start)
                if len(m2k_bars) > 0:
                    break
                m2k_bars = None

    if m2k_bars is None:
        logger.warning("No M2K data — skipping M2K backtest.")
    else:
        logger.info(f"M2K: {len(m2k_bars):,} bars  {m2k_bars.index[0].date()} → {m2k_bars.index[-1].date()}")
        m2k_spec = INSTRUMENT_SPECS["M2K"]
        best_m2k: dict | None = None

        for nc in m2k_spec["test_contracts"]:
            logger.info(f"Running M2K {nc}c ...")
            r = run_backtest_instrument(m2k_bars, m2k_spec, n_contracts=nc)
            r["instrument"] = "M2K"
            _print_result("M2K", r)
            key = f"M2K_{nc}c"
            results[key] = r

            if best_m2k is None or r.get("sharpe", 0) > best_m2k.get("sharpe", 0):
                best_m2k = r

        if best_m2k is not None:
            daily_pnl_per_instr["M2K"] = best_m2k["daily_pnl"]

    # -----------------------------------------------------------------------
    # Portfolio correlation analysis
    # -----------------------------------------------------------------------
    if len(daily_pnl_per_instr) >= 2:
        logger.info("─── Portfolio analysis ───")
        portfolio = _portfolio_analysis(daily_pnl_per_instr)
        _print_portfolio(portfolio)
        results["portfolio_analysis"] = portfolio
    else:
        results["portfolio_analysis"] = {}

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_summary(results)

    results["config"] = {
        "or_end_time":      OR_END_TIME,
        "min_or_bars":      MIN_OR_BARS,
        "pt_mult":          PT_MULT,
        "sl_mult":          SL_MULT,
        "entry_cutoff":     ENTRY_CUTOFF,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "max_daily_loss":   MAX_DAILY_LOSS,
        "drawdown_buffer":  DRAWDOWN_BUFFER,
        "time_stop_bars":   TIME_STOP_BARS,
        "trailing_act_atr": TRAILING_ACT_ATR,
        "direction":        "LONG_ONLY",
    }
    results["run_metadata"] = {
        "period_start": args.start,
        "period_end":   args.end,
        "run_date":     str(pd.Timestamp.now().date()),
    }

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved → {RESULTS_PATH}")
    print(f"\n  Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
