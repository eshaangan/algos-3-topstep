"""Grid search for $15k / 5-contract / $2k-DD strategy on 2026 YTD MNQ data.

Tests ORB-only variants and ORB+VWAP portfolio combinations.
All results scaled to 5 contracts.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/goal_search_15k.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
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

logging.basicConfig(level=logging.WARNING)

CACHE_PATH    = ROOT / "data" / "processed" / "mnq_vc_backtest_5min.parquet"
RESULTS_PATH  = RBV1 / "diagnostics" / "goal_search_results.json"

# ── Fixed params ──────────────────────────────────────────────────────────────
OR_END_TIME       = "10:04"
MIN_OR_BARS       = 7
MIN_RANGE_ATR     = 0.3
ENTRY_CUTOFF      = "12:00"
ATR_PERIOD        = 14
TRAILING_ACT_ATR  = 999.0
TRAILING_DIST_ATR = 0.75
TIME_STOP_BARS    = 24

POINT_VALUE       = 2.0
TICK_SIZE         = 0.25
COMMISSION        = 0.62
SLIPPAGE_TICKS    = 1

TARGET_CONTRACTS  = 5
TARGET_PNL        = 15_000.0
TARGET_MAX_DD     = -2_000.0
MAX_DAILY_LOSS    = -950.0
DRAWDOWN_BUFFER   = 1_950.0
COOLDOWN_BARS     = 3

# ── VC metadata ───────────────────────────────────────────────────────────────
def build_vc_meta(bars: pd.DataFrame, lookback: int = 10) -> dict:
    dates = sorted(set(bars.index.date))
    daily = {}
    for d in dates:
        day = bars[bars.index.date == d]
        if not len(day):
            continue
        daily[d] = {
            "range": float(day["high"].max() - day["low"].min()),
            "direction": 1 if day["close"].iloc[-1] > day["open"].iloc[0]
                         else (-1 if day["close"].iloc[-1] < day["open"].iloc[0] else 0),
        }
    meta = {}
    for i, d in enumerate(dates):
        if i == 0:
            meta[d] = None
            continue
        prev = daily.get(dates[i - 1])
        if prev is None:
            meta[d] = None
            continue
        window = [daily[dates[j]]["range"] for j in range(max(0, i - lookback), i) if dates[j] in daily]
        if len(window) < 3:
            meta[d] = None
            continue
        arr = np.array(window)
        meta[d] = {
            "prev_range":     prev["range"],
            "prev_direction": prev["direction"],
            "wide_50":  prev["range"] > float(np.median(arr)),
            "wide_25":  prev["range"] > float(np.percentile(arr, 25)),
            "wide_75":  prev["range"] > float(np.percentile(arr, 75)),
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


def _slip(price, direction, is_entry):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry, exit_, direction, n_contracts):
    return (exit_ - entry) * direction * n_contracts * POINT_VALUE - 2 * COMMISSION * n_contracts


def _check_exit(pos: Position, bar, idx: int, sess_close: bool, pt_mult, sl_mult):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close or idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
    return False, 0.0, ""


# ── ORB backtest ──────────────────────────────────────────────────────────────
def run_orb(
    bars: pd.DataFrame,
    vc_meta: dict,
    n_contracts: int,
    pt_mult: float,
    sl_mult: float,
    max_trades_day: int,
    vc_mode: str,  # "none","dir","wide50_dir","wide25_dir","wide75_dir","wide50","dir_long"
    long_only: bool = False,
) -> dict:
    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME, min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR, entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD, long_only=long_only,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    rm = RiskManager(
        contracts=n_contracts, point_value=POINT_VALUE, tick_size=TICK_SIZE,
        tick_value=POINT_VALUE / 4, max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=abs(MAX_DAILY_LOSS) * 1.5,
        max_consecutive_losses=10, cooldown_bars=COOLDOWN_BARS,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(50_000.0)

    atr_s = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos = None
    trades: list = []
    equity = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bt_et = bars.index[i]
        if bt_et.tzinfo:
            bt_et = bt_et.tz_convert("US/Eastern")
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (
            bars.index[i+1].tz_convert("US/Eastern").date() != bdate
        )
        sess_close = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close, pt_mult, sl_mult)
            if exited:
                p = _pnl(pos.entry_price, exit_p, pos.direction, n_contracts)
                trades.append({"pnl": p, "reason": reason, "dir": pos.direction})
                rm.record_trade(TradeRecord(0, 0, pos.direction, pos.entry_price, exit_p, p, reason))
                equity += p
                eq_vals.append(equity)
                pos = None

        if pos is None and not sess_close and trades_today < max_trades_day:
            can, _ = rm.can_trade()
            if not can:
                continue

            # VC gate
            meta = vc_meta.get(bdate)
            if vc_mode != "none":
                if meta is None:
                    continue
                pd_ = meta["prev_direction"]
                w50, w25, w75 = meta["wide_50"], meta["wide_25"], meta["wide_75"]
                if vc_mode == "dir" and pd_ == 0:
                    continue
                elif vc_mode == "wide50_dir" and (not w50 or pd_ == 0):
                    continue
                elif vc_mode == "wide25_dir" and (not w25 or pd_ == 0):
                    continue
                elif vc_mode == "wide75_dir" and (not w75 or pd_ == 0):
                    continue
                elif vc_mode == "wide50" and not w50:
                    continue

            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg.evaluate(lookback)
            if not dec.should_trade:
                continue

            # Direction gate
            if vc_mode in ("dir", "wide50_dir", "wide25_dir", "wide75_dir") and meta:
                if meta["prev_direction"] != 0 and dec.direction != meta["prev_direction"]:
                    continue

            cur_atr = float(atr_s.iloc[i])
            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            ep = _slip(bar["close"], dec.direction, True)
            sl = ep - sl_mult * cur_atr * dec.direction
            pt = ep + pt_mult * cur_atr * dec.direction
            pos = Position(
                direction=dec.direction, entry_price=ep, entry_bar_idx=i,
                stop_loss=sl, profit_target=pt,
                time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr,
            )
            trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "win_rate": 0, "sharpe": 0, "max_drawdown": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    eq = pd.Series(eq_vals)
    max_dd = float((eq - eq.cummax()).min())
    daily = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(bars_2026: pd.DataFrame, vc_meta: dict) -> list[dict]:
    vc_modes = ["none", "dir", "wide25_dir", "wide50_dir", "wide75_dir"]
    pt_mults = [2.0, 2.5, 3.0, 4.0]
    sl_mults = [1.0, 1.5, 2.0]
    max_trades_opts = [1, 2]
    long_only_opts = [True, False]
    n = TARGET_CONTRACTS

    results = []
    total = len(vc_modes) * len(pt_mults) * len(sl_mults) * len(max_trades_opts) * len(long_only_opts)
    done = 0

    for vc_mode, pt, sl, mt, lo in product(vc_modes, pt_mults, sl_mults, max_trades_opts, long_only_opts):
        if lo and vc_mode == "none":
            done += 1
            continue  # skip: long-only without VC is covered in baseline
        r = run_orb(bars_2026, vc_meta, n, pt, sl, mt, vc_mode, long_only=lo)
        r.update({"vc_mode": vc_mode, "pt": pt, "sl": sl, "max_trades": mt, "long_only": lo})
        results.append(r)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{total} done...")

    return results


# ── VWAP mean reversion (simple) ──────────────────────────────────────────────
def run_vwap_mr(bars: pd.DataFrame, n_contracts: int,
                entry_dist_atr: float = 1.0, pt_mult: float = 2.0,
                sl_mult: float = 1.5, time_start: str = "10:30",
                time_end: str = "13:30", max_trades: int = 2) -> dict:
    """Simple VWAP mean reversion: enter when price deviates >= entry_dist_atr from VWAP."""
    atr_s = atr(bars["high"], bars["low"], bars["close"], 14)
    start_h, start_m = int(time_start.split(":")[0]), int(time_start.split(":")[1])
    end_h, end_m = int(time_end.split(":")[0]), int(time_end.split(":")[1])

    trades = []
    equity = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0
    daily_loss = 0.0
    pos = None

    dates = sorted(set(bars.index.date))
    for d in dates:
        day = bars[bars.index.date == d]
        if len(day) < 5:
            continue
        if cur_date is not None:
            daily_pnl[cur_date] = daily_loss
        cur_date = d
        daily_loss = 0.0
        trades_today = 0
        pos = None

        # Compute intraday VWAP
        tp = (day["high"] + day["low"] + day["close"]) / 3
        cum_vol = day["volume"].cumsum().replace(0, np.nan)
        vwap = (tp * day["volume"]).cumsum() / cum_vol

        for i in range(5, len(day)):
            bar = day.iloc[i]
            ts = day.index[i]
            ts_et = ts.tz_convert("US/Eastern") if ts.tzinfo else ts
            h, m = ts_et.hour, ts_et.minute

            in_window = (h > start_h or (h == start_h and m >= start_m)) and \
                        (h < end_h or (h == end_h and m <= end_m))

            if pos is not None:
                # Check exit
                cur_atr = float(atr_s.loc[ts]) if ts in atr_s.index else 0.0
                if cur_atr <= 0:
                    continue
                sl_price = pos["entry"] - pos["dir"] * sl_mult * pos["atr"]
                pt_price = pos["entry"] + pos["dir"] * pt_mult * pos["atr"]
                sess_close = (h == 15 and m >= 55)
                exited = False
                if pos["dir"] == 1:
                    if bar["low"] <= sl_price:
                        p = _pnl(pos["entry"], _slip(sl_price, 1, False), 1, n_contracts)
                        trades.append({"pnl": p, "reason": "stop_loss"})
                        exited = True
                    elif bar["high"] >= pt_price:
                        p = _pnl(pos["entry"], _slip(pt_price, 1, False), 1, n_contracts)
                        trades.append({"pnl": p, "reason": "profit_target"})
                        exited = True
                    elif sess_close or not in_window:
                        p = _pnl(pos["entry"], _slip(bar["close"], 1, False), 1, n_contracts)
                        trades.append({"pnl": p, "reason": "time_stop"})
                        exited = True
                else:
                    if bar["high"] >= sl_price:
                        p = _pnl(pos["entry"], _slip(sl_price, -1, False), -1, n_contracts)
                        trades.append({"pnl": p, "reason": "stop_loss"})
                        exited = True
                    elif bar["low"] <= pt_price:
                        p = _pnl(pos["entry"], _slip(pt_price, -1, False), -1, n_contracts)
                        trades.append({"pnl": p, "reason": "profit_target"})
                        exited = True
                    elif sess_close or not in_window:
                        p = _pnl(pos["entry"], _slip(bar["close"], -1, False), -1, n_contracts)
                        trades.append({"pnl": p, "reason": "time_stop"})
                        exited = True
                if exited:
                    daily_loss += trades[-1]["pnl"]
                    equity += trades[-1]["pnl"]
                    eq_vals.append(equity)
                    pos = None
                continue

            if not in_window or trades_today >= max_trades or daily_loss <= MAX_DAILY_LOSS:
                continue

            cur_atr = float(atr_s.loc[ts]) if ts in atr_s.index else 0.0
            if cur_atr <= 0:
                continue
            vwap_val = float(vwap.iloc[i])
            price = float(bar["close"])
            dev = (price - vwap_val) / cur_atr
            if abs(dev) < entry_dist_atr:
                continue
            # Mean reversion: if price > VWAP by enough → SHORT; below → LONG
            direction = -1 if dev > 0 else 1
            pos = {"entry": price, "dir": direction, "atr": cur_atr}
            trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "win_rate": 0, "sharpe": 0, "max_drawdown": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    eq = pd.Series(eq_vals)
    max_dd = float((eq - eq.cummax()).min())
    daily = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }


# ── Portfolio combiner ────────────────────────────────────────────────────────
def combine_portfolio(r1: dict, r2: dict) -> dict:
    """Combine two strategy results into a portfolio."""
    all_dates = sorted(set(list(r1["daily_pnl"].keys()) + list(r2["daily_pnl"].keys())))
    combined_pnl = {}
    for d in all_dates:
        combined_pnl[d] = r1["daily_pnl"].get(d, 0) + r2["daily_pnl"].get(d, 0)

    equity = 50_000.0
    eq_vals = [equity]
    for d in sorted(combined_pnl):
        equity += combined_pnl[d]
        eq_vals.append(equity)

    eq = pd.Series(eq_vals)
    max_dd = float((eq - eq.cummax()).min())
    total = sum(combined_pnl.values())
    daily = pd.Series({k: v for k, v in combined_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    n_trades = r1.get("n_trades", 0) + r2.get("n_trades", 0)
    wins_r1 = r1.get("win_rate", 0) * r1.get("n_trades", 0)
    wins_r2 = r2.get("win_rate", 0) * r2.get("n_trades", 0)
    wr = (wins_r1 + wins_r2) / n_trades if n_trades > 0 else 0

    return {
        "n_trades": n_trades,
        "win_rate": round(wr, 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    bars = pd.read_parquet(CACHE_PATH)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")

    bars_2026 = bars[bars.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
    print(f"2026 YTD: {len(bars_2026)} bars, {bars_2026.index[0].date()} → {bars_2026.index[-1].date()}")
    print(f"Trading days: {len(set(bars_2026.index.date))}")
    print(f"Target: ${TARGET_PNL:,.0f} PnL / {TARGET_CONTRACTS}c / ${TARGET_MAX_DD:,.0f} max DD\n")

    vc_meta = build_vc_meta(bars_2026)

    # ── ORB grid search ──────────────────────────────────────────────────────
    print("Running ORB grid search...")
    orb_results = grid_search(bars_2026, vc_meta)

    hits = [r for r in orb_results
            if r["total_pnl"] >= TARGET_PNL and r["max_drawdown"] >= TARGET_MAX_DD]

    print(f"\n{'='*70}")
    print(f"  ORB-ONLY: {len(hits)} configs hit the goal (${TARGET_PNL:,.0f} / ${TARGET_MAX_DD:,.0f} DD)")
    print(f"{'='*70}")

    # Show top 10 by PnL regardless of goal
    top = sorted(orb_results, key=lambda x: x["total_pnl"] - max(0, -x["max_drawdown"] - 2000) * 3, reverse=True)[:10]
    print(f"\n  Top 10 ORB configs (scored by PnL penalised for DD > $2k):")
    print(f"  {'VC mode':<14} {'PT':>4} {'SL':>4} {'MT':>3} {'LO':>4} | {'Trades':>7} {'WR':>6} {'PnL':>9} {'DD':>9} {'Sharpe':>7}")
    print(f"  {'-'*75}")
    for r in top:
        goal = "✓GOAL" if r["total_pnl"] >= TARGET_PNL and r["max_drawdown"] >= TARGET_MAX_DD else ""
        print(f"  {r['vc_mode']:<14} {r['pt']:>4.1f} {r['sl']:>4.1f} {r['max_trades']:>3} {'Y' if r['long_only'] else 'N':>4} | "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1%} ${r['total_pnl']:>8,.0f} ${r['max_drawdown']:>8,.0f} {r['sharpe']:>7.2f}  {goal}")

    # ── VWAP MR grid search ──────────────────────────────────────────────────
    print("\n\nRunning VWAP Mean Reversion search...")
    vwap_configs = [
        {"entry_dist_atr": d, "pt_mult": pt, "sl_mult": sl, "max_trades": mt}
        for d, pt, sl, mt in product([0.5, 0.75, 1.0, 1.5], [1.5, 2.0, 2.5], [1.0, 1.5], [2, 3])
    ]
    vwap_results = []
    for cfg in vwap_configs:
        r = run_vwap_mr(bars_2026, TARGET_CONTRACTS, **cfg)
        r.update(cfg)
        vwap_results.append(r)

    top_vwap = sorted(vwap_results, key=lambda x: x["total_pnl"], reverse=True)[:5]
    print(f"\n  Top VWAP MR configs:")
    print(f"  {'entry':>6} {'PT':>4} {'SL':>4} {'MT':>3} | {'Trades':>7} {'WR':>6} {'PnL':>9} {'DD':>9} {'Sharpe':>7}")
    print(f"  {'-'*65}")
    for r in top_vwap:
        print(f"  {r['entry_dist_atr']:>6.2f} {r['pt_mult']:>4.1f} {r['sl_mult']:>4.1f} {r['max_trades']:>3} | "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1%} ${r['total_pnl']:>8,.0f} ${r['max_drawdown']:>8,.0f} {r['sharpe']:>7.2f}")

    # ── Portfolio: best ORB + best VWAP ──────────────────────────────────────
    print("\n\nRunning ORB + VWAP portfolio search...")
    best_orb_candidates = sorted(
        [r for r in orb_results if r["n_trades"] >= 5],
        key=lambda x: x["sharpe"], reverse=True
    )[:10]
    best_vwap_candidates = sorted(
        [r for r in vwap_results if r["n_trades"] >= 5],
        key=lambda x: x["sharpe"], reverse=True
    )[:10]

    portfolio_hits = []
    for orb, vwap in product(best_orb_candidates, best_vwap_candidates):
        combined = combine_portfolio(orb, vwap)
        combined["orb_cfg"] = f"{orb['vc_mode']}/PT{orb['pt']}/SL{orb['sl']}/MT{orb['max_trades']}/LO{'Y' if orb['long_only'] else 'N'}"
        combined["vwap_cfg"] = f"dist{vwap['entry_dist_atr']}/PT{vwap['pt_mult']}/SL{vwap['sl_mult']}"
        combined["orb_pnl"] = orb["total_pnl"]
        combined["vwap_pnl"] = vwap["total_pnl"]
        portfolio_hits.append(combined)

    goal_hits = [p for p in portfolio_hits
                 if p["total_pnl"] >= TARGET_PNL and p["max_drawdown"] >= TARGET_MAX_DD]
    top_port = sorted(portfolio_hits, key=lambda x: x["total_pnl"] - max(0, -x["max_drawdown"] - 2000) * 5, reverse=True)[:10]

    print(f"\n  {len(goal_hits)} portfolio combos hit the goal!")
    print(f"\n  Top 10 ORB+VWAP portfolios:")
    print(f"  {'ORB config':<32} {'VWAP config':<22} | {'PnL':>9} {'DD':>9} {'Sharpe':>7}")
    print(f"  {'-'*80}")
    for p in top_port:
        goal = "✓GOAL" if p["total_pnl"] >= TARGET_PNL and p["max_drawdown"] >= TARGET_MAX_DD else ""
        print(f"  {p['orb_cfg']:<32} {p['vwap_cfg']:<22} | ${p['total_pnl']:>8,.0f} ${p['max_drawdown']:>8,.0f} {p['sharpe']:>7.2f}  {goal}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "target": {"pnl": TARGET_PNL, "max_dd": TARGET_MAX_DD, "contracts": TARGET_CONTRACTS},
        "orb_goal_hits": [r for r in hits],
        "top_orb": top,
        "top_vwap": top_vwap,
        "portfolio_goal_hits": goal_hits[:20],
        "top_portfolios": top_port,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
