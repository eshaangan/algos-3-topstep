"""Trend-pullback VWAP backtest for Topstep-style intraday futures.

Hypothesis:
  Trade MNQ/MES only during the active RTH window. Define trend with session
  VWAP plus an EMA, then enter the first pullback that holds VWAP and resumes
  in trend direction. Use fixed R exits, max two losses per day, and a modest
  daily profit stop.

Run:
  python rule_based_v1/diagnostics/trend_pullback_vwap_backtest.py
  python rule_based_v1/diagnostics/trend_pullback_vwap_backtest.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in (str(ROOT), str(RBV1)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr, ema

DATA = ROOT / "data" / "processed"
RESULTS_PATH = RBV1 / "diagnostics" / "trend_pullback_vwap_results.json"

DATASETS = {
    "MNQ": DATA / "mnq_2026ytd_5min.h5",
    "MNQ_DB": DATA / "mnq_2026ytd_databento_5min_rth.h5",
    "MES": DATA / "mes_2026_ytd_rth_5m.h5",
}

SPECS = {
    "MNQ": {"point_value": 2.0, "tick_size": 0.25, "commission": 0.62},
    "MNQ_DB": {"point_value": 2.0, "tick_size": 0.25, "commission": 0.62},
    "MES": {"point_value": 5.0, "tick_size": 0.25, "commission": 0.62},
}


@dataclass(frozen=True)
class Config:
    instrument: str = "MNQ"
    contracts: int = 3
    ema_period: int = 21
    atr_period: int = 14
    trend_bars: int = 3
    entry_start: str = "09:45"
    entry_end: str = "12:30"
    max_trades_day: int = 2
    max_losses_day: int = 2
    daily_goal: float = 900.0
    daily_loss_limit: float = -950.0
    target_r: float = 2.0
    stop_buffer_atr: float = 0.10
    min_stop_atr: float = 0.45
    max_stop_atr: float = 1.25
    pullback_tolerance_atr: float = 0.18
    trend_separation_atr: float = 0.05
    time_stop_bars: int = 18
    slippage_ticks: int = 1
    starting_equity: float = 50_000.0
    drawdown_buffer: float = 1_950.0
    allow_long: bool = True
    allow_short: bool = True
    confirm_breakout: bool = True


@dataclass
class Position:
    direction: int
    entry: float
    stop: float
    target: float
    entry_idx: int
    time_stop_idx: int
    date: str
    atr_entry: float


@dataclass
class PreparedBars:
    bars: pd.DataFrame
    vwap: pd.Series
    atr_by_period: dict[int, pd.Series]
    ema_by_period: dict[int, pd.Series]


def load_bars(instrument: str) -> pd.DataFrame:
    path = DATASETS[instrument]
    if not path.exists():
        raise FileNotFoundError(path)
    with pd.HDFStore(str(path), "r") as store:
        key = next((k.strip("/") for k in store.keys() if "5min" in k), store.keys()[0].strip("/"))
    df = pd.read_hdf(str(path), key=key).rename(columns=str.lower)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()
    rth = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df.loc[rth, ["open", "high", "low", "close", "volume"]].copy()


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    pv = typical * bars["volume"]
    out = pd.Series(np.nan, index=bars.index)
    for _, grp in bars.groupby(bars.index.date):
        idx = grp.index
        out.loc[idx] = (pv.loc[idx].cumsum() / bars.loc[idx, "volume"].cumsum().replace(0, np.nan)).values
    return out


def prepare_bars(bars: pd.DataFrame) -> PreparedBars:
    atr_by_period = {14: atr(bars["high"], bars["low"], bars["close"], 14)}
    ema_by_period = {p: ema(bars["close"], p) for p in (13, 21, 34)}
    return PreparedBars(
        bars=bars,
        vwap=session_vwap(bars),
        atr_by_period=atr_by_period,
        ema_by_period=ema_by_period,
    )


def slip(price: float, direction: int, is_entry: bool, cfg: Config) -> float:
    tick = SPECS[cfg.instrument]["tick_size"]
    amount = cfg.slippage_ticks * tick
    return price + direction * amount if is_entry else price - direction * amount


def pnl(entry: float, exit_: float, direction: int, cfg: Config) -> float:
    spec = SPECS[cfg.instrument]
    gross = (exit_ - entry) * direction * cfg.contracts * spec["point_value"]
    return gross - 2 * spec["commission"] * cfg.contracts


def parse_hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def in_entry_window(ts: pd.Timestamp, cfg: Config) -> bool:
    sh, sm = parse_hhmm(cfg.entry_start)
    eh, em = parse_hhmm(cfg.entry_end)
    t = (ts.hour, ts.minute)
    return (sh, sm) <= t < (eh, em)


def summarize(trades: list[dict], daily_pnl: dict[str, float], max_dd: float, cfg: Config) -> dict:
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    daily = pd.Series(daily_pnl, dtype=float).sort_index()
    active_daily = daily[daily != 0]
    sharpe = 0.0
    if len(active_daily) > 1 and active_daily.std(ddof=1) > 0:
        sharpe = float(active_daily.mean() / active_daily.std(ddof=1) * np.sqrt(252))
    monthly = {}
    for d, v in daily.items():
        ym = d[:7]
        monthly[ym] = monthly.get(ym, 0.0) + float(v)
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {
        "config": cfg.__dict__,
        "num_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "total_pnl": round(float(pnls.sum()) if len(pnls) else 0.0, 2),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 0.0, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float(max_dd), 2),
        "avg_trade": round(float(pnls.mean()) if len(pnls) else 0.0, 2),
        "best_day": round(float(daily.max()) if len(daily) else 0.0, 2),
        "worst_day": round(float(daily.min()) if len(daily) else 0.0, 2),
        "active_days": int((daily != 0).sum()),
        "monthly_pnl": {k: round(v, 2) for k, v in sorted(monthly.items())},
        "exit_reasons": reasons,
        "mll_ok": bool(max_dd > -cfg.drawdown_buffer),
        "daily_pnl": {k: round(float(v), 2) for k, v in daily.items()},
        "trades": trades,
    }


def run_backtest(prepared: PreparedBars, cfg: Config) -> dict:
    bars = prepared.bars
    vwap = prepared.vwap
    ema_s = prepared.ema_by_period.get(cfg.ema_period)
    if ema_s is None:
        ema_s = ema(bars["close"], cfg.ema_period)
    atr_s = prepared.atr_by_period.get(cfg.atr_period)
    if atr_s is None:
        atr_s = atr(bars["high"], bars["low"], bars["close"], cfg.atr_period)

    position: Position | None = None
    day = None
    trades_today = 0
    losses_today = 0
    day_pnl = 0.0
    trend_count = 0
    trend_dir = 0
    pullback_seen = False
    pull_low = np.nan
    pull_high = np.nan
    daily_pnl: dict[str, float] = {}
    trades: list[dict] = []
    equity = cfg.starting_equity
    peak = cfg.starting_equity
    max_dd = 0.0

    for i in range(max(cfg.atr_period, cfg.ema_period) + 3, len(bars)):
        ts = bars.index[i]
        cur_day = str(ts.date())
        bar = bars.iloc[i]
        prev = bars.iloc[i - 1]

        if day is not None and cur_day != day:
            daily_pnl[day] = daily_pnl.get(day, 0.0) + day_pnl
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            trend_count = 0
            trend_dir = 0
            pullback_seen = False
            pull_low = np.nan
            pull_high = np.nan
        day = cur_day

        is_last_bar = i + 1 >= len(bars) or str(bars.index[i + 1].date()) != cur_day
        session_close = is_last_bar or (ts.hour == 15 and ts.minute >= 55)

        if position is not None:
            exit_price = None
            reason = ""
            if session_close:
                exit_price = slip(float(bar["close"]), position.direction, False, cfg)
                reason = "session_close"
            elif i >= position.time_stop_idx:
                exit_price = slip(float(bar["close"]), position.direction, False, cfg)
                reason = "time_stop"
            elif position.direction == 1:
                if float(bar["low"]) <= position.stop:
                    exit_price = slip(position.stop, position.direction, False, cfg)
                    reason = "stop"
                elif float(bar["high"]) >= position.target:
                    exit_price = slip(position.target, position.direction, False, cfg)
                    reason = "target"
            else:
                if float(bar["high"]) >= position.stop:
                    exit_price = slip(position.stop, position.direction, False, cfg)
                    reason = "stop"
                elif float(bar["low"]) <= position.target:
                    exit_price = slip(position.target, position.direction, False, cfg)
                    reason = "target"

            if exit_price is not None:
                trade_pnl = pnl(position.entry, exit_price, position.direction, cfg)
                trades.append({
                    "date": position.date,
                    "time": str(ts),
                    "direction": "LONG" if position.direction == 1 else "SHORT",
                    "entry": round(position.entry, 2),
                    "exit": round(float(exit_price), 2),
                    "pnl": round(float(trade_pnl), 2),
                    "r_multiple": round(float(trade_pnl / max(abs(position.entry - position.stop) * cfg.contracts * SPECS[cfg.instrument]["point_value"], 1e-9)), 2),
                    "reason": reason,
                    "atr": round(position.atr_entry, 2),
                })
                equity += trade_pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                day_pnl += trade_pnl
                if trade_pnl < 0:
                    losses_today += 1
                position = None

        close = float(bar["close"])
        vwap_now = float(vwap.iloc[i])
        ema_now = float(ema_s.iloc[i])
        atr_now = float(atr_s.iloc[i])
        if not np.isfinite(vwap_now + ema_now + atr_now) or atr_now <= 0:
            continue

        ema_slope = ema_s.iloc[i] - ema_s.iloc[i - 3]
        vwap_slope = vwap.iloc[i] - vwap.iloc[i - 3]
        sep = cfg.trend_separation_atr * atr_now
        cur_dir = 0
        if close > vwap_now + sep and close > ema_now and ema_slope > 0 and vwap_slope >= 0:
            cur_dir = 1
        elif close < vwap_now - sep and close < ema_now and ema_slope < 0 and vwap_slope <= 0:
            cur_dir = -1

        if cur_dir == trend_dir and cur_dir != 0:
            trend_count += 1
        else:
            trend_dir = cur_dir
            trend_count = 1 if cur_dir != 0 else 0
            pullback_seen = False
            pull_low = np.nan
            pull_high = np.nan

        trend_ready = trend_count >= cfg.trend_bars
        if trend_ready and position is None:
            if trend_dir == 1:
                held = float(bar["low"]) >= vwap_now - cfg.pullback_tolerance_atr * atr_now
                touched = float(bar["low"]) <= max(vwap_now, ema_now) + cfg.pullback_tolerance_atr * atr_now
                if held and touched:
                    pullback_seen = True
                    pull_low = float(bar["low"]) if np.isnan(pull_low) else min(pull_low, float(bar["low"]))
            elif trend_dir == -1:
                held = float(bar["high"]) <= vwap_now + cfg.pullback_tolerance_atr * atr_now
                touched = float(bar["high"]) >= min(vwap_now, ema_now) - cfg.pullback_tolerance_atr * atr_now
                if held and touched:
                    pullback_seen = True
                    pull_high = float(bar["high"]) if np.isnan(pull_high) else max(pull_high, float(bar["high"]))

        can_enter = (
            position is None
            and not session_close
            and in_entry_window(ts, cfg)
            and trades_today < cfg.max_trades_day
            and losses_today < cfg.max_losses_day
            and day_pnl < cfg.daily_goal
            and day_pnl > cfg.daily_loss_limit
            and equity - peak > -cfg.drawdown_buffer
        )

        if can_enter and trend_ready and pullback_seen:
            direction = trend_dir
            if (direction == 1 and not cfg.allow_long) or (direction == -1 and not cfg.allow_short):
                continue
            signal = False
            if direction == 1:
                resumed = close > float(prev["high"]) if cfg.confirm_breakout else close > float(bar["open"])
                signal = close > vwap_now and close > ema_now and resumed
                raw_stop = min(float(pull_low), vwap_now) - cfg.stop_buffer_atr * atr_now
                risk_pts = close - raw_stop
            elif direction == -1:
                resumed = close < float(prev["low"]) if cfg.confirm_breakout else close < float(bar["open"])
                signal = close < vwap_now and close < ema_now and resumed
                raw_stop = max(float(pull_high), vwap_now) + cfg.stop_buffer_atr * atr_now
                risk_pts = raw_stop - close
            else:
                risk_pts = 0.0

            if signal and cfg.min_stop_atr * atr_now <= risk_pts <= cfg.max_stop_atr * atr_now:
                entry = slip(close, direction, True, cfg)
                stop = entry - direction * risk_pts
                target = entry + direction * risk_pts * cfg.target_r
                position = Position(
                    direction=direction,
                    entry=entry,
                    stop=stop,
                    target=target,
                    entry_idx=i,
                    time_stop_idx=i + cfg.time_stop_bars,
                    date=cur_day,
                    atr_entry=atr_now,
                )
                trades_today += 1
                pullback_seen = False
                pull_low = np.nan
                pull_high = np.nan

    if day is not None and day not in daily_pnl:
        daily_pnl[day] = daily_pnl.get(day, 0.0) + day_pnl
    return summarize(trades, daily_pnl, max_dd, cfg)


def candidate_configs(instruments: tuple[str, ...] = ("MNQ", "MES")) -> list[Config]:
    configs = []
    for instrument, contracts, ema_period, target_r, entry_end, tol, stop_buf, daily_goal, time_stop, allow_short, confirm_breakout in product(
        instruments,
        (5, 8, 12),
        (13, 21),
        (1.5, 2.0),
        ("12:30", "13:30"),
        (0.18, 0.25),
        (0.10,),
        (900.0,),
        (18,),
        (False, True),
        (False, True),
    ):
        configs.append(Config(
            instrument=instrument,
            contracts=contracts,
            ema_period=ema_period,
            target_r=target_r,
            entry_end=entry_end,
            pullback_tolerance_atr=tol,
            stop_buffer_atr=stop_buf,
            daily_goal=daily_goal,
            time_stop_bars=time_stop,
            allow_short=allow_short,
            confirm_breakout=confirm_breakout,
        ))
    return configs


def print_row(r: dict) -> None:
    c = r["config"]
    side = "bi" if c["allow_short"] else "long"
    confirm = "break" if c["confirm_breakout"] else "bounce"
    print(
        f"{c['instrument']:<3} {c['contracts']:>2}c ema={c['ema_period']:<2} "
        f"R={c['target_r']:<3} end={c['entry_end']} tol={c['pullback_tolerance_atr']:<4} "
        f"{side:<4} {confirm:<6} goal={c['daily_goal']:<5.0f} n={r['num_trades']:>3} wr={r['win_rate']:>5.1%} "
        f"pnl=${r['total_pnl']:>8,.0f} dd=${r['max_drawdown']:>7,.0f} "
        f"pf={r['profit_factor']:>4} sh={r['sharpe']:>5}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--instrument", choices=sorted(DATASETS), default="MNQ")
    args = parser.parse_args()

    if args.sweep:
        bars_cache = {args.instrument: prepare_bars(load_bars(args.instrument))}
        results = []
        for cfg in candidate_configs((args.instrument,)):
            r = run_backtest(bars_cache[cfg.instrument], cfg)
            if r["num_trades"] >= 8 and r["mll_ok"]:
                results.append(r)
        results.sort(key=lambda x: (x["total_pnl"], x["sharpe"]), reverse=True)
        payload = {
            "benchmarks": {
                "best_single_saved_novel_filter": 6004.99,
                "best_saved_combined_portfolio": 7344.12,
            },
            "top": results[:25],
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2))
        print(f"\nTop {min(15, len(results))} MLL-valid trend-pullback configs")
        print("-" * 132)
        for r in results[:15]:
            print_row(r)
        print(f"\nWrote {RESULTS_PATH}")
    else:
        cfg = Config(instrument=args.instrument)
        r = run_backtest(prepare_bars(load_bars(cfg.instrument)), cfg)
        print_row(r)
        RESULTS_PATH.write_text(json.dumps({"single": r}, indent=2))
        print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
