"""Honest multi-regime strategy search on index-futures (ES/MES) RTH 5-min.

Discipline:
  - DEV period (default 2010-2021): all parameter selection happens here.
  - HOLDOUT (default 2022-2025): each strategy family is tested ONCE; the number
    of families tested is fed to the DSR as n_trials so selection bias is paid for.
  - Economics: MES micro S&P ($5/pt, 0.25 tick), 1-tick slippage each way, real
    commission. ES and MES share the same underlying price series; we backtest on
    the 16-year ES series and apply MES contract economics.

Run:
    python rule_based_v1/validation/research_index_futures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rule_based_v1.validation.harness import (  # noqa: E402
    SimParams, simulate, aggregate_stats, monthly_breakdown,
    PreRegistration, evaluate,
)

ET = "America/New_York"

# MES economics
MES = dict(point_value=5.0, tick_size=0.25, commission_per_side=0.62, slippage_ticks=1)


# ─────────────────────────────────────────────────── data + features ─────────

def load_es() -> pd.DataFrame:
    d = pd.read_hdf(ROOT / "data/processed/es_bars_2010_2025.h5", "/bars_5min")
    ts = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index(ts).sort_index()[["open", "high", "low", "close", "volume"]]
    return d[~d.index.duplicated()]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    et = d.index.tz_convert(ET)
    d = d.copy()
    d["et_hour"] = et.hour
    d["et_min"] = et.minute
    d["et_date"] = et.date
    d["dow"] = et.dayofweek

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(span=14, min_periods=14).mean()

    # intraday VWAP (reset each ET day)
    tp = (h + l + c) / 3
    grp = pd.Series(et.date, index=d.index)
    cumv = v.groupby(grp).cumsum().replace(0, np.nan)
    d["vwap"] = (tp * v).groupby(grp).cumsum() / cumv
    d["vwap_dev_atr"] = (c - d["vwap"]) / d["atr"]

    # RSI(14)
    diff = c.diff()
    up = diff.clip(lower=0).ewm(com=13, min_periods=14).mean()
    dn = (-diff.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # session bar index (0 at 09:30)
    d["bar_of_day"] = d.groupby(grp).cumcount()
    return d


# ─────────────────────────────────────────────────── strategies ──────────────
# Each returns a `direction` Series (1/-1/0). Entry session limited to 09:35-15:00
# ET (bar_of_day 1..66) and no new entries in the last 12 bars before close.

def _session_ok(d: pd.DataFrame) -> pd.Series:
    return (d["bar_of_day"] >= 1) & (d["et_hour"] < 15)


def strat_mean_reversion(d: pd.DataFrame, dev_atr: float, rsi_hi: float) -> pd.Series:
    """Fade stretched moves: short when far above VWAP & RSI hot, long when far below & cold."""
    ok = _session_ok(d) & d["atr"].notna() & d["vwap_dev_atr"].notna() & d["rsi"].notna()
    direction = pd.Series(0, index=d.index, dtype=int)
    longs = ok & (d["vwap_dev_atr"] <= -dev_atr) & (d["rsi"] <= (100 - rsi_hi))
    shorts = ok & (d["vwap_dev_atr"] >= dev_atr) & (d["rsi"] >= rsi_hi)
    direction[longs] = 1
    direction[shorts] = -1
    return direction


def strat_orb(d: pd.DataFrame, or_bars: int, buffer_atr: float) -> pd.Series:
    """Opening-range breakout: after first `or_bars`, go with the first close beyond
    the OR high/low (+/- buffer*atr). One breakout direction per day."""
    direction = pd.Series(0, index=d.index, dtype=int)
    grp = pd.Series(d["et_date"].values, index=d.index)
    for _, idx in d.groupby(grp).groups.items():
        day = d.loc[idx]
        if len(day) <= or_bars + 2:
            continue
        orh = day["high"].iloc[:or_bars].max()
        orl = day["low"].iloc[:or_bars].min()
        atr0 = day["atr"].iloc[or_bars] if or_bars < len(day) else np.nan
        if not np.isfinite(atr0) or atr0 <= 0:
            continue
        buf = buffer_atr * atr0
        post = day.iloc[or_bars:]
        post = post[post["et_hour"] < 15]
        up = post[post["close"] > orh + buf]
        dn = post[post["close"] < orl - buf]
        first_up = up.index[0] if len(up) else None
        first_dn = dn.index[0] if len(dn) else None
        if first_up is not None and (first_dn is None or first_up <= first_dn):
            direction[first_up] = 1
        elif first_dn is not None:
            direction[first_dn] = -1
    return direction


def strat_momentum(d: pd.DataFrame, lookback: int, thresh_atr: float) -> pd.Series:
    """Continuation: go with a strong `lookback`-bar move that exceeds thresh*atr."""
    ok = _session_ok(d) & d["atr"].notna()
    mom = (d["close"] - d["close"].shift(lookback)) / d["atr"]
    direction = pd.Series(0, index=d.index, dtype=int)
    direction[ok & (mom >= thresh_atr)] = 1
    direction[ok & (mom <= -thresh_atr)] = -1
    return direction


# ─────────────────────────────────────────────────── runner ──────────────────

def run_family(name, d, signal_fn, grid, dev_end, hold_start, hold_end, p: SimParams,
               n_trials_for_dsr):
    """Select best param combo on DEV by total PnL; test once on HOLDOUT."""
    dev = d[d.index < dev_end]
    hold = d[(d.index >= hold_start) & (d.index < hold_end)]

    best, best_pnl, best_dev = None, -1e18, None
    for combo in grid:
        sig = signal_fn(dev, *combo)
        tr = simulate(dev[["open", "high", "low", "close"]],
                      pd.DataFrame({"direction": sig, "atr": dev["atr"]}), p)
        agg = aggregate_stats(tr)
        # require a sane trade count on DEV so we don't pick a degenerate combo
        if agg["n_trades"] < 200:
            continue
        if agg["total_pnl"] > best_pnl:
            best_pnl, best, best_dev = agg["total_pnl"], combo, agg

    if best is None:
        print(f"\n### {name}: no viable DEV combo")
        return None

    sig_h = signal_fn(hold, *best)
    tr_h = simulate(hold[["open", "high", "low", "close"]],
                    pd.DataFrame({"direction": sig_h, "atr": hold["atr"]}), p)

    prereg = PreRegistration(
        strategy_id=f"{name}{best}",
        holdout={"start": str(hold_start.date()), "end": str(hold_end.date())},
        dsr_n_trials=n_trials_for_dsr,
        gate={"min_oos_trades": 300, "min_months": 24,
              "min_positive_month_fraction": 0.55, "require_net_profitable": True,
              "min_dsr": 0.95, "max_drawdown_usd": 2500},
        sim={f: getattr(p, f) for f in SimParams().__dataclass_fields__},
    )
    res = evaluate(tr_h, prereg)
    mb = monthly_breakdown(tr_h)
    pos_frac = float((mb["pnl"] > 0).mean()) if len(mb) else 0.0

    print(f"\n### {name}  best_params={best}")
    print(f"  DEV   : {best_dev['n_trades']} tr  WR={best_dev['win_rate']:.1%}  "
          f"PnL=${best_dev['total_pnl']:,.0f}  DD=${best_dev['max_dd']:,.0f}  "
          f"Sharpe/tr={best_dev['sharpe_per_trade']:.3f}")
    a = res["aggregate"]
    dsr = res["dsr"].get("dsr")
    print(f"  HOLD  : {a['n_trades']} tr  WR={a['win_rate']:.1%}  "
          f"PnL=${a['total_pnl']:,.0f}  DD=${a['max_dd']:,.0f}  "
          f"Sharpe/tr={a['sharpe_per_trade']:.3f}  PF={a['profit_factor']:.2f}")
    print(f"  HOLD  : positive-months={pos_frac:.0%} ({len(mb)} mo)  "
          f"DSR={dsr:.3f}" if dsr is not None else f"  DSR=n/a")
    print(f"  VERDICT: {res['verdict']}  failures={res['failures']}")
    # yearly PnL
    if a["n_trades"] > 0:
        ty = tr_h.copy(); ty["yr"] = pd.to_datetime(ty["entry_time"]).dt.year
        yearly = ty.groupby("yr")["pnl"].agg(["sum", "count"])
        print("  yearly:", {int(y): (round(r['sum']), int(r['count'])) for y, r in yearly.iterrows()})
    return res


def main():
    print("Loading ES 2010-2025 RTH 5-min …")
    d = add_features(load_es())
    dev_end = pd.Timestamp("2022-01-01", tz="UTC")
    hold_start = pd.Timestamp("2022-01-01", tz="UTC")
    hold_end = pd.Timestamp("2026-01-01", tz="UTC")
    print(f"DEV < {dev_end.date()}  |  HOLDOUT {hold_start.date()}–{hold_end.date()}")

    # base sim: 2 MES contracts, ATR PT/SL, time stop, modest daily cap
    p = SimParams(pt_atr=2.0, sl_atr=1.0, horizon_bars=12, cooldown_bars=2,
                  max_trades_per_day=3, n_contracts=2, **MES)

    N_FAMILIES = 3  # honest count of distinct families tested on the holdout

    grids = {
        "MeanRev": [(dev, rsi) for dev in (1.0, 1.5, 2.0) for rsi in (65, 70, 75)],
        "ORB": [(ob, bf) for ob in (3, 6, 9) for bf in (0.0, 0.1, 0.25)],
        "Momentum": [(lb, th) for lb in (3, 6, 12) for th in (1.0, 1.5, 2.0)],
    }
    fns = {"MeanRev": strat_mean_reversion, "ORB": strat_orb, "Momentum": strat_momentum}

    for name in ("MeanRev", "ORB", "Momentum"):
        run_family(name, d, fns[name], grids[name], dev_end, hold_start, hold_end, p,
                   n_trials_for_dsr=N_FAMILIES * len(grids[name]))


if __name__ == "__main__":
    main()
