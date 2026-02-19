"""
Historical ORB Backtest: ES 2010-2025 + MES Jan-Feb 2026
=========================================================
Validates that the best ORB config has PERSISTENT edge across market regimes.

Runs the confirmed best ORB params year-by-year on ES futures (2010-2025),
then the Jan-Feb 2026 OOS on MES.  Risk limits are intentionally disabled
so we see the raw strategy edge, not risk-manager interference.

Usage:
    cd "algos 3 topstep"
    python ml_intraday_v3/diagnostics/historical_orb_backtest.py -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent      # algos 3 topstep/
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.backtest_engine import BacktestEngine          # noqa: E402
from engine.signal_aggregator import SignalAggregator      # noqa: E402
from engine.risk_manager import RiskManager                # noqa: E402
from rules.opening_range import OpeningRangeBreakoutRule   # noqa: E402
from rules.time_of_day import TimeOfDayRule                # noqa: E402
from utils.data_loader import load_bars                    # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Best ORB Config (confirmed on Jan-Feb 2026 OOS)
# ---------------------------------------------------------------------------
BEST_ORB = {
    "or_end_time": "09:55",          # 09:30-09:55 = 6 bars opening range
    "entry_cutoff_time": "12:00",    # No entries after noon
    "min_range_atr": 0.3,            # Range >= 0.3x ATR
    "pt_atr_mult": 2.0,              # Profit target = 2x ATR
    "sl_atr_mult": 1.5,              # Stop loss = 1.5x ATR
    "trailing_activation_atr": 999.0,# Disabled (critical for profitability)
    "trailing_distance_atr": 0.75,
    "atr_period": 14,
    "time_stop_bars": 24,
    "commission_per_side": 0.62,
    "slippage_ticks": 1,
}

# ---------------------------------------------------------------------------
# Instrument specs
# ---------------------------------------------------------------------------
ES_SPEC = {"point_value": 50.0, "tick_size": 0.25, "tick_value": 12.50}
MES_SPEC = {"point_value": 5.0,  "tick_size": 0.25, "tick_value": 1.25}

# Risk limits — set extremely large so they never fire during edge study
UNLIMITED_RISK = {
    "max_daily_loss": -9_999_999.0,
    "per_trade_max_loss": 9_999_999.0,
    "max_consecutive_losses": 9_999,
    "cooldown_bars": 3,
    "flatten_minutes_before_close": 5,
    "drawdown_buffer": 9_999_999.0,
}


def build_engine(params: dict, spec: dict, n_contracts: int = 1) -> BacktestEngine:
    primary = OpeningRangeBreakoutRule(
        or_end_time=params["or_end_time"],
        min_range_atr=params["min_range_atr"],
        entry_cutoff_time=params["entry_cutoff_time"],
        atr_period=params["atr_period"],
    )
    filters = [
        TimeOfDayRule(
            session_start="09:35",
            session_end="15:45",
            lunch_filter_enabled=False,
        ),
    ]
    aggregator = SignalAggregator(
        primary_rule=primary,
        filter_rules=filters,
        confirmation_rules=[],
        min_confirmations=0,
    )
    risk_manager = RiskManager(
        contracts=n_contracts,
        point_value=spec["point_value"],
        tick_size=spec["tick_size"],
        tick_value=spec["tick_value"],
        **UNLIMITED_RISK,
    )
    engine = BacktestEngine(
        aggregator=aggregator,
        risk_manager=risk_manager,
        commission_per_side=params["commission_per_side"],
        slippage_ticks=params["slippage_ticks"],
        profit_target_atr=params["pt_atr_mult"],
        stop_loss_atr=params["sl_atr_mult"],
        time_stop_bars=params["time_stop_bars"],
        trailing_activation_atr=params["trailing_activation_atr"],
        trailing_distance_atr=params["trailing_distance_atr"],
        atr_period=params["atr_period"],
    )
    return engine


def run_period(
    bars: pd.DataFrame,
    spec: dict,
    label: str,
    n_contracts: int = 1,
) -> dict:
    """Run ORB backtest on a period, return summary dict."""
    if len(bars) < 90:
        return {
            "label": label,
            "n_bars": len(bars),
            "num_trades": 0,
            "win_rate": None,
            "profit_factor": None,
            "total_pnl": 0.0,
            "avg_trade_pnl": None,
            "sharpe_ratio": None,
            "max_drawdown": 0.0,
            "skip_reason": "insufficient bars",
        }

    engine = build_engine(BEST_ORB, spec, n_contracts=n_contracts)
    result = engine.run(bars, starting_equity=50_000.0)
    s = result.summary()

    return {
        "label": label,
        "n_bars": len(bars),
        "num_trades": s.get("num_trades", 0),
        "win_rate": round(s.get("win_rate", 0.0), 4),
        "profit_factor": round(s.get("profit_factor", 0.0), 4),
        "total_pnl": round(s.get("total_pnl", 0.0), 2),
        "avg_trade_pnl": round(s.get("avg_trade_pnl", 0.0), 2),
        "sharpe_ratio": round(s.get("sharpe_ratio", 0.0), 4),
        "max_drawdown": round(s.get("max_drawdown", 0.0), 2),
        "trade_pnls": [round(t.pnl, 2) for t in result.trades],
    }


def to_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter to Regular Trading Hours (9:30–15:59 ET)."""
    h = bars.index.hour
    m = bars.index.minute
    after_open = (h > 9) | ((h == 9) & (m >= 30))
    before_close = h < 16
    return bars[after_open & before_close]


def print_table(rows: List[dict]) -> None:
    header = (
        f"{'Period':<18} {'Bars':>7} {'Trades':>7} {'WR':>6} "
        f"{'PF':>6} {'Sharpe':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'MaxDD':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("skip_reason"):
            print(f"  {r['label']:<16}  (skipped: {r['skip_reason']})")
            continue
        wr = (f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "N/A").rjust(6)
        pf = (f"{r['profit_factor']:.2f}" if r["profit_factor"] is not None else "N/A").rjust(6)
        sh = (f"{r['sharpe_ratio']:.2f}" if r["sharpe_ratio"] is not None else "N/A").rjust(7)
        avg = (f"${r['avg_trade_pnl']:.0f}" if r["avg_trade_pnl"] is not None else "N/A").rjust(8)
        print(
            f"  {r['label']:<16} {r['n_bars']:>7,} {r['num_trades']:>7} "
            f"{wr} {pf} {sh} {avg} "
            f"${r['total_pnl']:>9,.0f} ${r['max_drawdown']:>8,.0f}"
        )


def aggregate_stats(rows: List[dict]) -> dict:
    """Compute aggregate stats across all periods (equal-weight per trade)."""
    valid = [r for r in rows if r.get("num_trades", 0) > 0 and r.get("win_rate") is not None]
    if not valid:
        return {}

    all_pnls: List[float] = []
    for r in valid:
        all_pnls.extend(r.get("trade_pnls", []))

    if not all_pnls:
        return {}

    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    total_win = sum(wins)
    total_loss = abs(sum(losses))

    return {
        "total_trades": len(all_pnls),
        "win_rate": round(len(wins) / len(all_pnls), 4),
        "profit_factor": round(total_win / total_loss, 3) if total_loss > 0 else None,
        "total_pnl": round(sum(all_pnls), 2),
        "avg_trade_pnl": round(sum(all_pnls) / len(all_pnls), 2),
        "median_trade_pnl": round(float(np.median(all_pnls)), 2),
        "positive_years": sum(1 for r in valid if r["total_pnl"] > 0),
        "total_years": len(valid),
    }


def main():
    parser = argparse.ArgumentParser(description="Historical ORB backtest 2010-2025 + Jan-Feb 2026")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "processed" / "es_bars_2010_2025.h5"),
    )
    parser.add_argument("--data-key", default="bars_5min")
    parser.add_argument(
        "--oos-path",
        default=str(_PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"),
    )
    parser.add_argument("--oos-key", default="bars_5min")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "diagnostics" / "historical_orb_results.json"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ------------------------------------------------------------------
    # Load ES 2010-2025
    # ------------------------------------------------------------------
    logger.info(f"Loading ES bars from {args.data_path}")
    es_all = load_bars(args.data_path, key=args.data_key)
    es_rth = to_rth(es_all)
    logger.info(f"ES RTH bars: {len(es_rth):,}  ({es_rth.index[0]} → {es_rth.index[-1]})")

    # ------------------------------------------------------------------
    # Load MES Jan-Feb 2026 OOS
    # ------------------------------------------------------------------
    logger.info(f"Loading MES OOS bars from {args.oos_path}")
    mes_all = load_bars(args.oos_path, key=args.oos_key)
    mes_rth = to_rth(mes_all)
    logger.info(f"MES OOS RTH bars: {len(mes_rth):,}  ({mes_rth.index[0]} → {mes_rth.index[-1]})")

    # ------------------------------------------------------------------
    # Year-by-year ES backtest
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("OPENING RANGE BREAKOUT — HISTORICAL EDGE VALIDATION")
    print(f"Config: or_end=09:55, pt=2.0x ATR, sl=1.5x ATR, no trailing stop")
    print(f"ES (point_value=$50/pt) — risk limits disabled to measure pure edge")
    print("=" * 90)

    year_rows: List[dict] = []
    for year in range(2010, 2026):
        year_bars = es_rth[es_rth.index.year == year]
        row = run_period(year_bars, ES_SPEC, label=str(year), n_contracts=1)
        year_rows.append(row)
        logger.info(
            f"Year {year}: {row['num_trades']} trades, "
            f"WR={row['win_rate']:.1%}, "
            f"PnL=${row['total_pnl']:,.0f}, "
            f"Sharpe={row['sharpe_ratio']}"
        )

    # Aggregate across all ES years
    es_agg = aggregate_stats(year_rows)

    print_table(year_rows)

    print("\n  --- ES 2010-2025 AGGREGATE ---")
    print(f"  Total trades:   {es_agg.get('total_trades', 0)}")
    print(f"  Win rate:       {es_agg.get('win_rate', 0):.1%}")
    print(f"  Profit factor:  {es_agg.get('profit_factor', 'N/A')}")
    print(f"  Avg trade PnL:  ${es_agg.get('avg_trade_pnl', 0):.2f}")
    print(f"  Total PnL:      ${es_agg.get('total_pnl', 0):,.0f}")
    print(f"  Positive years: {es_agg.get('positive_years', 0)}/{es_agg.get('total_years', 0)}")

    # ------------------------------------------------------------------
    # Jan-Feb 2026 MES OOS (n=2, confirmed best)
    # ------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("MES Jan-Feb 2026 OOS (n=2 contracts, MES point_value=$5/pt)")
    print("-" * 90)

    oos_row_n1 = run_period(mes_rth, MES_SPEC, label="Jan-Feb 2026 n=1", n_contracts=1)
    oos_row_n2 = run_period(mes_rth, MES_SPEC, label="Jan-Feb 2026 n=2", n_contracts=2)
    print_table([oos_row_n1, oos_row_n2])

    # ------------------------------------------------------------------
    # Regime analysis (positive years by market environment)
    # ------------------------------------------------------------------
    REGIME_MAP = {
        2010: "Post-GFC recovery",
        2011: "European debt crisis",
        2012: "Low-vol grind up",
        2013: "Strong bull market",
        2014: "Bull market, low vol",
        2015: "Choppy, China scare",
        2016: "Trump rally",
        2017: "Low-vol bull",
        2018: "Vol spike, selloff",
        2019: "Bull recovery",
        2020: "COVID crash + recovery",
        2021: "Meme stocks, bull run",
        2022: "Rate hike bear market",
        2023: "Recovery, AI rally",
        2024: "AI bull market",
        2025: "Late cycle",
    }

    print("\n" + "-" * 90)
    print("REGIME BREAKDOWN (edge persistence across different market environments)")
    print("-" * 90)
    print(f"  {'Year':<6} {'Regime':<28} {'Trades':>7} {'WR':>6} {'PnL':>10}  {'Edge?':>6}")
    print("  " + "-" * 70)
    for r in year_rows:
        if r.get("skip_reason"):
            continue
        label = r["label"]
        regime = REGIME_MAP.get(int(label), "")
        edge = "YES" if r["win_rate"] and r["win_rate"] > 0.45 else "---"
        wr_str = f"{r['win_rate']:.1%}".rjust(6)
        pnl_str = f"${r['total_pnl']:>8,.0f}"
        print(
            f"  {label:<6} {regime:<28} {r['num_trades']:>7} "
            f"{wr_str} {pnl_str:>10}  {edge:>6}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "meta": {
            "strategy": "opening_range_breakout",
            "config": BEST_ORB,
            "es_data": args.data_path,
            "oos_data": args.oos_path,
            "note": "Risk limits disabled to measure pure strategy edge",
        },
        "es_years": year_rows,
        "es_aggregate": es_agg,
        "oos_n1": oos_row_n1,
        "oos_n2": oos_row_n2,
    }

    with open(output_path, "w") as f:
        # Exclude trade_pnls from final JSON to keep it readable
        clean_data = json.loads(json.dumps(output_data))
        for row in clean_data.get("es_years", []):
            row.pop("trade_pnls", None)
        for key in ("oos_n1", "oos_n2"):
            clean_data[key].pop("trade_pnls", None)
        json.dump(clean_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
