"""
MAE/MFE Empirical Barrier Optimization
=======================================
Analyzes Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
from completed trades to derive empirically-optimal stop loss and profit target
ATR multiples — replacing ATR-heuristic grid search with trade-level evidence.

Based on: "Maximum Adverse Excursion" (MAE) methodology for stop placement.

Usage:
    # After running validate_rule_based_oos.py to get trade data:
    python ml_intraday_v3/diagnostics/mae_mfe_analysis.py \
        --trades-file ml_intraday_v3/diagnostics/rule_based_oos_results.json \
        --data-path data/processed/mes_bars_databento_rth.h5 \
        --output ml_intraday_v3/diagnostics/mae_mfe_results.json

    # Or with raw trade CSV:
    python ml_intraday_v3/diagnostics/mae_mfe_analysis.py \
        --trades-csv /path/to/trades.csv \
        --output ml_intraday_v3/diagnostics/mae_mfe_results.json

Outputs:
    - Recommended {pt_atr_mult, sl_atr_mult} overriding ATR heuristics
    - JSON with full analysis
    - (Optional) matplotlib scatter plots saved to diagnostics/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# MES point value
MES_POINT_VALUE = 5.0


def compute_intra_trade_mae_mfe(
    trades: List[dict],
    bars: pd.DataFrame,
    atr_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    For each completed trade, compute intra-trade MAE and MFE using bar data.

    MAE = Maximum Adverse Excursion = deepest loss during the trade (absolute price units).
    MFE = Maximum Favorable Excursion = highest gain during the trade.

    Parameters
    ----------
    trades : list of dict
        Each dict must have: entry_bar, exit_bar, direction, entry_price, pnl, exit_reason.
        Optional: entry_time, exit_time.
    bars : pd.DataFrame
        OHLCV bars with DatetimeIndex. Must cover the trade periods.
    atr_series : pd.Series, optional
        ATR values aligned to bars index. If None, computed internally.

    Returns
    -------
    pd.DataFrame with columns:
        entry_bar, exit_bar, direction, entry_price, pnl, exit_reason,
        mae_price, mfe_price, mae_atr, mfe_atr, outcome (win/loss)
    """
    bars_arr = bars.reset_index()

    # Compute ATR if not provided
    if atr_series is None:
        from utils.indicators import atr as compute_atr  # type: ignore
        atr_series = compute_atr(bars["high"], bars["low"], bars["close"], period=14)

    records = []
    for trade in trades:
        entry_idx = trade.get("entry_bar", None)
        exit_idx = trade.get("exit_bar", None)

        if entry_idx is None or exit_idx is None:
            continue
        if entry_idx >= len(bars) or exit_idx > len(bars):
            continue
        if entry_idx >= exit_idx:
            continue

        direction = trade.get("direction", 1)
        entry_price = trade.get("entry_price", float("nan"))
        pnl = trade.get("pnl", 0.0)
        exit_reason = trade.get("exit_reason", "unknown")

        # Slice bars within the trade [entry_bar, exit_bar]
        trade_bars = bars.iloc[entry_idx:exit_idx + 1]

        if len(trade_bars) == 0:
            continue

        # ATR at entry bar
        try:
            atr_at_entry = float(atr_series.iloc[entry_idx])
        except (IndexError, KeyError):
            atr_at_entry = float("nan")

        if direction == 1:  # LONG
            # MAE: how far price fell below entry (adverse for LONG)
            mae_price = max(0.0, entry_price - float(trade_bars["low"].min()))
            # MFE: how far price rose above entry (favorable for LONG)
            mfe_price = max(0.0, float(trade_bars["high"].max()) - entry_price)
        else:  # SHORT
            mae_price = max(0.0, float(trade_bars["high"].max()) - entry_price)
            mfe_price = max(0.0, entry_price - float(trade_bars["low"].min()))

        mae_atr = mae_price / atr_at_entry if atr_at_entry > 0 else float("nan")
        mfe_atr = mfe_price / atr_at_entry if atr_at_entry > 0 else float("nan")

        records.append({
            "entry_bar": entry_idx,
            "exit_bar": exit_idx,
            "direction": direction,
            "entry_price": entry_price,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "mae_price": mae_price,
            "mfe_price": mfe_price,
            "mae_atr": mae_atr,
            "mfe_atr": mfe_atr,
            "outcome": "win" if pnl > 0 else "loss",
            "atr_at_entry": atr_at_entry,
        })

    return pd.DataFrame(records)


def find_optimal_sl(
    mae_df: pd.DataFrame,
    max_winner_crossing_pct: float = 0.05,
) -> Tuple[float, dict]:
    """
    Find optimal SL = MAE percentile where <= max_winner_crossing_pct of eventual
    winners would have been stopped out.

    Parameters
    ----------
    mae_df : pd.DataFrame
        Output from compute_intra_trade_mae_mfe.
    max_winner_crossing_pct : float
        Max fraction of winners we're willing to stop out early. Default 5%.

    Returns
    -------
    (optimal_sl_atr, diagnostics_dict)
    """
    winners = mae_df[mae_df["outcome"] == "win"]["mae_atr"].dropna()
    losers = mae_df[mae_df["outcome"] == "loss"]["mae_atr"].dropna()

    if len(winners) == 0:
        return 1.5, {"error": "no winners in dataset"}

    # For each candidate SL level, what fraction of winners would be stopped?
    sl_candidates = np.arange(0.25, 4.0, 0.05)
    results = []
    for sl in sl_candidates:
        winners_stopped = (winners <= sl).mean()
        losers_stopped = (losers <= sl).mean()
        results.append({
            "sl_atr": round(float(sl), 2),
            "pct_winners_stopped": round(float(winners_stopped), 4),
            "pct_losers_stopped": round(float(losers_stopped), 4),
        })

    results_df = pd.DataFrame(results)

    # Find smallest SL where <= max_winner_crossing_pct winners would be stopped
    # (i.e., tight enough to cut losers but loose enough not to kill winners)
    valid = results_df[results_df["pct_winners_stopped"] <= max_winner_crossing_pct]
    if len(valid) == 0:
        # Relax: take the level where pct_winners_stopped is minimized
        optimal_row = results_df.iloc[0]
    else:
        # Among valid, take the tightest (smallest SL)
        optimal_row = valid.iloc[0]

    optimal_sl = float(optimal_row["sl_atr"])

    diagnostics = {
        "optimal_sl_atr": optimal_sl,
        "pct_winners_stopped_at_optimal": float(optimal_row["pct_winners_stopped"]),
        "pct_losers_stopped_at_optimal": float(optimal_row["pct_losers_stopped"]),
        "winner_mae_p50": round(float(winners.quantile(0.50)), 3),
        "winner_mae_p90": round(float(winners.quantile(0.90)), 3),
        "winner_mae_p95": round(float(winners.quantile(0.95)), 3),
        "loser_mae_p50": round(float(losers.quantile(0.50)), 3),
        "loser_mae_p75": round(float(losers.quantile(0.75)), 3),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "sweep": results_df.to_dict(orient="records"),
    }
    return optimal_sl, diagnostics


def find_optimal_pt(
    mfe_df: pd.DataFrame,
    pt_candidates: Optional[np.ndarray] = None,
) -> Tuple[float, dict]:
    """
    Find optimal PT = MFE level that maximizes (win_rate × avg_win) - (loss_rate × avg_loss).

    Simulates: "if we had set PT = X × ATR, what would the outcome be?"

    Parameters
    ----------
    mfe_df : pd.DataFrame
        Must have columns: mfe_atr, mae_atr, outcome, pnl.
    pt_candidates : array-like, optional
        PT ATR multiples to test. Default: 0.5 to 5.0 in 0.1 steps.

    Returns
    -------
    (optimal_pt_atr, diagnostics_dict)
    """
    if pt_candidates is None:
        pt_candidates = np.arange(0.5, 5.1, 0.1)

    results = []
    n = len(mfe_df)
    if n == 0:
        return 2.0, {"error": "empty dataframe"}

    avg_loss = abs(mfe_df[mfe_df["outcome"] == "loss"]["pnl"].mean()) if (mfe_df["outcome"] == "loss").any() else 1.0

    for pt in pt_candidates:
        # Trades where MFE >= PT → would have hit profit target
        hit_target = mfe_df["mfe_atr"] >= pt
        n_hits = hit_target.sum()

        if n_hits == 0:
            continue

        win_rate_sim = n_hits / n
        # Avg win if we take profit at PT: PT × ATR × point_value per contract
        # We approximate: the win = pt × avg_atr_at_entry × MES_POINT_VALUE
        # But we can use actual pnl scaled to new PT
        # Simpler: expected_value = win_rate_sim * (pt * avg_atr * pv) - loss_rate_sim * avg_loss
        avg_atr = mfe_df["atr_at_entry"].mean()
        simulated_avg_win = pt * avg_atr * MES_POINT_VALUE if not np.isnan(avg_atr) else pt * 10

        loss_rate_sim = 1.0 - win_rate_sim
        expected_value = win_rate_sim * simulated_avg_win - loss_rate_sim * avg_loss
        profit_factor = (win_rate_sim * simulated_avg_win) / (loss_rate_sim * avg_loss + 1e-9)

        results.append({
            "pt_atr": round(float(pt), 2),
            "sim_win_rate": round(float(win_rate_sim), 4),
            "sim_expected_value": round(float(expected_value), 2),
            "sim_profit_factor": round(float(profit_factor), 4),
            "n_would_hit": int(n_hits),
        })

    if not results:
        return 2.0, {"error": "no valid PT candidates"}

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["sim_expected_value"].idxmax()]
    optimal_pt = float(best_row["pt_atr"])

    # MFE percentiles
    mfe_vals = mfe_df["mfe_atr"].dropna()
    diagnostics = {
        "optimal_pt_atr": optimal_pt,
        "sim_win_rate_at_optimal": float(best_row["sim_win_rate"]),
        "sim_ev_at_optimal": float(best_row["sim_expected_value"]),
        "sim_pf_at_optimal": float(best_row["sim_profit_factor"]),
        "mfe_p25": round(float(mfe_vals.quantile(0.25)), 3),
        "mfe_p50": round(float(mfe_vals.quantile(0.50)), 3),
        "mfe_p75": round(float(mfe_vals.quantile(0.75)), 3),
        "mfe_p90": round(float(mfe_vals.quantile(0.90)), 3),
        "n_total": len(mfe_df),
        "sweep": results_df.sort_values("sim_expected_value", ascending=False).head(20).to_dict(orient="records"),
    }
    return optimal_pt, diagnostics


def try_plot(mae_mfe_df: pd.DataFrame, output_dir: Path):
    """Generate scatter plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        wins = mae_mfe_df[mae_mfe_df["outcome"] == "win"]
        losses = mae_mfe_df[mae_mfe_df["outcome"] == "loss"]

        # MAE vs MFE scatter
        ax = axes[0]
        ax.scatter(wins["mae_atr"], wins["mfe_atr"], alpha=0.4, color="green", s=20, label="Win")
        ax.scatter(losses["mae_atr"], losses["mfe_atr"], alpha=0.4, color="red", s=20, label="Loss")
        ax.set_xlabel("MAE (ATR multiples)")
        ax.set_ylabel("MFE (ATR multiples)")
        ax.set_title("MAE vs MFE by Trade Outcome")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MFE distribution by outcome
        ax2 = axes[1]
        bins = np.arange(0, 5, 0.25)
        ax2.hist(wins["mfe_atr"].dropna(), bins=bins, alpha=0.5, color="green", label="Win", density=True)
        ax2.hist(losses["mfe_atr"].dropna(), bins=bins, alpha=0.5, color="red", label="Loss", density=True)
        ax2.set_xlabel("MFE (ATR multiples)")
        ax2.set_ylabel("Density")
        ax2.set_title("MFE Distribution by Outcome")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "mae_mfe_scatter.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to {plot_path}")
    except ImportError:
        logger.info("matplotlib not available; skipping plot generation")


def load_trades_from_oos_json(json_path: str) -> Tuple[List[dict], dict]:
    """Extract trade list and best params from validate_rule_based_oos.py output."""
    with open(json_path) as f:
        data = json.load(f)

    # Prefer top result
    top5 = data.get("top5", [])
    if not top5:
        all_results = data.get("all_results", [])
        top5 = all_results[:1] if all_results else []

    if not top5:
        return [], {}

    # The OOS JSON doesn't store raw intra-trade bar data, only summary.
    # We'll need to re-run the backtest with the best params to get trade-level data.
    best_params = top5[0].get("params", {})
    # Return empty trades — caller will re-run backtest
    return [], best_params


def main():
    parser = argparse.ArgumentParser(description="MAE/MFE empirical barrier optimization")
    parser.add_argument(
        "--oos-results",
        type=str,
        help="Path to rule_based_oos_results.json (output of validate_rule_based_oos.py)",
    )
    parser.add_argument(
        "--trades-json",
        type=str,
        help="Path to JSON with trade list (entry_bar, exit_bar, direction, entry_price, pnl)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"),
        help="Path to HDF5 bar data",
    )
    parser.add_argument("--data-key", type=str, default="bars_5min")
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-02-10")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "diagnostics" / "mae_mfe_results.json"),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load bar data
    from utils.data_loader import load_bars  # type: ignore
    logger.info(f"Loading bars from {args.data_path}")
    bars = load_bars(args.data_path, key=args.data_key, start_date=args.start, end_date=args.end)
    logger.info(f"Loaded {len(bars):,} bars")

    # Load or generate trades
    trades = []
    best_params = {}

    if args.trades_json:
        with open(args.trades_json) as f:
            trade_data = json.load(f)
        trades = trade_data if isinstance(trade_data, list) else trade_data.get("trades", [])
        logger.info(f"Loaded {len(trades)} trades from {args.trades_json}")

    elif args.oos_results:
        logger.info(f"Loading best params from {args.oos_results}")
        _, best_params = load_trades_from_oos_json(args.oos_results)

        if best_params:
            logger.info(f"Re-running backtest with best params: {best_params}")
            # Detect ORB vs EMA strategy by key presence
            if "or_end_time" in best_params:
                from validate_orb_oos import build_orb_engine, BACKTEST_DEFAULTS  # type: ignore
                params = {**BACKTEST_DEFAULTS, **best_params}
                engine = build_orb_engine(params)
            else:
                from validate_rule_based_oos import build_engine, BACKTEST_DEFAULTS  # type: ignore
                params = {**BACKTEST_DEFAULTS, **best_params}
                engine = build_engine(params)
            result = engine.run(bars, starting_equity=50_000.0)
            trades = [
                {
                    "entry_bar": t.entry_bar,
                    "exit_bar": t.exit_bar,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                }
                for t in result.trades
            ]
            logger.info(f"Re-ran backtest: {len(trades)} trades")

    if not trades:
        logger.error(
            "No trades found. Provide --trades-json or --oos-results with valid data.\n"
            "Run validate_rule_based_oos.py first."
        )
        # Create a synthetic example for smoke-test
        logger.warning("Creating synthetic trade data for demonstration...")
        np.random.seed(42)
        n_bars = len(bars)
        trades = []
        for i in range(0, min(n_bars - 30, 200), 30):
            direction = np.random.choice([1, -1])
            entry_price = float(bars["close"].iloc[i])
            exit_bar = min(i + np.random.randint(5, 25), n_bars - 1)
            exit_price = float(bars["close"].iloc[exit_bar])
            pnl = direction * (exit_price - entry_price) * MES_POINT_VALUE
            trades.append({
                "entry_bar": i,
                "exit_bar": exit_bar,
                "direction": int(direction),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": "profit_target" if pnl > 0 else "stop_loss",
            })

    # Compute MAE/MFE
    logger.info("Computing intra-trade MAE/MFE...")
    mae_mfe_df = compute_intra_trade_mae_mfe(trades, bars)
    logger.info(
        f"Analyzed {len(mae_mfe_df)} trades: "
        f"{(mae_mfe_df['outcome']=='win').sum()} wins, "
        f"{(mae_mfe_df['outcome']=='loss').sum()} losses"
    )

    # Find optimal barriers
    optimal_sl, sl_diagnostics = find_optimal_sl(mae_mfe_df, max_winner_crossing_pct=0.05)
    optimal_pt, pt_diagnostics = find_optimal_pt(mae_mfe_df)

    # Report
    print("\n" + "=" * 60)
    print("MAE/MFE EMPIRICAL BARRIER ANALYSIS")
    print("=" * 60)
    print(f"Trades analyzed: {len(mae_mfe_df)}")
    print(f"Win rate: {(mae_mfe_df['outcome']=='win').mean():.1%}")
    print()
    print(f"RECOMMENDED STOP LOSS:    {optimal_sl:.2f}x ATR")
    print(f"  (stops {sl_diagnostics.get('pct_winners_stopped_at_optimal', 0):.1%} of winners, "
          f"{sl_diagnostics.get('pct_losers_stopped_at_optimal', 0):.1%} of losers)")
    print()
    print(f"RECOMMENDED PROFIT TARGET: {optimal_pt:.2f}x ATR")
    print(f"  (sim win rate: {pt_diagnostics.get('sim_win_rate_at_optimal', 0):.1%}, "
          f"sim EV: ${pt_diagnostics.get('sim_ev_at_optimal', 0):.2f}/trade)")
    print()
    print("MAE Distribution (ATR):")
    mae_vals = mae_mfe_df["mae_atr"].dropna()
    for pct in [50, 75, 90, 95]:
        print(f"  p{pct}: {mae_vals.quantile(pct/100):.2f}x ATR")
    print("MFE Distribution (ATR):")
    mfe_vals = mae_mfe_df["mfe_atr"].dropna()
    for pct in [50, 75, 90, 95]:
        print(f"  p{pct}: {mfe_vals.quantile(pct/100):.2f}x ATR")

    # Plot
    if not args.no_plot:
        try_plot(mae_mfe_df, output_path.parent)

    # Save results
    output = {
        "recommended": {
            "pt_atr_mult": optimal_pt,
            "sl_atr_mult": optimal_sl,
        },
        "sl_analysis": sl_diagnostics,
        "pt_analysis": pt_diagnostics,
        "trade_summary": {
            "n_trades": len(mae_mfe_df),
            "win_rate": round(float((mae_mfe_df["outcome"] == "win").mean()), 4),
            "avg_mae_atr": round(float(mae_vals.mean()), 3),
            "avg_mfe_atr": round(float(mfe_vals.mean()), 3),
            "mae_p50": round(float(mae_vals.quantile(0.50)), 3),
            "mfe_p50": round(float(mfe_vals.quantile(0.50)), 3),
        },
        "best_primary_params": best_params,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
