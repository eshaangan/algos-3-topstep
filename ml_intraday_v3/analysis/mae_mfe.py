"""
Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) Analysis.

MAE: Worst intraday loss from entry before exit (how far against you did the trade go?)
MFE: Best intraday profit from entry before exit (how far in your favor did the trade go?)

These metrics help optimize stop-loss and profit target placement:
- If MAE avg is 0.3% but stops are at 0.5%: stops are too wide (giving up too much)
- If MFE avg is 0.8% but targets are at 0.4%: targets are too tight (leaving money on table)
- MFE/MAE ratio indicates strategy quality (should be >1.5 for profitability)

Reference: "Understanding MAE and MFE Metrics: A Guide for Traders"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mae_mfe(
    trades_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    contract_multiplier: float = 5.0,  # MES default
) -> pd.DataFrame:
    """
    Compute MAE and MFE for each executed trade.

    Args:
        trades_df: DataFrame with columns: event_id, entry_ts, exit_ts, entry_px,
                   executed, pnl_usd, direction (if present)
        bars_df: OHLCV bars indexed by timestamp
        contract_multiplier: USD per point (5.0 for MES, 50.0 for ES)

    Returns:
        DataFrame with columns:
        - event_id
        - mae: Maximum adverse excursion in USD
        - mfe: Maximum favorable excursion in USD
        - mae_mfe_ratio: MFE / MAE
        - realized_pnl: Actual trade PnL
        - efficiency: realized_pnl / MFE (what % of best-case was captured)
        - mae_pct: MAE as % of entry price
        - mfe_pct: MFE as % of entry price
        - bars_to_mae: Number of bars until MAE hit
        - bars_to_mfe: Number of bars until MFE hit
    """
    logger.info(f"Computing MAE/MFE for {len(trades_df)} trades")

    # Filter to executed trades only
    executed_trades = trades_df[trades_df["executed"] == True].copy()
    logger.info(f"  {len(executed_trades)} executed trades to analyze")

    if len(executed_trades) == 0:
        logger.warning("No executed trades to analyze")
        return pd.DataFrame()

    results = []

    for idx, trade in executed_trades.iterrows():
        event_id = trade["event_id"]
        entry_ts = trade["entry_ts"]
        exit_ts = trade["exit_ts"]
        entry_px = trade["entry_px"]
        realized_pnl = trade["pnl_usd"]

        # Infer direction from entry_px and exit_px if not provided
        if "exit_px" in trade.index and pd.notna(trade["exit_px"]):
            exit_px = trade["exit_px"]
            # If we made money, we went the right direction
            if realized_pnl > 0:
                direction = "long" if exit_px > entry_px else "short"
            else:
                direction = "short" if exit_px > entry_px else "long"
        else:
            # Default to long if we can't infer
            direction = "long"

        # Get bars between entry and exit (inclusive)
        try:
            trade_bars = bars_df.loc[entry_ts:exit_ts]
        except Exception as e:
            logger.warning(f"  Could not get bars for trade {event_id}: {e}")
            continue

        if len(trade_bars) == 0:
            logger.warning(f"  No bars found for trade {event_id} ({entry_ts} to {exit_ts})")
            continue

        # Compute MAE and MFE
        if direction == "long":
            # For longs: MAE is how far below entry we went, MFE is how far above
            drawdowns = entry_px - trade_bars["low"]
            runups = trade_bars["high"] - entry_px
        else:  # short
            # For shorts: MAE is how far above entry we went, MFE is how far below
            drawdowns = trade_bars["high"] - entry_px
            runups = entry_px - trade_bars["low"]

        # Convert to USD
        mae_points = drawdowns.max()
        mfe_points = runups.max()
        mae_usd = mae_points * contract_multiplier
        mfe_usd = mfe_points * contract_multiplier

        # Find when MAE and MFE occurred
        bars_to_mae = drawdowns.idxmax()
        bars_to_mfe = runups.idxmax()

        # Compute bars as integer index difference
        try:
            bars_to_mae_idx = trade_bars.index.get_loc(bars_to_mae)
            bars_to_mfe_idx = trade_bars.index.get_loc(bars_to_mfe)
        except:
            bars_to_mae_idx = 0
            bars_to_mfe_idx = 0

        results.append({
            "event_id": event_id,
            "mae": mae_usd,
            "mfe": mfe_usd,
            "mae_mfe_ratio": mfe_usd / mae_usd if mae_usd > 0 else np.inf,
            "realized_pnl": realized_pnl,
            "efficiency": realized_pnl / mfe_usd if mfe_usd > 0 else 0.0,
            "mae_pct": (mae_points / entry_px) * 100 if entry_px > 0 else 0.0,
            "mfe_pct": (mfe_points / entry_px) * 100 if entry_px > 0 else 0.0,
            "bars_to_mae": bars_to_mae_idx,
            "bars_to_mfe": bars_to_mfe_idx,
            "direction": direction,
            "entry_px": entry_px,
        })

    result_df = pd.DataFrame(results)
    logger.info(f"  Computed MAE/MFE for {len(result_df)} trades")

    return result_df


def analyze_mae_mfe_distributions(mae_mfe_df: pd.DataFrame) -> dict:
    """
    Analyze MAE/MFE distributions and generate recommendations.

    Returns:
        Dictionary with:
        - mae_stats: percentiles and statistics for MAE
        - mfe_stats: percentiles and statistics for MFE
        - efficiency_stats: how well we capture available profit
        - mae_mfe_ratio_stats: quality metric
        - recommendations: suggested barrier adjustments
    """
    if len(mae_mfe_df) == 0:
        logger.warning("No MAE/MFE data to analyze")
        return {}

    logger.info("Analyzing MAE/MFE distributions")

    # MAE statistics
    mae_stats = {
        "p05": float(mae_mfe_df["mae"].quantile(0.05)),
        "p25": float(mae_mfe_df["mae"].quantile(0.25)),
        "p50": float(mae_mfe_df["mae"].quantile(0.50)),
        "p75": float(mae_mfe_df["mae"].quantile(0.75)),
        "p95": float(mae_mfe_df["mae"].quantile(0.95)),
        "mean": float(mae_mfe_df["mae"].mean()),
        "std": float(mae_mfe_df["mae"].std()),
    }

    # MFE statistics
    mfe_stats = {
        "p05": float(mae_mfe_df["mfe"].quantile(0.05)),
        "p25": float(mae_mfe_df["mfe"].quantile(0.25)),
        "p50": float(mae_mfe_df["mfe"].quantile(0.50)),
        "p75": float(mae_mfe_df["mfe"].quantile(0.75)),
        "p95": float(mae_mfe_df["mfe"].quantile(0.95)),
        "mean": float(mae_mfe_df["mfe"].mean()),
        "std": float(mae_mfe_df["mfe"].std()),
    }

    # Percentage-based stats (more useful for barrier sizing)
    mae_pct_stats = {
        "p05": float(mae_mfe_df["mae_pct"].quantile(0.05)),
        "p25": float(mae_mfe_df["mae_pct"].quantile(0.25)),
        "p50": float(mae_mfe_df["mae_pct"].quantile(0.50)),
        "p75": float(mae_mfe_df["mae_pct"].quantile(0.75)),
        "p95": float(mae_mfe_df["mae_pct"].quantile(0.95)),
        "mean": float(mae_mfe_df["mae_pct"].mean()),
    }

    mfe_pct_stats = {
        "p05": float(mae_mfe_df["mfe_pct"].quantile(0.05)),
        "p25": float(mae_mfe_df["mfe_pct"].quantile(0.25)),
        "p50": float(mae_mfe_df["mfe_pct"].quantile(0.50)),
        "p75": float(mae_mfe_df["mfe_pct"].quantile(0.75)),
        "p95": float(mae_mfe_df["mfe_pct"].quantile(0.95)),
        "mean": float(mae_mfe_df["mfe_pct"].mean()),
    }

    # Efficiency: what % of MFE was captured as realized PnL
    efficiency_stats = {
        "p05": float(mae_mfe_df["efficiency"].quantile(0.05)),
        "p25": float(mae_mfe_df["efficiency"].quantile(0.25)),
        "p50": float(mae_mfe_df["efficiency"].quantile(0.50)),
        "p75": float(mae_mfe_df["efficiency"].quantile(0.75)),
        "p95": float(mae_mfe_df["efficiency"].quantile(0.95)),
        "mean": float(mae_mfe_df["efficiency"].mean()),
    }

    # MAE/MFE ratio: strategy quality indicator
    # Filter out infinite values for ratio calculation
    valid_ratios = mae_mfe_df[mae_mfe_df["mae_mfe_ratio"] != np.inf]["mae_mfe_ratio"]
    mae_mfe_ratio_stats = {
        "p05": float(valid_ratios.quantile(0.05)) if len(valid_ratios) > 0 else 0.0,
        "p25": float(valid_ratios.quantile(0.25)) if len(valid_ratios) > 0 else 0.0,
        "p50": float(valid_ratios.quantile(0.50)) if len(valid_ratios) > 0 else 0.0,
        "p75": float(valid_ratios.quantile(0.75)) if len(valid_ratios) > 0 else 0.0,
        "p95": float(valid_ratios.quantile(0.95)) if len(valid_ratios) > 0 else 0.0,
        "mean": float(valid_ratios.mean()) if len(valid_ratios) > 0 else 0.0,
    }

    # Bars to MAE/MFE (timing analysis)
    bars_to_mae_mean = float(mae_mfe_df["bars_to_mae"].mean())
    bars_to_mfe_mean = float(mae_mfe_df["bars_to_mfe"].mean())

    # Generate recommendations
    recommendations = generate_barrier_recommendations(
        mae_pct_stats, mfe_pct_stats, efficiency_stats, mae_mfe_ratio_stats
    )

    return {
        "mae_usd": mae_stats,
        "mfe_usd": mfe_stats,
        "mae_pct": mae_pct_stats,
        "mfe_pct": mfe_pct_stats,
        "efficiency": efficiency_stats,
        "mae_mfe_ratio": mae_mfe_ratio_stats,
        "bars_to_mae_mean": bars_to_mae_mean,
        "bars_to_mfe_mean": bars_to_mfe_mean,
        "n_trades": len(mae_mfe_df),
        "recommendations": recommendations,
    }


def generate_barrier_recommendations(
    mae_pct_stats: dict,
    mfe_pct_stats: dict,
    efficiency_stats: dict,
    mae_mfe_ratio_stats: dict,
) -> dict:
    """
    Generate barrier sizing recommendations based on MAE/MFE analysis.

    Logic:
    - Stop-loss should be placed beyond MAE p75 to avoid getting stopped out by normal noise
    - Profit target should be placed at MFE p50 (conservative) or p75 (aggressive)
    - Efficiency <50% suggests exits are too early (need trailing stops or wider targets)
    """
    mae_p50 = mae_pct_stats["p50"]
    mae_p75 = mae_pct_stats["p75"]
    mae_p95 = mae_pct_stats["p95"]

    mfe_p50 = mfe_pct_stats["p50"]
    mfe_p75 = mfe_pct_stats["p75"]
    mfe_p95 = mfe_pct_stats["p95"]

    efficiency_mean = efficiency_stats["mean"]
    mae_mfe_ratio_mean = mae_mfe_ratio_stats["mean"]

    # Recommended stop-loss: MAE p75 + 10-20% buffer
    # (avoid getting stopped out by normal adverse excursion)
    recommended_sl_pct = mae_p75 * 1.15  # 15% buffer

    # Recommended profit target: MFE p50 (conservative) to p75 (aggressive)
    # If efficiency is low, use wider targets (p75); if high, p50 is fine
    if efficiency_mean < 0.5:
        recommended_pt_pct = mfe_p75  # Wider targets to capture more profit
        target_rationale = "p75 (efficiency low, need wider targets)"
    else:
        recommended_pt_pct = mfe_p50  # Conservative targets are working
        target_rationale = "p50 (efficiency acceptable)"

    # Quality assessment
    if mae_mfe_ratio_mean > 2.0:
        quality = "Excellent (MFE >> MAE, strong edge)"
    elif mae_mfe_ratio_mean > 1.5:
        quality = "Good (MFE > MAE, positive expectancy)"
    elif mae_mfe_ratio_mean > 1.0:
        quality = "Fair (MFE slightly > MAE, marginal edge)"
    else:
        quality = "Poor (MAE >= MFE, no edge or wrong direction)"

    return {
        "recommended_sl_pct": recommended_sl_pct,
        "recommended_pt_pct": recommended_pt_pct,
        "pt_rationale": target_rationale,
        "quality_assessment": quality,
        "mae_mfe_ratio": mae_mfe_ratio_mean,
        "efficiency": efficiency_mean,
        "interpretation": {
            "mae_p75": f"{mae_p75:.3f}% - 75% of trades had adverse excursion below this",
            "mfe_p50": f"{mfe_p50:.3f}% - 50% of trades reached this favorable excursion",
            "mfe_p75": f"{mfe_p75:.3f}% - 25% of trades exceeded this upside",
            "efficiency": f"{efficiency_mean:.1%} - captured this much of available profit",
        },
    }


def compare_to_current_barriers(
    recommendations: dict,
    current_pt_mult: float,
    current_sl_mult: float,
    current_atr_pct: float = 0.25,  # Typical ATR for MES 5m bars as % of price
) -> dict:
    """
    Compare recommended barriers to current configuration.

    Args:
        recommendations: Output from generate_barrier_recommendations
        current_pt_mult: Current profit target multiplier (e.g., 1.0)
        current_sl_mult: Current stop-loss multiplier (e.g., 1.5)
        current_atr_pct: Current ATR as % of price (e.g., 0.25% for MES)

    Returns:
        Comparison dict with current vs recommended
    """
    current_pt_pct = current_pt_mult * current_atr_pct
    current_sl_pct = current_sl_mult * current_atr_pct

    rec_pt_pct = recommendations["recommended_pt_pct"]
    rec_sl_pct = recommendations["recommended_sl_pct"]

    # Calculate new multipliers
    rec_pt_mult = rec_pt_pct / current_atr_pct if current_atr_pct > 0 else current_pt_mult
    rec_sl_mult = rec_sl_pct / current_atr_pct if current_atr_pct > 0 else current_sl_mult

    return {
        "current": {
            "pt_mult": current_pt_mult,
            "sl_mult": current_sl_mult,
            "pt_pct": current_pt_pct,
            "sl_pct": current_sl_pct,
        },
        "recommended": {
            "pt_mult": rec_pt_mult,
            "sl_mult": rec_sl_mult,
            "pt_pct": rec_pt_pct,
            "sl_pct": rec_sl_pct,
        },
        "changes": {
            "pt_mult_change": rec_pt_mult - current_pt_mult,
            "sl_mult_change": rec_sl_mult - current_sl_mult,
            "pt_mult_change_pct": ((rec_pt_mult / current_pt_mult) - 1) * 100 if current_pt_mult > 0 else 0,
            "sl_mult_change_pct": ((rec_sl_mult / current_sl_mult) - 1) * 100 if current_sl_mult > 0 else 0,
        },
    }


def generate_mae_mfe_report(
    mae_mfe_df: pd.DataFrame,
    analysis: dict,
    comparison: dict,
    output_path: Path,
) -> None:
    """
    Generate human-readable MAE/MFE analysis report.
    """
    logger.info(f"Writing MAE/MFE report to {output_path}")

    with open(output_path, "w") as f:
        f.write("# MAE/MFE Analysis Report\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Trades Analyzed:** {analysis['n_trades']}\n\n")

        f.write("---\n\n")
        f.write("## What Are MAE and MFE?\n\n")
        f.write("- **MAE (Maximum Adverse Excursion):** The worst drawdown from entry before exit\n")
        f.write("- **MFE (Maximum Favorable Excursion):** The best profit from entry before exit\n\n")
        f.write("**Why This Matters:**\n")
        f.write("- MAE tells us if stops are too tight (getting stopped out by noise) or too wide (giving up too much)\n")
        f.write("- MFE tells us if targets are too tight (exiting too early) or if we're capturing available profit\n")
        f.write("- MFE/MAE ratio indicates strategy quality (>1.5 is good, >2.0 is excellent)\n\n")

        f.write("---\n\n")
        f.write("## MAE Distribution (Stop-Loss Optimization)\n\n")
        mae_pct = analysis["mae_pct"]
        f.write("| Percentile | MAE (%) | Interpretation |\n")
        f.write("|------------|---------|----------------|\n")
        f.write(f"| p05 | {mae_pct['p05']:.3f}% | 95% of trades went at least this far against you |\n")
        f.write(f"| p25 | {mae_pct['p25']:.3f}% | 75% of trades reached this adverse excursion |\n")
        f.write(f"| **p50 (median)** | **{mae_pct['p50']:.3f}%** | **Typical adverse excursion** |\n")
        f.write(f"| p75 | {mae_pct['p75']:.3f}% | 25% of trades exceeded this drawdown |\n")
        f.write(f"| p95 | {mae_pct['p95']:.3f}% | Worst 5% of trades had larger adverse moves |\n\n")

        f.write(f"**Average MAE:** {mae_pct['mean']:.3f}%\n\n")
        f.write(f"**USD Equivalent (MES):** Median MAE = ${analysis['mae_usd']['p50']:.2f}\n\n")

        f.write("---\n\n")
        f.write("## MFE Distribution (Profit Target Optimization)\n\n")
        mfe_pct = analysis["mfe_pct"]
        f.write("| Percentile | MFE (%) | Interpretation |\n")
        f.write("|------------|---------|----------------|\n")
        f.write(f"| p05 | {mfe_pct['p05']:.3f}% | Even worst trades reached this profit |\n")
        f.write(f"| p25 | {mfe_pct['p25']:.3f}% | 75% of trades reached this profit |\n")
        f.write(f"| **p50 (median)** | **{mfe_pct['p50']:.3f}%** | **Typical best-case profit** |\n")
        f.write(f"| p75 | {mfe_pct['p75']:.3f}% | 25% of trades exceeded this profit |\n")
        f.write(f"| p95 | {mfe_pct['p95']:.3f}% | Best 5% of trades went this far in your favor |\n\n")

        f.write(f"**Average MFE:** {mfe_pct['mean']:.3f}%\n\n")
        f.write(f"**USD Equivalent (MES):** Median MFE = ${analysis['mfe_usd']['p50']:.2f}\n\n")

        f.write("---\n\n")
        f.write("## Strategy Quality Metrics\n\n")

        f.write("### MFE/MAE Ratio\n")
        ratio = analysis["mae_mfe_ratio"]
        f.write(f"**Mean Ratio:** {ratio['mean']:.2f}\n")
        f.write(f"**Median Ratio:** {ratio['p50']:.2f}\n\n")

        rec = analysis["recommendations"]
        f.write(f"**Quality Assessment:** {rec['quality_assessment']}\n\n")

        if ratio['mean'] > 2.0:
            f.write("✅ **Excellent!** Your favorable excursions are more than 2x your adverse excursions.\n")
        elif ratio['mean'] > 1.5:
            f.write("✅ **Good.** Positive expectancy is clear.\n")
        elif ratio['mean'] > 1.0:
            f.write("⚠️  **Fair.** Marginal edge; improvements needed.\n")
        else:
            f.write("❌ **Poor.** Strategy may be directionally wrong or poorly timed.\n")
        f.write("\n")

        f.write("### Exit Efficiency\n")
        eff = analysis["efficiency"]
        f.write(f"**Mean Efficiency:** {eff['mean']:.1%}\n")
        f.write(f"**Median Efficiency:** {eff['p50']:.1%}\n\n")

        f.write("**Interpretation:** This is the % of available profit (MFE) that you actually captured.\n\n")

        if eff['mean'] < 0.3:
            f.write("❌ **Very Low.** Exiting far too early. Consider trailing stops or wider targets.\n")
        elif eff['mean'] < 0.5:
            f.write("⚠️  **Low.** Leaving significant money on the table. Optimize exits.\n")
        elif eff['mean'] < 0.7:
            f.write("✅ **Acceptable.** Reasonable balance between safety and profit capture.\n")
        else:
            f.write("✅ **Excellent.** Capturing most of the available move.\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Barrier Placement Recommendations\n\n")

        f.write("### Current Configuration\n")
        curr = comparison["current"]
        f.write(f"- **Profit Target Multiplier:** {curr['pt_mult']:.2f}x ATR = {curr['pt_pct']:.3f}%\n")
        f.write(f"- **Stop-Loss Multiplier:** {curr['sl_mult']:.2f}x ATR = {curr['sl_pct']:.3f}%\n\n")

        f.write("### Recommended Configuration (MAE/MFE Optimized)\n")
        reco = comparison["recommended"]
        f.write(f"- **Profit Target Multiplier:** {reco['pt_mult']:.2f}x ATR = {reco['pt_pct']:.3f}%\n")
        f.write(f"  - Rationale: {rec['pt_rationale']}\n\n")
        f.write(f"- **Stop-Loss Multiplier:** {reco['sl_mult']:.2f}x ATR = {reco['sl_pct']:.3f}%\n")
        f.write(f"  - Rationale: MAE p75 + 15% buffer to avoid noise stops\n\n")

        f.write("### Changes Required\n")
        changes = comparison["changes"]
        f.write(f"- **PT Multiplier:** {curr['pt_mult']:.2f} → {reco['pt_mult']:.2f} ")
        f.write(f"({changes['pt_mult_change']:+.2f}, {changes['pt_mult_change_pct']:+.1f}%)\n")
        f.write(f"- **SL Multiplier:** {curr['sl_mult']:.2f} → {reco['sl_mult']:.2f} ")
        f.write(f"({changes['sl_mult_change']:+.2f}, {changes['sl_mult_change_pct']:+.1f}%)\n\n")

        f.write("---\n\n")
        f.write("## Implementation\n\n")
        f.write("**To apply these recommendations:**\n\n")
        f.write("1. Update `ml_intraday_v3/configs/labeling.yaml`:\n")
        f.write("```yaml\n")
        f.write("triple_barrier:\n")
        f.write(f"  pt_mult: {reco['pt_mult']:.3f}  # Was {curr['pt_mult']:.3f}\n")
        f.write(f"  sl_mult: {reco['sl_mult']:.3f}  # Was {curr['sl_mult']:.3f}\n")
        f.write("```\n\n")

        f.write("2. Rebuild labels with optimized barriers:\n")
        f.write("```bash\n")
        f.write("python ml_intraday_v3/cli.py build-labels \\\n")
        f.write("  --run-dir runs/improved_v3_001 \\\n")
        f.write("  --bar-size 5m\n")
        f.write("```\n\n")

        f.write("3. Retrain models with new labels\n\n")
        f.write("4. Re-run backtest to validate improvement\n\n")

        f.write("**Expected Impact:**\n")
        if changes['sl_mult_change'] < 0:
            f.write(f"- Tighter stops ({changes['sl_mult_change_pct']:.1f}%) → fewer stopped out, better win rate\n")
        else:
            f.write(f"- Wider stops ({changes['sl_mult_change_pct']:.1f}%) → avoid getting stopped by noise\n")

        if changes['pt_mult_change'] > 0:
            f.write(f"- Wider targets ({changes['pt_mult_change_pct']:.1f}%) → capture more profit per winner\n")
        else:
            f.write(f"- Tighter targets ({changes['pt_mult_change_pct']:.1f}%) → lock in profit faster\n")

        f.write("\n---\n\n")
        f.write("## Additional Insights\n\n")
        f.write(f"**Average Bars to MAE:** {analysis['bars_to_mae_mean']:.1f} bars\n")
        f.write(f"**Average Bars to MFE:** {analysis['bars_to_mfe_mean']:.1f} bars\n\n")

        if analysis['bars_to_mae_mean'] < analysis['bars_to_mfe_mean']:
            f.write("✅ Trades typically move in your favor before hitting adverse levels (good timing).\n")
        else:
            f.write("⚠️  Trades typically move against you first (may indicate poor entry timing).\n")

        f.write("\n---\n\n")
        f.write("*Generated by ml_intraday_v3/analysis/mae_mfe.py*\n")

    logger.info(f"Report written to {output_path}")
