"""
Explore alternative quality metrics for predicting CATASTOP vs success.

This script analyzes various model-output-based metrics to find predictors
of trade success that are more effective than raw confidence.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("RISK_PRESET_NAME", "TOPSTEP_50K")

from core.risk_presets import get_risk_preset
from data.clean_bars import clean_bars
from features.engineer import add_features
from models.nn_inference import load_nn_bundle, predict_scores_for_bars
from backtesting.backtest import run_backtest_nn


def compute_quality_metrics(trades_df, bars_df, prob_df):
    """
    Compute alternative quality metrics for each trade.

    Metrics:
    1. score_margin: How far above threshold (score - threshold)
    2. score_percentile_day: Percentile of score within trading day
    3. directional_strength: Signed probability advantage for chosen direction
    4. ev_proxy: Simple expected value proxy
    5. prob_dominant: Probability of chosen direction
    """
    trades_enriched = trades_df.copy()

    # Merge with bars to get entry bar features
    bars_df = bars_df.copy()
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

    # For each trade, find the signal bar (entry_bar - 1)
    signal_data = []
    for idx, trade in trades_df.iterrows():
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)

        # Find the bar index for entry
        entry_idx = bars_df[bars_df["timestamp"] == entry_time].index
        if len(entry_idx) == 0:
            signal_data.append({})
            continue

        entry_idx = entry_idx[0]
        if entry_idx == 0:
            signal_data.append({})
            continue

        # Signal bar is the bar before entry
        signal_idx = entry_idx - 1
        signal_bar = bars_df.iloc[signal_idx]

        # Get probabilities for signal bar
        prob_row = prob_df.iloc[signal_idx] if signal_idx < len(prob_df) else None
        if prob_row is None:
            signal_data.append({})
            continue

        p_long = prob_row.get("long_prob", 0.5)
        p_short = prob_row.get("short_prob", 0.5)
        score = prob_row.get("score", 0.5)

        # Compute metrics
        direction = trade["direction"]
        prob_dominant = p_long if direction == "long" else p_short
        directional_strength = (p_long - p_short) if direction == "long" else (p_short - p_long)

        signal_data.append({
            "p_long": p_long,
            "p_short": p_short,
            "score": score,
            "prob_dominant": prob_dominant,
            "directional_strength": directional_strength,
            "signal_timestamp": signal_bar["timestamp"],
        })

    signal_df = pd.DataFrame(signal_data)

    # Merge with trades
    for col in signal_df.columns:
        trades_enriched[col] = signal_df[col]

    # Compute score_margin (requires knowing threshold)
    # We'll use a reference threshold from the config
    trades_enriched["score_margin"] = trades_enriched["score"] - 0.46  # Approximate threshold

    # Compute score_percentile_day (rank within trading day)
    trades_enriched["signal_date"] = pd.to_datetime(
        trades_enriched["signal_timestamp"]
    ).dt.tz_convert("America/Chicago").dt.date

    trades_enriched["score_percentile_day"] = trades_enriched.groupby("signal_date")["score"].rank(pct=True)

    # Compute EV proxy: score * win_amount - (1 - score) * loss_amount
    # Assume 2:1 RR, so win = 2x, loss = 1x (in stop ticks)
    stop_ticks = 48  # catastrophic stop
    target_ticks = stop_ticks * 2
    tick_value = 1.25

    trades_enriched["ev_proxy"] = (
        trades_enriched["prob_dominant"] * (target_ticks * tick_value)
        - (1 - trades_enriched["prob_dominant"]) * (stop_ticks * tick_value)
    )

    return trades_enriched


def analyze_metric_correlation(trades_df, metric_name, metric_col):
    """Analyze how a metric correlates with CATASTOP vs success."""

    if metric_col not in trades_df.columns or trades_df[metric_col].isna().all():
        return None

    valid_trades = trades_df[trades_df[metric_col].notna()].copy()
    if len(valid_trades) == 0:
        return None

    catastop = valid_trades[valid_trades["reason"] == "CATASTOP"]
    timeexit = valid_trades[valid_trades["reason"] == "TIME_EXIT"]

    result = {
        "metric": metric_name,
        "total_trades": len(valid_trades),
        "catastop_count": len(catastop),
        "catastop_rate": len(catastop) / len(valid_trades) * 100,
        "avg_all": valid_trades[metric_col].mean(),
        "median_all": valid_trades[metric_col].median(),
        "avg_catastop": catastop[metric_col].mean() if len(catastop) > 0 else None,
        "avg_timeexit": timeexit[metric_col].mean() if len(timeexit) > 0 else None,
        "avg_winners": valid_trades[valid_trades["pnl"] > 0][metric_col].mean(),
        "avg_losers": valid_trades[valid_trades["pnl"] <= 0][metric_col].mean(),
    }

    # Compute difference (how well does metric separate CATASTOP from TIME_EXIT)
    if result["avg_catastop"] is not None and result["avg_timeexit"] is not None:
        result["separation"] = abs(result["avg_timeexit"] - result["avg_catastop"])
        result["separation_pct"] = (
            result["separation"] / result["avg_all"] * 100 if result["avg_all"] != 0 else 0
        )

    return result


def test_filter_thresholds(trades_df, metric_col, metric_name, ascending=False):
    """Test various threshold values for a metric as a filter."""

    valid_trades = trades_df[trades_df[metric_col].notna()].copy()
    if len(valid_trades) == 0:
        return []

    # Get percentile thresholds
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    thresholds = [valid_trades[metric_col].quantile(p / 100.0) for p in percentiles]

    results = []
    for p, thresh in zip(percentiles, thresholds):
        if ascending:
            # Keep trades with metric >= threshold (higher is better)
            filtered = valid_trades[valid_trades[metric_col] >= thresh]
        else:
            # Keep trades with metric <= threshold (lower is better)
            filtered = valid_trades[valid_trades[metric_col] <= thresh]

        if len(filtered) == 0:
            continue

        catastop = filtered[filtered["reason"] == "CATASTOP"]
        winners = filtered[filtered["pnl"] > 0]
        losers = filtered[filtered["pnl"] <= 0]

        catastop_rate = len(catastop) / len(filtered) * 100
        win_rate = len(winners) / len(filtered) * 100
        avg_pnl = filtered["pnl"].mean()

        gross_wins = winners["pnl"].sum() if len(winners) > 0 else 0
        gross_losses = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        results.append({
            "percentile": p,
            "threshold": thresh,
            "trades": len(filtered),
            "catastop_rate": catastop_rate,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "profit_factor": pf,
            "meets_target": "✓" if catastop_rate < 10.0 else "",
        })

    return results


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print("=" * 100)
    print("ALTERNATIVE QUALITY METRICS EXPLORATION")
    print("=" * 100)

    # Load data
    print(f"\nLoading data from {data_path}...")
    with pd.HDFStore(data_path, "r") as store:
        bars = store[dataset_key].copy()

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    preset = get_risk_preset("TOPSTEP_50K")
    bars = clean_bars(bars, tick_size=preset.risk_config.tick_size, verbose=False)
    bars = bars.tail(fast_max_bars).reset_index(drop=True)

    print(f"Loaded {len(bars):,} bars")

    # Add features for volatility analysis
    print("\nComputing features...")
    bars_with_features = add_features(bars, verbose=False)

    # Load model
    print(f"\nLoading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)
    nn_cfg = bundle.config.get("nn_config", {})

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run backtest with NO confidence filter
    print("\nRunning backtest (confidence_min=0.0)...")
    risk_cfg = preset.risk_config

    results = run_backtest_nn(
        bars,
        prob_df,
        score_threshold=float(nn_cfg["score_threshold"]),
        selection_mode=str(nn_cfg.get("selection_mode", "global_threshold")),
        day_percentile_floor=float(nn_cfg.get("day_percentile_floor", 0.90)),
        global_floor_score=float(nn_cfg.get("global_floor_score", nn_cfg["score_threshold"])),
        max_trades_per_day=int(nn_cfg["max_trades_per_day"]),
        min_bars_between_trades=int(nn_cfg["min_bars_between_trades"]),
        enable_long=bool(nn_cfg["enable_long"]),
        enable_short=bool(nn_cfg["enable_short"]),
        horizon_bars=int(nn_cfg["horizon_bars"]),
        execution_mode=str(nn_cfg["execution_mode"]),
        exit_price_mode=str(nn_cfg["exit_price_mode"]),
        session_mode=str(nn_cfg["session_mode"]),
        deadline_time=nn_cfg.get("deadline_time"),
        deadline_relax_factor=float(nn_cfg.get("deadline_relax_factor", 0.98)),
        bar_minutes=int(nn_cfg["bar_minutes"]),
        session_start=risk_cfg.session_start,
        session_end=risk_cfg.session_end,
        stop_loss_ticks=int(nn_cfg["stop_loss_ticks"]),
        target_multiplier=float(nn_cfg["target_multiplier"]),
        catastrophic_stop_ticks=int(nn_cfg.get("catastrophic_stop_ticks", int(nn_cfg["threshold_ticks"]) * 4)),
        max_hold_bars=int(nn_cfg["max_hold_bars"]),
        tick_size=float(nn_cfg["tick_size"]),
        tick_value=float(nn_cfg["tick_value"]),
        save_trades_path=None,
    )

    trades_df = pd.DataFrame(results["trades"])
    print(f"Generated {len(trades_df)} trades")

    # Compute quality metrics
    print("\nComputing quality metrics...")
    trades_enriched = compute_quality_metrics(trades_df, bars, prob_df)

    # Add volatility features at entry
    print("Enriching with volatility features...")
    trades_with_vol = []
    for idx, trade in trades_enriched.iterrows():
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)
        entry_idx = bars_with_features[bars_with_features["timestamp"] == entry_time].index

        if len(entry_idx) == 0 or entry_idx[0] == 0:
            vol_data = {}
        else:
            signal_idx = entry_idx[0] - 1
            signal_bar = bars_with_features.iloc[signal_idx]

            vol_data = {
                "atr_ticks": signal_bar.get("atr_ticks", None),
                "vol_20": signal_bar.get("vol_20", None),
                "vol_50": signal_bar.get("vol_50", None),
                "vol_percentile": signal_bar.get("vol_percentile", None),
                "bb_width": signal_bar.get("bb_width", None),
            }

        trade_dict = trade.to_dict()
        trade_dict.update(vol_data)
        trades_with_vol.append(trade_dict)

    trades_enriched = pd.DataFrame(trades_with_vol)

    # Analyze correlations
    print("\n" + "=" * 100)
    print("METRIC CORRELATION ANALYSIS")
    print("=" * 100)
    print("\nHow well does each metric separate CATASTOP from TIME_EXIT?\n")

    metrics_to_test = [
        ("confidence", "confidence"),
        ("score", "score"),
        ("score_margin", "score_margin"),
        ("score_percentile_day", "score_percentile_day"),
        ("prob_dominant", "prob_dominant"),
        ("directional_strength", "directional_strength"),
        ("ev_proxy", "ev_proxy"),
    ]

    correlation_results = []
    for metric_name, metric_col in metrics_to_test:
        result = analyze_metric_correlation(trades_enriched, metric_name, metric_col)
        if result:
            correlation_results.append(result)

    # Print correlation table
    print(f"{'Metric':<25} {'Trades':<8} {'Avg(All)':<12} {'Avg(CATASTOP)':<15} {'Avg(TIME_EXIT)':<15} {'Separation':<12}")
    print("-" * 100)
    for res in correlation_results:
        sep_str = f"{res.get('separation', 0):.4f}" if res.get('separation') is not None else "N/A"
        print(
            f"{res['metric']:<25} "
            f"{res['total_trades']:<8} "
            f"{res['avg_all']:<12.4f} "
            f"{res.get('avg_catastop', 0):<15.4f} "
            f"{res.get('avg_timeexit', 0):<15.4f} "
            f"{sep_str:<12}"
        )

    # Test filters
    print("\n" + "=" * 100)
    print("FILTER THRESHOLD ANALYSIS")
    print("=" * 100)

    filter_tests = [
        ("score", "score", True),  # Higher score is better
        ("score_margin", "score_margin", True),
        ("score_percentile_day", "score_percentile_day", True),
        ("prob_dominant", "prob_dominant", True),
        ("directional_strength", "directional_strength", True),
        ("ev_proxy", "ev_proxy", True),
    ]

    for metric_name, metric_col, ascending in filter_tests:
        print(f"\n{metric_name.upper()} Filter (keep trades with {'higher' if ascending else 'lower'} values):")
        print("-" * 100)

        filter_results = test_filter_thresholds(trades_enriched, metric_col, metric_name, ascending)

        if filter_results:
            print(f"{'Pctl':<6} {'Threshold':<12} {'Trades':<8} {'CATASTOP%':<12} {'WinRate%':<12} {'AvgPnL':<12} {'PF':<8} {'Target':<8}")
            print("-" * 100)
            for fr in filter_results:
                pf_str = f"{fr['profit_factor']:.2f}" if fr['profit_factor'] != float("inf") else "inf"
                print(
                    f"p{fr['percentile']:<4} "
                    f"{fr['threshold']:<12.4f} "
                    f"{fr['trades']:<8} "
                    f"{fr['catastop_rate']:<12.1f} "
                    f"{fr['win_rate']:<12.1f} "
                    f"${fr['avg_pnl']:<11.2f} "
                    f"{pf_str:<8} "
                    f"{fr['meets_target']:<8}"
                )

    # Volatility regime analysis
    print("\n" + "=" * 100)
    print("VOLATILITY REGIME ANALYSIS")
    print("=" * 100)

    vol_metrics = ["atr_ticks", "vol_20", "vol_percentile", "bb_width"]

    for vol_metric in vol_metrics:
        if vol_metric not in trades_enriched.columns:
            continue

        valid = trades_enriched[trades_enriched[vol_metric].notna()].copy()
        if len(valid) == 0:
            continue

        print(f"\n{vol_metric.upper()} Correlation:")
        result = analyze_metric_correlation(valid, vol_metric, vol_metric)
        if result:
            print(f"  Avg (all): {result['avg_all']:.4f}")
            print(f"  Avg (CATASTOP): {result.get('avg_catastop', 0):.4f}")
            print(f"  Avg (TIME_EXIT): {result.get('avg_timeexit', 0):.4f}")
            print(f"  Separation: {result.get('separation', 0):.4f}")

        # Test as filter (keep low-volatility trades)
        print(f"\n{vol_metric.upper()} Filter (keep LOW volatility):")
        filter_results = test_filter_thresholds(valid, vol_metric, vol_metric, ascending=False)

        if filter_results:
            print(f"  {'Pctl':<6} {'Threshold':<12} {'Trades':<8} {'CATASTOP%':<12} {'AvgPnL':<12} {'PF':<8}")
            for fr in filter_results[:5]:  # Show first 5
                pf_str = f"{fr['profit_factor']:.2f}" if fr['profit_factor'] != float("inf") else "inf"
                print(
                    f"  p{fr['percentile']:<4} "
                    f"{fr['threshold']:<12.4f} "
                    f"{fr['trades']:<8} "
                    f"{fr['catastop_rate']:<12.1f} "
                    f"${fr['avg_pnl']:<11.2f} "
                    f"{pf_str:<8}"
                )

    # Summary recommendations
    print("\n" + "=" * 100)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 100)

    # Find best performing filters
    best_filters = []
    for metric_name, metric_col, ascending in filter_tests:
        filter_results = test_filter_thresholds(trades_enriched, metric_col, metric_name, ascending)
        for fr in filter_results:
            if fr["catastop_rate"] < 10.0 and fr["trades"] >= 100:  # Meets target with reasonable trade count
                best_filters.append({
                    "metric": metric_name,
                    "percentile": fr["percentile"],
                    "threshold": fr["threshold"],
                    "trades": fr["trades"],
                    "catastop_rate": fr["catastop_rate"],
                    "profit_factor": fr["profit_factor"],
                    "avg_pnl": fr["avg_pnl"],
                })

    if best_filters:
        print("\nFilters that achieve <10% CATASTOP with >=100 trades:")
        print(f"{'Metric':<25} {'Pctl':<6} {'Trades':<8} {'CATASTOP%':<12} {'AvgPnL':<12} {'PF':<8}")
        print("-" * 100)
        for bf in best_filters:
            pf_str = f"{bf['profit_factor']:.2f}" if bf['profit_factor'] != float("inf") else "inf"
            print(
                f"{bf['metric']:<25} "
                f"p{bf['percentile']:<5} "
                f"{bf['trades']:<8} "
                f"{bf['catastop_rate']:<12.1f} "
                f"${bf['avg_pnl']:<11.2f} "
                f"{pf_str:<8}"
            )
    else:
        print("\nNo filters achieved <10% CATASTOP target with >=100 trades.")
        print("Consider combining multiple filters or accepting higher CATASTOP rate.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
