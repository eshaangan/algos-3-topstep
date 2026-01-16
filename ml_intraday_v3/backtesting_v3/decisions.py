"""
Decision logic for offline backtest.

Includes regime-aware filtering to reduce losses during dangerous market conditions
(high volatility + downtrend).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_regime_at_events(
    events_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    vol_window: int = 20,
    trend_window: int = 50,
) -> pd.DataFrame:
    """
    Compute volatility and trend regime at each event time.

    Returns events_df with added columns:
    - vol_regime: 0 (low), 1 (medium), 2 (high)
    - trend_regime: 0 (downtrend), 1 (sideways), 2 (uptrend)
    - combined_regime: vol * 3 + trend (0-8)
    - regime_label: human-readable label
    """
    # Ensure bars are sorted by time
    bars = bars_df.copy()
    if 'timestamp' in bars.columns:
        bars = bars.set_index('timestamp').sort_index()
    elif not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars_df must have timestamp column or DatetimeIndex")

    # Compute returns
    returns = bars['close'].pct_change().fillna(0)

    # Compute rolling volatility (standard deviation of returns)
    rolling_vol = returns.rolling(window=vol_window, min_periods=1).std()

    # Compute rolling trend (price vs moving average)
    ma = bars['close'].rolling(window=trend_window, min_periods=1).mean()
    trend_strength = (bars['close'] - ma) / ma

    # Classify volatility regime using expanding quantiles (causal - no lookahead)
    vol_q33 = rolling_vol.expanding(min_periods=vol_window).quantile(0.33)
    vol_q67 = rolling_vol.expanding(min_periods=vol_window).quantile(0.67)

    vol_regime = pd.Series(1, index=bars.index)  # Default to medium
    vol_regime[rolling_vol <= vol_q33] = 0  # Low
    vol_regime[rolling_vol > vol_q67] = 2  # High

    # Classify trend regime using expanding quantiles (causal)
    trend_q33 = trend_strength.expanding(min_periods=trend_window).quantile(0.33)
    trend_q67 = trend_strength.expanding(min_periods=trend_window).quantile(0.67)

    trend_regime = pd.Series(1, index=bars.index)  # Default to sideways
    trend_regime[trend_strength <= trend_q33] = 0  # Downtrend
    trend_regime[trend_strength > trend_q67] = 2  # Uptrend

    # Combined regime
    combined_regime = vol_regime * 3 + trend_regime

    # Create regime DataFrame
    regime_df = pd.DataFrame({
        'vol_regime': vol_regime,
        'trend_regime': trend_regime,
        'combined_regime': combined_regime,
    }, index=bars.index)

    # Map events to their regime at t0
    events_out = events_df.copy()

    # Convert t0 to proper datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(events_out['t0']):
        events_out['t0'] = pd.to_datetime(events_out['t0'])

    # Use merge_asof to find regime at each event time (looking backward)
    regime_df_reset = regime_df.reset_index()
    regime_df_reset.columns = ['t0', 'vol_regime', 'trend_regime', 'combined_regime']

    events_out = events_out.sort_values('t0')
    events_out = pd.merge_asof(
        events_out,
        regime_df_reset,
        on='t0',
        direction='backward'
    )

    # Add human-readable labels
    regime_labels = {
        0: 'low_vol_downtrend',
        1: 'low_vol_sideways',
        2: 'low_vol_uptrend',
        3: 'med_vol_downtrend',
        4: 'med_vol_sideways',
        5: 'med_vol_uptrend',
        6: 'high_vol_downtrend',  # DANGEROUS
        7: 'high_vol_sideways',
        8: 'high_vol_uptrend',
    }
    events_out['regime_label'] = events_out['combined_regime'].map(regime_labels)

    return events_out


def decide_trades(
    events_df: pd.DataFrame,
    primary_preds: pd.DataFrame,
    meta_preds: pd.DataFrame | None,
    config: dict,
    bars_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Decide whether to take trades based on primary/meta predictions.

    Includes regime-aware filtering to reduce losses during dangerous market
    conditions (high volatility + downtrend).

    Returns events_df with columns:
    accept (bool), decision_reason, p_primary, p_meta, regime_label
    """
    decision_cfg = config.get("decision", {})
    use_meta = bool(decision_cfg.get("use_meta", False))
    score_col = decision_cfg.get("primary_score_column")
    primary_threshold = float(decision_cfg.get("primary_threshold", 0.5))
    meta_threshold = float(decision_cfg.get("meta_threshold", 0.5))
    require_meta = bool(decision_cfg.get("require_meta_for_trade", True))

    if score_col and score_col in primary_preds.columns:
        cols = ["event_id", score_col]
        extra = [c for c in ["p_target", "p_stop", "p_vertical"] if c in primary_preds.columns]
        if "predicted_side" in primary_preds.columns:
            extra.append("predicted_side")
        cols.extend(extra)
        preds = primary_preds[cols].rename(columns={score_col: "p_primary"})
    else:
        preds = primary_preds[["event_id", "y_prob"]].rename(
            columns={"y_prob": "p_primary"}
        )
    merged = events_df.merge(preds, on="event_id", how="left")
    missing_primary = merged["p_primary"].isna()
    if missing_primary.any():
        logger.warning(
            "Missing primary predictions for %d events; marking as skipped",
            int(missing_primary.sum()),
        )

    merged["proposed"] = merged["p_primary"] >= primary_threshold
    merged["accept"] = merged["proposed"].copy()
    merged["decision_reason"] = ""
    if missing_primary.any():
        merged.loc[missing_primary, "accept"] = False
        merged.loc[missing_primary, "decision_reason"] = "missing_primary"

    if "predicted_side" in merged.columns:
        merged["side"] = merged["predicted_side"].fillna(merged.get("side", 1))
        no_side = merged["predicted_side"] == 0
        if no_side.any():
            merged.loc[no_side, "accept"] = False
            merged.loc[no_side & (merged["decision_reason"] == ""), "decision_reason"] = "no_positive_ev"

    # Volatility filter (only trade in high-vol regimes)
    vol_filter_cfg = decision_cfg.get("volatility_filter", {})
    if vol_filter_cfg.get("enabled", False):
        min_sigma = float(vol_filter_cfg.get("min_sigma", 0.0))
        max_sigma = float(vol_filter_cfg.get("max_sigma", 0.0) or 0.0)
        if "sigma" in merged.columns:
            low_vol = merged["sigma"] < min_sigma
            if low_vol.any():
                merged.loc[low_vol & merged["accept"], "accept"] = False
                merged.loc[low_vol & (merged["decision_reason"] == ""), "decision_reason"] = "low_volatility"
                logger.info(
                    "Volatility filter: rejected %d/%d proposed trades (sigma < %.2f)",
                    int(low_vol.sum()),
                    int(merged["proposed"].sum()),
                    min_sigma,
                )

            if max_sigma > 0:
                high_vol = merged["sigma"] > max_sigma
                if high_vol.any():
                    merged.loc[high_vol & merged["accept"], "accept"] = False
                    merged.loc[high_vol & (merged["decision_reason"] == ""), "decision_reason"] = "high_volatility"
                    logger.info(
                        "Volatility filter: rejected %d/%d proposed trades (sigma > %.2f)",
                        int((high_vol & merged["proposed"]).sum()),
                        int(merged["proposed"].sum()),
                        max_sigma,
                    )
        else:
            logger.warning("Volatility filter enabled but sigma column not found in events")

    # Regime filter (reduce trades during dangerous regimes)
    regime_filter_cfg = decision_cfg.get("regime_filter", {})
    if regime_filter_cfg.get("enabled", False) and bars_df is not None:
        vol_window = int(regime_filter_cfg.get("vol_window", 20))
        trend_window = int(regime_filter_cfg.get("trend_window", 50))
        skip_regimes = regime_filter_cfg.get("skip_regimes", [6])  # Default: high_vol_downtrend
        threshold_boost_regimes = regime_filter_cfg.get("threshold_boost_regimes", [3, 7])  # med_vol_down, high_vol_sideways
        threshold_boost = float(regime_filter_cfg.get("threshold_boost", 0.10))  # Add this to threshold

        # NEW: Regime-based side selection (short in downtrends, long in uptrends)
        enable_shorts = bool(regime_filter_cfg.get("enable_shorts", False))
        downtrend_regimes = regime_filter_cfg.get("downtrend_regimes", [0, 3, 6])  # All downtrend regimes
        uptrend_regimes = regime_filter_cfg.get("uptrend_regimes", [2, 5, 8])  # All uptrend regimes

        # Compute regime for each event
        merged = compute_regime_at_events(merged, bars_df, vol_window, trend_window)

        # Regime-based side selection: Short in downtrends, Long in uptrends
        if enable_shorts:
            # Set side based on trend regime
            # trend_regime: 0=downtrend, 1=sideways, 2=uptrend
            is_downtrend = merged["trend_regime"] == 0
            is_uptrend = merged["trend_regime"] == 2
            is_sideways = merged["trend_regime"] == 1

            # Default to long
            merged["side"] = 1

            # Short in downtrends
            merged.loc[is_downtrend, "side"] = -1

            # In sideways, keep long (model predicts direction)
            # Could also skip sideways if desired

            n_shorts = (is_downtrend & merged["accept"]).sum()
            n_longs = ((is_uptrend | is_sideways) & merged["accept"]).sum()
            logger.info(
                "Regime-based sides: %d shorts (downtrend), %d longs (uptrend/sideways)",
                int(n_shorts),
                int(n_longs),
            )

            # Remove downtrend regimes from skip_regimes since we're now shorting them
            skip_regimes = [r for r in skip_regimes if r not in downtrend_regimes]

        # Skip trades in dangerous regimes entirely (only if not shorting those regimes)
        if skip_regimes:
            in_skip_regime = merged["combined_regime"].isin(skip_regimes)
            n_skip = (in_skip_regime & merged["accept"]).sum()
            if n_skip > 0:
                merged.loc[in_skip_regime & merged["accept"], "accept"] = False
                merged.loc[in_skip_regime & (merged["decision_reason"] == ""), "decision_reason"] = "regime_skip"
                logger.info(
                    "Regime filter: skipped %d trades in dangerous regimes %s",
                    int(n_skip),
                    skip_regimes,
                )

        # Require higher threshold in risky regimes
        if threshold_boost_regimes and threshold_boost > 0:
            in_boost_regime = merged["combined_regime"].isin(threshold_boost_regimes)
            boosted_threshold = primary_threshold + threshold_boost
            below_boosted = (merged["p_primary"] < boosted_threshold) & (merged["p_primary"] >= primary_threshold)
            regime_boost_reject = in_boost_regime & below_boosted & merged["accept"]
            n_boost_reject = regime_boost_reject.sum()
            if n_boost_reject > 0:
                merged.loc[regime_boost_reject, "accept"] = False
                merged.loc[regime_boost_reject & (merged["decision_reason"] == ""), "decision_reason"] = "regime_threshold_boost"
                logger.info(
                    "Regime filter: rejected %d trades in risky regimes (threshold boosted to %.2f)",
                    int(n_boost_reject),
                    boosted_threshold,
                )

        # Log regime distribution
        if merged["accept"].any():
            accepted_regimes = merged.loc[merged["accept"], "regime_label"].value_counts()
            logger.info("Accepted trades by regime:\n%s", accepted_regimes.to_string())

            # Log side distribution if shorts enabled
            if enable_shorts:
                side_counts = merged.loc[merged["accept"], "side"].value_counts()
                logger.info("Trade sides: %d longs, %d shorts",
                           side_counts.get(1, 0), side_counts.get(-1, 0))
    else:
        # Add empty regime columns if not computed
        merged["vol_regime"] = None
        merged["trend_regime"] = None
        merged["combined_regime"] = None
        merged["regime_label"] = None

    if use_meta:
        if meta_preds is None:
            raise ValueError("Meta predictions required but not provided")
        meta = meta_preds[["event_id", "p_meta"]]
        merged = merged.merge(meta, on="event_id", how="left")
        merged["p_meta"] = merged["p_meta"].astype(float)
        if merged["p_meta"].isna().any():
            merged.loc[merged["p_meta"].isna(), "decision_reason"] = "missing_meta"
            if require_meta:
                merged.loc[merged["p_meta"].isna(), "accept"] = False

        meta_accept = merged["p_meta"] >= meta_threshold
        if require_meta:
            merged["accept"] = merged["proposed"] & meta_accept
            merged.loc[~merged["proposed"], "decision_reason"] = "threshold_primary"
            merged.loc[
                merged["proposed"] & ~meta_accept, "decision_reason"
            ] = "threshold_meta"
        else:
            merged["accept"] = merged["proposed"] & (
                meta_accept | merged["p_meta"].isna()
            )
            merged.loc[~merged["proposed"], "decision_reason"] = "threshold_primary"
            merged.loc[
                merged["proposed"] & ~meta_accept & ~merged["p_meta"].isna(),
                "decision_reason",
            ] = "threshold_meta"
    else:
        merged.loc[~merged["proposed"], "decision_reason"] = "threshold_primary"

    if "p_meta" not in merged.columns:
        merged["p_meta"] = None

    return merged
