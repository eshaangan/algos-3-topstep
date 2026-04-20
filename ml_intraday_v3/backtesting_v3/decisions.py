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


def _overlay_side_mask(side_series: pd.Series, side_name: str) -> pd.Series:
    if side_name == "long":
        return side_series.fillna(1).astype(float) >= 0
    if side_name == "short":
        return side_series.fillna(1).astype(float) < 0
    return pd.Series(True, index=side_series.index)


def _normalize_regime_overlay_rules(regime_filter_cfg: dict) -> list[dict]:
    rules = []

    explicit_rules = regime_filter_cfg.get("overlay_rules", []) or []
    for idx, rule in enumerate(explicit_rules):
        if not rule:
            continue
        normalized = dict(rule)
        normalized["name"] = normalized.get("name", f"overlay_rule_{idx}")
        normalized["regimes"] = [int(r) for r in (normalized.get("regimes", []) or [])]
        normalized["side"] = str(normalized.get("side", "both")).lower()
        normalized["threshold_boost"] = float(normalized.get("threshold_boost", 0.0) or 0.0)
        normalized["block"] = bool(normalized.get("block", False))
        normalized["reason"] = normalized.get(
            "reason",
            "regime_overlay_block" if normalized["block"] else "regime_threshold_boost",
        )
        rules.append(normalized)

    legacy_regimes = regime_filter_cfg.get("threshold_boost_regimes", []) or []
    legacy_boost = float(regime_filter_cfg.get("threshold_boost", 0.0) or 0.0)
    if legacy_regimes and legacy_boost > 0.0:
        rules.append(
            {
                "name": "legacy_threshold_boost",
                "regimes": [int(r) for r in legacy_regimes],
                "side": "both",
                "threshold_boost": legacy_boost,
                "block": False,
                "reason": "regime_threshold_boost",
            }
        )

    return rules


def _normalize_regime_threshold_schedule(regime_filter_cfg: dict) -> list[dict]:
    rules = []
    explicit_rules = regime_filter_cfg.get("threshold_schedule", []) or []
    for idx, rule in enumerate(explicit_rules):
        if not rule or rule.get("threshold") is None:
            continue
        normalized = dict(rule)
        normalized["name"] = normalized.get("name", f"threshold_schedule_rule_{idx}")
        normalized["regimes"] = [int(r) for r in (normalized.get("regimes", []) or [])]
        normalized["side"] = str(normalized.get("side", "both")).lower()
        normalized["threshold"] = float(normalized.get("threshold"))
        normalized["reason"] = normalized.get("reason", "regime_threshold_schedule")
        rules.append(normalized)
    return rules


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
    side_threshold_cfg = decision_cfg.get("primary_threshold_by_side", {}) or {}
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
        preds = primary_preds[["event_id", "y_prob"]].rename(columns={"y_prob": "p_primary"})

    merged = events_df.merge(preds, on="event_id", how="left")
    merged["decision_reason"] = ""

    missing_primary = merged["p_primary"].isna()
    if missing_primary.any():
        logger.warning(
            "Missing primary predictions for %d events; marking as skipped",
            int(missing_primary.sum()),
        )
        merged.loc[missing_primary, "decision_reason"] = "missing_primary"

    if "predicted_side" in merged.columns:
        merged["side"] = merged["predicted_side"].fillna(merged.get("side", 1))
        no_side = merged["predicted_side"] == 0
        if no_side.any():
            merged.loc[no_side & (merged["decision_reason"] == ""), "decision_reason"] = "no_positive_ev"

    if "side" not in merged.columns:
        merged["side"] = 1

    regime_filter_cfg = decision_cfg.get("regime_filter", {})
    regime_enabled = bool(regime_filter_cfg.get("enabled", False)) and bars_df is not None
    if regime_enabled:
        vol_window = int(regime_filter_cfg.get("vol_window", 20))
        trend_window = int(regime_filter_cfg.get("trend_window", 50))
        merged = compute_regime_at_events(merged, bars_df, vol_window, trend_window)

        if bool(regime_filter_cfg.get("enable_shorts", False)):
            downtrend_regimes = regime_filter_cfg.get("downtrend_regimes", [0, 3, 6])
            is_downtrend = merged["combined_regime"].isin(downtrend_regimes)
            merged.loc[is_downtrend, "side"] = -1
            merged.loc[~is_downtrend, "side"] = 1
    else:
        merged["vol_regime"] = None
        merged["trend_regime"] = None
        merged["combined_regime"] = None
        merged["regime_label"] = None

    long_threshold = float(side_threshold_cfg.get("long", primary_threshold))
    short_threshold = float(side_threshold_cfg.get("short", primary_threshold))
    merged["default_primary_threshold"] = np.where(
        merged["side"].fillna(1).astype(float) >= 0,
        long_threshold,
        short_threshold,
    )
    merged["base_primary_threshold"] = merged["default_primary_threshold"].astype(float)
    merged["regime_threshold_schedule_applied"] = False
    merged["regime_threshold_schedule_reason"] = ""
    merged["regime_threshold_adjustment"] = 0.0
    merged["regime_overlay_reason"] = ""
    merged["regime_blocked"] = False

    if regime_enabled:
        for rule in _normalize_regime_threshold_schedule(regime_filter_cfg):
            if not rule["regimes"]:
                continue
            mask = merged["combined_regime"].isin(rule["regimes"])
            mask &= _overlay_side_mask(merged["side"], rule["side"])
            if not mask.any():
                continue

            merged.loc[mask, "base_primary_threshold"] = rule["threshold"]
            merged.loc[mask, "regime_threshold_schedule_applied"] = True
            merged.loc[mask, "regime_threshold_schedule_reason"] = rule["reason"]

        for rule in _normalize_regime_overlay_rules(regime_filter_cfg):
            if not rule["regimes"]:
                continue
            mask = merged["combined_regime"].isin(rule["regimes"])
            mask &= _overlay_side_mask(merged["side"], rule["side"])
            if not mask.any():
                continue

            if rule["block"]:
                merged.loc[mask, "regime_blocked"] = True
                empty_reason = mask & (merged["regime_overlay_reason"] == "")
                merged.loc[empty_reason, "regime_overlay_reason"] = rule["reason"]

            threshold_boost = float(rule.get("threshold_boost", 0.0) or 0.0)
            if threshold_boost > 0.0:
                stronger = mask & (merged["regime_threshold_adjustment"] < threshold_boost)
                merged.loc[stronger, "regime_threshold_adjustment"] = threshold_boost
                merged.loc[stronger, "regime_overlay_reason"] = rule["reason"]

    merged["applied_primary_threshold"] = (
        merged["base_primary_threshold"] + merged["regime_threshold_adjustment"]
    )
    merged["proposed"] = merged["p_primary"] >= merged["applied_primary_threshold"]
    merged["accept"] = merged["proposed"].copy()

    if missing_primary.any():
        merged.loc[missing_primary, "accept"] = False
    if "predicted_side" in merged.columns:
        no_side = merged["predicted_side"] == 0
        if no_side.any():
            merged.loc[no_side, "accept"] = False

    schedule_reject = (
        merged["regime_threshold_schedule_applied"]
        & (merged["base_primary_threshold"] > merged["default_primary_threshold"])
        & (merged["p_primary"] >= merged["default_primary_threshold"])
        & (merged["p_primary"] < merged["base_primary_threshold"])
    )
    if schedule_reject.any():
        reason_mask = schedule_reject & (merged["decision_reason"] == "")
        merged.loc[reason_mask, "decision_reason"] = merged.loc[
            reason_mask, "regime_threshold_schedule_reason"
        ].replace("", "regime_threshold_schedule")

    overlay_reject = (
        (merged["regime_threshold_adjustment"] > 0.0)
        & (merged["p_primary"] >= merged["base_primary_threshold"])
        & (merged["p_primary"] < merged["applied_primary_threshold"])
    )
    if overlay_reject.any():
        reason_mask = overlay_reject & (merged["decision_reason"] == "")
        merged.loc[reason_mask, "decision_reason"] = merged.loc[
            reason_mask, "regime_overlay_reason"
        ].replace("", "regime_threshold_boost")

    blocked = merged["regime_blocked"] & merged["accept"]
    if blocked.any():
        merged.loc[blocked, "accept"] = False
        reason_mask = blocked & (merged["decision_reason"] == "")
        merged.loc[reason_mask, "decision_reason"] = merged.loc[
            reason_mask, "regime_overlay_reason"
        ].replace("", "regime_overlay_block")

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

    if regime_enabled:
        skip_regimes = regime_filter_cfg.get("skip_regimes", []) or []
        if skip_regimes:
            in_skip_regime = merged["combined_regime"].isin(skip_regimes)
            n_skip = int((in_skip_regime & merged["accept"]).sum())
            if n_skip > 0:
                merged.loc[in_skip_regime & merged["accept"], "accept"] = False
                merged.loc[in_skip_regime & (merged["decision_reason"] == ""), "decision_reason"] = "regime_skip"
                logger.info(
                    "Regime filter: skipped %d trades in dangerous regimes %s",
                    n_skip,
                    skip_regimes,
                )

        if merged["accept"].any():
            accepted_regimes = merged.loc[merged["accept"], "regime_label"].value_counts()
            logger.info("Accepted trades by regime:\n%s", accepted_regimes.to_string())

            if bool(regime_filter_cfg.get("enable_shorts", False)):
                side_counts = merged.loc[merged["accept"], "side"].value_counts()
                logger.info(
                    "Trade sides: %d longs, %d shorts",
                    side_counts.get(1, 0),
                    side_counts.get(-1, 0),
                )

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
            pre_meta_accept = merged["accept"].copy()
            merged["accept"] = merged["accept"] & meta_accept
            merged.loc[
                ~merged["proposed"] & (merged["decision_reason"] == ""),
                "decision_reason",
            ] = "threshold_primary"
            merged.loc[
                pre_meta_accept & ~meta_accept & (merged["decision_reason"] == ""),
                "decision_reason",
            ] = "threshold_meta"
        else:
            pre_meta_accept = merged["accept"].copy()
            merged["accept"] = merged["accept"] & (meta_accept | merged["p_meta"].isna())
            merged.loc[
                ~merged["proposed"] & (merged["decision_reason"] == ""),
                "decision_reason",
            ] = "threshold_primary"
            merged.loc[
                pre_meta_accept
                & ~meta_accept
                & ~merged["p_meta"].isna()
                & (merged["decision_reason"] == ""),
                "decision_reason",
            ] = "threshold_meta"
    else:
        merged.loc[
            ~merged["proposed"] & (merged["decision_reason"] == ""),
            "decision_reason",
        ] = "threshold_primary"

    if "p_meta" not in merged.columns:
        merged["p_meta"] = None

    return merged
