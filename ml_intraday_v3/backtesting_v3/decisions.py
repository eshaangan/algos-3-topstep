"""
Decision logic for offline backtest.
"""

import pandas as pd


def decide_trades(events_df: pd.DataFrame, primary_preds: pd.DataFrame, meta_preds: pd.DataFrame | None, config: dict) -> pd.DataFrame:
    """
    Decide whether to take trades based on primary/meta predictions.

    Returns events_df with columns:
    accept (bool), decision_reason, p_primary, p_meta
    """
    decision_cfg = config.get("decision", {})
    use_meta = bool(decision_cfg.get("use_meta", False))
    primary_threshold = float(decision_cfg.get("primary_threshold", 0.5))
    meta_threshold = float(decision_cfg.get("meta_threshold", 0.5))
    require_meta = bool(decision_cfg.get("require_meta_for_trade", True))

    preds = primary_preds[["event_id", "y_prob"]].rename(
        columns={"y_prob": "p_primary"}
    )
    merged = events_df.merge(preds, on="event_id", how="left")
    if merged["p_primary"].isna().any():
        raise ValueError("Missing primary predictions for some events")

    merged["proposed"] = merged["p_primary"] >= primary_threshold
    merged["accept"] = merged["proposed"].copy()
    merged["decision_reason"] = ""

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
