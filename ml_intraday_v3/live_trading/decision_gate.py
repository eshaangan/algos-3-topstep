"""
Live decision helpers that mirror offline backtest gating.

These utilities adapt a single live prediction into the same decision surface
used by the backtest so routed meta vetoes and regime threshold schedules stay
in sync.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ml_intraday_v3.backtesting_v3.decisions import compute_regime_at_events, decide_trades

# Short explanations for decide_trades / live decision_reason codes (logging).
DECISION_REASON_GUIDE: dict[str, str] = {
    "threshold_primary": "Score (p_primary / |EV|) is below the applied primary threshold for this side.",
    "threshold_meta": "Primary gate passed, but meta probability is below meta_threshold.",
    "missing_meta": "Meta output is missing while require_meta_approval is true.",
    "missing_primary": "No primary score joined for this event.",
    "no_positive_ev": "Bidirectional model set side=0 (no side cleared the EV gate).",
    "low_volatility": "Decision volatility filter rejected (sigma too low).",
    "high_volatility": "Decision volatility filter rejected (sigma too high).",
    "regime_skip": "Regime filter skip_regimes blocked this bar.",
    "regime_overlay_block": "Regime overlay rule blocked trading in this combined regime.",
    "regime_threshold_schedule": "Regime-specific threshold schedule raised the bar above your score.",
    "regime_threshold_boost": "Regime overlay raised the effective threshold above your score.",
    "rejected": "Rejected (no specific reason set — check gates).",
}


def explain_live_decision(
    *,
    bar_time: pd.Timestamp,
    prediction: dict[str, Any],
    runtime_decision_cfg: dict,
    decision_row: dict[str, Any],
) -> str:
    """
    Build a multi-line summary for INFO logs: scores, thresholds, gates, reason.
    """
    dec = (runtime_decision_cfg.get("decision") or {}) if isinstance(runtime_decision_cfg, dict) else {}
    lines: list[str] = [f"  bar={bar_time}"]

    score_ev = prediction.get("score_ev")
    if score_ev is not None:
        lines.append(f"  model score_ev={float(score_ev):.4f} y_prob={float(prediction.get('y_prob') or 0.0):.4f}")
    side = prediction.get("side")
    if side is not None:
        lines.append(f"  predicted_side={int(side)} meta_route={prediction.get('meta_route')!r}")

    p_primary = decision_row.get("p_primary")
    if p_primary is not None and pd.notna(p_primary):
        lines.append(f"  decision p_primary={float(p_primary):.4f} (thresholded score)")

    def _f(key: str) -> str:
        v = decision_row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "n/a"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    lines.append(
        f"  thresholds: default_primary={_f('default_primary_threshold')} "
        f"base={_f('base_primary_threshold')} applied={_f('applied_primary_threshold')}"
    )

    proposed = bool(decision_row.get("proposed", False))
    accept = bool(decision_row.get("accept", False))
    lines.append(f"  gates: proposed_primary={proposed} final_accept={accept}")

    p_meta = decision_row.get("p_meta")
    if p_meta is not None and pd.notna(p_meta):
        mt = float(dec.get("meta_threshold", 0.5) or 0.5)
        pm = float(p_meta)
        req = bool(dec.get("require_meta_for_trade", True))
        lines.append(
            f"  meta: p_meta={pm:.4f} require={req} threshold={mt:.4f} -> "
            f"{'PASS' if pm >= mt else 'FAIL'}"
        )

    rl = decision_row.get("regime_label")
    if rl is not None and not (isinstance(rl, float) and pd.isna(rl)):
        cr = decision_row.get("combined_regime")
        lines.append(f"  regime: label={rl!r} combined_regime={cr}")

    reason = str(decision_row.get("decision_reason") or "").strip()
    if reason:
        lines.append(f"  decision_reason={reason}")
        guide = DECISION_REASON_GUIDE.get(reason)
        if guide:
            lines.append(f"  meaning: {guide}")

    return "\n".join(lines)


def build_live_decision_config(
    *,
    backtest_cfg: dict | None,
    live_cfg: dict | None = None,
    bundle_decision_cfg: dict | None = None,
) -> dict:
    """
    Merge live overrides on top of the promoted decision config.

    `bundle_decision_cfg` may be either a full config dict or just the nested
    `decision` payload saved inside a model bundle.
    """
    cfg = deepcopy(backtest_cfg or {})
    decision_cfg = cfg.setdefault("decision", {})

    if bundle_decision_cfg:
        bundle_decision = (
            bundle_decision_cfg.get("decision", {})
            if "decision" in bundle_decision_cfg
            else bundle_decision_cfg
        )
        for key, value in (bundle_decision or {}).items():
            decision_cfg[key] = deepcopy(value)

    signals_cfg = ((live_cfg or {}).get("signals", {}) or {}) if isinstance(live_cfg, dict) else {}
    if not signals_cfg and isinstance(live_cfg, dict):
        signals_cfg = live_cfg or {}

    if "primary_threshold" in signals_cfg:
        decision_cfg["primary_threshold"] = float(signals_cfg["primary_threshold"])

    side_thresholds = dict(decision_cfg.get("primary_threshold_by_side", {}) or {})
    if "primary_threshold_long" in signals_cfg:
        side_thresholds["long"] = float(signals_cfg["primary_threshold_long"])
    if "primary_threshold_short" in signals_cfg:
        side_thresholds["short"] = float(signals_cfg["primary_threshold_short"])
    if side_thresholds:
        decision_cfg["primary_threshold_by_side"] = side_thresholds

    if "use_meta_model" in signals_cfg:
        decision_cfg["use_meta"] = bool(signals_cfg["use_meta_model"])
    if "meta_threshold" in signals_cfg:
        decision_cfg["meta_threshold"] = float(signals_cfg["meta_threshold"])
    if "require_meta_approval" in signals_cfg:
        decision_cfg["require_meta_for_trade"] = bool(signals_cfg["require_meta_approval"])

    return cfg


def compute_live_regime_context(
    *,
    timestamp: pd.Timestamp,
    bars_df: pd.DataFrame,
    decision_config: dict,
    event_id: str = "live_event",
) -> dict[str, Any]:
    """
    Compute the current regime at a live event timestamp.
    """
    decision_cfg = (decision_config.get("decision", {}) or {})
    regime_cfg = (decision_cfg.get("regime_filter", {}) or {})
    if not regime_cfg.get("enabled", False) or bars_df is None or bars_df.empty:
        return {
            "event_id": event_id,
            "t0": pd.Timestamp(timestamp),
            "vol_regime": None,
            "trend_regime": None,
            "combined_regime": None,
            "regime_label": None,
        }

    event_df = pd.DataFrame({"event_id": [event_id], "t0": [pd.Timestamp(timestamp)]})
    regime_df = compute_regime_at_events(
        event_df,
        bars_df,
        vol_window=int(regime_cfg.get("vol_window", 20)),
        trend_window=int(regime_cfg.get("trend_window", 50)),
    )
    return regime_df.iloc[0].to_dict()


def evaluate_live_trade_decision(
    *,
    timestamp: pd.Timestamp,
    prediction: dict[str, Any],
    bars_df: pd.DataFrame,
    decision_config: dict,
    event_id: str = "live_event",
) -> dict[str, Any]:
    """
    Evaluate one live prediction with the offline decision graph.
    """
    local_cfg = deepcopy(decision_config)
    local_decision = local_cfg.setdefault("decision", {})
    if prediction.get("meta_threshold") is not None:
        local_decision["meta_threshold"] = float(prediction["meta_threshold"])

    predicted_side = int(prediction.get("side", 0) or 0)
    events_df = pd.DataFrame(
        {
            "event_id": [event_id],
            "t0": [pd.Timestamp(timestamp)],
            "side": [predicted_side if predicted_side != 0 else 1],
        }
    )
    if prediction.get("sigma") is not None:
        events_df["sigma"] = [float(prediction["sigma"])]

    primary_preds = pd.DataFrame(
        {
            "event_id": [event_id],
            "y_prob": [float(prediction.get("y_prob", 0.0) or 0.0)],
            "score_ev": [float(prediction.get("score_ev", 0.0) or 0.0)],
            "predicted_side": [predicted_side],
        }
    )
    for column in ("p_stop", "p_target", "p_vertical"):
        if prediction.get(column) is not None:
            primary_preds[column] = [float(prediction[column])]

    use_meta = bool(local_decision.get("use_meta", False))
    if use_meta:
        if prediction.get("meta_prob") is None:
            meta_preds = pd.DataFrame(columns=["event_id", "p_meta"])
        else:
            meta_preds = pd.DataFrame(
                {"event_id": [event_id], "p_meta": [float(prediction["meta_prob"])]}
            )
    else:
        meta_preds = None

    decisions = decide_trades(
        events_df=events_df,
        primary_preds=primary_preds,
        meta_preds=meta_preds,
        config=local_cfg,
        bars_df=bars_df,
    )
    result = decisions.iloc[0].to_dict()
    if prediction.get("meta_route"):
        result["meta_route"] = prediction.get("meta_route")
    if prediction.get("meta_route_reason"):
        result["meta_route_reason"] = prediction.get("meta_route_reason")
    return result
