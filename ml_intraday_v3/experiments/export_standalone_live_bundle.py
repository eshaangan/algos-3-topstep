#!/usr/bin/env python3
"""
Export a promoted standalone candidate into a live-consumable bundle.

This trains the primary model on the most recent rolling training window and
fits routed meta veto models on a recent holdout slice so the live artifact can
mirror the promoted dual-meta decision surface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.core.instrument import load_instrument_from_execution_spec
from ml_intraday_v3.experiments.run_standalone_topstep_candidate import (
    _derive_cost_mode,
    _fit_candidate_model,
    _load_bars,
    _load_yaml,
    _prepare_events_and_dataset,
    _score_candidate,
    _to_utc_timestamp,
    _train_targeted_meta_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_recent_training_slice(
    *,
    bars: pd.DataFrame,
    acceptance_cfg: dict,
    as_of: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    schedule_cfg = acceptance_cfg.get("training_window", {}) or {}
    lookback_days = int(schedule_cfg.get("lookback_days", 180))
    gap_days = int(schedule_cfg.get("gap_days", 0))

    if as_of is None:
        train_end = pd.Timestamp(bars.index.max())
    else:
        train_end = pd.Timestamp(as_of)
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    else:
        train_end = train_end.tz_convert("UTC")

    train_end = min(train_end, pd.Timestamp(bars.index.max()))
    train_end = train_end - pd.Timedelta(days=gap_days)
    train_start = train_end - pd.Timedelta(days=lookback_days)
    return train_start, train_end


def _extract_meta_training_slice(
    *,
    train_df: pd.DataFrame,
    train_events: pd.DataFrame,
    training_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(train_df) * training_fraction)
    split_idx = max(1, min(split_idx, len(train_df) - 1))
    primary_seed_df = train_df.iloc[:split_idx].reset_index(drop=True)
    meta_df = train_df.iloc[split_idx:].reset_index(drop=True)
    primary_seed_events = train_events.iloc[:split_idx].reset_index(drop=True)
    meta_events = train_events.iloc[split_idx:].reset_index(drop=True)
    return primary_seed_df, primary_seed_events, meta_df, meta_events


def _build_live_decision_payload(training_cfg: dict, backtest_cfg: dict) -> dict:
    decision = json.loads(json.dumps((backtest_cfg.get("decision", {}) or {})))
    meta_cfg = training_cfg.get("meta", {}) or {}
    if meta_cfg.get("enabled", False):
        decision["use_meta"] = True
        decision["meta_threshold"] = float(meta_cfg.get("threshold_meta", decision.get("meta_threshold", 0.5)))
        decision["require_meta_for_trade"] = bool(
            meta_cfg.get("require_meta_for_trade", decision.get("require_meta_for_trade", False))
        )
    return decision


def export_live_bundle(
    *,
    data_path: Path,
    hdf_key: str,
    training_cfg_path: Path,
    labeling_cfg_path: Path,
    feature_cfg_path: Path,
    execution_spec_path: Path,
    backtest_cfg_path: Path,
    acceptance_cfg_path: Path,
    output_path: Path,
    as_of: pd.Timestamp | None,
    meta_seed_fraction: float,
    meta_lookback_days: int | None,
) -> dict:
    training_cfg = _load_yaml(training_cfg_path)
    labeling_cfg = _load_yaml(labeling_cfg_path)
    feature_cfg = _load_yaml(feature_cfg_path)
    execution_spec = _load_yaml(execution_spec_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    acceptance_cfg = _load_yaml(acceptance_cfg_path)

    data_cfg = acceptance_cfg.get("data", {}) or {}
    additional_paths = [PROJECT_ROOT / path for path in (data_cfg.get("additional_paths", []) or [])]
    bars = _load_bars([data_path, *additional_paths], hdf_key)
    train_start, train_end = _resolve_recent_training_slice(
        bars=bars,
        acceptance_cfg=acceptance_cfg,
        as_of=as_of,
    )
    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)].copy()
    if bars_train.empty:
        raise ValueError("No training bars available for export window")

    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    bar_size = data_cfg.get("bar_size", "5m")
    label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

    train_events, train_df = _prepare_events_and_dataset(
        bars_df=bars_train,
        bar_size=bar_size,
        labeling_cfg=labeling_cfg,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
        feature_cfg=feature_cfg,
        training_cfg=training_cfg,
        balance_train=True,
    )

    logger.info(
        "Export training slice %s .. %s | bars=%d events=%d",
        train_start,
        train_end,
        len(bars_train),
        len(train_df),
    )

    primary_candidate = _fit_candidate_model(train_df, training_cfg)
    if primary_candidate.get("mode") != "single":
        raise ValueError(
            "Live export currently supports single primary models only; "
            f"got mode={primary_candidate.get('mode')!r}"
        )

    meta_result: dict = {"enabled": False}
    meta_cfg = training_cfg.get("meta", {}) or {}
    if meta_cfg.get("enabled", False):
        primary_lookback_days = int(
            (acceptance_cfg.get("training_window", {}) or {}).get("lookback_days", 180)
        )
        effective_meta_lookback = (
            int(meta_lookback_days)
            if meta_lookback_days is not None
            else max(primary_lookback_days, 365)
        )
        meta_start = train_end - pd.Timedelta(days=effective_meta_lookback)
        bars_meta = bars[(bars.index >= meta_start) & (bars.index <= train_end)].copy()
        meta_events_all, meta_df_all = _prepare_events_and_dataset(
            bars_df=bars_meta,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            balance_train=True,
        )
        logger.info(
            "Meta training slice %s .. %s | bars=%d events=%d",
            meta_start,
            train_end,
            len(bars_meta),
            len(meta_df_all),
        )
        primary_seed_df, primary_seed_events, meta_df, meta_events = _extract_meta_training_slice(
            train_df=meta_df_all,
            train_events=meta_events_all,
            training_fraction=meta_seed_fraction,
        )
        seed_candidate = _fit_candidate_model(primary_seed_df, training_cfg)
        _seed_classification, meta_primary_preds = _score_candidate(meta_df, seed_candidate, training_cfg)
        meta_result = _train_targeted_meta_model(
            train_df=meta_df,
            test_df=meta_df,
            train_events=meta_events,
            test_events=meta_events,
            primary_train_preds=meta_primary_preds,
            primary_test_preds=meta_primary_preds,
            training_cfg=training_cfg,
        )
        logger.info(
            "Routed meta export prepared: enabled=%s skipped=%s routes=%d",
            meta_result.get("enabled", False),
            meta_result.get("skipped", False),
            len(meta_result.get("routes", []) or []),
        )

    meta_routes = []
    for route in (meta_result.get("routes", []) or []):
        if route.get("skipped", False):
            continue
        meta_routes.append(
            {
                "name": route.get("name"),
                "side": (route.get("route", {}) or {}).get("side"),
                "regimes": (route.get("route", {}) or {}).get("regimes", []),
                "threshold_primary": (route.get("route", {}) or {}).get("threshold_primary"),
                "threshold_meta": (route.get("route", {}) or {}).get("threshold_meta"),
                "model": route.get("model"),
                "preprocessor": route.get("preprocessor").state() if route.get("preprocessor") else None,
                "feature_columns": route.get("feature_cols", []),
            }
        )

    bundle = {
        "primary_model": primary_candidate["model"],
        "primary_preprocessor": primary_candidate["preprocessor"].state(),
        "primary_feature_columns": primary_candidate["feature_columns"],
        "meta_model": None,
        "meta_preprocessor": None,
        "meta_feature_columns": None,
        "meta_routes": meta_routes,
        "thresholds": {
            "primary_threshold": float(
                ((backtest_cfg.get("decision", {}) or {}).get("primary_threshold", 0.5))
            ),
            "meta_threshold": float((meta_cfg.get("threshold_meta", 0.5))),
        },
        "live_decision": _build_live_decision_payload(training_cfg, backtest_cfg),
        "label_schema": label_schema,
        "training_window": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
            "lookback_days": int((acceptance_cfg.get("training_window", {}) or {}).get("lookback_days", 180)),
        },
        "metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "bar_size": bar_size,
            "train_rows": int(len(train_df)),
            "train_events": int(len(train_events)),
            "meta_route_count": int(len(meta_routes)),
            "meta_seed_fraction": float(meta_seed_fraction),
            "training_config_path": str(training_cfg_path),
            "backtest_config_path": str(backtest_cfg_path),
            "feature_config_path": str(feature_cfg_path),
            "execution_spec_path": str(execution_spec_path),
            "acceptance_config_path": str(acceptance_cfg_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    summary = {
        "bundle_path": str(output_path),
        "training_window": bundle["training_window"],
        "meta_routes": [
            {
                "name": route["name"],
                "side": route["side"],
                "regimes": route["regimes"],
                "threshold_primary": route["threshold_primary"],
                "threshold_meta": route["threshold_meta"],
            }
            for route in meta_routes
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export standalone live bundle")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/mes_bars_databento_rth.h5"),
    )
    parser.add_argument("--hdf-key", default="bars_5min")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml"),
    )
    parser.add_argument(
        "--labeling-config",
        type=Path,
        default=Path("ml_intraday_v3/configs/labeling.yaml"),
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=Path("ml_intraday_v3/configs/features_structure_context.yaml"),
    )
    parser.add_argument(
        "--execution-spec",
        type=Path,
        default=Path("ml_intraday_v3/configs/execution_spec.yaml"),
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=Path("ml_intraday_v3/configs/backtest_standalone_topstep_recent_decay_dual_meta.yaml"),
    )
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=Path("ml_intraday_v3/configs/standalone_viability.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml_intraday_v3/models/live/dual_meta_mes_live_bundle.pkl"),
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="UTC timestamp cut-off for the live export window",
    )
    parser.add_argument(
        "--meta-seed-fraction",
        type=float,
        default=0.70,
        help="Chronological fraction reserved for seeding primary predictions before fitting meta routes",
    )
    parser.add_argument(
        "--meta-lookback-days",
        type=int,
        default=365,
        help="Lookback window for routed meta training; can exceed the primary training lookback",
    )
    args = parser.parse_args()

    summary = export_live_bundle(
        data_path=args.data_path,
        hdf_key=args.hdf_key,
        training_cfg_path=args.training_config,
        labeling_cfg_path=args.labeling_config,
        feature_cfg_path=args.feature_config,
        execution_spec_path=args.execution_spec,
        backtest_cfg_path=args.backtest_config,
        acceptance_cfg_path=args.acceptance_config,
        output_path=args.output,
        as_of=_to_utc_timestamp(args.as_of) if args.as_of else None,
        meta_seed_fraction=float(args.meta_seed_fraction),
        meta_lookback_days=int(args.meta_lookback_days) if args.meta_lookback_days else None,
    )
    logger.info("Export complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
