#!/usr/bin/env python3
"""
Rolling daily refit OOS: for each Chicago session day in a test range, train on a
lookback window ending strictly before that day, then run the same promotion path
as run_standalone_topstep_candidate (fit + run_backtest).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.experiments.daily_refit_helpers import (  # noqa: E402
    assert_train_before_test,
    chicago_session_dates_in_range,
    slice_bars_for_daily_refit,
)
from ml_intraday_v3.experiments.run_standalone_topstep_candidate import (  # noqa: E402
    _derive_cost_mode,
    _fit_promotion_window_artifacts,
    _load_bars,
    _load_yaml,
    _period_bounds,
    _prepare_events_and_dataset,
    _prepare_events_and_dataset_session_day,
)
from ml_intraday_v3.backtesting_v3 import run_backtest  # noqa: E402
from ml_intraday_v3.core.instrument import load_instrument_from_execution_spec  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING = "ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml"
DEFAULT_LABELING = "ml_intraday_v3/configs/labeling.yaml"
DEFAULT_FEATURES = "ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml"
DEFAULT_EXECUTION = "ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec.yaml"
DEFAULT_BACKTEST = "ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml"
DEFAULT_RISK = "ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml"


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _expand_paths(paths: list[str] | None) -> list[Path]:
    if not paths:
        return []
    out: list[Path] = []
    for raw in paths:
        for part in raw.split(","):
            p = part.strip()
            if not p:
                continue
            path = Path(p)
            out.append(path if path.is_absolute() else PROJECT_ROOT / path)
    return out


def _resolve_cfg(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def run_daily_refit_oos(
    *,
    data_paths: list[Path],
    hdf_key: str,
    test_start: date,
    test_end: date,
    lookback_days: int,
    gap_days: int,
    training_cfg_path: Path,
    labeling_cfg_path: Path,
    feature_cfg_path: Path,
    execution_spec_path: Path,
    backtest_cfg_path: Path,
    risk_cfg_path: Path,
    output_dir: Path,
    bar_size: str,
    min_train_events: int,
    write_all_trades: bool,
    compare_single_window: bool,
) -> dict:
    training_cfg = _load_yaml(training_cfg_path)
    labeling_cfg = _load_yaml(labeling_cfg_path)
    feature_cfg = _load_yaml(feature_cfg_path)
    execution_spec = _load_yaml(execution_spec_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    risk_cfg = _load_yaml(risk_cfg_path)

    bars = _load_bars(data_paths, hdf_key)
    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

    session_days = chicago_session_dates_in_range(bars.index, test_start, test_end)
    logger.info(
        "Chicago session days in [%s, %s]: %d days",
        test_start,
        test_end,
        len(session_days),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_rows: list[dict] = []
    skipped: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    for d in session_days:
        day_str = d.isoformat()
        name = f"daily_{day_str}"
        bars_train, bars_test, train_start, train_end, test_start_ts, test_end_ts = (
            slice_bars_for_daily_refit(
                bars,
                session_date=d,
                lookback_days=lookback_days,
                gap_days=gap_days,
            )
        )
        base_row: dict = {
            "session_date": day_str,
            "train_bars": int(len(bars_train)),
            "test_bars": int(len(bars_test)),
        }

        if bars_test.empty or train_start is None:
            skipped.append({"session_date": day_str, "reason": "empty_test_bars"})
            daily_rows.append(
                {
                    **base_row,
                    "skipped": True,
                    "skip_reason": "empty_test_bars",
                    "train_events": 0,
                    "test_events": 0,
                    "total_pnl_usd": 0.0,
                    "trades_count": 0,
                    "max_drawdown_usd": 0.0,
                }
            )
            continue

        assert_train_before_test(bars_train, bars_test)

        if bars_train.empty:
            skipped.append({"session_date": day_str, "reason": "empty_train_bars"})
            daily_rows.append(
                {
                    **base_row,
                    "skipped": True,
                    "skip_reason": "empty_train_bars",
                    "train_events": 0,
                    "test_events": 0,
                    "total_pnl_usd": 0.0,
                    "trades_count": 0,
                    "max_drawdown_usd": 0.0,
                }
            )
            continue

        _te, train_df = _prepare_events_and_dataset(
            bars_df=bars_train,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            balance_train=True,
        )
        if len(train_df) < min_train_events:
            reason = f"train_events<{min_train_events}"
            skipped.append({"session_date": day_str, "reason": reason})
            daily_rows.append(
                {
                    **base_row,
                    "skipped": True,
                    "skip_reason": reason,
                    "train_events": int(len(train_df)),
                    "test_events": 0,
                    "total_pnl_usd": 0.0,
                    "trades_count": 0,
                    "max_drawdown_usd": 0.0,
                }
            )
            continue

        bars_label_ctx = pd.concat([bars_train, bars_test]).sort_index()
        if bars_label_ctx.index.duplicated().any():
            bars_label_ctx = bars_label_ctx[~bars_label_ctx.index.duplicated(keep="last")]

        _tev, test_df = _prepare_events_and_dataset_session_day(
            bars_df=bars_label_ctx,
            session_date=d,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
        )
        if len(test_df) == 0:
            daily_rows.append(
                {
                    **base_row,
                    "skipped": False,
                    "skip_reason": "no_test_events",
                    "train_events": int(len(train_df)),
                    "test_events": 0,
                    "total_pnl_usd": 0.0,
                    "trades_count": 0,
                    "max_drawdown_usd": 0.0,
                }
            )
            continue

        logger.info(
            "Day %s | train=%s..%s (%d bars, %d events) | test=%s..%s (%d bars, %d events)",
            day_str,
            train_start,
            train_end,
            len(bars_train),
            len(train_df),
            test_start_ts,
            test_end_ts,
            len(bars_test),
            len(test_df),
        )

        art = _fit_promotion_window_artifacts(
            name=name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start_ts,
            test_end=test_end_ts,
            bars_train=bars_train,
            bars_test=bars_test,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            backtest_cfg=backtest_cfg,
            bars_test_label_context=bars_label_ctx,
            session_day_chicago=d,
        )

        trades_df, equity_df, backtest_metrics = run_backtest(
            events_df=art["test_events_df"],
            bars_df=art["bars_test"],
            primary_preds_df=art["primary_preds"],
            meta_preds_df=art["meta_preds"],
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            label_schema=label_schema,
            risk_cfg=risk_cfg,
            backtest_cfg=art["local_backtest_cfg"],
            bar_size=bar_size,
        )

        trades_df = trades_df.copy()
        trades_df["session_date"] = day_str

        pnl = float(backtest_metrics.get("total_pnl_usd", 0.0) or 0.0)
        n_tr = int(backtest_metrics.get("trades_count", 0) or 0)
        mdd = float(backtest_metrics.get("max_drawdown_usd", 0.0) or 0.0)

        daily_rows.append(
            {
                **base_row,
                "skipped": False,
                "skip_reason": "",
                "train_events": int(art["train_events"]),
                "test_events": int(art["test_events"]),
                "total_pnl_usd": pnl,
                "trades_count": n_tr,
                "max_drawdown_usd": mdd,
            }
        )
        trade_frames.append(trades_df)

    daily_df = pd.DataFrame(daily_rows)
    daily_csv = output_dir / "daily_metrics.csv"
    daily_df.to_csv(daily_csv, index=False)

    ran_backtest = (daily_df["skipped"] == False) & (daily_df["skip_reason"] == "")  # noqa: E712
    sum_pnl = float(daily_df["total_pnl_usd"].sum())
    sum_trades = int(daily_df["trades_count"].sum())

    summary: dict = {
        "test_range": {"start": test_start.isoformat(), "end": test_end.isoformat()},
        "lookback_days": lookback_days,
        "gap_days": gap_days,
        "session_days_in_index": len(session_days),
        "days_with_backtest": int(ran_backtest.sum()),
        "aggregate_total_pnl_usd": sum_pnl,
        "aggregate_trades_count": sum_trades,
        "skipped_days": skipped,
        "daily_metrics_csv": str(daily_csv.relative_to(PROJECT_ROOT)),
        "labeling_note": (
            "Per day, test labeling uses concat(train_bars, session_bars) for indicator/event "
            "context; only events with t0 on that Chicago session_date are scored and backtested."
        ),
    }

    if write_all_trades and trade_frames:
        non_empty = [t for t in trade_frames if not t.empty]
        if non_empty:
            all_trades = pd.concat(non_empty, ignore_index=True)
            tp = output_dir / "all_trades.parquet"
            all_trades.to_parquet(tp)
            summary["all_trades_parquet"] = str(tp.relative_to(PROJECT_ROOT))

    if compare_single_window:
        wcfg = {
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        }
        tr_s, tr_e, ts_s, ts_e = _period_bounds(
            wcfg,
            default_lookback_days=lookback_days,
            default_gap_days=gap_days,
        )
        b_tr = bars[(bars.index >= tr_s) & (bars.index <= tr_e)].copy()
        b_te = bars[(bars.index >= ts_s) & (bars.index <= ts_e)].copy()
        if b_tr.empty or b_te.empty:
            summary["reference_single_window"] = {
                "error": "empty_train_or_test_bars_for_single_window",
            }
        else:
            art1 = _fit_promotion_window_artifacts(
                name="single_window_compare",
                train_start=tr_s,
                train_end=tr_e,
                test_start=ts_s,
                test_end=ts_e,
                bars_train=b_tr,
                bars_test=b_te,
                bar_size=bar_size,
                labeling_cfg=labeling_cfg,
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                feature_cfg=feature_cfg,
                training_cfg=training_cfg,
                backtest_cfg=backtest_cfg,
            )
            _t, _e, bt_m = run_backtest(
                events_df=art1["test_events_df"],
                bars_df=art1["bars_test"],
                primary_preds_df=art1["primary_preds"],
                meta_preds_df=art1["meta_preds"],
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_cfg,
                backtest_cfg=art1["local_backtest_cfg"],
                bar_size=bar_size,
            )
            summary["reference_single_window_pnl"] = {
                "total_pnl_usd": float(bt_m.get("total_pnl_usd", 0.0) or 0.0),
                "trades_count": int(bt_m.get("trades_count", 0) or 0),
                "max_drawdown_usd": float(bt_m.get("max_drawdown_usd", 0.0) or 0.0),
                "note": "Single fit on full test range; UTC bounds from _period_bounds differ from per-day Chicago slices.",
            }

    agg_path = output_dir / "aggregate_summary.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Wrote %s and %s", daily_csv, agg_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily refit OOS over Chicago session days.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--additional-path",
        action="append",
        default=[],
        help="Extra HDF paths (repeat flag or comma-separated in one value).",
    )
    parser.add_argument("--hdf-key", type=str, default="bars_5min")
    parser.add_argument("--test-start", type=str, required=True, help="YYYY-MM-DD (Chicago filter)")
    parser.add_argument("--test-end", type=str, required=True)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--gap-days", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="ml_intraday_v3/experiments/results/daily_refit_apr2026")
    parser.add_argument("--bar-size", type=str, default="5m")
    parser.add_argument("--training-cfg", type=str, default=DEFAULT_TRAINING)
    parser.add_argument("--labeling-cfg", type=str, default=DEFAULT_LABELING)
    parser.add_argument("--feature-cfg", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--execution-spec", type=str, default=DEFAULT_EXECUTION)
    parser.add_argument("--backtest-cfg", type=str, default=DEFAULT_BACKTEST)
    parser.add_argument("--risk-cfg", type=str, default=DEFAULT_RISK)
    parser.add_argument("--min-train-events", type=int, default=250)
    parser.add_argument(
        "--acceptance-cfg",
        type=str,
        default=None,
        help="Optional YAML; gates.min_train_events overrides --min-train-events when present.",
    )
    parser.add_argument("--write-all-trades", action="store_true")
    parser.add_argument(
        "--compare-single-window",
        action="store_true",
        help="Also one fit+backtest on full [test-start,test-end] via _period_bounds (UTC).",
    )
    args = parser.parse_args()

    dp = Path(args.data_path)
    data_paths = [(dp if dp.is_absolute() else PROJECT_ROOT / dp).resolve()]
    data_paths.extend(_expand_paths(args.additional_path))

    min_train = args.min_train_events
    if args.acceptance_cfg:
        acc = _load_yaml(_resolve_cfg(args.acceptance_cfg))
        g = acc.get("gates") or {}
        if "min_train_events" in g:
            min_train = int(g["min_train_events"])

    out = _resolve_cfg(args.output_dir)

    run_daily_refit_oos(
        data_paths=data_paths,
        hdf_key=args.hdf_key,
        test_start=_parse_date(args.test_start),
        test_end=_parse_date(args.test_end),
        lookback_days=args.lookback_days,
        gap_days=args.gap_days,
        training_cfg_path=_resolve_cfg(args.training_cfg),
        labeling_cfg_path=_resolve_cfg(args.labeling_cfg),
        feature_cfg_path=_resolve_cfg(args.feature_cfg),
        execution_spec_path=_resolve_cfg(args.execution_spec),
        backtest_cfg_path=_resolve_cfg(args.backtest_cfg),
        risk_cfg_path=_resolve_cfg(args.risk_cfg),
        output_dir=out,
        bar_size=args.bar_size,
        min_train_events=min_train,
        write_all_trades=args.write_all_trades,
        compare_single_window=args.compare_single_window,
    )


if __name__ == "__main__":
    main()
