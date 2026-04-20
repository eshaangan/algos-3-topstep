"""
Promotion gates for standalone ML candidates.
"""

from __future__ import annotations

import math
from typing import Any


def _collect_daily_pnl(window_results: list[dict]) -> list[float]:
    """Concatenate daily PnL series across all windows in chronological order.

    Uses the window's ``risk_tearsheet.daily.daily_pnl_by_day`` map. Windows
    that lack daily data (no trades) contribute nothing.
    """
    all_days: list[tuple[str, float]] = []
    for window in window_results:
        risk = window.get("risk_tearsheet", {}) or {}
        daily = risk.get("daily", {}) or {}
        by_day = daily.get("daily_pnl_by_day") or {}
        if not isinstance(by_day, dict):
            continue
        for day, value in by_day.items():
            try:
                all_days.append((str(day), float(value)))
            except (TypeError, ValueError):
                continue
    all_days.sort(key=lambda pair: pair[0])
    return [pnl for _, pnl in all_days]


def _compute_overall_dsr(
    window_results: list[dict],
    n_trials: int,
) -> dict | None:
    """Compute DSR from daily PnL aggregated across windows.

    Returns None if the diagnostics module is unavailable or there are too
    few observations. DSR > 0.95 is the canonical "more likely skill than
    luck" threshold (Bailey & López de Prado 2014).
    """
    returns = _collect_daily_pnl(window_results)
    if len(returns) < 2:
        return None
    try:
        from ml_intraday_v3.experiments.diagnostics import compute_dsr
    except Exception:
        return None
    # Annualize per-day returns to 252-day Sharpe to match daily_sharpe convention.
    return compute_dsr(returns=returns, n_trials=int(n_trials), annualization_factor=math.sqrt(252.0))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def summarize_trade_directions(long_trades: int, short_trades: int) -> dict:
    total = max(0, int(long_trades)) + max(0, int(short_trades))
    if total == 0:
        return {
            "long_trades": int(long_trades),
            "short_trades": int(short_trades),
            "total_trades": 0,
            "long_share": None,
            "short_share": None,
        }
    return {
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "total_trades": int(total),
        "long_share": long_trades / total,
        "short_share": short_trades / total,
    }


def summarize_trades_df(trades_df) -> dict:
    if trades_df is None or len(trades_df) == 0:
        return summarize_trade_directions(0, 0)

    executed = trades_df
    if "executed" in executed.columns:
        executed = executed[executed["executed"]]

    if executed.empty or "side" not in executed.columns:
        return summarize_trade_directions(0, 0)

    long_trades = int((executed["side"] > 0).sum())
    short_trades = int((executed["side"] < 0).sum())
    return summarize_trade_directions(long_trades, short_trades)


def _check_min(
    failures: list[dict],
    label: str,
    actual: Any,
    threshold: Any,
) -> None:
    actual_f = _as_float(actual)
    threshold_f = _as_float(threshold)
    if actual_f is None or threshold_f is None:
        return
    if actual_f < threshold_f:
        failures.append(
            {
                "metric": label,
                "actual": actual_f,
                "expected": f">= {threshold_f}",
            }
        )


def _check_max(
    failures: list[dict],
    label: str,
    actual: Any,
    threshold: Any,
) -> None:
    actual_f = _as_float(actual)
    threshold_f = _as_float(threshold)
    if actual_f is None or threshold_f is None:
        return
    if actual_f > threshold_f:
        failures.append(
            {
                "metric": label,
                "actual": actual_f,
                "expected": f"<= {threshold_f}",
            }
        )


def _best_day_fraction(window_result: dict) -> float | None:
    """Topstep consistency proxy: best session PnL / total PnL for the window.

    Returns None when total PnL is not strictly positive — a losing or
    break-even window is rejected by existing min_total_pnl_usd gates, and the
    ratio is undefined/misleading when the denominator is not positive.
    """
    backtest_metrics = window_result.get("backtest_metrics", {}) or {}
    risk_tearsheet = window_result.get("risk_tearsheet", {}) or {}
    daily = risk_tearsheet.get("daily", {}) or {}

    total_pnl = _as_float(backtest_metrics.get("total_pnl_usd"))
    best_day = _as_float(daily.get("best_day_usd"))
    if total_pnl is None or best_day is None or total_pnl <= 0:
        return None
    return best_day / total_pnl


def evaluate_window_gates(window_result: dict, gates: dict) -> dict:
    failures: list[dict] = []
    classification = window_result.get("classification", {}) or {}
    backtest_metrics = window_result.get("backtest_metrics", {}) or {}
    direction_summary = window_result.get("direction_summary", {}) or {}

    _check_min(failures, "train_events", window_result.get("train_events"), gates.get("min_train_events"))
    _check_min(failures, "test_events", window_result.get("test_events"), gates.get("min_test_events"))
    _check_min(failures, "test_auc", classification.get("test_auc"), gates.get("min_test_auc"))
    _check_min(
        failures,
        "test_accuracy",
        classification.get("test_accuracy"),
        gates.get("min_test_accuracy"),
    )
    _check_min(
        failures,
        "total_pnl_usd",
        backtest_metrics.get("total_pnl_usd"),
        gates.get("min_total_pnl_usd"),
    )
    _check_min(
        failures,
        "avg_trade_usd",
        backtest_metrics.get("avg_trade_usd"),
        gates.get("min_avg_trade_usd"),
    )
    _check_min(
        failures,
        "profit_factor",
        backtest_metrics.get("profit_factor"),
        gates.get("min_profit_factor"),
    )
    _check_min(
        failures,
        "win_rate",
        backtest_metrics.get("win_rate"),
        gates.get("min_win_rate"),
    )
    _check_min(
        failures,
        "trades_count",
        backtest_metrics.get("trades_count"),
        gates.get("min_trades_count"),
    )
    _check_min(
        failures,
        "positive_prediction_rate",
        classification.get("positive_prediction_rate"),
        gates.get("min_positive_prediction_rate"),
    )

    _check_max(
        failures,
        "max_drawdown_usd",
        backtest_metrics.get("max_drawdown_usd"),
        gates.get("max_drawdown_usd"),
    )
    _check_max(
        failures,
        "mtm_daily_loss_liquidations",
        backtest_metrics.get("mtm_daily_loss_liquidations"),
        gates.get("max_mtm_daily_liquidations"),
    )
    _check_max(
        failures,
        "mtm_trailing_dd_liquidations",
        backtest_metrics.get("mtm_trailing_dd_liquidations"),
        gates.get("max_mtm_trailing_liquidations"),
    )

    _check_min(
        failures,
        "long_share",
        direction_summary.get("long_share"),
        gates.get("min_long_share"),
    )
    _check_max(
        failures,
        "long_share",
        direction_summary.get("long_share"),
        gates.get("max_long_share"),
    )
    _check_min(
        failures,
        "short_share",
        direction_summary.get("short_share"),
        gates.get("min_short_share"),
    )
    _check_max(
        failures,
        "short_share",
        direction_summary.get("short_share"),
        gates.get("max_short_share"),
    )

    best_day_fraction = _best_day_fraction(window_result)
    _check_max(
        failures,
        "best_day_fraction",
        best_day_fraction,
        gates.get("max_best_day_fraction"),
    )

    passed = len(failures) == 0
    return {
        "passed": passed,
        "failures": failures,
        "window_name": window_result.get("name"),
        "best_day_fraction": best_day_fraction,
    }


def evaluate_promotion_gates(window_results: list[dict], gates: dict) -> dict:
    evaluated_windows = []
    passed_count = 0
    total_pnl = 0.0
    total_trades = 0
    total_long = 0
    total_short = 0
    max_best_day_usd = float("-inf")

    for window in window_results:
        gate_result = evaluate_window_gates(window, gates)
        merged = {**window, "gate_result": gate_result}
        evaluated_windows.append(merged)
        if gate_result["passed"]:
            passed_count += 1

        backtest_metrics = window.get("backtest_metrics", {}) or {}
        total_pnl += _as_float(backtest_metrics.get("total_pnl_usd")) or 0.0
        total_trades += int(_as_float(backtest_metrics.get("trades_count")) or 0)

        direction_summary = window.get("direction_summary", {}) or {}
        total_long += int(direction_summary.get("long_trades") or 0)
        total_short += int(direction_summary.get("short_trades") or 0)

        daily = (window.get("risk_tearsheet", {}) or {}).get("daily", {}) or {}
        best_day = _as_float(daily.get("best_day_usd"))
        if best_day is not None and best_day > max_best_day_usd:
            max_best_day_usd = best_day

    total_windows = len(evaluated_windows)
    pass_ratio = (passed_count / total_windows) if total_windows else 0.0
    overall_direction = summarize_trade_directions(total_long, total_short)
    overall_failures: list[dict] = []

    if gates.get("require_all_windows", False) and passed_count != total_windows:
        overall_failures.append(
            {
                "metric": "all_windows",
                "actual": passed_count,
                "expected": f"{total_windows} passing windows",
            }
        )

    _check_min(
        overall_failures,
        "pass_ratio",
        pass_ratio,
        gates.get("min_pass_ratio"),
    )
    _check_min(
        overall_failures,
        "overall_total_pnl_usd",
        total_pnl,
        gates.get("min_total_pnl_usd_overall"),
    )
    _check_min(
        overall_failures,
        "overall_long_share",
        overall_direction.get("long_share"),
        gates.get("min_long_share_overall"),
    )
    _check_max(
        overall_failures,
        "overall_long_share",
        overall_direction.get("long_share"),
        gates.get("max_long_share_overall"),
    )
    _check_min(
        overall_failures,
        "overall_short_share",
        overall_direction.get("short_share"),
        gates.get("min_short_share_overall"),
    )
    _check_max(
        overall_failures,
        "overall_short_share",
        overall_direction.get("short_share"),
        gates.get("max_short_share_overall"),
    )

    overall_best_day_fraction: float | None = None
    if total_pnl > 0 and max_best_day_usd > float("-inf"):
        overall_best_day_fraction = max_best_day_usd / total_pnl
    _check_max(
        overall_failures,
        "overall_best_day_fraction",
        overall_best_day_fraction,
        gates.get("max_best_day_fraction_overall"),
    )

    # P0.3: Deflated Sharpe on aggregated daily PnL.
    dsr_summary: dict | None = None
    dsr_value: float | None = None
    if gates.get("min_dsr") is not None:
        # n_trials is conservative when not explicitly tracked. 10 matches
        # Bailey & López de Prado's "moderate search" default; sweeps should
        # override via gates["dsr_n_trials"].
        n_trials = int(gates.get("dsr_n_trials") or 10)
        dsr_summary = _compute_overall_dsr(evaluated_windows, n_trials=n_trials)
        if dsr_summary is not None:
            dsr_value = _as_float(dsr_summary.get("dsr"))
        _check_min(overall_failures, "overall_dsr", dsr_value, gates.get("min_dsr"))

    passed = len(overall_failures) == 0
    return {
        "passed": passed,
        "passed_windows": passed_count,
        "total_windows": total_windows,
        "pass_ratio": pass_ratio,
        "overall_total_pnl_usd": total_pnl,
        "overall_total_trades": total_trades,
        "overall_direction_summary": overall_direction,
        "overall_best_day_fraction": overall_best_day_fraction,
        "overall_max_best_day_usd": (
            max_best_day_usd if max_best_day_usd > float("-inf") else None
        ),
        "overall_dsr": dsr_value,
        "overall_dsr_detail": dsr_summary,
        "overall_failures": overall_failures,
        "windows": evaluated_windows,
    }
