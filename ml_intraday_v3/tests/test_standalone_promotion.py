from ml_intraday_v3.analysis.standalone_promotion import (
    evaluate_promotion_gates,
    evaluate_window_gates,
    summarize_trade_directions,
)


def test_summarize_trade_directions_handles_empty():
    summary = summarize_trade_directions(0, 0)
    assert summary["total_trades"] == 0
    assert summary["long_share"] is None
    assert summary["short_share"] is None


def test_evaluate_window_gates_passes_balanced_profitable_window():
    window = {
        "name": "jan_2026",
        "train_events": 600,
        "test_events": 80,
        "classification": {
            "test_auc": 0.56,
            "test_accuracy": 0.53,
            "positive_prediction_rate": 0.32,
        },
        "backtest_metrics": {
            "total_pnl_usd": 420.0,
            "avg_trade_usd": 14.0,
            "profit_factor": 1.24,
            "win_rate": 0.49,
            "trades_count": 32,
            "max_drawdown_usd": 650.0,
            "mtm_daily_loss_liquidations": 0,
            "mtm_trailing_dd_liquidations": 0,
        },
        "direction_summary": {
            "long_trades": 18,
            "short_trades": 14,
            "long_share": 18 / 32,
            "short_share": 14 / 32,
        },
    }
    gates = {
        "min_train_events": 300,
        "min_test_events": 30,
        "min_test_auc": 0.52,
        "min_total_pnl_usd": 0.0,
        "min_avg_trade_usd": 5.0,
        "min_profit_factor": 1.05,
        "min_win_rate": 0.45,
        "min_trades_count": 20,
        "min_positive_prediction_rate": 0.1,
        "max_drawdown_usd": 1200.0,
        "max_mtm_daily_liquidations": 0,
        "max_mtm_trailing_liquidations": 0,
        "min_long_share": 0.2,
        "max_long_share": 0.8,
        "min_short_share": 0.2,
        "max_short_share": 0.8,
    }

    result = evaluate_window_gates(window, gates)
    assert result["passed"] is True
    assert result["failures"] == []


def test_evaluate_window_gates_flags_directional_degeneracy():
    window = {
        "name": "bad_window",
        "train_events": 500,
        "test_events": 60,
        "classification": {"test_auc": 0.55, "test_accuracy": 0.52},
        "backtest_metrics": {
            "total_pnl_usd": 100.0,
            "avg_trade_usd": 5.0,
            "profit_factor": 1.1,
            "win_rate": 0.46,
            "trades_count": 30,
            "max_drawdown_usd": 700.0,
            "mtm_daily_loss_liquidations": 0,
            "mtm_trailing_dd_liquidations": 0,
        },
        "direction_summary": {
            "long_trades": 29,
            "short_trades": 1,
            "long_share": 29 / 30,
            "short_share": 1 / 30,
        },
    }
    gates = {
        "min_short_share": 0.2,
        "max_long_share": 0.8,
    }

    result = evaluate_window_gates(window, gates)
    assert result["passed"] is False
    metrics = {failure["metric"] for failure in result["failures"]}
    assert "short_share" in metrics
    assert "long_share" in metrics


def test_evaluate_window_gates_rejects_no_trade_candidate():
    window = {
        "name": "no_trade_window",
        "train_events": 500,
        "test_events": 60,
        "classification": {
            "test_auc": 0.55,
            "test_accuracy": 0.53,
            "positive_prediction_rate": 0.0,
        },
        "backtest_metrics": {
            "total_pnl_usd": 0.0,
            "avg_trade_usd": None,
            "profit_factor": None,
            "win_rate": None,
            "trades_count": 0,
            "max_drawdown_usd": None,
            "mtm_daily_loss_liquidations": 0,
            "mtm_trailing_dd_liquidations": 0,
        },
        "direction_summary": {
            "long_trades": 0,
            "short_trades": 0,
            "long_share": None,
            "short_share": None,
        },
    }
    gates = {
        "min_trades_count": 10,
        "min_positive_prediction_rate": 0.02,
    }

    result = evaluate_window_gates(window, gates)
    assert result["passed"] is False
    metrics = {failure["metric"] for failure in result["failures"]}
    assert "trades_count" in metrics
    assert "positive_prediction_rate" in metrics


def test_evaluate_promotion_gates_requires_all_windows():
    gates = {
        "min_short_share": 0.2,
        "max_long_share": 0.8,
        "require_all_windows": True,
        "min_pass_ratio": 1.0,
        "min_total_pnl_usd_overall": 500.0,
        "min_short_share_overall": 0.2,
        "max_long_share_overall": 0.8,
    }
    windows = [
        {
            "name": "pass_window",
            "backtest_metrics": {"total_pnl_usd": 350.0, "trades_count": 20},
            "direction_summary": {
                "long_trades": 10,
                "short_trades": 10,
                "long_share": 0.5,
                "short_share": 0.5,
            },
            "gate_result": {"passed": True},
            "classification": {},
        },
        {
            "name": "fail_window",
            "backtest_metrics": {"total_pnl_usd": -50.0, "trades_count": 12},
            "direction_summary": {
                "long_trades": 11,
                "short_trades": 1,
                "long_share": 11 / 12,
                "short_share": 1 / 12,
            },
            "classification": {},
        },
    ]

    result = evaluate_promotion_gates(windows, gates)
    assert result["passed"] is False
    assert result["passed_windows"] == 1
    overall_metrics = {failure["metric"] for failure in result["overall_failures"]}
    assert "all_windows" in overall_metrics


def _window(name, total_pnl, best_day, **extra):
    return {
        "name": name,
        "train_events": 600,
        "test_events": 80,
        "classification": {"test_auc": 0.56, "test_accuracy": 0.53},
        "backtest_metrics": {
            "total_pnl_usd": total_pnl,
            "avg_trade_usd": 14.0,
            "profit_factor": 1.24,
            "win_rate": 0.49,
            "trades_count": 32,
            "max_drawdown_usd": 650.0,
            "mtm_daily_loss_liquidations": 0,
            "mtm_trailing_dd_liquidations": 0,
        },
        "direction_summary": {
            "long_trades": 18,
            "short_trades": 14,
            "long_share": 18 / 32,
            "short_share": 14 / 32,
        },
        "risk_tearsheet": {"daily": {"best_day_usd": best_day}},
        **extra,
    }


def test_best_day_fraction_gate_catches_concentration_risk():
    """Topstep consistency rule: single day > 50% of PnL = reject (40% w/ margin)."""
    window = _window("concentrated", total_pnl=600.0, best_day=870.0)
    gates = {"max_best_day_fraction": 0.40}
    result = evaluate_window_gates(window, gates)
    assert result["passed"] is False
    assert result["best_day_fraction"] > 1.0
    assert any(f["metric"] == "best_day_fraction" for f in result["failures"])


def test_best_day_fraction_gate_passes_diversified_pnl():
    window = _window("diversified", total_pnl=1000.0, best_day=200.0)
    gates = {"max_best_day_fraction": 0.40}
    result = evaluate_window_gates(window, gates)
    assert result["passed"] is True
    assert result["best_day_fraction"] == 0.20


def test_best_day_fraction_skipped_when_window_unprofitable():
    """Undefined ratio: negative total PnL means the concentration check is
    moot — the window will already fail min_total_pnl_usd."""
    window = _window("losing", total_pnl=-500.0, best_day=200.0)
    gates = {"max_best_day_fraction": 0.40}
    result = evaluate_window_gates(window, gates)
    assert result["best_day_fraction"] is None
    assert not any(f["metric"] == "best_day_fraction" for f in result["failures"])


def _window_with_daily(name, daily_map, **backtest_overrides):
    total = sum(daily_map.values())
    bt = {
        "total_pnl_usd": total,
        "avg_trade_usd": 14.0,
        "profit_factor": 1.24,
        "win_rate": 0.49,
        "trades_count": 32,
        "max_drawdown_usd": 650.0,
        "mtm_daily_loss_liquidations": 0,
        "mtm_trailing_dd_liquidations": 0,
    }
    bt.update(backtest_overrides)
    return {
        "name": name,
        "train_events": 600,
        "test_events": 80,
        "classification": {"test_auc": 0.56, "test_accuracy": 0.53},
        "backtest_metrics": bt,
        "direction_summary": {
            "long_trades": 18, "short_trades": 14,
            "long_share": 18/32, "short_share": 14/32,
        },
        "risk_tearsheet": {
            "daily": {
                "best_day_usd": max(daily_map.values()) if daily_map else 0.0,
                "daily_pnl_by_day": daily_map,
            }
        },
    }


def test_dsr_gate_passes_on_consistent_winner():
    """Daily PnL with positive mean and low noise should deflate to DSR≈1.0."""
    import random
    random.seed(0)
    # 100 days, mean $50, small dispersion — strong real edge
    daily = {f"2024-01-{i:03d}": 50.0 + random.uniform(-20, 20) for i in range(1, 101)}
    window = _window_with_daily("strong", daily)
    gates = {"min_dsr": 0.95, "dsr_n_trials": 10}
    result = evaluate_promotion_gates([window], gates)
    assert result["overall_dsr"] is not None
    assert result["overall_dsr"] > 0.95
    assert not any(f["metric"] == "overall_dsr" for f in result["overall_failures"])


def test_dsr_gate_rejects_noisy_breakeven():
    """A weak edge under heavy selection bias must fail min_dsr=0.95.

    The deflation comes from n_trials: with 500 trials tested the
    Sharpe bar rises substantially, and a modest real Sharpe should no
    longer clear 0.95.
    """
    import random
    random.seed(1)
    # Mean=$8/day, std≈$280 → raw Sharpe ~0.45 — weak but non-zero edge.
    daily = {f"2024-01-{i:03d}": 8.0 + random.uniform(-280, 280) for i in range(1, 51)}
    window = _window_with_daily("weak_edge", daily)
    gates = {"min_dsr": 0.95, "dsr_n_trials": 500}
    result = evaluate_promotion_gates([window], gates)
    assert result["overall_dsr"] is not None
    assert result["overall_dsr"] < 0.95
    assert any(f["metric"] == "overall_dsr" for f in result["overall_failures"])


def test_dsr_not_computed_when_gate_omitted():
    """DSR computation is opt-in via gates — no min_dsr means no compute cost."""
    daily = {"2024-01-01": 50.0, "2024-01-02": -20.0}
    window = _window_with_daily("tiny", daily)
    result = evaluate_promotion_gates([window], {})
    assert result["overall_dsr"] is None
    assert result["overall_dsr_detail"] is None
