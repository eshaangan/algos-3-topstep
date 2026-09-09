"""Tests for the alpha/beta attribution screen."""

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.analysis.alpha_beta import (
    newey_west_lag,
    ols_hac,
    daily_pnl_from_trades,
    daily_market_move,
    alpha_beta_screen,
    format_alpha_beta_report,
)


# ---------------------------------------------------------------- OLS / HAC

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = 3.0 + 2.5 * x + rng.normal(scale=0.1, size=n)
    fit = ols_hac(y, np.column_stack([np.ones(n), x]), ["a", "b"])
    assert fit.get("a")["coef"] == pytest.approx(3.0, abs=0.05)
    assert fit.get("b")["coef"] == pytest.approx(2.5, abs=0.05)
    assert fit.r_squared > 0.99


def test_hac_se_exceeds_ols_se_under_serial_correlation():
    """Positively autocorrelated residuals must inflate the HAC standard error."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    y = 1.0 + 0.0 * x + e
    X = np.column_stack([np.ones(n), x])

    hac = ols_hac(y, X, ["a", "b"], hac_lags=12)
    plain = ols_hac(y, X, ["a", "b"], hac_lags=0)
    assert hac.get("a")["se"] > plain.get("a")["se"]


def test_newey_west_lag_rule():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1) == 0
    assert newey_west_lag(0) == 0


def test_ols_validates_shapes():
    with pytest.raises(ValueError):
        ols_hac(np.zeros(5), np.zeros((4, 2)), ["a", "b"])
    with pytest.raises(ValueError):
        ols_hac(np.zeros(5), np.zeros((5, 2)), ["a"])
    with pytest.raises(ValueError):
        # more parameters than observations
        ols_hac(np.zeros(2), np.zeros((2, 3)), ["a", "b", "c"])


# ------------------------------------------------------------- aggregation

def test_daily_pnl_sums_trades_per_day():
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime([
            "2026-01-02 14:35", "2026-01-02 15:10", "2026-01-05 14:40",
        ], utc=True),
        "pnl": [100.0, -40.0, 25.0],
    })
    daily = daily_pnl_from_trades(trades)
    assert len(daily) == 2
    assert daily.iloc[0] == pytest.approx(60.0)
    assert daily.iloc[1] == pytest.approx(25.0)


def test_daily_market_move_is_intraday_not_gapped():
    """The overnight gap must be excluded: intraday strategies are flat then."""
    idx = pd.to_datetime([
        "2026-01-02 14:30", "2026-01-02 20:00",
        "2026-01-05 14:30", "2026-01-05 20:00",
    ], utc=True)
    bars = pd.DataFrame({"close": [100.0, 110.0, 200.0, 205.0]}, index=idx)
    mv = daily_market_move(bars)
    # Day 1 moves +10 intraday; day 2 moves +5. The +90 overnight gap is ignored.
    assert mv.iloc[0] == pytest.approx(10.0)
    assert mv.iloc[1] == pytest.approx(5.0)


def test_missing_columns_raise():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        daily_pnl_from_trades(df)
    with pytest.raises(KeyError):
        daily_market_move(df)


# ------------------------------------------------------------------ screen

def _dates(n):
    return pd.bdate_range("2024-01-01", periods=n).date


def test_pure_beta_strategy_shows_no_alpha():
    """P/L that is exactly 2 contracts of market move must yield alpha ~ 0."""
    rng = np.random.default_rng(2)
    n = 300
    d = _dates(n)
    mkt = pd.Series(rng.normal(scale=10.0, size=n), index=d)
    # 2 MES contracts plus execution noise (a noiseless fit collapses the
    # standard error and makes any residual intercept spuriously significant).
    pnl = pd.Series(
        2 * 5.0 * mkt.to_numpy() + rng.normal(scale=25.0, size=n), index=d
    )

    res = alpha_beta_screen(pnl, mkt, point_value=5.0, include_flat_days=False)
    assert res.directional["terms"]["alpha"]["p"] > 0.05
    assert res.beta_contracts_equiv == pytest.approx(2.0, abs=0.01)
    assert res.verdict == "BETA_ONLY"


def test_true_alpha_survives_both_specs():
    """Constant per-day edge, uncorrelated with the market, must be detected."""
    rng = np.random.default_rng(3)
    n = 300
    d = _dates(n)
    mkt = pd.Series(rng.normal(scale=10.0, size=n), index=d)
    pnl = pd.Series(50.0 + rng.normal(scale=5.0, size=n), index=d)

    res = alpha_beta_screen(pnl, mkt, point_value=5.0, include_flat_days=False)
    assert res.verdict == "ALPHA_SURVIVES"
    assert res.directional["terms"]["alpha"]["coef"] == pytest.approx(50.0, abs=2.0)
    assert res.directional["terms"]["beta"]["p"] > 0.05


def test_convexity_strategy_is_not_called_alpha():
    """
    A strategy paid for range (|move|) with no directional or residual edge must
    be flagged CONVEXITY_NOT_ALPHA, not as genuine alpha.
    """
    rng = np.random.default_rng(4)
    n = 400
    d = _dates(n)
    mkt = pd.Series(rng.normal(scale=10.0, size=n), index=d)
    pnl = pd.Series(3.0 * np.abs(mkt.to_numpy()) + rng.normal(scale=1.0, size=n), index=d)

    res = alpha_beta_screen(pnl, mkt, point_value=5.0, include_flat_days=False)
    assert res.verdict == "CONVEXITY_NOT_ALPHA"
    assert res.with_convexity["terms"]["gamma_abs"]["p"] < 0.05


def test_flat_day_zero_fill_changes_the_sample():
    rng = np.random.default_rng(5)
    n = 200
    d = _dates(n)
    mkt = pd.Series(rng.normal(scale=10.0, size=n), index=d)
    traded = d[::4]
    pnl = pd.Series(rng.normal(loc=20.0, scale=5.0, size=len(traded)), index=traded)

    filled = alpha_beta_screen(pnl, mkt, include_flat_days=True)
    traded_only = alpha_beta_screen(pnl, mkt, include_flat_days=False)
    assert filled.n_days > traded_only.n_days
    # Total P/L is identical; only the day count changes.
    assert filled.total_pnl == pytest.approx(traded_only.total_pnl)
    assert filled.mean_daily_pnl < traded_only.mean_daily_pnl


def test_too_few_days_raises():
    d = _dates(5)
    pnl = pd.Series([1.0] * 5, index=d)
    mkt = pd.Series([1.0] * 5, index=d)
    with pytest.raises(ValueError, match="aligned days"):
        alpha_beta_screen(pnl, mkt, include_flat_days=False)


def test_report_renders():
    rng = np.random.default_rng(6)
    n = 120
    d = _dates(n)
    mkt = pd.Series(rng.normal(scale=10.0, size=n), index=d)
    pnl = pd.Series(rng.normal(loc=10.0, scale=20.0, size=n), index=d)
    res = alpha_beta_screen(pnl, mkt, point_value=5.0, include_flat_days=False)
    text = format_alpha_beta_report(res, title="unit test")
    assert "VERDICT" in text and "Spec 2" in text
