"""Tests for the P/L distribution stability screen."""

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.analysis.stability import (
    kl_divergence,
    js_divergence,
    stability_screen,
    format_stability_report,
)


# ------------------------------------------------------------- divergences

def test_kl_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_asymmetric_and_js_is_not():
    # Deliberately not a mirror image: a reversed pair has equal KL both ways.
    p = np.array([0.60, 0.30, 0.10])
    q = np.array([0.20, 0.15, 0.65])
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-9)
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-12)


def test_js_is_bounded_by_ln2():
    p = np.array([1.0 - 1e-9, 5e-10, 5e-10])
    q = np.array([5e-10, 5e-10, 1.0 - 1e-9])
    assert 0.0 <= js_divergence(p, q) <= np.log(2) + 1e-9


def test_kl_rejects_invalid_support():
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.3, 0.2]))


# ------------------------------------------------------------------ screen

def _trades(pnl, start="2024-01-01"):
    ts = pd.bdate_range(start, periods=len(pnl))
    return pd.DataFrame({"entry_time": ts, "pnl": pnl})


def test_stationary_pnl_is_called_stable():
    """IID P/L must not be flagged as drifting."""
    rng = np.random.default_rng(0)
    pnl = rng.normal(loc=20.0, scale=100.0, size=400)
    res = stability_screen(_trades(pnl), n_permutations=500, random_state=1)
    assert res.verdict == "STABLE"
    assert res.p_value > 0.05


def test_regime_shift_is_detected_as_unstable():
    """A strategy whose P/L distribution changes halfway must be flagged."""
    rng = np.random.default_rng(1)
    good = rng.normal(loc=150.0, scale=50.0, size=200)
    bad = rng.normal(loc=-150.0, scale=50.0, size=200)
    res = stability_screen(
        _trades(np.concatenate([good, bad])), n_blocks=2,
        n_permutations=500, random_state=1,
    )
    assert res.verdict == "UNSTABLE"
    assert res.p_value < 0.05
    assert res.first_last_js > res.null_q95_js


def test_variance_only_shift_is_detected():
    """Drift in dispersion alone (same mean) must still be caught."""
    rng = np.random.default_rng(2)
    calm = rng.normal(loc=0.0, scale=20.0, size=250)
    wild = rng.normal(loc=0.0, scale=300.0, size=250)
    res = stability_screen(
        _trades(np.concatenate([calm, wild])), n_blocks=2,
        n_permutations=500, random_state=2,
    )
    assert res.verdict == "UNSTABLE"


def test_permutation_null_controls_small_samples():
    """
    Small IID samples produce a visibly non-zero divergence; the null must
    absorb it rather than reporting spurious drift.
    """
    rng = np.random.default_rng(3)
    pnl = rng.normal(loc=10.0, scale=80.0, size=25)
    res = stability_screen(_trades(pnl), n_blocks=3, n_permutations=800, random_state=3)
    assert res.mean_pairwise_js > 0.0
    assert res.null_mean_js > 0.0
    assert res.p_value > 0.05
    assert any("small" in n for n in res.notes)


def test_lte_decomposition_localises_the_drift():
    """Block stats must show which component of E[P/L] moved."""
    rng = np.random.default_rng(4)
    # Same winner/loser sizes, collapsing hit rate.
    first = np.where(rng.random(200) < 0.7, 100.0, -100.0)
    last = np.where(rng.random(200) < 0.3, 100.0, -100.0)
    res = stability_screen(
        _trades(np.concatenate([first, last])), n_blocks=2,
        n_permutations=400, random_state=4,
    )
    assert res.block_stats[0]["p_win"] > res.block_stats[-1]["p_win"]
    assert any("hit rate" in n for n in res.notes)


def test_trades_are_sorted_chronologically():
    """Row order must not matter; the time column defines the blocks."""
    rng = np.random.default_rng(5)
    pnl = np.concatenate([
        rng.normal(200.0, 30.0, 150), rng.normal(-200.0, 30.0, 150),
    ])
    df = _trades(pnl)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = stability_screen(df, n_blocks=2, n_permutations=300, random_state=0)
    b = stability_screen(shuffled, n_blocks=2, n_permutations=300, random_state=0)
    assert a.mean_pairwise_js == pytest.approx(b.mean_pairwise_js, abs=1e-12)
    assert a.verdict == b.verdict == "UNSTABLE"


def test_constant_pnl_does_not_crash():
    """Degenerate quantile edges must be handled, not raise."""
    res = stability_screen(
        _trades(np.full(60, 25.0)), n_blocks=2, n_permutations=100, random_state=0
    )
    assert res.mean_pairwise_js == pytest.approx(0.0, abs=1e-9)
    assert res.verdict == "STABLE"


def test_too_few_trades_raises():
    with pytest.raises(ValueError, match="need >="):
        stability_screen(_trades(np.arange(8.0)), n_blocks=3)


def test_invalid_block_count_raises():
    with pytest.raises(ValueError, match="at least 2"):
        stability_screen(_trades(np.arange(60.0)), n_blocks=1)


def test_skipping_null_yields_unknown_verdict():
    res = stability_screen(_trades(np.arange(60.0)), n_permutations=0)
    assert res.verdict == "UNKNOWN"
    assert np.isnan(res.p_value)


def test_report_renders():
    rng = np.random.default_rng(6)
    res = stability_screen(
        _trades(rng.normal(10.0, 50.0, 120)), n_permutations=200, random_state=0
    )
    text = format_stability_report(res, title="unit test")
    assert "VERDICT" in text and "Law of Total Expectation" in text
