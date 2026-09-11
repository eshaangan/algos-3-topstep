"""Tests for the prop-firm barrier screen.

The load-bearing test is `test_calibrates_to_the_measured_lucid_result`. The whole
model is an arithmetic shortcut for a simulation that took 1,685 contiguous sessions
to run; if the shortcut stops reproducing that simulation, the shortcut is wrong.
"""
from __future__ import annotations

import pytest

from rule_based_v1.propscreen import (Breach, Drawdown, MECHANICS, PLANS, Plan,
                                      rank, score, table)


def plan(**kw) -> Plan:
    base = dict(firm="F", plan="P", size=100_000, target=6_000, drawdown=3_000,
                dd_type=Drawdown.STATIC, breach=Breach.LIVE, fee=158.0)
    return Plan(**{**base, **kw})


# ------------------------------------------------------------- calibration --
def test_calibrates_to_the_measured_lucid_result():
    s = score(plan(dd_type=Drawdown.TRAILING_EOD, breach=Breach.LIVE))
    assert s.pass_prob == pytest.approx(0.228, abs=0.005)   # simulated 22.8%
    assert s.ev == pytest.approx(20.0, abs=15.0)            # simulated +$20


def test_close_only_breach_reproduces_the_other_measured_cell():
    s = score(plan(dd_type=Drawdown.TRAILING_EOD, breach=Breach.CLOSE))
    assert s.pass_prob == pytest.approx(0.36, abs=0.01)     # simulated 36.1%
    assert s.ev > 500                                        # simulated +$755 at 60


# ----------------------------------------------------------------- geometry --
def test_static_pass_probability_is_exactly_the_barrier_ratio():
    assert score(plan()).pass_prob == pytest.approx(3_000 / 9_000)
    assert score(plan(target=3_000)).pass_prob == pytest.approx(0.5)


def test_pass_probability_is_invariant_to_account_size():
    a = score(plan(size=50_000, target=3_000, drawdown=1_500))
    b = score(plan(size=150_000, target=9_000, drawdown=4_500))
    assert a.pass_prob == pytest.approx(b.pass_prob)


def test_a_static_floor_beats_a_trailing_one_at_identical_geometry():
    st = score(plan(dd_type=Drawdown.STATIC))
    tr = score(plan(dd_type=Drawdown.TRAILING_EOD))
    assert st.pass_prob > tr.pass_prob
    assert st.ev > tr.ev * 10          # the difference is an order of magnitude


def test_ev_scales_with_the_drawdown_allowance_not_the_account_size():
    """With the industry-standard T = 2D, gross EV reduces to s*D/3."""
    s = score(plan(drawdown=3_000, target=6_000, fee=1.0))
    assert s.ev + 1.0 == pytest.approx(0.9 * 3_000 * 0.75 / 3, rel=0.01)


# ------------------------------------------------------------------ ranking --
def test_rank_orders_by_ev_per_dollar_of_fee():
    cheap = plan(firm="cheap", fee=50.0, drawdown=3_000)
    dear = plan(firm="dear", fee=500.0, drawdown=3_000)
    assert [s.plan.firm for s in rank([dear, cheap])] == ["cheap", "dear"]


def test_a_negative_ev_plan_is_rejected():
    s = score(plan(dd_type=Drawdown.TRAILING_EOD, breach=Breach.LIVE, fee=2_000.0))
    assert s.ev < 0 and s.verdict == "REJECT"


def test_trailing_plus_live_is_only_ever_marginal():
    """The combination that killed Lucid must never reach the shortlist."""
    s = score(plan(dd_type=Drawdown.TRAILING_EOD, breach=Breach.LIVE, fee=10.0))
    assert s.ev > 0 and s.verdict == "MARGINAL"


def test_discounted_price_is_what_gets_charged():
    p = plan(fee=225.0, fee_discounted=158.0)
    assert p.price == 158.0 and score(p).ev == pytest.approx(
        score(plan(fee=158.0)).ev)


# --------------------------------------------------------------- validation --
@pytest.mark.parametrize("bad", [{"target": 0}, {"drawdown": -1}, {"fee": 0},
                                 {"split": 0}, {"split": 1.5}])
def test_nonsense_plans_are_refused(bad):
    with pytest.raises(ValueError):
        plan(**bad)


def test_efficiency_override_is_bounded():
    with pytest.raises(ValueError):
        score(plan(), efficiency=0)


# ------------------------------------------------------------- seed hygiene --
def test_every_seed_plan_cites_a_source_and_a_confidence_tag():
    for p in PLANS:
        assert p.source.startswith("http"), p
        assert p.verified is not None, p
        assert "CONFIDENCE=" in p.notes, p


def test_no_plan_claims_close_only_breach_without_source_confidence():
    """Assuming a close-only breach is the single most expensive mistake here:
    it is worth 1.08x vs 0.68x on pass and 1.18x vs 0.29x on extraction."""
    for p in PLANS:
        if p.breach is Breach.CLOSE:
            assert "CONFIDENCE=SOURCE" in p.notes, (
                f"{p.firm}/{p.plan} assumes close-only breach without confirming it")


def test_mechanics_table_covers_every_combination():
    for d in Drawdown:
        for b in Breach:
            assert (d, b) in MECHANICS


def test_table_renders_every_plan():
    out = table(rank(PLANS))
    assert len(out.splitlines()) == len(PLANS) + 2      # header + rule + one per plan
    for p in PLANS:
        assert p.firm[:16] in out
