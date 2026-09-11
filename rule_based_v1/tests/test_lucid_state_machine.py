"""Tests for the LucidFlex account state machine.

The rule that matters most is the one that costs money silently: requesting a
payout while the Max Loss Limit is still trailing relocks it upward on the spot.
`test_payout_trap_*` pin that behaviour down.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from rule_based_v1.lucid import LucidAccount, Phase, Portfolio, RuleViolation, SPECS
from rule_based_v1.lucid.rules import MIN_PAYOUT, PROFIT_SPLIT

D0 = date(2026, 9, 1)


def days(n: int) -> date:
    return D0 + timedelta(days=n)


def acct(size: int = 100_000, **kw) -> LucidAccount:
    return LucidAccount("A1", size, **kw)


def run(a: LucidAccount, pnls, start: int = 0):
    return [a.close_session(days(start + i), p) for i, p in enumerate(pnls)]


# ------------------------------------------------------------------- specs --
@pytest.mark.parametrize("size", sorted(SPECS))
def test_spec_table_matches_the_published_arithmetic(size):
    s = SPECS[size]
    assert s.locked_mll == size + 100
    assert s.trail_balance == size + s.mll_amount + 100
    assert s.trail_balance - s.mll_amount == s.locked_mll


def test_fresh_account_starts_at_the_initial_mll():
    a = acct()
    assert a.balance == 100_000 and a.mll == 97_000 and a.buffer == 3_000
    assert a.max_size("micros") == 60 and a.max_size("minis") == 6


# --------------------------------------------------------------------- MLL --
def test_mll_trails_the_highest_closing_balance():
    a = acct()
    run(a, [1_000])
    assert a.mll == 98_000
    run(a, [-500], start=1)          # a losing day must not lower the MLL
    assert a.mll == 98_000


def test_mll_never_exceeds_the_locked_level():
    a = acct()
    run(a, [10_000])
    assert a.mll == 100_100


def test_mll_locks_permanently_once_the_trail_balance_is_cleared():
    a = acct()
    run(a, [3_100])                  # balance 103,100 == trail balance
    assert a.mll_locked and a.mll == 100_100
    run(a, [-2_000], start=1)        # falling back must not unlock it
    assert a.mll_locked and a.mll == 100_100


def test_breach_is_evaluated_on_the_closing_balance():
    a = acct()
    r = run(a, [-3_000])[0]
    assert a.phase is Phase.BREACHED and "BREACHED" in r.note
    with pytest.raises(RuleViolation, match="breached"):
        a.close_session(days(1), 100.0)


def test_a_close_just_above_the_mll_survives():
    a = acct()
    run(a, [-2_999.99])
    assert a.phase is Phase.EVAL and a.buffer == pytest.approx(0.01)


# -------------------------------------------------------------- evaluation --
def test_evaluation_passes_on_target_with_two_even_days():
    a = acct()
    run(a, [3_000, 3_000])
    assert a.phase is Phase.FUNDED and a.cycle_start_balance == 100_000


def test_consistency_rule_blocks_a_lopsided_pass():
    """Two trading days, target met, but one day carries 98% of the profit."""
    a = acct()
    r = run(a, [5_900, 100])[-1]
    assert a.phase is Phase.EVAL and "consistency" in r.note
    assert "12,200" not in r.note and "11,800" in r.note


def test_minimum_two_trading_days_is_enforced_before_consistency():
    a = acct()
    r = run(a, [7_000])[0]           # 7,000 > 2x best day is impossible in one day
    assert a.phase is Phase.EVAL and "trading day" in r.note


def test_a_later_even_day_completes_a_previously_blocked_pass():
    a = acct()
    run(a, [6_000])                  # target hit, but only one trading day
    assert a.phase is Phase.EVAL
    run(a, [6_000], start=1)         # best day 6,000 is exactly 50% of 12,000
    assert a.phase is Phase.FUNDED


# ---------------------------------------------------------- qualifying days --
def test_qualifying_days_only_count_in_the_funded_phase():
    a = acct()
    run(a, [500, 500])               # both >= $200 but this is the evaluation
    assert a.qualifying_days == 0


def test_qualifying_days_need_the_minimum_daily_profit():
    a = acct(phase=Phase.FUNDED)
    run(a, [199.99, 200.0])
    assert a.qualifying_days == 1


# ------------------------------------------------------------ payout advice --
def test_payout_ineligible_without_five_qualifying_days():
    a = acct(phase=Phase.FUNDED)
    run(a, [300] * 4)
    adv = a.payout_advice()
    assert not adv.eligible and "4/5 qualifying days" in " ".join(adv.reasons)


def test_payout_trap_advises_waiting_and_prices_the_buffer_loss():
    """The headline rule. Below the trail balance a request costs real buffer."""
    a = acct(phase=Phase.FUNDED)
    run(a, [400] * 5)                # balance 102,000, peak 102,000, MLL 99,000
    adv = a.payout_advice()
    assert adv.eligible and adv.recommendation == "wait"
    assert adv.mll_jump_cost == pytest.approx(1_100.0)   # 100,100 - 99,000
    assert "destroys" in " ".join(adv.reasons)


def test_payout_costs_nothing_once_the_mll_has_locked():
    a = acct(phase=Phase.FUNDED)
    run(a, [700] * 5)                # balance 103,500 > trail, MLL locked
    adv = a.payout_advice()
    assert adv.eligible and adv.recommendation == "request"
    assert adv.mll_jump_cost == 0.0


def test_request_is_refused_while_the_advice_says_wait():
    a = acct(phase=Phase.FUNDED)
    run(a, [400] * 5)
    with pytest.raises(RuleViolation, match="refusing to request"):
        a.request_payout()
    cash = a.request_payout(override_wait=True)
    assert cash == pytest.approx(PROFIT_SPLIT * 1_000.0)


def test_payout_below_the_five_hundred_minimum_is_ineligible():
    a = acct(phase=Phase.FUNDED)
    run(a, [200] * 5)                # profit 1,000 -> 50% is 500, exactly the min
    assert a.payout_advice().amount == pytest.approx(MIN_PAYOUT)
    b = acct(phase=Phase.FUNDED)
    run(b, [200, 200, 200, 200, 199.99 + 200])
    b.balance -= 300                 # nudge profit under 1,000
    adv = b.payout_advice()
    assert not adv.eligible and "below the" in " ".join(adv.reasons)


# ---------------------------------------------------------- payout effects --
def test_requesting_a_payout_locks_the_mll_and_resets_the_cycle():
    a = acct(phase=Phase.FUNDED)
    run(a, [400] * 5)
    assert not a.mll_locked
    cash = a.request_payout(override_wait=True)
    assert a.mll_locked and a.mll == 100_100
    assert a.balance == pytest.approx(101_000.0)
    assert cash == pytest.approx(900.0) and a.cash_received == pytest.approx(900.0)
    assert a.qualifying_days == 0 and a.cycle_start_balance == a.balance


def test_the_patient_policy_keeps_more_buffer_than_the_greedy_one():
    """Same P&L, two policies. This is the whole reason the module exists."""
    greedy = acct(phase=Phase.FUNDED)
    run(greedy, [400] * 5)
    greedy.request_payout(override_wait=True)

    patient = acct(phase=Phase.FUNDED)
    run(patient, [400] * 5)
    run(patient, [400] * 3, start=5)          # carry on to 103,200, MLL locks
    patient.request_payout()

    assert patient.buffer > greedy.buffer
    assert greedy.buffer == pytest.approx(900.0)


def test_five_payouts_moves_the_account_live():
    a = acct(phase=Phase.FUNDED)
    for k in range(5):
        run(a, [700] * 5, start=5 * k)
        a.request_payout()
    assert a.phase is Phase.LIVE and a.payouts_taken == 5
    with pytest.raises(RuleViolation, match="live"):
        a.close_session(days(99), 100.0)


# ------------------------------------------------------------------ guards --
def test_trading_is_blocked_while_a_payout_is_pending():
    a = acct(phase=Phase.FUNDED)
    a.payout_pending = True
    with pytest.raises(RuleViolation, match="pending"):
        a.close_session(days(1), 100.0)


def test_sessions_must_be_recorded_in_order():
    a = acct()
    run(a, [100])
    with pytest.raises(RuleViolation, match="not after"):
        a.close_session(D0, 100.0)


def test_inactivity_countdown_tracks_the_thirty_day_policy():
    a = acct()
    run(a, [100])
    assert a.days_until_deletion(days(0)) == 30
    assert a.days_until_deletion(days(28)) == 2


# ---------------------------------------------------------------- intraday --
def test_intraday_status_separates_the_two_readings_of_the_breach_rule():
    a = acct()
    s = a.intraday_status(90_000.0)          # $7,000 below the MLL intraday
    assert s["breaches_if_intraday_rule"] is True
    assert s["breaches_if_closing_rule"] is False
    assert s["closing_balance_needed"] == pytest.approx(97_000.01)


# --------------------------------------------------------------- portfolio --
def test_portfolio_round_trips_through_json(tmp_path):
    pf = Portfolio()
    a = pf.add(acct(), fee=158.0)
    run(a, [400] * 5)
    p = pf.save(tmp_path / "state.json")
    back = Portfolio.load(p)
    b = back.get("A1")
    assert b.balance == a.balance and b.mll == a.mll and b.phase is a.phase
    assert back.fees_paid == 158.0 and back.net == pytest.approx(-158.0)


def test_portfolio_alerts_surface_the_hold_and_the_thin_buffer():
    pf = Portfolio()
    a = pf.add(acct(phase=Phase.FUNDED))
    run(a, [400] * 5)
    assert any("HOLD" in x for x in pf.alerts(days(5)))
    b = pf.add(LucidAccount("A2", 100_000))
    run(b, [-2_500])
    assert any("THIN BUFFER" in x for x in pf.alerts(days(5)))


def test_duplicate_account_ids_are_rejected():
    pf = Portfolio()
    pf.add(acct())
    with pytest.raises(ValueError, match="duplicate"):
        pf.add(acct())


# ------------------------------------------------- evaluation -> funded reset --
def test_passing_opens_a_fresh_funded_account_at_the_size():
    """The funded account does not inherit the evaluation's ending balance."""
    a = acct()
    r = run(a, [3_000, 3_000])[-1]
    assert a.phase is Phase.FUNDED
    assert a.balance == 100_000 and a.peak_balance == 100_000
    assert a.mll == 97_000 and not a.mll_locked
    assert a.cycle_start_balance == 100_000 and a.qualifying_days == 0
    assert "opens fresh" in r.note


def test_the_payout_trap_is_reachable_only_after_a_real_pass():
    """End to end: pass, bank five small days, and get told to hold."""
    a = acct()
    run(a, [3_000, 3_000])
    run(a, [400] * 5, start=2)               # funded balance 102,000, MLL 99,000
    adv = a.payout_advice()
    assert adv.eligible and adv.recommendation == "wait"
    assert adv.mll_jump_cost == pytest.approx(1_100.0)
    run(a, [400] * 3, start=7)               # clears 103,100, MLL locks
    assert a.payout_advice().recommendation == "request"
