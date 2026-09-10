"""Tests for the live two-leg book runner's schedule and sizing logic.

These cover the parts that can be wrong while the runner still looks healthy:
entering on the wrong day, exiting late, mis-sizing, or computing a stop that
does not match the $600/micro the 73.4% P(pass) was scored with.

The broker path is not exercised here -- it needs Rithmic. What IS pinned is
every pure function that decides whether money moves.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rule_based_v1.live.book_live_runner import (
    FOMC_QTY, MAX_CONCURRENT_QTY, STOP_PTS, STOP_USD_PER_MICRO, WEEKEND_QTY,
    due_exit, fomc_signal, front_month, stop_price, weekend_signal,
)

ET = "America/New_York"
DECISIONS = ["2026-09-16", "2026-10-28", "2026-12-09"]


def t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=ET)


# ---------------------------------------------------------------- frozen spec

def test_frozen_config_matches_the_scored_monte_carlo():
    """If any of these change, the 73.4% no longer describes what is deployed."""
    assert STOP_USD_PER_MICRO == 600.0
    assert STOP_PTS == 300.0            # MNQ is $2/point
    assert WEEKEND_QTY == 2
    assert FOMC_QTY == 2
    assert MAX_CONCURRENT_QTY == 2


def test_stop_price_is_600_dollars_per_micro_below_the_fill():
    assert stop_price(20000.0) == pytest.approx(19700.0)
    # and the dollar loss at that stop is exactly $600 per micro
    assert (20000.0 - stop_price(20000.0)) * 2.0 == pytest.approx(600.0)


def test_stop_price_snaps_to_a_tradeable_tick():
    px = stop_price(20000.13)
    assert (px / 0.25) == pytest.approx(round(px / 0.25))


# ------------------------------------------------------------ weekend signal

def test_weekend_enters_sunday_at_1800_and_is_keyed_to_monday():
    sig = weekend_signal(t("2026-09-13 18:00"))      # a Sunday
    assert sig is not None
    assert sig.leg == "weekend"
    assert sig.qty == 2
    assert sig.key == "wk-2026-09-14"                # the Monday it exits


def test_weekend_entry_window_closes_after_fifteen_minutes():
    assert weekend_signal(t("2026-09-13 18:15")) is not None
    assert weekend_signal(t("2026-09-13 18:16")) is None
    assert weekend_signal(t("2026-09-13 17:59")) is None


def test_weekend_does_not_fire_on_other_days():
    for day in ("2026-09-14", "2026-09-15", "2026-09-16",
                "2026-09-17", "2026-09-18", "2026-09-19"):
        assert weekend_signal(t(f"{day} 18:00")) is None


# --------------------------------------------------------------- fomc signal

def test_fomc_enters_1800_the_day_before_a_decision():
    sig = fomc_signal(t("2026-09-15 18:00"), DECISIONS)   # decision is the 16th
    assert sig is not None
    assert sig.leg == "fomc"
    assert sig.qty == 2
    assert sig.key == "fomc-2026-09-16"


def test_fomc_does_not_fire_on_the_decision_day_itself():
    assert fomc_signal(t("2026-09-16 18:00"), DECISIONS) is None


def test_fomc_does_not_fire_on_unrelated_days():
    assert fomc_signal(t("2026-09-10 18:00"), DECISIONS) is None
    assert fomc_signal(t("2026-11-02 18:00"), DECISIONS) is None


def test_fomc_entry_is_after_the_1645_flatten():
    """The whole point of the 18:00 entry is that it is legal under Lucid's
    mandatory 16:45 flatten. An entry before it would be force-closed."""
    assert fomc_signal(t("2026-09-15 16:30"), DECISIONS) is None
    assert fomc_signal(t("2026-09-15 18:00"), DECISIONS) is not None


# ---------------------------------------------------------------- exit timing

def test_weekend_exits_monday_1559_not_before():
    ot = {"leg": "weekend", "key": "wk-2026-09-14"}
    assert due_exit(t("2026-09-14 15:58"), ot) is None
    assert due_exit(t("2026-09-14 15:59"), ot) == "scheduled_exit_mon_1559"


def test_weekend_exit_precedes_the_lucid_flatten():
    """15:59 must be strictly before 16:45, or the broker closes it for us."""
    ot = {"leg": "weekend", "key": "wk-2026-09-14"}
    assert due_exit(t("2026-09-14 16:00"), ot) == "scheduled_exit_mon_1559"


def test_fomc_exits_1355_before_the_1400_announcement():
    ot = {"leg": "fomc", "key": "fomc-2026-09-16"}
    assert due_exit(t("2026-09-16 13:54"), ot) is None
    assert due_exit(t("2026-09-16 13:55"), ot) == "scheduled_exit_fomc_1355"


def test_a_position_left_open_past_its_day_is_flagged_stale():
    """A missed exit must not quietly become a multi-day hold."""
    assert due_exit(t("2026-09-15 10:00"),
                    {"leg": "weekend", "key": "wk-2026-09-14"}) == "stale_weekend_position"
    assert due_exit(t("2026-09-17 10:00"),
                    {"leg": "fomc", "key": "fomc-2026-09-16"}) == "stale_fomc_position"


def test_no_open_position_means_no_exit():
    assert due_exit(t("2026-09-14 16:00"), None) is None


def test_weekend_held_over_sunday_night_is_not_stale():
    """Sunday 18:00 -> Monday is the intended hold, not a stale position."""
    ot = {"leg": "weekend", "key": "wk-2026-09-14"}
    assert due_exit(t("2026-09-13 20:00"), ot) is None
    assert due_exit(t("2026-09-14 09:30"), ot) is None


# -------------------------------------------------------------- contract roll

def test_front_month_rolls_about_eight_days_before_third_friday(monkeypatch):
    monkeypatch.delenv("FADE_CONTRACT", raising=False)
    # Sept 2026 third Friday is the 18th; roll ~8 days before = the 10th
    assert front_month(t("2026-09-01")) == "MNQU6"
    assert front_month(t("2026-09-15")) == "MNQZ6"
    assert front_month(t("2026-11-01")) == "MNQZ6"


def test_fade_contract_env_overrides_the_calculation(monkeypatch):
    monkeypatch.setenv("FADE_CONTRACT", "MNQH7")
    assert front_month(t("2026-09-01")) == "MNQH7"


# ------------------------------------------------- the legs never overlap

def test_the_two_legs_cannot_want_to_enter_at_the_same_instant():
    """Weekend fires only on Sunday; the earliest FOMC entry is a Monday.

    They are serialised in the runner regardless, but if they ever collided the
    account-wide cancel would put a live stop at risk, so this is worth pinning.
    """
    for d in pd.date_range("2026-09-01", "2027-12-31", freq="D"):
        now = pd.Timestamp(f"{d.date()} 18:00", tz=ET)
        both = weekend_signal(now), fomc_signal(now, DECISIONS)
        assert not (both[0] and both[1]), f"collision on {d.date()}"
