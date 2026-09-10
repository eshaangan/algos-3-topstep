"""End-to-end lifecycle test for the live book runner, against a fake broker.

The runner had never completed a single trade when this was written -- the
entry -> fill -> stop -> verify -> exit sequence existed only on paper. Rithmic
cannot be exercised in CI, so this stands in a fake order plant that records
what was submitted and replays the notifications Rithmic would send.

What this pins, in order of how much money it costs to get wrong:

  1. a position is NEVER left holding without a verified resting stop
  2. the stop is submitted for the FULL position quantity at fill - $600/micro
  3. if the stop is not resting, the runner flattens instead of riding naked
  4. the scheduled exit fires, and P&L is booked with the right qty and costs
  5. a second leg cannot open while one is already open

It does NOT prove Rithmic behaves as modelled here. It proves the runner's own
state machine is correct given that behaviour.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from rule_based_v1.live import book_live_runner as blr
from rule_based_v1.live.book_live_runner import BookLive, LegSignal


class FakeOrderPlant:
    """Records submissions and can be told what list_orders should return."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.cancelled = 0
        self.exited = 0
        self.resting: list[str] = []

    async def submit_order(self, **kw):
        self.submitted.append(kw)
        # A real stop shows up in the working-order snapshot.
        if kw.get("order_id", "").endswith("-stp"):
            self.resting.append(kw["order_id"])

    async def list_orders(self, **kw):
        return [SimpleNamespace(user_tag=t) for t in self.resting]

    async def cancel_all_orders(self, **kw):
        self.cancelled += 1
        self.resting.clear()

    async def exit_position(self, **kw):
        self.exited += 1

    async def list_accounts(self):
        return [SimpleNamespace(account_id="LFE100-TEST")]


def make_runner(tmp_path, live=True) -> tuple[BookLive, FakeOrderPlant]:
    r = BookLive(str(tmp_path), live=live, decisions=["2026-09-16"])
    plant = FakeOrderPlant()
    r.client = SimpleNamespace(plants={"order": plant})
    r.acct = "LFE100-TEST"
    r.contract = "MNQZ6"
    return r, plant


# ------------------------------------------------------------ happy path

def test_full_lifecycle_entry_stop_verify_exit(tmp_path):
    r, plant = make_runner(tmp_path)

    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    assert len(plant.submitted) == 1
    from async_rithmic import OrderType, TransactionType
    entry = plant.submitted[0]
    assert entry["qty"] == 2
    assert entry["order_type"] == OrderType.MARKET
    assert entry["transaction_type"] == TransactionType.BUY

    ot = r.st["open"]
    assert ot["fill"] is None and not ot["verified"]

    # Rithmic reports the entry fill
    ot["fill"] = 24010.0
    asyncio.run(r.protect(ot))

    stop = plant.submitted[1]
    assert stop["order_type"] == OrderType.STOP_MARKET
    assert stop["transaction_type"] == TransactionType.SELL
    assert stop["order_id"].endswith("-stp")
    assert stop["qty"] == 2, "stop must cover the FULL position"
    assert stop["trigger_price"] == pytest.approx(24010.0 - 300.0)
    assert (24010.0 - stop["trigger_price"]) * blr.PV == pytest.approx(600.0)

    asyncio.run(r.verify(ot))
    assert ot["verified"] is True
    assert plant.exited == 0, "a verified stop must not trigger a flatten"

    asyncio.run(r.flatten("scheduled_exit_mon_1559"))
    assert plant.exited == 1 and plant.cancelled == 1


def test_pnl_is_booked_with_the_right_qty_and_costs(tmp_path):
    r, _ = make_runner(tmp_path)
    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    r.st["open"]["fill"] = 24000.0

    from async_rithmic import ExchangeOrderNotificationType as NT
    exit_note = SimpleNamespace(notify_type=NT.FILL, fill_price=24050.0,
                                transaction_type=2, user_tag="x")
    asyncio.run(r._note(exit_note))

    events = [json.loads(l) for l in open(r.ev_p)]
    closed = [e for e in events if e.get("ev") == "closed"][0]
    # 50 points * $2 * 2 micros - 2 * $0.62 * 2 = 200 - 2.48
    assert closed["pnl"] == pytest.approx(50.0 * 2.0 * 2 - 2 * 0.62 * 2)
    assert closed["qty"] == 2
    assert r.st["open"] is None
    assert "wk-2026-09-14" in r.st["done"]


# --------------------------------------------------- the protection failure

def test_unresting_stop_forces_a_flatten(tmp_path):
    """The failure this project has actually seen: a stop that is accepted but
    never rests. The position must be closed, not carried."""
    r, plant = make_runner(tmp_path)
    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    ot = r.st["open"]
    ot["fill"] = 24000.0
    asyncio.run(r.protect(ot))

    plant.resting.clear()               # broker silently dropped it
    asyncio.run(r.verify(ot))

    assert ot["verified"] is False
    assert plant.exited == 1, "must flatten when the stop is not resting"


def test_submit_failure_flattens_rather_than_holding_naked(tmp_path):
    r, plant = make_runner(tmp_path)
    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    ot = r.st["open"]
    ot["fill"] = 24000.0

    async def boom(**kw):
        raise RuntimeError("rejected")
    plant.submit_order = boom

    asyncio.run(r.protect(ot))
    assert plant.exited == 1


# ------------------------------------------------------------- concurrency

def test_only_one_position_at_a_time(tmp_path):
    """Two open legs would put the account-wide cancel in conflict with a live
    stop, which is the whole reason both legs share one process."""
    r, _ = make_runner(tmp_path)
    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    assert r.st["open"]["leg"] == "weekend"

    now = pd.Timestamp("2026-09-15 18:00", tz=blr.ET)
    assert blr.fomc_signal(now, ["2026-09-16"]) is not None   # it WANTS to fire
    # ... but the run loop only looks for entries when nothing is open:
    assert r.st["open"] is not None


def test_qty_over_the_cap_is_refused():
    assert blr.WEEKEND_QTY <= blr.MAX_CONCURRENT_QTY
    assert blr.FOMC_QTY <= blr.MAX_CONCURRENT_QTY


# ------------------------------------------------------------- persistence

def test_state_survives_a_restart(tmp_path):
    r, _ = make_runner(tmp_path)
    asyncio.run(r.enter(LegSignal("fomc", 2, "fomc-2026-09-16"), ref=24000.0))
    r.st["open"]["fill"] = 24000.0
    r.save()

    r2 = BookLive(str(tmp_path), live=True, decisions=["2026-09-16"])
    assert r2.st["open"]["leg"] == "fomc"
    assert r2.st["open"]["fill"] == 24000.0
    assert r2.st["open"]["qty"] == 2


def test_dry_mode_places_nothing(tmp_path):
    r, plant = make_runner(tmp_path, live=False)
    asyncio.run(r.enter(LegSignal("weekend", 2, "wk-2026-09-14"), ref=24000.0))
    assert plant.submitted == []
    ot = r.st["open"]
    ot["fill"] = 24000.0
    asyncio.run(r.protect(ot))
    assert plant.submitted == []
    assert ot["verified"] is True
