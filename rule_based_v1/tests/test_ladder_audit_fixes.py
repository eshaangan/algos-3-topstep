"""Regression tests for the 2026-09-06 pre-live audit of ladder_live_runner.

Every test here pins a defect that was demonstrated against the live-money runner
BEFORE it traded. Names map to the audit's finding ids.
"""
import asyncio, json, os, tempfile, time, types
from datetime import date, timedelta

import pandas as pd
import pytest

from rule_based_v1.live import ladder_live_runner as L


async def _aret_coro(v): return v
def _aret(v): return _aret_coro(v)


_REAL_SLEEP = asyncio.sleep      # captured before any test can patch it


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip real waits. Must restore the ORIGINAL: L.asyncio is the asyncio module
    itself, so a naive save/restore through it hands back the patched function and
    leaves asyncio.sleep broken for every later test."""
    monkeypatch.setattr(asyncio, "sleep", lambda s: _aret(None))
    yield
    monkeypatch.setattr(asyncio, "sleep", _REAL_SLEEP)


def _daemon(live=True, when="2026-07-13 16:00", out=None):
    d = L.Daemon(out or tempfile.mkdtemp(), live=live,
                 now_fn=lambda: pd.Timestamp(when, tz=L.ET),
                 quote_fn=lambda: (23000.0, 1.0))
    d.acct = "ACCT-A"; d.contract = "MNQZ6"
    return d


def _order(tag, sym="MNQZ6", side="2", qty=1, trg=22700.0, status="working"):
    return types.SimpleNamespace(user_tag=tag, symbol=sym, transaction_type=side,
                                 quantity=qty, trigger_price=trg, status=status)


# ── B1: the PnL subscription lives on the plant, not the client ──────────────────
def test_pnl_subscription_target_exists_on_plant_not_client():
    """The old call raised AttributeError, was swallowed, and left the daemon with no
    position feed and no balance -- blocking every entry and halting every flatten."""
    from async_rithmic import RithmicClient
    from async_rithmic.plants.pnl import PnlPlant
    assert not hasattr(RithmicClient, "subscribe_to_pnl_updates")
    assert hasattr(PnlPlant, "subscribe_to_pnl_updates")
    src = open(L.__file__).read()
    assert 'await self.client.plants["pnl"].subscribe_to_pnl_updates()' in src
    assert "await self.client.subscribe_to_pnl_updates" not in src


# ── B2: contract rollover ────────────────────────────────────────────────────────
@pytest.mark.parametrize("day,want", [
    ("2026-09-09", "MNQU6"),   # day before the roll
    ("2026-09-10", "MNQZ6"),   # roll day: 8 days before the 09-18 expiry
    ("2026-09-13", "MNQZ6"),   # the next tradeable weekend -- was hardcoded MNQU6
    ("2026-09-15", "MNQZ6"),   # the next FOMC entry -- was hardcoded MNQU6
    ("2026-12-10", "MNQH7"),
    ("2027-06-09", "MNQM7"),
])
def test_front_month_rolls_eight_days_before_expiry(day, want):
    assert L.front_month(date.fromisoformat(day))[0] == want


def test_stale_contract_override_is_refused():
    d = _daemon(when="2026-09-13 18:00")
    os.environ["LADDER_CONTRACT"] = "MNQU6"        # the old hardcoded default
    try:
        with pytest.raises(RuntimeError, match="not the front month"):
            d.resolve_contract()
    finally:
        del os.environ["LADDER_CONTRACT"]


def test_roll_block_refuses_a_window_ending_after_the_roll():
    d = _daemon(when="2026-12-09 18:00")
    d.roll_date = date(2026, 12, 10)
    d.acct_balance = 100000.0
    d._broker_cushion = lambda: _aret(3000.0)
    asyncio.run(d.enter("WK", "2026-12-10"))       # exit lands ON the roll
    assert d.st["open"] is None
    assert d.is_done("WK", "2026-12-10")


# ── B3: one de-dup key for all three legs ────────────────────────────────────────
@pytest.mark.parametrize("kind,when,key", [
    ("FOMC", "2026-10-27 18:05", "2026-10-27"),
    ("WK",   "2026-07-12 18:05", "2026-07-13"),
    ("EO",   "2026-07-14 02:05", "2026-07-14"),
])
def test_no_re_entry_after_completion(kind, when, key):
    """FOMC tested f'F{d}' but wrote the bare d, so it re-fired every 15s."""
    ts = pd.Timestamp(when, tz=L.ET)
    d = _daemon(live=False, when=when)
    d._eo_condition = lambda: True
    assert d.schedule(ts)[0] == kind
    d.mark_done(kind, key)
    assert d.schedule(ts) is None


def test_cushion_block_sticks_for_fomc():
    """A blocked FOMC used to be re-attempted on the next tick because the block wrote
    a key schedule() never read."""
    d = _daemon(when="2026-10-27 18:05")
    d.st["banked"] = 0.0
    d._broker_cushion = lambda: _aret(300.0)       # too thin for even one micro
    asyncio.run(d.enter("FOMC", "2026-10-27"))
    assert d.st["open"] is None


# ── B4: orphaned state.json ──────────────────────────────────────────────────────
def _orphan(td, **over):
    ot = {"kind": "WK", "key": "2026-08-17", "oid": "wk-abc", "qty": 1, "ref": 23000.0,
          "fill": 23000.0, "filled_qty": 1, "verified": True, "stp_sent": True,
          "stop_px": 22700.0, "entered_at": time.time() - 3 * 7 * 86400,
          "verify_after": None}
    ot.update(over)
    json.dump({"open": ot, "done": [], "banked": 0.0},
              open(os.path.join(td, "state.json"), "w"))
    return td


def test_orphan_with_flat_broker_is_cleared_without_sending_orders():
    """Broker says flat: the record is a crash artifact. Clear it, send nothing."""
    d = _daemon(out=_orphan(tempfile.mkdtemp()))
    sent = []
    d._cancel_all = lambda: _aret(None)
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        exit_position=lambda **k: sent.append("exit") or _aret(None))})
    d.pos_qty = 0; d.pos_qty_ts = time.time()
    asyncio.run(d.startup_recovery())
    assert d.st["open"] is None
    assert not d.st.get("halted")
    assert sent == [], "sent orders against a position the broker says does not exist"


def test_orphan_with_no_position_feed_halts_rather_than_guessing(no_sleep):
    d = _daemon(out=_orphan(tempfile.mkdtemp()))
    d.pos_qty = None
    asyncio.run(d.startup_recovery())
    assert d.st.get("halted") is True
    assert d.st["open"] is not None                # preserved for a human


def test_protect_refuses_to_price_a_stop_off_a_stale_record():
    """The money bug: a verified=False orphan made protect() submit a STOP_MARKET SELL
    off a 3-week-old fill. Below that trigger it is an immediate uncovered short."""
    d = _daemon(out=_orphan(tempfile.mkdtemp(), verified=False, stp_sent=False,
                            verify_after=time.time() - 1))
    sent = []
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        submit_order=lambda **k: sent.append(k) or _aret(None))})
    asyncio.run(d.protect(d.st["open"]))
    assert sent == [], "submitted a stop from stale state"
    assert d.st.get("halted") is True


def test_halted_daemon_stops_the_flatten_storm():
    """filled_qty=0 + a stale entered_at re-triggered flatten() every 15s forever."""
    d = _daemon(out=_orphan(tempfile.mkdtemp(), filled_qty=0))
    d.st["halted"] = True
    calls = []
    async def _f(why): calls.append(why)
    d.flatten = _f
    asyncio.run(d._tick())
    assert calls == []


def test_protect_refuses_an_inverted_stop():
    d = _daemon()
    d.quote = lambda: (22000.0, 1.0)               # market BELOW the would-be stop
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "ref": 23000.0,
                    "fill": 23000.0, "filled_qty": 1, "entered_at": time.time()}
    sent, flat = [], []
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        submit_order=lambda **k: sent.append(k) or _aret(None))})
    async def _f(why): flat.append(why)
    d.flatten = _f
    asyncio.run(d.protect(d.st["open"]))
    assert sent == []
    assert flat == ["stop_would_invert"]


# ── B5: a failed submit is not proof the order never landed ──────────────────────
def test_submit_failure_keeps_position_when_the_order_is_actually_live(no_sleep):
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "ref": 23000.0,
                    "entered_at": time.time()}
    d._list_orders = lambda: _aret([_order("wk-1", side="1", status="working")])
    asyncio.run(d._reconcile_after_submit_failure("wk-1", "WK", "k"))
    assert d.st["open"] is not None, "cleared state while the order was live -> double fill"
    assert not d.st.get("halted")


def test_submit_failure_halts_when_it_cannot_verify(no_sleep):
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "ref": 23000.0,
                    "entered_at": time.time()}
    d._list_orders = lambda: _aret(None)           # order list unavailable
    asyncio.run(d._reconcile_after_submit_failure("wk-1", "WK", "k"))
    assert d.st.get("halted") is True


# ── B6: the stop is re-verified for the life of the trade ────────────────────────
def test_verify_rearms_instead_of_latching_off():
    d = _daemon()
    ot = {"kind": "WK", "oid": "wk-1", "qty": 1, "filled_qty": 1, "stop_px": 22700.0,
          "key": "k", "entered_at": time.time()}
    d.st["open"] = ot
    d._list_orders = lambda: _aret([_order("wk-1-stp")])
    asyncio.run(d.verify(ot))
    assert ot["verified"] is True
    assert ot["verify_after"] is not None, "verification latched off for the whole hold"
    assert ot["verify_after"] > time.time()


def test_stop_is_submitted_gtc_not_day():
    """submit_order defaults to duration=DAY; these holds cross a trade-date rollover."""
    from async_rithmic import OrderDuration
    d = _daemon()
    ot = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "filled_qty": 1,
          "fill": 23000.0, "ref": 23000.0, "entered_at": time.time()}
    d.st["open"] = ot
    sent = {}
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        submit_order=lambda **k: sent.update(k) or _aret(None))})
    asyncio.run(d.protect(ot))
    assert sent["duration"] == OrderDuration.GTC


# ── B7: the cushion fail-safe can actually fail safe ─────────────────────────────
def test_unpopulated_min_account_balance_is_not_read_as_a_huge_cushion():
    """min_account_balance is a proto3 no-presence field: getattr returns 0.0, never
    None, so the old `floor is None` guard could not fire and an unset floor presented
    the whole ~$100,000 balance as cushion."""
    from async_rithmic.protocol_buffers import response_account_rms_info_pb2 as m
    rec = m.ResponseAccountRmsInfo(account_id="ACCT-A")     # floor never set
    assert getattr(rec, "min_account_balance", None) == 0.0
    d = _daemon()
    d.acct_balance = 100000.0
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        get_account_rms=lambda: _aret([rec]))})
    assert asyncio.run(d._broker_cushion()) is None


def test_broker_cushion_rejects_an_out_of_range_value():
    d = _daemon()
    d.acct_balance = 100000.0
    rec = types.SimpleNamespace(account_id="ACCT-A", min_account_balance=50000.0)
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        get_account_rms=lambda: _aret([rec]))})
    assert asyncio.run(d._broker_cushion()) is None        # $50k cushion is not real


def test_broker_cushion_allows_a_real_profit_above_the_lock_point():
    """The ceiling must not reject a legitimate cushion once banked exceeds the MLL."""
    d = _daemon()
    d.st["banked"] = 5000.0
    d.acct_balance = 105000.0
    rec = types.SimpleNamespace(account_id="ACCT-A", min_account_balance=100000.0)
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        get_account_rms=lambda: _aret([rec]))})
    assert asyncio.run(d._broker_cushion()) == pytest.approx(5000.0)


# ── R1: cushion tracks the PEAK, because the floor does ──────────────────────────
@pytest.mark.parametrize("peak,cur,want", [
    (0, 0, 3000),         # fresh account
    (2000, 500, 1500),    # peaked then gave back -- old code said 3000
    (2000, -500, 500),    # old code said 3000
    (2900, 0, 100),       # old code said 3000: a 30x overstatement
    (4000, 1000, 1000),
    (5000, 3500, 3500),   # floor locked at start; cushion == banked
    (2500, -1000, 0),     # busted
])
def test_cushion_matches_the_lock_at_start_balance_floor(peak, cur, want):
    assert L.cushion_for(cur, peak) == pytest.approx(want)


def test_sizing_shrinks_to_zero_on_a_thin_cushion():
    """A floor of 1 micro meant a nearly-busted account still took a full position."""
    assert L.ladder_size("FOMC", 0, peak_banked=2900) == 0
    assert L.ladder_size("WK", 0, peak_banked=2900) == 0
    assert L.ladder_size("FOMC", 0, peak_banked=0) == 2       # unchanged when healthy
    assert L.ladder_size("WK", 0, peak_banked=0) == 1


def test_no_rung_exceeds_the_tail_budget():
    for banked, peak in ((0, 0), (1000, 1000), (4500, 4500), (6000, 6000), (500, 2000)):
        c = L.cushion_for(banked, peak)
        for kind in ("WK", "FOMC", "EO"):
            n = L.ladder_size(kind, banked, peak_banked=peak)
            assert n * L.WORST_STOPPED_PER_MICRO <= L.TAIL_BUDGET * c + 1e-9


# ── R2: sizing uses the cushion it actually verified ─────────────────────────────
def test_sizing_follows_the_cushion_it_was_given():
    """enter() used to size off cushion_for(banked) and only GATE on the broker number,
    so the size and the verified cushion could disagree."""
    assert L.ladder_size("FOMC", 0, cushion=3000.0) == 2
    assert L.ladder_size("FOMC", 0, cushion=1200.0) == 1     # broker says thinner
    assert L.ladder_size("FOMC", 0, cushion=600.0) == 0


def test_enter_passes_the_broker_cushion_into_sizing():
    seen = {}
    real = L.ladder_size
    d = _daemon(when="2026-07-12 18:05")
    d.st["banked"] = 0.0                       # internal estimate would say $3,000
    d.roll_date = date(2026, 12, 10)
    d._broker_cushion = lambda: _aret(1200.0)  # broker says $1,200
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        submit_order=lambda **k: _aret(None))})
    L.ladder_size = lambda *a, **k: seen.update(k) or real(*a, **k)
    try:
        asyncio.run(d.enter("WK", "2026-07-13"))   # the one live-authorised leg
    finally:
        L.ladder_size = real
    assert seen["cushion"] == 1200.0, "sized off the internal estimate, not the broker"


# ── pre-registration authorisation caps ──────────────────────────────────────────
def test_unauthorised_legs_cannot_size_live():
    """euro_open_v1 was never promoted and the 18:00 FOMC failed its own criteria, yet
    both were sized at 2 micros -- larger than the one leg that IS registered."""
    assert L.ladder_size("EO", 0, cushion=3000.0, live=True) == 0
    assert L.ladder_size("FOMC", 0, cushion=3000.0, live=True) == 0
    assert L.ladder_size("WK", 0, cushion=3000.0, live=True) == 1


def test_weekend_capped_at_the_phase1_shakedown_size():
    """go_scale (2-3 micros) needs 8 forward weekends, mean > 0, >= 5/8 positive.
    At 5 weekends with mean -$286 that gate is not met."""
    assert L.AUTHORISED_MAX["WK"] == 1
    assert L.ladder_size("WK", 6000, cushion=6000.0, live=True) == 1   # even when rich
    assert L.ladder_size("WK", 6000, cushion=6000.0, live=False) == 2  # MC grid, paper


def test_paper_mode_is_unaffected_by_the_authorisation_caps():
    assert L.ladder_size("EO", 0, cushion=3000.0, live=False) == 2


# ── R3: banked accounting ────────────────────────────────────────────────────────
def test_firm_auto_flatten_fill_is_booked_not_discarded():
    """The 16:45 auto-liquidation arrives untagged; rejecting it left the trade booked
    open forever and `banked` stale -- and `banked` drives size_cap()."""
    from async_rithmic import ExchangeOrderNotificationType as NT
    d = _daemon(when="2026-07-13 16:46")       # inside the flatten window
    d.st["open"] = {"kind": "WK", "key": "2026-07-13", "oid": "wk-1", "qty": 1,
                    "ref": 23000.0, "fill": 23000.0, "filled_qty": 1,
                    "entered_at": time.time()}
    d._cancel_all = lambda: _aret(None)
    n = types.SimpleNamespace(notify_type=NT.FILL, fill_price=23100.0, transaction_type=2,
                              symbol="MNQZ6", user_tag="", account_id="ACCT-A", fill_size=1)
    asyncio.run(d._note(n))
    assert d.st["open"] is None
    assert d.st["banked"] == pytest.approx(100 * L.PV - 2 * L.COMM)
    assert d.is_done("WK", "2026-07-13")


def test_untagged_sell_outside_the_flatten_window_is_still_an_alarm():
    from async_rithmic import ExchangeOrderNotificationType as NT
    d = _daemon(when="2026-07-13 11:00")       # nowhere near 16:45
    d.st["open"] = {"kind": "WK", "key": "2026-07-13", "oid": "wk-1", "qty": 1,
                    "ref": 23000.0, "fill": 23000.0, "filled_qty": 1,
                    "entered_at": time.time()}
    n = types.SimpleNamespace(notify_type=NT.FILL, fill_price=23100.0, transaction_type=2,
                              symbol="MNQZ6", user_tag="", account_id="ACCT-A", fill_size=1)
    asyncio.run(d._note(n))
    assert d.st["open"] is not None
    assert d.st["banked"] == 0.0


def test_peak_banked_only_ever_rises():
    d = _daemon()
    d.book_pnl(1000.0, "WK", "WK:a", "exit")
    d.book_pnl(-400.0, "WK", "WK:b", "exit")
    assert d.st["banked"] == pytest.approx(600.0)
    assert d.st["peak_banked"] == pytest.approx(1000.0)
    assert L.cushion_for(600.0, 1000.0) == pytest.approx(2600.0)


# ── R8: cumulative vs incremental fill sizes ─────────────────────────────────────
@pytest.mark.parametrize("sizes", [[1, 1, 1], [1, 2, 3], [3]])
def test_fill_accumulation_survives_either_reporting_convention(sizes):
    """Blind addition on cumulative reports overstates filled_qty, which makes protect()
    place an OVERSIZED stop -- a residual naked short when it triggers."""
    from async_rithmic import ExchangeOrderNotificationType as NT
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 3, "ref": 23000.0,
                    "fill": None, "entered_at": time.time()}
    for s in sizes:
        asyncio.run(d._note(types.SimpleNamespace(
            notify_type=NT.FILL, fill_price=23000.0, transaction_type=1, symbol="MNQZ6",
            user_tag="wk-1", account_id="ACCT-A", fill_size=s)))
    assert d.st["open"]["filled_qty"] == 3


def test_entry_fill_price_is_vwap_not_first_slice():
    from async_rithmic import ExchangeOrderNotificationType as NT
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 2, "ref": 23000.0,
                    "fill": None, "entered_at": time.time()}
    for px in (23000.0, 23100.0):
        asyncio.run(d._note(types.SimpleNamespace(
            notify_type=NT.FILL, fill_price=px, transaction_type=1, symbol="MNQZ6",
            user_tag="wk-1", account_id="ACCT-A", fill_size=1)))
    assert d.st["open"]["fill"] == pytest.approx(23050.0)


# ── R4: the 16:45 flatten ────────────────────────────────────────────────────────
def test_safety_flatten_fires_before_the_mandatory_flatten():
    d = _daemon(live=False, when="2026-07-13 16:41")
    d.st["open"] = {"kind": "WK", "key": "2026-07-13", "oid": "x", "qty": 1,
                    "ref": 23000.0, "fill": 23000.0, "filled_qty": 1}
    assert d.schedule(pd.Timestamp("2026-07-13 16:41", tz=L.ET)) == ("WK", "2026-07-13", True)


def test_no_exit_is_attempted_inside_the_closed_window():
    """`hm >= 960` had no upper bound, so a restart at 17:30 fired exit_position into
    the 16:45-18:00 window where no orders are accepted."""
    d = _daemon(live=False, when="2026-07-13 17:30")
    d.st["open"] = {"kind": "WK", "key": "2026-07-13", "oid": "x", "qty": 1,
                    "ref": 23000.0, "fill": 23000.0, "filled_qty": 1}
    assert d.schedule(pd.Timestamp("2026-07-13 17:30", tz=L.ET)) is None


# ── R6: calendars cover the same span ────────────────────────────────────────────
def test_holiday_calendar_covers_every_fomc_year():
    fomc_years = {int(x[:4]) for x in L.FOMC_ANNOUNCE}
    holiday_years = {int(x[:4]) for x in L.HOLIDAYS}
    assert fomc_years <= holiday_years


def test_no_entry_sundays_are_derived_from_holidays():
    for s in L.NO_ENTRY_SUNDAYS:
        d = date.fromisoformat(s)
        assert d.weekday() == 6
        assert (d + timedelta(days=1)).isoformat() in L.HOLIDAYS


def test_weekend_entry_blocked_before_a_holiday_monday():
    d = _daemon(live=False, when="2027-05-30 18:05")     # Memorial Day weekend 2027
    assert d.schedule(pd.Timestamp("2027-05-30 18:05", tz=L.ET)) is None


# ── R7: the 50% consistency rule ─────────────────────────────────────────────────
def test_consistency_is_tracked_and_blocks_near_the_target():
    d = _daemon(when="2026-07-13 18:05")
    d.st["day_pnl"] = {"2026-07-01": 3500.0, "2026-07-08": 500.0}
    d.st["banked"] = 4000.0                              # > 60% of the $6,000 target
    assert d.consistency() == pytest.approx(0.875)
    assert d.consistency_block("WK") is not None


def test_consistency_does_not_block_far_from_the_target():
    d = _daemon(when="2026-07-13 18:05")
    d.st["day_pnl"] = {"2026-07-01": 900.0}
    d.st["banked"] = 900.0
    assert d.consistency_block("WK") is None


# ── R9: redaction ────────────────────────────────────────────────────────────────
def test_scrub_covers_every_credential_not_just_the_password():
    env = {"RITHMIC_PASSWORD": "pw-secret-1", "RITHMIC_USERNAME": "user-secret-1",
           "LUCID_ACCOUNT_ID": "ACCT-98765", "NTFY_TOPIC": "topic-secret-1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        msg = L._scrub("login user-secret-1/pw-secret-1 acct ACCT-98765 -> topic-secret-1")
        for v in env.values():
            assert v not in msg
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def test_redact_stream_catches_a_secret_split_across_writes():
    buf = types.SimpleNamespace(written=[])
    buf.write = lambda m: buf.written.append(m)
    s = L._RedactStream(buf, ["supersecret"])
    s.write("token=super"); s.write("secret end")
    assert "supersecret" not in "".join(buf.written)


# ── R5: timeouts ─────────────────────────────────────────────────────────────────
def test_broker_calls_are_wrapped_in_a_timeout():
    d = _daemon()
    async def _hang(): await _REAL_SLEEP(10)
    async def go(): return await d._call("hang", _hang(), timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())


def test_unauthorised_leg_settles_instead_of_retrying_all_window():
    """A 0-size leg must not re-attempt every 15s for the whole entry window."""
    d = _daemon(when="2026-10-27 18:05")
    calls = []
    d._broker_cushion = lambda: calls.append(1) or _aret(3000.0)
    asyncio.run(d.enter("FOMC", "2026-10-27"))
    assert d.st["open"] is None
    assert d.is_done("FOMC", "2026-10-27")
    assert calls == [], "did broker work for a leg that cannot trade"
