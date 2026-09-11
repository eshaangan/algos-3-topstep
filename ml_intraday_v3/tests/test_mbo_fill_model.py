"""Tests for the MBO fill model.

The load-bearing test is `test_book_matches_replay_core`: mbo_fill_model keeps
its own copy of the book because it needs the whole ladder, not just the best
quote. Two copies of the same logic drift, so the best-quote stream each one
produces is locked identical here.
"""
import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.features.depth_scaled_flow import (
    ADD, CANCEL, MODIFY, BID, ASK, TICK, POINT_VALUE, COMMISSION_PER_SIDE,
    replay_bbo_from_mbo,
)
from ml_intraday_v3.features.mbo_fill_model import price_orders


def mk(events):
    """events: (recv_ns, update_type, txn_type, price, size, order_id)"""
    return pd.DataFrame(events, columns=[
        "recv_ns", "update_type", "txn_type", "price", "size", "exch_order_id"])


def simple_book():
    """Asks 100.00 x2, 100.25 x3, 100.50 x10;  bids 99.75 x1, 99.50 x4."""
    return mk([
        (1, ADD, ASK, 100.00, 2, "a1"),
        (2, ADD, ASK, 100.25, 3, "a2"),
        (3, ADD, ASK, 100.50, 10, "a3"),
        (4, ADD, BID, 99.75, 1, "b1"),
        (5, ADD, BID, 99.50, 4, "b2"),
    ])


def test_top_of_book_is_the_touch():
    c = price_orders(simple_book(), np.array([10]), sizes=(1,))
    assert c.bid_px[0] == pytest.approx(99.75)
    assert c.ask_px[0] == pytest.approx(100.00)


def test_size_one_equals_the_quoted_spread():
    c = price_orders(simple_book(), np.array([10]), sizes=(1,))
    rt = c.round_trip_dollars()[0, 0]
    expected = (100.00 - 99.75) * POINT_VALUE + 2 * COMMISSION_PER_SIDE
    assert rt == pytest.approx(expected)


def test_vwap_walks_the_ladder():
    # buy 5: 2 @ 100.00 + 3 @ 100.25 -> (2*100 + 3*100.25)/5 = 100.15
    c = price_orders(simple_book(), np.array([10]), sizes=(5,))
    assert c.buy_vwap[0, 0] == pytest.approx(100.15)
    # sell 5: 1 @ 99.75 + 4 @ 99.50 -> (99.75 + 4*99.50)/5 = 99.55
    assert c.sell_vwap[0, 0] == pytest.approx(99.55)


def test_cost_rises_with_size():
    # simple_book() rests only 5 on the bid, so use sizes fillable on BOTH sides
    c = price_orders(simple_book(), np.array([10]), sizes=(1, 2, 5))
    rt = c.round_trip_dollars()[0]
    assert rt[0] < rt[1] < rt[2]


def test_unfillable_size_is_nan_not_a_silent_bad_price():
    # only 15 contracts rest on the ask side in total
    c = price_orders(simple_book(), np.array([10]), sizes=(100,))
    assert np.isnan(c.buy_vwap[0, 0])
    assert np.isnan(c.round_trip_dollars()[0, 0])


def test_cancel_removes_liquidity_and_widens_the_fill():
    d = simple_book()
    before = price_orders(d, np.array([10]), sizes=(2,)).buy_vwap[0, 0]
    d2 = pd.concat([d, mk([(6, CANCEL, ASK, 100.00, 2, "a1")])], ignore_index=True)
    after = price_orders(d2, np.array([10]), sizes=(2,)).buy_vwap[0, 0]
    assert before == pytest.approx(100.00)
    assert after == pytest.approx(100.25)   # top level gone, the sweep goes deeper


def test_modify_reprices_rather_than_duplicating():
    d = pd.concat([simple_book(),
                   mk([(6, MODIFY, ASK, 100.75, 2, "a1")])], ignore_index=True)
    c = price_orders(d, np.array([10]), sizes=(2,))
    # a1's 2 lots moved off 100.00, so a 2-lot buy now fills at 100.25
    assert c.buy_vwap[0, 0] == pytest.approx(100.25)


def test_latency_fills_against_the_later_book():
    d = pd.concat([simple_book(),
                   mk([(20, CANCEL, ASK, 100.00, 2, "a1")])], ignore_index=True)
    seen = price_orders(d, np.array([10]), sizes=(2,), latency_ns=0)
    met = price_orders(d, np.array([10]), sizes=(2,), latency_ns=15)
    assert seen.buy_vwap[0, 0] == pytest.approx(100.00)
    assert met.buy_vwap[0, 0] == pytest.approx(100.25)


def test_query_times_must_be_sorted():
    with pytest.raises(ValueError):
        price_orders(simple_book(), np.array([10, 5]), sizes=(1,))


def test_one_sided_book_is_skipped_not_priced_at_infinity():
    d = mk([(1, ADD, ASK, 100.00, 2, "a1")])       # no bids at all
    c = price_orders(d, np.array([10]), sizes=(1,))
    assert len(c.ts) == 0


def test_book_matches_replay_core():
    """The duplicated book maintenance must agree with depth_scaled_flow."""
    # Bids stay strictly below 100 and asks strictly above, so the book is not
    # crossed. A randomly-priced two-sided book is crossed most of the time and
    # both replays would then correctly refuse to quote it.
    rng = np.random.default_rng(0)
    rows, live = [], {}
    for i in range(1, 4000):
        side = BID if rng.random() < 0.5 else ASK
        off = int(rng.integers(1, 20))
        px = round(100.0 + (off if side == ASK else -off) * TICK, 4)
        if live and rng.random() < 0.45:
            oid = str(rng.choice(list(live)))
            u = CANCEL if rng.random() < 0.6 else MODIFY
            oside, opx, osz = live[oid]
            noff = int(rng.integers(1, 20))
            npx = round(100.0 + (noff if oside == ASK else -noff) * TICK, 4)
            rows.append((i, u, oside, npx if u == MODIFY else opx, osz, oid))
            if u == MODIFY:
                live[oid] = (oside, npx, osz)
            else:
                del live[oid]
        else:
            oid = "o%d" % i
            sz = int(rng.integers(1, 9))
            rows.append((i, ADD, side, px, sz, oid))
            live[oid] = (side, px, sz)
    d = mk(rows)
    ref = replay_bbo_from_mbo(d)
    q = np.arange(1, 4000, 7, dtype=np.int64)
    c = price_orders(d, q, sizes=(1,))
    # Convention: price_orders fills against the book STRICTLY BEFORE the
    # arrival instant, while replay_bbo_from_mbo timestamps the state AFTER
    # each event. side="left" selects the last event before the query, which
    # is the same instant, and the two must then agree exactly.
    j = np.searchsorted(ref.ts, c.ts, side="left") - 1
    ok = j >= 0
    assert ok.sum() > 100
    assert np.allclose(c.bid_px[ok], ref.bid_px[j[ok]])
    assert np.allclose(c.ask_px[ok], ref.ask_px[j[ok]])
