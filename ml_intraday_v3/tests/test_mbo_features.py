"""Tests for the MBO order-book replay and feature computation."""

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.features.mbo_features import (
    ADD, MODIFY, CANCEL, BID, ASK,
    replay_book_to_grid, compute_features,
)


def _depth(rows):
    """rows: list of (seq, update_type, txn_type, price, size, oid) at recv_ns=seq*1e6."""
    return pd.DataFrame([
        {"recv_ns": seq * 1_000_000, "seq": seq, "update_type": u, "txn_type": t,
         "price": p, "prev_price": 0.0, "size": s, "exch_order_id": o}
        for seq, u, t, p, s, o in rows
    ])


def test_add_builds_best_levels():
    # two bids, two asks; best bid = 100.0, best ask = 100.5
    d = _depth([
        (1, ADD, BID, 99.75, 5, "a"),
        (2, ADD, BID, 100.0, 3, "b"),
        (3, ADD, ASK, 100.5, 4, "c"),
        (4, ADD, ASK, 100.75, 6, "d"),
    ])
    g = replay_book_to_grid(d, grid_ms=1000)
    last = g.dropna(subset=["mid"]).iloc[-1]
    assert last["best_bid"] == 100.0
    assert last["best_ask"] == 100.5
    assert last["depth_bid"] == 3
    assert last["depth_ask"] == 4
    assert last["mid"] == pytest.approx(100.25)
    assert last["spread"] == pytest.approx(0.5)


def test_cancel_removes_size_and_reveals_next_level():
    d = _depth([
        (1, ADD, BID, 100.0, 3, "b"),
        (2, ADD, BID, 99.75, 5, "a"),
        (3, CANCEL, BID, 100.0, 3, "b"),      # remove the top bid
    ])
    g = replay_book_to_grid(d, grid_ms=1000)
    last = g.dropna(subset=["best_bid"]).iloc[-1]
    assert last["best_bid"] == 99.75         # next level revealed
    assert last["depth_bid"] == 5
    assert last["cancel_bid"] >= 3


def test_modify_updates_level():
    d = _depth([
        (1, ADD, ASK, 100.5, 4, "c"),
        (2, MODIFY, ASK, 100.5, 9, "c"),      # same price, larger size
    ])
    g = replay_book_to_grid(d, grid_ms=1000)
    last = g.dropna(subset=["best_ask"]).iloc[-1]
    assert last["best_ask"] == 100.5
    assert last["depth_ask"] == 9


def test_flow_accumulators_count_size():
    d = _depth([
        (1, ADD, BID, 100.0, 5, "b"),
        (2, ADD, ASK, 100.5, 2, "c"),
        (3, CANCEL, BID, 100.0, 5, "b"),
    ])
    g = replay_book_to_grid(d, grid_ms=100_000)   # one big bar
    row = g.iloc[0] if len(g) == 1 else g[g["n_updates"] > 0].iloc[0]
    assert row["add_bid"] == 5
    assert row["add_ask"] == 2
    assert row["cancel_bid"] == 5


def test_queue_imbalance_sign():
    d = _depth([
        (1, ADD, BID, 100.0, 9, "b"),         # heavy bid
        (2, ADD, ASK, 100.5, 1, "c"),         # thin ask
    ])
    g = compute_features(replay_book_to_grid(d, grid_ms=1000))
    last = g.dropna(subset=["queue_imbalance"]).iloc[-1]
    # bid-heavy book -> positive imbalance
    assert last["queue_imbalance"] > 0.5


def test_ofi_positive_when_bid_grows_ask_unchanged():
    # bar 0 sets a book; bar 1 (>=1s later) grows the bid queue -> positive OFI
    d = _depth([
        (1, ADD, BID, 100.0, 2, "b1"),
        (2, ADD, ASK, 100.5, 2, "c1"),
        (1002, ADD, BID, 100.0, 8, "b2"),     # ~1s later, more bid size at best
    ])
    g = compute_features(replay_book_to_grid(d, grid_ms=1000))
    g = g.dropna(subset=["mid"])
    assert g["ofi"].iloc[-1] > 0


def test_replay_is_deterministic():
    d = _depth([
        (1, ADD, BID, 100.0, 3, "b"),
        (2, ADD, ASK, 100.5, 4, "c"),
        (3, CANCEL, ASK, 100.5, 4, "c"),
        (4, ADD, ASK, 100.25, 2, "e"),
    ])
    a = replay_book_to_grid(d, grid_ms=1000)
    b = replay_book_to_grid(d, grid_ms=1000)
    pd.testing.assert_frame_equal(a, b)


def test_empty_book_yields_nan_mid():
    d = _depth([(1, ADD, BID, 100.0, 3, "b")])   # bid only, no ask
    g = replay_book_to_grid(d, grid_ms=1000)
    assert g["mid"].isna().all()
    assert (g["best_bid"] == 100.0).any()
