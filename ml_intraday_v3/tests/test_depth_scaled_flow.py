"""
Tests for depth_scaled_flow.

These lock the three things that would corrupt every downstream result while
leaving the numbers superficially plausible:

  1. the OFI sign convention (a sign flip inverts every verdict),
  2. the event-level accumulation (the defect that made the killed screen
     untestable: bar-endpoint differencing discards ~940 events/second),
  3. the causality of the trailing depth rank (a leak here would manufacture
     an edge out of nothing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.features.depth_scaled_flow import (
    ADD, ASK, BID, CANCEL, MODIFY, BboStream, add_depth_state,
    aggregate_fills, build_bars, infer_buy_aggressor_code,
    replay_bbo_from_mbo, replay_bbo_from_mbp1,
)

MS = 1_000_000


def _depth(rows):
    """rows: (seq, update_type, txn_type, price, size, oid), recv_ns = seq * 1ms."""
    df = pd.DataFrame(rows, columns=["seq", "update_type", "txn_type",
                                     "price", "size", "exch_order_id"])
    df["recv_ns"] = df["seq"] * MS
    return df


# ---------------------------------------------------------------- book replay

def test_replay_tracks_best_quote_through_add_cancel_modify():
    d = _depth([
        (1, ADD, BID, 100.00, 5, "b1"),
        (2, ADD, ASK, 100.50, 3, "a1"),
        (3, ADD, BID, 100.25, 2, "b2"),      # new better bid
        (4, CANCEL, BID, 100.25, 2, "b2"),   # ... withdrawn, best falls back
        (5, MODIFY, ASK, 100.25, 3, "a1"),   # ask re-prices down
    ])
    s = replay_bbo_from_mbo(d)
    assert list(s.bid_px) == [100.00, 100.00, 100.25, 100.00, 100.00]
    assert list(s.ask_px) == [np.inf, 100.50, 100.50, 100.50, 100.25]
    assert list(s.bid_sz) == [5, 5, 2, 5, 5]
    # the MODIFY moved the whole order, so no size is left behind at 100.50
    assert s.ask_sz[-1] == 3


def test_replay_emits_only_on_best_quote_change():
    """Deep-level churn must not produce rows; that is what keeps the OFI sum
    both correct and cheap."""
    d = _depth([
        (1, ADD, BID, 100.00, 5, "b1"),
        (2, ADD, ASK, 100.50, 3, "a1"),
        (3, ADD, BID, 90.00, 7, "b9"),       # far from touch
        (4, CANCEL, BID, 90.00, 7, "b9"),
        (5, ADD, ASK, 110.00, 4, "a9"),
    ])
    s = replay_bbo_from_mbo(d)
    assert len(s) == 2


def test_cancel_of_order_resting_before_the_file_starts():
    """An unseen order id is cancelled at its stated price rather than dropped,
    otherwise the book inflates monotonically through the session."""
    d = _depth([
        (1, ADD, BID, 100.00, 5, "b1"),
        (2, ADD, ASK, 100.50, 3, "a1"),
        (3, CANCEL, BID, 100.00, 5, "unseen-but-same-level"),
    ])
    s = replay_bbo_from_mbo(d)
    assert s.bid_px[-1] == -np.inf  # level fully removed


# ------------------------------------------------------------------ OFI sign

def _stream(states):
    ts = np.arange(len(states), dtype=np.int64) * MS
    a = np.array(states, dtype=float)
    return BboStream(ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def test_ofi_is_positive_when_the_bid_advances():
    # bid steps up: buying pressure at the touch
    s = _stream([(100.00, 5, 100.50, 5),
                 (100.25, 6, 100.50, 5)])
    bars = build_bars(s, grid_ms=10)
    assert bars["ofi"].sum() > 0


def test_ofi_is_negative_when_the_ask_advances_downward():
    # ask steps down: selling pressure at the touch
    s = _stream([(100.00, 5, 100.50, 5),
                 (100.00, 5, 100.25, 6)])
    bars = build_bars(s, grid_ms=10)
    assert bars["ofi"].sum() < 0


def test_ofi_is_positive_when_bid_size_grows_at_the_same_price():
    s = _stream([(100.00, 5, 100.50, 5),
                 (100.00, 9, 100.50, 5)])
    bars = build_bars(s, grid_ms=10)
    assert bars["ofi"].sum() == pytest.approx(4.0)


def test_ofi_matches_the_cks_formula_term_by_term():
    """One transition, worked through CKS 2014 by hand.

        e = 1{Pb>=Pb'} qb - 1{Pb<=Pb'} qb' - 1{Pa<=Pa'} qa + 1{Pa>=Pa'} qa'

    Both quotes step UP, which is buying pressure on both sides of the book: the
    bid advances and the offer is withdrawn upward.

        bid: 100.25 >= 100.00 -> +qb  = +6 ;  100.25 <= 100.00 is false -> -0
        ask: 100.75 <= 100.50 is false -> -0 ;  100.75 >= 100.50 -> +qa' = +7
        e = 6 + 7 = 13
    """
    s = _stream([(100.00, 5, 100.50, 7),
                 (100.25, 6, 100.75, 4)])
    bars = build_bars(s, grid_ms=10)
    assert bars["ofi"].sum() == pytest.approx(13.0)


def _ofi_by_endpoint_differencing(states) -> float:
    """The killed screen's method: apply the CKS formula to the bar's first and
    last state only, ignoring everything in between."""
    a = np.array(states, dtype=float)
    s = BboStream(np.array([0, 1], dtype=np.int64),
                  a[[0, -1], 0], a[[0, -1], 1], a[[0, -1], 2], a[[0, -1], 3])
    return float(build_bars(s, grid_ms=10).loc[0, "ofi"])


def test_ofi_accumulates_every_event_not_just_bar_endpoints():
    """The defect this module exists to fix.

    A bid that walks up one tick at a time accumulates flow at every step. The
    endpoints see a single displacement and report roughly one step's worth. At
    ~940 book events per second on MNQ, a one-second bar loses almost all of its
    flow to that approximation.
    """
    states = [(100.00 + 0.25 * i, 10.0, 200.00, 10.0) for i in range(11)]
    s = _stream(states)
    bars = build_bars(s, grid_ms=10_000)     # one bar covering everything
    assert len(bars) == 1

    event_level = bars["ofi"].iloc[0]
    endpoint = _ofi_by_endpoint_differencing(states)

    assert event_level == pytest.approx(100.0)   # ten steps of +10
    assert endpoint == pytest.approx(10.0)       # one displacement of +10
    assert event_level == pytest.approx(10 * endpoint)


# ------------------------------------------------------- depth scaling itself

def test_ofi_scaled_divides_by_time_weighted_depth():
    s = _stream([(100.00, 10, 100.50, 10),
                 (100.00, 20, 100.50, 10)])
    bars = build_bars(s, grid_ms=10)
    row = bars.iloc[0]
    assert row["ofi"] == pytest.approx(10.0)
    assert row["avg_depth"] == pytest.approx(10.0)   # state held for the bar
    assert row["ofi_scaled"] == pytest.approx(1.0)


def test_ofi_scaled_is_nan_not_inf_when_depth_is_zero():
    s = _stream([(100.00, 0, 100.50, 0),
                 (100.25, 0, 100.50, 0)])
    bars = build_bars(s, grid_ms=10)
    assert not np.isinf(bars["ofi_scaled"]).any()


def test_depth_is_time_weighted_not_event_weighted():
    """A burst of flicker must not outvote a long-held state."""
    ts = np.array([0, 1 * MS, 1 * MS + 1000, 100 * MS], dtype=np.int64)
    s = BboStream(ts,
                  np.array([100.0] * 4), np.array([100.0, 2.0, 2.0, 2.0]),
                  np.array([100.5] * 4), np.array([100.0, 2.0, 2.0, 2.0]))
    bars = build_bars(s, grid_ms=1000)
    # depth 100 held ~1ms, depth 2 held ~99ms -> time weighting lands near 2
    assert bars["avg_depth"].iloc[0] < 10


# ------------------------------------------------------------ trade handling

def test_aggregate_fills_groups_a_sweep_into_one_aggressive_order():
    t = pd.DataFrame({
        "recv_ns": [0, 1000, 2000, 50 * MS],
        "price": [100.50, 100.75, 101.00, 100.50],
        "size": [2, 3, 1, 4],
        "aggressor": [1, 1, 1, 1],
    })
    a = aggregate_fills(t)
    assert len(a) == 2
    assert a.iloc[0]["size"] == 6
    assert a.iloc[0]["n_fills"] == 3
    assert a.iloc[0]["n_levels"] == 3      # a sweep
    assert a.iloc[0]["span_ticks"] == 2
    assert a.iloc[1]["n_levels"] == 1


def test_aggregate_fills_splits_on_aggressor_change():
    t = pd.DataFrame({
        "recv_ns": [0, 100, 200],
        "price": [100.50, 100.50, 100.00],
        "size": [1, 1, 1],
        "aggressor": [1, 1, 2],
    })
    assert len(aggregate_fills(t)) == 2


def test_buy_aggressor_code_is_inferred_from_price_against_mid():
    s = _stream([(100.00, 5, 100.50, 5)] * 4)
    a = pd.DataFrame({
        "ts": [1 * MS, 1 * MS, 2 * MS, 2 * MS],
        "aggressor": [1, 1, 2, 2],
        "px_first": [100.50, 100.50, 100.00, 100.00],  # code 1 lifts the offer
        "size": [1, 1, 1, 1],
        "n_levels": [1, 1, 1, 1],
    })
    assert infer_buy_aggressor_code(a, s) == 1
    a["aggressor"] = [2, 2, 1, 1]
    assert infer_buy_aggressor_code(a, s) == 2


def test_buy_aggressor_code_refuses_to_guess_when_inseparable():
    s = _stream([(100.00, 5, 100.50, 5)] * 3)
    a = pd.DataFrame({
        "ts": [1 * MS, 1 * MS, 2 * MS],
        "aggressor": [1, 2, 1],
        "px_first": [100.25, 100.25, 100.25],   # both codes at mid
        "size": [1, 1, 1],
        "n_levels": [1, 1, 1],
    })
    with pytest.raises(ValueError):
        infer_buy_aggressor_code(a, s)


def test_signed_volume_uses_the_inferred_buy_code():
    s = _stream([(100.00, 5, 100.50, 5)] * 3)
    a = pd.DataFrame({
        "ts": [0, 0, 0],
        "aggressor": [1, 1, 2],
        "px_first": [100.50, 100.50, 100.00],
        "size": [3.0, 2.0, 4.0],
        "n_levels": [1, 1, 1],
    })
    bars = build_bars(s, aggr=a, buy_code=1, grid_ms=10_000)
    assert bars["trade_vol"].sum() == pytest.approx(9.0)
    assert bars["signed_vol"].sum() == pytest.approx(3.0 + 2.0 - 4.0)


# ------------------------------------------------------------ causality guard

def test_depth_pctile_never_sees_the_present_or_the_future():
    """A single huge depth spike at the end must not change earlier ranks."""
    n = 400
    base = np.linspace(1, 2, n)
    bars = pd.DataFrame({"avg_depth": base})
    a = add_depth_state(bars, lookback_bars=100)["depth_pctile"].to_numpy()

    spiked = base.copy()
    spiked[-1] = 1e6
    b = add_depth_state(pd.DataFrame({"avg_depth": spiked}),
                        lookback_bars=100)["depth_pctile"].to_numpy()

    np.testing.assert_allclose(a[:-1], b[:-1], equal_nan=True)


def test_depth_pctile_is_high_when_depth_is_unusually_thick():
    x = np.concatenate([np.full(300, 5.0), [50.0]])
    p = add_depth_state(pd.DataFrame({"avg_depth": x}),
                        lookback_bars=200)["depth_pctile"].to_numpy()
    assert p[-1] == pytest.approx(1.0)


def test_depth_pctile_is_low_when_depth_is_unusually_thin():
    x = np.concatenate([np.full(300, 5.0), [0.1]])
    p = add_depth_state(pd.DataFrame({"avg_depth": x}),
                        lookback_bars=200)["depth_pctile"].to_numpy()
    assert p[-1] == pytest.approx(0.0)


# ---------------------------------------------------------- mbp1 equivalence

def test_mbp1_and_mbo_paths_produce_identical_features():
    """The validation set must be computed by the same code as the discovery
    set, or the out-of-era test is not a test of the same thing."""
    d = _depth([
        (1, ADD, BID, 100.00, 5, "b1"),
        (2, ADD, ASK, 100.50, 3, "a1"),
        (3, ADD, BID, 100.25, 2, "b2"),
        (4, CANCEL, BID, 100.25, 2, "b2"),
    ])
    from_mbo = replay_bbo_from_mbo(d)
    keep = np.isfinite(from_mbo.bid_px) & np.isfinite(from_mbo.ask_px)
    mbp1 = pd.DataFrame({
        "recv_ns": from_mbo.ts[keep],
        "bid_px": from_mbo.bid_px[keep], "bid_sz": from_mbo.bid_sz[keep],
        "ask_px": from_mbo.ask_px[keep], "ask_sz": from_mbo.ask_sz[keep],
    })
    from_mbp1 = replay_bbo_from_mbp1(mbp1)

    a = build_bars(BboStream(from_mbo.ts[keep], from_mbo.bid_px[keep],
                             from_mbo.bid_sz[keep], from_mbo.ask_px[keep],
                             from_mbo.ask_sz[keep]), grid_ms=10)
    b = build_bars(from_mbp1, grid_ms=10)
    pd.testing.assert_frame_equal(a, b)
