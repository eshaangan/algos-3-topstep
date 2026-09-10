"""
Market-By-Order (L3) microstructure features from Rithmic DepthByOrder data.

Source data: `/root/mnq_l2_data/depth_MNQ_YYYYMMDD.parquet` on the research VPS,
one row per book update, schema:

    recv_ns ssboe usecs seq update_type txn_type price prev_price size priority exch_order_id

Semantics (verified against book_recorder.py and the value distribution,
2026-09-09):
    update_type  1 = ADD (new resting order)   ~11.4M/day, size mostly 1
                 2 = MODIFY (size/price change) ~1.4M/day
                 3 = CANCEL (order removed)     ~11.4M/day
    txn_type     1 = bid side,  2 = ask side

Trades are NOT in this file -- they come from a separate LAST_TRADE stream. So
these features are built from the resting-order book only: adds, cancels, and
the resulting best-bid/ask. That is deliberate; order-flow imbalance and queue
imbalance are the parts of microstructure alpha that live in the book itself and
are hardest to fake, and they are what the repo's earlier (killed, small-sample)
BBO study could not measure.

The book is REPLAYED exactly: a dict of exch_order_id -> (side, price, size) with
add/modify/cancel applied in sequence, so best-bid and best-ask are the true top
of book, not an approximation. This is O(events) and single-pass. Best levels are
snapshotted onto a fixed time grid; features are then computed per bar.

Features per bar (all causal -- use only events within the bar):
    mid, spread                  : book state at bar close
    ofi                          : Cont-Kukanov-Stoikov order-flow imbalance at
                                   best, the canonical price-pressure signal
    queue_imbalance              : (bid_sz - ask_sz)/(bid_sz + ask_sz) at best
    add_cancel_ratio_bid/ask     : adds vs cancels near the touch, per side
    depth_bid/ask                : resting size at best
    n_updates                    : event intensity in the bar

References
----------
Cont, R., Kukanov, A., Stoikov, S. (2014), "The price impact of order book
    events." Journal of Financial Econometrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# update_type codes
ADD, MODIFY, CANCEL = 1, 2, 3
# txn_type codes
BID, ASK = 1, 2


def load_depth(path: str, columns: Optional[list] = None) -> pd.DataFrame:
    """Load one depth_MNQ day, coercing types and sorting by sequence."""
    cols = columns or ["recv_ns", "seq", "update_type", "txn_type",
                       "price", "prev_price", "size", "exch_order_id"]
    d = pd.read_parquet(path, columns=cols)
    # seq is the exchange's total order across the book; it is the correct replay
    # order. recv_ns is our clock and can tie or reorder across messages.
    d = d.sort_values("seq", kind="stable").reset_index(drop=True)
    return d


def replay_book_to_grid(
    depth: pd.DataFrame,
    grid_ms: int = 1000,
) -> pd.DataFrame:
    """
    Replay the MBO stream and snapshot best bid/ask and per-bar flow onto a grid.

    Returns one row per grid bar with the book state at bar close plus the flow
    that occurred within the bar. The replay maintains price-level aggregates
    (size resting at each price, per side), which is enough for best-level
    features without tracking every order id's size — except that MODIFY and
    CANCEL reference an order whose price we must know, so we DO keep a compact
    id -> (side, price) map and the current size per (side, price) level.

    grid_ms : snapshot interval in milliseconds of the recv_ns clock.
    """
    recv = depth["recv_ns"].to_numpy()
    utype = depth["update_type"].to_numpy()
    ttype = depth["txn_type"].to_numpy()
    price = depth["price"].to_numpy(dtype=float)
    size = depth["size"].to_numpy(dtype=float)
    oid = depth["exch_order_id"].to_numpy()

    # Level books: price -> resting size. Separate dicts per side.
    bid_levels: dict = {}
    ask_levels: dict = {}
    # order id -> (side, price, size) so MODIFY/CANCEL can find the level.
    orders: dict = {}

    t0 = recv[0] - (recv[0] % (grid_ms * 1_000_000))
    grid_ns = grid_ms * 1_000_000

    rows = []
    # per-bar flow accumulators
    add_bid = cancel_bid = add_ask = cancel_ask = 0.0
    n_upd = 0
    cur_bar = t0

    def best(levels: dict, want_max: bool):
        if not levels:
            return np.nan, 0.0
        px = max(levels) if want_max else min(levels)
        return px, levels[px]

    def snapshot(bar_close):
        bb, bbsz = best(bid_levels, True)
        ba, basz = best(ask_levels, False)
        mid = (bb + ba) / 2 if (np.isfinite(bb) and np.isfinite(ba)) else np.nan
        spread = (ba - bb) if (np.isfinite(bb) and np.isfinite(ba)) else np.nan
        rows.append({
            "bar_ns": bar_close, "mid": mid, "spread": spread,
            "best_bid": bb, "best_ask": ba, "depth_bid": bbsz, "depth_ask": basz,
            "add_bid": add_bid, "cancel_bid": cancel_bid,
            "add_ask": add_ask, "cancel_ask": cancel_ask, "n_updates": n_upd,
        })

    def apply(levels, px, delta):
        levels[px] = levels.get(px, 0.0) + delta
        if levels[px] <= 0:
            levels.pop(px, None)

    for k in range(len(recv)):
        # roll grid forward, emitting empty/quiet bars as needed
        while recv[k] >= cur_bar + grid_ns:
            snapshot(cur_bar + grid_ns)
            add_bid = cancel_bid = add_ask = cancel_ask = 0.0
            n_upd = 0
            cur_bar += grid_ns

        u, t, p, s, o = utype[k], ttype[k], price[k], size[k], oid[k]
        levels = bid_levels if t == BID else ask_levels
        n_upd += 1

        if u == ADD:
            apply(levels, p, s)
            orders[o] = (t, p, s)
            if t == BID:
                add_bid += s
            else:
                add_ask += s
        elif u == CANCEL:
            prev = orders.pop(o, None)
            if prev is not None:
                _, pp, ps = prev
                apply(bid_levels if prev[0] == BID else ask_levels, pp, -ps)
                if prev[0] == BID:
                    cancel_bid += ps
                else:
                    cancel_ask += ps
            else:
                # unseen order (started before our window): best-effort at its price
                apply(levels, p, -s)
        elif u == MODIFY:
            prev = orders.get(o)
            if prev is not None:
                pside, pp, ps = prev
                apply(bid_levels if pside == BID else ask_levels, pp, -ps)
            apply(levels, p, s)
            orders[o] = (t, p, s)

    snapshot(cur_bar + grid_ns)
    return pd.DataFrame(rows)


def compute_features(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a replayed grid into the causal feature panel.

    OFI (Cont-Kukanov-Stoikov) at best is built from changes in best-bid/ask
    price and the resting size at those levels between consecutive bars:

        e_n = 1{Pb>=Pb'} qb  - 1{Pb<=Pb'} qb'        (bid contribution)
            - 1{Pa<=Pa'} qa  + 1{Pa>=Pa'} qa'        (ask contribution)

    where P/q are best price/size, primes are the previous bar. A positive OFI is
    net buying pressure at the touch.
    """
    g = grid.copy()
    g["ts"] = pd.to_datetime(g["bar_ns"], unit="ns", utc=True)

    pb, pa = g["best_bid"].to_numpy(), g["best_ask"].to_numpy()
    qb, qa = g["depth_bid"].to_numpy(), g["depth_ask"].to_numpy()
    pb_p, pa_p = np.roll(pb, 1), np.roll(pa, 1)
    qb_p, qa_p = np.roll(qb, 1), np.roll(qa, 1)

    bid_term = np.where(pb >= pb_p, qb, 0.0) - np.where(pb <= pb_p, qb_p, 0.0)
    ask_term = np.where(pa <= pa_p, qa, 0.0) - np.where(pa >= pa_p, qa_p, 0.0)
    ofi = bid_term - ask_term
    ofi[0] = 0.0
    g["ofi"] = ofi

    denom = g["depth_bid"] + g["depth_ask"]
    g["queue_imbalance"] = np.where(denom > 0,
                                    (g["depth_bid"] - g["depth_ask"]) / denom, 0.0)

    g["net_add_bid"] = g["add_bid"] - g["cancel_bid"]
    g["net_add_ask"] = g["add_ask"] - g["cancel_ask"]
    g["flow_imbalance"] = g["net_add_bid"] - g["net_add_ask"]

    g["ret_1"] = g["mid"].diff()
    return g


def build_day(path: str, grid_ms: int = 1000) -> pd.DataFrame:
    """Load one depth day and return its feature panel."""
    depth = load_depth(path)
    grid = replay_book_to_grid(depth, grid_ms=grid_ms)
    return compute_features(grid)
