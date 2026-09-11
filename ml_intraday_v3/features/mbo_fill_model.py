"""Measured fill cost for MNQ, from the L3 (MBO) book rather than an assumption.

WHY THIS EXISTS
---------------
Every cost-floor verdict in this project rests on one number: a taker round trip
costs $2.06/contract (mean quoted spread 1.64 ticks = $0.82, plus $1.24
commission). That number is a QUOTED-spread average. It is the right cost only
for an order small enough to be filled entirely at the touch, and it is silent
about what happens at the 6-13 contract sizes the feasibility map actually
wants to trade.

This module replays the full order book and walks the ladder, so the cost of a
marketable order of Q contracts is measured rather than assumed:

    effective round trip (Q) = buy_vwap(Q) - sell_vwap(Q)   [in points]

At Q=1 that reduces to the quoted spread and reproduces the old number. Above
Q=1 it adds whatever the book actually charges for depth.

MODEL AND ITS LIMITS
--------------------
* The order is assumed to sweep resting liquidity at the instant it arrives, in
  price priority. That is what a market order does.
* TIMING CONVENTION: the order fills against the book STRICTLY BEFORE its
  arrival instant. An event stamped at exactly the arrival time is treated as
  not yet applied. This is the conservative choice and it is what
  test_book_matches_replay_core pins against depth_scaled_flow.
* `latency_ns` shifts the arrival away from the decision time, so the order
  fills against the book it MEETS, not the one it SAW. This is the honest
  version and it is why a book-based model beats a fixed spread assumption.
* NOT modelled: the reaction of other participants to the order itself
  (transient impact beyond the sweep), and hidden/iceberg liquidity, which this
  Rithmic feed cannot express (see depth_scaled_flow, Zotikov note). Both push
  the true cost in OPPOSITE directions, so this is an estimate, not a bound.
* The book starts cold. Orders resting before the file begins are unknown, so
  DEEP levels are understated early in the session. Use `warmup_ns` and check
  `validate_against_bbo` before trusting a day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .depth_scaled_flow import (ADD, MODIFY, CANCEL, BID, ASK,
                                POINT_VALUE, TICK, COMMISSION_PER_SIDE,
                                load_depth)

MAX_LEVELS = 400          # ticks to walk before declaring an order unfillable


@dataclass
class FillCurve:
    """Fill prices for a grid of order sizes at a grid of query times."""
    ts: np.ndarray          # int64 recv_ns, one per query actually reached
    sizes: np.ndarray       # int64, the order sizes priced
    bid_px: np.ndarray      # (n_ts,)
    ask_px: np.ndarray      # (n_ts,)
    buy_vwap: np.ndarray    # (n_ts, n_sizes) NaN where the book could not fill
    sell_vwap: np.ndarray   # (n_ts, n_sizes)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid_px + self.ask_px)

    def round_trip_dollars(self) -> np.ndarray:
        """Per-contract cost of buying then selling Q, including commission."""
        gross = (self.buy_vwap - self.sell_vwap) * POINT_VALUE
        return gross + 2.0 * COMMISSION_PER_SIDE

    def slippage_ticks(self) -> tuple[np.ndarray, np.ndarray]:
        """One-way distance from mid to the fill, in ticks, buy then sell."""
        m = self.mid[:, None]
        return (self.buy_vwap - m) / TICK, (m - self.sell_vwap) / TICK

    def to_frame(self) -> pd.DataFrame:
        rt = self.round_trip_dollars()
        bs, ss = self.slippage_ticks()
        out = []
        for j, q in enumerate(self.sizes):
            out.append(pd.DataFrame({
                "ts": self.ts, "size": int(q),
                "bid_px": self.bid_px, "ask_px": self.ask_px,
                "buy_vwap": self.buy_vwap[:, j], "sell_vwap": self.sell_vwap[:, j],
                "rt_cost_usd": rt[:, j],
                "buy_slip_ticks": bs[:, j], "sell_slip_ticks": ss[:, j],
            }))
        return pd.concat(out, ignore_index=True)



def _price_one(bid_sz, ask_sz, best_bid, best_ask, sizes, max_levels, n_ticks,
               out_buy, out_sell, m):
    """Walk the ladder and price every requested size at the current book."""
    ns = len(sizes)
    for j in range(ns):
        want = float(sizes[j])
        need = want
        notional = 0.0
        t = best_ask
        lim = best_ask + max_levels
        if lim > n_ticks:
            lim = n_ticks
        while need > 0.0 and t < lim:
            avail = ask_sz[t]
            if avail > 0.0:
                take = avail if avail < need else need
                notional += take * t
                need -= take
            t += 1
        if need <= 0.0:
            out_buy[m, j] = notional / want
        need = want
        notional = 0.0
        t = best_bid
        lim = best_bid - max_levels
        if lim < 0:
            lim = -1
        while need > 0.0 and t > lim:
            avail = bid_sz[t]
            if avail > 0.0:
                take = avail if avail < need else need
                notional += take * t
                need -= take
            t -= 1
        if need <= 0.0:
            out_sell[m, j] = notional / want


def _fill_core(recv, utype, ttype, tick_idx, size, ocode,
               n_ticks, n_orders, query, sizes, max_levels):
    """Book replay identical to depth_scaled_flow._replay_core, but at each
    query timestamp it walks the ladder outward and prices every order size.

    The walk is the only addition. It is O(levels touched), which for the sizes
    of interest is a handful, so the cost is negligible against the replay.
    """
    bid_sz = np.zeros(n_ticks, dtype=np.float64)
    ask_sz = np.zeros(n_ticks, dtype=np.float64)
    ord_side = np.zeros(n_orders, dtype=np.int8)
    ord_tick = np.zeros(n_orders, dtype=np.int64)
    ord_size = np.zeros(n_orders, dtype=np.float64)

    best_bid = -1
    best_ask = n_ticks

    nq = len(query)
    ns = len(sizes)
    out_ts = np.empty(nq, dtype=np.int64)
    out_bp = np.empty(nq, dtype=np.int64)
    out_ap = np.empty(nq, dtype=np.int64)
    out_buy = np.full((nq, ns), np.nan, dtype=np.float64)
    out_sell = np.full((nq, ns), np.nan, dtype=np.float64)
    q = 0
    m = 0

    n = len(recv)
    for k in range(n):
        # ---- price every query time this event has passed -----------------
        while q < nq and recv[k] >= query[q]:
            if best_bid >= 0 and best_ask < n_ticks and best_ask > best_bid:
                out_ts[m] = query[q]
                out_bp[m] = best_bid
                out_ap[m] = best_ask
                _price_one(bid_sz, ask_sz, best_bid, best_ask, sizes,
                           max_levels, n_ticks, out_buy, out_sell, m)
                m += 1
            q += 1
        if q >= nq:
            break

        # ---- apply the event ---------------------------------------------
        u = utype[k]
        t = ttype[k]
        ti = tick_idx[k]
        s = size[k]
        o = ocode[k]
        if ti < 0 or ti >= n_ticks:
            continue

        if u == ADD:
            if t == BID:
                bid_sz[ti] += s
                if ti > best_bid:
                    best_bid = ti
            else:
                ask_sz[ti] += s
                if ti < best_ask:
                    best_ask = ti
            ord_side[o] = t
            ord_tick[o] = ti
            ord_size[o] = s
        elif u == CANCEL or u == MODIFY:
            ps = ord_side[o]
            if ps != 0:
                pt = ord_tick[o]
                pz = ord_size[o]
            else:
                pt = ti
                pz = s
                ps = t
            if ps == BID:
                bid_sz[pt] -= pz
                if bid_sz[pt] <= 0.0:
                    bid_sz[pt] = 0.0
                    if pt == best_bid:
                        while best_bid >= 0 and bid_sz[best_bid] <= 0.0:
                            best_bid -= 1
            else:
                ask_sz[pt] -= pz
                if ask_sz[pt] <= 0.0:
                    ask_sz[pt] = 0.0
                    if pt == best_ask:
                        while best_ask < n_ticks and ask_sz[best_ask] <= 0.0:
                            best_ask += 1
            ord_side[o] = 0
            ord_size[o] = 0.0
            if u == MODIFY:
                if t == BID:
                    bid_sz[ti] += s
                    if ti > best_bid:
                        best_bid = ti
                else:
                    ask_sz[ti] += s
                    if ti < best_ask:
                        best_ask = ti
                ord_side[o] = t
                ord_tick[o] = ti
                ord_size[o] = s

    # queries after the last book event still have a book to fill against
    while q < nq:
        if best_bid >= 0 and best_ask < n_ticks and best_ask > best_bid:
            out_ts[m] = query[q]
            out_bp[m] = best_bid
            out_ap[m] = best_ask
            _price_one(bid_sz, ask_sz, best_bid, best_ask, sizes,
                       max_levels, n_ticks, out_buy, out_sell, m)
            m += 1
        q += 1

    return out_ts[:m], out_bp[:m], out_ap[:m], out_buy[:m], out_sell[:m]


try:
    from numba import njit as _njit
    _price_one = _njit(cache=True)(_price_one)
    _fill_core = _njit(cache=True)(_fill_core)
except Exception:  # pragma: no cover
    pass


def price_orders(depth: pd.DataFrame, query_ns: np.ndarray,
                 sizes=(1, 2, 4, 6, 9, 13, 20, 40, 60),
                 latency_ns: int = 0,
                 max_levels: int = MAX_LEVELS) -> FillCurve:
    """Price a marketable order of each size at each query time.

    `latency_ns` moves the arrival later than the decision, so the order fills
    against the book it meets. Query times must be sorted.
    """
    query_ns = np.asarray(query_ns, dtype=np.int64)
    if np.any(np.diff(query_ns) < 0):
        raise ValueError("query_ns must be sorted")
    arrive = query_ns + int(latency_ns)

    price = depth["price"].to_numpy(dtype=float)
    lo = float(np.nanmin(price))
    tick_idx = np.rint((price - lo) / TICK).astype(np.int64)
    n_ticks = int(tick_idx.max()) + 2
    codes, _ = pd.factorize(depth["exch_order_id"], sort=False)
    codes = codes.astype(np.int64)

    ts, bp, ap, buy, sell = _fill_core(
        depth["recv_ns"].to_numpy(dtype=np.int64),
        depth["update_type"].to_numpy(dtype=np.int64),
        depth["txn_type"].to_numpy(dtype=np.int64),
        tick_idx,
        depth["size"].to_numpy(dtype=np.float64),
        codes,
        n_ticks,
        int(codes.max()) + 1,
        arrive,
        np.asarray(sizes, dtype=np.int64),
        int(max_levels),
    )
    return FillCurve(ts=ts, sizes=np.asarray(sizes, dtype=np.int64),
                     bid_px=lo + bp * TICK, ask_px=lo + ap * TICK,
                     buy_vwap=lo + buy * TICK, sell_vwap=lo + sell * TICK)


def validate_against_bbo(curve: FillCurve, bbo: pd.DataFrame) -> dict:
    """Cross-check the replayed top of book against the INDEPENDENT bbo recording.

    The bbo file is a separate capture of the same feed, one-sided per message,
    so each side is forward-filled before comparison. Agreement here is the
    evidence that the replayed ladder is real; a fill model built on a wrong
    book would be confidently wrong.
    """
    b = bbo.sort_values("recv_ns")
    ref_bid = b["bid_px"].ffill().to_numpy()
    ref_ask = b["ask_px"].ffill().to_numpy()
    ref_ts = b["recv_ns"].to_numpy(dtype=np.int64)
    i = np.searchsorted(ref_ts, curve.ts, side="right") - 1
    ok = i >= 0
    rb, ra = ref_bid[i[ok]], ref_ask[i[ok]]
    mb, ma = curve.bid_px[ok], curve.ask_px[ok]
    good = np.isfinite(rb) & np.isfinite(ra)
    return {
        "n_compared": int(good.sum()),
        "bid_exact_match": float(np.mean(np.abs(mb[good] - rb[good]) < 1e-9)),
        "ask_exact_match": float(np.mean(np.abs(ma[good] - ra[good]) < 1e-9)),
        "bid_mean_abs_diff_ticks": float(np.mean(np.abs(mb[good] - rb[good])) / TICK),
        "ask_mean_abs_diff_ticks": float(np.mean(np.abs(ma[good] - ra[good])) / TICK),
    }
