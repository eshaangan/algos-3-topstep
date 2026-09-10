"""
Depth-scaled order-flow features from MNQ book data.

WHY THIS EXISTS (and why it is not a re-run of the killed OFI screen)
--------------------------------------------------------------------
The September 2026 MBO screen concluded "OFI dead, IC ~ 0 at all horizons". Two
implementation defects in that screen make the conclusion untestable rather than
false, and this module fixes both.

1.  **OFI was measured by differencing 1-second snapshots.** Cont-Kukanov-Stoikov
    define OFI as a SUM of per-event contributions e_n over every change of the
    best quote in the interval. MNQ produces ~940 book events per second, so
    snapshot differencing kept a two-point difference and discarded essentially
    all of the flow it was supposed to measure. Here OFI is accumulated
    event-by-event off the true best-quote stream.

2.  **OFI was never divided by depth.** The CKS result is not "OFI predicts
    returns"; it is

        dP  =  beta * OFI / AD                        (CKS 2014, eq. 10-12)

    where AD is the average depth at the best quotes over the interval. The
    price-impact coefficient is INVERSELY PROPORTIONAL to available liquidity.
    Regressing raw OFI therefore averages the thin-book states, where a given
    imbalance moves price a long way, against the thick-book states, where it
    moves price almost not at all. That blended average is small by construction
    and will lose to any fixed cost. The tradeable quantity is `ofi_scaled`, and
    the economically interesting sample is its low-depth tail.

MEASURED COST FLOOR (2026-09-09, MNQ RTH): mean spread 1.64 ticks = $0.82,
commission $1.24 round trip, so a taker must clear $2.06/contract = 1.03 MNQ
points. Nothing in this module is tradeable unless it clears that.

DATA SOURCES
------------
Two input shapes are supported, and both are reduced to the SAME best-quote
change stream before any feature is computed. That is deliberate: discovery runs
on 16 days of L3 (2026-08-19 -> 2026-09-09) and validation must run on 57 days of
MBP-1 (2026-01 -> 2026-07), which is a different era and a thinner schema. A
feature that cannot be computed on both is not allowed in this module.

    MBO depth (L3)  recv_ns ssboe usecs seq update_type txn_type price
                    prev_price size priority exch_order_id
                    update_type 1=ADD 2=MODIFY 3=CANCEL, txn_type 1=bid 2=ask
                    NOTE: MODIFY on this Rithmic feed is order RE-PRICING, not an
                    iceberg refill, and no order id is ever ADDed twice. Native
                    iceberg detection (Zotikov 2019) is NOT possible here.

    MBP-1           recv_ns bid_px bid_sz ask_px ask_sz  (already a BBO stream)

    trades          recv_ns price size aggressor
                    Fill-level, not trade-summary: one row per resting order
                    matched, median size 1. Aggressive orders must be
                    reconstructed by grouping consecutive fills.

References
----------
Cont, R., Kukanov, A., Stoikov, S. (2014). "The price impact of order book
    events." Journal of Financial Econometrics 12(1), 47-88.
Cont, R., Cucuruzzu, R., Xu, T. (2023). "Cross-impact of order flow imbalance in
    equity markets."  (in ml_intraday_v3/research papers/)
Zotikov, D., Antonov, A. (2019). "CME Iceberg Order Detection and Prediction."
    arXiv:1909.09495 -- detection method, inapplicable to this feed, see above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---- MBO update_type / txn_type codes -------------------------------------
ADD, MODIFY, CANCEL = 1, 2, 3
BID, ASK = 1, 2

# ---- MNQ contract + cost constants ----------------------------------------
POINT_VALUE = 2.0            # $ per MNQ point
TICK = 0.25                  # MNQ minimum increment, in points
COMMISSION_PER_SIDE = 0.62   # $, repo-wide assumption

# Consecutive fills within this gap and with the same aggressor are treated as
# one aggressive order. CME matches a single aggressor against many resting
# orders in one event, so the fills share a timestamp to sub-millisecond.
AGGRESSOR_GROUP_NS = 2_000_000  # 2 ms


# ===========================================================================
# 1. Reduce either input shape to a best-quote change stream
# ===========================================================================

@dataclass
class BboStream:
    """Best-quote state at every instant the top of book changed.

    All arrays are the same length and aligned. `ts` is the recv_ns clock of the
    recorder, which is the only clock shared with the trade stream.
    """
    ts: np.ndarray        # int64 recv_ns
    bid_px: np.ndarray    # float
    bid_sz: np.ndarray    # float
    ask_px: np.ndarray    # float
    ask_sz: np.ndarray    # float

    def __len__(self) -> int:
        return len(self.ts)


def load_depth(path: str) -> pd.DataFrame:
    """Load one depth_MNQ day in exchange sequence order."""
    cols = ["recv_ns", "seq", "update_type", "txn_type",
            "price", "size", "exch_order_id"]
    d = pd.read_parquet(path, columns=cols)
    # seq is the exchange's total order across the book and is the correct
    # replay order; recv_ns is our clock and can tie or reorder across messages.
    return d.sort_values("seq", kind="stable").reset_index(drop=True)


def _replay_core(recv, utype, ttype, tick_idx, size, ocode,
                 n_ticks, n_orders):
    """
    Tick-indexed L3 book replay. Emits an index into the input for every event
    that changed the best quote, plus the resulting best state.

    The book is two size-by-tick arrays and an order table, so an add or remove
    is O(1). The best-quote pointers move by one step when a better price
    arrives and walk inward only when the current best empties, which is
    amortised O(1). The obvious dictionary version is O(levels) per event
    because it must rescan for max/min, and at 24M events a day that is the
    difference between two seconds and an hour.

    Orders resting before the file begins are unknown, so a CANCEL or MODIFY
    naming an unseen id is applied at its stated price as a best effort. Its
    influence decays to nothing within the first seconds of the session.
    """
    bid_sz = np.zeros(n_ticks, dtype=np.float64)
    ask_sz = np.zeros(n_ticks, dtype=np.float64)
    ord_side = np.zeros(n_orders, dtype=np.int8)      # 0 = never seen
    ord_tick = np.zeros(n_orders, dtype=np.int64)
    ord_size = np.zeros(n_orders, dtype=np.float64)

    best_bid = -1              # highest tick with resting bid size
    best_ask = n_ticks         # lowest tick with resting ask size

    n = len(recv)
    out_i = np.empty(n, dtype=np.int64)
    out_bp = np.empty(n, dtype=np.int64)
    out_bs = np.empty(n, dtype=np.float64)
    out_ap = np.empty(n, dtype=np.int64)
    out_as = np.empty(n, dtype=np.float64)
    m = 0

    pbb, pba, pbbs, pbas = -2, -2, -1.0, -1.0

    for k in range(n):
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
            # remove the order from wherever it currently rests
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
                # ... and re-add it at the new price
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

        bbs = bid_sz[best_bid] if best_bid >= 0 else 0.0
        bas = ask_sz[best_ask] if best_ask < n_ticks else 0.0
        if best_bid != pbb or best_ask != pba or bbs != pbbs or bas != pbas:
            pbb, pba, pbbs, pbas = best_bid, best_ask, bbs, bas
            out_i[m] = k
            out_bp[m] = best_bid
            out_bs[m] = bbs
            out_ap[m] = best_ask
            out_as[m] = bas
            m += 1

    return out_i[:m], out_bp[:m], out_bs[:m], out_ap[:m], out_as[:m]


try:
    from numba import njit as _njit
    _replay_core = _njit(cache=True)(_replay_core)
except Exception:  # pragma: no cover - numba is a speedup, not a requirement
    pass


def replay_bbo_from_mbo(depth: pd.DataFrame) -> BboStream:
    """
    Replay the full L3 book and emit one row per change of the best quote.

    Emitting only on CHANGE (rather than on every event) is what makes the
    event-level OFI sum both correct and cheap: OFI is defined on transitions of
    the best quote, and the large majority of events touch deeper levels and
    contribute nothing to it.

    An empty side is reported as -inf (bid) or +inf (ask), matching the
    convention downstream code drops on.
    """
    price = depth["price"].to_numpy(dtype=float)
    # Map prices onto a contiguous tick grid. The feed carries a few absurd
    # prices (seen: 30.0 and 299_956.0 on 2026-09-03) which are far outside any
    # plausible best quote; they are kept rather than filtered, because the grid
    # is cheap and silently dropping book events is how replays go wrong.
    lo = np.nanmin(price)
    tick_idx = np.rint((price - lo) / TICK).astype(np.int64)
    n_ticks = int(tick_idx.max()) + 2

    codes, _ = pd.factorize(depth["exch_order_id"], sort=False)
    codes = codes.astype(np.int64)

    i, bp, bs, ap, asz = _replay_core(
        depth["recv_ns"].to_numpy(dtype=np.int64),
        depth["update_type"].to_numpy(dtype=np.int64),
        depth["txn_type"].to_numpy(dtype=np.int64),
        tick_idx,
        depth["size"].to_numpy(dtype=np.float64),
        codes,
        n_ticks,
        int(codes.max()) + 1,
    )

    bid_px = np.where(bp >= 0, lo + bp * TICK, -np.inf)
    ask_px = np.where(ap < n_ticks, lo + ap * TICK, np.inf)
    return BboStream(depth["recv_ns"].to_numpy(dtype=np.int64)[i],
                     bid_px, bs, ask_px, asz)


def replay_bbo_from_mbp1(df: pd.DataFrame) -> BboStream:
    """Wrap an MBP-1 frame, which is already a best-quote stream, as a BboStream.

    Rows where either side is empty are dropped: OFI is undefined across a
    one-sided book, and they are a negligible fraction of RTH.
    """
    d = df.dropna(subset=["bid_px", "ask_px", "bid_sz", "ask_sz"])
    d = d.sort_values("recv_ns", kind="stable")
    return BboStream(
        d["recv_ns"].to_numpy(dtype=np.int64),
        d["bid_px"].to_numpy(dtype=float),
        d["bid_sz"].to_numpy(dtype=float),
        d["ask_px"].to_numpy(dtype=float),
        d["ask_sz"].to_numpy(dtype=float),
    )


# ===========================================================================
# 2. Aggressive-order reconstruction from fill-level trades
# ===========================================================================

def aggregate_fills(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Group fill-level trade rows into the aggressive orders that caused them.

    The trade feed reports one row per resting order matched (median size 1), so
    a single marketable order that sweeps three price levels appears as several
    rows sharing a timestamp. A run of consecutive rows with the same aggressor
    whose gap to the previous row is under AGGRESSOR_GROUP_NS is one aggressive
    order.

    Returns one row per aggressive order with:
        ts, aggressor, size, n_fills, n_levels, px_first, px_last, span_ticks
    `n_levels >= 2` is the definition of a sweep used downstream.
    """
    t = trades.sort_values("recv_ns", kind="stable").reset_index(drop=True)
    ts = t["recv_ns"].to_numpy()
    agg = t["aggressor"].to_numpy()
    px = t["price"].to_numpy(dtype=float)
    sz = t["size"].to_numpy(dtype=float)

    if len(t) == 0:
        return pd.DataFrame(columns=["ts", "aggressor", "size", "n_fills",
                                     "n_levels", "px_first", "px_last",
                                     "span_ticks"])

    new_group = np.empty(len(t), dtype=bool)
    new_group[0] = True
    new_group[1:] = (agg[1:] != agg[:-1]) | (np.diff(ts) > AGGRESSOR_GROUP_NS)
    gid = np.cumsum(new_group) - 1

    g = pd.DataFrame({"gid": gid, "ts": ts, "aggressor": agg,
                      "price": px, "size": sz})
    out = g.groupby("gid").agg(
        ts=("ts", "first"),
        aggressor=("aggressor", "first"),
        size=("size", "sum"),
        n_fills=("size", "count"),
        n_levels=("price", "nunique"),
        px_first=("price", "first"),
        px_last=("price", "last"),
        px_min=("price", "min"),
        px_max=("price", "max"),
    ).reset_index(drop=True)
    out["span_ticks"] = ((out["px_max"] - out["px_min"]) / TICK).round().astype(int)
    return out.drop(columns=["px_min", "px_max"])


def infer_buy_aggressor_code(aggr: pd.DataFrame, bbo: BboStream) -> int:
    """
    Determine empirically which `aggressor` code means "buyer initiated".

    The Rithmic field is documented inconsistently, and getting the sign wrong
    inverts every downstream result while leaving all magnitudes intact, which is
    exactly the kind of error a screen cannot catch. So we measure it: a buyer
    initiated trade lifts the offer, so its price sits at or above the prevailing
    mid. We compare the two codes and return whichever sits higher.
    """
    idx = np.searchsorted(bbo.ts, aggr["ts"].to_numpy(), side="right") - 1
    ok = idx >= 0
    with np.errstate(invalid="ignore"):
        mid = (bbo.bid_px[idx[ok]] + bbo.ask_px[idx[ok]]) / 2.0
        rel = aggr["px_first"].to_numpy()[ok] - mid
    code = aggr["aggressor"].to_numpy()[ok]
    # A one-sided book has no mid, so those trades carry no information about
    # the coding. Drop them rather than letting a NaN poison the group mean.
    good = np.isfinite(rel)
    rel, code = rel[good], code[good]
    m1 = np.nanmean(rel[code == 1]) if (code == 1).any() else np.nan
    m2 = np.nanmean(rel[code == 2]) if (code == 2).any() else np.nan
    if not np.isfinite(m1) or not np.isfinite(m2):
        raise ValueError("cannot infer aggressor coding: one code is absent")
    if abs(m1 - m2) < TICK / 4:
        raise ValueError(
            f"aggressor coding is not separable: mean price-vs-mid "
            f"code1={m1:.4f} code2={m2:.4f}")
    return 1 if m1 > m2 else 2


# ===========================================================================
# 3. Event-level OFI, time-weighted depth, and the bar panel
# ===========================================================================

def build_bars(
    bbo: BboStream,
    aggr: Optional[pd.DataFrame] = None,
    buy_code: Optional[int] = None,
    grid_ms: int = 1000,
) -> pd.DataFrame:
    """
    Accumulate event-level OFI and time-weighted book state onto a fixed grid.

    OFI (Cont-Kukanov-Stoikov 2014). For consecutive best-quote states n-1, n:

        e_n =  1{Pb_n >= Pb_{n-1}} qb_n  -  1{Pb_n <= Pb_{n-1}} qb_{n-1}
             - 1{Pa_n <= Pa_{n-1}} qa_n  +  1{Pa_n >= Pa_{n-1}} qa_{n-1}

    and OFI over a bar is the sum of e_n over the events inside it. A positive
    value is net buying pressure at the touch. This is the definition the killed
    screen approximated by differencing bar endpoints.

    Depth is TIME weighted, not event weighted. Event weighting would let a burst
    of 1-lot quote flicker dominate the average and would break the CKS scaling,
    whose AD is an average over the interval's duration.

    Every column is causal within its bar: it uses only events at or before the
    bar close, and the forward return columns are added separately by the
    labelling step, never here.
    """
    ts = bbo.ts
    pb, qb = bbo.bid_px, bbo.bid_sz
    pa, qa = bbo.ask_px, bbo.ask_sz
    n = len(ts)
    if n < 2:
        raise ValueError("need at least two best-quote states")

    # ---- event-level OFI increments, fully vectorised ---------------------
    bid_term = np.where(pb[1:] >= pb[:-1], qb[1:], 0.0) \
        - np.where(pb[1:] <= pb[:-1], qb[:-1], 0.0)
    ask_term = np.where(pa[1:] <= pa[:-1], qa[1:], 0.0) \
        - np.where(pa[1:] >= pa[:-1], qa[:-1], 0.0)
    e = bid_term - ask_term                      # length n-1, aligned to ts[1:]

    # ---- time weights: each state holds until the next event --------------
    dt = np.diff(ts).astype(float)               # length n-1, aligned to ts[:-1]
    mid = (pb + pa) / 2.0
    spread = pa - pb
    depth = (qb + qa) / 2.0
    denom = qb + qa
    qimb = np.divide(qb - qa, denom, out=np.zeros_like(denom), where=denom > 0)

    # ---- grid assignment ---------------------------------------------------
    grid_ns = grid_ms * 1_000_000
    t0 = (ts[0] // grid_ns) * grid_ns
    n_bars = int((ts[-1] - t0) // grid_ns) + 1
    bar_of_state = np.minimum(((ts - t0) // grid_ns).astype(np.int64), n_bars - 1)

    # OFI increment e[i] happens at ts[i+1], so it belongs to that state's bar.
    ofi = np.bincount(bar_of_state[1:], weights=e, minlength=n_bars)
    # Time-weighted quantities belong to the bar of the state that was resting.
    b_hold = bar_of_state[:-1]
    w_time = np.bincount(b_hold, weights=dt, minlength=n_bars)
    w_depth = np.bincount(b_hold, weights=depth[:-1] * dt, minlength=n_bars)
    w_spread = np.bincount(b_hold, weights=spread[:-1] * dt, minlength=n_bars)
    w_qimb = np.bincount(b_hold, weights=qimb[:-1] * dt, minlength=n_bars)
    n_events = np.bincount(bar_of_state, minlength=n_bars).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        avg_depth = np.where(w_time > 0, w_depth / w_time, np.nan)
        avg_spread = np.where(w_time > 0, w_spread / w_time, np.nan)
        avg_qimb = np.where(w_time > 0, w_qimb / w_time, np.nan)

    # mid/spread at bar close = state of the last event at or before the close
    last_state = np.full(n_bars, -1, dtype=np.int64)
    np.maximum.at(last_state, bar_of_state, np.arange(n, dtype=np.int64))
    # bars with no event inherit the previous bar's closing state
    last_state = pd.Series(np.where(last_state >= 0, last_state, np.nan)).ffill()
    valid = last_state.notna().to_numpy()
    ls = last_state.fillna(0).to_numpy().astype(np.int64)

    bar_ns = t0 + np.arange(n_bars, dtype=np.int64) * grid_ns + grid_ns

    out = pd.DataFrame({
        "bar_ns": bar_ns,
        "mid": np.where(valid, mid[ls], np.nan),
        "spread_close": np.where(valid, spread[ls], np.nan),
        "best_bid": np.where(valid, pb[ls], np.nan),
        "best_ask": np.where(valid, pa[ls], np.nan),
        "ofi": ofi,
        "avg_depth": avg_depth,
        "avg_spread": avg_spread,
        "queue_imbalance": avg_qimb,
        "n_events": n_events,
    })

    # ---- THE feature: CKS depth scaling -----------------------------------
    # dP = beta * OFI / AD. Guard the divide: a bar with no resting size has no
    # defined impact coefficient, and must not become an infinite signal.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ofi_scaled"] = np.where(out["avg_depth"] > 0,
                                     out["ofi"] / out["avg_depth"], np.nan)

    # ---- trade-side columns ------------------------------------------------
    if aggr is not None and len(aggr) > 0:
        if buy_code is None:
            raise ValueError("buy_code is required when aggr is given; call "
                             "infer_buy_aggressor_code first")
        a_ts = aggr["ts"].to_numpy()
        a_bar = ((a_ts - t0) // grid_ns).astype(np.int64)
        keep = (a_bar >= 0) & (a_bar < n_bars)
        a_bar = a_bar[keep]
        a_sz = aggr["size"].to_numpy(dtype=float)[keep]
        a_lv = aggr["n_levels"].to_numpy()[keep]
        sign = np.where(aggr["aggressor"].to_numpy()[keep] == buy_code, 1.0, -1.0)

        out["trade_vol"] = np.bincount(a_bar, weights=a_sz, minlength=n_bars)
        out["signed_vol"] = np.bincount(a_bar, weights=a_sz * sign,
                                        minlength=n_bars)
        out["n_aggressive"] = np.bincount(a_bar, minlength=n_bars).astype(float)
        sweep = a_lv >= 2
        out["n_sweeps"] = np.bincount(a_bar[sweep], minlength=n_bars).astype(float)
        out["sweep_vol"] = np.bincount(a_bar[sweep], weights=a_sz[sweep],
                                       minlength=n_bars)
        mx = np.zeros(n_bars)
        np.maximum.at(mx, a_bar, a_sz)
        out["max_aggressive_size"] = mx
        # Trade imbalance normalised the same way OFI is: the CKS argument is
        # about liquidity consumed relative to liquidity available, and applies
        # to marketable flow as directly as it does to quote flow.
        with np.errstate(invalid="ignore", divide="ignore"):
            out["signed_vol_scaled"] = np.where(
                out["avg_depth"] > 0, out["signed_vol"] / out["avg_depth"], np.nan)

    out["ts"] = pd.to_datetime(out["bar_ns"], unit="ns", utc=True)
    return out


def add_depth_state(bars: pd.DataFrame, lookback_bars: int = 1800) -> pd.DataFrame:
    """
    Rank each bar's depth against its own trailing distribution.

    The CKS coefficient is 1/depth, so the economically distinct regime is not
    "low depth in absolute contracts" but "low depth relative to what this market
    has been carrying recently". Depth has a strong intraday shape (thin at the
    open, thick midday), and an absolute threshold would simply select the open.

    The window is TRAILING and CLOSED (`closed="left"` semantics via shift), so a
    bar's percentile never sees its own depth or anything after it. Default 1800
    bars = 30 minutes on a 1-second grid.
    """
    b = bars.copy()
    d = b["avg_depth"]
    min_p = max(lookback_bars // 4, 2)
    b["depth_pctile"] = _pct_rank_trailing(d.to_numpy(dtype=float),
                                           lookback_bars, min_p)
    prior = d.shift(1).rolling(lookback_bars, min_periods=min_p)
    b["depth_z"] = (d - prior.mean()) / prior.std()
    return b


def _pct_rank_trailing(x: np.ndarray, window: int, min_p: int) -> np.ndarray:
    """Fraction of the preceding `window` values that are strictly below x[i].

    Strictly trailing: x[i] is compared against x[i-window:i] and never against
    itself or anything after it. Returns NaN until `min_p` prior values exist.
    """
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        v = x[i]
        if not np.isfinite(v):
            continue
        lo = i - window
        if lo < 0:
            lo = 0
        cnt = 0
        below = 0
        for j in range(lo, i):
            w = x[j]
            if np.isfinite(w):
                cnt += 1
                if w < v:
                    below += 1
        if cnt >= min_p:
            out[i] = below / cnt
    return out


try:  # optional: the trailing rank is the one O(n*window) loop in the module
    from numba import njit
    _pct_rank_trailing = njit(cache=True)(_pct_rank_trailing)
except Exception:  # pragma: no cover - numba is a convenience, not a requirement
    pass


# ===========================================================================
# 4. One-day driver
# ===========================================================================

def build_day_from_mbo(depth_path: str,
                       trade_path: Optional[str] = None,
                       grid_ms: int = 1000,
                       depth_lookback_bars: int = 1800) -> pd.DataFrame:
    """Build the full feature panel for one L3 day."""
    depth = load_depth(depth_path)
    bbo = replay_bbo_from_mbo(depth)
    aggr = buy_code = None
    if trade_path is not None:
        trades = pd.read_parquet(trade_path)
        aggr = aggregate_fills(trades)
        buy_code = infer_buy_aggressor_code(aggr, bbo)
    bars = build_bars(bbo, aggr=aggr, buy_code=buy_code, grid_ms=grid_ms)
    return add_depth_state(bars, lookback_bars=depth_lookback_bars)


def build_day_from_mbp1(mbp1_path: str,
                        grid_ms: int = 1000,
                        depth_lookback_bars: int = 1800) -> pd.DataFrame:
    """Build the same panel from an MBP-1 day, for out-of-era validation.

    Trade columns are only produced if the file carries trade fields; otherwise
    the quote-side features stand alone and the validation is restricted to them.
    """
    df = pd.read_parquet(mbp1_path)
    bbo = replay_bbo_from_mbp1(df)
    bars = build_bars(bbo, grid_ms=grid_ms)
    return add_depth_state(bars, lookback_bars=depth_lookback_bars)
