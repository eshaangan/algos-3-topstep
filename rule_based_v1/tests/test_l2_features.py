"""Tests for L2 feature extraction. Uses synthetic book/trade frames that match the
recorder schema, so the pipeline is verified before any real data exists."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rule_based_v1.validation.l2_features import (
    snapshot_features, build_bars, trade_flow, synthesize_l2, _AGG_BUY, _AGG_SELL,
)


def _one_book_row(bid_sz0, ask_sz0, bid_px0=20000.00, ask_px0=20000.25, n_levels=10):
    row = {"recv_ns": 1_000_000_000, "ssboe": 1, "usecs": 0, "update_type": 3}
    for i in range(n_levels):
        row[f"bid_px_{i}"] = bid_px0 - 0.25 * i
        row[f"ask_px_{i}"] = ask_px0 + 0.25 * i
        row[f"bid_sz_{i}"] = bid_sz0
        row[f"ask_sz_{i}"] = ask_sz0
        row[f"bid_ord_{i}"] = max(bid_sz0 // 5, 1)
        row[f"ask_ord_{i}"] = max(ask_sz0 // 5, 1)
    df = pd.DataFrame([row])
    df["ts"] = pd.to_datetime(df["recv_ns"], unit="ns", utc=True)
    return df


def test_snapshot_math_balanced():
    f = snapshot_features(_one_book_row(10, 10), n_levels=5).iloc[0]
    assert f["mid"] == pytest.approx(20000.125)
    assert f["spread"] == pytest.approx(0.25)
    assert f["l1_imb"] == pytest.approx(0.0)
    assert f["depth_imb"] == pytest.approx(0.0)
    assert f["micro_tilt"] == pytest.approx(0.0)


def test_snapshot_imbalance_sign():
    # more bid size than ask ⇒ positive imbalance and microprice tilted UP
    f = snapshot_features(_one_book_row(30, 10), n_levels=5).iloc[0]
    assert f["l1_imb"] > 0
    assert f["depth_imb"] > 0
    assert f["micro_tilt"] > 0   # microprice above mid


def test_microprice_formula():
    # bid_sz=30, ask_sz=10 → micro = (bpx*asz + apx*bsz)/(bsz+asz)
    f = snapshot_features(_one_book_row(30, 10), n_levels=1).iloc[0]
    expected = (20000.00 * 10 + 20000.25 * 30) / 40
    assert f["microprice"] == pytest.approx(expected)


def test_trade_flow_ofi():
    base = pd.Timestamp("2026-06-12 14:00:05", tz="UTC")
    t = pd.DataFrame({
        "ts": [base, base + pd.Timedelta(seconds=1), base + pd.Timedelta(seconds=2)],
        "size": [5, 3, 2],
        "aggressor": [_AGG_BUY, _AGG_SELL, _AGG_BUY],
    })
    tf = trade_flow(t, "1min")
    assert tf["t_ofi"].iloc[0] == 5 - 3 + 2          # +4
    assert tf["t_vol"].iloc[0] == 10
    assert tf["t_ofi_imb"].iloc[0] == pytest.approx(4 / 10)


def test_build_bars_shape_and_ohlc():
    book, trades = synthesize_l2(3000, seed=0, signal_strength=0.0)
    book["ts"] = pd.to_datetime(book["recv_ns"], unit="ns", utc=True)
    trades["ts"] = pd.to_datetime(trades["recv_ns"], unit="ns", utc=True)
    bars = build_bars(book, trades, freq="1min")
    assert len(bars) > 5
    for col in ["open", "high", "low", "close", "atr", "depth_imb", "micro_tilt", "t_ofi"]:
        assert col in bars.columns
    # OHLC sanity
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["close"]).all()


def test_planted_edge_is_detectable():
    # with a planted depth→price lead, depth_imb at t should correlate with next mid move
    book, trades = synthesize_l2(20000, seed=5, signal_strength=0.5)
    book["ts"] = pd.to_datetime(book["recv_ns"], unit="ns", utc=True)
    trades["ts"] = pd.to_datetime(trades["recv_ns"], unit="ns", utc=True)
    bars = build_bars(book, trades, freq="1min").dropna(subset=["depth_imb"])
    fwd = bars["close"].shift(-1) - bars["close"]
    corr = bars["depth_imb"].corr(fwd)
    assert corr > 0.1, f"planted edge not detected (corr={corr:.3f})"


def test_no_edge_control_is_near_zero():
    book, trades = synthesize_l2(20000, seed=6, signal_strength=0.0)
    book["ts"] = pd.to_datetime(book["recv_ns"], unit="ns", utc=True)
    trades["ts"] = pd.to_datetime(trades["recv_ns"], unit="ns", utc=True)
    bars = build_bars(book, trades, freq="1min").dropna(subset=["depth_imb"])
    fwd = bars["close"].shift(-1) - bars["close"]
    assert abs(bars["depth_imb"].corr(fwd)) < 0.1
