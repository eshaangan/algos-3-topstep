"""Tests for the pre-registered validation harness.

Fast and deterministic — no model or parquet needed. Exercises the trade
simulator mechanics, the DSR, the gate logic, the content hash, and the audit
ledger's p-hacking detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rule_based_v1.validation.harness import (
    SimParams, simulate, aggregate_stats, monthly_breakdown,
    deflated_sharpe_ratio, PreRegistration, evaluate, HoldoutLedger,
)


def _bars(rows):
    """rows: list of (ts, open, high, low, close). 5-min UTC index."""
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows]},
        index=idx,
    )


def _signals(index, direction, atr):
    return pd.DataFrame({"direction": direction, "atr": atr}, index=index)


# Use UTC 16:00 = h_et 11 so timestamps are inside any realistic window; the
# simulator itself is window-agnostic (filtering happens in the adapter).
T0 = pd.Timestamp("2026-04-01 16:00", tz="UTC")
def _ts(n): return [T0 + pd.Timedelta(minutes=5 * i) for i in range(n)]


# ─────────────────────────────────────────────────── simulator ───────────────

def test_long_profit_target_hit():
    ts = _ts(3)
    # Enter long at bar0 close=100 (+1 tick slip → 100.25), atr=10, pt=+2*10.
    # Bar1 high 121 >= target 120.25 → profit_target.
    bars = _bars([(ts[0], 100, 100, 100, 100),
                  (ts[1], 100, 121, 99, 110),
                  (ts[2], 110, 111, 109, 110)])
    sig = _signals(bars.index, [1, 0, 0], [10.0, 10.0, 10.0])
    p = SimParams(pt_atr=2.0, sl_atr=1.0, horizon_bars=12, cooldown_bars=0,
                  slippage_ticks=1, tick_size=0.25, commission_per_side=0.62,
                  point_value=2.0, n_contracts=1)
    tr = simulate(bars, sig, p)
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "profit_target"
    assert tr.iloc[0]["pnl"] > 0


def test_both_hit_resolves_to_stop():
    ts = _ts(2)
    # Bar1 spans both stop (90.25-1*10=90.25→ wait) — set atr=10, entry 100.25:
    # stop=90.25, target=120.25. Bar1 low 85 <= stop AND high 125 >= target.
    bars = _bars([(ts[0], 100, 100, 100, 100),
                  (ts[1], 100, 125, 85, 100)])
    sig = _signals(bars.index, [1, 0], [10.0, 10.0])
    p = SimParams(cooldown_bars=0, slippage_ticks=1, tick_size=0.25)
    tr = simulate(bars, sig, p)
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "stop_loss"   # conservative tie-break
    assert tr.iloc[0]["pnl"] < 0


def test_time_stop_at_horizon():
    n = 6
    ts = _ts(n)
    # Flat-ish price that never hits pt/sl; horizon=3 → time stop.
    bars = _bars([(t, 100, 100.5, 99.5, 100) for t in ts])
    sig = _signals(bars.index, [1] + [0] * (n - 1), [10.0] * n)
    p = SimParams(pt_atr=2.0, sl_atr=1.0, horizon_bars=3, cooldown_bars=0)
    tr = simulate(bars, sig, p)
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "time_stop"
    assert tr.iloc[0]["bars_held"] == 3


def test_max_trades_per_day_and_cooldown():
    n = 12
    ts = _ts(n)
    # Every bar signals; each trade resolves next bar via target. With
    # max_trades_per_day=2 we must see exactly 2 entries on this single day.
    rows = []
    for i in range(n):
        rows.append((ts[i], 100, 130, 99, 100))  # big high → target hits fast
    bars = _bars(rows)
    sig = _signals(bars.index, [1] * n, [10.0] * n)
    p = SimParams(pt_atr=2.0, sl_atr=1.0, horizon_bars=12,
                  max_trades_per_day=2, cooldown_bars=0)
    tr = simulate(bars, sig, p)
    assert len(tr) == 2


def test_no_entry_on_zero_direction_or_bad_atr():
    ts = _ts(3)
    bars = _bars([(t, 100, 130, 70, 100) for t in ts])
    sig = _signals(bars.index, [0, 1, 1], [10.0, np.nan, -5.0])
    p = SimParams(cooldown_bars=0)
    tr = simulate(bars, sig, p)
    assert len(tr) == 0   # dir 0, NaN atr, and atr<=0 all rejected


# ─────────────────────────────────────────────────── stats / DSR ─────────────

def test_aggregate_and_monthly():
    tr = pd.DataFrame({
        "entry_time": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02"),
                       pd.Timestamp("2026-05-01")],
        "pnl": [100.0, -50.0, 200.0],
        "exit_reason": ["profit_target", "stop_loss", "profit_target"],
    })
    agg = aggregate_stats(tr)
    assert agg["n_trades"] == 3
    assert agg["total_pnl"] == pytest.approx(250.0)
    assert agg["win_rate"] == pytest.approx(2 / 3)
    mb = monthly_breakdown(tr)
    assert list(mb["month"]) == ["2026-04", "2026-05"]
    assert mb.set_index("month").loc["2026-05", "pnl"] == pytest.approx(200.0)


def test_dsr_strong_vs_noise():
    rng = np.random.default_rng(0)
    strong = rng.normal(1.0, 1.0, 400)      # clear positive edge
    noise = rng.normal(0.0, 1.0, 400)       # no edge
    d_strong = deflated_sharpe_ratio(strong, n_trials=50)["dsr"]
    d_noise = deflated_sharpe_ratio(noise, n_trials=50)["dsr"]
    assert d_strong > 0.95
    assert d_noise < 0.5


def test_dsr_degenerate_inputs():
    assert deflated_sharpe_ratio([], n_trials=10)["dsr"] is None
    assert deflated_sharpe_ratio([5.0, 5.0, 5.0], n_trials=10)["dsr"] is None


# ─────────────────────────────────────────── preregistration / gate ──────────

def _prereg(tmp_path, **gate_overrides):
    import yaml
    gate = {"min_oos_trades": 3, "min_months": 2, "min_positive_month_fraction": 0.5,
            "require_net_profitable": True, "min_dsr": 0.95, "max_drawdown_usd": 10000}
    gate.update(gate_overrides)
    cfg = {
        "strategy_id": "test_strat",
        "holdout": {"start": "2026-03-01", "end": "2026-06-01"},
        "dsr_n_trials": 10,
        "gate": gate,
        "sim": {"pt_atr": 2.0, "sl_atr": 1.0, "conf": 0.5, "min_atr_pts": 5.0},
    }
    path = tmp_path / "preregister.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return PreRegistration.load(path)


def test_content_hash_changes_with_rule_but_not_strategy_id(tmp_path):
    a = _prereg(tmp_path)
    h1 = a.content_hash()
    a.strategy_id = "renamed"            # metadata, not part of the rule
    assert a.content_hash() == h1
    a.gate["min_dsr"] = 0.99            # rule change
    assert a.content_hash() != h1


def test_evaluate_nogo_on_negative(tmp_path):
    prereg = _prereg(tmp_path)
    tr = pd.DataFrame({
        "entry_time": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01"),
                       pd.Timestamp("2026-05-02")],
        "pnl": [-100.0, -50.0, -10.0],
        "exit_reason": ["stop_loss"] * 3,
    })
    res = evaluate(tr, prereg)
    assert res["verdict"] == "NO-GO"
    assert any("not > 0" in f for f in res["failures"])


def test_evaluate_go_on_strong(tmp_path):
    # Many positive trades across 3 months, low variance → high DSR, passes gate.
    prereg = _prereg(tmp_path, min_oos_trades=30)
    months = ["2026-03", "2026-04", "2026-05"]
    rows = []
    rng = np.random.default_rng(1)
    for m in months:
        for d in range(20):
            rows.append({"entry_time": pd.Timestamp(f"{m}-{(d % 27) + 1:02d}"),
                         "pnl": float(rng.normal(120, 60)), "exit_reason": "profit_target"})
    tr = pd.DataFrame(rows)
    res = evaluate(tr, prereg)
    assert res["dsr"]["dsr"] > 0.95
    assert res["verdict"] == "GO", res["failures"]


# ─────────────────────────────────────────────────── ledger ──────────────────

def test_ledger_records_and_detects_reuse_and_rule_change(tmp_path):
    prereg = _prereg(tmp_path)
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")

    assert ledger.warnings_for(prereg) == []            # clean first look
    res = evaluate(pd.DataFrame({"entry_time": [pd.Timestamp("2026-04-01")],
                                 "pnl": [10.0], "exit_reason": ["x"]}), prereg)
    ledger.record(prereg, res)

    warns = ledger.warnings_for(prereg)                 # second look → reuse warning
    assert any("already been evaluated" in w for w in warns)
    assert not any("CHANGED" in w for w in warns)

    prereg.gate["min_dsr"] = 0.5                        # rule edited after the fact
    warns2 = ledger.warnings_for(prereg)
    assert any("CHANGED" in w for w in warns2)
