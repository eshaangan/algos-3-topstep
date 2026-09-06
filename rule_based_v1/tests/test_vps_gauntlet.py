"""Tests for VPS CSV → harness gauntlet path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from rule_based_v1.validation.harness import PreRegistration, evaluate
from rule_based_v1.validation.vps_csv import load_vps_trades, slice_holdout


def _write_vps_csv(path: Path, rows: list[str], *, with_rr: bool = True) -> Path:
    if with_rr:
        header = "date,time,direction,entry,stop,target,exit,rr,pnl_pts,pnl_usd,outcome"
    else:
        header = "date,time,direction,entry,stop,target,exit,pnl_pts,pnl_usd,outcome"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_load_vps_trades_maps_columns(tmp_path: Path):
    path = _write_vps_csv(
        tmp_path / "book.csv",
        [
            "2025-03-10,10:30,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
            "2025-03-10,11:00,short,20100.0,20110.0,20080.0,20110.0,0.5,-10.0,-21.24,sl",
            "2025-03-11,09:45,long,20050.0,20040.0,20055.0,20050.0,0.5,-0.25,-1.54,be",
        ],
    )
    trades = load_vps_trades(path)
    assert list(trades.columns) == [
        "entry_time", "pnl", "direction", "entry_price", "exit_price", "exit_reason",
        # carried since 2026-09-05 so execution_realism() can audit the bracket
        "stop_price", "target_price",
    ]
    assert len(trades) == 3
    assert trades.iloc[0]["direction"] == 1
    assert trades.iloc[1]["direction"] == -1
    assert trades.iloc[0]["pnl"] == pytest.approx(18.76)
    assert trades.iloc[0]["entry_price"] == pytest.approx(20000.0)
    assert trades.iloc[1]["exit_price"] == pytest.approx(20110.0)
    assert trades.iloc[0]["exit_reason"] == "profit_target"
    assert trades.iloc[1]["exit_reason"] == "stop_loss"
    assert trades.iloc[2]["exit_reason"] == "break_even"
    # America/New_York March → EDT (UTC-4): 10:30 ET → 14:30 UTC
    assert trades.iloc[0]["entry_time"] == pd.Timestamp("2025-03-10 14:30", tz="UTC")


def test_load_vps_trades_filters_nofill(tmp_path: Path):
    path = _write_vps_csv(
        tmp_path / "book.csv",
        [
            "2025-03-10,10:30,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
            "2025-03-10,10:35,long,20001.0,19991.0,20021.0,20001.0,0.5,0.0,0.0,nofill",
            "2025-03-10,10:40,long,20002.0,19992.0,20022.0,20002.0,0.5,0.0,0.0,no_fill",
            "2025-03-10,10:45,short,20100.0,20110.0,20080.0,20100.0,0.5,0.0,0.0,skip",
            "2025-03-10,10:50,short,20101.0,20111.0,20081.0,20101.0,0.5,0.0,0.0,skip_wide",
            "2025-03-10,10:55,long,20003.0,19993.0,20023.0,20003.0,0.5,0.0,0.0,cancelled",
            "2025-03-10,11:00,short,20100.0,20110.0,20080.0,20110.0,0.5,-10.0,-21.24,sl",
        ],
    )
    trades = load_vps_trades(path)
    assert len(trades) == 2
    assert set(trades["exit_reason"]) == {"profit_target", "stop_loss"}


def test_load_vps_trades_without_rr(tmp_path: Path):
    path = _write_vps_csv(
        tmp_path / "mr.csv",
        ["2025-03-10,09:30,long,20000.0,19990.0,20020.0,20020.0,10.0,18.76,tp"],
        with_rr=False,
    )
    trades = load_vps_trades(path)
    assert len(trades) == 1
    assert trades.iloc[0]["pnl"] == pytest.approx(18.76)


def test_slice_holdout(tmp_path: Path):
    path = _write_vps_csv(
        tmp_path / "book.csv",
        [
            "2024-12-31,15:00,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
            "2025-01-01,10:00,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
            "2025-06-15,10:00,short,20100.0,20110.0,20080.0,20080.0,0.5,10.0,18.76,tp",
            "2026-07-01,10:00,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
            "2026-07-02,10:00,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,18.76,tp",
        ],
    )
    trades = load_vps_trades(path)
    hold = slice_holdout(trades, "2025-01-01", "2026-07-01")
    assert len(hold) == 3
    dates = pd.to_datetime(hold["entry_time"]).dt.tz_convert("America/New_York").dt.date
    assert dates.min().isoformat() == "2025-01-01"
    assert dates.max().isoformat() == "2026-07-01"


def test_evaluate_on_synthetic_vps_csv(tmp_path: Path):
    # Enough months/trades to exercise evaluate; gate thresholds set low for unit test.
    rows = []
    for i, month in enumerate(["2025-01", "2025-02", "2025-03"]):
        for d in range(1, 11):
            day = f"{month}-{d:02d}"
            # Alternate wins/losses so variance is non-zero for DSR.
            if (i + d) % 2 == 0:
                rows.append(
                    f"{day},10:00,long,20000.0,19990.0,20020.0,20020.0,0.5,10.0,20.0,tp"
                )
            else:
                rows.append(
                    f"{day},10:00,short,20100.0,20110.0,20080.0,20110.0,0.5,-5.0,-11.0,sl"
                )
    path = _write_vps_csv(tmp_path / "synth.csv", rows)
    trades = load_vps_trades(path)
    hold = slice_holdout(trades, "2025-01-01", "2025-03-31")

    prereg_path = tmp_path / "prereg.yaml"
    prereg_path.write_text(
        yaml.dump(
            {
                "strategy_id": "vps_test_synth",
                "holdout": {"start": "2025-01-01", "end": "2025-03-31"},
                "dsr_n_trials": 1,
                "gate": {
                    "min_oos_trades": 5,
                    "min_months": 2,
                    "min_positive_month_fraction": 0.0,
                    "require_net_profitable": False,
                    "min_dsr": 0.0,
                    "max_drawdown_usd": 1_000_000,
                },
                "sim": {
                    "point_value": 2.0,
                    "tick_size": 0.25,
                    "commission_per_side": 0.62,
                },
            }
        )
    )
    prereg = PreRegistration.load(prereg_path)
    result = evaluate(hold, prereg)

    assert set(result.keys()) >= {
        "verdict", "failures", "aggregate", "monthly", "dsr",
        "config_hash", "strategy_id", "holdout",
    }
    assert result["strategy_id"] == "vps_test_synth"
    assert result["verdict"] in ("GO", "NO-GO")
    assert isinstance(result["failures"], list)
    assert result["aggregate"]["n_trades"] == len(hold)
    assert "dsr" in result["dsr"]
