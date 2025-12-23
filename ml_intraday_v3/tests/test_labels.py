"""
Tests for triple-barrier labeling and label CLI integration.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from labels import apply_triplebarrier, generate_events
from labels.schema import write_label_schema
from core.instrument import load_instrument_from_execution_spec
from cli import build_labels_command


def _minimal_labeling_config(horizon_bars=2, account_for_costs=False):
    return {
        "primary_labeling": {
            "event_policy": "every_bar",
            "triple_barrier": {
                "volatility_estimator": "atr",
                "volatility_params": {"atr_period": 14},
                "pt_multipliers": [1.0],
                "sl_multipliers": [1.0],
                "horizon_bars": {"1m": [horizon_bars], "5m": [horizon_bars]},
                "label_encoding": {
                    "target_first": 1,
                    "stop_first": -1,
                    "vertical": 0,
                },
                "account_for_costs": account_for_costs,
            },
        },
    }


def _minimal_execution_spec(touch_ordering="ohlc_path", slippage_ticks=0.0):
    return {
        "instrument": {
            "symbol": "MES",
            "tick_size_points": 0.25,
            "contract_multiplier_usd_per_point": 5.0,
        },
        "fill_model": {
            "fill_price": "next_bar_open",
            "touch_ordering": touch_ordering,
        },
        "costs": {
            "slippage_ticks": {"1m": slippage_ticks, "5m": slippage_ticks},
            "commission_per_contract": 0.0,
        },
        "holding_constraints": {"max_holding_bars": {"1m": 2, "5m": 2}},
    }


def _make_bars(timestamps, ohlc_rows):
    df = pd.DataFrame(ohlc_rows, columns=["open", "high", "low", "close"])
    df["is_synthetic"] = False
    df.index = pd.DatetimeIndex(timestamps)
    return df


INSTRUMENT_SPEC = load_instrument_from_execution_spec(
    Path(__file__).parent.parent / "configs" / "execution_spec.yaml"
)


def test_triple_barrier_first_touch_upper_lower_known_path():
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.5, 99.5, 100.0],
            [100.0, 101.2, 100.2, 101.0],
            [101.0, 101.1, 100.8, 100.9],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [timestamps[0]],
            "t1": [timestamps[2]],
            "bar_size": ["1m"],
            "side": [0],
            "sigma": [1.0],
            "pt_mult": [1.0],
            "sl_mult": [1.0],
            "horizon_bars": [2],
        }
    )

    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=_minimal_labeling_config(),
        execution_spec=_minimal_execution_spec(),
        instrument_spec=INSTRUMENT_SPEC,
    )

    assert labeled.loc[0, "y"] == 1
    assert labeled.loc[0, "t_touch"] == timestamps[1]
    assert np.isclose(labeled.loc[0, "ret_gross"], 1.0)


def test_triple_barrier_neither_touch_vertical_exit():
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.4, 99.6, 100.0],
            [100.0, 100.4, 99.7, 100.2],
            [100.2, 100.3, 99.8, 100.1],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [timestamps[0]],
            "t1": [timestamps[2]],
            "bar_size": ["1m"],
            "side": [0],
            "sigma": [1.0],
            "pt_mult": [1.0],
            "sl_mult": [1.0],
            "horizon_bars": [2],
        }
    )

    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=_minimal_labeling_config(),
        execution_spec=_minimal_execution_spec(),
        instrument_spec=INSTRUMENT_SPEC,
    )

    assert labeled.loc[0, "y"] == 0
    assert labeled.loc[0, "t_touch"] == timestamps[2]
    assert np.isclose(labeled.loc[0, "exit_price"], 100.1)
    assert np.isclose(labeled.loc[0, "ret_gross"], 0.1)


def test_touch_ordering_stop_first_vs_target_first_differs_when_both_hit_same_bar():
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=2, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.2, 99.8, 100.0],
            [100.0, 101.5, 98.5, 100.0],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [timestamps[0]],
            "t1": [timestamps[1]],
            "bar_size": ["1m"],
            "side": [0],
            "sigma": [1.0],
            "pt_mult": [1.0],
            "sl_mult": [1.0],
            "horizon_bars": [1],
        }
    )

    labeled_stop = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=_minimal_labeling_config(horizon_bars=1),
        execution_spec=_minimal_execution_spec(touch_ordering="stop_first"),
        instrument_spec=INSTRUMENT_SPEC,
    )

    labeled_target = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=_minimal_labeling_config(horizon_bars=1),
        execution_spec=_minimal_execution_spec(touch_ordering="target_first"),
        instrument_spec=INSTRUMENT_SPEC,
    )

    assert labeled_stop.loc[0, "y"] == -1
    assert labeled_target.loc[0, "y"] == 1


def test_triple_barrier_short_side_hits_target_when_price_drops():
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.5, 99.5, 100.0],
            [100.0, 100.2, 98.8, 99.0],  # should hit short target (<= 99.0)
            [99.0, 99.2, 98.9, 99.1],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [timestamps[0]],
            "t1": [timestamps[2]],
            "bar_size": ["1m"],
            "side": [-1],
            "sigma": [1.0],
            "pt_mult": [1.0],
            "sl_mult": [1.0],
            "horizon_bars": [2],
        }
    )

    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=_minimal_labeling_config(),
        execution_spec=_minimal_execution_spec(),
        instrument_spec=INSTRUMENT_SPEC,
    )

    assert labeled.loc[0, "y"] == 1
    assert labeled.loc[0, "t_touch"] == timestamps[1]
    assert np.isclose(labeled.loc[0, "exit_price"], 99.0)
    assert np.isclose(labeled.loc[0, "ret_gross"], 1.0)


def test_generate_events_cusum_filters_bars():
    idx = pd.date_range("2025-01-01 09:30:00", periods=8, freq="1min", tz="UTC")
    close = np.array([100.0, 100.2, 100.4, 100.6, 100.2, 99.8, 100.0, 100.2])
    bars_df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "is_synthetic": False,
        },
        index=idx,
    )

    labeling_cfg = _minimal_labeling_config(horizon_bars=2)
    labeling_cfg["primary_labeling"]["event_policy"] = "cusum"
    labeling_cfg["primary_labeling"]["cusum"] = {"threshold_atr_mult": 0.5}

    events = generate_events(
        bars_df=bars_df,
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=_minimal_execution_spec(),
    )

    assert events["t0"].tolist() == [idx[3], idx[5]]


def test_generate_events_trend_scanning_sets_side_and_horizon():
    idx = pd.date_range("2025-01-01 09:30:00", periods=12, freq="1min", tz="UTC")
    close = np.linspace(100.0, 102.2, num=len(idx))
    bars_df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "is_synthetic": False,
        },
        index=idx,
    )

    labeling_cfg = _minimal_labeling_config(horizon_bars=3)
    labeling_cfg["primary_labeling"]["event_policy"] = "trend_scanning"
    labeling_cfg["primary_labeling"]["trend_scanning"] = {
        "tstat_threshold": 0.5,
        "use_cusum_prefilter": False,
        "cusum_threshold_atr_mult": 1.0,
    }
    labeling_cfg["primary_labeling"]["triple_barrier"]["volatility_params"] = {"atr_period": 1}

    events = generate_events(
        bars_df=bars_df,
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec={**_minimal_execution_spec(), "holding_constraints": {"max_holding_bars": {"1m": 3, "5m": 3}}},
    )

    assert not events.empty
    assert set(events["horizon_bars"].unique().tolist()) == {3}
    assert set(events["side"].unique().tolist()) == {1}


def test_build_labels_writes_artifacts_and_updates_manifest(tmp_path):
    run_dir = tmp_path / "run_labels"
    run_dir.mkdir()

    for bar_size in ["1m", "5m"]:
        bar_dir = run_dir / f"bar_size={bar_size}"
        bar_dir.mkdir()

        timestamps = pd.date_range(
            "2025-01-01 09:30:00", periods=10, freq="1min"
        )
        bars_df = _make_bars(
            timestamps,
            [[100.0, 100.3, 99.7, 100.0] for _ in range(10)],
        )
        bars_df.to_parquet(bar_dir / "bars.parquet")

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m", "5m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    labeling_config = _minimal_labeling_config(horizon_bars=2)
    execution_spec = _minimal_execution_spec()

    labeling_path = tmp_path / "labeling.yaml"
    execution_path = tmp_path / "execution_spec.yaml"
    with open(labeling_path, "w") as f:
        json.dump(labeling_config, f)
    with open(execution_path, "w") as f:
        json.dump(execution_spec, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        labeling_config=str(labeling_path),
        execution_spec=str(execution_path),
    )

    build_labels_command(args)

    for bar_size in ["1m", "5m"]:
        bar_dir = run_dir / f"bar_size={bar_size}"
        assert (bar_dir / "events.parquet").exists()
        assert (bar_dir / "label_schema.json").exists()

    with open(run_dir / "run_manifest.json", "r") as f:
        updated = json.load(f)

    for bar_size in ["1m", "5m"]:
        artifacts = updated["per_bar_artifacts"][bar_size]
        assert "events_path" in artifacts
        assert "label_schema_path" in artifacts
        assert "label_schema_hash" in artifacts
        assert artifacts["n_events"] > 0


def test_label_backtest_parity_harness_synthetic():
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=6, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.2, 99.8, 100.0],
            [100.0, 101.2, 99.9, 101.0],
            [101.0, 101.3, 100.7, 101.0],
            [101.0, 101.1, 100.9, 101.0],
            [101.0, 101.1, 100.9, 101.0],
            [101.0, 101.2, 100.8, 101.0],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t0": [timestamps[0], timestamps[2]],
            "t1": [timestamps[2], timestamps[4]],
            "bar_size": ["1m", "1m"],
            "side": [0, 0],
            "sigma": [1.0, 1.0],
            "pt_mult": [1.0, 1.0],
            "sl_mult": [1.0, 1.0],
            "horizon_bars": [2, 2],
        }
    )

    exec_spec = _minimal_execution_spec(slippage_ticks=1.0)
    labeling_cfg = _minimal_labeling_config(horizon_bars=2, account_for_costs=True)

    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
    )

    def parity_sim(row):
        tick_size = INSTRUMENT_SPEC.tick_size_points
        slippage = exec_spec["costs"]["slippage_ticks"]["1m"] * tick_size
        total_cost = 2.0 * slippage

        t0_idx = bars_df.index.get_loc(row["t0"])
        entry_idx = t0_idx + 1
        entry_price = bars_df.iloc[entry_idx]["open"]
        cost_buffer = total_cost
        upper = entry_price + row["pt_mult"] * row["sigma"] + cost_buffer
        lower = entry_price - row["sl_mult"] * row["sigma"] + cost_buffer

        t1_idx = bars_df.index.get_loc(row["t1"])
        touch = "vertical"
        exit_price = bars_df.iloc[t1_idx]["close"]

        for j in range(entry_idx, t1_idx + 1):
            high = bars_df.iloc[j]["high"]
            low = bars_df.iloc[j]["low"]
            if high >= upper and low <= lower:
                touch = "target"
                exit_price = upper
                break
            if high >= upper:
                touch = "target"
                exit_price = upper
                break
            if low <= lower:
                touch = "stop"
                exit_price = lower
                break

        if touch == "target":
            y = 1
        elif touch == "stop":
            y = -1
        else:
            y = 0

        ret_gross = exit_price - entry_price
        ret_net = ret_gross - total_cost
        return y, ret_net

    for i, row in labeled.iterrows():
        y_sim, ret_net_sim = parity_sim(row)
        assert row["y"] == y_sim
        assert np.isclose(row["ret_net"], ret_net_sim)


def test_triple_barrier_cost_mode_semantics(tmp_path):
    timestamps = pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min")
    bars_df = _make_bars(
        timestamps,
        [
            [100.0, 100.2, 99.8, 100.0],
            [100.0, 100.2, 99.8, 100.0],
            [100.0, 100.2, 99.8, 100.2],
        ],
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [timestamps[0]],
            "t1": [timestamps[2]],
            "bar_size": ["1m"],
            "side": [0],
            "sigma": [1.0],
            "pt_mult": [10.0],
            "sl_mult": [10.0],
            "horizon_bars": [2],
        }
    )

    exec_spec = _minimal_execution_spec(slippage_ticks=1.0)
    total_cost_points = (
        2.0
        * exec_spec["costs"]["slippage_ticks"]["1m"]
        * exec_spec["instrument"]["tick_size_points"]
    )

    # Case A: account_for_costs = false
    labeling_cfg = _minimal_labeling_config(horizon_bars=2, account_for_costs=False)
    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
    )
    assert np.isclose(labeled.loc[0, "ret_net"], labeled.loc[0, "ret_gross"])

    schema_path = tmp_path / "label_schema_gross.json"
    write_label_schema(
        output_path=schema_path,
        columns=list(labeled.columns),
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
        touch_ordering_definition="open->high->low->close",
        code_version="1.0.0",
    )
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert schema["cost_mode"] == "gross_in_events"

    # Case B: account_for_costs = true
    labeling_cfg = _minimal_labeling_config(horizon_bars=2, account_for_costs=True)
    labeled = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events_df,
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
    )
    expected_ret_net = labeled.loc[0, "ret_gross"] - total_cost_points
    assert np.isclose(labeled.loc[0, "ret_net"], expected_ret_net)

    schema_path = tmp_path / "label_schema_net.json"
    write_label_schema(
        output_path=schema_path,
        columns=list(labeled.columns),
        bar_size="1m",
        labeling_config=labeling_cfg,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
        touch_ordering_definition="open->high->low->close",
        code_version="1.0.0",
    )
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert schema["cost_mode"] == "net_in_events"
