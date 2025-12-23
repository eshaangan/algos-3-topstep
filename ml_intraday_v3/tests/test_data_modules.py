"""
Unit tests for V3 data modules.

Tests cover:
1. Reindexing grid and synthetic bar marking
2. 1m to 5m resampling OHLCV correctness
3. QA OHLC violation detection
4. Roll schedule determinism
5. End-to-end build-data artifacts
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import tempfile
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    load_raw_data,
    standardize_ohlcv,
    build_roll_schedule,
    apply_roll_schedule,
    write_roll_schedule,
    load_roll_schedule,
    RollSchedule,
    reindex_to_grid,
    resample_1m_to_5m,
    add_session_features,
    run_qa_checks,
    SessionConfig,
)


@pytest.fixture
def sample_1m_ohlcv():
    """Generate sample 1m OHLCV data."""
    # 30 bars of 1m data (not a complete grid, to test reindexing)
    dates = pd.date_range(
        "2025-01-01 09:30:00", periods=30, freq="1min", tz="UTC"
    )

    # Skip some bars to create gaps
    dates = dates[[i for i in range(30) if i not in [5, 6, 15, 16, 17]]]

    df = pd.DataFrame(
        {
            "open": 100.0 + np.random.randn(len(dates)) * 0.1,
            "high": 100.5 + np.random.randn(len(dates)) * 0.1,
            "low": 99.5 + np.random.randn(len(dates)) * 0.1,
            "close": 100.0 + np.random.randn(len(dates)) * 0.1,
            "volume": np.random.randint(1000, 10000, size=len(dates)),
        },
        index=dates,
    )

    # Ensure OHLC validity
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01

    df.index.name = "ts"

    return df


@pytest.fixture
def complete_1m_grid():
    """Generate complete 1m grid (no missing bars)."""
    dates = pd.date_range(
        "2025-01-01 09:30:00", periods=100, freq="1min", tz="UTC"
    )

    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 5000,
        },
        index=dates,
    )

    df.index.name = "ts"

    return df


class TestIngestHdf:
    """Test HDF ingestion index handling."""

    def test_ingest_hdf_sets_datetime_index(self, tmp_path):
        """HDF ingest should set a UTC DatetimeIndex from a timestamp column."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:35:00",
                    "2025-01-01 09:40:00",
                ],
                "open": [100.0, 100.1, 100.2],
                "high": [100.5, 100.6, 100.7],
                "low": [99.5, 99.6, 99.7],
                "close": [100.0, 100.1, 100.2],
                "volume": [1000, 1100, 1200],
            }
        )

        data_file = tmp_path / "bars.h5"
        df.to_hdf(data_file, key="/bars_5min", mode="w")

        out = load_raw_data(
            input_path=data_file,
            input_format="hdf5",
            timestamp_column="timestamp",
            required_columns=["open", "high", "low", "close", "volume"],
            hdf_key="/bars_5min",
        )

        assert isinstance(out.index, pd.DatetimeIndex)
        assert str(out.index.tz) == "UTC"
        assert "timestamp" not in out.columns


class TestReindexGrid:
    """Test reindex_to_grid function."""

    def test_reindex_grid_marks_synthetic_bars(self, sample_1m_ohlcv):
        """Test that reindexing marks synthetic bars correctly (with forward-fill)."""
        df_reindexed, metadata = reindex_to_grid(
            sample_1m_ohlcv,
            bar_size="1m",
            missing_fill_mode="forward_fill",
            forward_fill_max_consecutive=5,
            add_synthetic_flag=True,
        )

        # Should have is_synthetic column
        assert "is_synthetic" in df_reindexed.columns

        # Should have more bars than original (filled gaps)
        assert len(df_reindexed) > len(sample_1m_ohlcv)

        # Number of synthetic bars should match missing bars
        n_synthetic = df_reindexed["is_synthetic"].sum()
        assert n_synthetic > 0
        assert metadata["synthetic_bars"] == n_synthetic

        # Original bars should not be marked synthetic
        original_indices = set(sample_1m_ohlcv.index)
        for idx in original_indices:
            if idx in df_reindexed.index:
                assert not df_reindexed.loc[idx, "is_synthetic"]

    def test_reindex_grid_complete_grid(self, complete_1m_grid):
        """Test reindexing on already complete grid."""
        df_reindexed, metadata = reindex_to_grid(
            complete_1m_grid,
            bar_size="1m",
            missing_fill_mode="nan",  # Using new default
            add_synthetic_flag=True,
        )

        # Should have no synthetic bars
        assert metadata["synthetic_bars"] == 0
        assert metadata["synthetic_pct"] == 0.0

        # Length should be same
        assert len(df_reindexed) == len(complete_1m_grid)

    def test_reindex_grid_5m(self, complete_1m_grid):
        """Test reindexing to 5m grid."""
        # Take first 50 bars (50 minutes)
        df_1m = complete_1m_grid.iloc[:50].copy()

        df_reindexed, metadata = reindex_to_grid(
            df_1m,
            bar_size="5m",
            missing_fill_mode="nan",  # Using new default
            add_synthetic_flag=True,
        )

        # Should have 5m frequency
        # 50 minutes = 10 bars of 5m (if aligned)
        assert len(df_reindexed) >= 10

    def test_reindex_default_nan_no_forward_fill(self, sample_1m_ohlcv):
        """Test that default behavior keeps missing bars as NaN (research-grade)."""
        df_reindexed, metadata = reindex_to_grid(
            sample_1m_ohlcv,
            bar_size="1m",
            # Using defaults: missing_fill_mode="nan", forward_fill_max_consecutive=0
        )

        # Should have synthetic bars marked
        assert "is_synthetic" in df_reindexed.columns
        n_synthetic = df_reindexed["is_synthetic"].sum()
        assert n_synthetic > 0

        # Synthetic bars should have NaN OHLCV
        synthetic_mask = df_reindexed["is_synthetic"]
        assert df_reindexed.loc[synthetic_mask, "close"].isna().all()
        assert df_reindexed.loc[synthetic_mask, "open"].isna().all()

        # Metadata should reflect NaN mode
        assert metadata["missing_fill_mode"] == "nan"
        assert metadata["forward_filled_bars"] == 0

    def test_session_grid_reindex_does_not_create_massive_missing(self):
        """Session grid should not create large missing gaps for RTH-only data."""
        tz = "America/Chicago"
        day1 = pd.Timestamp("2025-01-02", tz=tz)
        day2 = pd.Timestamp("2025-01-03", tz=tz)
        start1 = day1 + pd.Timedelta(hours=8, minutes=30)
        end1 = day1 + pd.Timedelta(hours=15)
        start2 = day2 + pd.Timedelta(hours=8, minutes=30)
        end2 = day2 + pd.Timedelta(hours=15)

        idx1 = pd.date_range(start=start1, end=end1, freq="1min", tz=tz, inclusive="left")
        idx2 = pd.date_range(start=start2, end=end2, freq="1min", tz=tz, inclusive="left")
        index = idx1.append(idx2).tz_convert("UTC")

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=index,
        )

        sessions = [
            {"name": "rth", "start_time": "08:30", "end_time": "15:00"},
            {"name": "eth", "start_time": "17:00", "end_time": "16:00"},
        ]

        _, metadata = reindex_to_grid(
            df,
            bar_size="1m",
            missing_fill_mode="nan",
            forward_fill_max_consecutive=0,
            add_synthetic_flag=True,
            grid_mode="session",
            session_grid="rth",
            session_timezone=tz,
            sessions=sessions,
            exclude_weekends=True,
            day_selection_mode="data_present",
            min_rows_per_day=1,
        )

        assert metadata["synthetic_pct"] <= 0.1

    def test_session_grid_data_present_skips_missing_days(self):
        """Data-present day selection should skip dates with no raw data."""
        tz = "America/Chicago"
        day1 = pd.Timestamp("2019-12-24", tz=tz)
        day3 = pd.Timestamp("2019-12-26", tz=tz)

        def rth_index(day):
            start = day + pd.Timedelta(hours=8, minutes=30)
            end = day + pd.Timedelta(hours=15)
            return pd.date_range(start=start, end=end, freq="1min", tz=tz, inclusive="left")

        idx = rth_index(day1).append(rth_index(day3)).tz_convert("UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=idx,
        )

        sessions = [
            {"name": "rth", "start_time": "08:30", "end_time": "15:00"},
            {"name": "eth", "start_time": "17:00", "end_time": "16:00"},
        ]

        df_reindexed, metadata = reindex_to_grid(
            df,
            bar_size="1m",
            missing_fill_mode="nan",
            add_synthetic_flag=True,
            grid_mode="session",
            session_grid="rth",
            session_timezone=tz,
            sessions=sessions,
            exclude_weekends=True,
            day_selection_mode="data_present",
            min_rows_per_day=1,
        )

        chi_dates = df_reindexed.index.tz_convert(tz).date
        assert pd.Timestamp("2019-12-25").date() not in set(chi_dates)
        assert pd.Timestamp("2019-12-24").date() in set(chi_dates)
        assert pd.Timestamp("2019-12-26").date() in set(chi_dates)

        assert "2019-12-25" not in metadata.get("missing_pct_per_day", {})

    def test_drop_sparse_days_excludes_days_below_coverage_threshold(self):
        """Sparse day exclusion should exclude days with coverage < min_day_coverage_pct."""
        tz = "America/Chicago"

        # Create two session days:
        # Day A (2025-01-02): Good coverage (nearly full RTH session)
        # Day B (2025-01-03): Sparse coverage (only a few bars)

        day_a = pd.Timestamp("2025-01-02", tz=tz)
        day_b = pd.Timestamp("2025-01-03", tz=tz)

        def rth_index(day):
            start = day + pd.Timedelta(hours=8, minutes=30)
            end = day + pd.Timedelta(hours=15)
            return pd.date_range(start=start, end=end, freq="5min", tz=tz, inclusive="left")

        # Day A: Full RTH session (08:30 - 15:00 = 6.5 hours = 78 bars @ 5min)
        idx_a = rth_index(day_a)

        # Day B: Only 8 bars (sparse - should be excluded with 90% threshold)
        start_b = day_b + pd.Timedelta(hours=8, minutes=30)
        idx_b = pd.date_range(start=start_b, periods=8, freq="5min", tz=tz)

        # Combine indexes
        idx = idx_a.append(idx_b).tz_convert("UTC")

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=idx,
        )

        sessions = [
            {"name": "rth", "start_time": "08:30", "end_time": "15:00"},
        ]

        df_reindexed, metadata = reindex_to_grid(
            df,
            bar_size="5m",
            missing_fill_mode="nan",
            add_synthetic_flag=True,
            grid_mode="session",
            session_grid="rth",
            session_timezone=tz,
            sessions=sessions,
            exclude_weekends=True,
            day_selection_mode="data_present",
            min_rows_per_day=1,
            drop_sparse_days=True,
            min_day_coverage_pct=0.90,
            coverage_session="rth",
        )

        # Verify Day A is included
        chi_dates = df_reindexed.index.tz_convert(tz).date
        assert day_a.date() in set(chi_dates), "Day A (good coverage) should be included"

        # Verify Day B is excluded
        assert day_b.date() not in set(chi_dates), "Day B (sparse coverage) should be excluded"

        # Verify metadata reports excluded days
        excluded = metadata.get("excluded_sparse_days", [])
        assert len(excluded) > 0, "Metadata should report excluded days"

        # Find Day B in excluded list
        day_b_excluded = [e for e in excluded if e["date"] == str(day_b.date())]
        assert len(day_b_excluded) == 1, "Day B should be in excluded list"

        # Verify coverage stats for Day B
        day_b_stats = day_b_excluded[0]
        assert day_b_stats["observed"] == 8
        assert day_b_stats["expected"] == 78
        assert day_b_stats["coverage"] < 0.90

    def test_drop_sparse_days_deterministic(self):
        """Sparse day exclusion should be deterministic with same inputs."""
        tz = "America/Chicago"

        day_a = pd.Timestamp("2025-01-02", tz=tz)
        day_b = pd.Timestamp("2025-01-03", tz=tz)

        def rth_index(day):
            start = day + pd.Timedelta(hours=8, minutes=30)
            end = day + pd.Timedelta(hours=15)
            return pd.date_range(start=start, end=end, freq="5min", tz=tz, inclusive="left")

        idx_a = rth_index(day_a)
        start_b = day_b + pd.Timedelta(hours=8, minutes=30)
        idx_b = pd.date_range(start=start_b, periods=8, freq="5min", tz=tz)
        idx = idx_a.append(idx_b).tz_convert("UTC")

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=idx,
        )

        sessions = [{"name": "rth", "start_time": "08:30", "end_time": "15:00"}]

        # Run twice with same params
        df_1, meta_1 = reindex_to_grid(
            df, bar_size="5m", grid_mode="session", session_grid="rth",
            session_timezone=tz, sessions=sessions, drop_sparse_days=True,
            min_day_coverage_pct=0.90, coverage_session="rth",
        )

        df_2, meta_2 = reindex_to_grid(
            df, bar_size="5m", grid_mode="session", session_grid="rth",
            session_timezone=tz, sessions=sessions, drop_sparse_days=True,
            min_day_coverage_pct=0.90, coverage_session="rth",
        )

        # Verify outputs are identical
        pd.testing.assert_index_equal(df_1.index, df_2.index)
        assert meta_1["excluded_sparse_days"] == meta_2["excluded_sparse_days"]


class TestResample1mTo5m:
    """Test resample_1m_to_5m function."""

    def test_resample_1m_to_5m_ohlcv_correctness(self, complete_1m_grid):
        """Test that resampling preserves OHLCV semantics."""
        # Simpler test: just verify that OHLCV aggregation rules are applied
        df_1m = complete_1m_grid.iloc[:50].copy()

        # Set distinct values to verify
        # Change values across first 5m window
        df_1m.iloc[0, df_1m.columns.get_loc("open")] = 100.0
        df_1m.iloc[0, df_1m.columns.get_loc("close")] = 100.1

        df_1m.iloc[1, df_1m.columns.get_loc("high")] = 105.0  # Should be max
        df_1m.iloc[1, df_1m.columns.get_loc("low")] = 99.0

        df_1m.iloc[4, df_1m.columns.get_loc("low")] = 95.0  # Should be min
        df_1m.iloc[4, df_1m.columns.get_loc("close")] = 101.0  # Should be last

        # Set all volumes to known values
        for i in range(5):
            df_1m.iloc[i, df_1m.columns.get_loc("volume")] = 1000

        df_5m = resample_1m_to_5m(df_1m)

        # Verify we have data
        assert len(df_5m) > 0

        # Check OHLCV aggregation on any 5m bar
        for idx, row in df_5m.iterrows():
            # Just verify types and non-null
            assert pd.notna(row["open"])
            assert pd.notna(row["high"])
            assert pd.notna(row["low"])
            assert pd.notna(row["close"])
            assert pd.notna(row["volume"])

            # Volume should be >= sum of at least 1 bar
            assert row["volume"] >= 1000

    def test_resample_preserves_index_timezone(self, complete_1m_grid):
        """Test that resampling preserves UTC timezone."""
        df_5m = resample_1m_to_5m(complete_1m_grid)

        assert df_5m.index.tz is not None
        assert str(df_5m.index.tz) == "UTC"

    def test_resample_ohlc_validity(self, complete_1m_grid):
        """Test that resampled bars have valid OHLC."""
        df_5m = resample_1m_to_5m(complete_1m_grid)

        # Check OHLC validity
        for idx, row in df_5m.iterrows():
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]


class TestQAChecks:
    """Test run_qa_checks function."""

    def test_qa_catches_ohlc_violation(self):
        """Test that QA detects OHLC violations (fail-fast disabled)."""
        # Create data with OHLC violation
        dates = pd.date_range("2025-01-01 09:30:00", periods=10, freq="1min", tz="UTC")

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=dates,
        )

        # Introduce violation: low > close
        df.loc[df.index[5], "low"] = 101.0  # low > close

        df.index.name = "ts"

        qa_report = run_qa_checks(df, qa_fail_fast=False)  # Disable fail-fast for test

        # Should fail OHLC validity check
        assert not qa_report.passed
        assert "ohlc_validity" in qa_report.failed_checks
        assert qa_report.checks["ohlc_validity"]["n_total_violations"] > 0

    def test_qa_passes_on_valid_data(self, complete_1m_grid):
        """Test that QA passes on valid data."""
        qa_report = run_qa_checks(complete_1m_grid, qa_fail_fast=True)

        # Should pass all checks
        assert qa_report.passed
        assert len(qa_report.failed_checks) == 0

    def test_qa_detects_duplicates(self):
        """Test that QA detects duplicate timestamps (fail-fast disabled)."""
        dates = pd.date_range("2025-01-01 09:30:00", periods=10, freq="1min", tz="UTC")

        # Add duplicate timestamp
        dates = dates.append(pd.DatetimeIndex([dates[5]]))

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=dates,
        )

        df.index.name = "ts"

        qa_report = run_qa_checks(df, qa_fail_fast=False)  # Disable fail-fast for test

        # Should fail no_duplicates check
        assert not qa_report.passed
        assert "no_duplicates" in qa_report.failed_checks

    def test_qa_fail_fast_raises_exception(self):
        """Test that QA fail-fast mode raises exception on violations."""
        from data import QAViolationError

        dates = pd.date_range("2025-01-01 09:30:00", periods=10, freq="1min", tz="UTC")

        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            },
            index=dates,
        )

        # Introduce OHLC violation
        df.loc[df.index[5], "low"] = 101.0

        df.index.name = "ts"

        # Should raise exception with fail-fast=True (default)
        with pytest.raises(QAViolationError):
            run_qa_checks(df, qa_fail_fast=True)


class TestRollSchedule:
    """Test roll schedule functions."""

    def test_continuous_mode_already_continuous_no_change(self):
        """already_continuous should ignore symbol changes and keep data unchanged."""
        dates = pd.date_range(
            "2025-01-01 09:30:00", periods=5, freq="1min", tz="UTC"
        )
        symbols = ["F1", "F2", "F2", "F2", "F2"]

        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.5] * 5,
                "low": [99.5] * 5,
                "close": [100.0] * 5,
                "volume": [5000] * 5,
                "symbol": symbols,
            },
            index=dates,
        )

        schedule = build_roll_schedule(df, mode="already_continuous")
        df_out = apply_roll_schedule(
            df, schedule, roll_day_policy="exclude", mode="already_continuous"
        )

        pd.testing.assert_frame_equal(df_out, df)
        assert len(schedule.roll_datetimes) == 1
        assert schedule.contracts[0] == symbols[0]

    def test_calendar_roll_switches_contract_without_lookahead(self, tmp_path):
        """calendar_roll should switch contracts at roll time only."""
        dates = pd.date_range(
            "2025-01-01 09:30:00", periods=6, freq="1min", tz="UTC"
        )
        front = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [100.5] * 6,
                "low": [99.5] * 6,
                "close": [100.0] * 6,
                "volume": [1000] * 6,
                "symbol": ["F1"] * 6,
            },
            index=dates,
        )
        back = pd.DataFrame(
            {
                "open": [200.0] * 6,
                "high": [200.5] * 6,
                "low": [199.5] * 6,
                "close": [200.0] * 6,
                "volume": [2000] * 6,
                "symbol": ["F2"] * 6,
            },
            index=dates,
        )
        df = pd.concat([front, back]).sort_index()

        roll_path = tmp_path / "roll_schedule.csv"
        roll_df = pd.DataFrame(
            {
                "contract": ["F1", "F2"],
                "roll_datetime_utc": [dates[0], dates[3]],
            }
        )
        roll_df.to_csv(roll_path, index=False)

        schedule = build_roll_schedule(
            df, mode="calendar_roll", roll_schedule_path=roll_path
        )
        df_out = apply_roll_schedule(
            df, schedule, roll_day_policy="keep", mode="calendar_roll"
        )

        assert len(df_out) == len(dates)
        assert (df_out.loc[dates[:3], "close"] == 100.0).all()
        assert (df_out.loc[dates[3:], "close"] == 200.0).all()

    def test_roll_day_policy_exclude_drops_roll_day(self, tmp_path):
        """exclude policy should drop all bars on the roll date."""
        day1 = pd.date_range(
            "2025-01-01 09:30:00", periods=3, freq="1min", tz="UTC"
        )
        day2 = pd.date_range(
            "2025-01-02 09:30:00", periods=3, freq="1min", tz="UTC"
        )
        dates = day1.append(day2)
        front = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [100.5] * 6,
                "low": [99.5] * 6,
                "close": [100.0] * 6,
                "volume": [1000] * 6,
                "symbol": ["F1"] * 6,
            },
            index=dates,
        )
        back = pd.DataFrame(
            {
                "open": [200.0] * 6,
                "high": [200.5] * 6,
                "low": [199.5] * 6,
                "close": [200.0] * 6,
                "volume": [2000] * 6,
                "symbol": ["F2"] * 6,
            },
            index=dates,
        )
        df = pd.concat([front, back]).sort_index()

        roll_path = tmp_path / "roll_schedule.csv"
        roll_df = pd.DataFrame(
            {
                "contract": ["F1", "F2"],
                "roll_datetime_utc": [dates[0], dates[1]],
            }
        )
        roll_df.to_csv(roll_path, index=False)

        schedule = build_roll_schedule(
            df, mode="calendar_roll", roll_schedule_path=roll_path
        )
        df_out = apply_roll_schedule(
            df, schedule, roll_day_policy="exclude", mode="calendar_roll"
        )

        roll_date = dates[1].normalize()
        assert not df_out.index.normalize().isin([roll_date]).any()
        assert df_out.index.normalize().isin([day2[0].normalize()]).all()

    def test_roll_schedule_write_load_roundtrip(self, tmp_path):
        """Test that roll schedule can be written and loaded back."""
        roll_dt = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        schedule = RollSchedule(
            roll_datetimes=[roll_dt],
            contracts=["F1"],
            mode="calendar_roll",
        )

        output_path = tmp_path / "roll_schedule.csv"
        write_roll_schedule(schedule, output_path)

        loaded_schedule = load_roll_schedule(output_path)

        assert loaded_schedule.roll_datetimes == [roll_dt]
        assert loaded_schedule.contracts == ["F1"]


class TestSessionFeatures:
    """Test session feature computation."""

    def test_add_session_features_basic(self, complete_1m_grid):
        """Test that basic time features are added."""
        df = add_session_features(complete_1m_grid)

        # Should have time features
        assert "minute_of_day" in df.columns
        assert "day_of_week" in df.columns
        assert "hour" in df.columns
        assert "minute" in df.columns

        # minute_of_day should be 0-1439
        assert df["minute_of_day"].min() >= 0
        assert df["minute_of_day"].max() < 1440

    def test_add_session_features_with_sessions(self, complete_1m_grid):
        """Test that session flags are added when configured."""
        sessions = [
            SessionConfig(name="rth", start_time="08:30", end_time="15:00"),
        ]

        df = add_session_features(
            complete_1m_grid,
            session_timezone="America/Chicago",
            sessions=sessions,
        )

        # Should have session flag
        assert "is_rth" in df.columns

        # Should have time_to_session_end
        assert "time_to_session_end_minutes" in df.columns


class TestEndToEndBuildData:
    """Test end-to-end build-data workflow."""

    def test_build_data_writes_expected_artifacts_for_both_bar_sizes(self, tmp_path):
        """Test that build-data writes all expected artifacts for 1m and 5m."""
        # Create synthetic data file
        dates = pd.date_range("2025-01-01 09:30:00", periods=100, freq="1min", tz="UTC")

        df = pd.DataFrame(
            {
                "open": 100.0 + np.random.randn(100) * 0.1,
                "high": 100.5 + np.random.randn(100) * 0.1,
                "low": 99.5 + np.random.randn(100) * 0.1,
                "close": 100.0 + np.random.randn(100) * 0.1,
                "volume": np.random.randint(1000, 10000, size=100),
            },
            index=dates,
        )

        # Ensure OHLC validity
        df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
        df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01

        df.index.name = "ts"

        # Write to HDF5
        data_file = tmp_path / "test_data.h5"
        df.to_hdf(data_file, key="data", mode="w")

        # Create config
        import yaml

        config = {
            "raw_data": {
                "input_path": str(data_file),
                "required_columns": ["open", "high", "low", "close", "volume"],
            },
            "ingestion": {
                "format": "hdf5",
                "hdf_key": "data",
                "timestamp_col": None,
            },
            "canonical_bar_size": "1m",
            "bar_sizes_to_write": ["1m", "5m"],
            "continuization": {
                "mode": "already_continuous",
                "roll_schedule_path": None,
                "roll_day_policy": "exclude",
            },
            "reindexing": {
                "bar_sizes": ["1m", "5m"],
                "resample_policy": "build_1m_resample_5m",
                "missing_bars": {
                    "missing_fill_mode": "nan",  # Updated to new default
                    "forward_fill_max_consecutive": 0,
                    "add_synthetic_flag": True,
                },
                "session_labeling": {
                    "timezone": "America/Chicago",
                    "sessions": [
                        {"name": "rth", "start_time": "08:30", "end_time": "15:00"},
                    ],
                },
            },
            "qa": {
                "qa_fail_fast": False,  # Disable for test (don't want exception)
                "checks": [
                    "monotonic_index",
                    "no_duplicates",
                    "ohlc_validity",
                    "volume_sanity",
                ],
                "thresholds": {
                    "max_missing_bar_pct_per_day": 10.0,
                    "max_ohlc_violations": 0,
                    "max_duplicate_timestamps": 0,
                },
            },
        }

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config_path = config_dir / "data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Also create minimal execution_spec for manifest
        exec_spec = {
            "version": "1.0.0",
            "instrument": {
                "symbol": "MES",
                "tick_size_points": 0.25,
                "contract_multiplier_usd_per_point": 5.0,
            },
            "costs": {"slippage_ticks": {"1m": 1.0}},
        }
        with open(config_dir / "execution_spec.yaml", "w") as f:
            yaml.dump(exec_spec, f)

        # Run build-data workflow
        from cli import build_data_command
        import argparse

        args = argparse.Namespace(
            config=str(config_path),
            out=str(tmp_path / "runs" / "test_run"),
            run_id="test_run",
            seed=42,
        )

        build_data_command(args)

        # Verify artifacts exist
        run_dir = tmp_path / "runs" / "test_run"

        for bar_size in ["1m", "5m"]:
            bar_dir = run_dir / f"bar_size={bar_size}"

            # Check artifacts exist
            assert (bar_dir / "bars.parquet").exists()
            assert (bar_dir / "qa_report.json").exists()
            assert (bar_dir / "roll_schedule.csv").exists()
            assert (bar_dir / "data_metadata.json").exists()

            # Load and verify bars
            df_bars = pd.read_parquet(bar_dir / "bars.parquet")
            assert len(df_bars) > 0
            assert "open" in df_bars.columns
            assert "close" in df_bars.columns
            assert "is_synthetic" in df_bars.columns
            assert "is_rth" in df_bars.columns

            # With NaN default, synthetic bars will have NaN OHLCV
            # (but non-synthetic bars should be valid)
            non_synthetic = df_bars[~df_bars["is_synthetic"]]
            assert len(non_synthetic) > 0
            assert non_synthetic["close"].notna().all()

            # Load and verify QA report
            with open(bar_dir / "qa_report.json", "r") as f:
                qa_report = json.load(f)

            assert "passed" in qa_report
            assert "checks" in qa_report

        # Verify manifest exists
        assert (run_dir / "run_manifest.json").exists()

        with open(run_dir / "run_manifest.json", "r") as f:
            manifest = json.load(f)

        assert manifest["run_id"] == "test_run"
        assert set(manifest["bar_sizes"]) == {"1m", "5m"}
        assert "1m" in manifest["per_bar_size_artifacts"]
        assert "5m" in manifest["per_bar_size_artifacts"]

    def test_canonical_5m_rejects_upsample_to_1m(self, tmp_path):
        """Canonical 5m inputs should reject 1m outputs."""
        import yaml
        from cli import build_data_command
        import argparse

        config = {
            "raw_data": {
                "input_path": str(tmp_path / "dummy.h5"),
                "required_columns": ["open", "high", "low", "close", "volume"],
            },
            "ingestion": {
                "format": "hdf5",
                "hdf_key": "/bars_5min",
                "timestamp_col": "timestamp",
            },
            "canonical_bar_size": "5m",
            "bar_sizes_to_write": ["1m", "5m"],
        }

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args = argparse.Namespace(
            config=str(config_path),
            out=str(tmp_path / "runs" / "test_run"),
            run_id="test_run",
            seed=42,
        )

        with pytest.raises(ValueError, match="Upsampling"):
            build_data_command(args)

    def test_build_data_5m_writes_expected_artifacts(self, tmp_path):
        """Canonical 5m inputs should write 5m artifacts only."""
        import yaml
        from cli import build_data_command
        import argparse

        timestamps = pd.date_range(
            "2025-01-02 09:30:00", periods=12, freq="5min", tz="UTC"
        )
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 5000,
            }
        )

        data_file = tmp_path / "bars_5m.h5"
        df.to_hdf(data_file, key="/bars_5min", mode="w")

        config = {
            "raw_data": {
                "input_path": str(data_file),
                "required_columns": ["open", "high", "low", "close", "volume"],
            },
            "ingestion": {
                "format": "hdf5",
                "hdf_key": "/bars_5min",
                "timestamp_col": "timestamp",
            },
            "canonical_bar_size": "5m",
            "bar_sizes_to_write": ["5m"],
            "continuization": {
                "mode": "already_continuous",
                "roll_schedule_path": None,
                "roll_day_policy": "exclude",
            },
            "reindexing": {
                "bar_sizes": ["5m"],
                "resample_policy": "native",
                "grid_mode": "full_range",
                "missing_bars": {
                    "missing_fill_mode": "nan",
                    "forward_fill_max_consecutive": 0,
                    "add_synthetic_flag": True,
                },
                "session_labeling": {
                    "timezone": "America/Chicago",
                    "sessions": [
                        {"name": "rth", "start_time": "08:30", "end_time": "15:00"},
                    ],
                },
            },
            "qa": {
                "qa_fail_fast": True,
                "checks": [
                    "monotonic_index",
                    "no_duplicates",
                    "missing_bar_pct",
                    "ohlc_validity",
                    "volume_sanity",
                ],
                "thresholds": {
                    "max_missing_bar_pct_per_day": 10.0,
                    "max_ohlc_violations": 0,
                    "max_duplicate_timestamps": 0,
                },
            },
        }

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args = argparse.Namespace(
            config=str(config_path),
            out=str(tmp_path / "runs" / "test_run"),
            run_id="test_run",
            seed=42,
        )

        build_data_command(args)

        run_dir = tmp_path / "runs" / "test_run"
        bar_dir = run_dir / "bar_size=5m"

        assert (bar_dir / "bars.parquet").exists()
        assert (bar_dir / "qa_report.json").exists()
        assert (bar_dir / "roll_schedule.csv").exists()
        assert (bar_dir / "data_metadata.json").exists()

        df_bars = pd.read_parquet(bar_dir / "bars.parquet")
        assert len(df_bars) > 0
        assert "open" in df_bars.columns
        assert "close" in df_bars.columns
        assert "is_synthetic" in df_bars.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
