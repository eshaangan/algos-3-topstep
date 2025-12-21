"""
Validation script for RTH feasibility checks.

Tests the bars-left feasibility logic to ensure it correctly identifies
the last feasible entry time for a given horizon.
"""

from datetime import time
import pandas as pd

from core.session_utils import (
    is_entry_feasible,
    bars_left_in_rth,
    get_last_feasible_entry_time,
)


def test_feasibility_logic():
    """Test the feasibility logic with sample timestamps."""

    # Configuration
    horizon_bars = 12
    bar_minutes = 5
    rth_start = time(8, 30)  # 8:30 AM CT
    rth_end = time(14, 55)  # 2:55 PM CT
    execution_mode = "time_exit"
    session_mode = "RTH"

    print("=" * 80)
    print("RTH FEASIBILITY VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  RTH Session: {rth_start} - {rth_end} CT")
    print(f"  Bar size: {bar_minutes} minutes")
    print(f"  Horizon: {horizon_bars} bars ({horizon_bars * bar_minutes} minutes)")
    print(f"  Execution mode: {execution_mode}")

    # Compute last feasible entry time
    last_feasible_time = get_last_feasible_entry_time(
        horizon_bars=horizon_bars,
        bar_minutes=bar_minutes,
        rth_start_time=rth_start,
        rth_end_time=rth_end,
    )

    print(f"\nLast feasible signal time: {last_feasible_time} CT")
    print(f"  → Entry at next bar ({bar_minutes} min later)")
    print(f"  → Exit after {horizon_bars} bars ({horizon_bars * bar_minutes} min hold)")
    print(f"  → Exit completes before {rth_end}")

    # Test cases for a sample day (2025-01-15)
    test_cases = [
        ("2025-01-15 13:45:00", "Should be FEASIBLE (last valid signal time)"),
        ("2025-01-15 13:50:00", "Should be INFEASIBLE (too late by 5 min)"),
        ("2025-01-15 13:55:00", "Should be INFEASIBLE (too late)"),
        ("2025-01-15 14:00:00", "Should be INFEASIBLE (too late)"),
        ("2025-01-15 14:30:00", "Should be INFEASIBLE (way too late)"),
        ("2025-01-15 14:50:00", "Should be INFEASIBLE (almost session end)"),
        ("2025-01-15 12:00:00", "Should be FEASIBLE (mid-day)"),
        ("2025-01-15 09:00:00", "Should be FEASIBLE (early morning)"),
    ]

    print("\n" + "=" * 80)
    print("TEST CASES")
    print("=" * 80)

    for ts_str, description in test_cases:
        # Create timestamp in UTC (will be converted to Chicago internally)
        ts = pd.Timestamp(ts_str, tz="America/Chicago").tz_convert("UTC")

        # Check feasibility
        is_feasible = is_entry_feasible(
            ts,
            horizon_bars=horizon_bars,
            bar_minutes=bar_minutes,
            rth_end_time=rth_end,
            execution_mode=execution_mode,
            session_mode=session_mode,
        )

        # Get bars left
        bars_left = bars_left_in_rth(
            ts,
            bar_minutes=bar_minutes,
            rth_end_time=rth_end,
        )

        # Compute entry and exit times
        ts_chicago = ts.tz_convert("America/Chicago")
        entry_time = ts_chicago + pd.Timedelta(minutes=bar_minutes)
        exit_time = entry_time + pd.Timedelta(minutes=horizon_bars * bar_minutes)

        status = "✓ FEASIBLE" if is_feasible else "✗ INFEASIBLE"
        result_icon = "✓" if (
            ("FEASIBLE" in description and is_feasible) or
            ("INFEASIBLE" in description and not is_feasible)
        ) else "✗ MISMATCH"

        print(f"\n{result_icon} Test: {ts_chicago.strftime('%H:%M:%S')} CT")
        print(f"   {description}")
        print(f"   Bars left: {bars_left}")
        print(f"   Required bars: {horizon_bars + 1}")
        print(f"   Entry time: {entry_time.strftime('%H:%M:%S')} CT")
        print(f"   Exit time: {exit_time.strftime('%H:%M:%S')} CT")
        print(f"   Result: {status}")


def test_edge_cases():
    """Test edge cases around the session boundary."""

    print("\n" + "=" * 80)
    print("EDGE CASES")
    print("=" * 80)

    horizon_bars = 12
    bar_minutes = 5
    rth_end = time(14, 55)

    # Last feasible time should be 13:45
    # Signal at 13:45, entry at 13:50, exit at bar 14:50 (which is IN session)

    ts_1345 = pd.Timestamp("2025-01-15 13:45:00", tz="America/Chicago").tz_convert("UTC")
    ts_1350 = pd.Timestamp("2025-01-15 13:50:00", tz="America/Chicago").tz_convert("UTC")

    bars_1345 = bars_left_in_rth(ts_1345, bar_minutes=bar_minutes, rth_end_time=rth_end)
    bars_1350 = bars_left_in_rth(ts_1350, bar_minutes=bar_minutes, rth_end_time=rth_end)

    feasible_1345 = is_entry_feasible(
        ts_1345,
        horizon_bars=horizon_bars,
        bar_minutes=bar_minutes,
        rth_end_time=rth_end,
        execution_mode="time_exit",
        session_mode="RTH",
    )

    feasible_1350 = is_entry_feasible(
        ts_1350,
        horizon_bars=horizon_bars,
        bar_minutes=bar_minutes,
        rth_end_time=rth_end,
        execution_mode="time_exit",
        session_mode="RTH",
    )

    print(f"\nEdge case 1: Signal at 13:45 CT")
    print(f"  Bars left after 13:45: {bars_1345}")
    print(f"  Feasible: {feasible_1345} (expected: True)")
    print(f"  Entry: 13:50, Exit bar: 14:50 (which ends at 14:55, exactly at session end)")

    print(f"\nEdge case 2: Signal at 13:50 CT")
    print(f"  Bars left after 13:50: {bars_1350}")
    print(f"  Feasible: {feasible_1350} (expected: False)")
    print(f"  Entry: 13:55, Exit bar: 14:55 (which is OUTSIDE session!)")


if __name__ == "__main__":
    test_feasibility_logic()
    test_edge_cases()
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
