"""
Test concurrent position limit enforcement in backtest simulator.
"""

import pandas as pd
import numpy as np
from ml_intraday_v3.backtesting_v3.simulator import run_backtest
from ml_intraday_v3.core.instrument import InstrumentSpec


def test_max_concurrent_positions():
    """Test that max_concurrent_positions limit is enforced."""

    # Create 10 events that enter at same time (so no min_time conflicts)
    # but with long duration to test concurrent position limit
    # All events try to enter at minute 0, with staggered exits
    # Only first 5 should execute due to max_concurrent_positions=5
    base_time = pd.Timestamp("2023-01-01 09:30:00", tz="America/Chicago")

    events = []
    for i in range(10):
        # All events signal at the same bar (t0), sorted by event_id
        # This ensures they're processed in order and we can test the limit
        events.append({
            'event_id': f'event_{i:02d}',  # Zero-padded for sorting
            't0': base_time,  # All signal at same time
            't1': base_time + pd.Timedelta(minutes=20),  # All have same duration
            'stop_price': 4900.0 - 5,  # Stop
            'target_price': 4900.0 + 10,  # Target
            'vertical_exit_price': 4900.0 + 5,  # Vertical
        })
    events_df = pd.DataFrame(events)

    # Create minimal bars (need to cover all event durations)
    bars = []
    for i in range(50):  # Extended to cover all events
        bars.append({
            'timestamp': base_time + pd.Timedelta(minutes=i),
            'open': 4900.0,
            'high': 4905.0,
            'low': 4895.0,
            'close': 4900.0,
            'volume': 100,
        })
    bars_df = pd.DataFrame(bars).set_index('timestamp')

    # Create predictions - all events get high probability
    preds = []
    for i in range(10):
        preds.append({
            'event_id': f'event_{i:02d}',  # Match event_id format
            'y_prob': 0.9,
            'score_ev': 0.5,  # Above threshold
        })
    preds_df = pd.DataFrame(preds)

    # Backtest config with max_concurrent_positions = 5
    backtest_cfg = {
        'decision': {
            'use_meta': False,
            'primary_score_column': 'score_ev',
            'primary_threshold': 0.10,
        },
        'sizing': {
            'contracts': 1,
            'max_concurrent_positions': 5,  # LIMIT TO 5
        },
        'costs': {
            'use_execution_spec': True,
            'use_event_ret_net': False,
        },
        'risk': {
            'use_risk_yaml': False,
        },
        'session': {},
        'outputs': {
            'write_trade_log': False,
            'write_equity_curve': False,
        },
    }

    execution_spec = {
        'entry_delay_bars': 1,
        'slippage_points': 0.25,
        'commission_per_contract': 1.00,
    }

    instrument_spec = InstrumentSpec(
        symbol='MES',
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    label_schema = {
        'cost_mode': 'gross_in_events',
    }

    risk_cfg = {
        'topstep': {'starting_balance': 50000},
        'daily_loss_limit': {'enabled': False},
        'trailing_drawdown': {'enabled': False},
        'intraday_controls': {
            'max_trades_per_day': 999,
            'min_seconds_between_trades': 0,  # Allow immediate consecutive trades
            'max_consecutive_losses': 999,
        },
    }

    # Run backtest
    trades_df, equity_df, metrics = run_backtest(
        events_df=events_df,
        bars_df=bars_df,
        primary_preds_df=preds_df,
        meta_preds_df=None,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
        label_schema=label_schema,
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        bar_size='1m',
    )

    # Verify results
    print("=" * 70)
    print("POSITION LIMIT TEST RESULTS")
    print("=" * 70)
    print(f"\nTotal events: {len(events_df)}")
    print(f"Executed trades: {trades_df['executed'].sum()}")
    print(f"Rejected trades: {(~trades_df['executed']).sum()}")

    # Check rejection reasons
    rejection_reasons = trades_df[~trades_df['executed']]['reason_skipped'].value_counts()
    print(f"\nRejection reasons:")
    for reason, count in rejection_reasons.items():
        print(f"  {reason}: {count}")

    # Verify max concurrent was enforced
    executed = trades_df[trades_df['executed']].copy()
    print(f"\nExecuted trades: {len(executed)}")

    # Check max concurrent positions at any point in time
    max_concurrent = 0
    for idx, row in executed.iterrows():
        entry = row['entry_ts']
        # Count how many positions were open at this entry time
        open_at_entry = executed[
            (executed['entry_ts'] <= entry) & (executed['exit_ts'] > entry)
        ]
        max_concurrent = max(max_concurrent, len(open_at_entry))

    print(f"Max concurrent positions observed: {max_concurrent}")

    # Assertions
    assert trades_df['executed'].sum() == 5, "Should execute exactly 5 trades"
    assert (trades_df['reason_skipped'] == 'max_concurrent_positions').sum() == 5, \
        "Should reject 5 trades due to position limit"
    assert max_concurrent <= 5, "Should never exceed 5 concurrent positions"

    print("\n✅ All position limit tests PASSED!")
    return True


if __name__ == "__main__":
    test_max_concurrent_positions()
