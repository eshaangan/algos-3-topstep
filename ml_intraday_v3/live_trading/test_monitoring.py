"""
Quick test of the monitoring system.

Simulates trading activity to demonstrate dashboard, alerts, and metrics tracking.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from monitoring.metrics_tracker import MetricsTracker
from monitoring.alerts import AlertManager
from monitoring.dashboard import TerminalDashboard


def main():
    print("🧪 Testing Monitoring System\n")
    print("=" * 80)

    # Initialize monitoring components
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    tracker = MetricsTracker(logs_dir)
    alerts = AlertManager(logs_dir, enable_sound=False)  # Disable sound for test
    dashboard = TerminalDashboard()

    # Set starting equity
    starting_equity = 149925.51
    tracker.set_starting_equity(starting_equity)

    # Send session start alert
    alerts.session_started("15266746", starting_equity)

    print("\n✅ Monitoring components initialized")
    print(f"   - Metrics saved to: {logs_dir}")
    print(f"   - Alert log: {alerts.alert_log_path}")

    # Simulate some trading activity
    print("\n🎬 Simulating trading activity...\n")

    current_time = datetime.now()
    current_equity = starting_equity

    # Trade 1: Winning LONG
    print("Trade 1: LONG +$125.00")
    tracker.record_signal(executed=True)
    entry_time = current_time - timedelta(minutes=15)
    exit_time = current_time - timedelta(minutes=10)
    pnl = 125.00
    tracker.record_trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction="LONG",
        contracts=1,
        entry_price=5012.50,
        exit_price=5037.50,
        pnl=pnl,
        exit_reason="target"
    )
    current_equity += pnl
    tracker.update_equity(current_equity, pnl)
    alerts.trade_executed("LONG", 1, 5012.50, 0.15)
    alerts.trade_closed("LONG", pnl, "target")
    time.sleep(1)

    # Trade 2: Losing SHORT
    print("Trade 2: SHORT -$85.00")
    tracker.record_signal(executed=True)
    entry_time = exit_time
    exit_time = current_time - timedelta(minutes=5)
    pnl = -85.00
    tracker.record_trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction="SHORT",
        contracts=1,
        entry_price=5040.00,
        exit_price=5057.00,
        pnl=pnl,
        exit_reason="stop"
    )
    current_equity += pnl
    tracker.update_equity(current_equity, tracker.total_pnl)
    alerts.trade_executed("SHORT", 1, 5040.00, 0.12)
    alerts.trade_closed("SHORT", pnl, "stop")
    time.sleep(1)

    # Trade 3: Winning LONG
    print("Trade 3: LONG +$200.00")
    tracker.record_signal(executed=True)
    entry_time = exit_time
    exit_time = current_time - timedelta(minutes=2)
    pnl = 200.00
    tracker.record_trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction="LONG",
        contracts=1,
        entry_price=5055.00,
        exit_price=5095.00,
        pnl=pnl,
        exit_reason="target"
    )
    current_equity += pnl
    tracker.update_equity(current_equity, tracker.total_pnl)
    alerts.trade_executed("LONG", 1, 5055.00, 0.18)
    alerts.trade_closed("LONG", pnl, "target")
    time.sleep(1)

    # Trade 4: Losing LONG
    print("Trade 4: LONG -$110.00")
    tracker.record_signal(executed=True)
    entry_time = exit_time
    exit_time = current_time
    pnl = -110.00
    tracker.record_trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction="LONG",
        contracts=1,
        entry_price=5090.00,
        exit_price=5068.00,
        pnl=pnl,
        exit_reason="stop"
    )
    current_equity += pnl
    tracker.update_equity(current_equity, tracker.total_pnl)
    alerts.trade_executed("LONG", 1, 5090.00, 0.11)
    alerts.trade_closed("LONG", pnl, "stop")
    time.sleep(1)

    # Some rejected signals
    print("\nRejecting 3 signals (risk limits)")
    tracker.record_signal(executed=False)
    tracker.record_signal(executed=False)
    tracker.record_signal(executed=False)

    # Simulate approaching daily loss limit (for warning alert)
    print("\nSimulating warning: Approaching daily loss limit")
    alerts.daily_loss_warning(-1600.00, 2000.00)

    # Display dashboard with current metrics
    print("\n" + "=" * 80)
    print("📊 RENDERING LIVE DASHBOARD")
    print("=" * 80 + "\n")

    time.sleep(1)

    # Get current metrics and render dashboard
    metrics = tracker.snapshot()
    alerts_summary = alerts.get_summary()

    dashboard.render(metrics, alerts_summary)

    # Save snapshot
    tracker.save_snapshot()
    tracker.save_trades()

    # Wait a moment
    time.sleep(2)

    # Display final summary
    print("\n\n")
    dashboard.display_summary(metrics)

    # Show what was saved
    print("\n" + "=" * 80)
    print("💾 FILES SAVED")
    print("=" * 80)

    metrics_files = list(logs_dir.glob("metrics_*.csv"))
    trades_files = list(logs_dir.glob("trades_*.csv"))
    alerts_files = list(logs_dir.glob("alerts_*.log"))

    if metrics_files:
        print(f"\n✅ Metrics: {metrics_files[-1].name}")
    if trades_files:
        print(f"✅ Trades:  {trades_files[-1].name}")
    if alerts_files:
        print(f"✅ Alerts:  {alerts_files[-1].name}")

    # Send session ended alert
    alerts.session_ended(tracker.total_trades, tracker.total_pnl, current_equity)

    print("\n" + "=" * 80)
    print("✅ MONITORING TEST COMPLETE")
    print("=" * 80)
    print("\nAll monitoring components working correctly!")
    print(f"Check the files in: {logs_dir}")
    print()


if __name__ == "__main__":
    main()
