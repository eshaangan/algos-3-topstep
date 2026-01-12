"""
Smoke tests for Kelly Criterion integration with live trading system.

Tests basic imports, initialization, and integration points without requiring
live market data or full replay runs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import yaml
from datetime import datetime

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


def test_kelly_sizer_imports():
    """Test that KellySizer module can be imported."""
    try:
        from live_trading.kelly_sizer import KellySizer
        print("✓ KellySizer import successful")
        return True
    except Exception as e:
        print(f"✗ KellySizer import failed: {e}")
        return False


def test_execution_engine_imports():
    """Test that LiveExecutionEngine can be imported."""
    try:
        from live_trading.execution_engine import LiveExecutionEngine
        print("✓ LiveExecutionEngine import successful")
        return True
    except Exception as e:
        print(f"✗ LiveExecutionEngine import failed: {e}")
        return False


def test_live_runner_imports():
    """Test that LiveTradingRunner can be imported."""
    try:
        from live_trading.live_runner import LiveTradingRunner
        print("✓ LiveTradingRunner import successful")
        return True
    except Exception as e:
        print(f"✗ LiveTradingRunner import failed: {e}")
        return False


def test_metrics_tracker_kelly_methods():
    """Test that MetricsTracker has Kelly logging methods."""
    try:
        from monitoring.metrics_tracker import MetricsTracker
        from pathlib import Path
        import tempfile

        # Create temporary directory for metrics
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(output_dir=Path(tmpdir))

            # Verify Kelly methods exist
            assert hasattr(tracker, 'kelly_sizing_log'), "kelly_sizing_log attribute missing"
            assert hasattr(tracker, 'record_kelly_decision'), "record_kelly_decision method missing"
            assert hasattr(tracker, 'save_kelly_log'), "save_kelly_log method missing"

            # Test recording a Kelly decision
            test_decision = {
                'timestamp': datetime.now(),
                'contracts': 2,
                'reason': 'kelly_0.137_score_0.183',
                'raw_kelly': 0.55,
                'fractional_kelly': 0.1375,
                'score_ev': 0.183,
            }
            tracker.record_kelly_decision(test_decision)

            assert len(tracker.kelly_sizing_log) == 1, "Kelly decision not recorded"
            print("✓ MetricsTracker Kelly methods verified")
            return True

    except Exception as e:
        print(f"✗ MetricsTracker Kelly methods test failed: {e}")
        return False


def test_kelly_sizer_initialization():
    """Test KellySizer initialization with config."""
    try:
        from live_trading.kelly_sizer import KellySizer

        # Test with default config
        config = {
            'enabled': False,
            'min_trades_for_kelly': 20,
            'kelly_fraction': 0.25,
            'rolling_window_trades': 50,
            'max_contracts_per_trade': 5,
            'min_contracts': 1,
            'confidence_boost': {
                'enabled': True,
                'boost_factor': 1.5,
                'boost_threshold': 0.15,
            },
            'negative_kelly_threshold': 3,
            'log_sizing_decisions': False,
        }

        sizer = KellySizer(config)

        # Verify initialization
        assert sizer.config == config, "Config not stored correctly"
        assert sizer.negative_kelly_count == 0, "negative_kelly_count should initialize to 0"
        assert sizer.last_kelly_fraction == 0.0, "last_kelly_fraction should initialize to 0.0"
        assert sizer.last_position_size == 1, "last_position_size should initialize to 1"
        assert sizer.trades_seen == 0, "trades_seen should initialize to 0"

        print("✓ KellySizer initialization successful")
        return True

    except Exception as e:
        print(f"✗ KellySizer initialization failed: {e}")
        return False


def test_kelly_sizer_disabled_behavior():
    """Test that KellySizer returns 1 contract when disabled."""
    try:
        from live_trading.kelly_sizer import KellySizer

        config = {
            'enabled': False,  # Disabled
            'min_trades_for_kelly': 20,
            'kelly_fraction': 0.25,
            'rolling_window_trades': 50,
            'max_contracts_per_trade': 5,
            'min_contracts': 1,
            'confidence_boost': {'enabled': True, 'boost_factor': 1.5, 'boost_threshold': 0.15},
            'negative_kelly_threshold': 3,
            'log_sizing_decisions': False,
        }

        sizer = KellySizer(config)

        # Generate fake trade history
        trades = [{'pnl': 100.0} for _ in range(30)]

        # Should return 1 contract when disabled
        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        assert contracts == 1, f"Expected 1 contract when disabled, got {contracts}"
        assert reason == "disabled", f"Expected 'disabled' reason, got '{reason}'"

        print("✓ KellySizer disabled behavior correct")
        return True

    except Exception as e:
        print(f"✗ KellySizer disabled behavior test failed: {e}")
        return False


def test_config_files_exist():
    """Test that all required config files exist and have Kelly sections."""
    try:
        config_dir = Path(__file__).parent.parent / 'configs'

        # Check live_trading.yaml
        live_config_path = config_dir / 'live_trading.yaml'
        assert live_config_path.exists(), f"live_trading.yaml not found at {live_config_path}"

        with open(live_config_path) as f:
            live_cfg = yaml.safe_load(f)

        assert 'kelly_sizing' in live_cfg, "kelly_sizing section missing from live_trading.yaml"
        assert 'enabled' in live_cfg['kelly_sizing'], "enabled field missing from kelly_sizing"

        print(f"✓ live_trading.yaml has kelly_sizing section (enabled={live_cfg['kelly_sizing']['enabled']})")

        # Check risk.yaml
        risk_config_path = config_dir / 'risk.yaml'
        assert risk_config_path.exists(), f"risk.yaml not found at {risk_config_path}"

        with open(risk_config_path) as f:
            risk_cfg = yaml.safe_load(f)

        assert 'position_limits' in risk_cfg, "position_limits section missing from risk.yaml"
        max_per_position = risk_cfg['position_limits']['max_contracts_per_position']
        max_concurrent = risk_cfg['position_limits']['max_concurrent_positions']

        assert max_per_position == 5, f"Expected max_contracts_per_position=5, got {max_per_position}"
        assert max_concurrent == 15, f"Expected max_concurrent_positions=15, got {max_concurrent}"

        print(f"✓ risk.yaml position limits: {max_per_position} per position, {max_concurrent} concurrent")

        return True

    except Exception as e:
        print(f"✗ Config files test failed: {e}")
        return False


def test_kelly_status():
    """Test KellySizer status reporting."""
    try:
        from live_trading.kelly_sizer import KellySizer

        config = {
            'enabled': True,
            'min_trades_for_kelly': 20,
            'kelly_fraction': 0.25,
            'rolling_window_trades': 50,
            'max_contracts_per_trade': 5,
            'min_contracts': 1,
            'confidence_boost': {'enabled': True, 'boost_factor': 1.5, 'boost_threshold': 0.15},
            'negative_kelly_threshold': 3,
            'log_sizing_decisions': False,
        }

        sizer = KellySizer(config)

        # Initial status (no trades)
        status = sizer.get_status()
        assert status['phase'] == 'learning_0/20', f"Expected 'learning_0/20' phase, got '{status['phase']}'"
        assert status['trades_seen'] == 0, "trades_seen should be 0"

        print("✓ KellySizer status reporting works")
        return True

    except Exception as e:
        print(f"✗ KellySizer status test failed: {e}")
        return False


def run_smoke_tests():
    """Run all smoke tests and report results."""
    print("\n" + "="*60)
    print("Kelly Criterion Integration Smoke Tests")
    print("="*60 + "\n")

    tests = [
        ("Import KellySizer", test_kelly_sizer_imports),
        ("Import LiveExecutionEngine", test_execution_engine_imports),
        ("Import LiveTradingRunner", test_live_runner_imports),
        ("MetricsTracker Kelly methods", test_metrics_tracker_kelly_methods),
        ("KellySizer initialization", test_kelly_sizer_initialization),
        ("KellySizer disabled behavior", test_kelly_sizer_disabled_behavior),
        ("Config files exist and valid", test_config_files_exist),
        ("KellySizer status reporting", test_kelly_status),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 60)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("-" * 60)
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All smoke tests passed!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_tests()
    exit(exit_code)
