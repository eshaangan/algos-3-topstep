"""
Pytest configuration and fixtures for V3 tests.
"""

import pytest


@pytest.fixture
def sample_config_dict():
    """Sample config dictionary for testing."""
    return {
        "version": "1.0.0",
        "param1": 100,
        "param2": "test_value",
        "nested": {
            "sub_param1": [1, 2, 3],
            "sub_param2": {"key": "value"},
        },
    }


@pytest.fixture
def sample_execution_spec():
    """Sample execution spec config for testing."""
    return {
        "version": "1.0.0",
        "fill_model": {
            "fill_price": "next_bar_open",
            "touch_ordering": "ohlc_path",
        },
        "costs": {
            "slippage_ticks": {"1m": 1.0, "5m": 1.5},
            "commission_per_contract": 0.62,
        },
        "position_limits": {
            "max_contracts_per_position": 2,
            "max_concurrent_positions": 1,
        },
    }
