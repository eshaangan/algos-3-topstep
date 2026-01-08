"""
Tests for data filtering functionality.

Validates that filtering removes corrupted data (spread contracts, negative prices, etc.)
while preserving clean data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_intraday_v3.data.ingest import filter_data


class TestSpreadContractFiltering:
    """Tests for filtering spread contracts by symbol pattern."""

    def test_filter_spread_contracts(self):
        """Test that spread contracts (symbols with '-') are removed."""
        # Create test data with both outright and spread contracts
        df = pd.DataFrame({
            "symbol": ["ESU0", "ESM0-ESU0", "ESZ0", "ESU0-ESZ0", "ESH1"],
            "open": [4500.0, -4.30, 4510.0, -5.20, 4520.0],
            "high": [4505.0, -4.20, 4515.0, -5.10, 4525.0],
            "low": [4495.0, -4.40, 4505.0, -5.30, 4515.0],
            "close": [4500.0, -4.30, 4510.0, -5.20, 4520.0],
        }, index=pd.date_range("2020-01-01", periods=5, freq="1min", tz="UTC"))

        # Configure filtering to exclude spread contracts
        filter_cfg = {
            "enabled": True,
            "symbol_filter": {
                "enabled": True,
                "mode": "exclude_patterns",
                "exclude_patterns": [".*-.*"],  # Exclude symbols with "-"
            },
            "price_validation": {"enabled": False},
        }

        df_filtered, stats = filter_data(df, filter_cfg, symbol_column="symbol")

        # Verify spread contracts removed
        assert len(df_filtered) == 3, "Should keep 3 outright contracts"
        assert stats["symbol_filter_removed"] == 2, "Should remove 2 spread contracts"
        assert "ESM0-ESU0" not in df_filtered["symbol"].values
        assert "ESU0-ESZ0" not in df_filtered["symbol"].values
        assert "ESU0" in df_filtered["symbol"].values
        assert "ESZ0" in df_filtered["symbol"].values
        assert "ESH1" in df_filtered["symbol"].values

    def test_filter_include_patterns(self):
        """Test include_patterns mode (keep only matching symbols)."""
        df = pd.DataFrame({
            "symbol": ["ESU0", "ESM0-ESU0", "NQU0", "ESZ0"],
            "close": [4500.0, -4.30, 15000.0, 4510.0],
        }, index=pd.date_range("2020-01-01", periods=4, freq="1min", tz="UTC"))

        # Configure to keep only ES contracts
        filter_cfg = {
            "enabled": True,
            "symbol_filter": {
                "enabled": True,
                "mode": "include_patterns",
                "include_patterns": ["ES.*"],  # Keep only ES symbols
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg, symbol_column="symbol")

        # Verify only ES symbols kept
        assert len(df_filtered) == 3, "Should keep 3 ES symbols"
        assert "NQU0" not in df_filtered["symbol"].values


class TestPriceValidation:
    """Tests for price range validation."""

    def test_filter_negative_prices(self):
        """Test that negative prices are removed."""
        df = pd.DataFrame({
            "open": [4500.0, -4.30, 4510.0, -5.20, 4520.0],
            "high": [4505.0, -4.20, 4515.0, -5.10, 4525.0],
            "low": [4495.0, -4.40, 4505.0, -5.30, 4515.0],
            "close": [4500.0, -4.30, 4510.0, -5.20, 4520.0],
        }, index=pd.date_range("2020-01-01", periods=5, freq="1min", tz="UTC"))

        # Configure price validation
        filter_cfg = {
            "enabled": True,
            "symbol_filter": {"enabled": False},
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
                "max_price": 10000.0,
                "violation_action": "drop_bar",
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg)

        # Verify negative prices removed
        assert len(df_filtered) == 3, "Should keep 3 bars with valid prices"
        assert stats["price_validation_removed"] == 2, "Should remove 2 bars with negative prices"
        assert (df_filtered["open"] >= 100).all()
        assert (df_filtered["close"] >= 100).all()

    def test_filter_low_prices(self):
        """Test that prices below minimum threshold are removed."""
        df = pd.DataFrame({
            "open": [4500.0, 50.50, 4510.0, 75.25, 4520.0],
            "close": [4500.0, 50.50, 4510.0, 75.25, 4520.0],
        }, index=pd.date_range("2020-01-01", periods=5, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
                "violation_action": "drop_bar",
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg)

        # Verify low prices removed
        assert len(df_filtered) == 3, "Should keep 3 bars with prices >= 100"
        assert stats["price_validation_removed"] == 2, "Should remove 2 bars with low prices"
        assert (df_filtered["open"] >= 100).all()

    def test_filter_high_prices(self):
        """Test that prices above maximum threshold are removed."""
        df = pd.DataFrame({
            "open": [4500.0, 15000.0, 4510.0],
            "close": [4500.0, 15000.0, 4510.0],
        }, index=pd.date_range("2020-01-01", periods=3, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {
                "enabled": True,
                "max_price": 10000.0,
                "violation_action": "drop_bar",
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg)

        # Verify high prices removed
        assert len(df_filtered) == 2, "Should keep 2 bars with prices <= 10000"
        assert stats["price_validation_removed"] == 1, "Should remove 1 bar with high price"
        assert (df_filtered["open"] <= 10000).all()

    def test_price_validation_warn_action(self):
        """Test that warn action logs violations but keeps data."""
        df = pd.DataFrame({
            "close": [4500.0, -4.30, 50.50],
        }, index=pd.date_range("2020-01-01", periods=3, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
                "violation_action": "warn",  # Keep data but warn
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg)

        # Verify data kept despite violations
        assert len(df_filtered) == 3, "Should keep all bars in warn mode"
        assert stats["price_validation_removed"] == 0, "Should not remove any bars"

    def test_price_validation_raise_action(self):
        """Test that raise action throws exception on violations."""
        df = pd.DataFrame({
            "close": [4500.0, -4.30],
        }, index=pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
                "violation_action": "raise",
            },
        }

        # Should raise ValueError on violations
        with pytest.raises(ValueError, match="Price validation failed"):
            filter_data(df, filter_cfg)


class TestCombinedFiltering:
    """Tests for combined symbol + price filtering."""

    def test_combined_spread_and_price_filtering(self):
        """Test that both spread contracts and invalid prices are removed."""
        df = pd.DataFrame({
            "symbol": ["ESU0", "ESM0-ESU0", "ESZ0", "ESH1"],
            "open": [4500.0, -4.30, 50.50, 4520.0],
            "close": [4500.0, -4.30, 50.50, 4520.0],
        }, index=pd.date_range("2020-01-01", periods=4, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "symbol_filter": {
                "enabled": True,
                "mode": "exclude_patterns",
                "exclude_patterns": [".*-.*"],
            },
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
                "violation_action": "drop_bar",
            },
        }

        df_filtered, stats = filter_data(df, filter_cfg, symbol_column="symbol")

        # Verify both filters applied
        assert len(df_filtered) == 2, "Should keep 2 clean bars"
        assert stats["symbol_filter_removed"] == 1, "Should remove 1 spread contract"
        assert stats["price_validation_removed"] == 1, "Should remove 1 bar with low price"
        assert "ESU0" in df_filtered["symbol"].values
        assert "ESH1" in df_filtered["symbol"].values


class TestFilteringDisabled:
    """Tests for disabled filtering."""

    def test_filter_disabled(self):
        """Test that filtering can be disabled."""
        df = pd.DataFrame({
            "symbol": ["ESU0", "ESM0-ESU0"],
            "close": [4500.0, -4.30],
        }, index=pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": False,  # Disabled
        }

        df_filtered, stats = filter_data(df, filter_cfg, symbol_column="symbol")

        # Verify no filtering applied
        assert len(df_filtered) == 2, "Should keep all data when disabled"
        assert stats["rows_removed"] == 0, "Should not remove any rows"

    def test_filter_cfg_none(self):
        """Test that None filter_cfg returns data unchanged."""
        df = pd.DataFrame({
            "close": [4500.0, -4.30],
        }, index=pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC"))

        df_filtered, stats = filter_data(df, None)

        # Verify no filtering applied
        assert len(df_filtered) == 2, "Should keep all data when filter_cfg is None"
        assert stats["rows_removed"] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_symbol_column(self):
        """Test symbol filtering when symbol column doesn't exist."""
        df = pd.DataFrame({
            "close": [4500.0, 4510.0],
        }, index=pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "symbol_filter": {
                "enabled": True,
                "mode": "exclude_patterns",
                "exclude_patterns": [".*-.*"],
            },
        }

        # Should not fail if symbol column doesn't exist
        df_filtered, stats = filter_data(df, filter_cfg, symbol_column="symbol")
        assert len(df_filtered) == 2, "Should keep all data if symbol column missing"
        assert stats["symbol_filter_removed"] == 0

    def test_no_price_columns(self):
        """Test price validation when price columns don't exist."""
        df = pd.DataFrame({
            "volume": [1000, 2000],
        }, index=pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {
                "enabled": True,
                "min_price": 100.0,
            },
        }

        # Should not fail if price columns don't exist
        df_filtered, stats = filter_data(df, filter_cfg)
        assert len(df_filtered) == 2, "Should keep all data if price columns missing"
        assert stats["price_validation_removed"] == 0

    def test_empty_dataframe(self):
        """Test filtering on empty DataFrame."""
        df = pd.DataFrame({
            "close": [],
        }, index=pd.DatetimeIndex([], tz="UTC"))

        filter_cfg = {
            "enabled": True,
            "price_validation": {"enabled": True, "min_price": 100.0},
        }

        df_filtered, stats = filter_data(df, filter_cfg)
        assert len(df_filtered) == 0, "Should return empty DataFrame"
        assert stats["rows_removed"] == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
