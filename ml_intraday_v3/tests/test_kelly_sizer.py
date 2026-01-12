"""
Unit tests for Kelly Criterion position sizing.

Tests cover:
- Kelly fraction calculation (basic and edge cases)
- Position sizing logic (all decision paths)
- Safety mechanisms (learning phase, negative Kelly, caps)
- Status reporting
"""

import pytest
from datetime import datetime
from typing import Dict, List

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.kelly_sizer import KellySizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Standard Kelly sizing configuration for testing."""
    return {
        'enabled': True,
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
        'log_sizing_decisions': False,  # Reduce test output noise
    }


@pytest.fixture
def disabled_config(default_config):
    """Configuration with Kelly sizing disabled."""
    config = default_config.copy()
    config['enabled'] = False
    return config


def generate_trade_history(
    num_trades: int,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> List[Dict]:
    """
    Generate synthetic trade history for testing.

    Args:
        num_trades: Total number of trades
        win_rate: Win rate (0.0 to 1.0)
        avg_win: Average winning trade P&L
        avg_loss: Average losing trade P&L (positive value, will be negated)

    Returns:
        List of trade dicts with 'pnl' key
    """
    num_wins = int(num_trades * win_rate)
    num_losses = num_trades - num_wins

    trades = []

    # Add winners
    for _ in range(num_wins):
        trades.append({'pnl': avg_win, 'entry_time': datetime.now()})

    # Add losers
    for _ in range(num_losses):
        trades.append({'pnl': -avg_loss, 'entry_time': datetime.now()})

    return trades


# ============================================================================
# Test calculate_kelly_fraction
# ============================================================================

class TestCalculateKellyFraction:
    """Tests for Kelly fraction calculation logic."""

    def test_basic_kelly_calculation(self, default_config):
        """Test basic Kelly calculation with 55% WR and 1.5 payoff ratio."""
        sizer = KellySizer(default_config)

        # Generate 30 trades: 55% WR, avg_win=$60, avg_loss=$40
        # Note: With 30 trades, int(30*0.55)=16 wins, 14 losses → actual WR=53.33%
        # Expected: payoff_ratio = 60/40 = 1.5
        # Kelly = (0.5333 × 1.5 - 0.4667) / 1.5 = 0.3333 / 1.5 = 0.222
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.55,
            avg_win=60.0,
            avg_loss=40.0,
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        # Expected Kelly with actual 53.33% WR: 0.222
        assert 0.20 <= kelly <= 0.25, f"Expected Kelly ~0.22, got {kelly:.3f}"

    def test_insufficient_data(self, default_config):
        """Test Kelly returns 0.0 when trade history is too short."""
        sizer = KellySizer(default_config)

        # Only 10 trades (below min_trades_for_kelly=20)
        trades = generate_trade_history(
            num_trades=10,
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=50.0,
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        assert kelly == 0.0, f"Expected 0.0 for insufficient data, got {kelly:.3f}"

    def test_all_winners(self, default_config):
        """Test Kelly caps at 1.0 when all trades are winners."""
        sizer = KellySizer(default_config)

        # 25 winning trades, 0 losers
        trades = generate_trade_history(
            num_trades=25,
            win_rate=1.0,
            avg_win=100.0,
            avg_loss=0.0,  # No losses
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        assert kelly == 1.0, f"Expected Kelly=1.0 for all winners, got {kelly:.3f}"

    def test_all_losers(self, default_config):
        """Test Kelly returns -1.0 when all trades are losers."""
        sizer = KellySizer(default_config)

        # 25 losing trades, 0 winners
        trades = generate_trade_history(
            num_trades=25,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=50.0,
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        assert kelly == -1.0, f"Expected Kelly=-1.0 for all losers, got {kelly:.3f}"

    def test_negative_expectancy(self, default_config):
        """Test Kelly returns negative value for losing system."""
        sizer = KellySizer(default_config)

        # 40% WR, 1.0 payoff → negative expectancy
        # Kelly = (0.4 × 1.0 - 0.6) / 1.0 = -0.2
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=50.0,
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        assert kelly < 0, f"Expected negative Kelly for losing system, got {kelly:.3f}"
        assert -0.25 <= kelly <= -0.15, f"Expected Kelly ~-0.2, got {kelly:.3f}"

    def test_rolling_window(self, default_config):
        """Test rolling window limits calculation to recent trades."""
        sizer = KellySizer(default_config)

        # Create 100 trades: first 50 are losers, last 50 are winners
        old_trades = generate_trade_history(
            num_trades=50,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=50.0,
        )

        new_trades = generate_trade_history(
            num_trades=50,
            win_rate=1.0,
            avg_win=100.0,
            avg_loss=0.0,
        )

        all_trades = old_trades + new_trades

        # With rolling_window_trades=50, should only use last 50 (all winners)
        kelly = sizer.calculate_kelly_fraction(all_trades)

        # Should be positive (recent trades are winners)
        assert kelly > 0, f"Expected positive Kelly for recent winners, got {kelly:.3f}"
        assert kelly == 1.0, f"Expected Kelly=1.0 for all winners in window, got {kelly:.3f}"

    def test_empty_trade_history(self, default_config):
        """Test Kelly returns 0.0 for empty trade history."""
        sizer = KellySizer(default_config)

        kelly = sizer.calculate_kelly_fraction([])

        assert kelly == 0.0, f"Expected 0.0 for empty history, got {kelly:.3f}"

    def test_realistic_backtest_stats(self, default_config):
        """Test Kelly with realistic backtest statistics (51% WR, 0.83 payoff)."""
        sizer = KellySizer(default_config)

        # 51% WR, payoff_ratio = 0.83 (from user's backtest)
        # Expected Kelly = (0.51 × 0.83 - 0.49) / 0.83 = -0.087 (negative)
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.51,
            avg_win=41.5,  # Avg win
            avg_loss=50.0,  # Avg loss → payoff = 41.5/50 = 0.83
        )

        kelly = sizer.calculate_kelly_fraction(trades)

        # Should be slightly negative
        assert kelly < 0, f"Expected negative Kelly for poor payoff, got {kelly:.3f}"
        assert -0.15 <= kelly <= -0.05, f"Expected Kelly ~-0.087, got {kelly:.3f}"


# ============================================================================
# Test get_position_size
# ============================================================================

class TestGetPositionSize:
    """Tests for position sizing logic with all safety mechanisms."""

    def test_kelly_disabled(self, disabled_config):
        """Test position size = 1 when Kelly is disabled."""
        sizer = KellySizer(disabled_config)

        # Even with great trade history, should return 1
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=50.0,
        )

        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        assert contracts == 1, f"Expected 1 contract when disabled, got {contracts}"
        assert reason == "disabled", f"Expected 'disabled' reason, got '{reason}'"

    def test_learning_phase(self, default_config):
        """Test position size = 1 during learning phase (< min_trades)."""
        sizer = KellySizer(default_config)

        # Only 15 trades (below 20 threshold)
        trades = generate_trade_history(
            num_trades=15,
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=50.0,
        )

        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        assert contracts == 1, f"Expected 1 contract in learning phase, got {contracts}"
        assert "learning_phase" in reason, f"Expected 'learning_phase' in reason, got '{reason}'"
        assert "15/20" in reason, f"Expected '15/20' in reason, got '{reason}'"

    def test_negative_kelly_fallback(self, default_config):
        """Test position size = 1 when Kelly is negative."""
        sizer = KellySizer(default_config)

        # Poor performance: 40% WR, 1.0 payoff → negative Kelly
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=50.0,
        )

        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        assert contracts == 1, f"Expected 1 contract for negative Kelly, got {contracts}"
        assert "negative_expectancy" in reason, f"Expected 'negative_expectancy' in reason, got '{reason}'"

    def test_consecutive_negative_kelly_threshold(self, default_config):
        """Test consecutive negative Kelly detection."""
        sizer = KellySizer(default_config)

        # Losing trade history
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=50.0,
        )

        # Call get_position_size 3 times (should trigger threshold)
        for i in range(3):
            contracts, reason = sizer.get_position_size(
                trade_history=trades,
                score_ev=0.20,
                max_contracts_limit=5,
                current_equity=50000.0,
                contract_margin=1320.0,
            )

            assert contracts == 1, f"Expected 1 contract on call {i+1}, got {contracts}"

            if i < 2:
                assert "negative_expectancy" in reason
            else:
                # On 3rd consecutive negative Kelly
                assert "consecutive_negative_kelly" in reason, \
                    f"Expected 'consecutive_negative_kelly' on call 3, got '{reason}'"

    def test_confidence_boost_applied(self, default_config):
        """Test confidence boost multiplies position size on high score_ev."""
        sizer = KellySizer(default_config)

        # Good performance: 55% WR, 1.5 payoff
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.55,
            avg_win=75.0,
            avg_loss=50.0,
        )

        # Without boost (score_ev below threshold)
        contracts_no_boost, reason_no_boost = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.10,  # Below 0.15 threshold
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        # With boost (score_ev >= threshold)
        contracts_with_boost, reason_with_boost = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,  # Above 0.15 threshold
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        # Boosted should be larger (1.5x factor)
        assert contracts_with_boost >= contracts_no_boost, \
            f"Expected boost to increase contracts: {contracts_no_boost} → {contracts_with_boost}"

    def test_margin_capping(self, default_config):
        """Test position size capped by available margin."""
        sizer = KellySizer(default_config)

        # Excellent performance → high Kelly
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.65,
            avg_win=100.0,
            avg_loss=40.0,
        )

        # Low equity / high margin → can only afford 2 contracts
        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=3000.0,  # Only $3k
            contract_margin=1320.0,  # $1320 per contract → max 2 contracts
        )

        assert contracts <= 2, f"Expected contracts <= 2 due to margin, got {contracts}"
        # Note: May be capped by other factors too, so just check it doesn't exceed affordable

    def test_position_limit_capping(self, default_config):
        """Test position size capped by max_contracts_limit from risk.yaml."""
        # Use custom config with high max_contracts_per_trade to test position limit
        config = default_config.copy()
        config['max_contracts_per_trade'] = 10  # Raise config limit
        sizer = KellySizer(config)

        # Excellent performance
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.70,
            avg_win=120.0,
            avg_loss=40.0,
        )

        # High equity but strict position limit
        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=2,  # Hard cap at 2 (this should be the binding constraint)
            current_equity=100000.0,  # High equity
            contract_margin=1320.0,
        )

        assert contracts <= 2, f"Expected contracts <= 2 due to position limit, got {contracts}"
        assert "position_limit" in reason, f"Expected 'position_limit' in reason, got '{reason}'"

    def test_config_max_contracts_capping(self, default_config):
        """Test position size capped by max_contracts_per_trade config."""
        sizer = KellySizer(default_config)

        # Excellent performance
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.70,
            avg_win=120.0,
            avg_loss=40.0,
        )

        # High limits but config caps at 5
        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.25,  # High conviction
            max_contracts_limit=10,  # High limit
            current_equity=100000.0,  # High equity
            contract_margin=1320.0,
        )

        assert contracts <= 5, f"Expected contracts <= 5 (config max), got {contracts}"
        # Note: config max_contracts_per_trade = 5

    def test_floor_at_min_contracts(self, default_config):
        """Test position size floors at min_contracts (never returns 0)."""
        sizer = KellySizer(default_config)

        # Very weak positive Kelly (just above 0)
        # 50.1% WR, 1.0 payoff → tiny positive Kelly
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.501,
            avg_win=50.0,
            avg_loss=50.0,
        )

        # Low equity → Kelly * affordable ≈ 0
        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.05,  # Low score
            max_contracts_limit=5,
            current_equity=2000.0,  # Low equity
            contract_margin=1320.0,
        )

        # Should floor at 1 (min_contracts)
        assert contracts >= 1, f"Expected contracts >= 1 (floor), got {contracts}"

    def test_normal_kelly_sizing(self, default_config):
        """Test normal Kelly sizing with positive expectancy."""
        sizer = KellySizer(default_config)

        # Solid performance: 55% WR, 1.5 payoff
        # Kelly ≈ 0.55, fractional (1/4) = 0.1375
        trades = generate_trade_history(
            num_trades=30,
            win_rate=0.55,
            avg_win=75.0,
            avg_loss=50.0,
        )

        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.12,  # Below boost threshold
            max_contracts_limit=5,
            current_equity=52000.0,  # Typical Topstep equity
            contract_margin=1320.0,
        )

        # Expected: max_affordable = 52000 / 1320 = 39
        # Kelly = 0.55, fractional = 0.1375
        # Contracts = 39 * 0.1375 ≈ 5.36 → 5
        # But may be capped by max_contracts_per_trade=5 or other limits
        assert 1 <= contracts <= 5, f"Expected contracts in [1, 5], got {contracts}"
        assert "kelly" in reason.lower(), f"Expected 'kelly' in reason, got '{reason}'"

    def test_kelly_error_fallback(self, default_config):
        """Test fallback to 1 contract on Kelly calculation error."""
        sizer = KellySizer(default_config)

        # Malformed trade history (missing 'pnl' key)
        trades = [{'entry_time': datetime.now()} for _ in range(30)]

        contracts, reason = sizer.get_position_size(
            trade_history=trades,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        # Should gracefully fall back to 1 contract
        assert contracts == 1, f"Expected 1 contract on error, got {contracts}"
        # Reason may vary (could be kelly_error_fallback or other)


# ============================================================================
# Test get_status
# ============================================================================

class TestGetStatus:
    """Tests for status reporting."""

    def test_status_disabled(self, disabled_config):
        """Test status when Kelly is disabled."""
        sizer = KellySizer(disabled_config)

        status = sizer.get_status()

        assert status['phase'] == 'disabled', f"Expected phase='disabled', got '{status['phase']}'"
        assert status['trades_seen'] == 0, f"Expected trades_seen=0, got {status['trades_seen']}"

    def test_status_learning_phase(self, default_config):
        """Test status during learning phase."""
        sizer = KellySizer(default_config)

        # Simulate 10 trades seen
        trades = generate_trade_history(10, 0.60, 100.0, 50.0)
        sizer.get_position_size(
            trade_history=trades,
            score_ev=0.10,
            max_contracts_limit=5,
            current_equity=50000.0,
            contract_margin=1320.0,
        )

        status = sizer.get_status()

        assert "learning" in status['phase'], f"Expected 'learning' in phase, got '{status['phase']}'"
        assert status['trades_seen'] == 10, f"Expected trades_seen=10, got {status['trades_seen']}"

    def test_status_active(self, default_config):
        """Test status when Kelly is active."""
        sizer = KellySizer(default_config)

        # Simulate 30 trades seen (past learning phase)
        trades = generate_trade_history(30, 0.55, 75.0, 50.0)
        sizer.get_position_size(
            trade_history=trades,
            score_ev=0.15,
            max_contracts_limit=5,
            current_equity=52000.0,
            contract_margin=1320.0,
        )

        status = sizer.get_status()

        assert status['phase'] == 'active', f"Expected phase='active', got '{status['phase']}'"
        assert status['trades_seen'] == 30, f"Expected trades_seen=30, got {status['trades_seen']}"
        assert status['current_kelly_fraction'] != 0, "Expected non-zero Kelly fraction"
        assert status['last_position_size'] >= 1, "Expected position size >= 1"


# ============================================================================
# Integration Tests
# ============================================================================

class TestKellySizerIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_positive_expectancy(self, default_config):
        """Test complete workflow with positive expectancy system."""
        sizer = KellySizer(default_config)

        # Simulate 50 trades with good performance
        all_trades = []

        for i in range(50):
            # Generate batch of trades (i+1 trades total)
            num_trades = i + 1
            batch = generate_trade_history(num_trades, 0.55, 75.0, 50.0)

            contracts, reason = sizer.get_position_size(
                trade_history=batch,
                score_ev=0.12,
                max_contracts_limit=5,
                current_equity=52000.0,
                contract_margin=1320.0,
            )

            if num_trades < 20:
                # Learning phase: Should return 1 contract
                assert contracts == 1, f"Trade {num_trades}: Expected 1 contract in learning, got {contracts}"
                assert "learning_phase" in reason, f"Trade {num_trades}: Expected 'learning_phase' in reason"
            else:
                # Active phase: Should use Kelly (>= 1 with good performance)
                assert contracts >= 1, f"Trade {num_trades}: Expected contracts >= 1, got {contracts}"
                assert "kelly" in reason.lower(), f"Trade {num_trades}: Expected 'kelly' in reason"

        # Check final status
        status = sizer.get_status()
        assert status['phase'] == 'active'
        assert status['trades_seen'] == 50
        assert status['current_kelly_fraction'] > 0, "Expected positive Kelly"

    def test_full_workflow_negative_expectancy(self, default_config):
        """Test complete workflow with negative expectancy system."""
        sizer = KellySizer(default_config)

        # Simulate 30 trades with poor performance
        for i in range(30):
            batch = generate_trade_history(i+1, 0.40, 50.0, 50.0)
            contracts, reason = sizer.get_position_size(
                trade_history=batch,
                score_ev=0.10,
                max_contracts_limit=5,
                current_equity=52000.0,
                contract_margin=1320.0,
            )

            # Should always be 1 (learning phase or negative Kelly)
            assert contracts == 1, f"Trade {i+1}: Expected 1 contract for negative system, got {contracts}"

        # Check final status
        status = sizer.get_status()
        assert status['phase'] == 'active'
        assert status['current_kelly_fraction'] <= 0, "Expected negative Kelly"

    def test_transition_from_learning_to_active(self, default_config):
        """Test smooth transition from learning phase to active Kelly."""
        sizer = KellySizer(default_config)

        # Trade 19: Still learning
        trades_19 = generate_trade_history(19, 0.60, 100.0, 50.0)
        contracts_19, reason_19 = sizer.get_position_size(
            trade_history=trades_19,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=52000.0,
            contract_margin=1320.0,
        )
        assert contracts_19 == 1, "Trade 19: Expected learning phase"
        assert "learning_phase" in reason_19

        # Trade 20: Activates Kelly
        trades_20 = generate_trade_history(20, 0.60, 100.0, 50.0)
        contracts_20, reason_20 = sizer.get_position_size(
            trade_history=trades_20,
            score_ev=0.20,
            max_contracts_limit=5,
            current_equity=52000.0,
            contract_margin=1320.0,
        )
        # Should now use Kelly (likely > 1 with good stats and high score_ev)
        assert contracts_20 >= 1, "Trade 20: Expected Kelly activation"
        assert "kelly" in reason_20.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
