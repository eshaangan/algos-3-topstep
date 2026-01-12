"""
Kelly Criterion Dynamic Position Sizing.

Implements fractional Kelly sizing with multiple safety mechanisms:
- Learning phase (fixed 1 contract for first N trades)
- Negative expectancy fallback
- Confidence-based scaling
- Multiple position/margin caps
- Graceful degradation on errors
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KellySizer:
    """
    Dynamic position sizing using Kelly Criterion with robust safety mechanisms.

    The Kelly Criterion calculates optimal bet size based on win rate and payoff ratio:
        f* = (p × b - q) / b
    where:
        p = win probability
        q = loss probability (1 - p)
        b = payoff ratio (avg_win / avg_loss)

    This implementation uses fractional Kelly (typically 1/4) for safety and includes:
    - Learning phase before Kelly activates
    - Confidence boost on high-conviction signals
    - Multiple caps (margin, position limits, config limits)
    - Fallback to 1 contract on negative Kelly or errors
    """

    def __init__(self, config: Dict):
        """
        Initialize Kelly sizer with configuration.

        Args:
            config: Kelly sizing configuration dict with keys:
                - enabled: bool (master kill switch)
                - min_trades_for_kelly: int (learning phase threshold)
                - kelly_fraction: float (0.25 = 1/4 Kelly)
                - rolling_window_trades: int or None (window size for calculation)
                - max_contracts_per_trade: int (hard cap)
                - min_contracts: int (floor, typically 1)
                - confidence_boost: dict with enabled, boost_factor, boost_threshold
                - negative_kelly_threshold: int (consecutive negative Kelly limit)
                - log_sizing_decisions: bool
        """
        self.config = config

        # State tracking
        self.negative_kelly_count = 0
        self.last_kelly_fraction = 0.0
        self.last_position_size = 1
        self.trades_seen = 0

        logger.info(f"KellySizer initialized: enabled={config.get('enabled', False)}, "
                   f"kelly_fraction={config.get('kelly_fraction', 0.25)}, "
                   f"min_trades={config.get('min_trades_for_kelly', 20)}")

    def calculate_kelly_fraction(self, trade_history: List[Dict]) -> float:
        """
        Calculate Kelly fraction from trade history.

        Args:
            trade_history: List of trade dicts with 'pnl' key

        Returns:
            Kelly fraction (0.0 to 1.0 for positive edge, negative for losing system)
            Returns 0.0 for insufficient data

        Formula: f* = (p × b - q) / b
        where b = avg_win / abs(avg_loss)
        """
        # Edge case 1: Insufficient history
        min_trades = self.config.get('min_trades_for_kelly', 20)
        if len(trade_history) < min_trades:
            return 0.0

        # Apply rolling window if configured
        window_size = self.config.get('rolling_window_trades')
        if window_size and len(trade_history) > window_size:
            trades = trade_history[-window_size:]
        else:
            trades = trade_history

        # Edge case 2: No trades in window
        if len(trades) == 0:
            return 0.0

        # Separate winners and losers
        winners = [t for t in trades if t.get('pnl', 0) > 0]
        losers = [t for t in trades if t.get('pnl', 0) <= 0]

        # Calculate win rate
        win_rate = len(winners) / len(trades)

        # Edge case 3: All losers (win_rate = 0)
        if win_rate == 0.0:
            logger.warning("All trades are losers - Kelly = -1.0 (negative expectancy)")
            return -1.0

        # Edge case 4: All winners (win_rate = 1.0)
        if win_rate == 1.0:
            logger.info("All trades are winners - capping Kelly at 1.0")
            return 1.0

        # Calculate average win and loss
        avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0.0
        avg_loss = abs(sum(t['pnl'] for t in losers) / len(losers)) if losers else 0.0

        # Edge case 5: No losses (shouldn't happen if we have losers list, but be safe)
        if avg_loss == 0.0:
            logger.warning("No losing trades found - capping Kelly at 1.0")
            return 1.0

        # Calculate payoff ratio
        payoff_ratio = avg_win / avg_loss

        # Kelly formula: (p × b - q) / b
        kelly_fraction = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio

        # Cap between -1 and 1 for safety
        kelly_fraction = max(-1.0, min(1.0, kelly_fraction))

        # Log calculation details
        if self.config.get('log_sizing_decisions', True):
            logger.debug(
                f"Kelly calculation: win_rate={win_rate:.3f}, "
                f"avg_win=${avg_win:.2f}, avg_loss=${avg_loss:.2f}, "
                f"payoff_ratio={payoff_ratio:.3f}, kelly={kelly_fraction:.3f}"
            )

        return kelly_fraction

    def get_position_size(
        self,
        trade_history: List[Dict],
        score_ev: float,
        max_contracts_limit: int,
        current_equity: float,
        contract_margin: float,
    ) -> Tuple[int, str]:
        """
        Determine position size for next trade using Kelly Criterion.

        Decision flow:
        1. Check if Kelly is disabled → return (1, "disabled")
        2. Check if in learning phase → return (1, "learning_phase")
        3. Calculate Kelly from trade history
        4. Check if negative expectancy → return (1, "negative_expectancy")
        5. Apply fractional Kelly
        6. Apply confidence boost (if enabled and score_ev > threshold)
        7. Cap by margin available
        8. Cap by position limits
        9. Floor at min_contracts

        Args:
            trade_history: List of completed trades for Kelly calculation
            score_ev: Model prediction score (for confidence boost)
            max_contracts_limit: Maximum contracts allowed by risk config
            current_equity: Current account equity
            contract_margin: Margin required per contract

        Returns:
            Tuple of (contracts, sizing_reason)
            - contracts: Number of contracts (1 to max_contracts_limit)
            - sizing_reason: String explaining the sizing decision
        """
        self.trades_seen = len(trade_history)

        # Safety gate 1: Kill switch
        if not self.config.get('enabled', False):
            return 1, "disabled"

        # Safety gate 2: Learning phase
        min_trades = self.config.get('min_trades_for_kelly', 20)
        if len(trade_history) < min_trades:
            return 1, f"learning_phase_{len(trade_history)}/{min_trades}"

        # Calculate raw Kelly
        try:
            raw_kelly = self.calculate_kelly_fraction(trade_history)
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}, falling back to 1 contract")
            return 1, "kelly_error_fallback"

        self.last_kelly_fraction = raw_kelly

        # Safety gate 3: Negative expectancy
        if raw_kelly <= 0:
            self.negative_kelly_count += 1

            # Check consecutive negative Kelly threshold
            neg_threshold = self.config.get('negative_kelly_threshold', 3)
            if self.negative_kelly_count >= neg_threshold:
                logger.warning(
                    f"Consecutive negative Kelly count: {self.negative_kelly_count} "
                    f">= threshold {neg_threshold} - using 1 contract"
                )
                return 1, f"consecutive_negative_kelly_{self.negative_kelly_count}"

            return 1, f"negative_expectancy_kelly_{raw_kelly:.3f}"
        else:
            # Reset counter on positive Kelly
            self.negative_kelly_count = 0

        # Apply fractional Kelly
        kelly_fraction = self.config.get('kelly_fraction', 0.25)
        fractional_kelly = raw_kelly * kelly_fraction

        # Convert Kelly fraction to position size
        # Kelly tells us what % of equity to risk
        # Calculate how many contracts that translates to
        max_affordable = int(current_equity / contract_margin) if contract_margin > 0 else 1
        kelly_contracts = int(fractional_kelly * max_affordable)

        # Apply confidence boost (if enabled)
        confidence_cfg = self.config.get('confidence_boost', {})
        if confidence_cfg.get('enabled', False):
            boost_threshold = confidence_cfg.get('boost_threshold', 0.15)
            boost_factor = confidence_cfg.get('boost_factor', 1.5)

            if abs(score_ev) >= boost_threshold:
                boosted_contracts = int(kelly_contracts * boost_factor)

                if self.config.get('log_sizing_decisions', True):
                    logger.info(
                        f"Confidence boost applied: score_ev={score_ev:.3f} >= {boost_threshold:.3f}, "
                        f"boost_factor={boost_factor:.2f}, "
                        f"contracts: {kelly_contracts} → {boosted_contracts}"
                    )

                kelly_contracts = boosted_contracts

        # Apply all caps
        min_contracts = self.config.get('min_contracts', 1)
        max_contracts_config = self.config.get('max_contracts_per_trade', 5)

        # Floor at minimum
        kelly_contracts = max(min_contracts, kelly_contracts)

        # Store original Kelly contracts before capping
        original_kelly_contracts = kelly_contracts

        # Apply all caps - take the minimum (most restrictive)
        kelly_contracts = min(
            kelly_contracts,
            max_affordable,
            max_contracts_limit,
            max_contracts_config,
        )

        # Determine which cap was binding (for sizing_reason)
        if original_kelly_contracts <= kelly_contracts:
            # No cap was binding
            sizing_reason = f"kelly_{fractional_kelly:.3f}_score_{score_ev:.3f}"
        elif kelly_contracts == max_affordable:
            sizing_reason = f"kelly_{fractional_kelly:.3f}_capped_by_margin"
        elif kelly_contracts == max_contracts_limit:
            sizing_reason = f"kelly_{fractional_kelly:.3f}_capped_by_position_limit"
        elif kelly_contracts == max_contracts_config:
            sizing_reason = f"kelly_{fractional_kelly:.3f}_capped_by_config"
        else:
            # Shouldn't happen, but fallback
            sizing_reason = f"kelly_{fractional_kelly:.3f}_capped"

        # Final floor at 1
        kelly_contracts = max(1, kelly_contracts)

        self.last_position_size = kelly_contracts

        # Log sizing decision
        if self.config.get('log_sizing_decisions', True):
            logger.info(
                f"Kelly sizing decision: contracts={kelly_contracts}, "
                f"raw_kelly={raw_kelly:.3f}, fractional={fractional_kelly:.3f}, "
                f"score_ev={score_ev:.3f}, reason={sizing_reason}"
            )

        return kelly_contracts, sizing_reason

    def get_status(self) -> Dict:
        """
        Get current Kelly sizer status for monitoring.

        Returns:
            Dict with status information:
                - phase: "disabled", "learning", or "active"
                - trades_seen: Number of trades processed
                - current_kelly_fraction: Last calculated Kelly fraction
                - last_position_size: Last determined position size
                - negative_kelly_count: Consecutive negative Kelly count
        """
        min_trades = self.config.get('min_trades_for_kelly', 20)

        if not self.config.get('enabled', False):
            phase = "disabled"
        elif self.trades_seen < min_trades:
            phase = f"learning_{self.trades_seen}/{min_trades}"
        else:
            phase = "active"

        return {
            'phase': phase,
            'trades_seen': self.trades_seen,
            'current_kelly_fraction': self.last_kelly_fraction,
            'last_position_size': self.last_position_size,
            'negative_kelly_count': self.negative_kelly_count,
        }
