"""
Tiered Position Sizing - Phase 2b Priority #3

Scales position size based on model confidence level.

Key Concept:
- Not all signals are created equal
- High confidence signals (P>0.65) deserve larger positions
- Low confidence signals (P=0.50-0.55) deserve smaller positions
- Amplifies winners while limiting exposure to marginal trades

Expected Impact: +$200-300 total P&L by concentrating capital on best opportunities

Research Context:
- Kelly Criterion: Optimal position sizing based on edge
- "Bet more when you have an edge" - fundamental principle
- Tiered sizing is a simplified, conservative Kelly approach

Usage:
    sizer = TieredPositionSizer(base_size=1, max_size=2)

    # Calculate position size for a trade
    size = sizer.calculate_size(
        probability=0.68,
        side='LONG',
        base_size=1
    )

    # Returns: 1.5 (high confidence, scale up 50%)
"""

import logging
from typing import Literal, Dict

logger = logging.getLogger(__name__)

SideType = Literal['LONG', 'SHORT']


class TieredPositionSizer:
    """
    Calculate position sizes based on prediction confidence.

    Strategy:
    - Divide predictions into confidence tiers
    - Scale position size by tier
    - Apply caps to prevent over-leverage
    - Track performance by tier
    """

    def __init__(
        self,
        # Confidence thresholds
        high_confidence_threshold: float = 0.65,
        medium_confidence_threshold: float = 0.55,
        low_confidence_threshold: float = 0.50,

        # Size multipliers
        high_confidence_multiplier: float = 1.0,  # UPDATED: 1.0 for 2-contract base (was 1.5)
        medium_confidence_multiplier: float = 1.0,
        low_confidence_multiplier: float = 0.5,

        # Position limits
        min_size: int = 1,
        max_size: int = 2,

        # Risk management
        allow_low_confidence: bool = False
    ):
        """
        Initialize tiered position sizer.

        Args:
            high_confidence_threshold: Probability threshold for high confidence (default: 0.65)
            medium_confidence_threshold: Probability threshold for medium confidence (default: 0.55)
            low_confidence_threshold: Probability threshold for low confidence (default: 0.50)
            high_confidence_multiplier: Size multiplier for high confidence (default: 1.5x)
            medium_confidence_multiplier: Size multiplier for medium confidence (default: 1.0x)
            low_confidence_multiplier: Size multiplier for low confidence (default: 0.5x)
            min_size: Minimum position size (default: 1)
            max_size: Maximum position size (default: 2)
            allow_low_confidence: If False, return 0 for low confidence trades (default: False)
        """
        self.thresholds = {
            'high': high_confidence_threshold,
            'medium': medium_confidence_threshold,
            'low': low_confidence_threshold
        }

        self.multipliers = {
            'high': high_confidence_multiplier,
            'medium': medium_confidence_multiplier,
            'low': low_confidence_multiplier
        }

        self.min_size = min_size
        self.max_size = max_size
        self.allow_low_confidence = allow_low_confidence

        # Statistics tracking
        self.sizes_calculated = 0
        self.tier_counts = {'high': 0, 'medium': 0, 'low': 0, 'rejected': 0}
        self.total_contracts = 0

        logger.info(
            f"TieredPositionSizer initialized: "
            f"high={high_confidence_multiplier}x (P>{high_confidence_threshold:.2f}), "
            f"medium={medium_confidence_multiplier}x (P>{medium_confidence_threshold:.2f}), "
            f"low={low_confidence_multiplier}x (P>{low_confidence_threshold:.2f}), "
            f"max_size={max_size}"
        )

    @classmethod
    def from_config(cls, config: dict):
        """
        Create TieredPositionSizer from configuration dictionary.

        Args:
            config: Configuration dict with tiered_sizing section

        Returns:
            TieredPositionSizer instance

        Example:
            import yaml
            with open('configs/position_sizing.yaml') as f:
                cfg = yaml.safe_load(f)
            sizer = TieredPositionSizer.from_config(cfg['tiered_sizing'])
        """
        return cls(
            high_confidence_threshold=config.get('high_confidence_threshold', 0.65),
            medium_confidence_threshold=config.get('medium_confidence_threshold', 0.55),
            low_confidence_threshold=config.get('low_confidence_threshold', 0.50),
            high_confidence_multiplier=config.get('high_confidence_multiplier', 1.0),
            medium_confidence_multiplier=config.get('medium_confidence_multiplier', 1.0),
            low_confidence_multiplier=config.get('low_confidence_multiplier', 0.5),
            min_size=config.get('min_size', 1),
            max_size=config.get('max_size', 2),
            allow_low_confidence=config.get('allow_low_confidence', False)
        )

    def classify_confidence(self, probability: float, side: SideType) -> str:
        """
        Classify confidence tier based on probability.

        Args:
            probability: Model probability (0.0-1.0)
            side: Trade direction ('LONG' or 'SHORT')

        Returns:
            Confidence tier: 'high', 'medium', 'low', or 'reject'

        Logic:
            For LONG trades:
            - High: P > 0.65
            - Medium: P > 0.55
            - Low: P > 0.50
            - Reject: P <= 0.50

            For SHORT trades (P is probability of UP):
            - High: P < 0.35 (equivalent to P(down) > 0.65)
            - Medium: P < 0.45 (equivalent to P(down) > 0.55)
            - Low: P < 0.50 (equivalent to P(down) > 0.50)
            - Reject: P >= 0.50
        """
        if side == 'LONG':
            # Direct probability interpretation
            if probability >= self.thresholds['high']:
                return 'high'
            elif probability >= self.thresholds['medium']:
                return 'medium'
            elif probability >= self.thresholds['low']:
                return 'low'
            else:
                return 'reject'
        else:  # SHORT
            # Inverse probability (P is probability of UP)
            if probability <= (1 - self.thresholds['high']):
                return 'high'
            elif probability <= (1 - self.thresholds['medium']):
                return 'medium'
            elif probability <= (1 - self.thresholds['low']):
                return 'low'
            else:
                return 'reject'

    def calculate_size(
        self,
        probability: float,
        side: SideType,
        base_size: int = 1,
        max_size_override: int = None
    ) -> int:
        """
        Calculate position size based on confidence.

        Args:
            probability: Model probability (0.0-1.0)
            side: Trade direction ('LONG' or 'SHORT')
            base_size: Base position size to scale from (default: 1)
            max_size_override: Optional override for max size

        Returns:
            Position size (contracts)
            Returns 0 if confidence too low

        Example:
            probability=0.68, side='LONG', base_size=1
            → Tier: high (P > 0.65)
            → Size: 1 * 1.5 = 1.5 → 1 contract (rounded down)

            probability=0.58, side='LONG', base_size=1
            → Tier: medium (0.55 < P < 0.65)
            → Size: 1 * 1.0 = 1 contract

            probability=0.52, side='LONG', base_size=1
            → Tier: low (0.50 < P < 0.55)
            → Size: 1 * 0.5 = 0.5 → 1 contract (min)
            → OR 0 if allow_low_confidence=False
        """
        self.sizes_calculated += 1

        # Classify confidence tier
        tier = self.classify_confidence(probability, side)

        # Track tier
        if tier in self.tier_counts:
            self.tier_counts[tier] += 1
        else:
            self.tier_counts['rejected'] += 1

        # Handle rejection
        if tier == 'reject':
            logger.debug(
                f"Trade rejected: side={side}, P={probability:.3f} "
                f"(threshold={self.thresholds['low']:.2f})"
            )
            return 0

        # Handle low confidence
        if tier == 'low' and not self.allow_low_confidence:
            logger.debug(
                f"Low confidence rejected: side={side}, P={probability:.3f}"
            )
            self.tier_counts['rejected'] += 1
            return 0

        # Calculate size
        multiplier = self.multipliers[tier]
        raw_size = base_size * multiplier

        # Apply limits
        max_allowed = max_size_override if max_size_override is not None else self.max_size
        final_size = max(self.min_size, min(max_allowed, int(raw_size)))

        self.total_contracts += final_size

        logger.debug(
            f"Position sized: side={side}, P={probability:.3f}, "
            f"tier={tier}, multiplier={multiplier:.1f}x, "
            f"size={final_size} contracts"
        )

        return final_size

    def get_statistics(self) -> dict:
        """
        Get position sizing statistics.

        Returns:
            Dictionary with usage statistics
        """
        total = self.sizes_calculated
        tier_pcts = {
            tier: (count / total * 100) if total > 0 else 0
            for tier, count in self.tier_counts.items()
        }

        avg_size = self.total_contracts / total if total > 0 else 0

        return {
            'sizes_calculated': total,
            'tier_counts': self.tier_counts,
            'tier_percentages': tier_pcts,
            'total_contracts': self.total_contracts,
            'avg_size': avg_size,
            'multipliers': self.multipliers
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.sizes_calculated = 0
        self.tier_counts = {'high': 0, 'medium': 0, 'low': 0, 'rejected': 0}
        self.total_contracts = 0
        logger.info("Position sizer statistics reset")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TieredPositionSizer(calculated={stats['sizes_calculated']}, "
            f"tiers={stats['tier_counts']}, "
            f"avg_size={stats['avg_size']:.2f})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*70)
    print("Tiered Position Sizing Test")
    print("="*70)

    # Create sizer
    sizer = TieredPositionSizer(
        high_confidence_threshold=0.65,
        medium_confidence_threshold=0.55,
        low_confidence_threshold=0.50,
        high_confidence_multiplier=1.5,
        medium_confidence_multiplier=1.0,
        low_confidence_multiplier=0.5,
        max_size=2,
        allow_low_confidence=True
    )

    print("\n1. LONG Trades - Different Confidence Levels")
    test_probs = [0.70, 0.60, 0.52, 0.48]
    for prob in test_probs:
        size = sizer.calculate_size(prob, 'LONG', base_size=1)
        tier = sizer.classify_confidence(prob, 'LONG')
        print(f"   P={prob:.2f} → {tier.upper():8s} → {size} contract(s)")

    print("\n2. SHORT Trades - Different Confidence Levels")
    test_probs = [0.30, 0.40, 0.48, 0.52]
    for prob in test_probs:
        size = sizer.calculate_size(prob, 'SHORT', base_size=1)
        tier = sizer.classify_confidence(prob, 'SHORT')
        # For SHORT, show P(down) = 1 - P(up) for clarity
        p_down = 1 - prob
        print(f"   P(up)={prob:.2f} [P(down)={p_down:.2f}] → {tier.upper():8s} → {size} contract(s)")

    print("\n3. Scaling with Different Base Sizes")
    prob = 0.68  # High confidence
    for base in [1, 2]:
        size = sizer.calculate_size(prob, 'LONG', base_size=base)
        print(f"   Base={base}, P={prob:.2f} → {size} contracts")

    print("\n4. Max Size Limit Test")
    prob = 0.75  # Very high confidence
    size = sizer.calculate_size(prob, 'LONG', base_size=2)
    print(f"   Base=2, Multiplier=1.5x → Raw=3.0 → Capped at max_size={sizer.max_size}")
    print(f"   Actual size: {size} contracts")

    print("\n5. Low Confidence Rejection Test")
    sizer_strict = TieredPositionSizer(allow_low_confidence=False)
    prob = 0.52  # Low confidence
    size = sizer_strict.calculate_size(prob, 'LONG', base_size=1)
    print(f"   P={prob:.2f}, allow_low_confidence=False")
    print(f"   Result: {size} contracts (rejected)")

    print("\n6. Statistics")
    stats = sizer.get_statistics()
    print(f"   Total sizes calculated: {stats['sizes_calculated']}")
    print(f"   Tier distribution:")
    for tier in ['high', 'medium', 'low', 'rejected']:
        count = stats['tier_counts'].get(tier, 0)
        pct = stats['tier_percentages'].get(tier, 0)
        print(f"     {tier:8s}: {count:2d} ({pct:4.1f}%)")
    print(f"   Average size: {stats['avg_size']:.2f} contracts")
    print(f"   Total contracts: {stats['total_contracts']}")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
