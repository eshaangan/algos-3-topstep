"""
Dynamic Stop/Target Adjustment - Phase 2b Priority #2

Adjusts stop-loss and take-profit levels based on current volatility regime.

Key Concept:
- Fixed stops don't adapt to market conditions
- In high volatility: Need wider stops to avoid premature stop-outs
- In low volatility: Can use tighter stops for better R:R
- Dynamic stops scale with ATR based on volatility percentile

Expected Impact: Reduce avg loss by $3-5, increase win rate by reducing false stops

Research Context:
- "Adaptive Trading" (Perry Kaufman) shows dynamic stops outperform fixed stops
- Volatility-adjusted stops reduce premature exits by 20-30%
- Maintains minimum Risk:Reward ratio for positive expectancy

Usage:
    stops_calc = DynamicStops(min_risk_reward=2.0)

    # Calculate stops for a trade
    stops = stops_calc.calculate_stops(
        entry_price=5000.0,
        side='LONG',
        current_atr=12.5,
        vol_regime='normal'
    )

    # Returns: {'stop': 4975.0, 'target': 5050.0, 'risk': 25, 'reward': 50}
"""

import logging
from typing import Literal, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SideType = Literal['LONG', 'SHORT']
VolRegimeType = Literal['low', 'normal', 'high']


class DynamicStops:
    """
    Calculate dynamic stop-loss and take-profit levels based on volatility.

    Strategy:
    - Classify current volatility regime (low/normal/high)
    - Scale stop distance by ATR multiplier based on regime
    - Maintain minimum Risk:Reward ratio
    - Prevent premature stop-outs in volatile markets
    """

    def __init__(
        self,
        atr_multiplier_low_vol: float = 1.5,
        atr_multiplier_normal: float = 2.0,
        atr_multiplier_high_vol: float = 2.5,
        min_risk_reward: float = 2.0,
        vol_low_percentile: float = 30.0,
        vol_high_percentile: float = 70.0
    ):
        """
        Initialize dynamic stops calculator.

        Args:
            atr_multiplier_low_vol: ATR multiplier for low volatility (default: 1.5)
            atr_multiplier_normal: ATR multiplier for normal volatility (default: 2.0)
            atr_multiplier_high_vol: ATR multiplier for high volatility (default: 2.5)
            min_risk_reward: Minimum risk:reward ratio (default: 2.0)
            vol_low_percentile: Percentile threshold for low vol (default: 30)
            vol_high_percentile: Percentile threshold for high vol (default: 70)
        """
        self.atr_mult = {
            'low': atr_multiplier_low_vol,
            'normal': atr_multiplier_normal,
            'high': atr_multiplier_high_vol
        }
        self.min_rr = min_risk_reward
        self.vol_low_pct = vol_low_percentile
        self.vol_high_pct = vol_high_percentile

        # Statistics tracking
        self.stops_calculated = 0
        self.regime_counts = {'low': 0, 'normal': 0, 'high': 0}

        logger.info(
            f"DynamicStops initialized: "
            f"low={atr_multiplier_low_vol}x, "
            f"normal={atr_multiplier_normal}x, "
            f"high={atr_multiplier_high_vol}x, "
            f"min_RR={min_risk_reward}"
        )

    def classify_volatility_regime(
        self,
        current_atr: float,
        recent_bars: pd.DataFrame,
        atr_column: str = 'atr_14'
    ) -> VolRegimeType:
        """
        Classify current volatility regime.

        Args:
            current_atr: Current ATR value
            recent_bars: Recent bars with ATR history
            atr_column: Column name for ATR in recent_bars

        Returns:
            Volatility regime: 'low', 'normal', or 'high'

        Logic:
            - Calculate ATR percentile from recent history (last 100 bars)
            - < 30th percentile: Low volatility
            - 30-70th percentile: Normal volatility
            - > 70th percentile: High volatility
        """
        try:
            # Get recent ATR history
            if atr_column in recent_bars.columns:
                recent_atrs = recent_bars[atr_column].dropna().tail(100)
            else:
                # Fallback: calculate simple ATR if column not available
                recent_atrs = self._calculate_simple_atr(recent_bars, period=14)

            if len(recent_atrs) < 20:
                logger.warning(f"Insufficient ATR history ({len(recent_atrs)} bars), using 'normal'")
                return 'normal'

            # Calculate percentile
            percentile = (np.sum(recent_atrs < current_atr) / len(recent_atrs)) * 100

            # Classify regime
            if percentile < self.vol_low_pct:
                regime = 'low'
            elif percentile < self.vol_high_pct:
                regime = 'normal'
            else:
                regime = 'high'

            logger.debug(
                f"Volatility regime: {regime} "
                f"(ATR={current_atr:.2f}, percentile={percentile:.1f})"
            )

            return regime

        except Exception as e:
            logger.warning(f"Error classifying volatility regime: {e}, using 'normal'")
            return 'normal'

    def _calculate_simple_atr(self, bars_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate simple ATR from OHLC bars."""
        try:
            high = bars_df['high'].values
            low = bars_df['low'].values
            close = bars_df['close'].values

            # True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))

            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # Convert to Series for rolling calculation
            tr_series = pd.Series(true_range, index=bars_df.index)
            atr = tr_series.ewm(span=period, adjust=False).mean()

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([10.0] * len(bars_df), index=bars_df.index)

    def calculate_stops(
        self,
        entry_price: float,
        side: SideType,
        current_atr: float,
        vol_regime: VolRegimeType = 'normal',
        risk_override: float = None
    ) -> Dict[str, float]:
        """
        Calculate stop and target prices based on volatility regime.

        Args:
            entry_price: Entry price for the trade
            side: 'LONG' or 'SHORT'
            current_atr: Current ATR value
            vol_regime: Volatility regime ('low', 'normal', 'high')
            risk_override: Optional manual risk amount (overrides ATR calculation)

        Returns:
            Dictionary with:
                - 'stop': Stop-loss price
                - 'target': Take-profit price
                - 'risk': Risk amount (dollars)
                - 'reward': Reward amount (dollars)
                - 'risk_reward': Risk:Reward ratio
                - 'stop_distance_atr': Stop distance in ATR multiples

        Example:
            For LONG at 5000 with ATR=10, normal vol (2.0x):
            - Stop distance = 10 * 2.0 = 20 points
            - Stop = 5000 - 20 = 4980
            - Target = 5000 + (20 * 2.0) = 5040
            - Risk = $20 * $5 = $100
            - Reward = $40 * $5 = $200
            - R:R = 2.0
        """
        self.stops_calculated += 1
        self.regime_counts[vol_regime] = self.regime_counts.get(vol_regime, 0) + 1

        # Get ATR multiplier for this regime
        atr_mult = self.atr_mult.get(vol_regime, 2.0)

        # Calculate stop distance
        if risk_override is not None:
            stop_distance = risk_override
        else:
            stop_distance = current_atr * atr_mult

        # Calculate reward distance (maintain min R:R)
        reward_distance = stop_distance * self.min_rr

        # Calculate stop and target prices
        if side == 'LONG':
            stop_price = entry_price - stop_distance
            target_price = entry_price + reward_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
            target_price = entry_price - reward_distance

        # Calculate dollar amounts (MES: $5 per point)
        risk_dollars = stop_distance * 5.0
        reward_dollars = reward_distance * 5.0

        result = {
            'stop': stop_price,
            'target': target_price,
            'risk': risk_dollars,
            'reward': reward_dollars,
            'risk_reward': self.min_rr,
            'stop_distance_atr': atr_mult,
            'stop_distance_points': stop_distance,
            'regime': vol_regime
        }

        logger.debug(
            f"Calculated stops: side={side}, regime={vol_regime}, "
            f"entry={entry_price:.2f}, stop={stop_price:.2f}, "
            f"target={target_price:.2f}, R:R={self.min_rr:.1f}"
        )

        return result

    def get_statistics(self) -> dict:
        """
        Get dynamic stops statistics.

        Returns:
            Dictionary with usage statistics
        """
        total = self.stops_calculated
        regime_pcts = {
            regime: (count / total * 100) if total > 0 else 0
            for regime, count in self.regime_counts.items()
        }

        return {
            'stops_calculated': total,
            'regime_counts': self.regime_counts,
            'regime_percentages': regime_pcts,
            'atr_multipliers': self.atr_mult
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stops_calculated = 0
        self.regime_counts = {'low': 0, 'normal': 0, 'high': 0}
        logger.info("Dynamic stops statistics reset")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"DynamicStops(calculated={stats['stops_calculated']}, "
            f"regimes={stats['regime_counts']}, "
            f"min_RR={self.min_rr})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*70)
    print("Dynamic Stops Test")
    print("="*70)

    # Create calculator
    stops_calc = DynamicStops(
        atr_multiplier_low_vol=1.5,
        atr_multiplier_normal=2.0,
        atr_multiplier_high_vol=2.5,
        min_risk_reward=2.0
    )

    # Test scenarios
    entry_price = 5000.0
    current_atr = 10.0

    print("\n1. LONG Trade - Different Volatility Regimes")
    print(f"   Entry: ${entry_price:.2f}, ATR: {current_atr:.2f}")

    for regime in ['low', 'normal', 'high']:
        stops = stops_calc.calculate_stops(entry_price, 'LONG', current_atr, regime)
        print(f"\n   {regime.upper()} Volatility ({stops_calc.atr_mult[regime]}x ATR):")
        print(f"     Stop:   ${stops['stop']:.2f} (risk: ${stops['risk']:.2f})")
        print(f"     Target: ${stops['target']:.2f} (reward: ${stops['reward']:.2f})")
        print(f"     R:R:    {stops['risk_reward']:.1f}")

    print("\n2. SHORT Trade - Normal Volatility")
    stops = stops_calc.calculate_stops(entry_price, 'SHORT', current_atr, 'normal')
    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Stop:   ${stops['stop']:.2f} (risk: ${stops['risk']:.2f})")
    print(f"   Target: ${stops['target']:.2f} (reward: ${stops['reward']:.2f})")
    print(f"   R:R:    {stops['risk_reward']:.1f}")

    print("\n3. High Volatility Example (ATR=20)")
    current_atr = 20.0
    stops = stops_calc.calculate_stops(entry_price, 'LONG', current_atr, 'high')
    print(f"   Entry: ${entry_price:.2f}, ATR: {current_atr:.2f}")
    print(f"   Stop:   ${stops['stop']:.2f} (${current_atr * 2.5:.1f} below entry)")
    print(f"   Target: ${stops['target']:.2f} (${current_atr * 2.5 * 2:.1f} above entry)")
    print(f"   Risk:   ${stops['risk']:.2f}")
    print(f"   Reward: ${stops['reward']:.2f}")

    print("\n4. Volatility Regime Classification Test")
    # Create synthetic ATR history
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    recent_bars = pd.DataFrame({
        'high': np.random.uniform(4990, 5010, 100),
        'low': np.random.uniform(4990, 5010, 100),
        'close': np.random.uniform(4990, 5010, 100),
        'atr_14': np.random.uniform(8, 15, 100)
    }, index=dates)

    test_atrs = [7.0, 10.0, 16.0]
    for test_atr in test_atrs:
        regime = stops_calc.classify_volatility_regime(test_atr, recent_bars)
        print(f"   ATR={test_atr:.1f} → {regime.upper()} volatility")

    print("\n5. Statistics")
    stats = stops_calc.get_statistics()
    print(f"   Stops calculated: {stats['stops_calculated']}")
    print(f"   Regime distribution:")
    for regime, pct in stats['regime_percentages'].items():
        print(f"     {regime}: {pct:.1f}%")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
