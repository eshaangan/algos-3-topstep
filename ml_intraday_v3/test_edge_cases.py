#!/usr/bin/env python3
"""
Edge Case Tests for BALANCED_V3 Model

Tests critical edge cases and stress scenarios for Topstep compliance.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class EdgeCaseTester:
    """Test edge cases and stress scenarios."""

    def __init__(self):
        self.test_results = []

    def test_regime_transition_lag(self):
        """
        Test: What happens during a regime transition when SMA lags?
        
        Scenario: Flash crash from bull to bear
        - Day 0: Market crashes -10%
        - Day 1-50: SMA still points UP (lag period)
        - Model may attempt LONG trades in a new bear market
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 1: REGIME TRANSITION LAG")
        logger.info("="*80)
        
        logger.info("\nScenario: Flash crash from bull market to bear market")
        logger.info("Expected: SMA will lag ~50 days, potentially allowing LONG trades in bear")
        
        # Simulate: Create synthetic data with flash crash
        # Bull market: 100 days of uptrend
        # Crash: -10% in 1 day
        # Bear market: 50 days of downtrend
        
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        prices = []
        
        # Bull: linear uptrend
        for i in range(100):
            prices.append(5000 + i * 10)  # +10 points/day
        
        # Crash: -10%
        crash_price = prices[-1] * 0.9
        prices.append(crash_price)
        
        # Bear: downtrend
        for i in range(49):
            prices.append(crash_price - i * 5)
        
        df = pd.DataFrame({'date': dates, 'close': prices})
        
        # Calculate SMA-50
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['regime'] = np.where(df['close'] > df['sma_50'], 1, -1)
        
        # Find lag period after crash
        crash_idx = 100
        lag_period = 0
        for i in range(crash_idx + 1, len(df)):
            if df.iloc[i]['regime'] == 1:  # Still bull regime
                lag_period += 1
            else:
                break
        
        logger.info(f"\nResults:")
        logger.info(f"  Crash Day: Day {crash_idx} (Price: {prices[crash_idx]:.0f})")
        logger.info(f"  Regime Lag Period: {lag_period} days")
        logger.info(f"  Risk: Model would attempt LONG trades for {lag_period} days in bear market")
        
        # Calculate potential loss
        # Assume 1 trade/day, avg loss $20
        potential_loss = lag_period * 20
        logger.info(f"  Estimated Risk Exposure: ~${potential_loss:.0f}")
        
        if lag_period > 30:
            status = "❌ HIGH RISK"
            recommendation = "Add fast SMA (10-day) or volatility filter"
        elif lag_period > 15:
            status = "⚠️  MODERATE RISK"
            recommendation = "Monitor closely, consider secondary filter"
        else:
            status = "✅ LOW RISK"
            recommendation = "Acceptable lag, monitor in live trading"
        
        logger.info(f"\n  Status: {status}")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info("="*80 + "\n")
        
        return {
            'test': 'regime_transition_lag',
            'lag_days': lag_period,
            'potential_loss': potential_loss,
            'status': status,
            'recommendation': recommendation
        }

    def test_whipsaw_market(self):
        """
        Test: What happens in a whipsaw/choppy market?
        
        Scenario: Price oscillates around SMA
        - Frequent regime changes
        - Many trades get filtered
        - Potential opportunity cost
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 2: WHIPSAW MARKET")
        logger.info("="*80)
        
        logger.info("\nScenario: Price oscillates around SMA (choppy market)")
        
        # Simulate: Sideways market with high chop
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = []
        base = 5000
        
        for i in range(100):
            # Oscillate +/- 2% around base
            noise = np.sin(i * 0.5) * 100  # ±100 points
            prices.append(base + noise)
        
        df = pd.DataFrame({'date': dates, 'close': prices})
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['regime'] = np.where(df['close'] > df['sma_50'], 1, -1)
        
        # Count regime changes
        regime_changes = (df['regime'] != df['regime'].shift()).sum()
        
        # Estimate trades filtered
        # Assume model generates 1 signal/day, 50% would be counter-trend
        total_potential_trades = 100
        filtered_trades = regime_changes * 5  # Rough estimate
        
        logger.info(f"\nResults:")
        logger.info(f"  Regime Changes: {regime_changes}")
        logger.info(f"  Estimated Trades Filtered: {filtered_trades}")
        logger.info(f"  Trade Opportunity Loss: {filtered_trades / total_potential_trades * 100:.0f}%")
        
        if regime_changes > 20:
            status = "❌ HIGH WHIPSAW RISK"
            recommendation = "Consider disabling filter in low-volatility periods"
        elif regime_changes > 10:
            status = "⚠️  MODERATE WHIPSAW"
            recommendation = "Monitor filter effectiveness"
        else:
            status = "✅ LOW WHIPSAW"
            recommendation = "Filter working as intended"
        
        logger.info(f"\n  Status: {status}")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info("="*80 + "\n")
        
        return {
            'test': 'whipsaw_market',
            'regime_changes': regime_changes,
            'filtered_trades': filtered_trades,
            'status': status,
            'recommendation': recommendation
        }

    def test_max_consecutive_losses(self):
        """
        Test: Can the model hit 40+ consecutive losses?
        
        From the latest run: Max Consecutive Losses = 40
        This is VERY high and indicates a fundamental issue.
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 3: MAX CONSECUTIVE LOSSES (CRITICAL)")
        logger.info("="*80)
        
        logger.info("\nObserved: Max Consecutive Losses = 40 (from latest run)")
        logger.info("Topstep Risk: At 5 consecutive losses, most traders reduce size.")
        logger.info("At 40 losses: Account is in severe distress.")
        
        # Calculate drawdown from 40 consecutive losses
        avg_loss = 21.37  # From metrics
        consecutive_loss_dd = 40 * avg_loss
        
        logger.info(f"\nImpact Analysis:")
        logger.info(f"  Avg Loss per Trade: ${avg_loss:.2f}")
        logger.info(f"  40 Consecutive Losses: ${consecutive_loss_dd:.2f}")
        logger.info(f"  % of Daily Loss Limit (${1000}): {consecutive_loss_dd / 1000 * 100:.0f}%")
        logger.info(f"  % of Max Drawdown (${2500}): {consecutive_loss_dd / 2500 * 100:.0f}%")
        
        # Check if this would breach Topstep limits
        if consecutive_loss_dd > 2500:
            status = "❌ CRITICAL: WOULD BREACH MAX DRAWDOWN"
        elif consecutive_loss_dd > 1000:
            status = "❌ HIGH RISK: WOULD BREACH DAILY LOSS LIMIT"
        elif consecutive_loss_dd > 750:
            status = "⚠️  WARNING: Close to limits"
        else:
            status = "✅ WITHIN LIMITS"
        
        logger.info(f"\n  Status: {status}")
        
        # Root cause analysis
        logger.info(f"\n  Root Cause:")
        logger.info(f"    - Win Rate: 34.9% (Too low)")
        logger.info(f"    - Avg Loss > Avg Win (Poor risk/reward)")
        logger.info(f"    - Model is in a losing streak phase")
        
        recommendation = "IMMEDIATE ACTION REQUIRED: Investigate why win rate is so low"
        logger.info(f"\n  Recommendation: {recommendation}")
        logger.info("="*80 + "\n")
        
        return {
            'test': 'max_consecutive_losses',
            'consecutive_losses': 40,
            'drawdown_from_losses': consecutive_loss_dd,
            'status': status,
            'recommendation': recommendation
        }

    def test_gap_events(self):
        """
        Test: What happens with large overnight gaps?
        
        Scenario: Market gaps up/down 5% overnight
        - SMA doesn't account for gap
        - Regime may be incorrect
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 4: GAP EVENTS")
        logger.info("="*80)
        
        logger.info("\nScenario: 5% overnight gap (news event)")
        
        # Simulate: Normal trading, then 5% gap
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = []
        
        for i in range(50):
            prices.append(5000 + i * 2)  # Slow uptrend
        
        # Gap up 5%
        gap_price = prices[-1] * 1.05
        prices.append(gap_price)
        
        for i in range(9):
            prices.append(gap_price + i * 2)  # Continue uptrend
        
        df = pd.DataFrame({'date': dates, 'close': prices})
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['regime_before'] = df['regime'] = np.where(df['close'] > df['sma_50'], 1, -1)
        
        # Check regime on gap day
        gap_day = 50
        regime_before = df.iloc[gap_day - 1]['regime']
        regime_after = df.iloc[gap_day]['regime']
        
        logger.info(f"\nResults:")
        logger.info(f"  Gap Day: Day {gap_day}")
        logger.info(f"  Gap Size: {(prices[gap_day] - prices[gap_day-1]) / prices[gap_day-1] * 100:.1f}%")
        logger.info(f"  Regime Before Gap: {'Bull' if regime_before == 1 else 'Bear'}")
        logger.info(f"  Regime After Gap: {'Bull' if regime_after == 1 else 'Bear'}")
        
        if regime_before == regime_after:
            status = "✅ NO ISSUE: Regime unchanged"
            risk = "Low"
        else:
            status = "⚠️  REGIME CHANGED: Potential false signal"
            risk = "Moderate"
        
        logger.info(f"\n  Status: {status}")
        logger.info(f"  Risk Level: {risk}")
        logger.info(f"  Recommendation: Monitor gap days closely in live trading")
        logger.info("="*80 + "\n")
        
        return {
            'test': 'gap_events',
            'regime_changed': regime_before != regime_after,
            'status': status,
            'risk': risk
        }

    def test_daily_loss_limit_buffer(self):
        """
        Test: How close did we get to daily loss limit?
        
        From latest run: Worst Day = -$552.36
        Buffer: $447.64 (44.8% of limit)
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 5: DAILY LOSS LIMIT BUFFER")
        logger.info("="*80)
        
        worst_day_loss = 552.36
        daily_limit = 1000.0
        buffer = daily_limit - worst_day_loss
        buffer_pct = buffer / daily_limit * 100
        
        logger.info(f"\nDaily Loss Limit: ${daily_limit}")
        logger.info(f"Worst Day Loss: ${worst_day_loss:.2f}")
        logger.info(f"Buffer: ${buffer:.2f} ({buffer_pct:.1f}%)")
        
        # Risk assessment
        if buffer_pct < 10:
            status = "❌ CRITICAL: Very close to limit"
            recommendation = "Reduce position size immediately"
        elif buffer_pct < 25:
            status = "⚠️  WARNING: Close to limit"
            recommendation = "Monitor closely, consider tighter stops"
        elif buffer_pct < 50:
            status = "⚠️  CAUTION: Moderate buffer"
            recommendation = "Acceptable but watch for consecutive bad days"
        else:
            status = "✅ HEALTHY: Good buffer"
            recommendation = "Continue monitoring"
        
        logger.info(f"\n  Status: {status}")
        logger.info(f"  Recommendation: {recommendation}")
        
        # Stress test: What if we have 2 worst days in a row?
        two_day_loss = worst_day_loss * 2
        logger.info(f"\n  Stress Test: 2 consecutive worst days")
        logger.info(f"    Total Loss: ${two_day_loss:.2f}")
        if two_day_loss > daily_limit:
            logger.info(f"    ❌ WOULD BREACH on Day 2")
        else:
            logger.info(f"    ✅ Still within limit")
        
        logger.info("="*80 + "\n")
        
        return {
            'test': 'daily_loss_limit_buffer',
            'worst_day': worst_day_loss,
            'buffer': buffer,
            'buffer_pct': buffer_pct,
            'status': status,
            'recommendation': recommendation
        }

    def test_max_drawdown_buffer(self):
        """
        Test: How close did we get to max drawdown?
        
        From latest run: Max Drawdown = -$2,382.52
        Buffer: $117.48 (4.7% of limit) - VERY CLOSE!
        """
        logger.info("="*80)
        logger.info("EDGE CASE TEST 6: MAX DRAWDOWN BUFFER (CRITICAL)")
        logger.info("="*80)
        
        max_dd = 2382.52
        dd_limit = 2500.0
        buffer = dd_limit - max_dd
        buffer_pct = buffer / dd_limit * 100
        
        logger.info(f"\nMax Drawdown Limit: ${dd_limit}")
        logger.info(f"Observed Max Drawdown: ${max_dd:.2f}")
        logger.info(f"Buffer: ${buffer:.2f} ({buffer_pct:.1f}%)")
        
        # Risk assessment
        if buffer_pct < 5:
            status = "❌ CRITICAL: DANGEROUSLY CLOSE TO BREACH"
            recommendation = "IMMEDIATE ACTION: Reduce size or stop trading"
        elif buffer_pct < 10:
            status = "❌ HIGH RISK: Very close to limit"
            recommendation = "Reduce position size by 50%"
        elif buffer_pct < 20:
            status = "⚠️  WARNING: Close to limit"
            recommendation = "Reduce position size, tighten stops"
        else:
            status = "✅ ACCEPTABLE: Moderate buffer"
            recommendation = "Continue monitoring"
        
        logger.info(f"\n  Status: {status}")
        logger.info(f"  Recommendation: {recommendation}")
        
        # Critical insight
        logger.info(f"\n  ⚠️  CRITICAL INSIGHT:")
        logger.info(f"    The model came within $117 of failing the combine.")
        logger.info(f"    This is TOO CLOSE for live trading.")
        logger.info(f"    Must improve risk management before deployment.")
        
        logger.info("="*80 + "\n")
        
        return {
            'test': 'max_drawdown_buffer',
            'max_drawdown': max_dd,
            'buffer': buffer,
            'buffer_pct': buffer_pct,
            'status': status,
            'recommendation': recommendation
        }


def main():
    logger.info("="*80)
    logger.info("BALANCED_V3: EDGE CASE & STRESS TEST BATTERY")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = EdgeCaseTester()
    
    # Run all edge case tests
    results = []
    
    results.append(tester.test_regime_transition_lag())
    results.append(tester.test_whipsaw_market())
    results.append(tester.test_max_consecutive_losses())
    results.append(tester.test_gap_events())
    results.append(tester.test_daily_loss_limit_buffer())
    results.append(tester.test_max_drawdown_buffer())
    
    # Summary
    logger.info("="*80)
    logger.info("EDGE CASE TEST SUMMARY")
    logger.info("="*80)
    
    critical_issues = [r for r in results if '❌' in r.get('status', '')]
    warnings = [r for r in results if '⚠️' in r.get('status', '')]
    
    logger.info(f"\nCritical Issues: {len(critical_issues)}")
    for issue in critical_issues:
        logger.info(f"  - {issue['test']}: {issue['status']}")
    
    logger.info(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        logger.info(f"  - {warning['test']}: {warning['status']}")
    
    logger.info(f"\n{'='*80}")
    
    if len(critical_issues) > 0:
        logger.error("❌ MODEL HAS CRITICAL ISSUES - NOT READY FOR LIVE DEPLOYMENT")
        logger.error("   MUST ADDRESS CRITICAL ISSUES BEFORE PROCEEDING")
    elif len(warnings) > 0:
        logger.warning("⚠️  MODEL HAS WARNINGS - DEPLOY WITH CAUTION")
        logger.warning("   MONITOR CLOSELY IN LIVE TRADING")
    else:
        logger.info("✅ ALL EDGE CASES PASSED - MODEL READY FOR DEPLOYMENT")
    
    logger.info(f"{'='*80}\n")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path("ml_intraday_v3/edge_case_test_results.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
