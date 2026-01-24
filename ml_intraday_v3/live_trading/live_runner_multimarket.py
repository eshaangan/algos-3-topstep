#!/usr/bin/env python3
"""
Multi-Market Live Trading Runner
Manages trading across MES (US) and NKD (Asian) sessions with portfolio-level risk management.

Usage:
    python live_runner_multimarket.py --config configs/live_trading_multimarket.yaml
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import pytz
import yaml
import json

# Placeholder imports - adjust based on your actual module structure
# from model_predictor import ModelPredictor
# from event_detector import EventDetector
# from data_fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """Manages aggregate risk across all markets"""

    def __init__(self, config: dict):
        self.config = config['portfolio_risk']
        self.max_daily_loss = self.config['max_daily_loss']
        self.max_trailing_drawdown = self.config['max_trailing_drawdown']
        self.max_concurrent_positions = self.config['max_concurrent_positions']

        # State tracking
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.active_positions = {}
        self.trades_today = []

    def can_trade(self, market: str) -> bool:
        """Check if portfolio-level risk allows new trade"""
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl} <= -{self.max_daily_loss}")
            return False

        # Check position limits
        if len(self.active_positions) >= self.max_concurrent_positions:
            logger.info(f"Max concurrent positions reached: {len(self.active_positions)}")
            return False

        # Check trailing drawdown
        current_drawdown = self.peak_balance - self.current_balance
        if current_drawdown >= self.max_trailing_drawdown:
            logger.warning(f"Max drawdown reached: {current_drawdown} >= {self.max_trailing_drawdown}")
            return False

        return True

    def update_pnl(self, market: str, pnl: float):
        """Update daily P&L from a trade"""
        self.daily_pnl += pnl
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        logger.info(f"Updated P&L - Market: {market}, Trade: ${pnl:.2f}, Daily: ${self.daily_pnl:.2f}")

    def reset_daily(self):
        """Reset daily counters (called at start of new trading day)"""
        logger.info("Resetting daily P&L counters")
        self.daily_pnl = 0.0
        self.trades_today = []


class MarketSession:
    """Represents a trading session for a specific market"""

    def __init__(self, market_name: str, config: dict):
        self.market_name = market_name
        self.config = config
        self.symbol = config['symbol']
        self.timezone = pytz.timezone(config['timezone'])

        # Parse trading hours (in CT)
        self.start_hour = int(config['trading_hours']['start'].split(':')[0])
        self.start_minute = int(config['trading_hours']['start'].split(':')[1])
        self.end_hour = int(config['trading_hours']['end'].split(':')[0])
        self.end_minute = int(config['trading_hours']['end'].split(':')[1])

        # Load model (placeholder - implement actual loading)
        self.model_path = config['model_bundle_path']
        logger.info(f"Loading {market_name} model from {self.model_path}")
        # self.model = ModelPredictor.load(self.model_path)

    def is_active(self, current_time_ct: datetime) -> bool:
        """Check if this market's session is currently active"""
        current_hour = current_time_ct.hour
        current_minute = current_time_ct.minute

        # Handle sessions that cross midnight
        if self.end_hour < self.start_hour:
            # Example: NKD 18:00-03:00
            return (current_hour >= self.start_hour or current_hour < self.end_hour) or \
                   (current_hour == self.start_hour and current_minute >= self.start_minute) or \
                   (current_hour == self.end_hour and current_minute < self.end_minute)
        else:
            # Example: MES 08:30-15:00
            if current_hour < self.start_hour or current_hour > self.end_hour:
                return False
            if current_hour == self.start_hour and current_minute < self.start_minute:
                return False
            if current_hour == self.end_hour and current_minute >= self.end_minute:
                return False
            return True

    def should_stop_trading(self, current_time_ct: datetime, buffer_minutes: int = 15) -> bool:
        """Check if we're within the buffer before session end"""
        # Calculate time until session end
        end_time = current_time_ct.replace(hour=self.end_hour, minute=self.end_minute, second=0)
        if current_time_ct > end_time:
            return True

        time_until_end = (end_time - current_time_ct).total_seconds() / 60
        return time_until_end <= buffer_minutes


class MultiMarketTrader:
    """Main orchestrator for multi-market trading"""

    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize markets
        self.markets = {}
        for market_name, market_config in self.config['markets'].items():
            self.markets[market_name] = MarketSession(market_name, market_config)
            logger.info(f"Initialized market: {market_name} ({market_config['symbol']})")

        # Initialize portfolio risk manager
        self.risk_manager = PortfolioRiskManager(self.config)

        # State file for persistence
        self.state_file = Path(self.config['gcp']['state_file_path'])
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_state()

        # CT timezone
        self.ct_tz = pytz.timezone('America/Chicago')

        logger.info("Multi-Market Trader initialized")
        logger.info(f"Environment: {self.config['trading']['environment']}")
        logger.info(f"Markets: {list(self.markets.keys())}")

    def load_state(self):
        """Load persistent state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.risk_manager.daily_pnl = state.get('daily_pnl', 0.0)
                    self.risk_manager.current_balance = state.get('current_balance', 0.0)
                    self.risk_manager.peak_balance = state.get('peak_balance', 0.0)
                    logger.info(f"Loaded state: P&L=${self.risk_manager.daily_pnl:.2f}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def save_state(self):
        """Save persistent state to file"""
        try:
            state = {
                'daily_pnl': self.risk_manager.daily_pnl,
                'current_balance': self.risk_manager.current_balance,
                'peak_balance': self.risk_manager.peak_balance,
                'last_updated': datetime.now(self.ct_tz).isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def get_active_market(self) -> Optional[MarketSession]:
        """Determine which market should be active right now"""
        current_time = datetime.now(self.ct_tz)

        for market_name, market in self.markets.items():
            if market.is_active(current_time):
                return market

        return None

    def run_trading_loop(self):
        """Main 24/7 trading loop"""
        logger.info("Starting 24/7 multi-market trading loop")

        last_active_market = None

        while True:
            try:
                current_time = datetime.now(self.ct_tz)

                # Check if we need to reset daily counters (new trading day)
                if current_time.hour == 0 and current_time.minute < 1:
                    self.risk_manager.reset_daily()

                # Determine active market
                active_market = self.get_active_market()

                if active_market is None:
                    if last_active_market is not None:
                        logger.info(f"No active markets at {current_time.strftime('%H:%M:%S')} CT")
                        last_active_market = None
                    time.sleep(60)  # Check every minute
                    continue

                # Log market switch
                if last_active_market != active_market:
                    logger.info(f"🔄 Market switch: {active_market.market_name} session active")
                    logger.info(f"   Trading {active_market.symbol} | Session: {active_market.config['trading_hours']}")
                    last_active_market = active_market

                # Check if we should stop trading (approaching session end)
                buffer_minutes = self.config['session_management']['transition_buffer_minutes']
                if active_market.should_stop_trading(current_time, buffer_minutes):
                    logger.info(f"Within {buffer_minutes}min of session end, stopping new trades")
                    time.sleep(60)
                    continue

                # Check portfolio-level risk
                if not self.risk_manager.can_trade(active_market.market_name):
                    logger.warning(f"Portfolio risk limits prevent trading in {active_market.market_name}")
                    time.sleep(60)
                    continue

                # Execute trading logic for active market
                self.trade_market(active_market, current_time)

                # Save state periodically
                if current_time.minute % self.config['gcp']['backup_frequency_minutes'] == 0:
                    self.save_state()

                # Sleep until next bar (adjust based on bar size)
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)

    def trade_market(self, market: MarketSession, current_time: datetime):
        """Execute trading logic for a specific market"""
        logger.debug(f"Trading {market.market_name} at {current_time.strftime('%H:%M:%S')}")

        # Placeholder for actual trading logic
        # 1. Fetch latest bar data
        # 2. Generate features
        # 3. Get model prediction
        # 4. Generate signal
        # 5. Execute trade if signal valid
        # 6. Update risk manager

        # Example:
        # bar_data = self.fetch_data(market.symbol)
        # features = self.generate_features(bar_data)
        # prediction = market.model.predict(features)
        # if prediction > threshold:
        #     self.execute_trade(market, prediction)

        pass

    def execute_trade(self, market: MarketSession, signal: dict):
        """Execute a trade and update risk manager"""
        logger.info(f"Executing trade in {market.market_name}: {signal}")

        # Placeholder for actual execution
        # 1. Send order to broker
        # 2. Get fill confirmation
        # 3. Update risk manager with P&L

        # Example:
        # fill = self.broker.send_order(...)
        # pnl = self.calculate_pnl(fill)
        # self.risk_manager.update_pnl(market.market_name, pnl)

        pass


def main():
    parser = argparse.ArgumentParser(description='Multi-Market Live Trading Runner')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to multi-market config YAML')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ML INTRADAY V3 - MULTI-MARKET LIVE TRADING")
    logger.info("=" * 60)

    trader = MultiMarketTrader(args.config)
    trader.run_trading_loop()


if __name__ == "__main__":
    main()
