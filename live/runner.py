"""
Live runner that polls TopstepX bars, evaluates ML signals, and routes orders.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from core.config import RiskConfig as CoreRiskConfig
from core.config import SessionConfig, StrategyConfig
from core.execution_live import LiveExecutionEngine
from core.projectx_client import ProjectXClient
from core.risk_management import RiskManager
from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from live.strategy import MLStrategy

LOGGER = logging.getLogger("live_runner")


def _build_configs() -> tuple[CoreRiskConfig, SessionConfig, StrategyConfig]:
    risk_cfg = CoreRiskConfig()
    risk_cfg.starting_balance = RISK_CONFIG.starting_balance
    risk_cfg.max_daily_loss = RISK_CONFIG.max_daily_loss
    risk_cfg.trailing_drawdown = RISK_CONFIG.trailing_drawdown
    risk_cfg.fixed_risk_per_trade = RISK_CONFIG.fixed_risk_per_trade
    risk_cfg.max_contracts = RISK_CONFIG.max_contracts
    risk_cfg.tick_size = RISK_CONFIG.tick_size
    risk_cfg.tick_value = RISK_CONFIG.tick_value
    risk_cfg.flat_by_time = RISK_CONFIG.session_end.strftime("%H:%M")
    risk_cfg.use_live_account_state = True
    risk_cfg.live_trading_enabled = True

    session_cfg = SessionConfig(
        timezone="America/Chicago",
        session_start=RISK_CONFIG.session_start.strftime("%H:%M"),
        session_end=RISK_CONFIG.session_end.strftime("%H:%M"),
        flat_buffer_minutes=5,
    )

    strategy_cfg = StrategyConfig()
    strategy_cfg.mode = "ml_only"
    strategy_cfg.stop_ticks = TRAINING_CONFIG.stop_loss_ticks
    strategy_cfg.target_rr_multiple = TRAINING_CONFIG.target_multiplier

    return risk_cfg, session_cfg, strategy_cfg


def _bars_to_df(bars: list) -> pd.DataFrame:
    rows = [
        {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    return pd.DataFrame(rows)


def run_loop(
    model_dir: str,
    symbol: str,
    poll_seconds: int,
    lookback_bars: int,
    unit_number: int,
    live_trading: bool,
) -> None:
    load_dotenv()

    client = ProjectXClient()
    risk_cfg, session_cfg, strategy_cfg = _build_configs()
    risk_manager = RiskManager(risk_cfg, session_cfg)
    execution = LiveExecutionEngine(client, risk_manager, risk_cfg, strategy_cfg)
    strategy = MLStrategy(model_dir=model_dir, symbol=symbol)

    last_bar_ts: Optional[pd.Timestamp] = None

    while True:
        try:
            if live_trading:
                account = client.get_account_state()
                risk_manager.sync_from_live(account.equity, account.open_pnl, account.realized_pnl)

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=lookback_bars * unit_number)

            bars = client.retrieve_bars(
                start_time=start_time,
                end_time=end_time,
                unit=2,
                unit_number=unit_number,
                limit=lookback_bars,
                include_partial_bar=False,
            )

            if not bars:
                time.sleep(poll_seconds)
                continue

            bars_df = _bars_to_df(bars)
            latest_ts = pd.to_datetime(bars_df["timestamp"].iloc[-1], utc=True)
            if last_bar_ts is not None and latest_ts <= last_bar_ts:
                time.sleep(poll_seconds)
                continue

            signal = strategy.generate_signal(bars_df)
            if signal:
                if live_trading:
                    result = execution.handle_signal(signal)
                    if result.order:
                        LOGGER.info("Order placed: %s", result.order.order_id)
                    else:
                        LOGGER.warning("Signal rejected: %s", result.rejection_reason)
                else:
                    LOGGER.info("PAPER signal: %s", signal.reason)

            if live_trading:
                execution.reconcile_open_orders()

            last_bar_ts = latest_ts
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Live loop error: %s", exc)

        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live ML strategy")
    parser.add_argument("--model-dir", default="models/nn_saved")
    parser.add_argument("--symbol", default="MES")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--lookback-bars", type=int, default=200)
    parser.add_argument("--unit-number", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Enable live order routing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_loop(
        model_dir=args.model_dir,
        symbol=args.symbol,
        poll_seconds=args.poll_seconds,
        lookback_bars=args.lookback_bars,
        unit_number=args.unit_number,
        live_trading=args.live,
    )


if __name__ == "__main__":
    main()
