"""
NN strategy wrapper that converts model probabilities into trade signals.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import time as dt_time
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from core.selection import DailyTopNSelector, in_session
from core.models import Signal, SignalAction
from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from models.nn_inference import load_nn_bundle, predict_latest

LOGGER = logging.getLogger(__name__)


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


class MLStrategy:
    """Loads tiny MLP models and emits trade signals based on calibrated scores."""

    def __init__(self, model_dir: str = "models/nn_saved", symbol: str = "MES", fold: int = 0) -> None:
        self.model_dir = Path(model_dir)
        self.symbol = symbol
        self.bundle = load_nn_bundle(str(self.model_dir), fold=fold)
        self.metadata = self.bundle.config
        nn_cfg = self.metadata.get("nn_config", {})
        self.nn_cfg = nn_cfg

        risk_meta = self.metadata.get("risk_config", {}) if isinstance(self.metadata, dict) else {}
        self.session_start = RISK_CONFIG.session_start
        self.session_end = RISK_CONFIG.session_end
        if isinstance(risk_meta.get("session_start"), str):
            self.session_start = dt_time.fromisoformat(risk_meta["session_start"])
        if isinstance(risk_meta.get("session_end"), str):
            self.session_end = dt_time.fromisoformat(risk_meta["session_end"])

        self.score_threshold = float(nn_cfg["score_threshold"])
        self.enable_long = bool(nn_cfg["enable_long"])
        self.enable_short = bool(nn_cfg["enable_short"])
        self.max_trades_per_day = int(nn_cfg["max_trades_per_day"])
        self.min_bars_between_trades = int(nn_cfg["min_bars_between_trades"])
        self.session_mode = str(nn_cfg.get("session_mode", "RTH"))
        self.deadline_time = nn_cfg.get("deadline_time")
        self.deadline_relax_factor = float(nn_cfg.get("deadline_relax_factor", 0.98))
        self.execution_mode = str(nn_cfg.get("execution_mode", "time_exit"))
        self.exit_price_mode = str(nn_cfg.get("exit_price_mode", "next_open"))
        self.horizon_bars = int(nn_cfg["horizon_bars"])
        self.bar_minutes = int(nn_cfg.get("bar_minutes", 5))
        self.stop_loss_ticks = int(nn_cfg.get("stop_loss_ticks", TRAINING_CONFIG.stop_loss_ticks))
        self.target_multiplier = float(nn_cfg.get("target_multiplier", TRAINING_CONFIG.target_multiplier))
        self.tick_size = float(nn_cfg.get("tick_size", RISK_CONFIG.tick_size))
        self.tick_value = float(nn_cfg.get("tick_value", RISK_CONFIG.tick_value))

        self.selector = DailyTopNSelector(
            max_trades_per_day=self.max_trades_per_day,
            min_bars_between_trades=self.min_bars_between_trades,
            score_threshold=self.score_threshold,
            bar_minutes=self.bar_minutes,
            session_mode=self.session_mode,
            session_start=self.session_start,
            session_end=self.session_end,
            deadline_time=self.deadline_time or pd.Timestamp("11:30").time(),
            deadline_relax_factor=self.deadline_relax_factor,
        )
        self.open_position: Optional[Dict[str, object]] = None

    def _bar_index(self, ts: pd.Timestamp) -> int:
        ts = pd.to_datetime(ts, utc=True)
        nanos_per_bar = int(self.bar_minutes * 60 * 1_000_000_000)
        return int(ts.value // nanos_per_bar)

    def generate_signal(self, bars_df: pd.DataFrame) -> Optional[Signal]:
        if bars_df.empty:
            return None

        bars_df = bars_df.copy()
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

        probs = predict_latest(bars_df, self.bundle)
        if probs is None:
            return None

        last_row = bars_df.iloc[-1]
        bar_index = self._bar_index(last_row["timestamp"])

        if self.open_position is not None and self.execution_mode == "time_exit":
            entry_idx = int(self.open_position["entry_idx"])
            hold_bars = bar_index - entry_idx
            if hold_bars >= self.horizon_bars:
                reason = f"TIME_EXIT after {hold_bars} bars"
                if self.exit_price_mode == "next_open":
                    exit_price = float(last_row["close"])
                else:
                    exit_price = float(last_row["close"])
                self.open_position = None
                return Signal(
                    action=SignalAction.EXIT,
                    symbol=self.symbol,
                    timestamp=last_row["timestamp"].to_pydatetime(),
                    stop_price=exit_price,
                    target_price=exit_price,
                    reason=reason,
                    metadata={
                        "strategy_mode": "ml_only",
                        "execution_mode": self.execution_mode,
                        "exit_price_mode": self.exit_price_mode,
                    },
                )

        if self.open_position is not None:
            return None

        if not in_session(
            last_row["timestamp"],
            session_mode=self.session_mode,
            session_start=self.session_start,
            session_end=self.session_end,
        ):
            return None

        long_prob = probs["long_prob"]
        short_prob = probs["short_prob"]
        score = probs["score"]
        direction = probs["direction"]

        if direction == "long" and not self.enable_long:
            return None
        if direction == "short" and not self.enable_short:
            return None

        if not self.selector.should_enter(
            last_row["timestamp"], score=float(score), direction=str(direction), bar_index=bar_index
        ):
            self.selector.log_rejection()
            return None

        # Use fixed stop from config to match backtest and model training
        # This ensures live performance matches backtest expectations
        stop_ticks = self.stop_loss_ticks

        last_close = float(last_row["close"])
        stop_distance = stop_ticks * self.tick_size
        target_distance = stop_distance * self.target_multiplier

        if direction == "long":
            action = SignalAction.ENTER_LONG
            stop_price = last_close - stop_distance
            target_price = last_close + target_distance
        else:
            action = SignalAction.ENTER_SHORT
            stop_price = last_close + stop_distance
            target_price = last_close - target_distance

        metadata = {
            "ideal_entry": last_close,
            "risk_ticks": stop_ticks,
            "target_rr_multiple": self.target_multiplier,
            "strategy_mode": "ml_only",
            "long_prob": long_prob,
            "short_prob": short_prob,
            "score": score,
            "score_threshold": float(self.score_threshold),
            "execution_mode": self.execution_mode,
        }

        reason = (
            f"ML signal {direction.upper()} | score={score:.3f} "
            f"long_prob={long_prob:.3f} short_prob={short_prob:.3f}"
        )

        self.selector.register_trade(last_row["timestamp"], bar_index=bar_index)
        self.open_position = {
            "entry_time": last_row["timestamp"],
            "entry_idx": bar_index,
            "direction": direction,
        }

        return Signal(
            action=action,
            symbol=self.symbol,
            timestamp=last_row["timestamp"].to_pydatetime(),
            stop_price=stop_price,
            target_price=target_price,
            reason=reason,
            metadata=metadata,
        )

    def describe(self) -> Dict[str, object]:
        return {
            "model_dir": str(self.model_dir),
            "symbol": self.symbol,
            "config": {
                "risk": asdict(RISK_CONFIG),
                "training": asdict(TRAINING_CONFIG),
                "nn": self.metadata.get("nn_config", {}),
            },
        }
