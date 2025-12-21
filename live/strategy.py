"""
NN strategy wrapper that converts model probabilities into trade signals.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from core.daily_trade_budget import DailyTradeBudget
from core.models import Signal, SignalAction
from core.simple_config import NN_CONFIG, RISK_CONFIG, TRAINING_CONFIG
from models.nn_inference import load_nn_bundle, predict_latest


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

        self.score_threshold = nn_cfg.get("score_threshold", NN_CONFIG.score_threshold)
        if self.score_threshold is None:
            raise ValueError("score_threshold missing from NN config artifacts")

        self.enable_long = bool(nn_cfg.get("enable_long", NN_CONFIG.enable_long))
        self.enable_short = bool(nn_cfg.get("enable_short", NN_CONFIG.enable_short))
        self.max_trades_per_day = int(nn_cfg.get("max_trades_per_day", NN_CONFIG.max_trades_per_day))
        self.min_bars_between_trades = int(
            nn_cfg.get("min_bars_between_trades", NN_CONFIG.min_bars_between_trades)
        )
        self.budget = DailyTradeBudget(
            max_trades_per_day=self.max_trades_per_day,
            min_bars_between_trades=self.min_bars_between_trades,
            bar_minutes=5,
        )

    def generate_signal(self, bars_df: pd.DataFrame) -> Optional[Signal]:
        if bars_df.empty:
            return None

        bars_df = bars_df.copy()
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

        probs = predict_latest(bars_df, self.bundle)
        if probs is None:
            return None

        last_row = bars_df.iloc[-1]
        ts_local = _to_chicago(last_row["timestamp"])
        if ts_local.time() < RISK_CONFIG.session_start or ts_local.time() >= RISK_CONFIG.session_end:
            return None

        long_prob = probs["long_prob"]
        short_prob = probs["short_prob"]
        score = probs["score"]
        direction = probs["direction"]

        if score < float(self.score_threshold):
            return None

        if direction == "long" and not self.enable_long:
            return None
        if direction == "short" and not self.enable_short:
            return None

        if not self.budget.can_take(last_row["timestamp"]):
            return None

        # Use fixed stop from config to match backtest and model training
        # This ensures live performance matches backtest expectations
        stop_ticks = TRAINING_CONFIG.stop_loss_ticks

        last_close = float(last_row["close"])
        stop_distance = stop_ticks * RISK_CONFIG.tick_size
        target_distance = stop_distance * TRAINING_CONFIG.target_multiplier

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
            "target_rr_multiple": TRAINING_CONFIG.target_multiplier,
            "strategy_mode": "ml_only",
            "long_prob": long_prob,
            "short_prob": short_prob,
            "score": score,
            "score_threshold": float(self.score_threshold),
        }

        reason = (
            f"ML signal {direction.upper()} | score={score:.3f} "
            f"long_prob={long_prob:.3f} short_prob={short_prob:.3f}"
        )

        self.budget.register_trade(last_row["timestamp"])

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
