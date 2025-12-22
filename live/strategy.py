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

from core.selection import DailyTopNSelector, DayAdaptiveTopNSelector, in_session
from core.session_utils import is_entry_feasible
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
        self.selection_mode = str(nn_cfg.get("selection_mode", "global_threshold"))
        self.day_percentile_floor = float(nn_cfg.get("day_percentile_floor", 0.90))
        self.global_floor_score = float(nn_cfg.get("global_floor_score", self.score_threshold))
        self.session_mode = str(nn_cfg.get("session_mode", "RTH"))
        self.deadline_time = nn_cfg.get("deadline_time")
        self.deadline_relax_factor = float(nn_cfg.get("deadline_relax_factor", 0.98))
        self.execution_mode = str(nn_cfg.get("execution_mode", "time_exit"))
        self.exit_price_mode = str(nn_cfg.get("exit_price_mode", "bar_close"))
        self.horizon_bars = int(nn_cfg["horizon_bars"])
        self.bar_minutes = int(nn_cfg.get("bar_minutes", 5))
        self.stop_loss_ticks = int(nn_cfg["stop_loss_ticks"])
        self.target_multiplier = float(nn_cfg["target_multiplier"])
        self.catastrophic_stop_ticks = int(
            nn_cfg.get("catastrophic_stop_ticks", int(nn_cfg["threshold_ticks"]) * 4)
        )
        self.tick_size = float(nn_cfg["tick_size"])
        self.tick_value = float(nn_cfg["tick_value"])

        # Dynamic catastrophic stop parameters
        self.use_dynamic_catastop = bool(nn_cfg.get("use_dynamic_catastop", False))
        self.catastop_atr_multiplier = float(nn_cfg.get("catastop_atr_multiplier", 2.0))
        self.catastop_min_ticks = int(nn_cfg.get("catastop_min_ticks", 24))
        self.catastop_max_ticks = int(nn_cfg.get("catastop_max_ticks", 72))

        if self.selection_mode.lower() == "day_adaptive_topn":
            self.selector = DayAdaptiveTopNSelector(
                max_trades_per_day=self.max_trades_per_day,
                min_bars_between_trades=self.min_bars_between_trades,
                day_percentile_floor=self.day_percentile_floor,
                global_floor_score=self.global_floor_score,
                bar_minutes=self.bar_minutes,
                session_mode=self.session_mode,
                session_start=self.session_start,
                session_end=self.session_end,
                deadline_time=self.deadline_time or pd.Timestamp("11:30").time(),
                deadline_relax_factor=self.deadline_relax_factor,
                confidence_min=float(nn_cfg.get("confidence_min", 0.05)),
                quality_margin=float(nn_cfg.get("quality_margin", 0.01)),
                allow_extra_trades_if_quality=bool(nn_cfg.get("allow_extra_trades_if_quality", True)),
                daily_stop_loss=float(nn_cfg.get("daily_stop_loss", 400.0)),
                daily_profit_lock=float(nn_cfg.get("daily_profit_lock", 500.0)),
            )
        else:
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
        self.pending_entry: Optional[Dict[str, object]] = None

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
            direction = str(self.open_position.get("direction", "long"))
            stop_price = float(self.open_position.get("stop_price", 0.0))
            close_price = float(last_row["close"])

            cat_trigger = False
            if direction == "long" and close_price <= stop_price:
                cat_trigger = True
            elif direction == "short" and close_price >= stop_price:
                cat_trigger = True

            if cat_trigger:
                reason = "CATASTOP close"
                exit_price = close_price
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

            if hold_bars >= self.horizon_bars:
                reason = f"TIME_EXIT after {hold_bars} bars"
                if self.exit_price_mode == "bar_close":
                    exit_price = float(last_row["close"])
                else:
                    exit_price = float(last_row["open"])
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

        if self.pending_entry is not None:
            if not in_session(
                last_row["timestamp"],
                session_mode=self.session_mode,
                session_start=self.session_start,
                session_end=self.session_end,
            ):
                LOGGER.warning("Dropping pending entry outside session.")
                self.pending_entry = None
                return None
            entry_idx = int(self.pending_entry["entry_idx"])
            if bar_index >= entry_idx:
                direction = str(self.pending_entry["direction"])
                long_prob = float(self.pending_entry["long_prob"])
                short_prob = float(self.pending_entry["short_prob"])
                score = float(self.pending_entry["score"])

                entry_price = float(last_row["open"])
                if self.execution_mode == "time_exit":
                    # Compute dynamic catastrophic stop if enabled
                    if self.use_dynamic_catastop:
                        # Get ATR from previous bar (causal)
                        from features.engineer import add_features
                        bars_with_features = add_features(bars_df, verbose=False)
                        if len(bars_with_features) >= 2:
                            signal_bar = bars_with_features.iloc[-2]  # Previous bar (signal bar)
                            atr_ticks = signal_bar.get("atr_ticks", 0)
                            if pd.isna(atr_ticks) or atr_ticks <= 0:
                                stop_ticks = self.catastrophic_stop_ticks
                            else:
                                # Dynamic stop: clamp(atr * multiplier, min, max)
                                raw_stop = int(atr_ticks * self.catastop_atr_multiplier)
                                stop_ticks = max(self.catastop_min_ticks, min(self.catastop_max_ticks, raw_stop))
                        else:
                            stop_ticks = self.catastrophic_stop_ticks
                    else:
                        stop_ticks = self.catastrophic_stop_ticks
                else:
                    stop_ticks = self.stop_loss_ticks
                stop_distance = stop_ticks * self.tick_size
                target_distance = stop_distance * self.target_multiplier

                if direction == "long":
                    action = SignalAction.ENTER_LONG
                    stop_price = entry_price - stop_distance
                    target_price = entry_price + target_distance
                else:
                    action = SignalAction.ENTER_SHORT
                    stop_price = entry_price + stop_distance
                    target_price = entry_price - target_distance

                metadata = {
                    "ideal_entry": entry_price,
                    "risk_ticks": stop_ticks,
                    "target_rr_multiple": self.target_multiplier,
                    "strategy_mode": "ml_only",
                    "long_prob": long_prob,
                    "short_prob": short_prob,
                    "score": score,
                    "score_threshold": float(self.score_threshold),
                    "execution_mode": self.execution_mode,
                    "exit_price_mode": self.exit_price_mode,
                }
                reason = (
                    f"ML entry {direction.upper()} | score={score:.3f} "
                    f"long_prob={long_prob:.3f} short_prob={short_prob:.3f}"
                )
                self.selector.register_trade(last_row["timestamp"], bar_index=bar_index)
                self.open_position = {
                    "entry_time": last_row["timestamp"],
                    "entry_idx": bar_index,
                    "direction": direction,
                    "stop_price": stop_price,
                }
                self.pending_entry = None
                return Signal(
                    action=action,
                    symbol=self.symbol,
                    timestamp=last_row["timestamp"].to_pydatetime(),
                    stop_price=stop_price,
                    target_price=target_price,
                    reason=reason,
                    metadata=metadata,
                )
            return None

        if not in_session(
            last_row["timestamp"],
            session_mode=self.session_mode,
            session_start=self.session_start,
            session_end=self.session_end,
        ):
            return None

        # Feasibility check: ensure enough bars left in RTH for time-exit
        if not is_entry_feasible(
            last_row["timestamp"],
            horizon_bars=self.horizon_bars,
            bar_minutes=self.bar_minutes,
            rth_end_time=self.session_end,
            execution_mode=self.execution_mode,
            session_mode=self.session_mode,
        ):
            LOGGER.info(
                "Signal rejected by feasibility: time=%s horizon=%s bars_left insufficient",
                last_row["timestamp"],
                self.horizon_bars,
            )
            return None

        long_prob = probs["long_prob"]
        short_prob = probs["short_prob"]
        score = probs["score"]
        direction = probs["direction"]
        confidence = probs.get("confidence")  # New: get confidence for quality gates

        if direction == "long" and not self.enable_long:
            return None
        if direction == "short" and not self.enable_short:
            return None

        # Pass confidence to selector for quality gates on trades 3-4
        if not self.selector.should_enter(
            last_row["timestamp"],
            score=float(score),
            direction=str(direction),
            bar_index=bar_index,
            confidence=float(confidence) if confidence is not None else None,
        ):
            self.selector.log_rejection()
            return None

        self.pending_entry = {
            "signal_time": last_row["timestamp"],
            "signal_idx": bar_index,
            "entry_idx": bar_index + 1,
            "direction": direction,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "score": score,
        }

        reason = (
            f"ML signal queued for next bar {direction.upper()} | score={score:.3f} "
            f"long_prob={long_prob:.3f} short_prob={short_prob:.3f}"
        )
        LOGGER.info(reason)
        return None

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
