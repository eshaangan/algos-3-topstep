"""
ML strategy wrapper that converts model probabilities into trade signals.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from features.engineer import add_features, select_features


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


class MLStrategy:
    """Loads trained models and emits trade signals based on probabilities."""

    def __init__(self, model_dir: str = "models/saved", symbol: str = "MES") -> None:
        self.model_dir = Path(model_dir)
        self.symbol = symbol
        self.long_model = joblib.load(self.model_dir / "model_long.joblib")
        self.short_model = joblib.load(self.model_dir / "model_short.joblib")
        self.metadata = self._load_metadata()
        self.feature_cols = self.metadata.get("feature_cols") or select_features()

    def _load_metadata(self) -> Dict[str, object]:
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_signal(self, bars_df: pd.DataFrame) -> Optional[Signal]:
        if bars_df.empty:
            return None

        bars_df = bars_df.copy()
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

        features_df = add_features(bars_df, verbose=False)
        last_row = features_df.iloc[-1]

        if last_row[self.feature_cols].isna().any():
            return None

        ts_local = _to_chicago(last_row["timestamp"])
        if ts_local.time() < RISK_CONFIG.session_start or ts_local.time() >= RISK_CONFIG.session_end:
            return None

        X = last_row[self.feature_cols].values.reshape(1, -1)
        long_prob = float(self.long_model.predict_proba(X)[:, 1][0])
        short_prob = float(self.short_model.predict_proba(X)[:, 1][0])

        trade_affordable = int(last_row.get("trade_affordable", 1))
        if trade_affordable < 1:
            return None

        long_ok = long_prob >= TRAINING_CONFIG.min_probability_long
        short_ok = short_prob >= TRAINING_CONFIG.min_probability_short
        if not long_ok and not short_ok:
            return None

        if long_ok and short_ok:
            direction = "long" if long_prob >= short_prob else "short"
        else:
            direction = "long" if long_ok else "short"

        stop_ticks = last_row.get("estimated_stop_ticks", TRAINING_CONFIG.stop_loss_ticks)
        if pd.isna(stop_ticks):
            stop_ticks = TRAINING_CONFIG.stop_loss_ticks
        stop_ticks = int(round(stop_ticks))
        stop_ticks = max(6, min(stop_ticks, 30))

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
            "atr_ticks": float(last_row.get("atr_ticks", np.nan))
            if not pd.isna(last_row.get("atr_ticks", np.nan))
            else None,
            "target_rr_multiple": TRAINING_CONFIG.target_multiplier,
            "strategy_mode": "ml_only",
            "long_prob": long_prob,
            "short_prob": short_prob,
        }

        reason = (
            f"ML signal {direction.upper()} | long_prob={long_prob:.3f} "
            f"short_prob={short_prob:.3f}"
        )

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
            },
        }
