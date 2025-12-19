"""
Run a walk-forward backtest using trained ML models and TopstepX risk rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from features.engineer import add_features, select_features


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def load_models(model_dir: str) -> Tuple[object, object, Dict[str, object]]:
    model_path = Path(model_dir)
    long_model = joblib.load(model_path / "model_long.joblib")
    short_model = joblib.load(model_path / "model_short.joblib")

    metadata_path = model_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return long_model, short_model, metadata


def load_bars(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    return bars


def _compute_probabilities(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
) -> pd.DataFrame:
    features_df = add_features(bars_df)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})

    valid_mask = features_df[feature_cols].notna().all(axis=1)
    valid_idx = features_df.loc[valid_mask, "idx"].values

    X = features_df.loc[valid_mask, feature_cols].values
    long_prob = long_model.predict_proba(X)[:, 1]
    short_prob = short_model.predict_proba(X)[:, 1]

    prob_df = pd.DataFrame(
        {
            "idx": valid_idx,
            "long_prob": long_prob,
            "short_prob": short_prob,
        }
    )

    full = pd.DataFrame({"idx": features_df["idx"].values})
    full = full.merge(prob_df, on="idx", how="left")
    full["estimated_stop_ticks"] = features_df["estimated_stop_ticks"].values
    full["trade_affordable"] = features_df["trade_affordable"].values
    full["max_contracts_by_risk"] = features_df["max_contracts_by_risk"].values
    return full


def _position_pnl(entry_price: float, exit_price: float, direction: str, contracts: int) -> float:
    ticks = (exit_price - entry_price) / RISK_CONFIG.tick_size
    if direction == "short":
        ticks *= -1
    return ticks * RISK_CONFIG.tick_value * contracts


def run_backtest(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
    save_trades_path: Optional[str] = None,
) -> Dict[str, object]:
    prob_df = _compute_probabilities(bars_df, long_model, short_model, feature_cols)

    trades: List[Dict[str, object]] = []
    equity_curve: List[float] = [RISK_CONFIG.starting_balance]
    equity = RISK_CONFIG.starting_balance
    trailing_peak = equity
    daily_pnl = 0.0
    daily_locked = False
    trailing_locked = False
    current_day = None

    position = None
    slippage_ticks = 1

    for i in range(0, len(bars_df) - 1):
        bar = bars_df.iloc[i]
        bar_time = _to_chicago(bar["timestamp"])
        bar_day = bar_time.date()

        if current_day != bar_day:
            current_day = bar_day
            daily_pnl = 0.0
            daily_locked = False

        if position is not None:
            hold_bars = i - position["entry_idx"]
            exit_reason = None
            exit_price = None

            if bar_time.time() >= RISK_CONFIG.session_end:
                exit_reason = "SESSION_FLAT"
                exit_price = bar["close"]
            elif hold_bars >= TRAINING_CONFIG.max_hold_bars:
                exit_reason = "MAX_HOLD"
                exit_price = bar["close"]
            else:
                if position["direction"] == "long":
                    if bar["low"] <= position["stop_price"]:
                        exit_reason = "STOP"
                        exit_price = position["stop_price"]
                    elif bar["high"] >= position["target_price"]:
                        exit_reason = "TARGET"
                        exit_price = position["target_price"]
                else:
                    if bar["high"] >= position["stop_price"]:
                        exit_reason = "STOP"
                        exit_price = position["stop_price"]
                    elif bar["low"] <= position["target_price"]:
                        exit_reason = "TARGET"
                        exit_price = position["target_price"]

            if exit_reason is not None:
                raw_pnl = _position_pnl(
                    position["entry_price"],
                    exit_price,
                    position["direction"],
                    position["contracts"],
                )
                fees = 2 * position["contracts"] * 2.35 + (
                    slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
                )
                pnl = raw_pnl - fees

                equity += pnl
                daily_pnl += pnl
                trailing_peak = max(trailing_peak, equity)
                equity_curve.append(equity)

                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": bar["timestamp"],
                        "direction": position["direction"],
                        "contracts": position["contracts"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "stop_ticks": position["stop_ticks"],
                    }
                )

                position = None

                if daily_pnl <= -RISK_CONFIG.max_daily_loss:
                    daily_locked = True
                if trailing_peak - equity >= RISK_CONFIG.trailing_drawdown:
                    trailing_locked = True

        if position is not None:
            continue

        if trailing_locked or daily_locked:
            continue

        if bar_time.time() < RISK_CONFIG.session_start or bar_time.time() >= RISK_CONFIG.session_end:
            continue

        probs = prob_df.iloc[i]
        long_prob = probs["long_prob"]
        short_prob = probs["short_prob"]
        if pd.isna(long_prob) or pd.isna(short_prob):
            continue

        if probs.get("trade_affordable", 1) < 1:
            continue

        long_ok = long_prob >= TRAINING_CONFIG.min_probability_long
        short_ok = short_prob >= TRAINING_CONFIG.min_probability_short
        if not long_ok and not short_ok:
            continue

        if long_ok and short_ok:
            direction = "long" if long_prob >= short_prob else "short"
        else:
            direction = "long" if long_ok else "short"

        stop_ticks = probs.get("estimated_stop_ticks", TRAINING_CONFIG.stop_loss_ticks)
        if pd.isna(stop_ticks):
            stop_ticks = TRAINING_CONFIG.stop_loss_ticks
        stop_ticks = int(round(stop_ticks))
        stop_ticks = max(6, min(stop_ticks, 30))

        risk_per_contract = stop_ticks * RISK_CONFIG.tick_value
        contracts = int(RISK_CONFIG.fixed_risk_per_trade / max(risk_per_contract, 1e-6))
        contracts = max(0, min(contracts, RISK_CONFIG.max_contracts))
        max_by_feature = probs.get("max_contracts_by_risk", contracts)
        if not pd.isna(max_by_feature):
            contracts = min(contracts, int(max_by_feature))

        if contracts < 1:
            continue

        next_bar = bars_df.iloc[i + 1]
        entry_price = next_bar["open"]
        slippage = slippage_ticks * RISK_CONFIG.tick_size
        if direction == "long":
            entry_price += slippage
        else:
            entry_price -= slippage

        stop_distance = stop_ticks * RISK_CONFIG.tick_size
        target_distance = stop_distance * TRAINING_CONFIG.target_multiplier

        if direction == "long":
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

        position = {
            "direction": direction,
            "contracts": contracts,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_time": next_bar["timestamp"],
            "entry_idx": i + 1,
            "stop_ticks": stop_ticks,
        }

    if position is not None:
        last_bar = bars_df.iloc[-1]
        exit_price = last_bar["close"]
        raw_pnl = _position_pnl(
            position["entry_price"],
            exit_price,
            position["direction"],
            position["contracts"],
        )
        fees = 2 * position["contracts"] * 2.35 + (
            slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
        )
        pnl = raw_pnl - fees
        equity += pnl
        equity_curve.append(equity)
        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": last_bar["timestamp"],
                "direction": position["direction"],
                "contracts": position["contracts"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": "FINAL_BAR",
                "stop_ticks": position["stop_ticks"],
            }
        )

    pnl_values = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]

    gross_win = wins.sum() if wins.size else 0.0
    gross_loss = abs(losses.sum()) if losses.size else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    win_rate = float((pnl_values > 0).mean()) if pnl_values.size else 0.0
    net_pnl = float(pnl_values.sum()) if pnl_values.size else 0.0

    equity_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_arr)
    drawdowns = peaks - equity_arr
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0

    summary = {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "max_drawdown": max_drawdown,
        "ending_equity": equity,
        "starting_balance": RISK_CONFIG.starting_balance,
        "config": {
            "risk": asdict(RISK_CONFIG),
            "training": asdict(TRAINING_CONFIG),
        },
    }

    if save_trades_path:
        trades_df = pd.DataFrame(trades)
        Path(save_trades_path).parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(save_trades_path, index=False)

    return {"summary": summary, "trades": trades}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML strategy")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--model-dir", default="models/saved")
    parser.add_argument("--save-trades", default=None)
    args = parser.parse_args()

    bars = load_bars(args.data_path)
    long_model, short_model, metadata = load_models(args.model_dir)
    feature_cols = metadata.get("feature_cols") or select_features()

    results = run_backtest(bars, long_model, short_model, feature_cols, args.save_trades)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
