"""
V2 Backtest: Pure ML strategy with score-based ranking and daily trade budget.

Key differences from backtest.py:
- NO heuristic gates (trend/hour/lunch/ATR filters)
- Entry decisions based ONLY on ML score + trade budget + risk checks
- Score = max(p_long, p_short)
- Daily trade budget enforces ~1-2 trades/day
- Minimum bars between trades prevents rapid-fire entries
- Execution (stop/target/max-hold) remains deterministic and separate
"""

from __future__ import annotations

import argparse
import json
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp to Chicago timezone."""
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def compute_probabilities_v2(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-bar probabilities and scores.

    Returns:
        DataFrame with idx, long_prob, short_prob, score, direction
    """
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})

    # Get valid rows (no NaN in features)
    valid_mask = features_df[feature_cols].notna().all(axis=1)
    valid_idx = features_df.loc[valid_mask, "idx"].values
    X = features_df.loc[valid_mask, feature_cols].values

    # Predict probabilities
    long_prob = long_model.predict_proba(X)[:, 1]
    if short_model is not None:
        short_prob = short_model.predict_proba(X)[:, 1]
    else:
        short_prob = np.zeros_like(long_prob)

    # Compute score and direction
    # Score = max(p_long, p_short)
    # Direction = argmax
    score = np.maximum(long_prob, short_prob)
    direction = np.where(long_prob >= short_prob, "long", "short")

    prob_df = pd.DataFrame(
        {
            "idx": valid_idx,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "score": score,
            "direction": direction,
        }
    )

    # Merge with full index
    full = pd.DataFrame({"idx": features_df["idx"].values})
    full = full.merge(prob_df, on="idx", how="left")

    return full


def run_backtest_v2(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
    *,
    score_threshold: float,
    max_trades_per_day: int = 2,
    min_bars_between_trades: int = 12,
    enable_long: bool = True,
    enable_short: bool = False,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    stop_loss_ticks: Optional[int] = None,
    target_multiplier: Optional[float] = None,
    max_hold_bars: Optional[int] = None,
    slippage_ticks: int = 1,
    commission_per_contract: float = 2.35,
    save_trades_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run backtest with V2 strategy: pure ML score + daily trade budget.

    Entry Logic (NO HEURISTIC GATES):
        1. Check RTH session (8:30 AM - 2:55 PM CT)
        2. Check risk manager (daily loss, trailing DD)
        3. Check no existing position
        4. Check score >= score_threshold
        5. Check direction enabled (long/short)
        6. Check daily trade budget not exceeded
        7. Check min bars since last trade
        8. → ENTER if all checks pass

    Exit Logic (DETERMINISTIC):
        - Stop loss hit
        - Target hit
        - Max hold bars reached
        - Session end (flatten before 2:55 PM CT)

    Args:
        bars_df: OHLCV bars DataFrame
        long_model: Trained long model
        short_model: Trained short model (can be None if shorts disabled)
        feature_cols: List of feature column names
        score_threshold: Minimum score to take trade (fitted from train quantile)
        max_trades_per_day: Hard daily trade limit (default 2)
        min_bars_between_trades: Minimum bars between entries (default 12 = 60 min)
        enable_long: Allow long trades
        enable_short: Allow short trades
        start_idx: Backtest window start
        end_idx: Backtest window end
        stop_loss_ticks: Stop loss size in ticks (default from config)
        target_multiplier: Target as multiple of stop (default from config)
        max_hold_bars: Max bars to hold (default from config)
        slippage_ticks: Slippage per entry/exit (default 1)
        commission_per_contract: Commission per side (default $2.35)
        save_trades_path: Optional CSV path to save trade log

    Returns:
        {
            "summary": {trade count, win rate, profit factor, net P&L, etc.},
            "trades": [list of trade dicts],
            "equity_curve": [list of equity values],
            "daily_stats": {trades per day, etc.},
        }
    """
    # Default parameters from config
    stop_loss_ticks = stop_loss_ticks or TRAINING_CONFIG.stop_loss_ticks
    target_multiplier = target_multiplier or TRAINING_CONFIG.target_multiplier
    max_hold_bars = max_hold_bars or TRAINING_CONFIG.max_hold_bars

    # Backtest window
    start = start_idx or 0
    end = end_idx or len(bars_df)

    if start < 0 or end > len(bars_df) or start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}, len={len(bars_df)}")

    # Compute probabilities and scores
    prob_df = compute_probabilities_v2(bars_df, long_model, short_model, feature_cols)

    # State
    trades: List[Dict[str, object]] = []
    equity_curve: List[float] = [RISK_CONFIG.starting_balance]
    equity = RISK_CONFIG.starting_balance
    trailing_peak = equity
    daily_pnl = 0.0
    daily_locked = False
    trailing_locked = False
    current_day: Optional[date] = None
    daily_trade_count = 0
    last_entry_idx = -999999  # Initialize far in past

    position: Optional[Dict[str, object]] = None

    # Daily tracking
    daily_trades_list: List[int] = []

    # Iterate through bars
    for i in range(start, end - 1):
        bar = bars_df.iloc[i]
        bar_time = _to_chicago(bar["timestamp"])
        bar_day = bar_time.date()

        # Reset daily state on new day
        if current_day != bar_day:
            if current_day is not None:
                daily_trades_list.append(daily_trade_count)
            current_day = bar_day
            daily_pnl = 0.0
            daily_locked = False
            daily_trade_count = 0

        # ========== EXIT LOGIC (if in position) ==========
        if position is not None:
            hold_bars = i - position["entry_idx"]
            exit_reason = None
            exit_price = None

            # Check exit conditions
            # 1. Session end (flatten before close)
            if bar_time.time() >= RISK_CONFIG.session_end:
                exit_reason = "SESSION_FLAT"
                exit_price = bar["close"]

            # 2. Max hold reached
            elif hold_bars >= max_hold_bars:
                exit_reason = "MAX_HOLD"
                exit_price = bar["close"]

            # 3. Stop/Target hit (check bar high/low)
            else:
                if position["direction"] == "long":
                    if bar["low"] <= position["stop_price"]:
                        exit_reason = "STOP"
                        exit_price = position["stop_price"]
                    elif bar["high"] >= position["target_price"]:
                        exit_reason = "TARGET"
                        exit_price = position["target_price"]
                else:  # short
                    if bar["high"] >= position["stop_price"]:
                        exit_reason = "STOP"
                        exit_price = position["stop_price"]
                    elif bar["low"] <= position["target_price"]:
                        exit_reason = "TARGET"
                        exit_price = position["target_price"]

            # Execute exit
            if exit_reason is not None:
                # Calculate P&L
                ticks = (exit_price - position["entry_price"]) / RISK_CONFIG.tick_size
                if position["direction"] == "short":
                    ticks *= -1
                raw_pnl = ticks * RISK_CONFIG.tick_value * position["contracts"]

                # Fees: commission (2 sides) + slippage
                fees = (
                    2 * position["contracts"] * commission_per_contract
                    + slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
                )
                pnl = raw_pnl - fees

                # Update equity
                equity += pnl
                daily_pnl += pnl
                trailing_peak = max(trailing_peak, equity)
                equity_curve.append(equity)

                # Record trade
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
                        "score": position["score"],
                    }
                )

                # Clear position
                position = None

                # Check risk limits
                if daily_pnl <= -RISK_CONFIG.max_daily_loss:
                    daily_locked = True
                if trailing_peak - equity >= RISK_CONFIG.trailing_drawdown:
                    trailing_locked = True

        # ========== ENTRY LOGIC (if no position) ==========
        if position is not None:
            continue  # Already in position

        # Risk checks
        if daily_locked or trailing_locked:
            continue  # Risk limit hit

        # Session check
        if bar_time.time() < RISK_CONFIG.session_start or bar_time.time() >= RISK_CONFIG.session_end:
            continue  # Outside regular trading hours

        # Get ML signal
        probs = prob_df.iloc[i]
        long_prob = probs.get("long_prob")
        short_prob = probs.get("short_prob")
        score = probs.get("score")
        direction = probs.get("direction")

        # Check for valid signal
        if pd.isna(score) or pd.isna(long_prob) or pd.isna(short_prob):
            continue  # No valid ML signal

        # ========== PURE ML ENTRY CHECKS (NO HEURISTIC GATES) ==========

        # 1. Score threshold check
        if score < score_threshold:
            continue  # Score too low

        # 2. Direction enabled check
        if direction == "long" and not enable_long:
            continue
        if direction == "short" and not enable_short:
            continue

        # 3. Daily trade budget check
        if daily_trade_count >= max_trades_per_day:
            continue  # Already took max trades today

        # 4. Minimum bars between trades check (prevent rapid-fire)
        bars_since_last = i - last_entry_idx
        if bars_since_last < min_bars_between_trades:
            continue  # Too soon after last trade

        # ========== ALL CHECKS PASSED → ENTER TRADE ==========

        # Entry will be next bar open (avoid lookahead bias)
        next_bar = bars_df.iloc[i + 1]
        next_bar_time = _to_chicago(next_bar["timestamp"])

        # Skip if next bar is outside session
        if next_bar_time.time() >= RISK_CONFIG.session_end:
            continue

        # Entry price with slippage
        if direction == "long":
            entry_price = next_bar["open"] + slippage_ticks * RISK_CONFIG.tick_size
        else:
            entry_price = next_bar["open"] - slippage_ticks * RISK_CONFIG.tick_size

        # Position sizing (simple: 1 contract for now)
        # TODO: Could add dynamic sizing based on fixed_risk_per_trade
        contracts = 1

        # Stop and target prices
        stop_distance = stop_loss_ticks * RISK_CONFIG.tick_size
        target_distance = stop_distance * target_multiplier

        if direction == "long":
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

        # Create position
        position = {
            "entry_idx": i + 1,  # Next bar index
            "entry_time": next_bar["timestamp"],
            "direction": direction,
            "contracts": contracts,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "stop_ticks": stop_loss_ticks,
            "score": score,
        }

        # Update state
        daily_trade_count += 1
        last_entry_idx = i + 1

    # End of backtest - close any remaining position
    if position is not None:
        final_bar = bars_df.iloc[end - 1]
        exit_price = final_bar["close"]

        ticks = (exit_price - position["entry_price"]) / RISK_CONFIG.tick_size
        if position["direction"] == "short":
            ticks *= -1
        raw_pnl = ticks * RISK_CONFIG.tick_value * position["contracts"]

        fees = (
            2 * position["contracts"] * commission_per_contract
            + slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
        )
        pnl = raw_pnl - fees

        equity += pnl
        daily_pnl += pnl
        equity_curve.append(equity)

        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": final_bar["timestamp"],
                "direction": position["direction"],
                "contracts": position["contracts"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": "END_OF_BACKTEST",
                "stop_ticks": position["stop_ticks"],
                "score": position["score"],
            }
        )

    # Add final day's trade count
    if current_day is not None:
        daily_trades_list.append(daily_trade_count)

    # ========== COMPUTE SUMMARY STATISTICS ==========
    trades_df = pd.DataFrame(trades)

    if len(trades) == 0:
        summary = {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": equity,
            "starting_balance": RISK_CONFIG.starting_balance,
        }
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0
        gross_wins = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Drawdown
        equity_array = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity_array)
        drawdowns = peaks - equity_array
        max_drawdown = float(np.max(drawdowns))

        summary = {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "gross_wins": float(gross_wins),
            "gross_losses": float(gross_losses),
            "net_pnl": float(trades_df["pnl"].sum()),
            "avg_pnl": float(trades_df["pnl"].mean()),
            "max_drawdown": max_drawdown,
            "ending_equity": float(equity),
            "starting_balance": RISK_CONFIG.starting_balance,
        }

    # Daily stats
    daily_stats = {
        "total_trading_days": len(daily_trades_list),
        "avg_trades_per_day": float(np.mean(daily_trades_list)) if daily_trades_list else 0.0,
        "max_trades_in_day": int(np.max(daily_trades_list)) if daily_trades_list else 0,
        "days_with_trades": int(np.sum(np.array(daily_trades_list) > 0)),
        "days_with_zero_trades": int(np.sum(np.array(daily_trades_list) == 0)),
    }

    # Save trades if requested
    if save_trades_path and len(trades) > 0:
        trades_df.to_csv(save_trades_path, index=False)
        print(f"\n✓ Saved {len(trades)} trades to {save_trades_path}")

    return {
        "summary": summary,
        "trades": trades,
        "equity_curve": equity_curve,
        "daily_stats": daily_stats,
    }


def load_models_v2(model_dir: str) -> Tuple[object, object, Dict[str, object]]:
    """Load V2 models and metadata."""
    model_path = Path(model_dir)
    long_model = joblib.load(model_path / "model_long.joblib")

    # Short model may not exist if shorts disabled
    short_model_path = model_path / "model_short.joblib"
    if short_model_path.exists():
        short_model = joblib.load(short_model_path)
    else:
        short_model = None

    # Load metadata
    metadata_path = model_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return long_model, short_model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Backtest: Pure ML with score + trade budget")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--model-dir", default="models/saved_v2")
    parser.add_argument("--save-trades", default=None, help="CSV path to save trades")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    # Load models
    long_model, short_model, metadata = load_models_v2(args.model_dir)
    feature_cols = metadata.get("feature_cols", [])

    # Get policy V2 params
    policy_v2 = metadata.get("policy_v2", {})
    score_threshold = policy_v2.get("score_threshold", 0.95)
    max_trades_per_day = policy_v2.get("max_trades_per_day", 2)
    min_bars_between_trades = policy_v2.get("min_bars_between_trades", 12)
    enable_long = policy_v2.get("enable_long", True)
    enable_short = policy_v2.get("enable_short", False)

    print("\n" + "=" * 60)
    print("V2 BACKTEST: PURE ML STRATEGY")
    print("=" * 60)
    print(f"\nModel: {args.model_dir}")
    print(f"Data: {args.data_path}")
    print(f"\nPolicy V2:")
    print(f"  Score threshold: {score_threshold:.4f}")
    print(f"  Max trades/day: {max_trades_per_day}")
    print(f"  Min bars between: {min_bars_between_trades}")
    print(f"  Long enabled: {enable_long}")
    print(f"  Short enabled: {enable_short}")

    # Load data
    with pd.HDFStore(args.data_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars, _ = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=False)

    print(f"\nBars: {len(bars):,}")

    # Run backtest
    results = run_backtest_v2(
        bars,
        long_model,
        short_model,
        feature_cols,
        score_threshold=score_threshold,
        max_trades_per_day=max_trades_per_day,
        min_bars_between_trades=min_bars_between_trades,
        enable_long=enable_long,
        enable_short=enable_short,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_trades_path=args.save_trades,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print("\nSummary:")
    print(json.dumps(results["summary"], indent=2))
    print("\nDaily Stats:")
    print(json.dumps(results["daily_stats"], indent=2))


if __name__ == "__main__":
    main()
