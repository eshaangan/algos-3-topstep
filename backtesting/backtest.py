"""
Run a walk-forward backtest using trained ML models and TopstepX risk rules.
"""

from __future__ import annotations

import argparse
import logging
import json
from dataclasses import asdict
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.risk_presets import RISK_PRESET_NAME, get_risk_config
from core.selection import DailyTopNSelector, DayAdaptiveTopNSelector, in_session
from core.session_utils import is_entry_feasible
from core.simple_config import TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, select_features
from models.nn_inference import load_nn_bundle, predict_scores_for_bars

RISK_CONFIG = get_risk_config(RISK_PRESET_NAME)


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def compute_probabilities(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-bar long/short probabilities and attach a small set of features
    used for optional trade gating.
    """
    # Backtest gets called in loops (threshold tuning, split eval); keep it quiet by default.
    features_df = add_features(bars_df, verbose=False)
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

    # Gating helpers (all derived from non-lookahead features).
    for col in [
        "estimated_stop_ticks",
        "trade_affordable",
        "max_contracts_by_risk",
        "ema_spread_21_50",
        "price_above_ema50",
        "atr_ticks",
        "lunch_period",
    ]:
        if col in features_df.columns:
            full[col] = features_df[col].values
        else:
            full[col] = np.nan

    return full


def _is_nn_model_dir(model_dir: str) -> bool:
    base = Path(model_dir)
    candidates = [base / "config.json", base / "fold_0" / "config.json"]
    for cfg in candidates:
        if cfg.exists():
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload.get("model_type") == "tiny_mlp"
            except Exception:
                continue
    return False


def select_top_n_indices_per_day(
    bars_df: pd.DataFrame,
    prob_df: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
    score_threshold: float,
    max_trades_per_day: int,
    min_bars_between_trades: int,
) -> set[int]:
    """
    Select top-N scored bars per day, spaced by min_bars_between_trades.
    """
    ts = pd.to_datetime(bars_df["timestamp"], utc=True)
    bar_day = ts.dt.tz_convert("America/Chicago").dt.date

    day_by_idx = pd.Series(bar_day.values, index=np.arange(len(bars_df)))

    subset = prob_df.copy()
    subset = subset[(subset["idx"] >= start_idx) & (subset["idx"] < end_idx)]
    subset = subset[subset["score"].notna() & (subset["score"] >= score_threshold)]
    subset["bar_day"] = subset["idx"].map(day_by_idx)

    selected: set[int] = set()
    for day, group in subset.groupby("bar_day"):
        group_sorted = group.sort_values("score", ascending=False)
        picks: List[int] = []
        for idx in group_sorted["idx"].astype(int).tolist():
            if len(picks) >= max_trades_per_day:
                break
            if all(abs(idx - chosen) >= min_bars_between_trades for chosen in picks):
                picks.append(idx)
        selected.update(picks)

    return selected


def run_backtest_nn(
    bars_df: pd.DataFrame,
    prob_df: pd.DataFrame,
    *,
    score_threshold: float,
    selection_mode: str = "global_threshold",
    day_percentile_floor: float = 0.90,
    global_floor_score: Optional[float] = None,
    max_trades_per_day: int,
    min_bars_between_trades: int,
    enable_long: bool,
    enable_short: bool,
    horizon_bars: int,
    execution_mode: str = "time_exit",
    exit_price_mode: str = "bar_close",
    session_mode: str = "RTH",
    deadline_time: Optional[time] = None,
    deadline_relax_factor: float = 0.98,
    bar_minutes: int = 5,
    session_start: Optional[time] = None,
    session_end: Optional[time] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    stop_loss_ticks: Optional[int] = None,
    target_multiplier: Optional[float] = None,
    catastrophic_stop_ticks: Optional[int] = None,
    max_hold_bars: Optional[int] = None,
    slippage_ticks: int = 1,
    commission_per_contract: float = 2.35,
    save_trades_path: Optional[str] = None,
    tick_size: Optional[float] = None,
    tick_value: Optional[float] = None,
    use_dynamic_catastop: bool = False,
    catastop_atr_multiplier: float = 2.0,
    catastop_min_ticks: int = 24,
    catastop_max_ticks: int = 72,
) -> Dict[str, object]:
    """
    Pure ML backtest using either global threshold or day-adaptive top-N selection.
    """
    logger = logging.getLogger(__name__)
    stop_loss_ticks = stop_loss_ticks or TRAINING_CONFIG.stop_loss_ticks
    target_multiplier = target_multiplier or TRAINING_CONFIG.target_multiplier
    max_hold_bars = max_hold_bars or horizon_bars
    tick_size = tick_size or RISK_CONFIG.tick_size
    tick_value = tick_value or RISK_CONFIG.tick_value
    session_start = session_start or RISK_CONFIG.session_start
    session_end = session_end or RISK_CONFIG.session_end
    if catastrophic_stop_ticks is None:
        catastrophic_stop_ticks = int(TRAINING_CONFIG.threshold_ticks) * 4

    # Add features to bars if using dynamic catastop (need ATR)
    if use_dynamic_catastop:
        from features.engineer import add_features
        bars_with_features = add_features(bars_df, verbose=False)
    else:
        bars_with_features = None

    if execution_mode == "time_exit":
        if exit_price_mode != "bar_close":
            raise ValueError("Aligned time-exit requires exit_price_mode='bar_close'.")
        if max_hold_bars != horizon_bars:
            raise ValueError("Aligned time-exit requires max_hold_bars == horizon_bars.")

    start = 0 if start_idx is None else int(start_idx)
    end = len(bars_df) if end_idx is None else int(end_idx)
    if start < 0 or end > len(bars_df) or start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}, len={len(bars_df)}")

    selection_mode = str(selection_mode).lower()
    if global_floor_score is None:
        global_floor_score = float(score_threshold)
    if selection_mode == "day_adaptive_topn":
        selector = DayAdaptiveTopNSelector(
            max_trades_per_day=max_trades_per_day,
            min_bars_between_trades=min_bars_between_trades,
            day_percentile_floor=day_percentile_floor,
            global_floor_score=float(global_floor_score),
            bar_minutes=bar_minutes,
            session_mode=session_mode,
            session_start=session_start,
            session_end=session_end,
            deadline_time=deadline_time or time(11, 30),
            deadline_relax_factor=deadline_relax_factor,
            confidence_min=0.05,  # Default, can be overridden by config
            quality_margin=0.01,  # Default, can be overridden by config
            allow_extra_trades_if_quality=True,
            daily_stop_loss=400.0,  # Default, can be overridden by config
            daily_profit_lock=500.0,  # Default, can be overridden by config
        )
    else:
        selector = DailyTopNSelector(
            max_trades_per_day=max_trades_per_day,
            min_bars_between_trades=min_bars_between_trades,
            score_threshold=score_threshold,
            bar_minutes=bar_minutes,
            session_mode=session_mode,
            session_start=session_start,
            session_end=session_end,
            deadline_time=deadline_time or time(11, 30),
            deadline_relax_factor=deadline_relax_factor,
        )

    trades: List[Dict[str, object]] = []
    equity_curve: List[float] = [RISK_CONFIG.starting_balance]
    equity = RISK_CONFIG.starting_balance
    trailing_peak = equity
    daily_pnl = 0.0
    daily_locked = False
    trailing_locked = False
    current_day: Optional[date] = None
    position: Optional[Dict[str, object]] = None

    daily_trade_count = 0
    daily_trades_list: List[int] = []

    # RTH days tracking (for accurate trading days count)
    rth_bars_per_day: Dict[date, int] = {}

    # Confidence tracking (for quality analysis)
    confidence_winners: List[float] = []
    confidence_losers: List[float] = []
    confidence_catastop: List[float] = []
    confidence_timeexit: List[float] = []

    # Feasibility tracking
    rejected_by_feasibility = 0

    for i in range(start, end - 1):
        bar = bars_df.iloc[i]
        bar_time = _to_chicago(bar["timestamp"])
        bar_day = bar_time.date()

        # Track RTH bars per day for accurate trading days count
        if in_session(
            bar["timestamp"],
            session_mode=session_mode,
            session_start=session_start,
            session_end=session_end,
        ):
            rth_bars_per_day[bar_day] = rth_bars_per_day.get(bar_day, 0) + 1

        if current_day != bar_day:
            if current_day is not None:
                daily_trades_list.append(daily_trade_count)
            current_day = bar_day
            daily_pnl = 0.0
            daily_locked = False
            daily_trade_count = 0

        if position is not None:
            hold_bars = i - position["entry_idx"]
            exit_reason = None
            exit_price = None

            if session_mode.upper() == "RTH" and bar_time.time() >= session_end:
                exit_reason = "SESSION_FLAT"
                exit_price = bar["close"]
            elif execution_mode == "time_exit":
                if position["direction"] == "long":
                    if bar["close"] <= position["stop_price"]:
                        exit_reason = "CATASTOP"
                        exit_price = bar["close"]
                else:
                    if bar["close"] >= position["stop_price"]:
                        exit_reason = "CATASTOP"
                        exit_price = bar["close"]
                if exit_reason is None and hold_bars >= horizon_bars:
                    exit_reason = "TIME_EXIT"
                    exit_price = bar["close"] if exit_price_mode == "bar_close" else bar["open"]
            else:
                if hold_bars >= max_hold_bars:
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
                    tick_size=tick_size,
                    tick_value=tick_value,
                )
                fees = (
                    2 * position["contracts"] * commission_per_contract
                    + slippage_ticks * position["contracts"] * tick_value
                )
                pnl = raw_pnl - fees

                equity += pnl
                daily_pnl += pnl
                trailing_peak = max(trailing_peak, equity)
                equity_curve.append(equity)

                # Update selector's daily PnL tracking for quality gates
                selector.update_daily_pnl(pnl, bar["timestamp"])

                # Track confidence for quality analysis
                conf_val = position.get("confidence")
                if conf_val is not None and not pd.isna(conf_val):
                    if pnl > 0:
                        confidence_winners.append(float(conf_val))
                    else:
                        confidence_losers.append(float(conf_val))
                    if exit_reason == "CATASTOP":
                        confidence_catastop.append(float(conf_val))
                    elif exit_reason == "TIME_EXIT":
                        confidence_timeexit.append(float(conf_val))

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
                        "confidence": position.get("confidence"),
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Signal rejected: rejected_by=risk daily=%s trailing=%s idx=%s",
                    daily_locked,
                    trailing_locked,
                    i,
                )
            continue

        if not in_session(
            bar["timestamp"],
            session_mode=session_mode,
            session_start=session_start,
            session_end=session_end,
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Signal rejected: outside_session idx=%s", i)
            continue

        # Feasibility check: ensure enough bars left in RTH for time-exit
        if not is_entry_feasible(
            bar["timestamp"],
            horizon_bars=horizon_bars,
            bar_minutes=bar_minutes,
            rth_end_time=session_end,
            execution_mode=execution_mode,
            session_mode=session_mode,
        ):
            rejected_by_feasibility += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Signal rejected: feasibility idx=%s time=%s horizon=%s",
                    i,
                    bar["timestamp"],
                    horizon_bars,
                )
            continue

        probs = prob_df.iloc[i]
        score = probs.get("score")
        direction = probs.get("direction")
        long_prob = probs.get("long_prob")
        short_prob = probs.get("short_prob")
        confidence = probs.get("confidence")  # New: get confidence for quality gates

        if pd.isna(score) or pd.isna(long_prob) or pd.isna(short_prob):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Signal rejected: missing_score idx=%s", i)
            continue

        if direction == "long" and not enable_long:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Signal rejected: long_disabled idx=%s", i)
            continue
        if direction == "short" and not enable_short:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Signal rejected: short_disabled idx=%s", i)
            continue

        # Pass confidence to selector for quality gates on trades 3-4
        if not selector.should_enter(
            bar["timestamp"],
            score=float(score),
            direction=str(direction),
            bar_index=i,
            confidence=float(confidence) if not pd.isna(confidence) else None,
        ):
            selector.log_rejection()
            continue

        next_bar = bars_df.iloc[i + 1]
        entry_idx = i + 1
        if not in_session(
            next_bar["timestamp"],
            session_mode=session_mode,
            session_start=session_start,
            session_end=session_end,
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Signal rejected: next_bar_outside_session idx=%s", i)
            continue

        entry_price = next_bar["open"]
        slippage = slippage_ticks * tick_size
        if direction == "long":
            entry_price += slippage
        else:
            entry_price -= slippage

        # Compute catastrophic stop (dynamic or fixed)
        if execution_mode == "time_exit":
            if use_dynamic_catastop and bars_with_features is not None:
                # Dynamic stop based on ATR at signal time (causal)
                signal_bar = bars_with_features.iloc[i]
                atr_ticks = signal_bar.get("atr_ticks", 0)
                if pd.isna(atr_ticks) or atr_ticks <= 0:
                    # Fallback to fixed if ATR unavailable
                    stop_ticks = int(catastrophic_stop_ticks)
                else:
                    # Compute dynamic stop: clamp(atr * multiplier, min, max)
                    raw_stop = int(atr_ticks * catastop_atr_multiplier)
                    stop_ticks = max(catastop_min_ticks, min(catastop_max_ticks, raw_stop))
            else:
                # Fixed stop
                stop_ticks = int(catastrophic_stop_ticks)
            stop_distance = stop_ticks * tick_size
            target_distance = stop_distance * target_multiplier
        else:
            stop_ticks = int(stop_loss_ticks)
            stop_distance = stop_ticks * tick_size
            target_distance = stop_distance * target_multiplier

        if direction == "long":
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

        position = {
            "direction": direction,
            "contracts": 1,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_time": next_bar["timestamp"],
            "entry_idx": entry_idx,
            "stop_ticks": stop_ticks,
            "score": score,
            "confidence": float(confidence) if not pd.isna(confidence) else None,
        }

        selector.register_trade(next_bar["timestamp"], bar_index=entry_idx)
        daily_trade_count += 1

    if position is not None:
        last_bar = bars_df.iloc[end - 1]
        exit_price = last_bar["close"]
        raw_pnl = _position_pnl(
            position["entry_price"],
            exit_price,
            position["direction"],
            position["contracts"],
            tick_size=tick_size,
            tick_value=tick_value,
        )
        fees = (
            2 * position["contracts"] * commission_per_contract
            + slippage_ticks * position["contracts"] * tick_value
        )
        pnl = raw_pnl - fees
        equity += pnl
        equity_curve.append(equity)

        # Update selector's daily PnL tracking
        selector.update_daily_pnl(pnl, last_bar["timestamp"])

        # Track confidence for final position
        conf_val = position.get("confidence")
        if conf_val is not None and not pd.isna(conf_val):
            if pnl > 0:
                confidence_winners.append(float(conf_val))
            else:
                confidence_losers.append(float(conf_val))

        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": last_bar["timestamp"],
                "direction": position["direction"],
                "contracts": position["contracts"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": "SEGMENT_END" if end < len(bars_df) else "FINAL_BAR",
                "stop_ticks": position["stop_ticks"],
                "score": position["score"],
                "confidence": position.get("confidence"),
            }
        )

    if current_day is not None:
        daily_trades_list.append(daily_trade_count)

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
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
        win_rate = len(wins) / len(trades) if len(trades) else 0.0
        gross_wins = wins["pnl"].sum() if len(wins) else 0.0
        gross_losses = abs(losses["pnl"].sum()) if len(losses) else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        equity_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = peaks - equity_arr
        max_drawdown = float(np.max(drawdowns))

        summary = {
            "trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
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

    trade_counts = np.array(daily_trades_list)
    if trade_counts.size:
        unique_counts, count_totals = np.unique(trade_counts, return_counts=True)
        distribution = {int(k): int(v) for k, v in zip(unique_counts, count_totals)}
        days_with_trades = int(np.sum(trade_counts > 0))
        avg_active = float(trade_counts[trade_counts > 0].mean()) if days_with_trades > 0 else 0.0
        pct_days_with_1 = float(distribution.get(1, 0) / len(trade_counts) * 100.0)
        pct_days_with_2 = float(distribution.get(2, 0) / len(trade_counts) * 100.0)
    else:
        distribution = {}
        days_with_trades = 0
        avg_active = 0.0
        pct_days_with_1 = 0.0
        pct_days_with_2 = 0.0

    # Compute RTH days metrics (accurate trading days count)
    min_rth_bars = 30  # Days need >=30 RTH bars to be considered trading days
    rth_days_in_data = sum(1 for count in rth_bars_per_day.values() if count >= min_rth_bars)

    # Count RTH days with trades (need to map trade days to RTH days)
    trade_days_set = set()
    if not trades_df.empty and "entry_time" in trades_df.columns:
        for entry_time in trades_df["entry_time"]:
            trade_day = _to_chicago(entry_time).date()
            trade_days_set.add(trade_day)
    rth_days_with_trades = sum(1 for day in trade_days_set if rth_bars_per_day.get(day, 0) >= min_rth_bars)
    pct_rth_days_traded = (
        float(rth_days_with_trades) / rth_days_in_data * 100.0 if rth_days_in_data > 0 else 0.0
    )

    # Compute confidence statistics
    confidence_stats = {}
    if confidence_winners or confidence_losers:
        all_confidence = confidence_winners + confidence_losers
        confidence_stats = {
            "avg_confidence_all": float(np.mean(all_confidence)) if all_confidence else 0.0,
            "avg_confidence_winners": float(np.mean(confidence_winners)) if confidence_winners else 0.0,
            "avg_confidence_losers": float(np.mean(confidence_losers)) if confidence_losers else 0.0,
            "avg_confidence_catastop": float(np.mean(confidence_catastop)) if confidence_catastop else 0.0,
            "avg_confidence_timeexit": float(np.mean(confidence_timeexit)) if confidence_timeexit else 0.0,
            "median_confidence_all": float(np.median(all_confidence)) if all_confidence else 0.0,
        }
        # Confidence percentiles
        if all_confidence:
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                confidence_stats[f"p{p}_confidence"] = float(np.percentile(all_confidence, p))

    daily_stats = {
        "calendar_days": len(daily_trades_list),
        "rth_days_in_data": rth_days_in_data,
        "rth_days_with_trades": rth_days_with_trades,
        "pct_rth_days_traded": pct_rth_days_traded,
        "total_trading_days": len(daily_trades_list),
        "avg_trades_per_day": float(np.mean(daily_trades_list)) if daily_trades_list else 0.0,
        "avg_trades_per_active_day": avg_active,
        "max_trades_in_day": int(np.max(daily_trades_list)) if daily_trades_list else 0,
        "days_with_trades": days_with_trades,
        "days_with_zero_trades": int(np.sum(trade_counts == 0)) if trade_counts.size else 0,
        "trades_per_day_distribution": distribution,
        "pct_days_with_1_trade": pct_days_with_1,
        "pct_days_with_2_trades": pct_days_with_2,
    }

    if save_trades_path:
        trades_df.to_csv(save_trades_path, index=False)

    exit_reason_counts = {}
    if not trades_df.empty and "reason" in trades_df.columns:
        exit_reason_counts = trades_df["reason"].value_counts().to_dict()
    exit_reason_avg_pnl = {}
    if not trades_df.empty and "reason" in trades_df.columns:
        exit_reason_avg_pnl = trades_df.groupby("reason")["pnl"].mean().to_dict()

    # Feasibility stats
    total_bars_checked = end - start - 1
    feasibility_stats = {
        "rejected_by_feasibility": rejected_by_feasibility,
        "total_bars_checked": total_bars_checked,
        "feasibility_rejection_rate": (
            float(rejected_by_feasibility) / total_bars_checked if total_bars_checked > 0 else 0.0
        ),
    }

    return {
        "summary": summary,
        "trades": trades,
        "daily_stats": daily_stats,
        "exit_reason_counts": exit_reason_counts,
        "exit_reason_avg_pnl": exit_reason_avg_pnl,
        "feasibility_stats": feasibility_stats,
        "confidence_stats": confidence_stats,
    }

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


def load_bars(h5_path: str, *, dataset_key: str = "bars_5min") -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        if dataset_key not in store:
            raise KeyError(f"Dataset key {dataset_key!r} not found in H5.")
        bars = store[dataset_key].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    return bars


def _compute_probabilities(*args, **kwargs) -> pd.DataFrame:  # pragma: no cover
    # Backward-compatible alias.
    return compute_probabilities(*args, **kwargs)


def _position_pnl(
    entry_price: float,
    exit_price: float,
    direction: str,
    contracts: int,
    *,
    tick_size: float,
    tick_value: float,
) -> float:
    ticks = (exit_price - entry_price) / tick_size
    if direction == "short":
        ticks *= -1
    return ticks * tick_value * contracts


def run_backtest(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
    save_trades_path: Optional[str] = None,
    *,
    prob_df: Optional[pd.DataFrame] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    min_probability_long: Optional[float] = None,
    min_probability_short: Optional[float] = None,
    enable_long: Optional[bool] = None,
    enable_short: Optional[bool] = None,
    blocked_hours: Optional[List[int]] = None,
    allowed_hours: Optional[List[int]] = None,
    exclude_lunch: Optional[bool] = None,
    require_trend_long: Optional[bool] = None,
    require_trend_short: Optional[bool] = None,
    min_atr_ticks: Optional[float] = None,
    max_atr_ticks: Optional[float] = None,
    stop_loss_ticks: Optional[int] = None,
    target_multiplier: Optional[float] = None,
    max_hold_bars: Optional[int] = None,
    slippage_ticks: Optional[int] = None,
    commission_per_contract: Optional[float] = None,
) -> Dict[str, object]:
    if prob_df is None:
        prob_df = compute_probabilities(bars_df, long_model, short_model, feature_cols)

    start = 0 if start_idx is None else int(start_idx)
    end = len(bars_df) if end_idx is None else int(end_idx)
    if start < 0 or end > len(bars_df) or start >= end:
        raise ValueError(f"Invalid start/end window: start={start}, end={end}, rows={len(bars_df)}")
    if end - start < 2:
        raise ValueError("Backtest window must contain at least 2 bars")

    min_prob_long = TRAINING_CONFIG.min_probability_long if min_probability_long is None else float(min_probability_long)
    min_prob_short = (
        TRAINING_CONFIG.min_probability_short if min_probability_short is None else float(min_probability_short)
    )
    allow_long = TRAINING_CONFIG.enable_long if enable_long is None else bool(enable_long)
    allow_short = TRAINING_CONFIG.enable_short if enable_short is None else bool(enable_short)
    blocked_hours_set = set(int(h) for h in (blocked_hours or []))
    allowed_hours_set = set(int(h) for h in (allowed_hours or []))
    exclude_lunch = False if exclude_lunch is None else bool(exclude_lunch)
    require_trend_long = False if require_trend_long is None else bool(require_trend_long)
    require_trend_short = False if require_trend_short is None else bool(require_trend_short)
    stop_loss_ticks = TRAINING_CONFIG.stop_loss_ticks if stop_loss_ticks is None else int(stop_loss_ticks)
    target_multiplier = TRAINING_CONFIG.target_multiplier if target_multiplier is None else float(target_multiplier)
    max_hold_bars = TRAINING_CONFIG.max_hold_bars if max_hold_bars is None else int(max_hold_bars)
    slippage_ticks = 1 if slippage_ticks is None else int(slippage_ticks)
    commission = 2.35 if commission_per_contract is None else float(commission_per_contract)

    trades: List[Dict[str, object]] = []
    equity_curve: List[float] = [RISK_CONFIG.starting_balance]
    equity = RISK_CONFIG.starting_balance
    trailing_peak = equity
    daily_pnl = 0.0
    daily_locked = False
    trailing_locked = False
    current_day = None

    position = None

    def _gate_direction(direction: str, bar_time: pd.Timestamp, row: pd.Series) -> bool:
        hour = int(bar_time.hour)
        if allowed_hours_set and hour not in allowed_hours_set:
            return False
        if blocked_hours_set and hour in blocked_hours_set:
            return False
        if exclude_lunch and int(row.get("lunch_period", 0) or 0) == 1:
            return False
        atr = row.get("atr_ticks")
        if min_atr_ticks is not None and pd.notna(atr) and float(atr) < float(min_atr_ticks):
            return False
        if max_atr_ticks is not None and pd.notna(atr) and float(atr) > float(max_atr_ticks):
            return False

        if direction == "long" and require_trend_long:
            spread = row.get("ema_spread_21_50")
            above = row.get("price_above_ema50")
            if pd.isna(spread) or float(spread) <= 0:
                return False
            if pd.isna(above) or int(above) != 1:
                return False
        if direction == "short" and require_trend_short:
            spread = row.get("ema_spread_21_50")
            above = row.get("price_above_ema50")
            if pd.isna(spread) or float(spread) >= 0:
                return False
            if pd.isna(above) or int(above) != 0:
                return False

        return True

    for i in range(start, end - 1):
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
            elif hold_bars >= max_hold_bars:
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
                    tick_size=RISK_CONFIG.tick_size,
                    tick_value=RISK_CONFIG.tick_value,
                )
                fees = (
                    2 * position["contracts"] * commission
                    + slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
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

        long_ok = allow_long and long_prob >= min_prob_long and _gate_direction("long", bar_time, probs)
        short_ok = allow_short and short_prob >= min_prob_short and _gate_direction("short", bar_time, probs)
        if not long_ok and not short_ok:
            continue

        if long_ok and short_ok:
            direction = "long" if long_prob >= short_prob else "short"
        else:
            direction = "long" if long_ok else "short"

        stop_ticks = stop_loss_ticks

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
        target_distance = stop_distance * target_multiplier

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
        # Force flat at the end of the requested window to avoid leaking beyond split.
        last_bar = bars_df.iloc[end - 1]
        exit_price = last_bar["close"]
        raw_pnl = _position_pnl(
            position["entry_price"],
            exit_price,
            position["direction"],
            position["contracts"],
            tick_size=RISK_CONFIG.tick_size,
            tick_value=RISK_CONFIG.tick_value,
        )
        fees = 2 * position["contracts"] * commission + slippage_ticks * position["contracts"] * RISK_CONFIG.tick_value
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
                "reason": "SEGMENT_END" if end < len(bars_df) else "FINAL_BAR",
                "stop_ticks": position["stop_ticks"],
            }
        )

    pnl_values = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]

    gross_win = wins.sum() if wins.size else 0.0
    gross_loss = abs(losses.sum()) if losses.size else 0.0
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    win_rate = float((pnl_values > 0).mean()) if pnl_values.size else 0.0
    net_pnl = float(pnl_values.sum()) if pnl_values.size else 0.0

    equity_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_arr)
    drawdowns = peaks - equity_arr
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0

    # Convert config to dict and serialize time objects to strings
    risk_dict = asdict(RISK_CONFIG)
    training_dict = asdict(TRAINING_CONFIG)
    
    # Convert time objects to strings for JSON serialization
    if "session_start" in risk_dict and isinstance(risk_dict["session_start"], time):
        risk_dict["session_start"] = risk_dict["session_start"].isoformat()
    if "session_end" in risk_dict and isinstance(risk_dict["session_end"], time):
        risk_dict["session_end"] = risk_dict["session_end"].isoformat()
    
    summary = {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "max_drawdown": max_drawdown,
        "ending_equity": equity,
        "starting_balance": RISK_CONFIG.starting_balance,
        "window": {"start_idx": start, "end_idx": end},
        "policy": {
            "enable_long": allow_long,
            "enable_short": allow_short,
            "min_probability_long": min_prob_long,
            "min_probability_short": min_prob_short,
            "blocked_hours": sorted(blocked_hours_set),
            "allowed_hours": sorted(allowed_hours_set),
            "exclude_lunch": exclude_lunch,
            "require_trend_long": require_trend_long,
            "require_trend_short": require_trend_short,
            "min_atr_ticks": min_atr_ticks,
            "max_atr_ticks": max_atr_ticks,
            "stop_loss_ticks": stop_loss_ticks,
            "target_multiplier": target_multiplier,
            "max_hold_bars": max_hold_bars,
            "slippage_ticks": slippage_ticks,
            "commission_per_contract": commission,
        },
        "config": {
            "risk": risk_dict,
            "training": training_dict,
        },
    }

    if save_trades_path:
        trades_df = pd.DataFrame(trades)
        Path(save_trades_path).parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(save_trades_path, index=False)

    return {"summary": summary, "trades": trades}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML strategy")
    parser.add_argument("--data-path", default="/mnt/data/es_bars_2010_2025.h5")
    parser.add_argument("--dataset-key", default="bars_5min")
    parser.add_argument("--model-dir", default="models/nn_saved")
    parser.add_argument("--artifact-dir", dest="model_dir")
    parser.add_argument("--strategy", choices=["auto", "rf", "nn"], default="auto")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for NN artifacts")
    parser.add_argument("--save-trades", dest="save_trades", default=None)
    parser.add_argument("--out-trades-csv", dest="save_trades")
    parser.add_argument("--split", choices=["full", "training", "validation", "test"], default="full")
    parser.add_argument("--start", default=None, help="Start timestamp/date (inclusive), e.g. 2024-01-01")
    parser.add_argument("--end", default=None, help="End timestamp/date (inclusive), e.g. 2024-06-01")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--rth-only", action="store_true", default=False)
    parser.set_defaults(save_trades=None)
    parser.add_argument("--min-prob-long", type=float, default=None)
    parser.add_argument("--min-prob-short", type=float, default=None)
    parser.add_argument("--blocked-hours", default=None, help="Comma-separated hours (CT) to skip, e.g. 14,13")
    parser.add_argument("--allowed-hours", default=None, help="Comma-separated hours (CT) to allow, e.g. 9,10,11")
    parser.add_argument("--exclude-lunch", action="store_true", default=False)
    parser.add_argument("--require-trend-long", action="store_true", default=False)
    parser.add_argument("--require-trend-short", action="store_true", default=False)
    parser.add_argument("--min-atr-ticks", type=float, default=None)
    parser.add_argument("--max-atr-ticks", type=float, default=None)
    parser.add_argument("--stop-ticks", type=int, default=None)
    parser.add_argument("--target-mult", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument("--enable-long", action="store_true", default=False)
    parser.add_argument("--disable-long", action="store_true", default=False)
    parser.add_argument("--enable-short", action="store_true", default=False)
    parser.add_argument("--disable-short", action="store_true", default=False)
    args = parser.parse_args()

    use_nn = args.strategy == "nn" or (args.strategy == "auto" and _is_nn_model_dir(args.model_dir))
    bars = load_bars(args.data_path, dataset_key=args.dataset_key)
    if args.fast and args.max_bars is None and args.max_days is None:
        args.max_days = 90

    if args.max_bars is not None:
        max_bars = int(args.max_bars)
        if max_bars <= 0:
            raise ValueError("--max-bars must be > 0")
        bars = bars.tail(max_bars).reset_index(drop=True)
    elif args.max_days is not None:
        max_days = int(args.max_days)
        if max_days <= 0:
            raise ValueError("--max-days must be > 0")
        bar_times = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/Chicago")
        bar_days = bar_times.dt.date
        unique_days = np.array(sorted(bar_days.unique()))
        if len(unique_days) > max_days:
            start_day = unique_days[-max_days]
            mask = bar_days >= start_day
            bars = bars.loc[mask].reset_index(drop=True)

    if use_nn:
        bundle = load_nn_bundle(args.model_dir, fold=args.fold)
        nn_cfg = bundle.config.get("nn_config", {})

        score_threshold = float(nn_cfg["score_threshold"])
        selection_mode = str(nn_cfg.get("selection_mode", "global_threshold"))
        day_percentile_floor = float(nn_cfg.get("day_percentile_floor", 0.90))
        global_floor_score = nn_cfg.get("global_floor_score")
        if global_floor_score is None:
            global_floor_score = float(score_threshold)
        max_trades_per_day = int(nn_cfg["max_trades_per_day"])
        min_bars_between_trades = int(nn_cfg["min_bars_between_trades"])
        enable_long = bool(nn_cfg["enable_long"])
        enable_short = bool(nn_cfg["enable_short"])
        horizon_bars = int(nn_cfg["horizon_bars"])
        execution_mode = str(nn_cfg.get("execution_mode", "time_exit"))
        exit_price_mode = str(nn_cfg.get("exit_price_mode", "bar_close"))
        session_mode = str(nn_cfg.get("session_mode", "RTH"))
        if args.rth_only:
            session_mode = "RTH"
        deadline_time = nn_cfg.get("deadline_time")
        deadline_relax_factor = float(nn_cfg.get("deadline_relax_factor", 0.98))
        bar_minutes = int(nn_cfg.get("bar_minutes", 5))
        max_hold_bars = int(nn_cfg.get("max_hold_bars", horizon_bars))
        tick_size = float(nn_cfg["tick_size"])
        tick_value = float(nn_cfg["tick_value"])
        stop_loss_ticks = int(nn_cfg["stop_loss_ticks"])
        target_multiplier = float(nn_cfg["target_multiplier"])
        catastrophic_stop_ticks = nn_cfg.get("catastrophic_stop_ticks")
        if catastrophic_stop_ticks is None:
            catastrophic_stop_ticks = int(nn_cfg["threshold_ticks"]) * 4
        risk_cfg = bundle.config.get("risk_config", {})
        session_start = RISK_CONFIG.session_start
        session_end = RISK_CONFIG.session_end
        if isinstance(risk_cfg.get("session_start"), str):
            session_start = time.fromisoformat(risk_cfg["session_start"])
        if isinstance(risk_cfg.get("session_end"), str):
            session_end = time.fromisoformat(risk_cfg["session_end"])

        if args.split != "full":
            raise SystemExit("NN backtest supports --start/--end; split metadata is not available.")

        window_start_idx: Optional[int] = None
        window_end_idx: Optional[int] = None
        if args.start or args.end:
            start_ts = pd.to_datetime(args.start, utc=True) if args.start else None
            end_ts = pd.to_datetime(args.end, utc=True) if args.end else None
            ts = bars["timestamp"]
            if start_ts is not None:
                window_start_idx = int(ts.searchsorted(start_ts, side="left"))
            if end_ts is not None:
                window_end_idx = int(ts.searchsorted(end_ts, side="right"))

        prob_df = predict_scores_for_bars(bars, bundle)
        results = run_backtest_nn(
            bars,
            prob_df,
            score_threshold=score_threshold,
            selection_mode=selection_mode,
            day_percentile_floor=day_percentile_floor,
            global_floor_score=float(global_floor_score),
            max_trades_per_day=max_trades_per_day,
            min_bars_between_trades=min_bars_between_trades,
            enable_long=enable_long,
            enable_short=enable_short,
            horizon_bars=horizon_bars,
            execution_mode=execution_mode,
            exit_price_mode=exit_price_mode,
            session_mode=session_mode,
            deadline_time=deadline_time,
            deadline_relax_factor=deadline_relax_factor,
            bar_minutes=bar_minutes,
            session_start=session_start,
            session_end=session_end,
            start_idx=window_start_idx,
            end_idx=window_end_idx,
            stop_loss_ticks=stop_loss_ticks,
            target_multiplier=target_multiplier,
            catastrophic_stop_ticks=int(catastrophic_stop_ticks),
            max_hold_bars=max_hold_bars,
            tick_size=tick_size,
            tick_value=tick_value,
            save_trades_path=args.save_trades,
        )
        print(json.dumps(results["summary"], indent=2))
        print(json.dumps(results["daily_stats"], indent=2))
        if results.get("exit_reason_counts"):
            print(json.dumps(results["exit_reason_counts"], indent=2))
        if results.get("exit_reason_avg_pnl"):
            print(json.dumps(results["exit_reason_avg_pnl"], indent=2))
        return

    long_model, short_model, metadata = load_models(args.model_dir)
    feature_cols = metadata.get("feature_cols") or select_features()

    # Allow metadata to provide tuned policy unless explicitly overridden.
    policy = (metadata.get("policy") or {}) if isinstance(metadata, dict) else {}
    min_prob_long = args.min_prob_long if args.min_prob_long is not None else policy.get("min_probability_long")
    min_prob_short = args.min_prob_short if args.min_prob_short is not None else policy.get("min_probability_short")

    enable_long = None
    if args.enable_long:
        enable_long = True
    if args.disable_long:
        enable_long = False
    if enable_long is None:
        enable_long = policy.get("enable_long", None)

    enable_short = None
    if args.enable_short:
        enable_short = True
    if args.disable_short:
        enable_short = False
    if enable_short is None:
        enable_short = policy.get("enable_short", None)

    def _parse_hours(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        vals = []
        for part in str(s).split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(int(part))
        return vals or None

    blocked_hours = _parse_hours(args.blocked_hours)
    allowed_hours = _parse_hours(args.allowed_hours)
    exclude_lunch = bool(args.exclude_lunch) if args.exclude_lunch else bool(policy.get("exclude_lunch", False))
    require_trend_long = bool(args.require_trend_long) if args.require_trend_long else bool(policy.get("require_trend_long", False))
    require_trend_short = bool(args.require_trend_short) if args.require_trend_short else bool(policy.get("require_trend_short", False))
    min_atr_ticks = args.min_atr_ticks if args.min_atr_ticks is not None else policy.get("min_atr_ticks", None)
    max_atr_ticks = args.max_atr_ticks if args.max_atr_ticks is not None else policy.get("max_atr_ticks", None)
    stop_ticks = args.stop_ticks if args.stop_ticks is not None else policy.get("stop_loss_ticks", None)
    target_mult = args.target_mult if args.target_mult is not None else policy.get("target_multiplier", None)
    max_hold_bars = args.max_hold_bars if args.max_hold_bars is not None else policy.get("max_hold_bars", None)
    if blocked_hours is None:
        blocked_hours = policy.get("blocked_hours", None)
    if allowed_hours is None:
        allowed_hours = policy.get("allowed_hours", None)

    # Window selection: keep full history for feature context, but restrict trading to indices.
    window_start_idx: Optional[int] = None
    window_end_idx: Optional[int] = None

    if args.start or args.end:
        start_ts = pd.to_datetime(args.start, utc=True) if args.start else None
        end_ts = pd.to_datetime(args.end, utc=True) if args.end else None
        ts = bars["timestamp"]
        if start_ts is not None:
            window_start_idx = int(ts.searchsorted(start_ts, side="left"))
        if end_ts is not None:
            window_end_idx = int(ts.searchsorted(end_ts, side="right"))
    elif args.split != "full":
        bt_meta = (metadata.get("backtest_metrics") or {}) if isinstance(metadata, dict) else {}
        win_ts = (bt_meta.get("window_timestamps") or {}).get(args.split) if isinstance(bt_meta, dict) else None
        if not win_ts or not win_ts.get("start_timestamp") or not win_ts.get("end_timestamp"):
            raise SystemExit(
                f"metadata.json missing window timestamps for split '{args.split}'. "
                "Re-train models with updated trainer or use --start/--end."
            )
        start_ts = pd.to_datetime(win_ts["start_timestamp"], utc=True)
        end_ts = pd.to_datetime(win_ts["end_timestamp"], utc=True)
        ts = bars["timestamp"]
        window_start_idx = int(ts.searchsorted(start_ts, side="left"))
        window_end_idx = int(ts.searchsorted(end_ts, side="right"))

    results = run_backtest(
        bars,
        long_model,
        short_model,
        feature_cols,
        args.save_trades,
        start_idx=window_start_idx,
        end_idx=window_end_idx,
        min_probability_long=min_prob_long,
        min_probability_short=min_prob_short,
        enable_long=enable_long,
        enable_short=enable_short,
        blocked_hours=blocked_hours,
        allowed_hours=allowed_hours,
        exclude_lunch=exclude_lunch,
        require_trend_long=require_trend_long,
        require_trend_short=require_trend_short,
        min_atr_ticks=min_atr_ticks,
        max_atr_ticks=max_atr_ticks,
        stop_loss_ticks=stop_ticks,
        target_multiplier=target_mult,
        max_hold_bars=max_hold_bars,
    )
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
