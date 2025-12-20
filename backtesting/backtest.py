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

from datetime import time

from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, select_features


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
    bars, _ = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    return bars


def _compute_probabilities(*args, **kwargs) -> pd.DataFrame:  # pragma: no cover
    # Backward-compatible alias.
    return compute_probabilities(*args, **kwargs)


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
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--model-dir", default="models/saved")
    parser.add_argument("--save-trades", default=None)
    parser.add_argument("--split", choices=["full", "training", "validation", "test"], default="full")
    parser.add_argument("--start", default=None, help="Start timestamp/date (inclusive), e.g. 2024-01-01")
    parser.add_argument("--end", default=None, help="End timestamp/date (inclusive), e.g. 2024-06-01")
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

    long_model, short_model, metadata = load_models(args.model_dir)
    feature_cols = metadata.get("feature_cols") or select_features()

    bars = load_bars(args.data_path)

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
