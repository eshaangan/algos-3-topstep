"""
Create labels by simulating what would happen if we took trades
rather than predicting raw price direction.
"""

from __future__ import annotations

from datetime import time
from typing import Literal

import pandas as pd


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def simulate_trade_outcome(
    bars_df: pd.DataFrame,
    entry_idx: int,
    direction: Literal["long", "short"] = "long",
    stop_ticks: int = 12,
    target_multiplier: float = 1.5,
    max_hold_bars: int = 12,
    tick_size: float = 0.25,
    tick_value: float = 1.25,
    slippage_ticks: int = 1,
    commission: float = 2.35,
) -> Literal["win", "loss", "neutral"]:
    """
    Simulate a trade from entry_idx forward.

    CRITICAL FIX: Entry uses NEXT bar's open (matching backtest execution)
    to eliminate 1-bar lookahead bias.

    Returns:
        "win": Trade would be profitable
        "loss": Trade would hit stop
        "neutral": Unclear outcome (skip in training)
    """
    # Need extra bar for next-bar entry
    if entry_idx + max_hold_bars + 1 >= len(bars_df):
        return "neutral"

    # FIXED: Entry is next bar's open, not current bar's close
    next_bar = bars_df.iloc[entry_idx + 1]

    if direction == "long":
        entry_price = next_bar["open"] + (slippage_ticks * tick_size)
    else:
        entry_price = next_bar["open"] - (slippage_ticks * tick_size)

    stop_distance = stop_ticks * tick_size
    target_distance = stop_distance * target_multiplier

    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    # FIXED: Start from entry_idx + 2 (first bar after next-bar entry)
    for i in range(entry_idx + 2, min(entry_idx + max_hold_bars + 2, len(bars_df))):
        bar = bars_df.iloc[i]

        if direction == "long":
            if bar["low"] <= stop_price:
                return "loss"

            if bar["high"] >= target_price:
                pnl = (target_price - entry_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "win" if pnl > 0 else "loss"

        else:
            if bar["high"] >= stop_price:
                return "loss"

            if bar["low"] <= target_price:
                pnl = (entry_price - target_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "win" if pnl > 0 else "loss"

    # FIXED: Max hold exit at entry_idx + max_hold_bars + 1 (accounting for next-bar entry)
    exit_price = bars_df.iloc[entry_idx + max_hold_bars + 1]["close"]

    if direction == "long":
        pnl = (exit_price - entry_price) / tick_size * tick_value
    else:
        pnl = (entry_price - exit_price) / tick_size * tick_value

    pnl -= (2 * commission + 2 * slippage_ticks * tick_value)

    # Match execution: max-hold exits are realized P&L, not "neutral".
    return "win" if pnl > 0 else "loss"


def simulate_trade(
    bars_df: pd.DataFrame,
    entry_idx: int,
    *,
    direction: Literal["long", "short"],
    stop_ticks: int,
    target_multiplier: float,
    max_hold_bars: int,
    tick_size: float,
    tick_value: float,
    session_end: time,
    slippage_ticks: int = 1,
    commission: float = 2.35,
) -> dict:
    """
    Simulate a single trade and return realized P&L and exit index.

    Mirrors `backtesting/backtest.py`:
      - Entry is next bar open with slippage applied to entry price.
      - Stop/target are evaluated on subsequent bars.
      - Forced exit at session_end close.
      - Max-hold exit at close.
    """
    if entry_idx + 2 >= len(bars_df):
        raise ValueError("Not enough bars for next-bar entry")

    next_bar = bars_df.iloc[entry_idx + 1]
    if direction == "long":
        entry_price = float(next_bar["open"]) + slippage_ticks * tick_size
    else:
        entry_price = float(next_bar["open"]) - slippage_ticks * tick_size

    stop_distance = stop_ticks * tick_size
    target_distance = stop_distance * target_multiplier

    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    entry_time = pd.to_datetime(next_bar["timestamp"], utc=True)

    last_bar_idx = min(entry_idx + max_hold_bars + 1, len(bars_df) - 1)
    exit_idx = last_bar_idx
    exit_price = float(bars_df.iloc[last_bar_idx]["close"])
    exit_reason = "MAX_HOLD"

    for i in range(entry_idx + 2, last_bar_idx + 1):
        bar = bars_df.iloc[i]
        bar_time = _to_chicago(pd.to_datetime(bar["timestamp"], utc=True))

        if bar_time.time() >= session_end:
            exit_idx = i
            exit_price = float(bar["close"])
            exit_reason = "SESSION_FLAT"
            break

        if direction == "long":
            if float(bar["low"]) <= stop_price:
                exit_idx = i
                exit_price = stop_price
                exit_reason = "STOP"
                break
            if float(bar["high"]) >= target_price:
                exit_idx = i
                exit_price = target_price
                exit_reason = "TARGET"
                break
        else:
            if float(bar["high"]) >= stop_price:
                exit_idx = i
                exit_price = stop_price
                exit_reason = "STOP"
                break
            if float(bar["low"]) <= target_price:
                exit_idx = i
                exit_price = target_price
                exit_reason = "TARGET"
                break

    ticks = (exit_price - entry_price) / tick_size
    if direction == "short":
        ticks *= -1
    raw_pnl = ticks * tick_value
    fees = 2 * commission + slippage_ticks * tick_value
    pnl = raw_pnl - fees

    return {
        "entry_idx": entry_idx + 1,
        "exit_idx": int(exit_idx),
        "entry_time": entry_time,
        "exit_time": pd.to_datetime(bars_df.iloc[exit_idx]["timestamp"], utc=True),
        "direction": direction,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "pnl": float(pnl),
        "reason": exit_reason,
    }


def create_sequential_trade_labels(
    bars_df: pd.DataFrame,
    *,
    direction: Literal["long", "short"],
    lookback: int,
    stop_ticks: int,
    target_multiplier: float,
    max_hold_bars: int,
    tick_size: float,
    tick_value: float,
    session_start: time,
    session_end: time,
    slippage_ticks: int = 1,
    commission: float = 2.35,
    max_trades: int | None = None,
) -> pd.DataFrame:
    """
    Create a label dataset consisting of NON-OVERLAPPING sequential trades.

    This matches the backtest reality (one position at a time), addressing the
    mismatch between per-bar overlapping labels and sequential execution.
    """
    records = []
    i = int(lookback)
    n = len(bars_df)
    max_i = n - max_hold_bars - 2

    while i <= max_i:
        bar_time = _to_chicago(pd.to_datetime(bars_df.iloc[i]["timestamp"], utc=True))
        if bar_time.time() < session_start or bar_time.time() >= session_end:
            i += 1
            continue

        trade = simulate_trade(
            bars_df,
            i,
            direction=direction,
            stop_ticks=stop_ticks,
            target_multiplier=target_multiplier,
            max_hold_bars=max_hold_bars,
            tick_size=tick_size,
            tick_value=tick_value,
            session_end=session_end,
            slippage_ticks=slippage_ticks,
            commission=commission,
        )
        label = 1 if trade["pnl"] > 0 else 0
        records.append(
            {
                "idx": int(i),
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "label": int(label),
                "pnl": float(trade["pnl"]),
                "reason": trade["reason"],
            }
        )

        # After exit, you can re-enter immediately on the same bar in the backtest loop.
        i = int(trade["exit_idx"])

        if max_trades is not None and len(records) >= int(max_trades):
            break

    return pd.DataFrame(records)


def create_per_bar_trade_labels(
    bars_df: pd.DataFrame,
    *,
    direction: Literal["long", "short"],
    lookback: int,
    stop_ticks: int,
    target_multiplier: float,
    max_hold_bars: int,
    tick_size: float,
    tick_value: float,
    session_start: time,
    session_end: time,
    slippage_ticks: int = 1,
    commission: float = 2.35,
    max_trades: int | None = None,
) -> pd.DataFrame:
    """
    Create a label dataset for EVERY eligible bar independently (no sequential skipping).

    This answers: "If we enter on bar i (next-bar open), would this trade win or lose?"
    The backtest still enforces one-position-at-a-time; these labels just provide more
    training examples than the sequential 'always-enter' path.
    """
    records = []
    i0 = int(lookback)
    n = len(bars_df)
    max_i = n - max_hold_bars - 2

    for i in range(i0, max_i + 1):
        bar_time = _to_chicago(pd.to_datetime(bars_df.iloc[i]["timestamp"], utc=True))
        if bar_time.time() < session_start or bar_time.time() >= session_end:
            continue

        trade = simulate_trade(
            bars_df,
            i,
            direction=direction,
            stop_ticks=stop_ticks,
            target_multiplier=target_multiplier,
            max_hold_bars=max_hold_bars,
            tick_size=tick_size,
            tick_value=tick_value,
            session_end=session_end,
            slippage_ticks=slippage_ticks,
            commission=commission,
        )
        label = 1 if trade["pnl"] > 0 else 0
        records.append(
            {
                "idx": int(i),
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "label": int(label),
                "pnl": float(trade["pnl"]),
                "reason": trade["reason"],
            }
        )

        if max_trades is not None and len(records) >= int(max_trades):
            break

    return pd.DataFrame(records)


def create_labels(
    bars_df: pd.DataFrame,
    lookback: int = 100,
    stop_ticks: int = 12,
    target_multiplier: float = 1.5,
    max_hold_bars: int = 12,
    tick_size: float = 0.25,
    tick_value: float = 1.25,
) -> pd.DataFrame:
    """
    Create labels for entire dataset.

    Args:
        bars_df: DataFrame with OHLCV
        lookback: Bars needed for feature calculation

    Returns:
        DataFrame with label_long, label_short columns
    """
    print("\n" + "=" * 60)
    print("LABEL CREATION - TRADE SIMULATION")
    print("=" * 60)

    print(f"\nSimulating trades for {len(bars_df):,} bars...")
    print("(This can take a few minutes...)\n")

    labels = []
    label_map = {"loss": 0, "win": 1, "neutral": 2}

    # FIXED: Account for next-bar entry by subtracting 1 from range
    for i in range(lookback, len(bars_df) - max_hold_bars - 1):
        if i % 1000 == 0:
            pct = (i - lookback) / (len(bars_df) - max_hold_bars - lookback - 1) * 100
            print(f"  Progress: {pct:.1f}% ({i:,}/{len(bars_df):,} bars)")

        long_outcome = simulate_trade_outcome(
            bars_df,
            i,
            direction="long",
            stop_ticks=stop_ticks,
            target_multiplier=target_multiplier,
            max_hold_bars=max_hold_bars,
            tick_size=tick_size,
            tick_value=tick_value,
        )
        short_outcome = simulate_trade_outcome(
            bars_df,
            i,
            direction="short",
            stop_ticks=stop_ticks,
            target_multiplier=target_multiplier,
            max_hold_bars=max_hold_bars,
            tick_size=tick_size,
            tick_value=tick_value,
        )

        labels.append(
            {
                "idx": i,
                "timestamp": bars_df.iloc[i]["timestamp"],
                "label_long": label_map[long_outcome],
                "label_short": label_map[short_outcome],
            }
        )

    labels_df = pd.DataFrame(labels)

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    print("\nLONG TRADES:")
    for label_name, label_val in [("Win", 1), ("Loss", 0), ("Neutral", 2)]:
        count = (labels_df["label_long"] == label_val).sum()
        pct = count / len(labels_df) * 100
        print(f"  {label_name:8s}: {count:6,} ({pct:5.1f}%)")

    print("\nSHORT TRADES:")
    for label_name, label_val in [("Win", 1), ("Loss", 0), ("Neutral", 2)]:
        count = (labels_df["label_short"] == label_val).sum()
        pct = count / len(labels_df) * 100
        print(f"  {label_name:8s}: {count:6,} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    long_wins = (labels_df["label_long"] == 1).sum()
    long_losses = (labels_df["label_long"] == 0).sum()
    long_wr = long_wins / (long_wins + long_losses) if (long_wins + long_losses) > 0 else 0

    short_wins = (labels_df["label_short"] == 1).sum()
    short_losses = (labels_df["label_short"] == 0).sum()
    short_wr = short_wins / (short_wins + short_losses) if (short_wins + short_losses) > 0 else 0

    print(f"\nLong win rate (excl. neutral): {long_wr * 100:.1f}%")
    print(f"Short win rate (excl. neutral): {short_wr * 100:.1f}%")

    if long_wr < 0.35 or long_wr > 0.65:
        print("WARNING: Long win rate outside 35-65% range (check stop/target ratio)")
    if short_wr < 0.35 or short_wr > 0.65:
        print("WARNING: Short win rate outside 35-65% range (check stop/target ratio)")

    neutral_pct = (labels_df["label_long"] == 2).mean() * 100
    if neutral_pct > 50:
        print(f"WARNING: {neutral_pct:.1f}% neutral labels (too many unclear trades)")

    print("\n" + "=" * 60 + "\n")

    return labels_df


if __name__ == "__main__":
    print("Loading data...")
    with pd.HDFStore("data/processed/mes_bars.h5", "r") as store:
        bars = store["bars_5min"]

    test_bars = bars.tail(3000).reset_index(drop=True)
    labels = create_labels(test_bars, lookback=100)

    print("Sample labels:")
    print(labels.head(20))
