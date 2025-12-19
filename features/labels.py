"""
Create labels by simulating what would happen if we took trades
rather than predicting raw price direction.
"""

from typing import Literal

import pandas as pd


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

    Returns:
        "win": Trade would be profitable
        "loss": Trade would hit stop
        "neutral": Unclear outcome (skip in training)
    """
    if entry_idx + max_hold_bars >= len(bars_df):
        return "neutral"

    entry_bar = bars_df.iloc[entry_idx]

    if direction == "long":
        entry_price = entry_bar["close"] + (slippage_ticks * tick_size)
    else:
        entry_price = entry_bar["close"] - (slippage_ticks * tick_size)

    stop_distance = stop_ticks * tick_size
    target_distance = stop_distance * target_multiplier

    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    for i in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, len(bars_df))):
        bar = bars_df.iloc[i]

        if direction == "long":
            if bar["low"] <= stop_price:
                pnl = (stop_price - entry_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "loss" if pnl < 0 else "neutral"

            if bar["high"] >= target_price:
                pnl = (target_price - entry_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "win" if pnl > 0 else "neutral"

        else:
            if bar["high"] >= stop_price:
                pnl = (entry_price - stop_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "loss" if pnl < 0 else "neutral"

            if bar["low"] <= target_price:
                pnl = (entry_price - target_price) / tick_size * tick_value
                pnl -= (2 * commission + slippage_ticks * tick_value)
                return "win" if pnl > 0 else "neutral"

    exit_price = bars_df.iloc[entry_idx + max_hold_bars]["close"]

    if direction == "long":
        pnl = (exit_price - entry_price) / tick_size * tick_value
    else:
        pnl = (entry_price - exit_price) / tick_size * tick_value

    pnl -= (2 * commission + 2 * slippage_ticks * tick_value)

    risk_amount = stop_distance / tick_size * tick_value

    if pnl > risk_amount * 0.3:
        return "win"
    if pnl < -risk_amount * 0.5:
        return "loss"
    return "neutral"


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

    for i in range(lookback, len(bars_df) - max_hold_bars):
        if i % 1000 == 0:
            pct = (i - lookback) / (len(bars_df) - max_hold_bars - lookback) * 100
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
