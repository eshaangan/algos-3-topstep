"""
Backtest metrics computation.
"""

import numpy as np


def compute_backtest_metrics(trades_df, equity_df):
    executed = trades_df[trades_df["executed"]].copy()
    skipped = trades_df[~trades_df["executed"]]

    total_pnl = executed["pnl_usd"].sum() if not executed.empty else 0.0
    wins = executed[executed["pnl_usd"] > 0]
    losses = executed[executed["pnl_usd"] < 0]
    win_rate = float(len(wins) / len(executed)) if len(executed) else None
    profit_factor = (
        float(wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum()))
        if len(losses) and len(wins)
        else None
    )
    avg_trade = float(executed["pnl_usd"].mean()) if len(executed) else None

    max_dd = None
    if equity_df is not None and not equity_df.empty:
        eq = equity_df["equity"].to_numpy()
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        max_dd = float(dd.max()) if len(dd) else None

    mtm_liquidations = 0
    daily_loss_liq = 0
    trailing_dd_liq = 0
    if "liquidation_reason" in executed.columns:
        mtm_liquidations = int(executed["liquidation_reason"].notna().sum())
        daily_loss_liq = int(
            (executed["liquidation_reason"] == "daily_loss_breach").sum()
        )
        trailing_dd_liq = int(
            (executed["liquidation_reason"] == "trailing_dd_breach").sum()
        )

    return {
        "total_pnl_usd": float(total_pnl),
        "max_drawdown_usd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_usd": avg_trade,
        "trades_count": int(len(executed)),
        "skipped_count": int(len(skipped)),
        "mtm_liquidations": mtm_liquidations,
        "mtm_daily_loss_liquidations": daily_loss_liq,
        "mtm_trailing_dd_liquidations": trailing_dd_liq,
    }
