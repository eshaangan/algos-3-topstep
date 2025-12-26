"""
Offline backtest simulator.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decisions import decide_trades
from .fills import get_entry_exit, compute_cost_points, apply_forced_flatten
from .risk import RiskManager
from .metrics import compute_backtest_metrics
from core.instrument import InstrumentSpec


def run_backtest(
    events_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    primary_preds_df: pd.DataFrame,
    meta_preds_df: pd.DataFrame | None,
    execution_spec: dict,
    instrument_spec: InstrumentSpec,
    label_schema: dict,
    risk_cfg: dict,
    backtest_cfg: dict,
    bar_size: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run offline backtest over events with decisions, fills, and risk gates.
    """
    decisions_df = decide_trades(
        events_df=events_df,
        primary_preds=primary_preds_df,
        meta_preds=meta_preds_df,
        config=backtest_cfg,
        bars_df=bars_df,
    )

    decisions_df = decisions_df.sort_values("t0").reset_index(drop=True)

    contracts = int(backtest_cfg.get("sizing", {}).get("contracts", 1))
    flatten_time = backtest_cfg.get("session", {}).get(
        "flatten_time_chicago", None
    )
    contract_multiplier = float(
        instrument_spec.contract_multiplier_usd_per_point
    )
    cost_points = compute_cost_points(execution_spec, instrument_spec, bar_size)
    use_event_ret_net = bool(
        backtest_cfg.get("costs", {}).get("use_event_ret_net", False)
    )
    cost_mode_policy = (
        "event_ret_net_preferred" if use_event_ret_net else "price_minus_costs"
    )
    cost_mode = (label_schema or {}).get("cost_mode")
    if cost_mode not in ["net_in_events", "gross_in_events"]:
        raise ValueError("label_schema.cost_mode missing or invalid")

    risk_mgr = RiskManager(risk_cfg)

    trades = []
    equity_rows = []
    equity = risk_mgr.equity

    for _, row in decisions_df.iterrows():
        event_id = row["event_id"]
        accept = bool(row["accept"])
        reason = row["decision_reason"]
        side = row.get("side", 1)
        try:
            side = int(side)
        except Exception:
            side = 1
        if side == 0:
            side = 1
        side = 1 if side > 0 else -1

        entry_ts = None
        exit_ts = None
        entry_px = None
        exit_px = None
        pnl_points = None
        pnl_usd = None
        costs_usd = None
        exit_reason = None
        liquidation_reason = None
        exit_source = None
        executed = False

        if accept:
            fill = get_entry_exit(row, bars_df, execution_spec)
            fill = apply_forced_flatten(
                fill, bars_df, flatten_time_chicago=flatten_time
            )

            can_trade, risk_reason = risk_mgr.can_trade(fill.entry_ts)
            if not can_trade:
                reason = risk_reason
            else:
                entry_ts = fill.entry_ts
                exit_ts = fill.exit_ts
                entry_px = fill.entry_px
                exit_px = fill.exit_px
                exit_reason = fill.exit_reason

                bars_idx = bars_df.index
                if bars_idx.tz is not None:
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.tz_localize(bars_idx.tz)
                    else:
                        entry_ts = entry_ts.tz_convert(bars_idx.tz)
                    if exit_ts.tzinfo is None:
                        exit_ts = exit_ts.tz_localize(bars_idx.tz)
                    else:
                        exit_ts = exit_ts.tz_convert(bars_idx.tz)
                else:
                    if entry_ts.tzinfo is not None:
                        entry_ts = entry_ts.tz_convert(None)
                    if exit_ts.tzinfo is not None:
                        exit_ts = exit_ts.tz_convert(None)

                entry_pos = bars_idx.searchsorted(entry_ts, side="left")
                exit_pos = bars_idx.searchsorted(exit_ts, side="right") - 1
                exit_pos = min(exit_pos, len(bars_idx) - 1)
                breach_reason = None
                for pos in range(entry_pos, exit_pos + 1):
                    ts = bars_idx[pos]
                    price = float(bars_df.iloc[pos]["close"])
                    unrealized_points = side * (price - entry_px)
                    equity_unrealized = (
                        risk_mgr.equity
                        + unrealized_points * contract_multiplier * contracts
                    )
                    breached, reason_breach = risk_mgr.check_breach(
                        ts, equity_unrealized
                    )
                    if breached:
                        exit_ts = ts
                        exit_px = price
                        exit_reason = "mtm_risk"
                        breach_reason = reason_breach
                        break

                if breach_reason == "risk_daily_loss":
                    liquidation_reason = "daily_loss_breach"
                elif breach_reason == "risk_drawdown":
                    liquidation_reason = "trailing_dd_breach"

                costs_usd = cost_points * contract_multiplier * contracts
                if cost_mode == "net_in_events" and exit_reason == "event_exit":
                    if "ret_net" not in row or pd.isna(row["ret_net"]):
                        raise ValueError(
                            "label_schema.cost_mode=net_in_events but ret_net missing"
                        )
                    # ret_net from events is computed for LONG position
                    # For SHORT, flip the sign: side=-1 gives negative of long return
                    gross_points = side * (float(row["ret_net"]) + cost_points)
                    pnl_points = gross_points
                    trade_cost_mode = "event_ret_net"
                else:
                    gross_points = side * (exit_px - entry_px)
                    pnl_points = gross_points
                    trade_cost_mode = "price_minus_costs"

                pnl_usd = (
                    pnl_points * contract_multiplier * contracts - costs_usd
                )

                if liquidation_reason:
                    exit_source = "mtm_risk"
                elif exit_reason == "event_exit":
                    exit_source = "barrier"
                elif exit_reason == "forced_flatten":
                    exit_source = "forced_flatten"
                else:
                    exit_source = "other"

                risk_mgr.record_trade(entry_ts, exit_ts, pnl_usd)
                equity = risk_mgr.equity
                executed = True

                equity_rows.append(
                    {
                        "timestamp": exit_ts,
                        "equity": equity,
                        "pnl_usd": pnl_usd,
                    }
                )

        trades.append(
            {
                "event_id": event_id,
                "side": side,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "pnl_points": pnl_points,
                "pnl_usd": pnl_usd,
                "costs_usd": costs_usd,
                "executed": executed,
                "reason_skipped": "" if executed else reason,
                "exit_reason": exit_reason,
                "exit_source": exit_source,
                "liquidation_reason": liquidation_reason,
                "p_primary": row.get("p_primary"),
                "p_meta": row.get("p_meta"),
                "cost_mode": trade_cost_mode if executed else None,
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_backtest_metrics(trades_df, equity_df)
    if "cost_mode" in trades_df.columns:
        modes = trades_df.loc[trades_df["executed"], "cost_mode"].unique()
        metrics["cost_mode"] = (
            modes[0] if len(modes) == 1 else "mixed"
        )
        metrics["cost_mode_policy"] = cost_mode_policy

    return trades_df, equity_df, metrics
