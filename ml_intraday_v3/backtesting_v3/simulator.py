"""
Offline backtest simulator.
"""

import json
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .decisions import decide_trades
from .fills import get_entry_exit, compute_cost_points, apply_forced_flatten
from .risk import RiskManager
from .metrics import compute_backtest_metrics
from ml_intraday_v3.core.instrument import InstrumentSpec


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
    # Filter out events with NaN ret_net (vertical exits without stop/target hit)
    # These are incomplete events that can't be properly backtested
    cost_mode = (label_schema or {}).get("cost_mode")
    if cost_mode == "net_in_events" and "ret_net" in events_df.columns:
        n_before = len(events_df)
        events_df = events_df[events_df["ret_net"].notna()].copy()
        n_filtered = n_before - len(events_df)
        if n_filtered > 0:
            # Also filter predictions to match
            valid_ids = set(events_df["event_id"])
            primary_preds_df = primary_preds_df[
                primary_preds_df["event_id"].isin(valid_ids)
            ].copy()
            if meta_preds_df is not None:
                meta_preds_df = meta_preds_df[
                    meta_preds_df["event_id"].isin(valid_ids)
                ].copy()

    decisions_df = decide_trades(
        events_df=events_df,
        primary_preds=primary_preds_df,
        meta_preds=meta_preds_df,
        config=backtest_cfg,
        bars_df=bars_df,
    )

    decisions_df = decisions_df.sort_values("t0").reset_index(drop=True)

    sizing_cfg = backtest_cfg.get("sizing", {}) or {}
    base_contracts = int(sizing_cfg.get("contracts", 1))
    flatten_time = backtest_cfg.get("session", {}).get(
        "flatten_time_chicago", None
    )
    trade_start_time = backtest_cfg.get("session", {}).get("trade_start_time_chicago")
    trade_end_time = backtest_cfg.get("session", {}).get("trade_end_time_chicago")
    trade_tz = ZoneInfo("America/Chicago")
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

    # Position tracking for concurrent position limit
    max_concurrent_positions = int(backtest_cfg.get("sizing", {}).get("max_concurrent_positions", 1))
    open_positions = []  # List of (entry_ts, exit_ts) tuples

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
            contracts = base_contracts
            dyn_cfg = (sizing_cfg.get("dynamic_contracts", {}) or {})
            if dyn_cfg.get("enabled", False):
                sigma = row.get("sigma")
                if sigma is not None and not pd.isna(sigma):
                    try:
                        sigma_val = float(sigma)
                    except Exception:
                        sigma_val = None
                    if sigma_val is not None:
                        rules = dyn_cfg.get("sigma_thresholds", []) or []
                        for rule in rules:
                            try:
                                max_sigma = float(rule.get("max_sigma"))
                                rule_contracts = int(rule.get("contracts"))
                            except Exception:
                                continue
                            if sigma_val <= max_sigma:
                                contracts = rule_contracts
                                break
            contracts = max(0, int(contracts))
            if contracts <= 0:
                accept = False
                reason = "dynamic_contracts"

            fill = get_entry_exit(row, bars_df, execution_spec)
            fill = apply_forced_flatten(
                fill, bars_df, flatten_time_chicago=flatten_time
            )

            # Optional: restrict entries to a specific intraday window (Chicago time).
            can_trade = True
            if trade_start_time and trade_end_time and fill.entry_ts is not None:
                entry_ts = pd.Timestamp(fill.entry_ts)
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize("UTC")
                entry_ts_local = entry_ts.tz_convert(trade_tz)

                start_t = dt_time.fromisoformat(trade_start_time)
                end_t = dt_time.fromisoformat(trade_end_time)
                now_t = entry_ts_local.to_pydatetime().time()
                in_window = (
                    start_t <= now_t < end_t
                    if start_t <= end_t
                    else (now_t >= start_t or now_t < end_t)
                )
                if not in_window:
                    reason = "time_filter"
                    can_trade = False

            if can_trade:
                can_trade, risk_reason = risk_mgr.can_trade(fill.entry_ts)
                if not can_trade:
                    reason = risk_reason
            else:
                # Check concurrent position limit
                # Clean up positions that have closed before this entry time
                open_positions = [(e, x) for e, x in open_positions if x > fill.entry_ts]

                if len(open_positions) >= max_concurrent_positions:
                    can_trade = False
                    reason = "max_concurrent_positions"

            if can_trade:
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
                    bar = bars_df.iloc[pos]

                    # Conservative intrabar bounds:
                    # - For LONG: best = high, worst = low
                    # - For SHORT: best = low, worst = high
                    high = float(bar.get("high", bar.get("close")))
                    low = float(bar.get("low", bar.get("close")))
                    if side > 0:
                        price_best = high
                        price_worst = low
                    else:
                        price_best = low
                        price_worst = high

                    equity_best = risk_mgr.equity + side * (price_best - entry_px) * contract_multiplier * contracts
                    equity_worst = risk_mgr.equity + side * (price_worst - entry_px) * contract_multiplier * contracts

                    breached, reason_breach = risk_mgr.check_breach(
                        ts, equity_worst, equity_best=equity_best
                    )
                    if breached:
                        exit_ts = ts
                        exit_px = price_worst
                        exit_reason = (
                            "mtm_risk_soft"
                            if reason_breach in {"risk_daily_loss_soft", "risk_drawdown_soft"}
                            else "mtm_risk"
                        )
                        breach_reason = reason_breach
                        break

                if breach_reason == "risk_daily_loss":
                    liquidation_reason = "daily_loss_breach"
                elif breach_reason == "risk_drawdown":
                    liquidation_reason = "trailing_dd_breach"

                # If labels encode PnL directly in events (ret_net), prefer that.
                #
                # IMPORTANT:
                # In our triple-barrier implementation, `events.ret_net` is already:
                # - side-adjusted (signed by event side)
                # - net of costs if labeling.triple_barrier.account_for_costs = true
                #
                # Therefore we MUST NOT multiply by `side` again or subtract costs again.
                # We still compute `costs_usd` for reporting, but we don't subtract it.
                costs_usd = cost_points * contract_multiplier * contracts
                if cost_mode == "net_in_events" and exit_reason == "event_exit":
                    if "ret_net" not in row or pd.isna(row["ret_net"]):
                        raise ValueError(
                            "label_schema.cost_mode=net_in_events but ret_net missing"
                        )
                    pnl_points = float(row["ret_net"])
                    pnl_usd = pnl_points * contract_multiplier * contracts
                    trade_cost_mode = "event_ret_net"
                else:
                    gross_points = side * (exit_px - entry_px)
                    pnl_points = gross_points
                    pnl_usd = pnl_points * contract_multiplier * contracts - costs_usd
                    trade_cost_mode = "price_minus_costs"

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

                # Add to open positions for concurrent tracking
                open_positions.append((entry_ts, exit_ts))

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
