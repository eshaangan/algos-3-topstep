"""Pre-registered month/quarter-end stock-bond rebalancing research adapter.

Primary hypothesis
------------------
At the final trading day of a month, institutional target-allocation funds trade
against the month-to-date stock-minus-bond return spread. If equities have
outperformed bonds, the equity leg is SHORT; if bonds have outperformed, it is
LONG. The signal is fixed using data through the PRIOR close, preventing lookahead.

The exact test requires an equity close series and a Treasury-futures close series.
The repository currently commits session PnL tapes but not the Treasury history,
so ``--session-placebo`` tests only the weaker unconditional month-end hypothesis.
That placebo is diagnostic and must not be promoted as the literature strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_close(path: Path) -> pd.Series:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        frame = pd.read_hdf(path)
    else:
        frame = pd.read_csv(path)

    if isinstance(frame, pd.Series):
        series = frame
    else:
        ts_col = next((c for c in ("timestamp", "datetime", "date", "day") if c in frame.columns), None)
        close_col = next((c for c in ("close", "settle", "price") if c in frame.columns), None)
        if ts_col is None or close_col is None:
            raise ValueError(f"{path} needs timestamp/date and close/settle columns")
        idx = pd.to_datetime(frame[ts_col], utc=True, errors="raise")
        series = pd.Series(pd.to_numeric(frame[close_col], errors="raise").to_numpy(), index=idx)
    series.index = pd.to_datetime(series.index, utc=True)
    return series.sort_index().groupby(series.index.normalize()).last().dropna()


def build_rebalance_events(
    equity_close: pd.Series,
    bond_close: pd.Series,
    *,
    spread_threshold: float = 0.02,
) -> pd.DataFrame:
    aligned = pd.concat({"equity": equity_close, "bond": bond_close}, axis=1).dropna().sort_index()
    month = aligned.index.tz_localize(None).to_period("M")
    is_month_end = month != month.shift(-1)

    prior_eq = aligned["equity"].shift(1)
    prior_bd = aligned["bond"].shift(1)
    month_first_eq = aligned["equity"].groupby(month).transform("first")
    month_first_bd = aligned["bond"].groupby(month).transform("first")
    eq_mtd_prior = prior_eq / month_first_eq - 1.0
    bd_mtd_prior = prior_bd / month_first_bd - 1.0
    spread = eq_mtd_prior - bd_mtd_prior

    direction = pd.Series(0, index=aligned.index, dtype=int)
    direction[spread >= spread_threshold] = -1
    direction[spread <= -spread_threshold] = 1
    equity_day_return = aligned["equity"] / prior_eq - 1.0

    out = pd.DataFrame(
        {
            "day": aligned.index,
            "spread_prior_close": spread,
            "direction": direction,
            "equity_day_return": equity_day_return,
        }
    )
    out = out[is_month_end & (out["direction"] != 0)].copy()
    out["strategy_return"] = out["direction"] * out["equity_day_return"]
    out["quarter_end"] = out["day"].dt.month.isin((3, 6, 9, 12))
    return out.reset_index(drop=True)


def _stats(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"n": 0}
    r = events["strategy_return"].to_numpy(float)
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return {
        "n": int(len(r)),
        "mean_return": float(np.mean(r)),
        "win_rate": float(np.mean(r > 0)),
        "t_stat": float(np.mean(r) / (std / np.sqrt(len(r)))) if std > 0 else 0.0,
        "cumulative_simple_return": float(np.sum(r)),
    }


def summarize_returns(events: pd.DataFrame) -> dict:
    result = _stats(events)
    if not events.empty:
        result["quarter_end"] = _stats(events[events["quarter_end"]])
        result["non_quarter_end"] = _stats(events[~events["quarter_end"]])
    return result


def session_placebo(path: Path) -> dict:
    frame = pd.read_csv(path)
    mae_col = "mae_usd" if "mae_usd" in frame else "mae" if "mae" in frame else None
    frame["session_start"] = pd.to_datetime(frame["day"], errors="raise")
    frame["trade_date"] = frame["session_start"] + pd.Timedelta(days=1)
    frame = frame.sort_values("trade_date")
    month = frame["trade_date"].dt.to_period("M")
    month_end = month != month.shift(-1)
    first = month != month.shift(1)

    def stats(mask: pd.Series) -> dict:
        p = frame.loc[mask, "pnl"].to_numpy(float)
        std = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
        return {
            "n": int(len(p)),
            "mean_pnl": float(np.mean(p)) if len(p) else 0.0,
            "win_rate": float(np.mean(p > 0)) if len(p) else 0.0,
            "t_stat": float(np.mean(p) / (std / np.sqrt(len(p)))) if std > 0 else 0.0,
            "worst_pnl": float(np.min(p)) if len(p) else None,
            "worst_mae": float(frame.loc[mask, mae_col].min()) if mae_col and len(p) else None,
        }

    return {
        "warning": "Unconditional session-hold placebo; not the stock-bond directional strategy.",
        "month_end": stats(month_end),
        "first_trading_day": stats(first),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity")
    ap.add_argument("--bond")
    ap.add_argument("--spread-threshold", type=float, default=0.02)
    ap.add_argument("--session-placebo", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.session_placebo:
        result = session_placebo(args.session_placebo)
    else:
        if not args.equity or not args.bond:
            raise SystemExit("provide --equity and --bond, or --session-placebo")
        events = build_rebalance_events(
            load_close(Path(args.equity)),
            load_close(Path(args.bond)),
            spread_threshold=args.spread_threshold,
        )
        result = {"summary": summarize_returns(events), "events": events.to_dict("records")}
    print(json.dumps(result, indent=2, default=str))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
