"""
Aggregation helpers for experiment backtests.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


SUM_FIELDS = {"total_pnl_usd", "trades_count", "skipped_count"}
MAX_FIELDS = {"max_drawdown_usd"}


def _numeric_values(rows: Iterable[Dict], key: str) -> List[float]:
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return vals


def aggregate_split_metrics(split_rows: List[Dict]) -> Dict:
    """
    Aggregate backtest metrics across splits.

    Uses sum for totals, max for drawdown, and mean for other numeric metrics.
    """
    if not split_rows:
        return {"n_splits": 0}

    keys = set()
    for row in split_rows:
        keys.update(row.keys())

    agg = {"n_splits": len(split_rows)}
    for key in sorted(keys):
        if key == "split_id":
            continue
        vals = _numeric_values(split_rows, key)
        if not vals:
            continue
        if key in SUM_FIELDS:
            agg[key] = float(sum(vals))
        elif key in MAX_FIELDS:
            agg[key] = float(max(vals))
        else:
            agg[key] = float(sum(vals) / len(vals))
    return agg
