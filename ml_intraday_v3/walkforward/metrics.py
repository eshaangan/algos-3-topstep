"""
Walk-forward metrics aggregation.
"""

from __future__ import annotations

from typing import Dict, List


def aggregate_window_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n_windows": 0}

    keys = set()
    for row in rows:
        keys.update(row.keys())

    agg = {"n_windows": len(rows)}
    for key in sorted(keys):
        if key in ["window_id", "status", "reason"]:
            continue
        vals = []
        for row in rows:
            val = row.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        if vals:
            agg[key] = sum(vals) / len(vals)
    return agg
