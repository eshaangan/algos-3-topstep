"""
Leakage checks for CV splits.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _shift_by_bars(
    bars_index: pd.Index, timestamp: pd.Timestamp, embargo_bars: int
) -> pd.Timestamp | None:
    end_pos = bars_index.searchsorted(timestamp, side="left")
    if end_pos >= len(bars_index):
        return None
    embargo_end_pos = min(end_pos + int(embargo_bars), len(bars_index) - 1)
    return bars_index[embargo_end_pos]


def _extract_splits(cv_data: dict) -> Dict[str, List[dict]]:
    splits = {}
    if "purged_kfold" in cv_data:
        splits["purged_kfold"] = cv_data.get("purged_kfold", [])
    if "cpcv" in cv_data:
        splits["cpcv"] = cv_data.get("cpcv", [])
    return splits


def check_train_test_disjoint(cv_data: dict) -> dict:
    splits_by_kind = _extract_splits(cv_data)
    if not splits_by_kind:
        return {"status": "SKIP", "reason": "no_splits"}

    violations = 0
    detail = {}
    for kind, splits in splits_by_kind.items():
        kind_violations = 0
        for split in splits:
            train = set(split.get("train_event_ids", []))
            test = set(split.get("test_event_ids", []))
            inter = train.intersection(test)
            if inter:
                kind_violations += len(inter)
        detail[kind] = {"overlap_count": int(kind_violations)}
        violations += kind_violations

    status = "PASS" if violations == 0 else "FAIL"
    return {
        "status": status,
        "details": detail,
        "total_overlap_count": int(violations),
        "definition": "train_event_ids ∩ test_event_ids == ∅",
    }


def check_purge_integrity(
    events_df: pd.DataFrame, cv_data: dict
) -> dict:
    splits_by_kind = _extract_splits(cv_data)
    if not splits_by_kind:
        return {"status": "SKIP", "reason": "no_splits"}

    df = events_df.copy()
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])
    df = df.set_index("event_id", drop=False)

    total_violations = 0
    detail = {}
    for kind, splits in splits_by_kind.items():
        kind_violations = 0
        for split in splits:
            train_ids = split.get("train_event_ids", [])
            if not train_ids:
                continue
            test_interval = split.get("test_interval", {})
            if not test_interval:
                continue
            test_start = pd.Timestamp(test_interval["start"])
            test_end = pd.Timestamp(test_interval["end"])

            train = df.loc[df.index.intersection(train_ids)]
            overlap = (train["t0"] <= test_end) & (train["t1"] >= test_start)
            kind_violations += int(overlap.sum())
        detail[kind] = {"overlap_count": int(kind_violations)}
        total_violations += kind_violations

    status = "PASS" if total_violations == 0 else "FAIL"
    return {
        "status": status,
        "details": detail,
        "total_overlap_count": int(total_violations),
        "definition": "(t0_train <= test_end) & (t1_train >= test_start)",
    }


def check_embargo_integrity(
    events_df: pd.DataFrame, bars_index: pd.Index, cv_data: dict
) -> dict:
    splits_by_kind = _extract_splits(cv_data)
    if not splits_by_kind:
        return {"status": "SKIP", "reason": "no_splits"}

    df = events_df.copy()
    df["t0"] = pd.to_datetime(df["t0"])
    df = df.set_index("event_id", drop=False)

    total_violations = 0
    detail = {}
    for kind, splits in splits_by_kind.items():
        kind_violations = 0
        for split in splits:
            train_ids = split.get("train_event_ids", [])
            test_interval = split.get("test_interval", {})
            params = split.get("params", {})
            embargo_bars = int(params.get("embargo_bars", 0))
            if embargo_bars <= 0 or not train_ids or not test_interval:
                continue
            test_end = pd.Timestamp(test_interval["end"])
            embargo_end = _shift_by_bars(bars_index, test_end, embargo_bars)
            if embargo_end is None:
                continue
            train = df.loc[df.index.intersection(train_ids)]
            mask = (train["t0"] > test_end) & (train["t0"] <= embargo_end)
            kind_violations += int(mask.sum())
        detail[kind] = {"embargo_violations": int(kind_violations)}
        total_violations += kind_violations

    status = "PASS" if total_violations == 0 else "FAIL"
    return {
        "status": status,
        "details": detail,
        "total_overlap_count": int(total_violations),
        "definition": "t0_train > test_end & t0_train <= embargo_end",
    }
