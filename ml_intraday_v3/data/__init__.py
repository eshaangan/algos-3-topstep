"""
Data pipeline modules for V3.

This package contains all data ingestion, continuization, reindexing,
resampling, and QA modules for the V3 pipeline.
"""

from .ingest import load_raw_data, standardize_ohlcv
from .continuous import (
    build_roll_schedule,
    apply_roll_schedule,
    write_roll_schedule,
    load_roll_schedule,
    RollSchedule,
)
from .reindex import reindex_to_grid
from .resample import resample_1m_to_5m
from .session import add_session_features, SessionConfig
from .qa import run_qa_checks, QAReport, QAViolationError

__all__ = [
    "load_raw_data",
    "standardize_ohlcv",
    "build_roll_schedule",
    "apply_roll_schedule",
    "write_roll_schedule",
    "load_roll_schedule",
    "RollSchedule",
    "reindex_to_grid",
    "resample_1m_to_5m",
    "add_session_features",
    "SessionConfig",
    "run_qa_checks",
    "QAReport",
    "QAViolationError",
]
