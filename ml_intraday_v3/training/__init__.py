"""
Training modules for V3 pipeline (primary model only).
"""

from .dataset import build_event_dataset, build_meta_dataset
from .preprocess import FoldPreprocessor
from .train import train_on_splits
from .metrics import compute_metrics
from .schema import write_training_schema, compute_training_schema_hash

__all__ = [
    "build_event_dataset",
    "build_meta_dataset",
    "FoldPreprocessor",
    "train_on_splits",
    "compute_metrics",
    "write_training_schema",
    "compute_training_schema_hash",
]
