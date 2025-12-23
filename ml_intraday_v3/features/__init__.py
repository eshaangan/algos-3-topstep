"""
Feature engineering modules for V3 pipeline.

Exports:
- build_features: Main feature computation function
- get_feature_registry: Get ordered feature registry
- write_feature_schema: Write schema artifacts
- compute_schema_hash: Compute schema hash
"""

from .build import build_features
from .registry import (
    get_feature_registry,
    filter_registry_for_bar_size,
    FeatureSpec,
)
from .schema import (
    write_feature_schema,
    load_feature_schema,
    compute_schema_hash,
    validate_features_match_schema,
)

__all__ = [
    "build_features",
    "get_feature_registry",
    "filter_registry_for_bar_size",
    "FeatureSpec",
    "write_feature_schema",
    "load_feature_schema",
    "compute_schema_hash",
    "validate_features_match_schema",
]
