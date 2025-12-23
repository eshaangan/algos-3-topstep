"""
Feature schema module for V3 pipeline.

Writes feature schema artifacts and computes schema hashes for reproducibility.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Literal

from .registry import FeatureSpec, filter_registry_for_bar_size

logger = logging.getLogger(__name__)


def compute_schema_hash(
    feature_columns: List[str],
    registry: List[FeatureSpec],
) -> str:
    """
    Compute deterministic hash of feature schema.

    Hash includes:
    - Feature column names in order
    - Feature specs (lookback, etc.)

    Args:
        feature_columns: List of feature column names in order
        registry: List of FeatureSpec for these features

    Returns:
        SHA256 hash (first 12 chars)
    """
    # Build deterministic dict
    schema_dict = {
        "feature_columns": feature_columns,
        "feature_specs": [
            {
                "name": spec.name,
                "lookback_bars": spec.lookback_bars,
                "uses_rolling_stats": spec.uses_rolling_stats,
                "requires_scaling": spec.requires_scaling,
                "fit_on_train_only": spec.fit_on_train_only,
                "bar_sizes_supported": spec.bar_sizes_supported,
            }
            for spec in registry
        ],
    }

    # Serialize to JSON with sorted keys for determinism
    schema_json = json.dumps(schema_dict, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    return hash_hex[:12]  # First 12 chars


def write_feature_schema(
    output_path: Path,
    feature_columns: List[str],
    registry: List[FeatureSpec],
    bar_size: Literal["1m", "5m"],
    config: dict,
    code_version: str = "1.0.0",
    config_hash: str = None,
) -> str:
    """
    Write feature schema JSON file.

    Args:
        output_path: Path to write feature_schema.json
        feature_columns: List of feature column names in order
        registry: List of FeatureSpec
        bar_size: Bar size this schema applies to
        config: Features config dict
        code_version: Version of feature code
        config_hash: Hash of features.yaml config (optional)

    Returns:
        Schema hash (for run manifest)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute schema hash
    schema_hash = compute_schema_hash(feature_columns, registry)

    # Build schema dict
    schema = {
        "schema_version": "1.0.0",
        "schema_hash": schema_hash,
        "code_version": code_version,
        "config_hash": config_hash,
        "bar_size": bar_size,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_specs": [
            {
                "name": spec.name,
                "lookback_bars": spec.lookback_bars,
                "uses_rolling_stats": spec.uses_rolling_stats,
                "requires_scaling": spec.requires_scaling,
                "fit_on_train_only": spec.fit_on_train_only,
                "bar_sizes_supported": spec.bar_sizes_supported,
                "description": spec.description,
            }
            for spec in registry
        ],
        "config_snapshot": config,
    }

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote feature schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")

    return schema_hash


def load_feature_schema(schema_path: Path) -> dict:
    """
    Load feature schema from JSON file.

    Args:
        schema_path: Path to feature_schema.json

    Returns:
        Schema dict
    """
    with open(schema_path, "r") as f:
        schema = json.load(f)

    return schema


def validate_features_match_schema(
    features_df,
    schema: dict,
) -> None:
    """
    Validate that features DataFrame matches schema.

    Args:
        features_df: Features DataFrame
        schema: Schema dict (from load_feature_schema)

    Raises:
        ValueError: If columns don't match schema
    """
    expected_cols = schema["feature_columns"]
    actual_cols = list(features_df.columns)

    if expected_cols != actual_cols:
        raise ValueError(
            f"Feature columns mismatch!\n"
            f"Expected: {expected_cols}\n"
            f"Got: {actual_cols}"
        )

    logger.info(f"Features match schema (hash: {schema['schema_hash']})")
