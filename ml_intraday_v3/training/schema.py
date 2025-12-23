"""
Training schema module for V3 pipeline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from run_manifest import hash_content

logger = logging.getLogger(__name__)


def compute_training_schema_hash(schema_dict: Dict) -> str:
    """Compute deterministic hash of training schema content."""
    schema_json = json.dumps(schema_dict, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_training_schema(
    output_path: Path,
    training_config: dict,
    feature_schema_hash: str,
    label_schema_hash: str,
    weight_schema_hash: str,
    cv_schema_hash: str,
    model_kind: str,
    model_params: dict,
    seed: int,
    cv_kind: str,
    n_splits: int,
    meta_enabled: bool,
    meta_config: dict,
    code_version: str = "1.0.0",
) -> str:
    """Write training schema JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_hash = hash_content(training_config)

    meta_config_hash = hash_content(meta_config) if meta_config else ""

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "config_hash": config_hash,
        "training_config": training_config,
        "feature_schema_hash": feature_schema_hash,
        "label_schema_hash": label_schema_hash,
        "weight_schema_hash": weight_schema_hash,
        "cv_schema_hash": cv_schema_hash,
        "model_kind": model_kind,
        "model_params": model_params,
        "seed": seed,
        "cv_kind": cv_kind,
        "n_splits": n_splits,
        "meta_enabled": meta_enabled,
        "meta_config_hash": meta_config_hash,
        "meta_config": meta_config,
    }

    schema_hash = compute_training_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote training schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")

    return schema_hash
