"""
Validation schema module for V3 pipeline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from run_manifest import hash_content

logger = logging.getLogger(__name__)


def compute_cv_schema_hash(schema_dict: Dict) -> str:
    """Compute deterministic hash of CV schema content."""
    schema_json = json.dumps(schema_dict, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_cv_schema(
    output_path: Path,
    config_snapshot: dict,
    summary: dict,
    code_version: str = "1.0.0",
) -> str:
    """
    Write CV schema JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_hash = hash_content(config_snapshot)

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "purge_definition": "remove any train event whose [t0,t1] overlaps test interval [test_start,test_end]",
        "embargo_definition": "remove train events with t0 in (test_end, embargo_end]",
        "config_hash": config_hash,
        "config_snapshot": config_snapshot,
        "summary": summary,
    }

    schema_hash = compute_cv_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote CV schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")

    return schema_hash
