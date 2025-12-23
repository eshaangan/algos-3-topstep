"""
Weight schema module for V3 pipeline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from run_manifest import hash_content

logger = logging.getLogger(__name__)


def compute_weight_schema_hash(schema_dict: Dict) -> str:
    """Compute deterministic hash of weight schema content."""
    schema_json = json.dumps(schema_dict, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_weight_schema(
    output_path: Path,
    columns: List[str],
    config_snapshot: dict,
    code_version: str = "1.0.0",
    notes: str = "",
) -> str:
    """
    Write weight schema JSON file.

    Includes formulas, normalization flags, and config snapshot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_hash = hash_content(config_snapshot)

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "n_columns": len(columns),
        "columns": columns,
        "formulas": {
            "concurrency": "c(t) = # active events overlapping t",
            "uniqueness": "u_i = mean_{t in [t0_i, t1_i]} 1 / c(t)",
            "magnitude": "w_mag = abs(ret_net|ret_gross|ret_points) with clipping",
            "final": "w_final = w_uniqueness * w_magnitude (if enabled)",
        },
        "normalization_applied": False,
        "notes": notes,
        "config_hash": config_hash,
        "config_snapshot": config_snapshot,
    }

    schema_hash = compute_weight_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote weight schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")

    return schema_hash
