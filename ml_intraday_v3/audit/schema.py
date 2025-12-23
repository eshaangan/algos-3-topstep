"""
Audit schema helper.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def compute_audit_schema_hash(schema: Dict) -> str:
    schema_json = json.dumps(schema, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_audit_schema(
    output_path: Path,
    checks: Dict,
    code_version: str = "1.0.0",
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "checks": checks,
    }

    schema_hash = compute_audit_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    return schema_hash
