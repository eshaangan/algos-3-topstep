"""
Backtest schema module for V3 pipeline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from run_manifest import hash_content

logger = logging.getLogger(__name__)


def compute_backtest_schema_hash(schema_dict: Dict) -> str:
    schema_json = json.dumps(schema_dict, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_backtest_schema(
    output_path: Path,
    backtest_config: dict,
    execution_spec: dict,
    risk_config: dict,
    training_schema_hash: str,
    cv_kind: str,
    n_splits: int,
    cost_mode_policy: str,
    code_version: str = "1.0.0",
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "backtest_config_hash": hash_content(backtest_config),
        "backtest_config": backtest_config,
        "execution_spec_hash": hash_content(execution_spec),
        "risk_config_hash": hash_content(risk_config),
        "training_schema_hash": training_schema_hash,
        "cv_kind": cv_kind,
        "n_splits": n_splits,
        "cost_mode_policy": cost_mode_policy,
    }

    schema_hash = compute_backtest_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote backtest schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")
    return schema_hash
