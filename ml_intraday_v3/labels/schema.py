"""
Label schema module for V3 pipeline.

Writes label schema artifacts and computes schema hashes for reproducibility.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from run_manifest import hash_content
from core.instrument import InstrumentSpec

logger = logging.getLogger(__name__)


def compute_label_schema_hash(schema_dict: Dict) -> str:
    """
    Compute deterministic hash of label schema content.

    Args:
        schema_dict: Schema dict (excluding schema_hash field)

    Returns:
        SHA256 hash (first 12 chars)
    """
    schema_json = json.dumps(schema_dict, sort_keys=True)
    hash_obj = hashlib.sha256(schema_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def write_label_schema(
    output_path: Path,
    columns: List[str],
    bar_size: str,
    labeling_config: dict,
    execution_spec: dict,
    instrument_spec: InstrumentSpec,
    touch_ordering_definition: str,
    code_version: str = "1.0.0",
) -> str:
    """
    Write label schema JSON file.

    Args:
        output_path: Path to write label_schema.json
        columns: List of label columns in order
        bar_size: Bar size this schema applies to
        labeling_config: Labeling config dict
        execution_spec: Execution spec dict
        touch_ordering_definition: Human-readable ordering definition
        code_version: Version of label code

    Returns:
        Schema hash (for run manifest)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    execution_spec_hash = hash_content(execution_spec)
    label_cfg = labeling_config.get("primary_labeling", {}).get("triple_barrier", {})
    label_encoding = label_cfg.get(
        "label_encoding", {"target_first": 1, "stop_first": -1, "vertical": 0}
    )

    schema = {
        "schema_version": "1.0.0",
        "code_version": code_version,
        "bar_size": bar_size,
        "n_columns": len(columns),
        "columns": columns,
        "cost_mode": "net_in_events" if "ret_net" in columns else "gross_in_events",
        "label_encoding": label_encoding,
        "touch_ordering": execution_spec.get("fill_model", {}).get(
            "touch_ordering", "ohlc_path"
        ),
        "touch_ordering_definition": touch_ordering_definition,
        "fill_price_model": execution_spec.get("fill_model", {}).get(
            "fill_price", "next_bar_open"
        ),
        "return_units": "price_points",
        "tick_size_points": instrument_spec.tick_size_points,
        "tick_value_usd": instrument_spec.tick_value_usd,
        "contract_multiplier_usd_per_point": instrument_spec.contract_multiplier_usd_per_point,
        "account_for_costs": bool(label_cfg.get("account_for_costs", False)),
        "execution_spec_hash": execution_spec_hash,
        "config_snapshot": labeling_config,
    }

    schema_hash = compute_label_schema_hash(schema)
    schema["schema_hash"] = schema_hash

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Wrote label schema to {output_path}")
    logger.info(f"Schema hash: {schema_hash}")

    return schema_hash
