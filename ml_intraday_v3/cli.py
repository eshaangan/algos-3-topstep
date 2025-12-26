"""
CLI entrypoint for V3 pipeline.

Commands:
- build-data: Build canonical data for all bar sizes
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

import yaml
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import (
    load_raw_data,
    standardize_ohlcv,
    build_roll_schedule,
    apply_roll_schedule,
    write_roll_schedule,
    reindex_to_grid,
    resample_1m_to_5m,
    add_session_features,
    run_qa_checks,
    SessionConfig,
    QAViolationError,
)
from run_manifest import write_multibar_run_manifest, hash_content
from features import (
    build_features,
    get_feature_registry,
    filter_registry_for_bar_size,
    write_feature_schema,
    compute_schema_hash,
)
from labels import (
    generate_events,
    apply_triplebarrier,
    write_label_schema,
)
from weights import (
    map_event_intervals_to_index,
    compute_concurrency,
    compute_uniqueness_weights,
    compute_magnitude_weights,
    write_weight_schema,
)
from validation import (
    build_purged_kfold_splits,
    build_cpcv_paths,
    write_cv_schema,
)
from training import train_on_splits
from backtesting_v3 import run_backtest, write_backtest_schema
from experiments import run_experiments
from audit import run_audit
from walkforward import run_walkforward
from core.instrument import (
    load_instrument_from_execution_spec,
    validate_risk_config_no_instrument_economics,
)
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_data_command(args):
    """
    Build canonical data for all configured bar sizes.

    This command:
    1. Loads raw data
    2. Standardizes to OHLCV format
    3. Builds contract roll schedule
    4. Applies roll schedule
    5. Reindexes to grid (for each bar size)
    6. Resamples 1m to 5m (if needed)
    7. Adds session features
    8. Runs QA checks
    9. Writes artifacts per bar size
    10. Writes run manifest
    """
    logger.info("=" * 80)
    logger.info("V3 DATA PIPELINE - BUILD DATA")
    logger.info("=" * 80)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Determine run_id
    if args.run_id:
        run_id = args.run_id
    else:
        # Auto-generate run_id with timestamp
        run_id = f"v3_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Run ID: {run_id}")

    # Determine output directory
    if args.out:
        run_dir = Path(args.out)
    else:
        run_dir = Path("runs") / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # Get bar sizes from config
    canonical_bar_size = config.get("canonical_bar_size", "1m")
    bar_sizes = config.get(
        "bar_sizes_to_write",
        config.get("reindexing", {}).get("bar_sizes", ["1m", "5m"]),
    )
    logger.info(f"Canonical bar size: {canonical_bar_size}")
    logger.info(f"Bar sizes to process: {bar_sizes}")

    if canonical_bar_size not in {"1m", "5m"}:
        raise ValueError(f"Unsupported canonical_bar_size: {canonical_bar_size}")
    if canonical_bar_size == "5m":
        invalid_sizes = [bs for bs in bar_sizes if bs != "5m"]
        if invalid_sizes:
            raise ValueError(
                "Canonical bar size is 5m. Upsampling to 1m is not supported "
                f"by default. Remove {invalid_sizes} from bar_sizes_to_write."
            )

    # ------------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading raw data")
    logger.info("-" * 80)

    raw_data_config = config.get("raw_data", {})
    ingestion_config = config.get("ingestion", {})
    input_path = Path(
        ingestion_config.get("input_path", raw_data_config.get("input_path", ""))
    )

    df_raw = load_raw_data(
        input_path=input_path,
        input_format=ingestion_config.get(
            "format", raw_data_config.get("input_format", "hdf5")
        ),
        timestamp_column=ingestion_config.get(
            "timestamp_col",
            ingestion_config.get(
                "timestamp_column", raw_data_config.get("timestamp_column")
            ),
        ),
        required_columns=raw_data_config.get("required_columns"),
        hdf_key=ingestion_config.get("hdf_key"),
        filter_cfg=config.get("filtering"),
        symbol_column="symbol",
    )

    # ------------------------------------------------------------------------
    # 2. Standardize OHLCV
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Standardizing OHLCV data")
    logger.info("-" * 80)

    df_std = standardize_ohlcv(df_raw)

    # ------------------------------------------------------------------------
    # 3. Build roll schedule
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Building contract roll schedule")
    logger.info("-" * 80)

    continuization_config = config.get("continuization", {})
    roll_schedule = build_roll_schedule(
        df_std,
        mode=continuization_config.get("mode", "already_continuous"),
        roll_schedule_path=continuization_config.get("roll_schedule_path"),
    )

    # ------------------------------------------------------------------------
    # 4. Apply roll schedule
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Applying roll schedule")
    logger.info("-" * 80)

    df_continuous = apply_roll_schedule(
        df_std,
        roll_schedule,
        roll_day_policy=continuization_config.get("roll_day_policy", "exclude"),
        mode=continuization_config.get("mode", "already_continuous"),
    )

    # ------------------------------------------------------------------------
    # Process each bar size
    # ------------------------------------------------------------------------
    per_bar_artifacts = {}

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        # Create bar_size output directory
        bar_dir = run_dir / f"bar_size={bar_size}"
        bar_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------------
        # 5. Reindex to grid OR resample
        # --------------------------------------------------------------------
        logger.info("")
        logger.info(f"Step 5: Reindexing/resampling to {bar_size} grid")
        logger.info("-" * 80)

        reindex_config = config.get("reindexing", {})
        resample_policy = reindex_config.get(
            "resample_policy", "build_1m_resample_5m"
        )
        session_config = reindex_config.get("session_labeling", {})
        session_defs = session_config.get("sessions", [])
        grid_mode = reindex_config.get("grid_mode", "full_range")
        session_grid = reindex_config.get("session_grid", "rth")
        exclude_weekends = bool(reindex_config.get("exclude_weekends", True))
        day_selection_mode = reindex_config.get(
            "day_selection_mode", "all_days_in_range"
        )
        min_rows_per_day = int(reindex_config.get("min_rows_per_day", 1))
        session_tz = session_config.get("timezone", "America/Chicago")

        if bar_size == "1m":
            if canonical_bar_size != "1m":
                raise ValueError(
                    "Cannot build 1m bars from canonical 5m input. "
                    "Upsampling is not supported by default."
                )
            # Reindex to 1m grid
            df_bars, reindex_metadata = reindex_to_grid(
                df_continuous,
                bar_size="1m",
                missing_fill_mode=reindex_config.get("missing_bars", {}).get(
                    "missing_fill_mode", "nan"
                ),
                forward_fill_max_consecutive=reindex_config.get("missing_bars", {}).get(
                    "forward_fill_max_consecutive", 0
                ),
                add_synthetic_flag=reindex_config.get("missing_bars", {}).get(
                    "add_synthetic_flag", True
                ),
                grid_mode=grid_mode,
                session_grid=session_grid,
                session_timezone=session_tz,
                sessions=session_defs,
                exclude_weekends=exclude_weekends,
                day_selection_mode=day_selection_mode,
                min_rows_per_day=min_rows_per_day,
                drop_sparse_days=reindex_config.get("drop_sparse_days", False),
                min_day_coverage_pct=reindex_config.get("min_day_coverage_pct", 0.90),
                coverage_session=reindex_config.get("coverage_session"),
            )

            # Store 1m data for potential 5m resampling
            df_1m = df_bars.copy()

        elif bar_size == "5m":
            if canonical_bar_size == "5m":
                resample_policy = "native"
            if resample_policy == "build_1m_resample_5m":
                # Resample from 1m
                logger.info("Resampling from 1m to 5m")

                # First ensure we have 1m data
                if "df_1m" not in locals():
                    logger.error("1m data not available for resampling to 5m")
                    logger.error("Ensure 1m is processed before 5m")
                    sys.exit(1)

                df_bars = resample_1m_to_5m(df_1m)

                # Reindex metadata for 5m is derived
                reindex_metadata = {
                    "total_bars": len(df_bars),
                    "original_bars": len(df_bars),
                    "synthetic_bars": 0,
                    "synthetic_pct": 0.0,
                    "missing_pct_per_day": {},
                    "max_gap_bars": 0,
                    "fill_method": "resampled_from_1m",
                }

            else:  # native
                # Reindex to 5m grid directly
                df_bars, reindex_metadata = reindex_to_grid(
                    df_continuous,
                    bar_size="5m",
                    missing_fill_mode=reindex_config.get("missing_bars", {}).get(
                        "missing_fill_mode", "nan"
                    ),
                    forward_fill_max_consecutive=reindex_config.get("missing_bars", {}).get(
                        "forward_fill_max_consecutive", 0
                    ),
                    add_synthetic_flag=reindex_config.get("missing_bars", {}).get(
                        "add_synthetic_flag", True
                    ),
                    grid_mode=grid_mode,
                    session_grid=session_grid,
                    session_timezone=session_tz,
                    sessions=session_defs,
                    exclude_weekends=exclude_weekends,
                    day_selection_mode=day_selection_mode,
                    min_rows_per_day=min_rows_per_day,
                    drop_sparse_days=reindex_config.get("drop_sparse_days", False),
                    min_day_coverage_pct=reindex_config.get("min_day_coverage_pct", 0.90),
                    coverage_session=reindex_config.get("coverage_session"),
                )

        else:
            logger.error(f"Unsupported bar_size: {bar_size}")
            sys.exit(1)

        # --------------------------------------------------------------------
        # 6. Add session features
        # --------------------------------------------------------------------
        logger.info("")
        logger.info("Step 6: Adding session features")
        logger.info("-" * 80)

        session_config = reindex_config.get("session_labeling", {})
        sessions = None

        if "sessions" in session_config:
            sessions = [
                SessionConfig(
                    name=s["name"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                )
                for s in session_config["sessions"]
            ]

        df_bars = add_session_features(
            df_bars,
            session_timezone=session_config.get("timezone", "America/Chicago"),
            sessions=sessions,
        )

        # --------------------------------------------------------------------
        # 7. Run QA checks
        # --------------------------------------------------------------------
        logger.info("")
        logger.info("Step 7: Running QA checks")
        logger.info("-" * 80)

        qa_config = config.get("qa", {})

        try:
            qa_report = run_qa_checks(
                df_bars,
                checks=qa_config.get("checks"),
                thresholds=qa_config.get("thresholds"),
                qa_fail_fast=qa_config.get("qa_fail_fast", True),
            )
        except QAViolationError as e:
            logger.error("")
            logger.error("=" * 80)
            logger.error("QA CHECKS FAILED - PIPELINE HALTED")
            logger.error("=" * 80)
            logger.error(str(e))
            logger.error("")
            logger.error("To disable fail-fast mode (NOT RECOMMENDED), set:")
            logger.error("  qa_fail_fast: false")
            logger.error("in configs/data.yaml")
            logger.error("")
            sys.exit(1)

        # --------------------------------------------------------------------
        # 8. Write artifacts
        # --------------------------------------------------------------------
        logger.info("")
        logger.info("Step 8: Writing artifacts")
        logger.info("-" * 80)

        # Write bars (try pyarrow, fallback to fastparquet if not available)
        bars_path = bar_dir / "bars.parquet"
        try:
            df_bars.to_parquet(bars_path, engine="pyarrow", compression="snappy")
        except ImportError:
            logger.warning("pyarrow not available, using default parquet engine")
            df_bars.to_parquet(bars_path)
        logger.info(f"Wrote bars to {bars_path}")

        # Write QA report
        qa_report_path = bar_dir / "qa_report.json"
        qa_report.to_json(qa_report_path)

        # Write roll schedule (same for all bar sizes, but write per bar_size for consistency)
        roll_schedule_path = bar_dir / "roll_schedule.csv"
        write_roll_schedule(roll_schedule, roll_schedule_path)

        # Write data metadata
        data_metadata = {
            "bar_size": bar_size,
            "total_bars": len(df_bars),
            "start_date": str(df_bars.index.min()),
            "end_date": str(df_bars.index.max()),
            "columns": list(df_bars.columns),
            "reindex_metadata": reindex_metadata,
            "qa_passed": qa_report.passed,
        }

        metadata_path = bar_dir / "data_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(data_metadata, f, indent=2)

        logger.info(f"Wrote data metadata to {metadata_path}")

        # Track artifact hashes for manifest
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            else:
                return obj

        qa_checks_native = convert_types(qa_report.checks)

        per_bar_artifacts[bar_size] = {
            "data_metadata": hash_content(json.dumps(data_metadata, sort_keys=True)),
            "qa_report": hash_content(json.dumps(qa_checks_native, sort_keys=True)),
            "roll_schedule": hash_content(
                roll_schedule.to_dataframe().to_json(orient="records")
            ),
        }

    # ------------------------------------------------------------------------
    # 9. Write run manifest
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("Step 9: Writing run manifest")
    logger.info("-" * 80)

    config_dir = config_path.parent
    manifest_path = write_multibar_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        bar_sizes=bar_sizes,
        config_dir=config_dir,
        per_bar_artifacts=per_bar_artifacts,
        repo_path=Path.cwd(),
        metadata={
            "stage": "data",
            "config_path": str(config_path),
        },
    )

    logger.info(f"Wrote run manifest to {manifest_path}")

    # ------------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(f"  - {run_dir}/bar_size={bar_size}/bars.parquet")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/qa_report.json")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/roll_schedule.csv")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/data_metadata.json")
    logger.info(f"  - {run_dir}/run_manifest.json")
    logger.info("")


def build_features_command(args):
    """
    Build features for all bar sizes in a run directory.

    This command:
    1. Loads bars.parquet for each bar size
    2. Computes features causally (no lookahead)
    3. Writes features.parquet and feature_schema.json
    4. Updates run manifest with feature artifacts
    """
    logger.info("=" * 80)
    logger.info("V3 FEATURE PIPELINE - BUILD FEATURES")
    logger.info("=" * 80)

    # Get run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    logger.info(f"Run directory: {run_dir}")

    # Load features config
    features_config_path = Path(args.features_config)
    if not features_config_path.exists():
        logger.error(f"Features config not found: {features_config_path}")
        sys.exit(1)

    features_config = load_config(features_config_path)
    logger.info(f"Loaded features config from {features_config_path}")

    # Compute config hash for schema
    with open(features_config_path, "r") as f:
        config_content = f.read()
    features_config_hash = hash_content(config_content)

    # Load existing run manifest (if exists)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        # No manifest, discover bar_size directories
        bar_size_dirs = [d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("bar_size=")]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    if not bar_sizes:
        logger.error("No bar_size directories found in run directory")
        sys.exit(1)

    # Process each bar size
    per_bar_feature_artifacts = {}

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        bar_dir = run_dir / f"bar_size={bar_size}"
        if not bar_dir.exists():
            logger.warning(f"Bar directory not found: {bar_dir} - skipping")
            continue

        # Load bars
        bars_path = bar_dir / "bars.parquet"
        if not bars_path.exists():
            logger.error(f"bars.parquet not found: {bars_path}")
            sys.exit(1)

        logger.info(f"Loading bars from {bars_path}")
        bars_df = pd.read_parquet(bars_path)
        logger.info(f"Loaded {len(bars_df)} bars")

        # Build features
        logger.info("")
        logger.info("Building features")
        logger.info("-" * 80)

        features_df = build_features(
            bars_df=bars_df,
            bar_size=bar_size,
            config=features_config,
        )

        # Write features.parquet
        logger.info("")
        logger.info("Writing artifacts")
        logger.info("-" * 80)

        features_path = bar_dir / "features.parquet"
        try:
            features_df.to_parquet(features_path, engine="pyarrow", compression="snappy")
        except ImportError:
            logger.warning("pyarrow not available, using default parquet engine")
            features_df.to_parquet(features_path)
        logger.info(f"Wrote features to {features_path}")

        # Write feature_schema.json
        feature_columns = list(features_df.columns)
        full_registry = get_feature_registry(features_config)
        bar_registry = filter_registry_for_bar_size(full_registry, bar_size)

        schema_path = bar_dir / "feature_schema.json"
        schema_hash = write_feature_schema(
            output_path=schema_path,
            feature_columns=feature_columns,
            registry=bar_registry,
            bar_size=bar_size,
            config=features_config,
            code_version="1.0.0",
            config_hash=features_config_hash,
        )

        logger.info(f"Wrote feature schema to {schema_path}")

        # Track artifacts for manifest
        per_bar_feature_artifacts[bar_size] = {
            "features_path": str(features_path.relative_to(run_dir)),
            "feature_schema_path": str(schema_path.relative_to(run_dir)),
            "feature_schema_hash": schema_hash,
            "n_features": len(feature_columns),
        }

    # Update run manifest
    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        # Load and update existing manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Add feature artifacts per bar size
        if "per_bar_artifacts" not in manifest:
            manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_feature_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        # Add features config to configs if not present
        features_config_entry = {
            "name": "features",
            "path": str(features_config_path),
            "content_hash": features_config_hash,
            "content": features_config,
        }

        # Check if features config already in manifest
        config_names = [c["name"] for c in manifest.get("configs", [])]
        if "features" not in config_names:
            manifest["configs"].append(features_config_entry)

        # Write updated manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    # Done
    logger.info("")
    logger.info("=" * 80)
    logger.info("FEATURE PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(f"  - {run_dir}/bar_size={bar_size}/features.parquet")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/feature_schema.json")
    logger.info(f"  - {manifest_path} (updated)")
    logger.info("")


def build_labels_command(args):
    """
    Build triple-barrier labels for all bar sizes in a run directory.

    This command:
    1. Loads bars.parquet for each bar size
    2. Generates feasible events (every_bar baseline)
    3. Applies triple-barrier labeling
    4. Writes events.parquet and label_schema.json
    5. Updates run manifest with label artifacts
    """
    logger.info("=" * 80)
    logger.info("V3 LABEL PIPELINE - BUILD LABELS")
    logger.info("=" * 80)

    # Get run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    logger.info(f"Run directory: {run_dir}")

    # Load labeling config
    labeling_config_path = Path(args.labeling_config)
    if not labeling_config_path.exists():
        logger.error(f"Labeling config not found: {labeling_config_path}")
        sys.exit(1)

    labeling_config = load_config(labeling_config_path)
    logger.info(f"Loaded labeling config from {labeling_config_path}")

    # Load execution spec
    execution_spec_path = Path(args.execution_spec)
    if not execution_spec_path.exists():
        logger.error(f"Execution spec not found: {execution_spec_path}")
        sys.exit(1)

    execution_spec = load_config(execution_spec_path)
    logger.info(f"Loaded execution spec from {execution_spec_path}")

    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    logger.info("Loaded instrument spec from execution_spec")

    # Compute config hashes for schema/manifest
    with open(labeling_config_path, "r") as f:
        labeling_config_hash = hash_content(f.read())

    # Load existing run manifest (if exists)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        # No manifest, discover bar_size directories
        bar_size_dirs = [
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("bar_size=")
        ]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    if not bar_sizes:
        logger.error("No bar_size directories found in run directory")
        sys.exit(1)

    per_bar_label_artifacts = {}

    touch_ordering = execution_spec.get("fill_model", {}).get(
        "touch_ordering", "ohlc_path"
    )
    if touch_ordering == "ohlc_path":
        touch_def = "open->high->low->close (open takes precedence, then high, then low)"
    elif touch_ordering == "stop_first":
        touch_def = "if both hit in same bar, stop barrier takes precedence"
    elif touch_ordering == "target_first":
        touch_def = "if both hit in same bar, target barrier takes precedence"
    else:
        touch_def = f"unsupported touch_ordering: {touch_ordering}"

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        bar_dir = run_dir / f"bar_size={bar_size}"
        if not bar_dir.exists():
            logger.warning(f"Bar directory not found: {bar_dir} - skipping")
            continue

        bars_path = bar_dir / "bars.parquet"
        if not bars_path.exists():
            logger.error(f"bars.parquet not found: {bars_path}")
            sys.exit(1)

        logger.info(f"Loading bars from {bars_path}")
        bars_df = pd.read_parquet(bars_path)
        logger.info(f"Loaded {len(bars_df)} bars")

        logger.info("")
        logger.info("Generating events")
        logger.info("-" * 80)
        events_df = generate_events(
            bars_df=bars_df,
            bar_size=bar_size,
            labeling_config=labeling_config,
            execution_spec=execution_spec,
        )
        logger.info(f"Generated {len(events_df)} events")

        logger.info("")
        logger.info("Applying triple-barrier labeling")
        logger.info("-" * 80)
        labeled_df = apply_triplebarrier(
            bars_df=bars_df,
            events_df=events_df,
            bar_size=bar_size,
            labeling_config=labeling_config,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
        )
        logger.info(f"Labeled {len(labeled_df)} events")

        logger.info("")
        logger.info("Writing artifacts")
        logger.info("-" * 80)

        events_path = bar_dir / "events.parquet"
        try:
            labeled_df.to_parquet(events_path, engine="pyarrow", compression="snappy")
        except ImportError:
            logger.warning("pyarrow not available, using default parquet engine")
            labeled_df.to_parquet(events_path)
        logger.info(f"Wrote events to {events_path}")

        schema_path = bar_dir / "label_schema.json"
        schema_hash = write_label_schema(
            output_path=schema_path,
            columns=list(labeled_df.columns),
            bar_size=bar_size,
            labeling_config=labeling_config,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            touch_ordering_definition=touch_def,
            code_version="1.0.0",
        )
        logger.info(f"Wrote label schema to {schema_path}")

        per_bar_label_artifacts[bar_size] = {
            "events_path": str(events_path.relative_to(run_dir)),
            "label_schema_path": str(schema_path.relative_to(run_dir)),
            "label_schema_hash": schema_hash,
            "n_events": int(len(labeled_df)),
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_label_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )

        if "configs" not in manifest:
            manifest["configs"] = []

        labeling_config_entry = {
            "name": "labeling",
            "path": str(labeling_config_path),
            "content_hash": labeling_config_hash,
            "content": labeling_config,
        }
        config_names = [c.get("name") for c in manifest.get("configs", [])]
        if "labeling" not in config_names:
            manifest["configs"].append(labeling_config_entry)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    logger.info("")
    logger.info("=" * 80)
    logger.info("LABEL PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(f"  - {run_dir}/bar_size={bar_size}/events.parquet")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/label_schema.json")
    logger.info(f"  - {manifest_path} (updated)")
    logger.info("")


def build_weights_command(args):
    """
    Build sample weights for all bar sizes in a run directory.

    This command:
    1. Loads events.parquet and bars.parquet for each bar size
    2. Computes uniqueness weights via concurrency
    3. Optionally computes magnitude weights
    4. Writes weights.parquet and weight_schema.json
    5. Updates run manifest with weight artifacts
    """
    logger.info("=" * 80)
    logger.info("V3 WEIGHT PIPELINE - BUILD WEIGHTS")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    logger.info(f"Run directory: {run_dir}")

    labeling_config_path = Path(args.labeling_config)
    if not labeling_config_path.exists():
        logger.error(f"Labeling config not found: {labeling_config_path}")
        sys.exit(1)

    labeling_config = load_config(labeling_config_path)
    logger.info(f"Loaded labeling config from {labeling_config_path}")

    weights_cfg = labeling_config.get("sample_weights", {})
    uniqueness_cfg = weights_cfg.get("uniqueness", {})
    magnitude_cfg = weights_cfg.get("magnitude", {})
    formula_cfg = weights_cfg.get("weight_formula", {})

    with open(labeling_config_path, "r") as f:
        labeling_config_hash = hash_content(f.read())

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        bar_size_dirs = [
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("bar_size=")
        ]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    if not bar_sizes:
        logger.error("No bar_size directories found in run directory")
        sys.exit(1)

    per_bar_weight_artifacts = {}

    uniqueness_enabled = bool(uniqueness_cfg.get("enabled", True))
    uniqueness_method = uniqueness_cfg.get("method", "lopez_de_prado")
    if uniqueness_method not in ["lopez_de_prado", "simple_overlap"]:
        raise ValueError(f"Unsupported uniqueness method: {uniqueness_method}")

    magnitude_enabled = bool(magnitude_cfg.get("enabled", False))
    clip_quantile = magnitude_cfg.get("clip_quantile", None)
    clip_quantiles = (0.01, 0.99)
    if clip_quantile is not None:
        clip_quantiles = (1.0 - float(clip_quantile), float(clip_quantile))

    uniq_exp = float(formula_cfg.get("uniqueness_exponent", 1.0))
    mag_exp = float(formula_cfg.get("magnitude_exponent", 0.0))

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        bar_dir = run_dir / f"bar_size={bar_size}"
        if not bar_dir.exists():
            logger.warning(f"Bar directory not found: {bar_dir} - skipping")
            continue

        events_path = bar_dir / "events.parquet"
        if not events_path.exists():
            logger.error(f"events.parquet not found: {events_path}")
            sys.exit(1)

        bars_path = bar_dir / "bars.parquet"
        if not bars_path.exists():
            logger.error(f"bars.parquet not found: {bars_path}")
            sys.exit(1)

        logger.info(f"Loading events from {events_path}")
        events_df = pd.read_parquet(events_path)
        logger.info(f"Loaded {len(events_df)} events")

        if events_df.empty:
            logger.warning("No events found - skipping weights")
            continue

        events_df = events_df.sort_values("event_id").reset_index(drop=True)

        logger.info(f"Loading bars from {bars_path}")
        bars_df = pd.read_parquet(bars_path)
        logger.info(f"Loaded {len(bars_df)} bars")

        logger.info("")
        logger.info("Computing uniqueness weights")
        logger.info("-" * 80)

        start_idx, end_idx = map_event_intervals_to_index(
            events_df=events_df, index=bars_df.index
        )
        valid_mask = (start_idx >= 0) & (end_idx >= 0) & (end_idx >= start_idx)
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            logger.warning(
                f"Skipping {n_invalid} events with invalid t0/t1 alignment"
            )

        events_valid = events_df[valid_mask].reset_index(drop=True)
        start_valid = start_idx[valid_mask]
        end_valid = end_idx[valid_mask]

        if events_valid.empty:
            logger.warning("No valid events after alignment - skipping weights")
            continue

        if uniqueness_enabled:
            concurrency = compute_concurrency(
                index_len=len(bars_df.index),
                start_idx=start_valid,
                end_idx=end_valid,
            )
            zero_mask = (concurrency == 0).astype(int)
            zero_prefix = np.concatenate(
                [[0], np.cumsum(zero_mask, dtype=int)]
            )
            zero_counts = zero_prefix[end_valid + 1] - zero_prefix[start_valid]
            if np.any(zero_counts > 0):
                raise ValueError(
                    "Found c(t)=0 inside one or more event spans"
                )

            inv_conc = np.zeros_like(concurrency, dtype=float)
            inv_conc[concurrency > 0] = 1.0 / concurrency[concurrency > 0]
            inv_prefix = np.concatenate([[0.0], np.cumsum(inv_conc)])

            w_uniqueness = compute_uniqueness_weights(
                start_idx=start_valid,
                end_idx=end_valid,
                inv_conc_prefix=inv_prefix,
            )
        else:
            logger.warning("Uniqueness weights disabled - using 1.0")
            w_uniqueness = np.ones(len(events_valid), dtype=float)

        logger.info("")
        logger.info("Computing magnitude weights")
        logger.info("-" * 80)

        if magnitude_enabled:
            w_magnitude = compute_magnitude_weights(
                events_df=events_valid,
                method="abs_ret_net",
                clip_quantiles=clip_quantiles,
            ).to_numpy()
        else:
            w_magnitude = np.ones(len(events_valid), dtype=float)

        w_final = (w_uniqueness ** uniq_exp) * (w_magnitude ** mag_exp)

        weights_df = pd.DataFrame(
            {
                "event_id": events_valid["event_id"].to_numpy(),
                "t0": events_valid["t0"].to_numpy(),
                "t1": events_valid["t1"].to_numpy(),
                "w_uniqueness": w_uniqueness,
                "w_magnitude": w_magnitude,
                "w_final": w_final,
            }
        )

        weights_df = weights_df[
            [
                "event_id",
                "t0",
                "t1",
                "w_uniqueness",
                "w_magnitude",
                "w_final",
            ]
        ]

        logger.info("")
        logger.info("Writing artifacts")
        logger.info("-" * 80)

        weights_path = bar_dir / "weights.parquet"
        try:
            weights_df.to_parquet(
                weights_path, engine="pyarrow", compression="snappy"
            )
        except ImportError:
            logger.warning("pyarrow not available, using default parquet engine")
            weights_df.to_parquet(weights_path)
        logger.info(f"Wrote weights to {weights_path}")

        schema_path = bar_dir / "weight_schema.json"
        schema_hash = write_weight_schema(
            output_path=schema_path,
            columns=list(weights_df.columns),
            config_snapshot=weights_cfg,
            code_version="1.0.0",
            notes=f"uniqueness={uniqueness_method}, magnitude_enabled={magnitude_enabled}",
        )
        logger.info(f"Wrote weight schema to {schema_path}")

        per_bar_weight_artifacts[bar_size] = {
            "weights_path": str(weights_path.relative_to(run_dir)),
            "weight_schema_path": str(schema_path.relative_to(run_dir)),
            "weight_schema_hash": schema_hash,
            "n_weighted_events": int(len(weights_df)),
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_weight_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )

        if "configs" not in manifest:
            manifest["configs"] = []

        config_names = [c.get("name") for c in manifest.get("configs", [])]
        if "labeling" not in config_names:
            manifest["configs"].append(
                {
                    "name": "labeling",
                    "path": str(labeling_config_path),
                    "content_hash": labeling_config_hash,
                    "content": labeling_config,
                }
            )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    logger.info("")
    logger.info("=" * 80)
    logger.info("WEIGHT PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(f"  - {run_dir}/bar_size={bar_size}/weights.parquet")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/weight_schema.json")
    logger.info(f"  - {manifest_path} (updated)")
    logger.info("")


def build_cv_command(args):
    """
    Build leakage-safe CV splits for all bar sizes in a run directory.

    This command:
    1. Loads events.parquet and bars.parquet for each bar size
    2. Builds purged K-Fold splits with embargo
    3. Optionally builds CPCV paths
    4. Writes cv_splits.json and cv_schema.json
    5. Updates run manifest with CV artifacts
    """
    logger.info("=" * 80)
    logger.info("V3 VALIDATION PIPELINE - BUILD CV")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    logger.info(f"Run directory: {run_dir}")

    validation_config_path = Path(args.validation_config)
    if not validation_config_path.exists():
        logger.error(f"Validation config not found: {validation_config_path}")
        sys.exit(1)

    validation_config = load_config(validation_config_path)
    logger.info(f"Loaded validation config from {validation_config_path}")

    with open(validation_config_path, "r") as f:
        validation_config_hash = hash_content(f.read())

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        bar_size_dirs = [
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("bar_size=")
        ]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    if not bar_sizes:
        logger.error("No bar_size directories found in run directory")
        sys.exit(1)

    purged_cfg = validation_config.get("purged_cv", {})
    test_set_policy = purged_cfg.get("test_set_policy", "equal_samples")
    if test_set_policy != "equal_samples":
        raise ValueError(
            f"Unsupported test_set_policy: {test_set_policy} "
            "(only 'equal_samples' is supported)"
        )
    embargo_cfg = purged_cfg.get("embargo", {})
    embargo_enabled = bool(embargo_cfg.get("enabled", True))
    embargo_map = embargo_cfg.get("embargo_bars", {})
    n_splits = int(purged_cfg.get("n_splits", 5))

    cpcv_cfg = validation_config.get("cpcv", {})
    cpcv_enabled = bool(cpcv_cfg.get("enabled", False))
    cpcv_groups = int(cpcv_cfg.get("n_groups", n_splits))
    cpcv_test_groups = int(cpcv_cfg.get("test_groups", 2))
    cpcv_max_paths = cpcv_cfg.get("max_paths", None)
    selection = cpcv_cfg.get("path_selection", "lexicographic")
    random_state = cpcv_cfg.get("random_state", None)

    # Validate selection strategy
    valid_selections = ["lexicographic", "balanced", "random"]
    if selection not in valid_selections and cpcv_enabled:
        raise ValueError(
            f"Unsupported cpcv.path_selection: {selection} "
            f"(must be one of {valid_selections})"
        )

    per_bar_cv_artifacts = {}

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        bar_dir = run_dir / f"bar_size={bar_size}"
        if not bar_dir.exists():
            logger.warning(f"Bar directory not found: {bar_dir} - skipping")
            continue

        events_path = bar_dir / "events.parquet"
        if not events_path.exists():
            logger.error(f"events.parquet not found: {events_path}")
            sys.exit(1)

        bars_path = bar_dir / "bars.parquet"
        if not bars_path.exists():
            logger.error(f"bars.parquet not found: {bars_path}")
            sys.exit(1)

        logger.info(f"Loading events from {events_path}")
        events_df = pd.read_parquet(events_path)
        logger.info(f"Loaded {len(events_df)} events")

        logger.info(f"Loading bars from {bars_path}")
        bars_df = pd.read_parquet(bars_path)
        logger.info(f"Loaded {len(bars_df)} bars")

        embargo_bars = int(embargo_map.get(bar_size, 0)) if embargo_enabled else 0

        logger.info("")
        logger.info("Building purged K-Fold splits")
        logger.info("-" * 80)

        purged_splits = build_purged_kfold_splits(
            events_df=events_df,
            bars_index=bars_df.index,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
        )

        cpcv_paths = []
        if cpcv_enabled:
            logger.info("")
            logger.info("Building CPCV paths")
            logger.info("-" * 80)

            n_groups_config = cpcv_groups
            n_groups_used = cpcv_groups
            if cpcv_groups != n_splits:
                logger.warning(
                    "CPCV n_groups differs from purged_cv n_splits; using n_splits"
                )
                n_groups_used = n_splits

            logger.info(f"Path selection strategy: {selection}")
            if random_state is not None:
                logger.info(f"Random state: {random_state}")

            cpcv_paths = build_cpcv_paths(
                base_folds=purged_splits,
                events_df=events_df,
                bars_index=bars_df.index,
                K=n_groups_used,
                test_groups=cpcv_test_groups,
                embargo_bars=embargo_bars,
                max_paths=cpcv_max_paths,
                selection=selection,
                random_state=random_state,
            )

        cv_splits = {
            "bar_size": bar_size,
            "purged_kfold": purged_splits,
        }
        if cpcv_paths:
            cv_splits["cpcv"] = cpcv_paths

        cv_splits_path = bar_dir / "cv_splits.json"
        with open(cv_splits_path, "w") as f:
            json.dump(cv_splits, f, indent=2)
        logger.info(f"Wrote CV splits to {cv_splits_path}")

        summary = {
            "bar_size": bar_size,
            "embargo_bars": embargo_bars,
            "n_events": int(len(events_df)),
            "n_folds": int(len(purged_splits)),
            "n_paths": int(len(cpcv_paths)),
            "cv_kind": "cpcv" if cpcv_paths else "purged_kfold",
            "n_splits": n_splits,
            "test_groups": cpcv_test_groups if cpcv_paths else None,
            "max_paths": cpcv_max_paths if cpcv_paths else None,
            "selection": selection if cpcv_paths else None,
            "test_set_policy": test_set_policy,
            "cpcv_n_groups_config": (
                n_groups_config if cpcv_paths else None
            ),
            "cpcv_n_groups_used": (
                n_groups_used if cpcv_paths else None
            ),
        }

        cv_schema_path = bar_dir / "cv_schema.json"
        cv_schema_hash = write_cv_schema(
            output_path=cv_schema_path,
            config_snapshot=validation_config,
            summary=summary,
            code_version="1.0.0",
        )

        per_bar_cv_artifacts[bar_size] = {
            "cv_splits_path": str(cv_splits_path.relative_to(run_dir)),
            "cv_schema_path": str(cv_schema_path.relative_to(run_dir)),
            "cv_schema_hash": cv_schema_hash,
            "n_splits": int(len(purged_splits)),
            "cv_kind": summary["cv_kind"],
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_cv_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )

        if "configs" not in manifest:
            manifest["configs"] = []

        config_names = [c.get("name") for c in manifest.get("configs", [])]
        if "validation" not in config_names:
            manifest["configs"].append(
                {
                    "name": "validation",
                    "path": str(validation_config_path),
                    "content_hash": validation_config_hash,
                    "content": validation_config,
                }
            )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(f"  - {run_dir}/bar_size={bar_size}/cv_splits.json")
        logger.info(f"  - {run_dir}/bar_size={bar_size}/cv_schema.json")
    logger.info(f"  - {manifest_path} (updated)")
    logger.info("")


def build_train_command(args):
    """
    Train baseline model on CV splits for all bar sizes in a run directory.
    """
    logger.info("=" * 80)
    logger.info("V3 TRAINING PIPELINE - BUILD TRAIN")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    logger.info(f"Run directory: {run_dir}")

    training_config_path = Path(args.training_config)
    if not training_config_path.exists():
        logger.error(f"Training config not found: {training_config_path}")
        sys.exit(1)

    training_config = load_config(training_config_path)
    logger.info(f"Loaded training config from {training_config_path}")

    with open(training_config_path, "r") as f:
        training_config_hash = hash_content(f.read())

    cv_kind = args.cv_kind
    meta_enabled = bool(training_config.get("meta", {}).get("enabled", False))

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        bar_size_dirs = [
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("bar_size=")
        ]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    if not bar_sizes:
        logger.error("No bar_size directories found in run directory")
        sys.exit(1)

    per_bar_training_artifacts = {}

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        result = train_on_splits(
            run_dir=run_dir,
            bar_size=bar_size,
            training_config=training_config,
            cv_kind=cv_kind,
        )

        training_dir = result["training_dir"]
        schema_path = result["training_schema_path"]
        schema_hash = result["training_schema_hash"]
        n_splits = result["n_splits"]

        per_bar_training_artifacts[bar_size] = {
            "training_dir": str(training_dir.relative_to(run_dir)),
            "training_schema_path": str(schema_path.relative_to(run_dir)),
            "training_schema_hash": schema_hash,
            "cv_kind_trained": cv_kind,
            "n_splits_trained": int(n_splits),
            "meta_training_enabled": meta_enabled,
            "meta_training_dir": str(training_dir.relative_to(run_dir)),
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_training_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )

        if "configs" not in manifest:
            manifest["configs"] = []

        config_names = [c.get("name") for c in manifest.get("configs", [])]
        if "training" not in config_names:
            manifest["configs"].append(
                {
                    "name": "training",
                    "path": str(training_config_path),
                    "content_hash": training_config_hash,
                    "content": training_config,
                }
            )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar sizes: {bar_sizes}")
    logger.info("")
    logger.info("Artifacts written:")
    for bar_size in bar_sizes:
        logger.info(
            f"  - {run_dir}/bar_size={bar_size}/training/{cv_kind}/summary.json"
        )
        logger.info(
            f"  - {run_dir}/bar_size={bar_size}/training/{cv_kind}/training_schema.json"
        )
    logger.info(f"  - {manifest_path} (updated)")
    logger.info("")


def _resolve_training_dir(
    run_dir: Path, bar_size: str, cv_kind: str, training_dir_arg: str | None
) -> Path:
    if training_dir_arg:
        base = Path(training_dir_arg)
        if (base / f"bar_size={bar_size}").exists():
            return base / f"bar_size={bar_size}" / "training" / cv_kind
        if base.name == cv_kind:
            return base
        if base.name == "training":
            return base / cv_kind
        if "bar_size=" in base.parts:
            return base
    return run_dir / f"bar_size={bar_size}" / "training" / cv_kind


def build_backtest_command(args):
    """
    Run offline backtest using trained predictions and risk gates.
    """
    logger.info("=" * 80)
    logger.info("V3 BACKTEST PIPELINE - BUILD BACKTEST")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    backtest_config_path = Path(args.backtest_config)
    if not backtest_config_path.exists():
        logger.error(f"Backtest config not found: {backtest_config_path}")
        sys.exit(1)

    backtest_config = load_config(backtest_config_path)
    logger.info(f"Loaded backtest config from {backtest_config_path}")

    execution_spec_path = Path(args.execution_spec)
    if not execution_spec_path.exists():
        logger.error(f"Execution spec not found: {execution_spec_path}")
        sys.exit(1)
    execution_spec = load_config(execution_spec_path)

    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    logger.info("Loaded instrument spec from execution_spec")

    risk_config_path = Path(args.risk_config)
    if not risk_config_path.exists():
        logger.error(f"Risk config not found: {risk_config_path}")
        sys.exit(1)
    risk_config = load_config(risk_config_path)
    validate_risk_config_no_instrument_economics(risk_config)

    cv_kind = args.cv_kind
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        bar_sizes = manifest.get("bar_sizes", ["1m", "5m"])
        logger.info(f"Found existing manifest with bar_sizes: {bar_sizes}")
    else:
        bar_size_dirs = [
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("bar_size=")
        ]
        bar_sizes = [d.replace("bar_size=", "") for d in bar_size_dirs]
        logger.info(f"Discovered bar_sizes from directories: {bar_sizes}")

    per_bar_backtest_artifacts = {}

    for bar_size in bar_sizes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing bar_size: {bar_size}")
        logger.info("=" * 80)

        bar_dir = run_dir / f"bar_size={bar_size}"
        events_path = bar_dir / "events.parquet"
        bars_path = bar_dir / "bars.parquet"
        cv_path = bar_dir / "cv_splits.json"
        label_schema_path = bar_dir / "label_schema.json"

        if (
            not events_path.exists()
            or not bars_path.exists()
            or not cv_path.exists()
            or not label_schema_path.exists()
        ):
            logger.error(f"Missing events/bars/cv_splits for {bar_size}")
            sys.exit(1)

        events_df = pd.read_parquet(events_path)
        bars_df = pd.read_parquet(bars_path)
        with open(cv_path, "r") as f:
            cv_data = json.load(f)
        with open(label_schema_path, "r") as f:
            label_schema = json.load(f)
        try:
            event_policy = (
                (label_schema.get("config_snapshot", {}) or {})
                .get("primary_labeling", {})
                .get("event_policy")
            )
        except Exception:
            event_policy = None
        if event_policy == "trend_scanning":
            logger.warning(
                "label_schema.primary_labeling.event_policy=trend_scanning uses forward returns to set `side`; "
                "if backtest PnL uses events['side'], results will be lookahead-biased. "
                "Use event_policy='cusum' for unbiased evaluation unless `side` comes from a t0-available signal/model."
            )
        label_cost_mode = label_schema.get("cost_mode")
        if label_cost_mode not in ["net_in_events", "gross_in_events"]:
            raise ValueError("label_schema.cost_mode missing or invalid")
        if label_cost_mode == "net_in_events" and "ret_net" not in events_df.columns:
            raise ValueError(
                "label_schema.cost_mode=net_in_events but events.parquet is missing ret_net"
            )
        pnl_mode = (
            "use_events_ret_net"
            if label_cost_mode == "net_in_events"
            else "compute_from_prices_then_subtract_costs"
        )

        if cv_kind == "purged_kfold":
            splits = cv_data.get("purged_kfold", [])
            split_id_key = "fold"
            prefix = "fold"
        elif cv_kind == "cpcv":
            splits = cv_data.get("cpcv", [])
            split_id_key = "path_id"
            prefix = "path"
        else:
            raise ValueError(f"Unsupported cv_kind: {cv_kind}")

        training_dir = _resolve_training_dir(
            run_dir, bar_size, cv_kind, args.training_dir
        )
        if not training_dir.exists():
            logger.error(f"Training dir not found: {training_dir}")
            sys.exit(1)

        backtest_dir = bar_dir / "backtests" / cv_kind
        backtest_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for split in splits:
            split_id = split.get(split_id_key)
            split_dir = training_dir / f"{prefix}_{split_id}"
            primary_path = split_dir / "preds.parquet"
            meta_path = split_dir / "meta_preds.parquet"
            if not primary_path.exists():
                raise FileNotFoundError(f"Missing preds: {primary_path}")

            primary_preds = pd.read_parquet(primary_path)
            meta_preds = None
            if backtest_config.get("decision", {}).get("use_meta", False):
                if not meta_path.exists():
                    raise FileNotFoundError(f"Missing meta preds: {meta_path}")
                meta_preds = pd.read_parquet(meta_path)

            test_ids = split.get("test_event_ids", [])
            test_events = events_df[events_df["event_id"].isin(test_ids)].copy()
            test_events = test_events.sort_values("t0")
            if label_cost_mode == "net_in_events":
                missing_ret_net = (
                    test_events["ret_net"].isna().sum()
                    if "ret_net" in test_events.columns
                    else len(test_events)
                )
                if missing_ret_net > 0:
                    logger.warning(
                        "Dropping %d test events with missing ret_net (cost_mode=net_in_events)",
                        int(missing_ret_net),
                    )
                    test_events = test_events.dropna(subset=["ret_net"])

            split_out_dir = backtest_dir / f"{prefix}_{split_id}"
            split_out_dir.mkdir(parents=True, exist_ok=True)

            trades_df, equity_df, metrics = run_backtest(
                events_df=test_events,
                bars_df=bars_df,
                primary_preds_df=primary_preds,
                meta_preds_df=meta_preds,
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_config,
                backtest_cfg=backtest_config,
                bar_size=bar_size,
            )

            if backtest_config.get("outputs", {}).get("write_trade_log", True):
                trades_df.to_parquet(split_out_dir / "trades.parquet")
            if backtest_config.get("outputs", {}).get(
                "write_equity_curve", True
            ):
                equity_df.to_parquet(split_out_dir / "equity.parquet")

            with open(split_out_dir / "backtest_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            summary_rows.append(
                {"split_id": split_id, **metrics}
            )

        summary = {
            "cv_kind": cv_kind,
            "n_splits": len(summary_rows),
            "metrics_by_split": summary_rows,
        }
        with open(backtest_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        training_schema_path = training_dir / "training_schema.json"
        if training_schema_path.exists():
            with open(training_schema_path, "r") as f:
                training_schema = json.load(f)
            training_schema_hash = training_schema.get("schema_hash", "")
        else:
            training_schema_hash = ""

        backtest_schema_path = backtest_dir / "backtest_schema.json"
        cost_mode_policy = (
            "event_ret_net_preferred"
            if backtest_config.get("costs", {}).get("use_event_ret_net", False)
            else "price_minus_costs"
        )
        backtest_schema_hash = write_backtest_schema(
            output_path=backtest_schema_path,
            backtest_config=backtest_config,
            execution_spec=execution_spec,
            risk_config=risk_config,
            training_schema_hash=training_schema_hash,
            cv_kind=cv_kind,
            n_splits=len(summary_rows),
            cost_mode_policy=cost_mode_policy,
            pnl_mode=pnl_mode,
            code_version="1.0.0",
        )

        per_bar_backtest_artifacts[bar_size] = {
            "backtest_dir": str(backtest_dir.relative_to(run_dir)),
            "backtest_schema_path": str(backtest_schema_path.relative_to(run_dir)),
            "backtest_schema_hash": backtest_schema_hash,
            "n_splits_backtested": len(summary_rows),
            "thresholds": backtest_config.get("decision", {}),
        }

    logger.info("")
    logger.info("=" * 80)
    logger.info("Updating run manifest")
    logger.info("-" * 80)

    if manifest_path.exists():
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        if "configs" not in manifest:
            manifest["configs"] = []
        config_names = [c.get("name") for c in manifest.get("configs", [])]
        for bar_size, artifacts in per_bar_backtest_artifacts.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated run manifest: {manifest_path}")
    else:
        logger.warning("No existing manifest found - artifacts written but manifest not updated")
        logger.warning("Run build-data first to create initial manifest")

    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST PIPELINE COMPLETE")
    logger.info("=" * 80)


def run_experiments_command(args):
    """
    Run experiment grid (train/backtest) and diagnostics.
    """
    logger.info("=" * 80)
    logger.info("V3 EXPERIMENT RUNNER")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    grid_config_path = Path(args.grid_config)
    if not grid_config_path.exists():
        logger.error(f"Grid config not found: {grid_config_path}")
        sys.exit(1)

    result = run_experiments(run_dir=run_dir, grid_config_path=grid_config_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Experiment ID: {result['exp_id']}")
    logger.info(f"Experiment dir: {result['exp_dir']}")
    logger.info(f"Variants: {result['n_variants']}")
    logger.info("")


def run_audit_command(args):
    """
    Run audit harness for a run directory.
    """
    logger.info("=" * 80)
    logger.info("V3 AUDIT HARNESS")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    strict = str(args.strict).lower() in ["1", "true", "yes", "y"]

    result = run_audit(run_dir=run_dir, strict=strict)

    logger.info("")
    logger.info("=" * 80)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Bar sizes: {result['bar_sizes']}")
    logger.info("")


def run_walkforward_command(args):
    """
    Run walk-forward evaluation and write live bundles.
    """
    logger.info("=" * 80)
    logger.info("V3 WALK-FORWARD EVALUATION")
    logger.info("=" * 80)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    wf_config_path = Path(args.walkforward_config)
    if not wf_config_path.exists():
        logger.error(f"Walkforward config not found: {wf_config_path}")
        sys.exit(1)

    result = run_walkforward(run_dir=run_dir, walkforward_config_path=wf_config_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("WALK-FORWARD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Walkforward dir: {result['walkforward_dir']}")
    logger.info("")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="V3 ML Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build-data command
    build_data_parser = subparsers.add_parser(
        "build-data",
        help="Build canonical data for all bar sizes",
    )
    build_data_parser.add_argument(
        "--config",
        type=str,
        default="ml_intraday_v3/configs/data.yaml",
        help="Path to data config YAML file",
    )
    build_data_parser.add_argument(
        "--out",
        type=str,
        help="Output directory (default: runs/<run_id>)",
    )
    build_data_parser.add_argument(
        "--run-id",
        type=str,
        help="Run identifier (default: auto-generate with timestamp)",
    )
    build_data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )

    # build-features command
    build_features_parser = subparsers.add_parser(
        "build-features",
        help="Build features for all bar sizes in a run",
    )
    build_features_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_features_parser.add_argument(
        "--features-config",
        type=str,
        default="ml_intraday_v3/configs/features.yaml",
        help="Path to features config YAML file",
    )

    # build-labels command
    build_labels_parser = subparsers.add_parser(
        "build-labels",
        help="Build triple-barrier labels for all bar sizes in a run",
    )
    build_labels_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_labels_parser.add_argument(
        "--labeling-config",
        type=str,
        default="ml_intraday_v3/configs/labeling.yaml",
        help="Path to labeling config YAML file",
    )
    build_labels_parser.add_argument(
        "--execution-spec",
        type=str,
        default="ml_intraday_v3/configs/execution_spec.yaml",
        help="Path to execution spec YAML file",
    )

    # build-weights command
    build_weights_parser = subparsers.add_parser(
        "build-weights",
        help="Build sample weights for all bar sizes in a run",
    )
    build_weights_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_weights_parser.add_argument(
        "--labeling-config",
        type=str,
        default="ml_intraday_v3/configs/labeling.yaml",
        help="Path to labeling config YAML file",
    )

    # build-cv command
    build_cv_parser = subparsers.add_parser(
        "build-cv",
        help="Build leakage-safe CV splits for all bar sizes in a run",
    )
    build_cv_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_cv_parser.add_argument(
        "--validation-config",
        type=str,
        default="ml_intraday_v3/configs/validation.yaml",
        help="Path to validation config YAML file",
    )

    # build-train command
    build_train_parser = subparsers.add_parser(
        "build-train",
        help="Train baseline model on CV splits for all bar sizes in a run",
    )
    build_train_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_train_parser.add_argument(
        "--training-config",
        type=str,
        default="ml_intraday_v3/configs/training.yaml",
        help="Path to training config YAML file",
    )
    build_train_parser.add_argument(
        "--cv-kind",
        type=str,
        default="purged_kfold",
        choices=["purged_kfold", "cpcv"],
        help="Which CV splits to train on",
    )

    # build-backtest command
    build_backtest_parser = subparsers.add_parser(
        "build-backtest",
        help="Run offline backtest on CV test splits",
    )
    build_backtest_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    build_backtest_parser.add_argument(
        "--training-dir",
        type=str,
        help="Optional training base directory (defaults to run_dir)",
    )
    build_backtest_parser.add_argument(
        "--backtest-config",
        type=str,
        default="ml_intraday_v3/configs/backtest.yaml",
        help="Path to backtest config YAML file",
    )
    build_backtest_parser.add_argument(
        "--execution-spec",
        type=str,
        default="ml_intraday_v3/configs/execution_spec.yaml",
        help="Path to execution spec YAML file",
    )
    build_backtest_parser.add_argument(
        "--risk-config",
        type=str,
        default="ml_intraday_v3/configs/risk.yaml",
        help="Path to risk config YAML file",
    )
    build_backtest_parser.add_argument(
        "--cv-kind",
        type=str,
        default="purged_kfold",
        choices=["purged_kfold", "cpcv"],
        help="Which CV splits to backtest",
    )

    # run-experiments command
    run_experiments_parser = subparsers.add_parser(
        "run-experiments",
        help="Run experiment grid and diagnostics",
    )
    run_experiments_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    run_experiments_parser.add_argument(
        "--grid-config",
        type=str,
        default="ml_intraday_v3/configs/experiment_grid.yaml",
        help="Path to experiment grid YAML file",
    )

    # run-audit command
    run_audit_parser = subparsers.add_parser(
        "run-audit",
        help="Run audit harness for a run directory",
    )
    run_audit_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    run_audit_parser.add_argument(
        "--strict",
        type=str,
        default="false",
        help="If true, raise on any FAIL",
    )

    # run-walkforward command
    run_wf_parser = subparsers.add_parser(
        "run-walkforward",
        help="Run walk-forward evaluation",
    )
    run_wf_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/baseline_v3_001)",
    )
    run_wf_parser.add_argument(
        "--walkforward-config",
        type=str,
        default="ml_intraday_v3/configs/walkforward.yaml",
        help="Path to walkforward config YAML file",
    )

    args = parser.parse_args()

    if args.command == "build-data":
        build_data_command(args)
    elif args.command == "build-features":
        build_features_command(args)
    elif args.command == "build-labels":
        build_labels_command(args)
    elif args.command == "build-weights":
        build_weights_command(args)
    elif args.command == "build-cv":
        build_cv_command(args)
    elif args.command == "build-train":
        build_train_command(args)
    elif args.command == "build-backtest":
        build_backtest_command(args)
    elif args.command == "run-experiments":
        run_experiments_command(args)
    elif args.command == "run-audit":
        run_audit_command(args)
    elif args.command == "run-walkforward":
        run_walkforward_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
