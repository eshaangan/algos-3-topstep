"""
Run manifest schema and persistence.

The RunManifest captures complete reproducibility metadata for every pipeline run:
- Git state (hash or dirty diff)
- All config snapshots
- Schema hashes for features/labels/splits
- Model/calibration IDs
- Execution spec + cost model hashes
- Timestamp and environment info

Every run writes a run_manifest.json to runs/<run_id>/run_manifest.json.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import platform
import sys


@dataclass
class GitState:
    """Git repository state at run time."""

    commit_hash: Optional[str] = None
    is_dirty: bool = False
    dirty_diff_hash: Optional[str] = None  # Hash of git diff if dirty
    branch: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ConfigSnapshot:
    """Snapshot of a single config file."""

    name: str  # e.g., "execution_spec", "data", "labeling"
    path: str  # Relative path to config file
    content_hash: str  # SHA256 hash of config content
    content: Dict[str, Any]  # Full config content


@dataclass
class SchemaHash:
    """Hash of a data schema (features, labels, CV splits, etc.)."""

    name: str  # e.g., "feature_schema", "label_schema", "cv_splits"
    hash: str  # SHA256 hash
    artifact_path: str  # Path to artifact file


@dataclass
class EnvironmentInfo:
    """Environment and dependency information."""

    python_version: str
    platform: str
    os_version: str
    packages: Dict[str, str] = field(default_factory=dict)  # package: version


@dataclass
class RunManifest:
    """
    Complete run manifest for reproducibility.

    This is the single source of truth for what was run, with what configs,
    on what data, at what time.

    NOTE: Supports both single bar_size (legacy) and multi-bar-size runs.
    For multi-bar-size runs, use bar_sizes list and per_bar_size_artifacts.
    """

    # Run identification
    run_id: str
    timestamp: str  # ISO 8601 format
    bar_size: str = ""  # Legacy: single bar size ("1m" or "5m"), deprecated in favor of bar_sizes
    bar_sizes: List[str] = field(default_factory=list)  # Multi-bar support: ["1m", "5m"]

    # Git state
    git_state: GitState = field(default_factory=GitState)

    # Config snapshots (all YAML + metrics_contract.json)
    configs: List[ConfigSnapshot] = field(default_factory=list)

    # Schema hashes (features, labels, CV splits)
    # Legacy: single list of schemas (deprecated for multi-bar runs)
    schemas: List[SchemaHash] = field(default_factory=list)

    # Per-bar-size artifact metadata (for multi-bar runs)
    # Dict[bar_size, Dict[schema_name, schema_hash]]
    # Example: {"1m": {"data_metadata": "hash123", "roll_schedule": "hash456"}, "5m": {...}}
    per_bar_size_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Model and calibration IDs (empty initially, populated during training)
    model_ids: Dict[str, str] = field(default_factory=dict)  # stage: model_id
    calibration_ids: Dict[str, str] = field(default_factory=dict)  # stage: cal_id

    # Execution spec hash (critical for label/backtest parity)
    execution_spec_hash: str = ""

    # Cost model hash (derived from execution_spec)
    cost_model_hash: str = ""

    # Environment
    environment: EnvironmentInfo = field(default_factory=lambda: EnvironmentInfo(
        python_version="",
        platform="",
        os_version="",
    ))

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Manifest version (increment if schema changes)
    manifest_version: str = "1.1.0"  # Bumped for multi-bar-size support


def get_git_state(repo_path: Path) -> GitState:
    """
    Capture current git state.

    Returns:
        GitState with commit hash, dirty status, and diff hash if dirty.
    """
    try:
        # Check if directory is a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return GitState(commit_hash=None, is_dirty=False, branch=None)

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else None

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

        # Get diff hash if dirty
        dirty_diff_hash = None
        if is_dirty:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                dirty_diff_hash = hashlib.sha256(
                    result.stdout.encode("utf-8")
                ).hexdigest()[:16]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = result.stdout.strip() if result.returncode == 0 else None

        # Get tags
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        tags = (
            result.stdout.strip().split("\n")
            if result.returncode == 0 and result.stdout.strip()
            else []
        )

        return GitState(
            commit_hash=commit_hash,
            is_dirty=is_dirty,
            dirty_diff_hash=dirty_diff_hash,
            branch=branch,
            tags=tags,
        )

    except Exception as e:
        # If git commands fail, return minimal state
        return GitState(commit_hash=None, is_dirty=False, branch=None)


def hash_content(content: Any) -> str:
    """
    Compute SHA256 hash of content (dict, string, or bytes).

    Args:
        content: Content to hash (dict will be JSON serialized)

    Returns:
        Hex string of SHA256 hash
    """
    if isinstance(content, dict):
        # Serialize dict deterministically
        content_str = json.dumps(content, sort_keys=True)
        content_bytes = content_str.encode("utf-8")
    elif isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")

    return hashlib.sha256(content_bytes).hexdigest()


def load_config_snapshot(config_path: Path, name: str) -> ConfigSnapshot:
    """
    Load a config file and create a snapshot with hash.

    Args:
        config_path: Path to YAML or JSON config file
        name: Config name (e.g., "execution_spec")

    Returns:
        ConfigSnapshot with content and hash
    """
    import yaml

    # Read file content
    with open(config_path, "r") as f:
        if config_path.suffix == ".json":
            content = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            content = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Compute hash
    content_hash = hash_content(content)

    return ConfigSnapshot(
        name=name,
        path=str(config_path),
        content_hash=content_hash,
        content=content,
    )


def get_environment_info(include_packages: bool = False) -> EnvironmentInfo:
    """
    Capture environment information.

    Args:
        include_packages: If True, include all installed package versions (slow)

    Returns:
        EnvironmentInfo with python version, platform, and optionally packages
    """
    env = EnvironmentInfo(
        python_version=sys.version,
        platform=platform.platform(),
        os_version=platform.version(),
    )

    if include_packages:
        try:
            import pkg_resources

            env.packages = {
                pkg.key: pkg.version for pkg in pkg_resources.working_set
            }
        except Exception:
            # If pkg_resources fails, skip package listing
            pass

    return env


def write_run_manifest(
    run_dir: Path,
    run_id: str,
    bar_size: str,
    config_dir: Path,
    feature_schema_hash: Optional[str] = None,
    feature_schema_path: Optional[str] = None,
    label_schema_hash: Optional[str] = None,
    label_schema_path: Optional[str] = None,
    cv_split_ids: Optional[str] = None,
    cv_split_path: Optional[str] = None,
    git_hash: Optional[str] = None,
    repo_path: Optional[Path] = None,
    include_packages: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a run manifest to disk.

    Args:
        run_dir: Run directory (e.g., runs/<run_id>/bar_size=1m/)
        run_id: Unique run identifier
        bar_size: Bar size ("1m" or "5m")
        config_dir: Path to configs directory
        feature_schema_hash: Optional hash of feature schema
        feature_schema_path: Optional path to feature schema artifact
        label_schema_hash: Optional hash of label schema
        label_schema_path: Optional path to label schema artifact
        cv_split_ids: Optional hash of CV split IDs
        cv_split_path: Optional path to CV splits artifact
        git_hash: Optional git commit hash (if not provided, will auto-detect)
        repo_path: Optional path to git repo (defaults to config_dir parent)
        include_packages: Whether to include full package list (slow)
        metadata: Optional additional metadata dict

    Returns:
        Path to written manifest file
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get git state
    if repo_path is None:
        repo_path = config_dir.parent.parent  # Go up from ml_intraday_v3/configs/

    if git_hash:
        # Use provided hash
        git_state = GitState(commit_hash=git_hash, is_dirty=False)
    else:
        # Auto-detect
        git_state = get_git_state(repo_path)

    # Load all config snapshots
    config_files = [
        ("execution_spec", "execution_spec.yaml"),
        ("data", "data.yaml"),
        ("labeling", "labeling.yaml"),
        ("validation", "validation.yaml"),
        ("risk", "risk.yaml"),
        ("retrain_policy", "retrain_policy.yaml"),
        ("metrics_contract", "metrics_contract.json"),
    ]

    configs = []
    execution_spec_hash = ""

    for name, filename in config_files:
        config_path = config_dir / filename
        if config_path.exists():
            snapshot = load_config_snapshot(config_path, name)
            configs.append(snapshot)

            # Capture execution spec hash separately (critical for parity)
            if name == "execution_spec":
                execution_spec_hash = snapshot.content_hash

    # Build schema hashes list
    schemas = []
    if feature_schema_hash and feature_schema_path:
        schemas.append(
            SchemaHash(
                name="feature_schema",
                hash=feature_schema_hash,
                artifact_path=feature_schema_path,
            )
        )

    if label_schema_hash and label_schema_path:
        schemas.append(
            SchemaHash(
                name="label_schema",
                hash=label_schema_hash,
                artifact_path=label_schema_path,
            )
        )

    if cv_split_ids and cv_split_path:
        schemas.append(
            SchemaHash(
                name="cv_splits",
                hash=cv_split_ids,
                artifact_path=cv_split_path,
            )
        )

    # Cost model hash (derived from execution_spec costs section)
    cost_model_hash = ""
    if execution_spec_hash:
        # Find execution_spec config
        exec_config = next(
            (c for c in configs if c.name == "execution_spec"), None
        )
        if exec_config and "costs" in exec_config.content:
            cost_model_hash = hash_content(exec_config.content["costs"])

    # Get environment info
    environment = get_environment_info(include_packages=include_packages)

    # Build manifest
    manifest = RunManifest(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        bar_size=bar_size,
        git_state=git_state,
        configs=configs,
        schemas=schemas,
        execution_spec_hash=execution_spec_hash,
        cost_model_hash=cost_model_hash,
        environment=environment,
        metadata=metadata or {},
    )

    # Write to disk
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=False)

    return manifest_path


def load_run_manifest(manifest_path: Path) -> RunManifest:
    """
    Load a run manifest from disk.

    Args:
        manifest_path: Path to run_manifest.json

    Returns:
        RunManifest object
    """
    with open(manifest_path, "r") as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
    data["git_state"] = GitState(**data["git_state"])
    data["configs"] = [ConfigSnapshot(**c) for c in data["configs"]]
    data["schemas"] = [SchemaHash(**s) for s in data["schemas"]]
    data["environment"] = EnvironmentInfo(**data["environment"])

    return RunManifest(**data)


def write_multibar_run_manifest(
    run_dir: Path,
    run_id: str,
    bar_sizes: List[str],
    config_dir: Path,
    per_bar_artifacts: Optional[Dict[str, Dict[str, str]]] = None,
    git_hash: Optional[str] = None,
    repo_path: Optional[Path] = None,
    include_packages: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a multi-bar-size run manifest to disk.

    This is the preferred method for V3 runs that process multiple bar sizes.
    Writes a single manifest at runs/<run_id>/run_manifest.json that captures
    all bar sizes and their respective artifacts.

    Args:
        run_dir: Run directory (e.g., runs/<run_id>/)
        run_id: Unique run identifier
        bar_sizes: List of bar sizes processed (e.g., ["1m", "5m"])
        config_dir: Path to configs directory
        per_bar_artifacts: Dict[bar_size, Dict[artifact_name, hash]]
            Example: {"1m": {"data_metadata": "hash123", "roll_schedule": "hash456"}}
        git_hash: Optional git commit hash (if not provided, will auto-detect)
        repo_path: Optional path to git repo (defaults to config_dir parent)
        include_packages: Whether to include full package list (slow)
        metadata: Optional additional metadata dict

    Returns:
        Path to written manifest file
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get git state
    if repo_path is None:
        repo_path = config_dir.parent.parent  # Go up from ml_intraday_v3/configs/

    if git_hash:
        git_state = GitState(commit_hash=git_hash, is_dirty=False)
    else:
        git_state = get_git_state(repo_path)

    # Load all config snapshots
    config_files = [
        ("execution_spec", "execution_spec.yaml"),
        ("data", "data.yaml"),
        ("labeling", "labeling.yaml"),
        ("validation", "validation.yaml"),
        ("risk", "risk.yaml"),
        ("retrain_policy", "retrain_policy.yaml"),
        ("metrics_contract", "metrics_contract.json"),
    ]

    configs = []
    execution_spec_hash = ""

    for name, filename in config_files:
        config_path = config_dir / filename
        if config_path.exists():
            snapshot = load_config_snapshot(config_path, name)
            configs.append(snapshot)

            # Capture execution spec hash separately (critical for parity)
            if name == "execution_spec":
                execution_spec_hash = snapshot.content_hash

    # Cost model hash (derived from execution_spec costs section)
    cost_model_hash = ""
    if execution_spec_hash:
        exec_config = next(
            (c for c in configs if c.name == "execution_spec"), None
        )
        if exec_config and "costs" in exec_config.content:
            cost_model_hash = hash_content(exec_config.content["costs"])

    # Get environment info
    environment = get_environment_info(include_packages=include_packages)

    # Build manifest
    manifest = RunManifest(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        bar_sizes=bar_sizes,
        git_state=git_state,
        configs=configs,
        per_bar_size_artifacts=per_bar_artifacts or {},
        execution_spec_hash=execution_spec_hash,
        cost_model_hash=cost_model_hash,
        environment=environment,
        metadata=metadata or {},
    )

    # Write to disk
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=False)

    return manifest_path


def compare_manifests(
    manifest1: RunManifest, manifest2: RunManifest
) -> Dict[str, Any]:
    """
    Compare two run manifests and report differences.

    Useful for debugging reproducibility issues or understanding
    what changed between runs.

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Dict with comparison results
    """
    differences = {
        "git_state": {},
        "configs": {},
        "schemas": {},
        "execution_spec": {},
    }

    # Compare git state
    if manifest1.git_state.commit_hash != manifest2.git_state.commit_hash:
        differences["git_state"]["commit_hash"] = {
            "manifest1": manifest1.git_state.commit_hash,
            "manifest2": manifest2.git_state.commit_hash,
        }

    # Compare config hashes
    config1_hashes = {c.name: c.content_hash for c in manifest1.configs}
    config2_hashes = {c.name: c.content_hash for c in manifest2.configs}

    for name in set(config1_hashes.keys()) | set(config2_hashes.keys()):
        hash1 = config1_hashes.get(name)
        hash2 = config2_hashes.get(name)
        if hash1 != hash2:
            differences["configs"][name] = {
                "manifest1": hash1,
                "manifest2": hash2,
            }

    # Compare schema hashes
    schema1_hashes = {s.name: s.hash for s in manifest1.schemas}
    schema2_hashes = {s.name: s.hash for s in manifest2.schemas}

    for name in set(schema1_hashes.keys()) | set(schema2_hashes.keys()):
        hash1 = schema1_hashes.get(name)
        hash2 = schema2_hashes.get(name)
        if hash1 != hash2:
            differences["schemas"][name] = {
                "manifest1": hash1,
                "manifest2": hash2,
            }

    # Compare execution specs (critical)
    if manifest1.execution_spec_hash != manifest2.execution_spec_hash:
        differences["execution_spec"]["hash"] = {
            "manifest1": manifest1.execution_spec_hash,
            "manifest2": manifest2.execution_spec_hash,
        }

    # Remove empty sections
    differences = {k: v for k, v in differences.items() if v}

    return differences
