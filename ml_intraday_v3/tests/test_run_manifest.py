"""
Unit tests for run_manifest.py

Tests cover:
1. Schema stability (manifest structure doesn't change unexpectedly)
2. Hash determinism (same content = same hash)
3. Config snapshot loading
4. Git state capture
5. Manifest write/read round-trip
6. Manifest comparison
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_manifest import (
    ConfigSnapshot,
    EnvironmentInfo,
    GitState,
    RunManifest,
    SchemaHash,
    compare_manifests,
    get_git_state,
    hash_content,
    load_config_snapshot,
    load_run_manifest,
    write_run_manifest,
)


class TestHashContent:
    """Test hash_content function."""

    def test_hash_dict_deterministic(self):
        """Same dict should produce same hash regardless of key order."""
        dict1 = {"b": 2, "a": 1, "c": 3}
        dict2 = {"a": 1, "c": 3, "b": 2}

        hash1 = hash_content(dict1)
        hash2 = hash_content(dict2)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_string(self):
        """Test hashing string content."""
        content = "test string"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

        assert hash_content(content) == expected

    def test_hash_bytes(self):
        """Test hashing bytes content."""
        content = b"test bytes"
        expected = hashlib.sha256(content).hexdigest()

        assert hash_content(content) == expected

    def test_hash_different_dicts(self):
        """Different dicts should produce different hashes."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3}

        hash1 = hash_content(dict1)
        hash2 = hash_content(dict2)

        assert hash1 != hash2


class TestConfigSnapshot:
    """Test ConfigSnapshot loading and hashing."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML config and creating snapshot."""
        import yaml

        # Create test YAML config
        config_data = {
            "version": "1.0.0",
            "param1": 100,
            "param2": [1, 2, 3],
        }

        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load snapshot
        snapshot = load_config_snapshot(config_path, "test")

        assert snapshot.name == "test"
        assert snapshot.path == str(config_path)
        assert snapshot.content == config_data
        assert len(snapshot.content_hash) == 64

    def test_load_json_config(self, tmp_path):
        """Test loading JSON config and creating snapshot."""
        config_data = {
            "version": "1.0.0",
            "param1": 100,
        }

        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load snapshot
        snapshot = load_config_snapshot(config_path, "test")

        assert snapshot.name == "test"
        assert snapshot.content == config_data

    def test_config_hash_stability(self, tmp_path):
        """Test that same config content produces same hash."""
        import yaml

        config_data = {"a": 1, "b": 2}

        # Write config twice
        config_path1 = tmp_path / "test1.yaml"
        config_path2 = tmp_path / "test2.yaml"

        with open(config_path1, "w") as f:
            yaml.dump(config_data, f)

        with open(config_path2, "w") as f:
            yaml.dump(config_data, f)

        snapshot1 = load_config_snapshot(config_path1, "test")
        snapshot2 = load_config_snapshot(config_path2, "test")

        assert snapshot1.content_hash == snapshot2.content_hash


class TestGitState:
    """Test git state capture."""

    def test_git_state_in_repo(self):
        """Test git state capture in actual repo."""
        # This test runs in the actual repo
        repo_path = Path(__file__).parent.parent.parent

        git_state = get_git_state(repo_path)

        # Should have a commit hash (unless not in a git repo)
        if git_state.commit_hash:
            assert len(git_state.commit_hash) == 40  # Git SHA-1 length
            assert git_state.branch is not None

    def test_git_state_non_repo(self, tmp_path):
        """Test git state capture in non-repo directory."""
        git_state = get_git_state(tmp_path)

        # Should return minimal state
        assert git_state.commit_hash is None
        assert git_state.is_dirty is False


class TestRunManifest:
    """Test RunManifest creation and persistence."""

    def test_manifest_schema_stability(self):
        """Test that RunManifest schema is stable."""
        # Create a minimal manifest
        manifest = RunManifest(
            run_id="test_run_001",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(commit_hash="abc123", is_dirty=False),
            configs=[],
            schemas=[],
        )

        # Check required fields exist
        assert hasattr(manifest, "run_id")
        assert hasattr(manifest, "timestamp")
        assert hasattr(manifest, "bar_size")
        assert hasattr(manifest, "git_state")
        assert hasattr(manifest, "configs")
        assert hasattr(manifest, "schemas")
        assert hasattr(manifest, "model_ids")
        assert hasattr(manifest, "calibration_ids")
        assert hasattr(manifest, "execution_spec_hash")
        assert hasattr(manifest, "cost_model_hash")
        assert hasattr(manifest, "environment")
        assert hasattr(manifest, "metadata")
        assert hasattr(manifest, "manifest_version")

        # Check types
        assert isinstance(manifest.run_id, str)
        assert isinstance(manifest.bar_size, str)
        assert isinstance(manifest.git_state, GitState)
        assert isinstance(manifest.configs, list)
        assert isinstance(manifest.schemas, list)
        assert isinstance(manifest.model_ids, dict)

    def test_write_and_load_manifest(self, tmp_path):
        """Test writing manifest to disk and loading it back."""
        import yaml

        # Create test configs
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        test_config = {"version": "1.0.0", "test": True}
        config_path = config_dir / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Create run directory
        run_dir = tmp_path / "runs" / "test_run_001" / "bar_size=1m"

        # Write manifest
        manifest_path = write_run_manifest(
            run_dir=run_dir,
            run_id="test_run_001",
            bar_size="1m",
            config_dir=config_dir,
            feature_schema_hash="feat_hash_123",
            feature_schema_path="features.parquet",
            label_schema_hash="label_hash_456",
            label_schema_path="labels.parquet",
            cv_split_ids="cv_hash_789",
            cv_split_path="cv_splits.json",
            git_hash="abc123def456",
            metadata={"experiment": "baseline"},
        )

        assert manifest_path.exists()

        # Load it back
        loaded_manifest = load_run_manifest(manifest_path)

        assert loaded_manifest.run_id == "test_run_001"
        assert loaded_manifest.bar_size == "1m"
        assert loaded_manifest.git_state.commit_hash == "abc123def456"
        assert loaded_manifest.metadata["experiment"] == "baseline"

        # Check schemas were captured
        schema_names = [s.name for s in loaded_manifest.schemas]
        assert "feature_schema" in schema_names
        assert "label_schema" in schema_names
        assert "cv_splits" in schema_names

    def test_manifest_includes_execution_spec_hash(self, tmp_path):
        """Test that execution spec hash is captured separately."""
        import yaml

        # Create execution_spec config
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        execution_spec = {
            "version": "1.0.0",
            "instrument": {
                "symbol": "MES",
                "tick_size_points": 0.25,
                "contract_multiplier_usd_per_point": 5.0,
            },
            "costs": {
                "slippage_ticks": {"1m": 1.0, "5m": 1.5},
                "commission_per_contract": 0.62,
            },
        }

        config_path = config_dir / "execution_spec.yaml"
        with open(config_path, "w") as f:
            yaml.dump(execution_spec, f)

        # Create run directory
        run_dir = tmp_path / "runs" / "test_run_002" / "bar_size=1m"

        # Write manifest
        manifest_path = write_run_manifest(
            run_dir=run_dir,
            run_id="test_run_002",
            bar_size="1m",
            config_dir=config_dir,
        )

        # Load manifest
        manifest = load_run_manifest(manifest_path)

        # Check execution spec hash is present
        assert manifest.execution_spec_hash != ""
        assert len(manifest.execution_spec_hash) == 64

        # Check cost model hash is present
        assert manifest.cost_model_hash != ""

    def test_manifest_json_serializable(self, tmp_path):
        """Test that manifest serializes to valid JSON."""
        from dataclasses import asdict

        manifest = RunManifest(
            run_id="test_run_003",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="5m",
            git_state=GitState(
                commit_hash="xyz789",
                is_dirty=True,
                dirty_diff_hash="diff_hash_123",
                branch="main",
                tags=["v1.0"],
            ),
            configs=[
                ConfigSnapshot(
                    name="test",
                    path="/path/to/test.yaml",
                    content_hash="hash123",
                    content={"key": "value"},
                )
            ],
            schemas=[
                SchemaHash(
                    name="test_schema",
                    hash="schema_hash_123",
                    artifact_path="schema.json",
                )
            ],
        )

        # Convert to dict and serialize to JSON
        manifest_dict = asdict(manifest)
        json_str = json.dumps(manifest_dict, indent=2)

        # Should be valid JSON
        loaded = json.loads(json_str)

        assert loaded["run_id"] == "test_run_003"
        assert loaded["git_state"]["commit_hash"] == "xyz789"
        assert loaded["git_state"]["is_dirty"] is True


class TestCompareManifests:
    """Test manifest comparison utility."""

    def test_compare_identical_manifests(self):
        """Identical manifests should have no differences."""
        manifest1 = RunManifest(
            run_id="run1",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(commit_hash="abc123"),
            configs=[],
            schemas=[],
            execution_spec_hash="exec_hash_123",
        )

        manifest2 = RunManifest(
            run_id="run2",  # Different run ID is OK
            timestamp="2025-01-01T01:00:00Z",  # Different timestamp is OK
            bar_size="1m",
            git_state=GitState(commit_hash="abc123"),
            configs=[],
            schemas=[],
            execution_spec_hash="exec_hash_123",
        )

        diff = compare_manifests(manifest1, manifest2)

        assert diff == {}  # No differences

    def test_compare_different_git_state(self):
        """Different git state should be detected."""
        manifest1 = RunManifest(
            run_id="run1",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(commit_hash="abc123"),
            configs=[],
            schemas=[],
        )

        manifest2 = RunManifest(
            run_id="run2",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(commit_hash="def456"),
            configs=[],
            schemas=[],
        )

        diff = compare_manifests(manifest1, manifest2)

        assert "git_state" in diff
        assert diff["git_state"]["commit_hash"]["manifest1"] == "abc123"
        assert diff["git_state"]["commit_hash"]["manifest2"] == "def456"

    def test_compare_different_configs(self):
        """Different config hashes should be detected."""
        config1 = ConfigSnapshot(
            name="test", path="test.yaml", content_hash="hash1", content={}
        )
        config2 = ConfigSnapshot(
            name="test", path="test.yaml", content_hash="hash2", content={}
        )

        manifest1 = RunManifest(
            run_id="run1",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[config1],
            schemas=[],
        )

        manifest2 = RunManifest(
            run_id="run2",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[config2],
            schemas=[],
        )

        diff = compare_manifests(manifest1, manifest2)

        assert "configs" in diff
        assert "test" in diff["configs"]

    def test_compare_different_execution_spec(self):
        """Different execution spec hash should be detected (critical)."""
        manifest1 = RunManifest(
            run_id="run1",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[],
            schemas=[],
            execution_spec_hash="exec_hash_1",
        )

        manifest2 = RunManifest(
            run_id="run2",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[],
            schemas=[],
            execution_spec_hash="exec_hash_2",
        )

        diff = compare_manifests(manifest1, manifest2)

        assert "execution_spec" in diff
        assert diff["execution_spec"]["hash"]["manifest1"] == "exec_hash_1"
        assert diff["execution_spec"]["hash"]["manifest2"] == "exec_hash_2"


class TestManifestVersionStability:
    """Test that manifest version is stable and versioned correctly."""

    def test_manifest_version_present(self):
        """Manifest should have a version field."""
        manifest = RunManifest(
            run_id="test",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[],
            schemas=[],
        )

        assert manifest.manifest_version == "1.1.0"  # Updated for multi-bar-size support

    def test_manifest_version_in_json(self, tmp_path):
        """Manifest version should be persisted in JSON."""
        from dataclasses import asdict

        manifest = RunManifest(
            run_id="test",
            timestamp="2025-01-01T00:00:00Z",
            bar_size="1m",
            git_state=GitState(),
            configs=[],
            schemas=[],
        )

        # Write to JSON
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)

        # Read back as raw JSON
        with open(manifest_path, "r") as f:
            data = json.load(f)

        assert "manifest_version" in data
        assert data["manifest_version"] == "1.1.0"  # Updated for multi-bar-size support


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
