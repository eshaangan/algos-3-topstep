"""
Trial Tracking System for PBO (Probability of Backtest Overfitting) Analysis

This module implements comprehensive trial tracking to properly compute PBO
as described in López de Prado's work. Unlike naive implementations that only
track winning configurations, this tracks ALL trials to avoid selection bias.

Key Concepts:
- Trial: A single configuration (hyperparameters, features, model type)
- IS Performance: In-sample (training) performance on CPCV paths
- OOS Performance: Out-of-sample (test) performance on CPCV paths
- PBO: Probability that backtest overfitting occurred (fraction of paths where
  best-by-IS config ranks below median on OOS)

Reference: López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trial:
    """
    Represents a single trial (configuration) in the optimization process.

    Attributes:
        trial_id: Unique identifier for this trial
        config_hash: Hash of configuration (for deduplication)
        timestamp: When trial was created
        config: Full configuration dictionary
        model_type: Type of model (e.g., 'logreg', 'lgbm', 'relogit')
        hyperparameters: Model hyperparameters
        features: Feature set used (or feature config)
        path_metrics: Dict mapping path_id -> {'is_metric': float, 'oos_metric': float}
        metadata: Additional trial-specific metadata
    """
    trial_id: str
    config_hash: str
    timestamp: str
    config: Dict[str, Any]
    model_type: str
    hyperparameters: Dict[str, Any]
    features: Optional[List[str]] = None
    path_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trial':
        """Create Trial from dictionary."""
        return cls(**data)


class TrialTracker:
    """
    Tracks all configurations/trials tested during model development.

    This class maintains a complete record of all trials, not just winners,
    which is essential for properly computing PBO. Selection bias occurs when
    only tracking successful configurations.

    Usage:
        tracker = TrialTracker(run_dir)

        # Log a trial
        trial_id = tracker.log_trial(
            config=config_dict,
            model_type='lgbm',
            hyperparameters={'n_estimators': 500, ...},
            features=['rsi', 'macd', ...]
        )

        # Update with CPCV results
        tracker.update_path_metrics(
            trial_id=trial_id,
            path_id='path_0',
            is_metric=0.65,  # In-sample ROC-AUC
            oos_metric=0.58  # Out-of-sample ROC-AUC
        )

        # Save to disk
        tracker.save()

        # Load for PBO computation
        df = tracker.to_dataframe()
        pbo = compute_pbo(df)
    """

    def __init__(self, run_dir: Path | str):
        """
        Initialize trial tracker.

        Args:
            run_dir: Root directory for this run (will create trials/ subdirectory)
        """
        self.run_dir = Path(run_dir)
        self.trials_dir = self.run_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self.trials_file = self.trials_dir / "trials.json"
        self.trials: Dict[str, Trial] = {}

        # Load existing trials if file exists
        if self.trials_file.exists():
            self._load()

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            SHA256 hash (first 16 chars)
        """
        # Sort keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def log_trial(
        self,
        config: Dict[str, Any],
        model_type: str,
        hyperparameters: Dict[str, Any],
        features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trial_id: Optional[str] = None,
    ) -> str:
        """
        Log a new trial.

        Args:
            config: Full configuration dictionary
            model_type: Model type ('logreg', 'lgbm', 'relogit', etc.)
            hyperparameters: Model hyperparameters
            features: Feature set used (optional)
            metadata: Additional metadata (optional)
            trial_id: Optional custom trial ID (auto-generated if None)

        Returns:
            trial_id: Unique identifier for this trial
        """
        config_hash = self._compute_config_hash(config)

        # Generate trial ID if not provided
        if trial_id is None:
            trial_id = f"trial_{len(self.trials):04d}_{config_hash}"

        # Check for duplicates
        if trial_id in self.trials:
            logger.warning(f"Trial {trial_id} already exists. Skipping.")
            return trial_id

        # Create trial
        trial = Trial(
            trial_id=trial_id,
            config_hash=config_hash,
            timestamp=datetime.now().isoformat(),
            config=config,
            model_type=model_type,
            hyperparameters=hyperparameters,
            features=features,
            metadata=metadata or {},
        )

        self.trials[trial_id] = trial
        logger.info(f"Logged trial: {trial_id} (hash: {config_hash})")

        return trial_id

    def update_path_metrics(
        self,
        trial_id: str,
        path_id: str,
        is_metric: float,
        oos_metric: float,
    ):
        """
        Update trial with IS/OOS metrics for a specific CPCV path.

        Args:
            trial_id: Trial identifier
            path_id: CPCV path identifier (e.g., 'path_0')
            is_metric: In-sample metric (e.g., ROC-AUC on train)
            oos_metric: Out-of-sample metric (e.g., ROC-AUC on test)
        """
        if trial_id not in self.trials:
            raise ValueError(f"Trial {trial_id} not found")

        self.trials[trial_id].path_metrics[path_id] = {
            'is_metric': float(is_metric),
            'oos_metric': float(oos_metric),
        }

        logger.debug(
            f"Updated trial {trial_id}, path {path_id}: "
            f"IS={is_metric:.4f}, OOS={oos_metric:.4f}"
        )

    def update_metadata(self, trial_id: str, metadata: Dict[str, Any]):
        """
        Update trial metadata.

        Args:
            trial_id: Trial identifier
            metadata: Metadata dictionary to merge
        """
        if trial_id not in self.trials:
            raise ValueError(f"Trial {trial_id} not found")

        self.trials[trial_id].metadata.update(metadata)

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get trial by ID."""
        return self.trials.get(trial_id)

    def list_trials(self) -> List[str]:
        """Get list of all trial IDs."""
        return list(self.trials.keys())

    def save(self):
        """Save trials to JSON file."""
        data = {
            'trials': {tid: trial.to_dict() for tid, trial in self.trials.items()},
            'n_trials': len(self.trials),
            'last_updated': datetime.now().isoformat(),
        }

        with open(self.trials_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.trials)} trials to {self.trials_file}")

    def _load(self):
        """Load trials from JSON file."""
        with open(self.trials_file, 'r') as f:
            data = json.load(f)

        trials_data = data.get('trials', {})
        self.trials = {
            tid: Trial.from_dict(trial_data)
            for tid, trial_data in trials_data.items()
        }

        logger.info(f"Loaded {len(self.trials)} trials from {self.trials_file}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert trials to DataFrame for PBO computation.

        Returns:
            DataFrame with columns:
                - trial_id
                - config_hash
                - model_type
                - timestamp
                - path_0_is, path_0_oos, path_1_is, path_1_oos, ...
                  (one pair of columns per CPCV path)
        """
        if not self.trials:
            return pd.DataFrame()

        # Collect all unique path IDs
        all_paths = set()
        for trial in self.trials.values():
            all_paths.update(trial.path_metrics.keys())
        all_paths = sorted(all_paths)

        # Build records
        records = []
        for trial_id, trial in self.trials.items():
            record = {
                'trial_id': trial_id,
                'config_hash': trial.config_hash,
                'model_type': trial.model_type,
                'timestamp': trial.timestamp,
            }

            # Add IS/OOS metrics for each path
            for path_id in all_paths:
                metrics = trial.path_metrics.get(path_id, {})
                record[f'{path_id}_is'] = metrics.get('is_metric')
                record[f'{path_id}_oos'] = metrics.get('oos_metric')

            records.append(record)

        df = pd.DataFrame(records)
        return df

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about tracked trials.

        Returns:
            Dictionary with summary stats
        """
        if not self.trials:
            return {
                'n_trials': 0,
                'n_paths': 0,
                'model_types': [],
                'date_range': None,
            }

        # Count paths
        all_paths = set()
        for trial in self.trials.values():
            all_paths.update(trial.path_metrics.keys())

        # Model types
        model_types = {}
        for trial in self.trials.values():
            model_types[trial.model_type] = model_types.get(trial.model_type, 0) + 1

        # Date range
        timestamps = [trial.timestamp for trial in self.trials.values()]
        timestamps = sorted(timestamps)

        return {
            'n_trials': len(self.trials),
            'n_paths': len(all_paths),
            'model_types': model_types,
            'date_range': {
                'first': timestamps[0] if timestamps else None,
                'last': timestamps[-1] if timestamps else None,
            },
        }

    def filter_trials(
        self,
        model_type: Optional[str] = None,
        min_timestamp: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> 'TrialTracker':
        """
        Create a new TrialTracker with filtered trials.

        Args:
            model_type: Filter by model type
            min_timestamp: Filter trials after this timestamp
            max_timestamp: Filter trials before this timestamp

        Returns:
            New TrialTracker with filtered trials
        """
        filtered = TrialTracker(self.run_dir)

        for trial_id, trial in self.trials.items():
            # Apply filters
            if model_type and trial.model_type != model_type:
                continue
            if min_timestamp and trial.timestamp < min_timestamp:
                continue
            if max_timestamp and trial.timestamp > max_timestamp:
                continue

            filtered.trials[trial_id] = trial

        return filtered


def create_trial_from_config(
    config: Dict[str, Any],
    trial_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper function to extract trial information from a training config.

    Args:
        config: Training configuration dictionary
        trial_id: Optional trial ID

    Returns:
        Dictionary with trial fields ready for TrialTracker.log_trial()
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('kind', 'unknown')
    hyperparameters = model_cfg.get('params', {})

    # Add rare events config if present
    if 'rare_events' in config:
        hyperparameters['rare_events'] = config['rare_events']

    features_cfg = config.get('features', {})

    return {
        'config': config,
        'model_type': model_type,
        'hyperparameters': hyperparameters,
        'features': features_cfg.get('use_columns'),
        'metadata': {
            'seed': config.get('seed', 42),
            'target_mode': config.get('target', {}).get('mode', 'binary'),
        },
        'trial_id': trial_id,
    }
