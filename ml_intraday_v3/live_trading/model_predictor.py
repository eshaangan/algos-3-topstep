"""
Model predictor for live trading.

Loads trained model bundle and generates predictions from features.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiveModelPredictor:
    """
    Generates predictions using trained model bundle.

    Loads model, preprocessor, and thresholds from the saved bundle
    and applies them to live features.
    """

    def __init__(self, model_bundle_path: Path):
        """
        Initialize the predictor.

        Args:
            model_bundle_path: Path to model_bundle.pkl file
        """
        self.bundle_path = Path(model_bundle_path)

        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")

        # Load bundle
        logger.info(f"Loading model bundle from: {model_bundle_path}")
        self.bundle = joblib.load(self.bundle_path)

        # Extract components
        self.model = self.bundle.get("primary_model")
        self.preprocessor_state = self.bundle.get("primary_preprocessor")
        self.feature_columns = self.bundle.get("primary_feature_columns")
        self.thresholds = self.bundle.get("thresholds", {})
        self.meta_model = self.bundle.get("meta_model")
        self.meta_preprocessor_state = self.bundle.get("meta_preprocessor")
        self.meta_feature_columns = self.bundle.get("meta_feature_columns")

        # Validate
        if self.model is None:
            raise ValueError("No primary_model in bundle")
        if self.preprocessor_state is None:
            raise ValueError("No primary_preprocessor in bundle")
        if self.feature_columns is None:
            raise ValueError("No primary_feature_columns in bundle")

        # Extract preprocessing parameters
        self.impute = self.preprocessor_state.get('impute', 'median')
        self.scaler = self.preprocessor_state.get('scaler', 'standard')
        self.medians = np.array(self.preprocessor_state['medians'])
        self.means = np.array(self.preprocessor_state['means'])
        self.stds = np.array(self.preprocessor_state['stds'])

        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Preprocessor: impute={self.impute}, scaler={self.scaler}")

        if self.meta_model is not None:
            logger.info("Meta model available")

        # Extract thresholds
        self.primary_threshold = self.thresholds.get('primary_threshold', 0.10)
        logger.info(f"Primary threshold: {self.primary_threshold}")

    def predict(
        self,
        features: pd.Series,
        use_meta: bool = False,
    ) -> Dict[str, float]:
        """
        Generate prediction from features.

        Args:
            features: Series with feature values
            use_meta: Whether to use meta model

        Returns:
            Dictionary with prediction scores:
                - y_prob: Probability of target hit
                - score_ev: EV score (p_target - p_stop)
                - p_stop: Probability of stop hit (if available)
                - p_target: Probability of target hit (if available)
                - p_vertical: Probability of vertical exit (if available)
                - meta_prob: Meta model probability (if use_meta=True)
        """
        # Prepare features as array
        X = features[self.feature_columns].values.reshape(1, -1)

        # Apply preprocessing
        X_scaled = self._preprocess(X)

        # Generate primary prediction
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)

            # Check if multiclass (3 outcomes: stop, target, vertical)
            if proba.shape[1] == 3:
                pred = {
                    'p_stop': float(proba[0, 0]),
                    'p_target': float(proba[0, 1]),
                    'p_vertical': float(proba[0, 2]),
                    'y_prob': float(proba[0, 1]),  # Target probability
                    'score_ev': float(proba[0, 1] - proba[0, 0]),  # EV score
                }
            else:
                # Binary classification
                pred = {
                    'y_prob': float(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]),
                    'score_ev': float(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]),
                }
        else:
            # Regression model
            y_pred = self.model.predict(X_scaled)
            pred = {
                'y_prob': float(y_pred[0]),
                'score_ev': float(y_pred[0]),
            }

        # Generate meta prediction if requested
        if use_meta and self.meta_model is not None:
            # Add primary prediction to features for meta model
            meta_features = pd.concat([
                features[self.feature_columns],
                pd.Series({
                    'p_primary': pred['y_prob'],
                    'p_primary_logit': np.log(pred['y_prob'] / (1 - pred['y_prob'] + 1e-8)),
                })
            ])

            X_meta = meta_features[self.meta_feature_columns].values.reshape(1, -1)
            X_meta_scaled = self._preprocess_meta(X_meta)

            if hasattr(self.meta_model, 'predict_proba'):
                meta_proba = self.meta_model.predict_proba(X_meta_scaled)
                pred['meta_prob'] = float(meta_proba[0, 1] if meta_proba.shape[1] > 1 else meta_proba[0, 0])
            else:
                meta_pred = self.meta_model.predict(X_meta_scaled)
                pred['meta_prob'] = float(meta_pred[0])

        return pred

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to features.

        Args:
            X: Raw feature matrix

        Returns:
            Preprocessed feature matrix
        """
        # Impute missing values
        if self.impute == 'median':
            mask = np.isnan(X)
            X[mask] = np.take(self.medians, np.where(mask)[1])
        elif self.impute == 'zero':
            X = np.nan_to_num(X, 0.0)

        # Scale
        if self.scaler == 'standard':
            X_scaled = (X - self.means) / (self.stds + 1e-8)
        elif self.scaler == 'minmax':
            # MinMax not stored in state, just use X as-is
            X_scaled = X
        else:
            X_scaled = X

        return X_scaled

    def _preprocess_meta(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to meta features.

        Args:
            X: Raw meta feature matrix

        Returns:
            Preprocessed meta feature matrix
        """
        if self.meta_preprocessor_state is None:
            return X

        impute = self.meta_preprocessor_state.get('impute', 'median')
        scaler = self.meta_preprocessor_state.get('scaler', 'standard')
        medians = np.array(self.meta_preprocessor_state['medians'])
        means = np.array(self.meta_preprocessor_state['means'])
        stds = np.array(self.meta_preprocessor_state['stds'])

        # Impute
        if impute == 'median':
            mask = np.isnan(X)
            X[mask] = np.take(medians, np.where(mask)[1])
        elif impute == 'zero':
            X = np.nan_to_num(X, 0.0)

        # Scale
        if scaler == 'standard':
            X_scaled = (X - means) / (stds + 1e-8)
        else:
            X_scaled = X

        return X_scaled

    def should_trade(
        self,
        prediction: Dict[str, float],
        primary_threshold: Optional[float] = None,
        meta_threshold: Optional[float] = None,
        require_meta_approval: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be executed based on prediction.

        Args:
            prediction: Prediction dictionary from predict()
            primary_threshold: Primary model threshold (overrides default)
            meta_threshold: Meta model threshold
            require_meta_approval: Whether meta approval is required

        Returns:
            (should_trade, reason) tuple
        """
        # Use thresholds from arguments or defaults
        primary_thresh = primary_threshold or self.primary_threshold
        meta_thresh = meta_threshold or 0.5

        # Check primary threshold
        score = prediction.get('score_ev', prediction.get('y_prob', 0.0))
        if score < primary_thresh:
            return False, f"primary_threshold (score={score:.3f} < {primary_thresh:.3f})"

        # Check meta threshold if required
        if require_meta_approval:
            if 'meta_prob' not in prediction:
                return False, "meta_approval_required_but_missing"

            if prediction['meta_prob'] < meta_thresh:
                return False, f"meta_threshold (prob={prediction['meta_prob']:.3f} < {meta_thresh:.3f})"

        return True, "approved"

    @classmethod
    def find_latest_model(cls, runs_dir: Path, bar_size: str = "1m") -> Optional[Path]:
        """
        Find the latest trained model bundle.

        Args:
            runs_dir: Path to runs directory
            bar_size: Bar size to look for

        Returns:
            Path to model_bundle.pkl, or None if not found
        """
        runs_dir = Path(runs_dir)

        # Find all run directories
        run_dirs = sorted(runs_dir.glob("*"), key=lambda x: x.name, reverse=True)

        for run_dir in run_dirs:
            # Look for walk-forward results
            wf_dir = run_dir / "walkforward" / f"bar_size={bar_size}"
            if not wf_dir.exists():
                continue

            # Find the last window
            windows = sorted(wf_dir.glob("window_*"), key=lambda x: int(x.name.split("_")[1]))
            if not windows:
                continue

            final_window = windows[-1]
            bundle_path = final_window / "model_bundle.pkl"

            if bundle_path.exists():
                logger.info(f"Found latest model: {bundle_path}")
                return bundle_path

        logger.warning(f"No model bundle found in {runs_dir}")
        return None

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        info = {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_columns),
            'primary_threshold': self.primary_threshold,
            'has_meta_model': self.meta_model is not None,
            'bundle_path': str(self.bundle_path),
        }

        # Add training range if available
        training_range = self.bundle.get('training_range')
        if training_range:
            info['training_start'] = training_range.get('start')
            info['training_end'] = training_range.get('end')

        # Add git state if available
        git_state = self.bundle.get('provenance', {}).get('git_state')
        if git_state:
            info['git_commit'] = git_state.get('commit_hash')
            info['git_branch'] = git_state.get('branch')
            info['git_dirty'] = git_state.get('is_dirty')

        return info
