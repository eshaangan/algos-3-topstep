"""
Model Ensemble - Phase 2b Priority #5

Ensemble of models with different training windows for better regime adaptation.

Key Concept:
- Single model is vulnerable to regime shifts
- Different lookback windows capture different market dynamics:
  - Short-term (3 months): Most adaptive to recent regime
  - Medium-term (6 months): Balance of adaptivity and stability
  - Long-term (12 months): Most stable across regimes
- Weighted ensemble combines strengths of all models

Expected Impact: +3-5% win rate, better regime handling

Research Context:
- "Ensemble Methods in Machine Learning" (Dietterich, 2000)
- Multiple models reduce overfitting and improve generalization
- Diverse training windows capture different market regimes
- Weighted voting is more robust than simple averaging

Usage:
    from models.ensemble import ModelEnsemble

    # Create ensemble
    ensemble = ModelEnsemble(
        short_term_window_months=3,
        medium_term_window_months=6,
        long_term_window_months=12
    )

    # Train ensemble (trains all 3 models)
    ensemble.fit(training_data, features, labels)

    # Predict with ensemble
    probability = ensemble.predict_proba(current_features)

    # Get ensemble details
    details = ensemble.get_prediction_details(current_features)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Ensemble of models with different training windows.

    Strategy:
    - Short-term model: Fast adaptation to recent regime
    - Medium-term model: Balance of stability and adaptation
    - Long-term model: Stable baseline across regimes
    - Weighted average: Emphasize recent (short-term) model
    """

    def __init__(
        self,
        short_term_window_months: int = 3,
        medium_term_window_months: int = 6,
        long_term_window_months: int = 12,
        short_term_weight: float = 0.50,
        medium_term_weight: float = 0.30,
        long_term_weight: float = 0.20,
        model_type: str = 'lightgbm'
    ):
        """
        Initialize model ensemble.

        Args:
            short_term_window_months: Lookback for short-term model (default: 3)
            medium_term_window_months: Lookback for medium-term model (default: 6)
            long_term_window_months: Lookback for long-term model (default: 12)
            short_term_weight: Weight for short-term predictions (default: 0.50)
            medium_term_weight: Weight for medium-term predictions (default: 0.30)
            long_term_weight: Weight for long-term predictions (default: 0.20)
            model_type: Type of base model ('lightgbm', 'xgboost', 'random_forest')
        """
        # Validate weights sum to 1.0
        total_weight = short_term_weight + medium_term_weight + long_term_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.windows = {
            'short_term': short_term_window_months,
            'medium_term': medium_term_window_months,
            'long_term': long_term_window_months
        }

        self.weights = {
            'short_term': short_term_weight,
            'medium_term': medium_term_weight,
            'long_term': long_term_weight
        }

        self.model_type = model_type
        self.models = {
            'short_term': None,
            'medium_term': None,
            'long_term': None
        }

        self.feature_cols = None
        self.is_fitted = False

        logger.info(
            f"ModelEnsemble initialized: "
            f"windows={self.windows}, weights={self.weights}, model={model_type}"
        )

    def fit(
        self,
        training_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        timestamp_col: str = 'timestamp'
    ):
        """
        Train all models in the ensemble.

        Args:
            training_data: Full training dataset with timestamp
            feature_cols: List of feature column names
            label_col: Name of label column (default: 'label')
            timestamp_col: Name of timestamp column (default: 'timestamp')
        """
        logger.info("Training model ensemble...")

        self.feature_cols = feature_cols

        # Validate data
        required_cols = feature_cols + [label_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in training_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Get latest timestamp for windowing
        latest_timestamp = training_data[timestamp_col].max()

        # Train each model with its respective window
        for model_name, window_months in self.windows.items():
            logger.info(f"Training {model_name} model (window={window_months} months)...")

            # Calculate cutoff date
            cutoff_date = latest_timestamp - pd.DateOffset(months=window_months)

            # Filter data for this model
            model_data = training_data[training_data[timestamp_col] >= cutoff_date].copy()

            if len(model_data) < 100:
                logger.warning(
                    f"{model_name} model has only {len(model_data)} samples. "
                    f"Consider using longer history."
                )

            # Train model
            X = model_data[feature_cols]
            y = model_data[label_col]

            model = self._create_base_model()
            model.fit(X, y)

            self.models[model_name] = model

            logger.info(
                f"  ✅ {model_name} trained on {len(model_data)} samples "
                f"({cutoff_date.date()} to {latest_timestamp.date()})"
            )

        self.is_fitted = True
        logger.info("✅ Ensemble training complete")

    def _create_base_model(self):
        """Create a base model instance."""

        if self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM not installed, falling back to sklearn")
                self.model_type = 'random_forest'

        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42
                )
            except ImportError:
                logger.warning("XGBoost not installed, falling back to sklearn")
                self.model_type = 'random_forest'

        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )

        raise ValueError(f"Unknown model type: {self.model_type}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Validate features
        missing_cols = [col for col in self.feature_cols if col not in features.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        X = features[self.feature_cols]

        # Get predictions from each model
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict_proba(X)

        # Weighted average
        ensemble_proba = (
            predictions['short_term'] * self.weights['short_term'] +
            predictions['medium_term'] * self.weights['medium_term'] +
            predictions['long_term'] * self.weights['long_term']
        )

        return ensemble_proba

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels using ensemble.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(features)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_prediction_details(
        self,
        features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get detailed predictions from each model in ensemble.

        Args:
            features: DataFrame with feature columns (single row)

        Returns:
            Dictionary with predictions from each model and weighted ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        X = features[self.feature_cols]

        details = {}

        # Get prediction from each model
        for model_name, model in self.models.items():
            proba = model.predict_proba(X)[0, 1]  # Probability of class 1
            details[f'{model_name}_proba'] = proba
            details[f'{model_name}_weight'] = self.weights[model_name]
            details[f'{model_name}_weighted'] = proba * self.weights[model_name]

        # Ensemble prediction
        ensemble_proba = sum(details[f'{name}_weighted'] for name in self.models.keys())
        details['ensemble_proba'] = ensemble_proba
        details['ensemble_prediction'] = int(ensemble_proba >= 0.5)

        # Agreement metrics
        probas = [details[f'{name}_proba'] for name in self.models.keys()]
        details['min_proba'] = min(probas)
        details['max_proba'] = max(probas)
        details['proba_std'] = np.std(probas)
        details['model_agreement'] = 1.0 - (details['proba_std'] / 0.5)  # Normalized

        return details

    def get_model_info(self) -> Dict:
        """
        Get information about the ensemble and its models.

        Returns:
            Dictionary with ensemble configuration and status
        """
        return {
            'is_fitted': self.is_fitted,
            'model_type': self.model_type,
            'windows': self.windows,
            'weights': self.weights,
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'models_trained': sum(1 for m in self.models.values() if m is not None)
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"ModelEnsemble(fitted={info['is_fitted']}, "
            f"models={info['models_trained']}/3, "
            f"features={info['feature_count']})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*70)
    print("Model Ensemble Test")
    print("="*70)

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2025-01-01', periods=n_samples, freq='5min')

    # Create synthetic features
    data = {
        'timestamp': dates,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples)
    }

    # Create synthetic labels (with some signal)
    data['label'] = (
        (data['feature_1'] + data['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.5) > 0
    ).astype(int)

    training_data = pd.DataFrame(data)

    print(f"\nTraining Data: {len(training_data)} samples")
    print(f"Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
    print(f"Label distribution: {training_data['label'].value_counts().to_dict()}")

    # Create ensemble
    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

    ensemble = ModelEnsemble(
        short_term_window_months=3,
        medium_term_window_months=6,
        long_term_window_months=12,
        model_type='random_forest'
    )

    print(f"\n✅ Ensemble created: {ensemble}")

    # Train ensemble
    ensemble.fit(training_data, feature_cols)

    # Test prediction
    test_sample = training_data[feature_cols].iloc[-1:].copy()

    print("\n📊 Testing Prediction:")
    print(f"Test sample:\n{test_sample.values}")

    # Get ensemble prediction
    proba = ensemble.predict_proba(test_sample)
    prediction = ensemble.predict(test_sample)

    print(f"\nEnsemble Prediction:")
    print(f"  Probability: {proba[0, 1]:.3f}")
    print(f"  Prediction: {prediction[0]}")

    # Get detailed predictions
    details = ensemble.get_prediction_details(test_sample)

    print(f"\nDetailed Predictions:")
    print(f"  Short-term (50%): {details['short_term_proba']:.3f}")
    print(f"  Medium-term (30%): {details['medium_term_proba']:.3f}")
    print(f"  Long-term (20%): {details['long_term_proba']:.3f}")
    print(f"  Ensemble: {details['ensemble_proba']:.3f}")
    print(f"  Model agreement: {details['model_agreement']:.3f}")

    # Get model info
    info = ensemble.get_model_info()
    print(f"\nModel Info:")
    print(f"  Fitted: {info['is_fitted']}")
    print(f"  Models trained: {info['models_trained']}")
    print(f"  Features: {info['feature_count']}")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
