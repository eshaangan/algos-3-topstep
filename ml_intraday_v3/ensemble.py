"""
Ensemble wrapper for LightGBM models.

Used to average probabilities across multiple base models.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


class LGBMEnsemble:
    def __init__(self, models: Iterable):
        self.models: List = list(models)
        if not self.models:
            raise ValueError("LGBMEnsemble requires at least one base model")

        self.n_models = len(self.models)
        self.classes_ = getattr(self.models[0], "classes_", None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [model.predict_proba(X) for model in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.classes_ is not None:
            return self.classes_[np.argmax(proba, axis=1)]
        return np.argmax(proba, axis=1)


class LGBMPreprocessedEnsemble:
    """
    Ensemble that applies per-model preprocessing before predicting.

    This is required when CV fold models were trained with fold-specific scalers/imputers.
    LiveModelPredictor can be configured to use a passthrough preprocessor so that this
    wrapper receives raw feature vectors and applies each model's own preprocessing.
    """

    def __init__(self, models: Iterable, preprocessors: Iterable[dict]):
        self.models: List = list(models)
        self.preprocessors: List[dict] = list(preprocessors)
        if not self.models:
            raise ValueError("LGBMPreprocessedEnsemble requires at least one base model")
        if len(self.models) != len(self.preprocessors):
            raise ValueError("models and preprocessors must have the same length")

        self.n_models = len(self.models)
        self.classes_ = getattr(self.models[0], "classes_", None)

    @staticmethod
    def _apply_preprocessor(X: np.ndarray, state: dict) -> np.ndarray:
        X = X.astype(float, copy=True)

        impute = (state or {}).get("impute", "median")
        scaler = (state or {}).get("scaler", "standard")

        if impute == "median":
            medians = np.array(state.get("medians"), dtype=float)
            mask = np.isnan(X)
            if mask.any():
                X[mask] = np.take(medians, np.where(mask)[1])
        elif impute == "zero":
            X = np.nan_to_num(X, 0.0)

        if scaler == "standard":
            means = np.array(state.get("means"), dtype=float)
            stds = np.array(state.get("stds"), dtype=float)
            X = (X - means) / (stds + 1e-8)

        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for model, prep in zip(self.models, self.preprocessors):
            Xi = self._apply_preprocessor(X, prep)
            probs.append(model.predict_proba(Xi))
        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.classes_ is not None:
            return self.classes_[np.argmax(proba, axis=1)]
        return np.argmax(proba, axis=1)


class DualSideModel:
    """
    Wrapper for separate long/short models to support dual-side inference.

    NOTE: If 'side' feature is present (index 0), this model automatically sets it to:
    - 1.0 for long_model predictions
    - -1.0 for short_model predictions
    """

    def __init__(self, long_model, short_model, side_feature_idx: int | None = None):
        self.long_model = long_model
        self.short_model = short_model
        self.classes_ = getattr(self.long_model, "classes_", None)
        # side_feature_idx: Index of 'side' feature in input array (usually 0, or None if not present)
        self.side_feature_idx = side_feature_idx

    def predict_proba_dual(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # If side feature exists, set it appropriately for each model
        if self.side_feature_idx is not None and X.shape[1] > self.side_feature_idx:
            X_long = X.copy()
            X_short = X.copy()
            X_long[:, self.side_feature_idx] = 1.0
            X_short[:, self.side_feature_idx] = -1.0
            return self.long_model.predict_proba(X_long), self.short_model.predict_proba(X_short)
        else:
            return self.long_model.predict_proba(X), self.short_model.predict_proba(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.long_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.long_model.predict(X)
