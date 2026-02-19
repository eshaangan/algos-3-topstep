"""Two-stage meta-labeling model."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier


class TwoStageMetaLabeler(BaseEstimator, ClassifierMixin):
    """Primary direction model + secondary trade-filter model."""

    def __init__(self, primary_estimator, secondary_estimator=None, secondary_threshold: float = 0.55):
        self.primary_estimator = primary_estimator
        self.secondary_estimator = secondary_estimator or RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        self.secondary_threshold = float(secondary_threshold)

    def _meta_features(self, X, primary_proba: np.ndarray) -> np.ndarray:
        p = np.clip(primary_proba, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        x_vals = X.values if hasattr(X, "values") else np.asarray(X)
        return np.column_stack([x_vals, p, logit])

    def fit(self, X, y_primary, y_meta=None, sample_weight_primary=None):
        y_primary_bin = (np.asarray(y_primary) == 1).astype(int)

        self.primary_ = clone(self.primary_estimator)
        if sample_weight_primary is None:
            self.primary_.fit(X, y_primary_bin)
        else:
            self.primary_.fit(X, y_primary_bin, sample_weight=sample_weight_primary)

        p_primary = self.primary_.predict_proba(X)[:, 1]
        pred_primary = (p_primary >= 0.5).astype(int)

        if y_meta is None:
            y_meta = (pred_primary == y_primary_bin).astype(int)
        else:
            y_meta = np.asarray(y_meta).astype(int)

        self.secondary_ = clone(self.secondary_estimator)
        self.secondary_.fit(self._meta_features(X, p_primary), y_meta)
        return self

    def predict_proba(self, X):
        p_primary = self.primary_.predict_proba(X)[:, 1]
        p_secondary = self.secondary_.predict_proba(self._meta_features(X, p_primary))[:, 1]
        p_trade = p_primary * p_secondary
        return np.column_stack([1.0 - p_trade, p_trade])

    def predict(self, X):
        p_secondary = self.secondary_.predict_proba(self._meta_features(X, self.primary_.predict_proba(X)[:, 1]))[:, 1]
        p_primary = self.primary_.predict_proba(X)[:, 1]
        return ((p_primary >= 0.5) & (p_secondary >= self.secondary_threshold)).astype(int)
