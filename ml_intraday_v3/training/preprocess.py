"""
Fold-safe preprocessing utilities (impute + scale).
"""

import numpy as np


class FoldPreprocessor:
    """Impute and scale using train-only statistics."""

    def __init__(self, feature_columns, config):
        self.feature_columns = list(feature_columns)
        self.impute = config.get("preprocessing", {}).get("impute", "median")
        self.scaler = config.get("preprocessing", {}).get("scaler", "standard")
        self.medians_ = None
        self.means_ = None
        self.stds_ = None

    def fit(self, train_df):
        X = train_df[self.feature_columns].to_numpy(dtype=float)

        if self.impute == "median":
            self.medians_ = np.nanmedian(X, axis=0)
            X = np.where(np.isnan(X), self.medians_, X)
        elif self.impute == "none":
            self.medians_ = None
        else:
            raise ValueError(f"Unsupported impute: {self.impute}")

        if self.scaler == "standard":
            self.means_ = X.mean(axis=0)
            self.stds_ = X.std(axis=0)
            self.stds_ = np.where(self.stds_ == 0.0, 1.0, self.stds_)
        elif self.scaler == "none":
            self.means_ = None
            self.stds_ = None
        else:
            raise ValueError(f"Unsupported scaler: {self.scaler}")

        return self

    def transform(self, df):
        X = df[self.feature_columns].to_numpy(dtype=float)

        if self.impute == "median":
            X = np.where(np.isnan(X), self.medians_, X)

        if self.scaler == "standard":
            X = (X - self.means_) / self.stds_

        y = df["y"].to_numpy()
        w = df["w_final"].to_numpy() if "w_final" in df.columns else None
        return X, y, w

    def state(self):
        """Return serializable preprocessor state."""
        return {
            "impute": self.impute,
            "scaler": self.scaler,
            "medians": None if self.medians_ is None else self.medians_.tolist(),
            "means": None if self.means_ is None else self.means_.tolist(),
            "stds": None if self.stds_ is None else self.stds_.tolist(),
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_state(cls, state_dict, config=None):
        """Restore preprocessor from saved state."""
        if config is None:
            config = {}
        obj = cls(state_dict.get("feature_columns", []), config)
        obj.impute = state_dict.get("impute", "median")
        obj.scaler = state_dict.get("scaler", "standard")
        obj.medians_ = np.array(state_dict["medians"]) if state_dict.get("medians") else None
        obj.means_ = np.array(state_dict["means"]) if state_dict.get("means") else None
        obj.stds_ = np.array(state_dict["stds"]) if state_dict.get("stds") else None
        return obj
