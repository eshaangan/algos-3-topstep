"""
Model predictor for live trading.

Loads trained model bundle and generates predictions from features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        self.bundle_decision_cfg = self.bundle.get("live_decision", {}) or {}
        self.meta_routes = self._load_meta_routes()

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

        # Check if model is DualSideModel with 'side' feature
        self.has_dual_model = hasattr(self.model, "predict_proba_dual")
        self.has_side_feature = 'side' in self.feature_columns

        # For DualSideModel with 'side' feature, set the side_feature_idx so it can inject values
        if self.has_dual_model and self.has_side_feature:
            side_idx = self.feature_columns.index('side')
            self.model.side_feature_idx = side_idx
            logger.info(f"DualSideModel configured with side_feature_idx={side_idx}")

        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Preprocessor: impute={self.impute}, scaler={self.scaler}")

        if self.meta_model is not None:
            logger.info("Legacy single meta model available")
        if self.meta_routes:
            logger.info("Loaded %d routed meta models", len(self.meta_routes))

        # Extract thresholds
        self.primary_threshold = self.thresholds.get('primary_threshold', 0.10)
        logger.info(f"Primary threshold: {self.primary_threshold}")

    def predict(
        self,
        features: pd.Series,
        use_meta: bool = False,
        side: int | None = None,
        combined_regime: int | None = None,
    ) -> Dict[str, float]:
        """
        Generate prediction from features.

        For bidirectional models with 'side' feature, evaluates both LONG and SHORT
        and returns the direction with higher expected value.

        Args:
            features: Series with feature values
            use_meta: Whether to use meta model
            side: Optional explicit side (1=LONG, -1=SHORT). If None and 'side' in features,
                  evaluates both directions.

        Returns:
            Dictionary with prediction scores:
                - y_prob: Probability of target hit
                - score_ev: EV score (p_target - p_stop)
                - p_stop: Probability of stop hit (if available)
                - p_target: Probability of target hit (if available)
                - p_vertical: Probability of vertical exit (if available)
                - side: Recommended side (1=LONG, -1=SHORT, 0=skip) if bidirectional
                - score_ev_long: EV score for LONG (if bidirectional)
                - score_ev_short: EV score for SHORT (if bidirectional)
                - meta_prob: Meta model probability (if use_meta=True)
        """
        # Prepare features as DataFrame to preserve feature names for sklearn
        X = pd.DataFrame(
            [features[self.feature_columns].to_numpy(dtype=float)],
            columns=self.feature_columns,
        )

        # Apply preprocessing
        X_scaled = self._preprocess(X)
        
        # For dual-model bundles, evaluate both LONG and SHORT
        if self.has_dual_model:
            proba_long, proba_short = self.model.predict_proba_dual(X_scaled)

            # Get class ordering from the underlying model
            # DualSideModel wraps long_model and short_model, check long_model for classes
            # Typically classes are [0, 1, 2] representing [stop, vertical, target]
            if hasattr(self.model.long_model, 'classes_') and self.model.long_model.classes_ is not None:
                classes = list(self.model.long_model.classes_)
            elif hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                classes = list(self.model.classes_)
            else:
                classes = [0, 1, 2]  # Default

            # Map outcome labels to indices
            # Class labels are typically [0, 1, 2] for [stop=-1, vertical=0, target=1]
            if classes == [0, 1, 2]:
                stop_idx, vertical_idx, target_idx = 0, 1, 2
            else:
                # Try to find actual outcome values in classes
                try:
                    target_idx = classes.index(1)
                    stop_idx = classes.index(-1)
                    vertical_idx = classes.index(0)
                except ValueError as e:
                    raise ValueError(
                        f"Model has unexpected class encoding: {classes}. "
                        f"Expected to find -1, 0, 1 for stop/vertical/target outcomes. "
                        f"Error: {e}"
                    )

                # Validate indices are sane
                if not (0 <= stop_idx < 3 and 0 <= vertical_idx < 3 and 0 <= target_idx < 3):
                    raise ValueError(
                        f"Invalid class indices after mapping: "
                        f"stop={stop_idx}, vertical={vertical_idx}, target={target_idx}"
                    )

            score_ev_long = float(proba_long[0, target_idx] - proba_long[0, stop_idx])
            score_ev_short = float(proba_short[0, target_idx] - proba_short[0, stop_idx])

            if side is not None:
                if side > 0:
                    chosen_side = 1
                    proba = proba_long
                    chosen_score_ev = score_ev_long
                else:
                    chosen_side = -1
                    proba = proba_short
                    chosen_score_ev = score_ev_short
            else:
                if score_ev_long > score_ev_short and score_ev_long > 0:
                    chosen_side = 1
                    proba = proba_long
                    chosen_score_ev = score_ev_long
                elif score_ev_short > score_ev_long and score_ev_short > 0:
                    chosen_side = -1
                    proba = proba_short
                    chosen_score_ev = score_ev_short
                elif score_ev_long == score_ev_short and score_ev_long > 0:
                    chosen_side = 1
                    proba = proba_long
                    chosen_score_ev = score_ev_long
                else:
                    chosen_side = 0
                    proba = proba_long
                    chosen_score_ev = 0.0

            pred = {
                'p_stop': float(proba[0, stop_idx]),
                'p_target': float(proba[0, target_idx]),
                'p_vertical': float(proba[0, vertical_idx]) if vertical_idx is not None else None,
                'y_prob': float(proba[0, target_idx]),
                'score_ev': chosen_score_ev,
                'side': chosen_side,
                'score_ev_long': score_ev_long,
                'score_ev_short': score_ev_short,
            }

        # For bidirectional models with side feature, evaluate both LONG and SHORT
        elif self.has_side_feature and side is None:
            side_col = 'side'

            # Evaluate LONG (side=1)
            X_long = X_scaled.copy()
            if isinstance(X_long, pd.DataFrame):
                X_long.iloc[0, X_long.columns.get_loc(side_col)] = 1.0
            else:
                side_idx = self.feature_columns.index(side_col)
                X_long[0, side_idx] = 1.0
            proba_long = self.model.predict_proba(X_long)

            # Evaluate SHORT (side=-1)
            X_short = X_scaled.copy()
            if isinstance(X_short, pd.DataFrame):
                X_short.iloc[0, X_short.columns.get_loc(side_col)] = -1.0
            else:
                side_idx = self.feature_columns.index(side_col)
                X_short[0, side_idx] = -1.0
            proba_short = self.model.predict_proba(X_short)

            n_classes = proba_long.shape[1]

            if n_classes == 2:
                # Binary model: class 0 = stop, class 1 = target
                stop_idx_l, target_idx_l = 0, 1
                vertical_idx_l = None
            else:
                # Multiclass: resolve indices from model classes
                if hasattr(self.model, 'classes_'):
                    target_idx_l = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 2
                    stop_idx_l = list(self.model.classes_).index(-1) if -1 in self.model.classes_ else 0
                    vertical_idx_l = list(self.model.classes_).index(0) if 0 in self.model.classes_ else 1
                else:
                    stop_idx_l, vertical_idx_l, target_idx_l = 0, 1, 2

            score_ev_long = float(proba_long[0, target_idx_l] - proba_long[0, stop_idx_l])
            score_ev_short = float(proba_short[0, target_idx_l] - proba_short[0, stop_idx_l])

            # Choose the side with better positive EV
            if score_ev_long > score_ev_short and score_ev_long > 0:
                chosen_side = 1
                proba = proba_long
                chosen_score_ev = score_ev_long
            elif score_ev_short > score_ev_long and score_ev_short > 0:
                chosen_side = -1
                proba = proba_short
                chosen_score_ev = score_ev_short
            else:
                # Neither side has positive EV - skip trade
                chosen_side = 0
                proba = proba_long  # Use LONG for reporting
                chosen_score_ev = 0.0

            pred = {
                'p_stop': float(proba[0, stop_idx_l]),
                'p_target': float(proba[0, target_idx_l]),
                'p_vertical': float(proba[0, vertical_idx_l]) if vertical_idx_l is not None else 0.0,
                'y_prob': float(proba[0, target_idx_l]),
                'score_ev': chosen_score_ev,
                'side': chosen_side,
                'score_ev_long': score_ev_long,
                'score_ev_short': score_ev_short,
            }
            
        else:
            # Non-bidirectional or explicit side provided
            if self.has_side_feature and side is not None:
                if isinstance(X_scaled, pd.DataFrame):
                    X_scaled.iloc[0, X_scaled.columns.get_loc('side')] = float(side)
                else:
                    side_idx = self.feature_columns.index('side')
                    X_scaled[0, side_idx] = float(side)
            
            # Generate primary prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)

                # Check if multiclass (3 outcomes: stop, vertical, target)
                if proba.shape[1] == 3:
                    # proba[0, 0] = stop loss hit (-1)
                    # proba[0, 1] = vertical exit (0)
                    # proba[0, 2] = profit target hit (1)
                    p_stop = float(proba[0, 0])
                    p_target = float(proba[0, 2])
                    p_vertical = float(proba[0, 1])
                    
                    # Calculate EV score
                    # For Long: P(Up) - P(Down)
                    score_ev = float(p_target - p_stop)
                    
                    # Determine side based on EV
                    # If score_ev is positive -> LONG (side 1)
                    # If score_ev is negative -> SHORT (side -1)
                    if score_ev > 0:
                        side = 1
                        final_score = score_ev
                    else:
                        side = -1
                        final_score = abs(score_ev)
                    
                    pred = {
                        'p_stop': p_stop,
                        'p_target': p_target,
                        'p_vertical': p_vertical,
                        'y_prob': p_target if side == 1 else p_stop,
                        'score_ev': final_score,
                        'raw_score_ev': score_ev,
                        'side': side
                    }
                else:
                    # Binary classification
                    # Assumes class 0 = negative outcome (stop), class 1 = positive outcome (target)
                    if proba.shape[1] > 1:
                        p_stop = float(proba[0, 0])  # Probability of class 0 (negative)
                        p_target = float(proba[0, 1])  # Probability of class 1 (positive)
                    else:
                        # Single column - treat as probability of positive class
                        p_target = float(proba[0, 0])
                        p_stop = 1.0 - p_target
                    
                    # Calculate EV score
                    score_ev = p_target - p_stop
                    
                    # Determine side based on which class is more likely
                    if score_ev > 0:
                        side = 1  # LONG
                        final_score = score_ev
                    else:
                        side = -1  # SHORT
                        final_score = abs(score_ev)
                    
                    pred = {
                        'p_stop': p_stop,
                        'p_target': p_target,
                        'p_vertical': 0.0,  # No vertical exit in binary classification
                        'y_prob': p_target if side == 1 else p_stop,
                        'score_ev': final_score,
                        'raw_score_ev': score_ev,
                        'side': side
                    }
            else:
                # Regression model
                y_pred = self.model.predict(X_scaled)
                score = float(y_pred[0])
                pred = {
                    'y_prob': abs(score),
                    'score_ev': abs(score),
                    'raw_score_ev': score,
                    'side': 1 if score > 0 else -1
                }

        # Generate routed meta prediction if requested.
        if use_meta and self.meta_routes:
            route_matches = self._predict_routed_meta(
                features=features,
                primary_prediction=pred,
                combined_regime=combined_regime,
            )
            pred["meta_routes"] = route_matches
            eligible_routes = [route for route in route_matches if route.get("eligible", False)]
            if len(eligible_routes) > 1:
                raise ValueError(
                    "Multiple meta routes matched one live event: "
                    f"{[route['name'] for route in eligible_routes]}"
                )
            if eligible_routes:
                route = eligible_routes[0]
                pred["meta_prob"] = float(route["meta_prob"])
                pred["meta_route"] = route["name"]
                pred["meta_threshold"] = float(route["threshold_meta"])
                pred["meta_primary_threshold"] = float(route["threshold_primary"])
            elif route_matches:
                pred["meta_route_reason"] = ", ".join(
                    f"{route['name']}:{route.get('reason', 'not_applied')}"
                    for route in route_matches
                )

        # Generate legacy single-route meta prediction if requested
        elif use_meta and self.meta_model is not None:
            # Add primary prediction to features for meta model
            meta_features = self._build_meta_features(features, pred)

            X_meta = meta_features[self.meta_feature_columns].to_numpy(dtype=float, copy=False).reshape(1, -1)
            X_meta_scaled = self._preprocess_meta(X_meta)

            if hasattr(self.meta_model, 'predict_proba'):
                meta_proba = self.meta_model.predict_proba(X_meta_scaled)
                pred['meta_prob'] = float(meta_proba[0, 1] if meta_proba.shape[1] > 1 else meta_proba[0, 0])
            else:
                meta_pred = self.meta_model.predict(X_meta_scaled)
                pred['meta_prob'] = float(meta_pred[0])

        return pred

    def _load_meta_routes(self) -> list[dict[str, Any]]:
        raw_routes = self.bundle.get("meta_routes", []) or []
        routes: list[dict[str, Any]] = []
        for idx, route in enumerate(raw_routes):
            if not route:
                continue
            model = route.get("model") or route.get("meta_model")
            preprocessor = route.get("preprocessor") or route.get("meta_preprocessor")
            feature_columns = route.get("feature_columns") or route.get("meta_feature_columns") or []
            if model is None or not feature_columns:
                logger.warning("Skipping incomplete meta route at index %d", idx)
                continue
            routes.append(
                {
                    "name": str(route.get("name", f"meta_route_{idx}")),
                    "side": str(route.get("side", "both")).lower(),
                    "regimes": [int(r) for r in (route.get("regimes", []) or [])],
                    "threshold_primary": float(
                        route.get(
                            "threshold_primary",
                            self.thresholds.get("primary_threshold", 0.10),
                        )
                    ),
                    "threshold_meta": float(
                        route.get(
                            "threshold_meta",
                            self.thresholds.get("meta_threshold", 0.50),
                        )
                    ),
                    "model": model,
                    "preprocessor": preprocessor,
                    "feature_columns": list(feature_columns),
                }
            )
        return routes

    @staticmethod
    def _route_side_matches(route_side: str, prediction_side: int) -> bool:
        if route_side == "long":
            return prediction_side > 0
        if route_side == "short":
            return prediction_side < 0
        return prediction_side != 0

    def _build_meta_features(
        self,
        features: pd.Series,
        prediction: Dict[str, float],
    ) -> pd.Series:
        return pd.concat(
            [
                features[self.feature_columns],
                pd.Series(
                    {
                        "p_primary": prediction["y_prob"],
                        "p_primary_logit": np.log(
                            prediction["y_prob"] / (1 - prediction["y_prob"] + 1e-8)
                        ),
                    }
                ),
            ]
        )

    def _predict_routed_meta(
        self,
        *,
        features: pd.Series,
        primary_prediction: Dict[str, float],
        combined_regime: int | None,
    ) -> list[dict[str, Any]]:
        side = int(primary_prediction.get("side", 0) or 0)
        if side == 0:
            return []

        meta_features = self._build_meta_features(features, primary_prediction)
        route_results: list[dict[str, Any]] = []
        for route in self.meta_routes:
            result: dict[str, Any] = {
                "name": route["name"],
                "side": route["side"],
                "regimes": route["regimes"],
                "threshold_primary": route["threshold_primary"],
                "threshold_meta": route["threshold_meta"],
                "eligible": False,
            }

            if not self._route_side_matches(route["side"], side):
                result["reason"] = "side_mismatch"
                route_results.append(result)
                continue
            if route["regimes"] and combined_regime not in route["regimes"]:
                result["reason"] = "regime_mismatch"
                route_results.append(result)
                continue
            if float(primary_prediction.get("y_prob", 0.0)) < float(route["threshold_primary"]):
                result["reason"] = "below_route_primary_threshold"
                route_results.append(result)
                continue

            X_meta = (
                meta_features[route["feature_columns"]]
                .to_numpy(dtype=float, copy=False)
                .reshape(1, -1)
            )
            X_meta_scaled = self._preprocess_meta(X_meta, route.get("preprocessor"))
            model = route["model"]
            if hasattr(model, "predict_proba"):
                meta_proba = model.predict_proba(X_meta_scaled)
                result["meta_prob"] = float(
                    meta_proba[0, 1] if meta_proba.shape[1] > 1 else meta_proba[0, 0]
                )
            else:
                meta_pred = model.predict(X_meta_scaled)
                result["meta_prob"] = float(meta_pred[0])
            result["eligible"] = True
            result["reason"] = "eligible"
            route_results.append(result)

        return route_results

    def _preprocess(self, X):
        """
        Apply preprocessing to features.

        Args:
            X: Raw feature matrix (DataFrame or ndarray)

        Returns:
            Preprocessed feature matrix (same type as input)
        """
        # Handle both DataFrame and ndarray inputs
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = X.columns if is_dataframe else None

        # Convert to numpy for processing
        X_array = X.to_numpy(dtype=float, copy=False) if is_dataframe else X.astype(float, copy=False)

        # Impute missing values
        if self.impute == 'median':
            mask = np.isnan(X_array)
            X_array[mask] = np.take(self.medians, np.where(mask)[1])
        elif self.impute == 'zero':
            X_array = np.nan_to_num(X_array, 0.0)

        # Scale
        if self.scaler == 'standard':
            X_scaled = (X_array - self.means) / (self.stds + 1e-8)
        elif self.scaler == 'minmax':
            # MinMax not stored in state, just use X as-is
            X_scaled = X_array
        else:
            X_scaled = X_array

        # Return same type as input
        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=columns)
        return X_scaled

    def _preprocess_meta(
        self,
        X: np.ndarray,
        preprocessor_state: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Apply preprocessing to meta features.

        Args:
            X: Raw meta feature matrix

        Returns:
            Preprocessed meta feature matrix
        """
        state = preprocessor_state if preprocessor_state is not None else self.meta_preprocessor_state
        if state is None:
            return X

        impute = state.get('impute', 'median')
        scaler = state.get('scaler', 'standard')
        medians = np.array(state['medians']) if state.get("medians") is not None else None
        means = np.array(state['means']) if state.get("means") is not None else None
        stds = np.array(state['stds']) if state.get("stds") is not None else None

        # Ensure numeric
        X = X.astype(float, copy=False)

        # Impute
        if impute == 'median' and medians is not None:
            mask = np.isnan(X)
            X[mask] = np.take(medians, np.where(mask)[1])
        elif impute == 'zero':
            X = np.nan_to_num(X, 0.0)

        # Scale
        if scaler == 'standard' and means is not None and stds is not None:
            X_scaled = (X - means) / (stds + 1e-8)
        else:
            X_scaled = X

        return X_scaled

    def should_trade(
        self,
        prediction: Dict[str, float],
        primary_threshold: Optional[float] = None,
        primary_threshold_long: Optional[float] = None,
        primary_threshold_short: Optional[float] = None,
        meta_threshold: Optional[float] = None,
        require_meta_approval: bool = False,
        check_negative_edge: bool = True,
        allowed_directions: Optional[list[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be executed based on prediction.

        Args:
            prediction: Prediction dictionary from predict()
            primary_threshold: Primary model threshold (overrides default)
            primary_threshold_long: Optional LONG-only override threshold
            primary_threshold_short: Optional SHORT-only override threshold
            meta_threshold: Meta model threshold
            require_meta_approval: Whether meta approval is required
            check_negative_edge: If True, reject trades where p_stop >= p_target
            allowed_directions: Optional allow-list (e.g., ["SHORT"] or ["LONG"]).

        Returns:
            (should_trade, reason) tuple
        """
        # Optional direction allow-list (useful for quickly disabling a losing side).
        if allowed_directions:
            allowed = {str(d).strip().upper() for d in allowed_directions if str(d).strip()}
            # Infer direction from prediction['side'] when available.
            side = prediction.get("side", 0.0)
            try:
                side_i = int(side)
            except Exception:
                side_i = 0

            if side_i > 0 and "LONG" not in allowed:
                return False, "direction_blocked (LONG)"
            if side_i < 0 and "SHORT" not in allowed:
                return False, "direction_blocked (SHORT)"

        # Use thresholds from arguments or defaults
        base_primary_thresh = primary_threshold or self.primary_threshold
        meta_thresh = meta_threshold or prediction.get("meta_threshold") or 0.5

        # Check for negative edge (sanity filter) - direction-aware
        if check_negative_edge:
            p_stop = prediction.get('p_stop', 0.0)
            p_target = prediction.get('p_target', 0.0)
            side = prediction.get('side', 0)

            if side == 0:
                return False, "no_direction"

            # With bidirectional evaluation (has_side_feature), p_stop/p_target
            # come from the chosen side's perspective. For both LONG and SHORT,
            # p_stop >= p_target means negative edge.
            # Without bidirectional (non-side models), p_stop/p_target are from
            # LONG perspective: SHORT signals naturally have p_stop > p_target.
            if self.has_side_feature:
                # Bidirectional: both sides use same check
                if p_stop >= p_target:
                    return False, f"negative_edge (side={side}: p_stop={p_stop:.3f} >= p_target={p_target:.3f})"
            else:
                # Non-bidirectional: LONG perspective probabilities
                if side == 1 and p_stop >= p_target:
                    return False, f"negative_edge (LONG: p_stop={p_stop:.3f} >= p_target={p_target:.3f})"
                elif side == -1 and p_target >= p_stop:
                    return False, f"negative_edge (SHORT: p_target={p_target:.3f} >= p_stop={p_stop:.3f})"

        # Determine which directional threshold to apply (if configured)
        score = prediction.get("score_ev", prediction.get("y_prob", 0.0))
        side = prediction.get("side", 0.0)
        try:
            side_i = int(side)
        except Exception:
            side_i = 0

        primary_thresh = base_primary_thresh
        if side_i > 0 and primary_threshold_long is not None:
            primary_thresh = float(primary_threshold_long)
        elif side_i < 0 and primary_threshold_short is not None:
            primary_thresh = float(primary_threshold_short)

        # Check primary threshold
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
            'has_meta_model': self.meta_model is not None or bool(self.meta_routes),
            'meta_route_count': len(self.meta_routes),
            'meta_route_names': [route["name"] for route in self.meta_routes],
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
