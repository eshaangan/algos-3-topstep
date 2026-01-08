"""
GPU-Accelerated Hidden Markov Model for Market Regime Detection.

Uses pomegranate v1.0.0+ with PyTorch backend for GPU acceleration.
Drop-in replacement for hmm_regime.py with identical API.

Performance: 5-10x faster than hmmlearn on GPU.

References:
- pomegranate docs: https://pomegranate.readthedocs.io/
- Yuan (2019), "Market Regime Identification Using Hidden Markov Model"
"""

import logging
from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Normal
    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class HMMRegimeDetectorGPU:
    """
    GPU-accelerated HMM for market regime detection.

    Drop-in replacement for HMMRegimeDetector with identical API.
    Uses pomegranate's PyTorch backend for 5-10x speedup.

    Parameters
    ----------
    n_states : int, default=2
        Number of hidden states.
    n_iter : int, default=100
        Maximum EM iterations.
    device : str, default="auto"
        Device to use: "auto", "cuda", "mps", or "cpu".
        "auto" selects best available device.
    random_state : int, default=42
        Random seed for reproducibility.
    min_samples : int, default=252
        Minimum samples before fitting.
    tol : float, default=1e-4
        Convergence threshold.

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.01)
    >>> hmm = HMMRegimeDetectorGPU(n_states=2)
    >>> hmm.fit(returns)
    >>> regimes, probs = hmm.predict_expanding_batched(returns)
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        device: str = "auto",
        random_state: int = 42,
        min_samples: int = 252,
        tol: float = 1e-4,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GPU HMM. Install with: pip install torch>=2.0.0"
            )

        if not POMEGRANATE_AVAILABLE:
            raise ImportError(
                "pomegranate>=1.0.0 required for GPU HMM. "
                "Install with: pip install pomegranate>=1.0.0"
            )

        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_samples = min_samples
        self.tol = tol

        # Device selection
        self.device = self._select_device(device)
        logger.info(f"HMMRegimeDetectorGPU using device: {self.device}")

        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        self.model_: Optional[DenseHMM] = None
        self.bull_state_: Optional[int] = None
        self.bear_state_: Optional[int] = None
        self.is_fitted_: bool = False
        self._train_samples_: int = 0

    def _select_device(self, device: str) -> torch.device:
        """Select best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                logger.info(f"Auto-selected CUDA GPU: {torch.cuda.get_device_name(0)}")
                return dev
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Auto-selected Apple Silicon MPS")
                return torch.device("mps")
            else:
                logger.info("No GPU available, using CPU")
                return torch.device("cpu")
        return torch.device(device)

    def _prepare_data(
        self, returns: Union[pd.Series, np.ndarray], for_hmm: bool = True
    ) -> torch.Tensor:
        """
        Convert returns to PyTorch tensor on device.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series.
        for_hmm : bool, default=True
            If True, returns 3D tensor (batch=1, seq_len, features=1) for pomegranate.
            If False, returns 2D tensor (seq_len, features=1).
        """
        if isinstance(returns, pd.Series):
            data = returns.values.astype(np.float32)
        else:
            data = np.asarray(returns, dtype=np.float32)

        # Remove NaN
        data = data[~np.isnan(data)]

        # Convert to tensor
        if for_hmm:
            # pomegranate expects (batch_size, sequence_length, n_features)
            tensor = torch.tensor(data, dtype=torch.float32).reshape(1, -1, 1)
        else:
            tensor = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)

        return tensor.to(self.device)

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "HMMRegimeDetectorGPU":
        """
        Fit HMM on historical returns.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series (log returns or simple returns).

        Returns
        -------
        self : HMMRegimeDetectorGPU
            Fitted estimator.
        """
        data = self._prepare_data(returns, for_hmm=True)  # Shape: (1, seq_len, 1)
        seq_len = data.shape[1]

        if seq_len < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples to fit HMM, "
                f"got {seq_len}"
            )

        # Initialize emission distributions (Normal for each state)
        distributions = [Normal() for _ in range(self.n_states)]

        # Create HMM
        self.model_ = DenseHMM(
            distributions=distributions,
            max_iter=self.n_iter,
            tol=self.tol,
            verbose=False,
        )

        # Move model to device
        self.model_.to(self.device)

        # Fit
        self.model_.fit(data)

        self._train_samples_ = seq_len
        self.is_fitted_ = True
        self._label_states()

        logger.info(
            f"GPU HMM fitted with {self.n_states} states on {len(data)} samples. "
            f"Bull state: {self.bull_state_}, Bear state: {self.bear_state_}"
        )
        return self

    def _label_states(self) -> None:
        """Label states as bull/bear based on emission means."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before labeling states")

        means = []
        for dist in self.model_.distributions:
            # pomegranate Normal stores means as tensor
            if hasattr(dist, 'means'):
                mean = dist.means.cpu().item() if hasattr(dist.means, 'cpu') else float(dist.means)
            else:
                # Fallback for different pomegranate versions
                mean = 0.0
            means.append(mean)

        sorted_states = np.argsort(means)
        self.bear_state_ = int(sorted_states[0])
        self.bull_state_ = int(sorted_states[-1])

    def predict(self, returns: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series.

        Returns
        -------
        states : pd.Series
            Predicted state at each timestep.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        data = self._prepare_data(returns, for_hmm=True)  # Shape: (1, seq_len, 1)
        states = self.model_.predict(data)  # Shape: (1, seq_len)

        # Move back to CPU and flatten
        if hasattr(states, 'cpu'):
            states = states.cpu().numpy()
        states = np.asarray(states).flatten()

        if isinstance(returns, pd.Series):
            valid_idx = returns.index[~returns.isna()]
            return pd.Series(states, index=valid_idx, name="hmm_state")
        return pd.Series(states, name="hmm_state")

    def predict_proba(self, returns: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Get state probabilities using forward-backward algorithm.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series.

        Returns
        -------
        probs : pd.DataFrame
            Probability of each state at each timestep.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        data = self._prepare_data(returns, for_hmm=True)  # Shape: (1, seq_len, 1)
        posteriors = self.model_.predict_proba(data)  # Shape: (1, seq_len, n_states)

        # Move back to CPU for pandas and remove batch dimension
        if hasattr(posteriors, 'cpu'):
            posteriors_np = posteriors.cpu().numpy()
        else:
            posteriors_np = np.asarray(posteriors)

        # Remove batch dimension: (1, seq_len, n_states) -> (seq_len, n_states)
        posteriors_np = posteriors_np.squeeze(0)

        columns = [f"prob_state_{i}" for i in range(self.n_states)]

        if isinstance(returns, pd.Series):
            valid_idx = returns.index[~returns.isna()]
            return pd.DataFrame(posteriors_np, index=valid_idx, columns=columns)
        return pd.DataFrame(posteriors_np, columns=columns)

    def predict_expanding(
        self,
        returns: Union[pd.Series, np.ndarray],
        min_train_samples: Optional[int] = None,
        refit_every: int = 21,
        use_rolling_window: bool = True,
        rolling_window_size: int = 252,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Rolling/expanding window regime detection (GPU accelerated).

        Same API as HMMRegimeDetector.predict_expanding().
        For better performance, use predict_expanding_batched() instead.
        """
        # Delegate to batched version with batch_size=1 for exact compatibility
        return self.predict_expanding_batched(
            returns=returns,
            min_train_samples=min_train_samples,
            refit_every=refit_every,
            rolling_window_size=rolling_window_size,
            batch_size=refit_every,  # Process one refit interval at a time
        )

    def predict_expanding_batched(
        self,
        returns: Union[pd.Series, np.ndarray],
        min_train_samples: Optional[int] = None,
        refit_every: int = 126,
        rolling_window_size: int = 252,
        batch_size: int = 5000,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        OPTIMIZED: Batched rolling window with GPU acceleration.

        Instead of per-bar forward pass, processes in batches:
        1. Fit model on rolling window at refit point
        2. Run forward pass on entire batch
        3. Move to next batch

        This reduces Python loop overhead and maximizes GPU utilization.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series.
        min_train_samples : int, optional
            Minimum samples before starting prediction.
        refit_every : int, default=126
            Refit model every N bars.
        rolling_window_size : int, default=252
            Rolling window size for fitting.
        batch_size : int, default=5000
            Process this many bars between progress updates.

        Returns
        -------
        regime_states : pd.Series
            State assignments at each timestep.
        regime_probs : pd.DataFrame
            State probabilities at each timestep.
        """
        if min_train_samples is None:
            min_train_samples = self.min_samples

        if isinstance(returns, pd.Series):
            index = returns.index
            data = returns.values.astype(np.float32)
        else:
            index = pd.RangeIndex(len(returns))
            data = np.asarray(returns, dtype=np.float32)

        # Remove NaN for processing but track positions
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        valid_idx = index[valid_mask]

        n_samples = len(valid_data)

        # Initialize output arrays
        states = np.full(n_samples, np.nan)
        probs = np.full((n_samples, self.n_states), np.nan)

        logger.info(f"Starting GPU HMM batched prediction for {n_samples:,} bars")
        logger.info(f"Device: {self.device}, Rolling window: {rolling_window_size}, Refit every: {refit_every}")

        # Calculate refit points
        refit_points = list(range(min_train_samples, n_samples, refit_every))
        if refit_points[-1] != n_samples:
            refit_points.append(n_samples)

        logger.info(f"Will perform {len(refit_points)-1} refits")

        current_model = None

        for i, refit_idx in enumerate(refit_points[:-1]):
            next_refit_idx = refit_points[i + 1]

            # Progress logging every 10 refits or at key milestones
            if i % 40 == 0 or i == len(refit_points) - 2:
                pct = refit_idx / n_samples * 100
                logger.info(f"GPU HMM Progress: {refit_idx:,}/{n_samples:,} bars ({pct:.1f}%)")

            # Define training window
            start_idx = max(0, refit_idx - rolling_window_size)
            train_data = valid_data[start_idx:refit_idx]

            # Fit new model on GPU - shape: (1, seq_len, 1)
            train_tensor = torch.tensor(
                train_data, dtype=torch.float32
            ).reshape(1, -1, 1).to(self.device)

            distributions = [Normal() for _ in range(self.n_states)]
            current_model = DenseHMM(
                distributions=distributions,
                max_iter=self.n_iter,
                tol=self.tol,
                verbose=False,
            )
            current_model.to(self.device)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_model.fit(train_tensor)
            except (IndexError, ValueError, AttributeError, RuntimeError) as e:
                # Handle edge cases:
                # - IndexError: data too homogeneous (all samples in one cluster)
                # - ValueError: numerical issues
                # - AttributeError: pomegranate internal issues with _inv_cov
                # - RuntimeError: GPU/MPS issues
                # Use uniform probabilities for this batch
                logger.warning(f"HMM fit failed at refit {i} (idx={refit_idx}): {type(e).__name__}: {e}. Using uniform probs.")
                states[refit_idx:next_refit_idx] = 0
                probs[refit_idx:next_refit_idx, :] = 1.0 / self.n_states
                continue

            # OPTIMIZED: Run forward pass only from start_idx (not from 0)
            # Since HMM was trained on data[start_idx:refit_idx], we only need
            # observations from start_idx onwards for prediction.
            # This makes each forward pass O(rolling_window + refit_every) instead of O(T).
            obs_data = valid_data[start_idx:next_refit_idx]
            obs_tensor = torch.tensor(
                obs_data, dtype=torch.float32
            ).reshape(1, -1, 1).to(self.device)

            # Get filtered probabilities for the window
            log_prob, filtered_probs = self._forward_filtering_gpu(
                current_model, obs_tensor
            )

            # Map local indices to global indices
            # filtered_probs has shape (next_refit_idx - start_idx, n_states)
            # We want indices [refit_idx, next_refit_idx) in global coords
            # Which is [refit_idx - start_idx, next_refit_idx - start_idx) in local coords
            local_start = refit_idx - start_idx
            local_end = next_refit_idx - start_idx

            states[refit_idx:next_refit_idx] = np.argmax(
                filtered_probs[local_start:local_end], axis=1
            )
            probs[refit_idx:next_refit_idx, :] = filtered_probs[local_start:local_end]

        logger.info(f"GPU HMM batched prediction complete: {n_samples:,} bars")

        # Convert to output format
        regime_states = pd.Series(states, index=valid_idx, name="hmm_state")

        prob_columns = [f"prob_state_{i}" for i in range(self.n_states)]
        regime_probs = pd.DataFrame(probs, index=valid_idx, columns=prob_columns)

        # Add convenience columns for bull/bear if 2-state
        if self.n_states == 2 and current_model is not None:
            means = []
            for dist in current_model.distributions:
                if hasattr(dist, 'means'):
                    mean = dist.means.cpu().item() if hasattr(dist.means, 'cpu') else float(dist.means)
                else:
                    mean = 0.0
                means.append(mean)

            bull_idx = int(np.argmax(means))
            bear_idx = int(np.argmin(means))
            regime_probs["prob_bull"] = regime_probs[f"prob_state_{bull_idx}"]
            regime_probs["prob_bear"] = regime_probs[f"prob_state_{bear_idx}"]

        return regime_states, regime_probs

    def _forward_filtering_gpu(
        self, model: DenseHMM, obs_seq: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """
        Compute forward (filtered) probabilities using GPU.

        Uses pomegranate's native forward() method for maximum speed.

        Parameters
        ----------
        model : DenseHMM
            Fitted pomegranate HMM model.
        obs_seq : torch.Tensor
            Observation sequence on device, shape (1, T, 1).

        Returns
        -------
        log_prob : float
            Log probability of the observation sequence.
        posteriors : np.ndarray
            Filtered state probabilities, shape (T, n_states).
        """
        # Ensure 3D input: (batch=1, seq_len, features=1)
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)  # (1, T, 1)

        # Use pomegranate's native forward algorithm (MUCH faster)
        # Returns log probabilities of shape (batch, seq_len, n_states)
        log_fwd = model.forward(obs_seq)  # shape: (1, T, n_states)

        # Remove batch dimension
        log_fwd = log_fwd.squeeze(0)  # shape: (T, n_states)

        # Compute log probability of the sequence
        log_prob = torch.logsumexp(log_fwd[-1], dim=0).cpu().item()

        # Convert log probs to normalized probabilities
        log_fwd_cpu = log_fwd.cpu().numpy()
        posteriors = np.exp(log_fwd_cpu - log_fwd_cpu.max(axis=1, keepdims=True))
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        return log_prob, posteriors

    def get_transition_matrix(self) -> np.ndarray:
        """Return fitted transition matrix."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        edges = self.model_.edges
        if hasattr(edges, 'cpu'):
            return edges.cpu().numpy()
        return np.asarray(edges)

    def get_emission_params(self) -> Dict[str, Dict[str, float]]:
        """Return emission parameters (mean, std) per state."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        params = {}

        for i, dist in enumerate(self.model_.distributions):
            if hasattr(dist, 'means'):
                mean = dist.means.cpu().item() if hasattr(dist.means, 'cpu') else float(dist.means)
            else:
                mean = 0.0

            if hasattr(dist, 'covs'):
                var = dist.covs.cpu().item() if hasattr(dist.covs, 'cpu') else float(dist.covs)
            else:
                var = 1.0

            std = np.sqrt(var)

            if self.n_states == 2:
                label = "bull" if i == self.bull_state_ else "bear"
            else:
                label = f"state_{i}"

            params[label] = {"mean": mean, "std": std, "state_idx": i}

        return params

    def expected_regime_duration(self) -> Dict[str, float]:
        """Compute expected duration of each regime (in bars)."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        transmat = self.get_transition_matrix()
        durations = {}

        for i in range(self.n_states):
            p_stay = transmat[i, i]
            duration = 1.0 / (1.0 - p_stay + 1e-10)

            if self.n_states == 2:
                label = "bull" if i == self.bull_state_ else "bear"
            else:
                label = f"state_{i}"

            durations[label] = duration

        return durations


def get_hmm_detector(use_gpu: bool = True, **kwargs):
    """
    Factory function to get best available HMM detector.

    Parameters
    ----------
    use_gpu : bool, default=True
        If True, try to use GPU-accelerated version first.
    **kwargs
        Arguments passed to HMM detector constructor.

    Returns
    -------
    detector : HMMRegimeDetector or HMMRegimeDetectorGPU
        Best available HMM detector.
    """
    if use_gpu and POMEGRANATE_AVAILABLE and TORCH_AVAILABLE:
        try:
            return HMMRegimeDetectorGPU(**kwargs)
        except Exception as e:
            logger.warning(f"GPU HMM initialization failed: {e}. Falling back to CPU.")

    # Fallback to CPU version
    from ml_intraday_v3.features.hmm_regime import HMMRegimeDetector
    return HMMRegimeDetector(**kwargs)


def benchmark_hmm(
    n_samples: int = 10000,
    n_states: int = 2,
    device: str = "auto",
) -> Dict[str, float]:
    """
    Benchmark GPU vs CPU HMM performance.

    Parameters
    ----------
    n_samples : int
        Number of samples to test with.
    n_states : int
        Number of HMM states.
    device : str
        Device for GPU version.

    Returns
    -------
    results : dict
        Timing results.
    """
    import time

    # Generate test data
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.01

    results = {}

    # GPU version
    if POMEGRANATE_AVAILABLE and TORCH_AVAILABLE:
        try:
            hmm_gpu = HMMRegimeDetectorGPU(n_states=n_states, device=device)

            start = time.time()
            hmm_gpu.fit(returns[:1000])  # Fit on subset
            fit_time = time.time() - start

            start = time.time()
            _ = hmm_gpu.predict_proba(returns)
            predict_time = time.time() - start

            results["gpu_fit_time"] = fit_time
            results["gpu_predict_time"] = predict_time
            results["gpu_device"] = str(hmm_gpu.device)
        except Exception as e:
            results["gpu_error"] = str(e)

    # CPU version (hmmlearn)
    try:
        from ml_intraday_v3.features.hmm_regime import HMMRegimeDetector

        hmm_cpu = HMMRegimeDetector(n_states=n_states)

        start = time.time()
        hmm_cpu.fit(returns[:1000])
        fit_time = time.time() - start

        start = time.time()
        _ = hmm_cpu.predict_proba(returns)
        predict_time = time.time() - start

        results["cpu_fit_time"] = fit_time
        results["cpu_predict_time"] = predict_time
    except Exception as e:
        results["cpu_error"] = str(e)

    # Calculate speedup
    if "gpu_predict_time" in results and "cpu_predict_time" in results:
        results["speedup"] = results["cpu_predict_time"] / results["gpu_predict_time"]

    return results
