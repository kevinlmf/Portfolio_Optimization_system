"""
Bayesian Posterior Updater

Online Bayesian learning system for adaptive portfolio optimization.

Key Features:
- Incremental posterior updates (no full retraining)
- Exponential forgetting for non-stationary markets
- Anomaly detection and robust updating
- Adaptive learning rates based on prediction errors
- Change point detection for regime transitions

Mathematical Framework:
    Posterior update (conjugate):
        p(θ_t | data_1:t) ∝ p(data_t | θ_t) * p(θ_t | data_1:t-1)

    With forgetting factor λ:
        p(θ_t | data_1:t) ∝ p(data_t | θ_t) * p(θ_t | data_1:t-1)^λ
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UpdateResult:
    """Container for update results."""
    updated_params: Dict           # Updated posterior parameters
    prediction_error: float        # Forecast error
    anomaly_score: float          # Anomaly detection score
    is_anomaly: bool              # Whether observation is anomalous
    learning_rate: float          # Effective learning rate used
    change_point_prob: float      # Probability of regime change


class ExponentialForgetting:
    """
    Implements exponential forgetting for non-stationary time series.

    Older observations receive exponentially decaying weights:
        w_t = λ^(T-t) where λ ∈ (0, 1]
    """

    def __init__(self, decay_factor: float = 0.95):
        """
        Initialize exponential forgetting.

        Args:
            decay_factor: Decay rate (λ). Higher = more memory
                         λ = 0.95: effective window ~20 periods
                         λ = 0.99: effective window ~100 periods
        """
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be in (0, 1]")

        self.decay_factor = decay_factor
        self.effective_window = int(1 / (1 - decay_factor)) if decay_factor < 1 else np.inf

    def compute_weights(self, n_periods: int) -> np.ndarray:
        """
        Compute forgetting weights for n periods.

        Args:
            n_periods: Number of time periods

        Returns:
            Normalized weights (most recent = 1.0)
        """
        t = np.arange(n_periods)
        weights = self.decay_factor ** (n_periods - 1 - t)
        return weights / weights.sum()

    def weighted_mean(self, values: np.ndarray) -> float:
        """Compute exponentially weighted mean."""
        weights = self.compute_weights(len(values))
        return np.sum(weights * values)

    def weighted_cov(self, data: np.ndarray) -> np.ndarray:
        """
        Compute exponentially weighted covariance.

        Args:
            data: 2D array (n_samples, n_features)

        Returns:
            Weighted covariance matrix
        """
        n, p = data.shape
        weights = self.compute_weights(n)

        # Weighted mean
        mu = np.sum(weights[:, np.newaxis] * data, axis=0)

        # Weighted covariance
        centered = data - mu
        cov = (centered.T @ np.diag(weights) @ centered) / (1 - np.sum(weights**2))

        return cov


class AnomalyDetector:
    """
    Detects anomalous observations using multiple methods.

    Methods:
    1. Mahalanobis distance (multivariate outliers)
    2. Z-score (univariate outliers)
    3. Prediction error threshold
    """

    def __init__(self,
                 method: str = 'mahalanobis',
                 threshold: float = 3.0,
                 min_history: int = 30):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method ('mahalanobis', 'zscore', 'prediction_error')
            threshold: Detection threshold (standard deviations)
            min_history: Minimum observations before detection
        """
        self.method = method
        self.threshold = threshold
        self.min_history = min_history

        self.history = deque(maxlen=100)
        self.mean = None
        self.cov = None

    def update_statistics(self, observation: np.ndarray):
        """Update running statistics."""
        self.history.append(observation)

        if len(self.history) >= self.min_history:
            data = np.array(self.history)
            self.mean = data.mean(axis=0)
            self.cov = np.cov(data.T)

    def detect(self, observation: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if observation is anomalous.

        Args:
            observation: New observation

        Returns:
            (is_anomaly, anomaly_score)
        """
        if len(self.history) < self.min_history:
            return False, 0.0

        if self.method == 'mahalanobis':
            score = self._mahalanobis_distance(observation)
        elif self.method == 'zscore':
            score = self._zscore(observation)
        else:
            score = 0.0

        is_anomaly = score > self.threshold

        # Update history
        self.update_statistics(observation)

        return is_anomaly, score

    def _mahalanobis_distance(self, observation: np.ndarray) -> float:
        """Compute Mahalanobis distance."""
        diff = observation - self.mean
        cov_inv = np.linalg.pinv(self.cov + np.eye(len(self.cov)) * 1e-6)
        distance = np.sqrt(diff @ cov_inv @ diff)
        return distance

    def _zscore(self, observation: np.ndarray) -> float:
        """Compute maximum z-score across dimensions."""
        std = np.sqrt(np.diag(self.cov))
        zscores = np.abs((observation - self.mean) / (std + 1e-8))
        return np.max(zscores)


class ChangePointDetector:
    """
    Detects structural breaks / regime changes.

    Uses CUSUM (Cumulative Sum) control chart approach.
    """

    def __init__(self,
                 threshold: float = 5.0,
                 drift: float = 0.5):
        """
        Initialize change point detector.

        Args:
            threshold: Detection threshold (h)
            drift: Allowable drift (k)
        """
        self.threshold = threshold
        self.drift = drift

        self.cumsum_pos = 0.0
        self.cumsum_neg = 0.0
        self.mean = 0.0
        self.std = 1.0
        self.n_obs = 0

    def update(self, observation: float) -> Tuple[bool, float]:
        """
        Update CUSUM and detect change point.

        Args:
            observation: New scalar observation (e.g., prediction error)

        Returns:
            (is_change_point, change_probability)
        """
        # Update running statistics
        self.n_obs += 1
        delta = observation - self.mean
        self.mean += delta / self.n_obs
        self.std = np.sqrt(((self.n_obs - 1) * self.std**2 + delta * (observation - self.mean)) / self.n_obs)

        # Standardize
        z = (observation - self.mean) / (self.std + 1e-8)

        # Update CUSUM
        self.cumsum_pos = max(0, self.cumsum_pos + z - self.drift)
        self.cumsum_neg = max(0, self.cumsum_neg - z - self.drift)

        # Detect change
        max_cumsum = max(self.cumsum_pos, self.cumsum_neg)
        is_change = max_cumsum > self.threshold

        # Change probability (sigmoid of CUSUM)
        change_prob = 1 / (1 + np.exp(-max_cumsum + self.threshold))

        if is_change:
            # Reset CUSUM after detection
            self.cumsum_pos = 0.0
            self.cumsum_neg = 0.0

        return is_change, change_prob


class AdaptiveLearningRate:
    """
    Adaptive learning rate based on prediction errors.

    Higher errors → higher learning rate (faster adaptation)
    Lower errors → lower learning rate (more stability)
    """

    def __init__(self,
                 base_rate: float = 0.1,
                 min_rate: float = 0.01,
                 max_rate: float = 0.5,
                 adaptation_speed: float = 0.1):
        """
        Initialize adaptive learning rate.

        Args:
            base_rate: Baseline learning rate
            min_rate: Minimum learning rate
            max_rate: Maximum learning rate
            adaptation_speed: How fast to adapt (α)
        """
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_speed = adaptation_speed

        self.current_rate = base_rate
        self.error_history = deque(maxlen=20)

    def update(self, prediction_error: float) -> float:
        """
        Update and return learning rate.

        Args:
            prediction_error: Absolute prediction error

        Returns:
            Updated learning rate
        """
        self.error_history.append(prediction_error)

        if len(self.error_history) < 5:
            return self.current_rate

        # Recent error vs historical average
        recent_error = np.mean(list(self.error_history)[-5:])
        historical_error = np.mean(list(self.error_history)[:-5]) if len(self.error_history) > 5 else recent_error

        error_ratio = recent_error / (historical_error + 1e-8)

        # Adjust learning rate
        if error_ratio > 1.5:
            # Errors increasing → increase learning rate
            self.current_rate = min(self.current_rate * (1 + self.adaptation_speed), self.max_rate)
        elif error_ratio < 0.5:
            # Errors decreasing → decrease learning rate
            self.current_rate = max(self.current_rate * (1 - self.adaptation_speed), self.min_rate)

        return self.current_rate


class BayesianPosteriorUpdater:
    """
    Main updater class integrating all online learning components.

    Performs robust, adaptive Bayesian updates with:
    - Exponential forgetting
    - Anomaly detection
    - Change point detection
    - Adaptive learning rates
    """

    def __init__(self,
                 decay_factor: float = 0.95,
                 anomaly_threshold: float = 3.0,
                 change_threshold: float = 5.0,
                 base_learning_rate: float = 0.1,
                 robust_mode: bool = True):
        """
        Initialize Bayesian updater.

        Args:
            decay_factor: Exponential forgetting factor
            anomaly_threshold: Anomaly detection threshold
            change_threshold: Change point detection threshold
            base_learning_rate: Base learning rate
            robust_mode: Enable robust updating (ignore anomalies)
        """
        self.decay_factor = decay_factor
        self.robust_mode = robust_mode

        # Components
        self.forgetting = ExponentialForgetting(decay_factor)
        self.anomaly_detector = AnomalyDetector(threshold=anomaly_threshold)
        self.change_detector = ChangePointDetector(threshold=change_threshold)
        self.learning_rate = AdaptiveLearningRate(base_rate=base_learning_rate)

        # State
        self.mu_posterior = None
        self.Sigma_posterior = None
        self.n_updates = 0
        self.update_history = []

    def initialize(self, mu_prior: np.ndarray, Sigma_prior: np.ndarray):
        """
        Initialize posterior with prior.

        Args:
            mu_prior: Prior mean
            Sigma_prior: Prior covariance
        """
        self.mu_posterior = mu_prior.copy()
        self.Sigma_posterior = Sigma_prior.copy()

    def update(self,
               observation: np.ndarray,
               predicted: Optional[np.ndarray] = None) -> UpdateResult:
        """
        Perform one-step Bayesian update.

        Args:
            observation: New observed returns
            predicted: Optional predicted returns (for error calculation)

        Returns:
            UpdateResult with updated parameters
        """
        if self.mu_posterior is None:
            raise ValueError("Updater not initialized. Call initialize() first.")

        # Compute prediction error
        if predicted is not None:
            prediction_error = np.linalg.norm(observation - predicted)
        else:
            prediction_error = np.linalg.norm(observation - self.mu_posterior)

        # Anomaly detection
        is_anomaly, anomaly_score = self.anomaly_detector.detect(observation)

        # Change point detection
        is_change, change_prob = self.change_detector.update(prediction_error)

        # Adaptive learning rate
        lr = self.learning_rate.update(prediction_error)

        # Perform update (if not anomalous or in non-robust mode)
        if not (self.robust_mode and is_anomaly):
            # Standard Bayesian update with forgetting
            self._bayesian_update(observation, lr)
        else:
            # Skip update for anomalous observations
            pass

        self.n_updates += 1

        result = UpdateResult(
            updated_params={
                'mu': self.mu_posterior.copy(),
                'Sigma': self.Sigma_posterior.copy()
            },
            prediction_error=prediction_error,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            learning_rate=lr,
            change_point_prob=change_prob
        )

        self.update_history.append(result)

        return result

    def _bayesian_update(self, observation: np.ndarray, learning_rate: float):
        """
        Core Bayesian update (conjugate Normal-Inverse-Wishart).

        Simplified update:
            μ_new = (1 - α) * μ_old + α * observation
            Σ_new = (1 - α) * Σ_old + α * (observation - μ_new)(observation - μ_new)'

        Where α = learning_rate adjusted by forgetting factor.
        """
        effective_lr = learning_rate * (1 - self.decay_factor + self.decay_factor / self.n_updates)

        # Update mean
        mu_old = self.mu_posterior.copy()
        self.mu_posterior = (1 - effective_lr) * mu_old + effective_lr * observation

        # Update covariance (exponentially weighted)
        diff_old = (observation - mu_old).reshape(-1, 1)
        self.Sigma_posterior = (
            (1 - effective_lr) * self.Sigma_posterior +
            effective_lr * (diff_old @ diff_old.T)
        )

    def get_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current posterior parameters.

        Returns:
            (mu_posterior, Sigma_posterior)
        """
        return self.mu_posterior, self.Sigma_posterior

    def reset_on_change(self, new_mu: np.ndarray, new_Sigma: np.ndarray):
        """
        Reset posterior after detected regime change.

        Args:
            new_mu: New posterior mean
            new_Sigma: New posterior covariance
        """
        self.mu_posterior = new_mu
        self.Sigma_posterior = new_Sigma
        self.n_updates = 0
        self.change_detector.cumsum_pos = 0.0
        self.change_detector.cumsum_neg = 0.0

    def get_update_diagnostics(self) -> pd.DataFrame:
        """
        Get diagnostics of update history.

        Returns:
            DataFrame with update statistics
        """
        if not self.update_history:
            return pd.DataFrame()

        diagnostics = pd.DataFrame([
            {
                'prediction_error': r.prediction_error,
                'anomaly_score': r.anomaly_score,
                'is_anomaly': r.is_anomaly,
                'learning_rate': r.learning_rate,
                'change_point_prob': r.change_point_prob
            }
            for r in self.update_history
        ])

        return diagnostics


class RegimeAwareUpdater:
    """
    Regime-aware version that maintains separate updaters per regime.
    """

    def __init__(self,
                 n_regimes: int = 5,
                 decay_factor: float = 0.95,
                 anomaly_threshold: float = 3.0):
        """
        Initialize regime-aware updater.

        Args:
            n_regimes: Number of regimes
            decay_factor: Forgetting factor
            anomaly_threshold: Anomaly threshold
        """
        self.n_regimes = n_regimes

        # Separate updater for each regime
        self.regime_updaters = {
            k: BayesianPosteriorUpdater(
                decay_factor=decay_factor,
                anomaly_threshold=anomaly_threshold
            )
            for k in range(n_regimes)
        }

    def initialize_all(self, mu_priors: Dict[int, np.ndarray], Sigma_priors: Dict[int, np.ndarray]):
        """Initialize all regime updaters."""
        for k in range(self.n_regimes):
            self.regime_updaters[k].initialize(mu_priors[k], Sigma_priors[k])

    def update_all(self,
                   observation: np.ndarray,
                   regime_probs: np.ndarray) -> Dict[int, UpdateResult]:
        """
        Update all regimes weighted by probabilities.

        Args:
            observation: New observation
            regime_probs: Current regime probabilities

        Returns:
            Dictionary of update results per regime
        """
        results = {}
        for k in range(self.n_regimes):
            if regime_probs[k] > 0.1:  # Only update active regimes
                # Weight observation by regime probability (optional)
                results[k] = self.regime_updaters[k].update(observation)

        return results

    def get_posteriors(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get all regime posteriors."""
        return {
            k: updater.get_posterior()
            for k, updater in self.regime_updaters.items()
        }


if __name__ == "__main__":
    # Example usage
    print("Bayesian Posterior Updater")
    print("=" * 60)
    print("Usage example:")
    print("""
    from strategy.bayesian_updater import BayesianPosteriorUpdater
    import numpy as np

    # Initialize
    updater = BayesianPosteriorUpdater(
        decay_factor=0.95,
        anomaly_threshold=3.0,
        robust_mode=True
    )

    # Set prior
    n_assets = 5
    mu_prior = np.zeros(n_assets)
    Sigma_prior = np.eye(n_assets) * 0.01
    updater.initialize(mu_prior, Sigma_prior)

    # Online updates
    for t in range(100):
        # New observation
        observation = np.random.randn(n_assets) * 0.01

        # Update
        result = updater.update(observation)

        print(f"t={t}: Error={result.prediction_error:.4f}, "
              f"Anomaly={result.is_anomaly}, LR={result.learning_rate:.4f}")

    # Get final posterior
    mu_post, Sigma_post = updater.get_posterior()
    """)
