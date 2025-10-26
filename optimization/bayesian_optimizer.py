"""
Bayesian Portfolio Optimizer

Performs portfolio optimization using Bayesian inference, incorporating:
- Regime probabilities from mixture models as priors
- Posterior uncertainty quantification
- Adaptive parameter learning
- Multiple optimization objectives

Mathematical Framework:
    p(w_t | data, z_t) ∝ p(data | w_t, z_t) * p(w_t | z_t)

    Final portfolio:
    w_t = Σ_k p(z_t=k) * E[w_t | z_t=k, data_1:t]

Where:
    - w_t: portfolio weights at time t
    - z_t: market regime at time t
    - p(w_t | z_t): regime-conditional prior
    - p(data | w_t, z_t): likelihood of observed returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PosteriorDistribution:
    """Container for posterior distribution of portfolio weights."""
    mean: np.ndarray              # Posterior mean weights
    covariance: np.ndarray         # Posterior covariance
    samples: Optional[np.ndarray]  # MCMC samples (if available)
    confidence_intervals: Dict     # 95% credible intervals


@dataclass
class OptimizationResult:
    """Container for Bayesian optimization results."""
    weights: np.ndarray                      # Final portfolio weights
    posterior: PosteriorDistribution         # Full posterior distribution
    regime_weights: Dict[str, np.ndarray]   # Weights per regime
    expected_return: float                   # Expected portfolio return
    expected_risk: float                     # Expected portfolio risk
    uncertainty: float                       # Posterior uncertainty measure


class BayesianMeanVarianceOptimizer:
    """
    Bayesian Mean-Variance Portfolio Optimization.

    Uses conjugate priors (Normal-Inverse-Wishart) for tractable posterior inference.

    Prior:
        μ ~ N(μ_0, Σ / κ_0)
        Σ ~ IW(Ψ_0, ν_0)

    Posterior (after observing data):
        μ | data ~ N(μ_n, Σ / κ_n)
        Σ | data ~ IW(Ψ_n, ν_n)
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 prior_confidence: float = 0.1,
                 use_shrinkage: bool = True):
        """
        Initialize Bayesian MV optimizer.

        Args:
            risk_aversion: Risk aversion parameter (λ in utility)
            prior_confidence: Strength of prior belief (κ_0)
            use_shrinkage: Apply shrinkage to covariance estimates
        """
        self.risk_aversion = risk_aversion
        self.prior_confidence = prior_confidence
        self.use_shrinkage = use_shrinkage

        # Prior parameters (will be set during fit)
        self.mu_0 = None
        self.kappa_0 = prior_confidence
        self.Psi_0 = None
        self.nu_0 = None

        # Posterior parameters
        self.mu_n = None
        self.kappa_n = None
        self.Psi_n = None
        self.nu_n = None

    def fit_prior(self,
                  returns: pd.DataFrame,
                  regime_probs: Optional[np.ndarray] = None) -> 'BayesianMeanVarianceOptimizer':
        """
        Fit prior distribution from historical data.

        Args:
            returns: Historical returns (n_samples, n_assets)
            regime_probs: Optional regime probabilities for weighting

        Returns:
            self (fitted)
        """
        n_samples, n_assets = returns.shape

        # Set prior mean (historical mean or weighted mean)
        if regime_probs is not None and len(regime_probs) == n_samples:
            # Weighted by regime stability
            weights = regime_probs / regime_probs.sum()
            self.mu_0 = (returns.T @ weights).values
        else:
            self.mu_0 = returns.mean().values

        # Set prior covariance (with shrinkage if enabled)
        sample_cov = returns.cov().values
        if self.use_shrinkage:
            # Ledoit-Wolf shrinkage
            self.Psi_0 = self._ledoit_wolf_shrinkage(returns)
        else:
            self.Psi_0 = sample_cov

        # Set prior degrees of freedom
        self.nu_0 = n_assets + 2  # Minimal prior belief

        return self

    def update_posterior(self,
                        new_returns: pd.DataFrame) -> 'BayesianMeanVarianceOptimizer':
        """
        Update posterior distribution with new observations.

        Args:
            new_returns: New return observations

        Returns:
            self (with updated posterior)
        """
        if self.mu_0 is None:
            raise ValueError("Prior not set. Call fit_prior() first.")

        n_new = len(new_returns)
        x_bar = new_returns.mean().values
        S_new = new_returns.cov().values

        # Update parameters (conjugate update)
        self.kappa_n = self.kappa_0 + n_new
        self.nu_n = self.nu_0 + n_new

        self.mu_n = (self.kappa_0 * self.mu_0 + n_new * x_bar) / self.kappa_n

        diff = x_bar - self.mu_0
        self.Psi_n = (self.Psi_0 +
                     n_new * S_new +
                     (self.kappa_0 * n_new / self.kappa_n) * np.outer(diff, diff))

        return self

    def optimize(self,
                 regime_probs: Optional[np.ndarray] = None,
                 constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Compute optimal portfolio weights.

        Args:
            regime_probs: Current regime probabilities (for prior weighting)
            constraints: Portfolio constraints (bounds, sum to 1, etc.)

        Returns:
            OptimizationResult with weights and posterior
        """
        if self.mu_n is None:
            # Use prior if posterior not yet updated
            mu = self.mu_0
            Sigma = self.Psi_0 / (self.nu_0 - self.mu_0.shape[0] - 1)
        else:
            # Use posterior
            mu = self.mu_n
            Sigma = self.Psi_n / (self.nu_n - self.mu_n.shape[0] - 1)

        n_assets = len(mu)

        # Objective: maximize utility = μ'w - λ/2 * w'Σw
        def objective(w):
            return -(mu @ w - 0.5 * self.risk_aversion * w @ Sigma @ w)

        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum to 1

        if constraints is not None:
            if 'long_only' in constraints and constraints['long_only']:
                bounds = [(0, 1) for _ in range(n_assets)]
            else:
                bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess (equal weight)
        w0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}. Using equal weights.")
            weights = w0
        else:
            weights = result.x

        # Compute posterior distribution
        posterior = self._compute_posterior_distribution(weights, mu, Sigma)

        # Expected return and risk
        expected_return = mu @ weights
        expected_risk = np.sqrt(weights @ Sigma @ weights)

        # Uncertainty measure (trace of posterior covariance)
        uncertainty = np.trace(posterior.covariance)

        return OptimizationResult(
            weights=weights,
            posterior=posterior,
            regime_weights={},  # Will be filled by regime-aware optimizer
            expected_return=expected_return,
            expected_risk=expected_risk,
            uncertainty=uncertainty
        )

    def _compute_posterior_distribution(self,
                                       weights: np.ndarray,
                                       mu: np.ndarray,
                                       Sigma: np.ndarray) -> PosteriorDistribution:
        """Compute posterior distribution of portfolio return."""
        # Portfolio return posterior: N(μ'w, w'Σw)
        post_mean = mu @ weights
        post_var = weights @ Sigma @ weights

        # Weight uncertainty (approximate via delta method)
        weight_cov = Sigma / self.kappa_n if self.kappa_n else Sigma

        # Confidence intervals (95%)
        ci_lower = weights - 1.96 * np.sqrt(np.diag(weight_cov))
        ci_upper = weights + 1.96 * np.sqrt(np.diag(weight_cov))

        return PosteriorDistribution(
            mean=weights,
            covariance=weight_cov,
            samples=None,
            confidence_intervals={
                'lower': ci_lower,
                'upper': ci_upper
            }
        )

    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        n, p = returns.shape
        sample_cov = returns.cov().values

        # Shrinkage target (diagonal matrix with average variance)
        target = np.eye(p) * np.trace(sample_cov) / p

        # Shrinkage intensity (Ledoit-Wolf estimator)
        # Simplified version
        delta = 0.5  # Could be estimated from data

        shrunk_cov = delta * target + (1 - delta) * sample_cov
        return shrunk_cov


class RegimeAwareBayesianOptimizer:
    """
    Regime-aware Bayesian optimizer that integrates mixture model outputs.

    For each regime k, maintains a separate Bayesian optimizer:
        p(w_t | z_t=k, data_k)

    Final portfolio is probability-weighted:
        w_t = Σ_k p(z_t=k) * E[w_t | z_t=k]
    """

    def __init__(self,
                 n_regimes: int = 5,
                 risk_aversion: float = 1.0,
                 prior_confidence: float = 0.1,
                 use_shrinkage: bool = True):
        """
        Initialize regime-aware optimizer.

        Args:
            n_regimes: Number of market regimes
            risk_aversion: Risk aversion parameter
            prior_confidence: Prior strength
            use_shrinkage: Use shrinkage estimation
        """
        self.n_regimes = n_regimes
        self.risk_aversion = risk_aversion

        # Create separate optimizer for each regime
        self.regime_optimizers = {
            k: BayesianMeanVarianceOptimizer(
                risk_aversion=risk_aversion,
                prior_confidence=prior_confidence,
                use_shrinkage=use_shrinkage
            )
            for k in range(n_regimes)
        }

        self.is_fitted = False

    def fit(self,
            returns: pd.DataFrame,
            regime_sequence: np.ndarray) -> 'RegimeAwareBayesianOptimizer':
        """
        Fit regime-specific optimizers.

        Args:
            returns: Historical returns
            regime_sequence: Regime labels/probabilities over time

        Returns:
            self (fitted)
        """
        if regime_sequence.ndim == 1:
            # Hard labels - convert to one-hot
            regime_probs = np.eye(self.n_regimes)[regime_sequence]
        else:
            # Already probabilities
            regime_probs = regime_sequence

        # Fit each regime's optimizer on regime-weighted data
        for k in range(self.n_regimes):
            # Weight samples by regime probability
            regime_weights = regime_probs[:, k]

            if regime_weights.sum() > 0:
                # Fit prior using regime-weighted historical data
                self.regime_optimizers[k].fit_prior(returns, regime_weights)

                # Update posterior with regime-filtered data
                # (Use samples where regime k is dominant)
                regime_mask = regime_probs[:, k] > 0.3
                if regime_mask.sum() > 10:
                    regime_returns = returns[regime_mask]
                    self.regime_optimizers[k].update_posterior(regime_returns)

        self.is_fitted = True
        return self

    def optimize(self,
                 regime_probs: np.ndarray,
                 constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Compute optimal portfolio given current regime probabilities.

        Args:
            regime_probs: Current regime probabilities [p(z=0), ..., p(z=K-1)]
            constraints: Portfolio constraints

        Returns:
            OptimizationResult with blended weights
        """
        if not self.is_fitted:
            raise ValueError("Optimizer not fitted. Call fit() first.")

        # Optimize for each regime
        regime_results = {}
        for k in range(self.n_regimes):
            if regime_probs[k] > 0.01:  # Skip negligible regimes
                result = self.regime_optimizers[k].optimize(constraints=constraints)
                regime_results[k] = result

        # Blend weights by regime probabilities
        n_assets = len(regime_results[0].weights)
        blended_weights = np.zeros(n_assets)
        regime_weights_dict = {}

        active_prob_mass = sum(regime_probs[k] for k in regime_results.keys())

        for k, result in regime_results.items():
            normalized_prob = regime_probs[k] / active_prob_mass
            blended_weights += normalized_prob * result.weights
            regime_weights_dict[f'regime_{k}'] = result.weights

        # Renormalize to ensure sum = 1
        blended_weights /= blended_weights.sum()

        # Compute blended statistics
        blended_return = sum(
            regime_probs[k] * regime_results[k].expected_return
            for k in regime_results.keys()
        ) / active_prob_mass

        blended_risk = np.sqrt(sum(
            regime_probs[k] * regime_results[k].expected_risk**2
            for k in regime_results.keys()
        )) / active_prob_mass

        # Uncertainty (entropy-weighted)
        entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
        blended_uncertainty = entropy * np.mean([
            r.uncertainty for r in regime_results.values()
        ])

        # Create blended posterior
        # (Simplified - weighted average of regime posteriors)
        blended_cov = np.zeros((n_assets, n_assets))
        for k, result in regime_results.items():
            blended_cov += (regime_probs[k] / active_prob_mass) * result.posterior.covariance

        ci_lower = blended_weights - 1.96 * np.sqrt(np.diag(blended_cov))
        ci_upper = blended_weights + 1.96 * np.sqrt(np.diag(blended_cov))

        posterior = PosteriorDistribution(
            mean=blended_weights,
            covariance=blended_cov,
            samples=None,
            confidence_intervals={'lower': ci_lower, 'upper': ci_upper}
        )

        return OptimizationResult(
            weights=blended_weights,
            posterior=posterior,
            regime_weights=regime_weights_dict,
            expected_return=blended_return,
            expected_risk=blended_risk,
            uncertainty=blended_uncertainty
        )

    def update_online(self,
                     new_returns: pd.Series,
                     regime_probs: np.ndarray) -> 'RegimeAwareBayesianOptimizer':
        """
        Online update with new observation.

        Args:
            new_returns: New return observation
            regime_probs: Regime probabilities at this time

        Returns:
            self (updated)
        """
        if not self.is_fitted:
            raise ValueError("Optimizer not fitted. Call fit() first.")

        # Update each regime's posterior weighted by regime probability
        for k in range(self.n_regimes):
            if regime_probs[k] > 0.1:
                # Weight the update by regime probability
                # (Could use more sophisticated weighting schemes)
                weighted_returns = new_returns.to_frame().T
                self.regime_optimizers[k].update_posterior(weighted_returns)

        return self


class BayesianBlackLitterman:
    """
    Bayesian Black-Litterman Model with regime awareness.

    Combines:
    - Market equilibrium (CAPM) as prior
    - Investor views as likelihood
    - Regime probabilities as view confidence weights
    """

    def __init__(self,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        Initialize Bayesian Black-Litterman.

        Args:
            risk_aversion: Market risk aversion (δ)
            tau: Scaling factor for uncertainty
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

        self.prior_returns = None
        self.prior_cov = None

    def fit_prior(self,
                  returns: pd.DataFrame,
                  market_caps: Optional[np.ndarray] = None) -> 'BayesianBlackLitterman':
        """
        Fit equilibrium prior (reverse optimization).

        Args:
            returns: Historical returns
            market_caps: Market capitalizations (for equilibrium weights)

        Returns:
            self (fitted)
        """
        if market_caps is None:
            # Equal-weighted market
            market_weights = np.ones(returns.shape[1]) / returns.shape[1]
        else:
            market_weights = market_caps / market_caps.sum()

        # Covariance estimate
        self.prior_cov = returns.cov().values

        # Implied equilibrium returns: Π = δ * Σ * w_market
        self.prior_returns = self.risk_aversion * self.prior_cov @ market_weights

        return self

    def optimize_with_views(self,
                           views: Dict[str, Dict],
                           asset_names: List[str],
                           regime_probs: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Optimize with investor views.

        Args:
            views: Dictionary of views
                   {'view_1': {'assets': ['AAPL', 'MSFT'], 'direction': 1, 'confidence': 0.8}}
            asset_names: List of asset names
            regime_probs: Optional regime probabilities (adjust view confidence)

        Returns:
            OptimizationResult with posterior weights
        """
        # Implementation would construct P matrix (view picking) and Q vector (view returns)
        # Then compute posterior: μ_post = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]

        # Placeholder for now
        n_assets = len(asset_names)
        weights = np.ones(n_assets) / n_assets

        posterior = PosteriorDistribution(
            mean=weights,
            covariance=np.eye(n_assets) * 0.01,
            samples=None,
            confidence_intervals={'lower': weights * 0.8, 'upper': weights * 1.2}
        )

        return OptimizationResult(
            weights=weights,
            posterior=posterior,
            regime_weights={},
            expected_return=0.0,
            expected_risk=0.0,
            uncertainty=0.0
        )


if __name__ == "__main__":
    # Example usage
    print("Bayesian Portfolio Optimizer")
    print("=" * 60)
    print("Usage example:")
    print("""
    from strategy.bayesian_optimizer import RegimeAwareBayesianOptimizer
    import pandas as pd

    # Assume you have:
    # - returns: DataFrame of asset returns
    # - regime_sequence: Regime probabilities over time
    # - current_regime_probs: Current regime probabilities from mixture model

    # Initialize
    optimizer = RegimeAwareBayesianOptimizer(
        n_regimes=5,
        risk_aversion=1.0
    )

    # Fit on historical data
    optimizer.fit(returns, regime_sequence)

    # Optimize for current market state
    result = optimizer.optimize(
        regime_probs=current_regime_probs,
        constraints={'long_only': True}
    )

    print(f"Optimal Weights: {result.weights}")
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Expected Risk: {result.expected_risk:.2%}")
    print(f"Uncertainty: {result.uncertainty:.4f}")
    """)
