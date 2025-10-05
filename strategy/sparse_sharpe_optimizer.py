"""
m-Sparse Sharpe Ratio Maximization via Proximal Gradient Algorithm (mSSRM-PGA)

Implementation of the algorithm from:
"A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization"
Lin et al., NeurIPS 2024

This module implements the mSSRM-PGA algorithm which achieves globally optimal
m-sparse Sharpe ratio under certain conditions with convergence guarantees.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SparseSharpeOptimizer:
    """
    m-Sparse Sharpe Ratio Maximization using Proximal Gradient Algorithm.

    This optimizer directly maximizes the Sharpe ratio with exact sparsity control
    (selecting exactly m assets) and provides theoretical guarantees for global optimality.

    Key Features:
    - Exact sparsity control via ℓ0 constraint
    - Direct optimization of original Sharpe ratio (not a proxy)
    - Global optimality guarantee under certain conditions
    - Convergence rates: O(1/√k) for iterates, O(1/k) for function values

    Reference:
        Lin, Y., Lai, Z.-R., & Li, C. (2024). A Globally Optimal Portfolio for
        m-Sparse Sharpe Ratio Maximization. NeurIPS 2024.
    """

    def __init__(self,
                 epsilon: float = 1e-3,
                 alpha_step: Optional[float] = None,
                 max_iter: int = 10000,
                 tol: float = 1e-5,
                 verbose: bool = True):
        """
        Initialize the Sparse Sharpe Optimizer.

        Args:
            epsilon: Regularization parameter for positive definite Q_epsilon
            alpha_step: Step size for proximal gradient (auto-computed if None)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print progress information
        """
        self.epsilon = epsilon
        self.alpha_step = alpha_step
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Storage for optimization results
        self.v_star = None
        self.w_star = None
        self.sharpe_ratio = None
        self.convergence_history = []

    def _compute_matrices(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute p, Q, and Q_epsilon matrices from return data.

        Args:
            returns: T×N matrix of asset returns (excess of risk-free rate)

        Returns:
            p: Mean return vector (N×1)
            Q: Covariance decomposition matrix
            Q_epsilon: Regularized positive definite matrix
        """
        T, N = returns.shape

        # p = (1/T) * R^T * 1_T
        p = np.mean(returns, axis=0)

        # Q = (1/√(T-1)) * (R - (1/T) * 1_T×T * R)
        mean_returns = np.mean(returns, axis=0, keepdims=True)
        centered_returns = returns - mean_returns
        Q = centered_returns / np.sqrt(T - 1)

        # Q_epsilon = Q^T * Q + epsilon * I
        Q_epsilon = Q.T @ Q + self.epsilon * np.eye(N)

        return p, Q, Q_epsilon

    def _prox_omega(self, v: np.ndarray, m: int) -> np.ndarray:
        """
        Proximity operator for the constraint set Omega.

        Omega = {v ∈ R^N | v ≥ 0 and ||v||_0 ≤ m}

        Args:
            v: Input vector
            m: Sparsity level (maximum number of non-zero components)

        Returns:
            Projection of v onto Omega
        """
        N = len(v)
        h = np.zeros(N)

        # Count positive components
        positive_mask = v > 0
        m_v = np.sum(positive_mask)

        if m_v > m:
            # Select m largest positive components
            indices = np.argsort(v)[::-1][:m]
            h[indices] = v[indices]
        else:
            # Keep all positive components
            h[positive_mask] = v[positive_mask]

        return h

    def _objective(self, v: np.ndarray, Q_epsilon: np.ndarray, p: np.ndarray) -> float:
        """
        Compute objective function value: f(v) = (1/2) * v^T * Q_epsilon * v - p^T * v

        Args:
            v: Portfolio vector
            Q_epsilon: Regularized covariance matrix
            p: Mean return vector

        Returns:
            Objective function value
        """
        return 0.5 * v.T @ Q_epsilon @ v - p.T @ v

    def _sharpe_ratio(self, w: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute Sharpe ratio for portfolio weights.

        Args:
            w: Portfolio weights
            returns: Return matrix

        Returns:
            Sharpe ratio
        """
        portfolio_returns = returns @ w
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns, ddof=1)

        if std_return < 1e-10:
            return 0.0

        return mean_return / std_return

    def optimize(self,
                 returns: np.ndarray,
                 m: int,
                 initial_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Optimize portfolio using mSSRM-PGA algorithm.

        Args:
            returns: T×N matrix of asset returns (excess of risk-free rate)
            m: Sparsity level (select at most m assets)
            initial_weights: Initial portfolio weights (default: equal to p)

        Returns:
            Dictionary containing:
                - weights: Optimal portfolio weights
                - sparsity: Actual number of selected assets
                - sharpe_ratio: Achieved Sharpe ratio
                - converged: Whether algorithm converged
                - iterations: Number of iterations
                - objective_values: History of objective function values
        """
        T, N = returns.shape

        if m > N:
            raise ValueError(f"Sparsity level m={m} cannot exceed number of assets N={N}")

        # Compute matrices
        p, Q, Q_epsilon = self._compute_matrices(returns)

        # Compute step size if not provided
        if self.alpha_step is None:
            eigenvalues = np.linalg.eigvalsh(Q_epsilon)
            lambda_max = np.max(eigenvalues)
            alpha = 0.999 / lambda_max
        else:
            alpha = self.alpha_step

        # Initialize v
        if initial_weights is not None:
            v_k = initial_weights.copy()
        else:
            v_k = p.copy()

        # Storage for convergence history
        self.convergence_history = []
        converged = False

        if self.verbose:
            print(f"\nStarting mSSRM-PGA optimization:")
            print(f"  Assets: {N}, Sparsity: {m}, Max iterations: {self.max_iter}")
            print(f"  Step size α: {alpha:.6f}, Tolerance: {self.tol}")

        # Main optimization loop
        for k in range(self.max_iter):
            # Gradient step: v - α * ∇f(v) = v - α * (Q_epsilon * v - p)
            gradient = Q_epsilon @ v_k - p
            v_temp = v_k - alpha * gradient

            # Proximal step
            v_k_plus_1 = self._prox_omega(v_temp, m)

            # Compute objective value
            obj_value = self._objective(v_k_plus_1, Q_epsilon, p)
            self.convergence_history.append(obj_value)

            # Check convergence
            diff = np.linalg.norm(v_k_plus_1 - v_k)
            relative_diff = diff / (np.linalg.norm(v_k) + 1e-10)

            if relative_diff < self.tol:
                converged = True
                if self.verbose:
                    print(f"\n✓ Converged at iteration {k+1}")
                    print(f"  Relative change: {relative_diff:.2e}")
                break

            # Update
            v_k = v_k_plus_1

            # Print progress
            if self.verbose and (k + 1) % 1000 == 0:
                print(f"  Iteration {k+1}: obj={obj_value:.6f}, diff={relative_diff:.2e}")

        # Store optimal v
        self.v_star = v_k

        # Convert to portfolio weights
        if np.sum(v_k) > 1e-10:
            self.w_star = v_k / np.sum(v_k)
        else:
            # If all components are zero, return equal weights as fallback
            self.w_star = np.zeros(N)
            if self.verbose:
                print("\n⚠ Warning: Optimal solution is zero vector. Using zero portfolio.")

        # Compute Sharpe ratio
        self.sharpe_ratio = self._sharpe_ratio(self.w_star, returns)

        # Get actual sparsity
        actual_sparsity = np.sum(self.w_star > 1e-8)

        if self.verbose:
            print(f"\nOptimization Results:")
            print(f"  Converged: {converged}")
            print(f"  Iterations: {k+1}/{self.max_iter}")
            print(f"  Actual sparsity: {actual_sparsity}/{m}")
            print(f"  Sharpe ratio: {self.sharpe_ratio:.4f}")
            print(f"  Non-zero weights: {np.sum(self.w_star > 0.01)}")

        return {
            'weights': self.w_star,
            'sparsity': actual_sparsity,
            'sharpe_ratio': self.sharpe_ratio,
            'converged': converged,
            'iterations': k + 1,
            'objective_values': self.convergence_history,
            'v_star': self.v_star
        }

    def get_portfolio_weights(self) -> np.ndarray:
        """
        Get optimal portfolio weights.

        Returns:
            Optimal weights array
        """
        if self.w_star is None:
            raise ValueError("Must call optimize() first")
        return self.w_star

    def get_selected_assets(self, asset_names: list, threshold: float = 1e-8) -> list:
        """
        Get list of selected assets (with non-zero weights).

        Args:
            asset_names: List of asset names/tickers
            threshold: Minimum weight to consider as selected

        Returns:
            List of selected asset names
        """
        if self.w_star is None:
            raise ValueError("Must call optimize() first")

        selected_indices = np.where(self.w_star > threshold)[0]
        return [asset_names[i] for i in selected_indices]

    def get_portfolio_statistics(self, returns: np.ndarray) -> Dict:
        """
        Compute comprehensive portfolio statistics.

        Args:
            returns: Asset return matrix

        Returns:
            Dictionary of portfolio statistics
        """
        if self.w_star is None:
            raise ValueError("Must call optimize() first")

        portfolio_returns = returns @ self.w_star

        return {
            'sharpe_ratio': self.sharpe_ratio,
            'mean_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns, ddof=1),
            'max_weight': np.max(self.w_star),
            'min_weight': np.min(self.w_star[self.w_star > 1e-8]) if np.sum(self.w_star > 1e-8) > 0 else 0,
            'sparsity': np.sum(self.w_star > 1e-8),
            'total_weight': np.sum(self.w_star)
        }


class MultiSparsityOptimizer:
    """
    Run mSSRM-PGA optimization across multiple sparsity levels.

    This class facilitates comparison of portfolios with different sparsity constraints
    to find the optimal trade-off between diversification and concentration.
    """

    def __init__(self, epsilon: float = 1e-3, verbose: bool = True):
        """
        Initialize multi-sparsity optimizer.

        Args:
            epsilon: Regularization parameter
            verbose: Whether to print progress
        """
        self.epsilon = epsilon
        self.verbose = verbose
        self.results = {}

    def optimize_grid(self,
                      returns: np.ndarray,
                      sparsity_levels: list,
                      asset_names: Optional[list] = None) -> pd.DataFrame:
        """
        Optimize portfolios across a grid of sparsity levels.

        Args:
            returns: Asset return matrix
            sparsity_levels: List of m values to try
            asset_names: Optional list of asset names

        Returns:
            DataFrame with results for each sparsity level
        """
        N = returns.shape[1]

        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(N)]

        results_list = []

        for m in sparsity_levels:
            if m > N:
                continue

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Optimizing with sparsity m = {m}")
                print(f"{'='*60}")

            optimizer = SparseSharpeOptimizer(
                epsilon=self.epsilon,
                verbose=self.verbose
            )

            result = optimizer.optimize(returns, m)
            stats = optimizer.get_portfolio_statistics(returns)
            selected_assets = optimizer.get_selected_assets(asset_names)

            results_list.append({
                'm_target': m,
                'm_actual': stats['sparsity'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'mean_return': stats['mean_return'],
                'volatility': stats['volatility'],
                'max_weight': stats['max_weight'],
                'converged': result['converged'],
                'iterations': result['iterations'],
                'selected_assets': ', '.join(selected_assets)
            })

            self.results[m] = {
                'optimizer': optimizer,
                'result': result,
                'stats': stats
            }

        return pd.DataFrame(results_list)

    def get_best_portfolio(self, metric: str = 'sharpe_ratio') -> Tuple[int, Dict]:
        """
        Get the best portfolio based on specified metric.

        Args:
            metric: Metric to optimize ('sharpe_ratio', 'mean_return', etc.)

        Returns:
            Tuple of (best_m, best_result)
        """
        if not self.results:
            raise ValueError("Must call optimize_grid() first")

        best_m = None
        best_value = -np.inf

        for m, result_dict in self.results.items():
            value = result_dict['stats'].get(metric, -np.inf)
            if value > best_value:
                best_value = value
                best_m = m

        return best_m, self.results[best_m]
