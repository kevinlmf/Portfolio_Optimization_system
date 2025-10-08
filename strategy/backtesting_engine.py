"""
Backtesting Engine for Portfolio Optimization

Comprehensive backtesting framework that supports:
- Rolling window optimization
- Regime-based strategy switching
- Transaction costs and slippage
- Multiple performance metrics
- Monte Carlo simulation validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class BacktestingEngine:
    """
    Professional-grade backtesting engine for portfolio strategies.

    Features:
    - Rolling window optimization with regime detection
    - Transaction cost modeling
    - Comprehensive performance metrics
    - Risk analytics (VaR, CVaR, drawdown)
    - Benchmark comparison
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 prices: pd.DataFrame = None,
                 transaction_cost: float = 0.001,
                 rebalance_frequency: str = 'monthly',
                 risk_free_rate: float = 0.03):
        """
        Initialize backtesting engine.

        Args:
            returns: DataFrame of asset returns (assets as columns)
            prices: DataFrame of asset prices (optional, for visualization)
            transaction_cost: Transaction cost per trade (default: 0.1%)
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.prices = prices if prices is not None else (1 + returns).cumprod()
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        # Rebalance frequency mapping
        freq_map = {
            'daily': 1,
            'weekly': 5,
            'biweekly': 10,
            'monthly': 21,
            'quarterly': 63,
            'semiannual': 126,
            'annual': 252
        }
        self.rebalance_days = freq_map.get(rebalance_frequency, 21)

        # Results storage
        self.portfolio_values = None
        self.weights_history = []
        self.rebalance_dates = []
        self.transaction_costs_history = []
        self.performance_metrics = {}

    def run_backtest(self,
                    optimization_func: Callable,
                    lookback_window: int = 252,
                    min_history: int = 60,
                    **opt_kwargs) -> Dict:
        """
        Run rolling window backtest.

        Args:
            optimization_func: Function that takes returns and returns weights
                              Should have signature: func(returns_df, **kwargs) -> weights
            lookback_window: Historical window for optimization (trading days)
            min_history: Minimum history required to start optimization
            **opt_kwargs: Additional arguments to pass to optimization function

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*70}")
        print("BACKTESTING ENGINE")
        print(f"{'='*70}")
        print(f"Period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print(f"Assets: {len(self.returns.columns)}")
        print(f"Trading days: {len(self.returns)}")
        print(f"Rebalance frequency: {self.rebalance_days} days")
        print(f"Transaction cost: {self.transaction_cost:.2%}")

        n_periods = len(self.returns)
        n_assets = len(self.returns.columns)

        # Initialize portfolio
        portfolio_values = [1.0]  # Start with $1
        current_weights = np.ones(n_assets) / n_assets  # Start with equal weight
        turnover_list = []

        rebalance_count = 0

        # Rolling window backtest
        for t in range(min_history, n_periods):
            current_date = self.returns.index[t]

            # Check if it's a rebalance date
            is_rebalance = (t - min_history) % self.rebalance_days == 0

            if is_rebalance:
                # Get historical data for optimization
                start_idx = max(0, t - lookback_window)
                hist_returns = self.returns.iloc[start_idx:t]

                try:
                    # Run optimization
                    new_weights = optimization_func(hist_returns, **opt_kwargs)

                    # Ensure weights are valid
                    if not isinstance(new_weights, np.ndarray):
                        new_weights = np.array(new_weights)

                    if len(new_weights) != n_assets:
                        raise ValueError(f"Weight dimension mismatch: got {len(new_weights)}, expected {n_assets}")

                    # Normalize weights
                    new_weights = np.maximum(new_weights, 0)  # No short selling
                    if new_weights.sum() > 0:
                        new_weights = new_weights / new_weights.sum()
                    else:
                        new_weights = np.ones(n_assets) / n_assets

                    # Calculate turnover
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    turnover_list.append(turnover)

                    # Apply transaction costs
                    tc_drag = turnover * self.transaction_cost
                    portfolio_values[-1] *= (1 - tc_drag)

                    # Update weights
                    current_weights = new_weights
                    rebalance_count += 1

                    # Store rebalance info
                    self.weights_history.append({
                        'date': current_date,
                        'weights': current_weights.copy(),
                        'turnover': turnover,
                        'transaction_cost': tc_drag
                    })
                    self.rebalance_dates.append(current_date)

                except Exception as e:
                    print(f"Warning: Optimization failed at {current_date}: {e}")
                    # Keep previous weights

            # Calculate daily return
            daily_returns = self.returns.iloc[t].values
            portfolio_return = np.dot(current_weights, daily_returns)

            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)

        # Store results
        self.portfolio_values = pd.Series(
            portfolio_values[1:],  # Skip initial value
            index=self.returns.index[min_history:]
        )

        print(f"\n✓ Backtest complete")
        print(f"  Rebalances: {rebalance_count}")
        print(f"  Final portfolio value: ${self.portfolio_values.iloc[-1]:.4f}")
        print(f"  Total return: {(self.portfolio_values.iloc[-1] - 1) * 100:.2f}%")

        # Calculate performance metrics
        self.performance_metrics = self.calculate_metrics()

        return {
            'portfolio_values': self.portfolio_values,
            'weights_history': self.weights_history,
            'metrics': self.performance_metrics,
            'rebalance_dates': self.rebalance_dates
        }

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if self.portfolio_values is None:
            raise ValueError("Must run backtest first")

        portfolio_returns = self.portfolio_values.pct_change().dropna()

        # Basic metrics
        total_return = self.portfolio_values.iloc[-1] - 1
        n_days = len(portfolio_returns)
        n_years = n_days / 252

        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Max drawdown
        cumulative = self.portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)

        # Average turnover
        avg_turnover = np.mean([w['turnover'] for w in self.weights_history]) if self.weights_history else 0

        # Skewness and Kurtosis
        from scipy import stats
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)

        # Cumulative transaction costs
        total_tc = sum([w['transaction_cost'] for w in self.weights_history])

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'avg_turnover': avg_turnover,
            'total_transaction_costs': total_tc,
            'n_rebalances': len(self.weights_history),
            'n_years': n_years
        }

        return metrics

    def compare_with_benchmark(self, benchmark_returns: pd.Series) -> Dict:
        """
        Compare portfolio performance with benchmark.

        Args:
            benchmark_returns: Benchmark return series

        Returns:
            Dictionary with comparison metrics
        """
        if self.portfolio_values is None:
            raise ValueError("Must run backtest first")

        portfolio_returns = self.portfolio_values.pct_change().dropna()

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]

        # Alpha and Beta
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        portfolio_mean = port_ret.mean() * 252
        benchmark_mean = bench_ret.mean() * 252
        alpha = portfolio_mean - beta * benchmark_mean

        # Information ratio
        active_returns = port_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        # Benchmark metrics
        benchmark_cumulative = (1 + bench_ret).cumprod()
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(bench_ret)) - 1
        benchmark_volatility = bench_ret.std() * np.sqrt(252)

        # Max drawdown for benchmark
        bench_running_max = benchmark_cumulative.expanding().max()
        bench_drawdown = (benchmark_cumulative - bench_running_max) / bench_running_max
        benchmark_max_drawdown = bench_drawdown.min()

        comparison = {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'benchmark_return': benchmark_annual_return,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'excess_return': portfolio_mean - benchmark_mean,
            'win_rate_vs_benchmark': (active_returns > 0).sum() / len(active_returns)
        }

        return comparison

    def print_metrics_report(self):
        """Print formatted performance metrics report."""
        if not self.performance_metrics:
            print("No metrics available. Run backtest first.")
            return

        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS")
        print(f"{'='*70}")

        m = self.performance_metrics

        print(f"\nReturn Metrics:")
        print(f"  Total Return:          {m['total_return']:>8.2%}")
        print(f"  Annual Return:         {m['annual_return']:>8.2%}")
        print(f"  Annual Volatility:     {m['annual_volatility']:>8.2%}")

        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio:          {m['sharpe_ratio']:>8.4f}")
        print(f"  Sortino Ratio:         {m['sortino_ratio']:>8.4f}")
        print(f"  Calmar Ratio:          {m['calmar_ratio']:>8.4f}")

        print(f"\nRisk Metrics:")
        print(f"  Maximum Drawdown:      {m['max_drawdown']:>8.2%}")
        print(f"  VaR (95%):             {m['var_95']:>8.2%}")
        print(f"  CVaR (95%):            {m['cvar_95']:>8.2%}")

        print(f"\nDistribution:")
        print(f"  Skewness:              {m['skewness']:>8.4f}")
        print(f"  Kurtosis:              {m['kurtosis']:>8.4f}")
        print(f"  Win Rate:              {m['win_rate']:>8.2%}")

        print(f"\nTrading Statistics:")
        print(f"  Number of Rebalances:  {m['n_rebalances']:>8d}")
        print(f"  Average Turnover:      {m['avg_turnover']:>8.2%}")
        print(f"  Transaction Costs:     {m['total_transaction_costs']:>8.4f}")

        print(f"\n{'='*70}")

    def export_results(self, output_dir: str = 'results'):
        """Export backtest results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export portfolio values
        self.portfolio_values.to_csv(f'{output_dir}/portfolio_values.csv')

        # Export weights history
        if self.weights_history:
            weights_df = pd.DataFrame([
                {
                    'date': w['date'],
                    **{f'weight_{i}': w['weights'][i] for i in range(len(w['weights']))},
                    'turnover': w['turnover'],
                    'transaction_cost': w['transaction_cost']
                }
                for w in self.weights_history
            ])
            weights_df.to_csv(f'{output_dir}/weights_history.csv', index=False)

        # Export metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)

        print(f"\n✓ Results exported to {output_dir}/")


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio validation.

    Generates synthetic return paths to assess:
    - Distribution of returns
    - Probability of achieving targets
    - Risk of large losses
    - Robustness of optimization
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 n_simulations: int = 10000,
                 n_days: int = 252):
        """
        Initialize Monte Carlo simulator.

        Args:
            returns: Historical returns DataFrame
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate
        """
        self.returns = returns
        self.n_simulations = n_simulations
        self.n_days = n_days

        # Estimate parameters
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values

        self.simulated_paths = None
        self.simulation_results = None

    def run_simulation(self,
                      weights: np.ndarray,
                      method: str = 'parametric') -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            weights: Portfolio weights
            method: 'parametric' (normal), 'bootstrap', or 'garch'

        Returns:
            Dictionary with simulation results
        """
        print(f"\n{'='*70}")
        print("MONTE CARLO SIMULATION")
        print(f"{'='*70}")
        print(f"Simulations: {self.n_simulations:,}")
        print(f"Horizon: {self.n_days} days")
        print(f"Method: {method}")

        if method == 'parametric':
            paths = self._simulate_parametric(weights)
        elif method == 'bootstrap':
            paths = self._simulate_bootstrap(weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.simulated_paths = paths

        # Calculate statistics
        final_values = paths[:, -1]

        results = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'var_95': np.percentile(final_values, 5),
            'cvar_95': final_values[final_values <= np.percentile(final_values, 5)].mean(),
            'best_case': np.percentile(final_values, 95),
            'worst_case': np.percentile(final_values, 5),
            'prob_positive': (final_values > 1.0).sum() / len(final_values),
            'prob_loss_10pct': (final_values < 0.9).sum() / len(final_values),
            'prob_gain_20pct': (final_values > 1.2).sum() / len(final_values),
        }

        self.simulation_results = results

        print(f"\n✓ Simulation complete")
        print(f"\nResults:")
        print(f"  Mean final value:      ${results['mean_final_value']:.4f}")
        print(f"  Median final value:    ${results['median_final_value']:.4f}")
        print(f"  95% VaR:              ${results['var_95']:.4f}")
        print(f"  Prob of profit:        {results['prob_positive']:.2%}")
        print(f"  Prob of >20% gain:     {results['prob_gain_20pct']:.2%}")
        print(f"  Prob of >10% loss:     {results['prob_loss_10pct']:.2%}")

        return results

    def _simulate_parametric(self, weights: np.ndarray) -> np.ndarray:
        """Simulate using parametric (multivariate normal) method."""
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = 1.0  # Start at $1

        for sim in range(self.n_simulations):
            # Generate correlated returns
            random_returns = np.random.multivariate_normal(
                self.mean_returns,
                self.cov_matrix,
                self.n_days
            )

            # Calculate portfolio returns
            portfolio_returns = random_returns @ weights

            # Compute cumulative value
            for t in range(self.n_days):
                paths[sim, t + 1] = paths[sim, t] * (1 + portfolio_returns[t])

        return paths

    def _simulate_bootstrap(self, weights: np.ndarray) -> np.ndarray:
        """Simulate using bootstrap (historical resampling) method."""
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = 1.0

        returns_array = self.returns.values

        for sim in range(self.n_simulations):
            # Randomly sample historical returns (with replacement)
            indices = np.random.randint(0, len(returns_array), self.n_days)
            sampled_returns = returns_array[indices]

            # Calculate portfolio returns
            portfolio_returns = sampled_returns @ weights

            # Compute cumulative value
            for t in range(self.n_days):
                paths[sim, t + 1] = paths[sim, t] * (1 + portfolio_returns[t])

        return paths

    def get_confidence_intervals(self, confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for returns.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with confidence intervals
        """
        if self.simulated_paths is None:
            raise ValueError("Must run simulation first")

        alpha = 1 - confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        intervals = {}
        for day in [1, 5, 21, 63, 252]:  # 1d, 1w, 1m, 3m, 1y
            if day < self.simulated_paths.shape[1]:
                values = self.simulated_paths[:, day]
                intervals[f'{day}d'] = {
                    'lower': np.percentile(values, lower_pct),
                    'median': np.percentile(values, 50),
                    'upper': np.percentile(values, upper_pct)
                }

        return intervals
