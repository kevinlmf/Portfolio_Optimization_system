"""
Integrated Bayesian Portfolio System

Unified system that integrates:
1. Mixture-based Regime Detection (soft regime identification)
2. Bayesian Portfolio Optimization (regime-aware optimization)
3. Online Posterior Updating (adaptive learning)

Complete Pipeline:
    Market Data → Feature Extraction → Mixture Model → Regime Probs
                                                            ↓
    Historical Returns → Regime-Aware Bayesian Optimizer ← Regime Probs
                                                            ↓
    New Observations → Posterior Updater → Updated Weights

This system provides:
- Continuous regime probability estimates
- Regime-weighted portfolio optimization
- Online adaptation to market changes
- Uncertainty quantification
- Robust handling of anomalies and regime shifts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from m2_market.mixture_regime_model import MixtureRegimeDetector, RegimeResult
    from m4_optimization.bayesian_optimizer import RegimeAwareBayesianOptimizer, OptimizationResult
    from m5_evaluation.bayesian_updater import BayesianPosteriorUpdater, RegimeAwareUpdater, UpdateResult
except ImportError:
    # Relative imports as fallback
    from .mixture_regime_model import MixtureRegimeDetector, RegimeResult
    from .bayesian_optimizer import RegimeAwareBayesianOptimizer, OptimizationResult
    from .bayesian_updater import BayesianPosteriorUpdater, RegimeAwareUpdater, UpdateResult


@dataclass
class SystemState:
    """Complete system state at time t."""
    timestamp: pd.Timestamp
    regime_probs: np.ndarray
    dominant_regime: str
    portfolio_weights: np.ndarray
    expected_return: float
    expected_risk: float
    uncertainty: float
    confidence: float


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    regime_history: pd.DataFrame
    performance_metrics: Dict
    system_states: List[SystemState]


class IntegratedBayesianSystem:
    """
    Integrated Mixture-Bayesian Portfolio Optimization System.

    This is the main class that users interact with.
    """

    def __init__(self,
                 n_regimes: int = 5,
                 mixture_type: str = 'bayesian_gaussian',
                 auto_select_k: bool = True,
                 risk_aversion: float = 1.0,
                 decay_factor: float = 0.95,
                 prior_confidence: float = 0.1,
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 rebalance_frequency: str = 'monthly',
                 transaction_cost: float = 0.001,
                 robust_mode: bool = True):
        """
        Initialize integrated system.

        Args:
            n_regimes: Number of market regimes
            mixture_type: Mixture model type ('gaussian', 'bayesian_gaussian')
            auto_select_k: Automatically determine optimal K
            risk_aversion: Portfolio risk aversion parameter
            decay_factor: Exponential forgetting factor
            prior_confidence: Strength of prior beliefs
            lookback_short: Short-term feature window
            lookback_long: Long-term feature window
            rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
            transaction_cost: Transaction cost (proportion)
            robust_mode: Enable robust updating
        """
        self.n_regimes = n_regimes
        self.risk_aversion = risk_aversion
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost

        # Initialize components
        print("Initializing Mixture Regime Detector...")
        self.regime_detector = MixtureRegimeDetector(
            n_regimes=n_regimes,
            mixture_type=mixture_type,
            auto_select_k=auto_select_k,
            lookback_short=lookback_short,
            lookback_long=lookback_long
        )

        print("Initializing Bayesian Optimizer...")
        self.optimizer = RegimeAwareBayesianOptimizer(
            n_regimes=n_regimes,
            risk_aversion=risk_aversion,
            prior_confidence=prior_confidence
        )

        print("Initializing Posterior Updater...")
        self.updater = RegimeAwareUpdater(
            n_regimes=n_regimes,
            decay_factor=decay_factor
        )

        # State tracking
        self.is_fitted = False
        self.current_weights = None
        self.system_states = []
        self.asset_names = None

    def fit(self,
            prices: pd.DataFrame,
            volume: Optional[pd.DataFrame] = None,
            fit_window: int = 252) -> 'IntegratedBayesianSystem':
        """
        Fit the system on historical data.

        Args:
            prices: Historical price data (DataFrame with date index)
            volume: Optional volume data
            fit_window: Lookback window for fitting (default: 1 year)

        Returns:
            self (fitted)
        """
        print("\n" + "="*60)
        print("FITTING INTEGRATED BAYESIAN SYSTEM")
        print("="*60)

        self.asset_names = prices.columns.tolist()
        n_assets = len(self.asset_names)

        # Use last fit_window for fitting
        fit_prices = prices.iloc[-fit_window:]
        fit_returns = fit_prices.pct_change().dropna()

        # Step 1: Fit regime detector
        print("\n[1/3] Fitting Mixture Regime Detector...")
        if volume is not None:
            fit_volume = volume.iloc[-fit_window:]
            # Use first asset's volume as proxy
            self.regime_detector.fit(fit_prices, fit_volume.iloc[:, 0])
        else:
            self.regime_detector.fit(fit_prices)

        # Get regime sequence for training
        regime_sequence = self.regime_detector.predict_sequence(fit_prices)
        regime_probs = regime_sequence[[col for col in regime_sequence.columns
                                       if col not in ['dominant_regime', 'confidence', 'entropy']]].values

        print(f"   Optimal K detected: {self.regime_detector.optimal_k}")
        print(f"   Regime distribution:")
        for i, name in enumerate(self.regime_detector.REGIME_NAMES.values()):
            avg_prob = regime_probs[:, i].mean()
            print(f"      {name}: {avg_prob:.2%}")

        # Step 2: Fit Bayesian optimizer
        print("\n[2/3] Fitting Regime-Aware Bayesian Optimizer...")
        self.optimizer.fit(fit_returns, regime_probs)
        print("   Fitted regime-specific optimizers")

        # Step 3: Initialize posterior updaters
        print("\n[3/3] Initializing Posterior Updaters...")
        mu_priors = {}
        Sigma_priors = {}

        for k in range(self.regime_detector.optimal_k):
            # Get regime-specific historical returns
            regime_mask = regime_probs[:, k] > 0.3
            if regime_mask.sum() > 10:
                regime_returns = fit_returns[regime_mask]
                mu_priors[k] = regime_returns.mean().values
                Sigma_priors[k] = regime_returns.cov().values
            else:
                # Fallback to overall statistics
                mu_priors[k] = fit_returns.mean().values
                Sigma_priors[k] = fit_returns.cov().values

        self.updater.initialize_all(mu_priors, Sigma_priors)
        print("   Initialized regime-specific updaters")

        self.is_fitted = True

        print("\n" + "="*60)
        print("SYSTEM FITTING COMPLETE")
        print("="*60 + "\n")

        return self

    def predict(self,
                prices: pd.DataFrame,
                volume: Optional[pd.DataFrame] = None,
                constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Generate portfolio weights for current market state.

        Args:
            prices: Current price data
            volume: Optional volume data
            constraints: Portfolio constraints

        Returns:
            OptimizationResult with recommended weights
        """
        if not self.is_fitted:
            raise ValueError("System not fitted. Call fit() first.")

        # Step 1: Detect current regime
        if volume is not None:
            regime_result = self.regime_detector.predict_proba(prices, volume.iloc[:, 0])
        else:
            regime_result = self.regime_detector.predict_proba(prices)

        # Step 2: Optimize portfolio given regime probabilities
        optimization_result = self.optimizer.optimize(
            regime_probs=regime_result.regime_probs,
            constraints=constraints
        )

        # Store current weights
        self.current_weights = optimization_result.weights

        return optimization_result

    def update(self,
               new_returns: pd.Series,
               regime_probs: np.ndarray) -> Dict[int, UpdateResult]:
        """
        Online update with new observations.

        Args:
            new_returns: New return observations
            regime_probs: Current regime probabilities

        Returns:
            Update results per regime
        """
        if not self.is_fitted:
            raise ValueError("System not fitted. Call fit() first.")

        # Update posterior with new observation
        update_results = self.updater.update_all(new_returns.values, regime_probs)

        # Update optimizer posteriors
        posteriors = self.updater.get_posteriors()
        for k, (mu, Sigma) in posteriors.items():
            if k in self.optimizer.regime_optimizers:
                self.optimizer.regime_optimizers[k].mu_n = mu
                self.optimizer.regime_optimizers[k].Sigma_n = Sigma

        return update_results

    def backtest(self,
                 prices: pd.DataFrame,
                 volume: Optional[pd.DataFrame] = None,
                 initial_capital: float = 100000.0,
                 train_window: int = 252,
                 test_start: Optional[pd.Timestamp] = None,
                 constraints: Optional[Dict] = None) -> BacktestResult:
        """
        Run comprehensive backtest with rolling optimization.

        Args:
            prices: Historical price data
            volume: Optional volume data
            initial_capital: Initial portfolio value
            train_window: Training window size
            test_start: Start date for testing (if None, uses train_window)
            constraints: Portfolio constraints

        Returns:
            BacktestResult with complete backtest statistics
        """
        print("\n" + "="*60)
        print("RUNNING BAYESIAN BACKTEST")
        print("="*60)

        returns = prices.pct_change().dropna()

        if test_start is None:
            test_start_idx = train_window
        else:
            test_start_idx = prices.index.get_loc(test_start)

        # Initial training
        print(f"\nInitial training on {train_window} days...")
        train_prices = prices.iloc[:test_start_idx]
        train_volume = volume.iloc[:test_start_idx] if volume is not None else None
        self.fit(train_prices, train_volume, fit_window=train_window)

        # Determine rebalance dates
        test_dates = prices.index[test_start_idx:]
        rebalance_dates = self._get_rebalance_dates(test_dates, self.rebalance_frequency)

        print(f"Testing period: {test_dates[0]} to {test_dates[-1]}")
        print(f"Number of rebalances: {len(rebalance_dates)}")

        # Tracking variables
        portfolio_value = initial_capital
        portfolio_values = [initial_capital]
        weights_history = []
        regime_history_list = []
        system_states = []

        current_weights = np.ones(len(self.asset_names)) / len(self.asset_names)

        # Main backtest loop
        for i, date in enumerate(test_dates):
            # Get returns for this period
            if i > 0:
                period_returns = returns.loc[date]
                portfolio_return = current_weights @ period_returns.values

                # Apply transaction costs if rebalancing
                if date in rebalance_dates and i > 0:
                    weight_change = np.abs(current_weights - self.current_weights).sum()
                    transaction_costs = weight_change * self.transaction_cost
                    portfolio_return -= transaction_costs

                portfolio_value *= (1 + portfolio_return)
                portfolio_values.append(portfolio_value)

            # Rebalance if needed
            if date in rebalance_dates:
                # Get historical data up to this point
                hist_prices = prices.loc[:date]
                hist_volume = volume.loc[:date] if volume is not None else None

                # Detect regime
                if hist_volume is not None:
                    regime_result = self.regime_detector.predict_proba(hist_prices, hist_volume.iloc[:, 0])
                else:
                    regime_result = self.regime_detector.predict_proba(hist_prices)

                # Optimize
                opt_result = self.optimizer.optimize(
                    regime_probs=regime_result.regime_probs,
                    constraints=constraints
                )

                # Update
                if i > 0:
                    self.update(returns.loc[date], regime_result.regime_probs)

                # Store results
                current_weights = opt_result.weights
                self.current_weights = current_weights

                weights_history.append({
                    'date': date,
                    **{asset: w for asset, w in zip(self.asset_names, current_weights)}
                })

                regime_history_list.append({
                    'date': date,
                    'dominant_regime': regime_result.regime_names[regime_result.dominant_regime],
                    'confidence': regime_result.confidence,
                    'entropy': regime_result.entropy,
                    **{f'prob_{name}': prob for name, prob in zip(regime_result.regime_names, regime_result.regime_probs)}
                })

                system_states.append(SystemState(
                    timestamp=date,
                    regime_probs=regime_result.regime_probs,
                    dominant_regime=regime_result.regime_names[regime_result.dominant_regime],
                    portfolio_weights=current_weights.copy(),
                    expected_return=opt_result.expected_return,
                    expected_risk=opt_result.expected_risk,
                    uncertainty=opt_result.uncertainty,
                    confidence=regime_result.confidence
                ))

                if (i % 20 == 0):
                    print(f"   [{date.date()}] Value: ${portfolio_value:,.0f} | "
                          f"Regime: {regime_result.regime_names[regime_result.dominant_regime]} "
                          f"({regime_result.confidence:.1%})")

        # Create results
        portfolio_series = pd.Series(portfolio_values, index=test_dates[:len(portfolio_values)])
        weights_df = pd.DataFrame(weights_history).set_index('date')
        regime_df = pd.DataFrame(regime_history_list).set_index('date')

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_series, returns.loc[test_dates])

        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        print(f"\nPerformance Summary:")
        print(f"  Total Return:     {metrics['total_return']:.2%}")
        print(f"  Annual Return:    {metrics['annual_return']:.2%}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:         {metrics['win_rate']:.2%}")
        print()

        return BacktestResult(
            portfolio_values=portfolio_series,
            weights_history=weights_df,
            regime_history=regime_df,
            performance_metrics=metrics,
            system_states=system_states
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return dates.tolist()
        elif frequency == 'weekly':
            return dates[dates.to_series().dt.dayofweek == 0].tolist()
        elif frequency == 'monthly':
            return dates[dates.to_series().dt.is_month_end].tolist()
        elif frequency == 'quarterly':
            return dates[dates.to_series().dt.is_quarter_end].tolist()
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    def _calculate_performance_metrics(self,
                                      portfolio_values: pd.Series,
                                      returns: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        portfolio_returns = portfolio_values.pct_change().dropna()

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(portfolio_values) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = (portfolio_returns > 0).mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'n_periods': len(portfolio_values),
            'n_years': n_years
        }

    def save_results(self, backtest_result: BacktestResult, output_dir: str = 'results/bayesian_system'):
        """Save backtest results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save portfolio values
        backtest_result.portfolio_values.to_csv(f'{output_dir}/portfolio_values.csv')

        # Save weights history
        backtest_result.weights_history.to_csv(f'{output_dir}/weights_history.csv')

        # Save regime history
        backtest_result.regime_history.to_csv(f'{output_dir}/regime_history.csv')

        # Save metrics
        metrics_df = pd.DataFrame([backtest_result.performance_metrics])
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)

        print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("Integrated Bayesian Portfolio System")
    print("=" * 60)
    print("Complete example:")
    print("""
    from strategy.integrated_bayesian_system import IntegratedBayesianSystem
    import yfinance as yf
    import pandas as pd

    # Download data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'GLD']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')
    prices = data['Close']

    # Initialize system
    system = IntegratedBayesianSystem(
        n_regimes=5,
        mixture_type='bayesian_gaussian',
        auto_select_k=True,
        risk_aversion=1.0,
        rebalance_frequency='monthly',
        transaction_cost=0.001
    )

    # Run backtest
    result = system.backtest(
        prices=prices,
        initial_capital=100000,
        train_window=252,
        constraints={'long_only': True}
    )

    # Save results
    system.save_results(result)

    # Access results
    print(f"Final value: ${result.portfolio_values.iloc[-1]:,.0f}")
    print(f"Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.3f}")
    """)
