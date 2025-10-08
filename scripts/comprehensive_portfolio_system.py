"""
Comprehensive Multi-Asset Portfolio Optimization System

This system integrates:
1. Multi-asset class data fetching (stocks, bonds, commodities, crypto, ETFs)
2. Market regime detection
3. Intelligent optimizer selection
4. Rolling window backtesting
5. Monte Carlo simulation validation
6. Comprehensive performance analytics

Usage:
    python scripts/comprehensive_portfolio_system.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio
from strategy.market_regime_detector import MarketRegimeDetector
from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector
from strategy.sparse_sharpe_optimizer import SparseSharpeOptimizer
from strategy.backtesting_engine import BacktestingEngine, MonteCarloSimulator

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)


class ComprehensivePortfolioSystem:
    """
    End-to-end portfolio optimization system with adaptive allocation
    and comprehensive validation.
    """

    def __init__(self,
                 start_date: str = None,
                 end_date: str = None,
                 risk_free_rate: float = 0.03,
                 transaction_cost: float = 0.001,
                 rebalance_frequency: str = 'monthly'):
        """
        Initialize comprehensive portfolio system.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            risk_free_rate: Annual risk-free rate
            transaction_cost: Transaction cost per trade
            rebalance_frequency: Rebalancing frequency
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency

        # Components
        self.data_fetcher = MultiAssetFetcher(start_date=self.start_date, end_date=self.end_date)
        self.regime_detector = MarketRegimeDetector()
        self.optimizer_selector = IntelligentOptimizerSelector()

        # Data storage
        self.prices = None
        self.returns = None
        self.asset_info = None

        # Results
        self.regime_result = None
        self.recommended_method = None
        self.backtest_results = None
        self.monte_carlo_results = None

    def setup_portfolio(self, scenario: str = 'all_assets', custom_tickers: List[str] = None):
        """
        Setup portfolio assets.

        Args:
            scenario: 'aggressive', 'balanced', 'conservative', 'all_assets'
            custom_tickers: Custom list of tickers (overrides scenario)
        """
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*15 + "COMPREHENSIVE PORTFOLIO OPTIMIZATION SYSTEM" + " "*20 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        # Select tickers
        if custom_tickers:
            tickers = custom_tickers
            print(f"Using custom tickers: {len(tickers)} assets")
        else:
            tickers = create_example_portfolio(scenario)
            print(f"Using {scenario} portfolio: {len(tickers)} assets")

        # Fetch data
        self.prices, self.returns = self.data_fetcher.fetch_assets(tickers)
        self.asset_info = self.data_fetcher.asset_info

        # Print summary
        self.data_fetcher.print_summary()

        return self.prices, self.returns

    def detect_market_regime(self, market_proxy: str = 'SPY'):
        """
        Detect current market regime.

        Args:
            market_proxy: Ticker to use as market proxy
        """
        print(f"\n{'='*80}")
        print("STEP 1: MARKET REGIME DETECTION")
        print(f"{'='*80}")

        # Use market proxy for regime detection
        if market_proxy in self.prices.columns:
            market_prices = self.prices[market_proxy]
        else:
            # Use first equity ETF or first asset
            market_prices = self.prices.iloc[:, 0]
            print(f"Market proxy {market_proxy} not found, using {self.prices.columns[0]}")

        self.regime_result = self.regime_detector.detect_regime(market_prices)

        print(f"\n✓ Detected Regime: {self.regime_result['regime'].upper().replace('_', ' ')}")
        print(f"  Confidence: {self.regime_result['confidence']:.1%}")
        print(f"\n{self.regime_result['recommendation']}")

        return self.regime_result

    def select_optimization_method(self, preferences: Dict = None):
        """
        Select optimal optimization method based on regime and asset configuration.

        Args:
            preferences: User preferences dictionary
        """
        print(f"\n{'='*80}")
        print("STEP 2: INTELLIGENT OPTIMIZER SELECTION")
        print(f"{'='*80}")

        if preferences is None:
            preferences = {
                'prefer_sparse': False,
                'prefer_interpretable': True,
                'prefer_robust': True
            }

        # Use market proxy for regime detection
        market_prices = self.prices.iloc[:, 0]

        recommendation = self.optimizer_selector.select_optimizer(
            prices=market_prices,
            returns=self.returns,
            asset_info=self.asset_info,
            preferences=preferences
        )

        self.recommended_method = recommendation.recommended_method

        print(f"\n✓ Recommended Method: {self.recommended_method.upper()}")
        print(f"  Confidence: {recommendation.confidence:.1%}")
        print(f"\nReasoning:")
        for reason in recommendation.reasoning:
            print(f"  {reason}")

        print(f"\nExpected Characteristics:")
        exp = recommendation.expected_characteristics
        print(f"  Max Weight Range: {exp['expected_max_weight_range'][0]:.1%} - {exp['expected_max_weight_range'][1]:.1%}")
        print(f"  Number of Positions: {exp['expected_n_positions_range'][0]}-{exp['expected_n_positions_range'][1]}")
        print(f"  Expected Turnover: {exp['expected_turnover'].title()}")
        print(f"  Rebalancing Frequency: {exp['rebalancing_frequency_suggestion'].title()}")

        return recommendation

    def run_backtest(self, test_multiple_methods: bool = True):
        """
        Run comprehensive backtest with regime-adaptive optimization.

        Args:
            test_multiple_methods: Compare multiple optimization methods
        """
        print(f"\n{'='*80}")
        print("STEP 3: BACKTESTING WITH REGIME-ADAPTIVE OPTIMIZATION")
        print(f"{'='*80}")

        # Define optimization function that adapts to regime
        def adaptive_optimization(returns_df, regime_window=60):
            """Optimization function that detects regime and selects method."""
            # Detect regime using recent prices
            recent_prices = (1 + returns_df).cumprod().iloc[:, 0]
            if len(recent_prices) >= regime_window:
                regime_result = self.regime_detector.detect_regime(recent_prices)
                regime = regime_result['regime']
            else:
                regime = 'sideways'  # Default

            # Select method based on regime
            regime_to_method = {
                'bull_market': 'max_sharpe',
                'bear_market': 'min_variance',
                'sideways': 'risk_parity',
                'high_volatility': 'min_variance',
                'crisis': 'equal_weight'
            }
            method = regime_to_method.get(regime, 'risk_parity')

            # Optimize
            weights = self._optimize_portfolio(returns_df, method)
            return weights

        # Run backtest
        backtest_engine = BacktestingEngine(
            returns=self.returns,
            prices=self.prices,
            transaction_cost=self.transaction_cost,
            rebalance_frequency=self.rebalance_frequency,
            risk_free_rate=self.risk_free_rate
        )

        self.backtest_results = backtest_engine.run_backtest(
            optimization_func=adaptive_optimization,
            lookback_window=252,
            min_history=126
        )

        # Print metrics
        backtest_engine.print_metrics_report()

        # Compare with benchmark (SPY)
        if 'SPY' in self.returns.columns:
            print(f"\n{'='*80}")
            print("BENCHMARK COMPARISON (SPY)")
            print(f"{'='*80}")

            comparison = backtest_engine.compare_with_benchmark(self.returns['SPY'])

            print(f"\nAlpha & Beta:")
            print(f"  Alpha:                {comparison['alpha']:>8.2%}")
            print(f"  Beta:                 {comparison['beta']:>8.4f}")
            print(f"  Information Ratio:    {comparison['information_ratio']:>8.4f}")

            print(f"\nExcess Performance:")
            print(f"  Excess Return:        {comparison['excess_return']:>8.2%}")
            print(f"  Win Rate vs Benchmark: {comparison['win_rate_vs_benchmark']:>8.2%}")

        # Export results
        backtest_engine.export_results('results/comprehensive_backtest')

        # Test multiple methods if requested
        if test_multiple_methods:
            self._test_multiple_methods()

        return self.backtest_results

    def _optimize_portfolio(self, returns_df: pd.DataFrame, method: str) -> np.ndarray:
        """Helper function to optimize portfolio with given method."""
        from scipy.optimize import minimize

        returns_array = returns_df.values
        n_assets = returns_array.shape[1]
        mean_returns = np.mean(returns_array, axis=0)
        cov_matrix = np.cov(returns_array.T)

        if method == 'max_sharpe':
            def objective(w):
                ret = np.dot(w, mean_returns)
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                return -(ret / vol) if vol > 0 else 1e6

        elif method == 'min_variance':
            def objective(w):
                return np.dot(w.T, np.dot(cov_matrix, w))

        elif method == 'risk_parity':
            def objective(w):
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                marginal_contrib = np.dot(cov_matrix, w) / vol
                risk_contrib = w * marginal_contrib
                target_risk = vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)

        elif method == 'equal_weight':
            return np.ones(n_assets) / n_assets

        else:
            return np.ones(n_assets) / n_assets

        # Optimize
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.3) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def _test_multiple_methods(self):
        """Test and compare multiple optimization methods."""
        print(f"\n{'='*80}")
        print("TESTING MULTIPLE OPTIMIZATION METHODS")
        print(f"{'='*80}")

        methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
        comparison_results = []

        for method in methods:
            print(f"\nTesting {method}...")

            def method_optimization(returns_df):
                return self._optimize_portfolio(returns_df, method)

            engine = BacktestingEngine(
                returns=self.returns,
                prices=self.prices,
                transaction_cost=self.transaction_cost,
                rebalance_frequency=self.rebalance_frequency,
                risk_free_rate=self.risk_free_rate
            )

            try:
                result = engine.run_backtest(
                    optimization_func=method_optimization,
                    lookback_window=252,
                    min_history=126
                )

                metrics = result['metrics']
                comparison_results.append({
                    'Method': method.replace('_', ' ').title(),
                    'Total Return': f"{metrics['total_return']:.2%}",
                    'Annual Return': f"{metrics['annual_return']:.2%}",
                    'Volatility': f"{metrics['annual_volatility']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
                    'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                    'Win Rate': f"{metrics['win_rate']:.2%}"
                })

            except Exception as e:
                print(f"Error testing {method}: {e}")

        # Create comparison table
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            print(f"\n{'='*80}")
            print("METHOD COMPARISON")
            print(f"{'='*80}\n")
            print(comparison_df.to_string(index=False))

            # Export
            comparison_df.to_csv('results/comprehensive_backtest/method_comparison.csv', index=False)

    def run_monte_carlo_validation(self, n_simulations: int = 10000, n_days: int = 252):
        """
        Run Monte Carlo simulation for validation.

        Args:
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate
        """
        print(f"\n{'='*80}")
        print("STEP 4: MONTE CARLO SIMULATION VALIDATION")
        print(f"{'='*80}")

        # Get current optimal weights (last rebalance)
        if self.backtest_results and self.backtest_results['weights_history']:
            weights = self.backtest_results['weights_history'][-1]['weights']
        else:
            # Use equal weight as fallback
            weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        # Run parametric simulation
        print("\n--- Parametric (Normal) Method ---")
        mc_simulator = MonteCarloSimulator(
            returns=self.returns,
            n_simulations=n_simulations,
            n_days=n_days
        )

        parametric_results = mc_simulator.run_simulation(weights, method='parametric')

        # Run bootstrap simulation
        print("\n--- Bootstrap (Historical Resampling) Method ---")
        bootstrap_results = mc_simulator.run_simulation(weights, method='bootstrap')

        # Get confidence intervals
        ci_parametric = mc_simulator.get_confidence_intervals(confidence_level=0.95)

        print(f"\n95% Confidence Intervals (Parametric):")
        for horizon, interval in ci_parametric.items():
            print(f"  {horizon:5s}: [{interval['lower']:.4f}, {interval['upper']:.4f}]  (median: {interval['median']:.4f})")

        self.monte_carlo_results = {
            'parametric': parametric_results,
            'bootstrap': bootstrap_results,
            'confidence_intervals': ci_parametric,
            'simulated_paths': mc_simulator.simulated_paths
        }

        return self.monte_carlo_results

    def visualize_results(self, save_path: str = 'results/comprehensive_backtest'):
        """Create comprehensive visualization of results."""
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        if self.backtest_results:
            portfolio_values = self.backtest_results['portfolio_values']
            ax1.plot(portfolio_values.index, portfolio_values.values,
                    linewidth=2, color='navy', label='Portfolio')

            # Add benchmark if available
            if 'SPY' in self.prices.columns:
                spy_values = self.prices['SPY'] / self.prices['SPY'].iloc[0]
                spy_values = spy_values.loc[portfolio_values.index]
                ax1.plot(spy_values.index, spy_values.values,
                        linewidth=2, color='orange', alpha=0.7, label='SPY Benchmark')

            ax1.set_title('Portfolio Performance', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Value ($)')
            ax1.legend()
            ax1.grid(alpha=0.3)

        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, :])
        if self.backtest_results:
            cumulative = self.backtest_results['portfolio_values']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            ax2.fill_between(drawdown.index, drawdown.values, 0,
                            color='red', alpha=0.3, label='Drawdown')
            ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
            ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(alpha=0.3)

        # 3. Asset Class Allocation
        ax3 = fig.add_subplot(gs[2, 0])
        if self.asset_info is not None:
            asset_class_counts = self.asset_info['asset_class'].value_counts()
            colors = plt.cm.Set3(range(len(asset_class_counts)))
            ax3.pie(asset_class_counts.values, labels=asset_class_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Asset Class Distribution', fontsize=12, fontweight='bold')

        # 4. Monte Carlo Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        if self.monte_carlo_results:
            final_values = self.monte_carlo_results['simulated_paths'][:, -1]
            ax4.hist(final_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.axvline(np.median(final_values), color='red', linestyle='--',
                       linewidth=2, label=f'Median: ${np.median(final_values):.2f}')
            ax4.set_title('Monte Carlo Final Value Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Final Portfolio Value ($)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(alpha=0.3)

        # 5. Performance Metrics Bar Chart
        ax5 = fig.add_subplot(gs[2, 2])
        if self.backtest_results:
            metrics = self.backtest_results['metrics']
            metric_names = ['Annual\nReturn', 'Sharpe\nRatio', 'Max\nDrawdown', 'Win\nRate']
            metric_values = [
                metrics['annual_return'] * 100,
                metrics['sharpe_ratio'],
                metrics['max_drawdown'] * 100,
                metrics['win_rate'] * 100
            ]
            colors = ['green' if v > 0 else 'red' for v in metric_values]
            ax5.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
            ax5.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
            ax5.axhline(0, color='black', linewidth=0.8)

        plt.savefig(f'{save_path}/comprehensive_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}/comprehensive_results.png")

        return fig

    def generate_final_report(self):
        """Generate comprehensive final report."""
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*25 + "FINAL COMPREHENSIVE REPORT" + " "*26 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        report = []
        report.append(f"Analysis Period: {self.start_date} to {self.end_date}")
        report.append(f"Number of Assets: {len(self.returns.columns)}")
        report.append(f"Rebalance Frequency: {self.rebalance_frequency}")

        if self.regime_result:
            report.append(f"\nMarket Regime: {self.regime_result['regime'].upper()}")
            report.append(f"  Confidence: {self.regime_result['confidence']:.1%}")

        if self.recommended_method:
            report.append(f"\nRecommended Method: {self.recommended_method.upper()}")

        if self.backtest_results:
            metrics = self.backtest_results['metrics']
            report.append(f"\nBacktest Performance:")
            report.append(f"  Total Return:       {metrics['total_return']:>8.2%}")
            report.append(f"  Annual Return:      {metrics['annual_return']:>8.2%}")
            report.append(f"  Annual Volatility:  {metrics['annual_volatility']:>8.2%}")
            report.append(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>8.4f}")
            report.append(f"  Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
            report.append(f"  Win Rate:           {metrics['win_rate']:>8.2%}")

        if self.monte_carlo_results:
            mc_results = self.monte_carlo_results['parametric']
            report.append(f"\nMonte Carlo Validation:")
            report.append(f"  Expected Value:     ${mc_results['mean_final_value']:>8.4f}")
            report.append(f"  95% VaR:           ${mc_results['var_95']:>8.4f}")
            report.append(f"  Prob of Profit:     {mc_results['prob_positive']:>8.2%}")

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        with open('results/comprehensive_backtest/final_report.txt', 'w') as f:
            f.write(report_text)

        print(f"\n✓ Report saved to results/comprehensive_backtest/final_report.txt")


def main():
    """Main execution function."""
    # Initialize system
    system = ComprehensivePortfolioSystem(
        start_date='2020-01-01',
        end_date='2024-01-01',
        risk_free_rate=0.03,
        transaction_cost=0.001,
        rebalance_frequency='monthly'
    )

    # Step 1: Setup portfolio with diverse assets
    system.setup_portfolio(scenario='all_assets')

    # Step 2: Detect market regime
    system.detect_market_regime(market_proxy='SPY')

    # Step 3: Select optimization method
    system.select_optimization_method()

    # Step 4: Run backtest
    system.run_backtest(test_multiple_methods=True)

    # Step 5: Run Monte Carlo validation
    system.run_monte_carlo_validation(n_simulations=10000, n_days=252)

    # Step 6: Visualize results
    system.visualize_results()

    # Step 7: Generate final report
    system.generate_final_report()

    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "ANALYSIS COMPLETE" + " "*35 + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}\n")

    print("All results saved to results/comprehensive_backtest/")


if __name__ == '__main__':
    main()
