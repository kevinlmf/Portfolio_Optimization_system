"""
Adaptive Portfolio Optimization System

This script demonstrates a dynamic portfolio optimization approach that:
1. Detects current market regime
2. Selects the optimal optimization method for that regime
3. Generates regime-specific portfolio allocations
4. Compares portfolios across different regimes

Automatically selects the best optimization method based on market conditions
and generates corresponding asset allocations

Usage:
    python scripts/adaptive_portfolio_optimizer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from strategy.market_regime_detector import MarketRegimeDetector
from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector
from strategy.sparse_sharpe_optimizer import SparseSharpeOptimizer

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


class AdaptivePortfolioOptimizer:
    """
    Adaptive Portfolio Optimizer

    Dynamically selects optimization methods and generates corresponding
    asset allocations based on market conditions
    """

    def __init__(self, tickers: List[str], lookback_days: int = 252,
                 risk_free_rate: float = 0.03):
        """
        Initialize adaptive optimizer.

        Args:
            tickers: List of asset tickers
            lookback_days: Historical data window
            risk_free_rate: Annual risk-free rate
        """
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

        self.prices = None
        self.returns = None
        self.regime_detector = MarketRegimeDetector()
        self.optimizer_selector = IntelligentOptimizerSelector()

        self.results = {}  # Store optimization results

    def fetch_data(self, end_date: datetime = None):
        """Download market data."""
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 100)

        print(f"\n{'='*70}")
        print("FETCHING MARKET DATA")
        print(f"{'='*70}")
        print(f"Assets: {', '.join(self.tickers)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        data = yf.download(self.tickers, start=start_date, end=end_date,
                          progress=False)

        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']

        # Clean data
        prices = prices.dropna(axis=1, how='all')
        prices = prices.fillna(method='ffill').fillna(method='bfill')

        # Take last lookback_days
        self.prices = prices.tail(self.lookback_days)
        self.returns = self.prices.pct_change().dropna()

        print(f"Downloaded {len(self.prices.columns)} assets")
        print(f"{len(self.prices)} trading days")

        return self.prices, self.returns

    def detect_current_regime(self, market_index: str = None):
        """Detect current market regime."""
        print(f"\n{'='*70}")
        print("DETECTING MARKET REGIME")
        print(f"{'='*70}")

        if market_index and market_index in self.prices.columns:
            market_prices = self.prices[market_index]
        else:
            # Use first asset or SPY as market proxy
            market_prices = self.prices.iloc[:, 0]

        regime_result = self.regime_detector.detect_regime(market_prices)

        print(f"\nDetected Regime: {regime_result['regime'].upper().replace('_', ' ')}")
        print(f"   Confidence: {regime_result['confidence']:.1%}")
        print(f"\n{regime_result['recommendation']}")

        return regime_result

    def get_recommended_method(self, regime_result: Dict,
                               preferences: Dict = None) -> Tuple[str, Dict]:
        """Get recommended optimization method."""
        print(f"\n{'='*70}")
        print("SELECTING OPTIMIZATION METHOD")
        print(f"{'='*70}")

        if preferences is None:
            preferences = {
                'prefer_sparse': False,
                'prefer_interpretable': True,
                'prefer_robust': True
            }

        recommendation = self.optimizer_selector.select_optimizer(
            prices=self.prices.iloc[:, 0],  # Market proxy
            returns=self.returns,
            preferences=preferences
        )

        print(f"\nRecommended Method: {recommendation.recommended_method.upper()}")
        print(f"   Confidence: {recommendation.confidence:.1%}")
        print(f"\nReasoning:")
        for reason in recommendation.reasoning:
            print(f"  - {reason}")

        return recommendation.recommended_method, recommendation

    def optimize_portfolio(self, method: str, m: int = None) -> Dict:
        """
        Optimize portfolio using specified method.

        Args:
            method: Optimization method name
            m: Sparsity parameter (for sparse methods)

        Returns:
            Dictionary with weights and statistics
        """
        print(f"\n{'─'*70}")
        print(f"OPTIMIZING WITH: {method.upper()}")
        print(f"{'─'*70}")

        returns_array = self.returns.values

        if method == 'sparse_sharpe':
            # Use sparse Sharpe optimizer
            if m is None:
                m = max(5, len(self.tickers) // 3)  # Default: 1/3 of assets

            optimizer = SparseSharpeOptimizer(epsilon=1e-3, max_iter=5000,
                                             verbose=True)
            result = optimizer.optimize(returns_array, m=m)

            weights = result['weights']
            sharpe = result['sharpe_ratio']

        elif method == 'max_sharpe':
            weights = self._optimize_max_sharpe(returns_array)
            sharpe = self._calculate_sharpe(weights, returns_array)

        elif method == 'min_variance':
            weights = self._optimize_min_variance(returns_array)
            sharpe = self._calculate_sharpe(weights, returns_array)

        elif method == 'risk_parity':
            weights = self._optimize_risk_parity(returns_array)
            sharpe = self._calculate_sharpe(weights, returns_array)

        elif method == 'equal_weight':
            weights = np.ones(len(self.tickers)) / len(self.tickers)
            sharpe = self._calculate_sharpe(weights, returns_array)

        else:
            # Default to equal weight
            print(f"Warning: Method {method} not fully implemented, using equal weight")
            weights = np.ones(len(self.tickers)) / len(self.tickers)
            sharpe = self._calculate_sharpe(weights, returns_array)

        # Calculate portfolio statistics
        portfolio_returns = returns_array @ weights

        stats = {
            'method': method,
            'weights': weights,
            'sharpe_ratio': sharpe,
            'annual_return': np.mean(portfolio_returns) * 252,
            'annual_volatility': np.std(portfolio_returns) * np.sqrt(252),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights[weights > 1e-8]) if np.sum(weights > 1e-8) > 0 else 0,
            'num_holdings': np.sum(weights > 0.01),
            'top_holdings': self._get_top_holdings(weights, n=5)
        }

        print(f"\nOptimization Complete:")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Annual Return: {stats['annual_return']:.2%}")
        print(f"  Annual Volatility: {stats['annual_volatility']:.2%}")
        print(f"  Number of Holdings: {stats['num_holdings']}")

        return stats

    def _optimize_max_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Simple max Sharpe optimization."""
        from scipy.optimize import minimize

        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return / portfolio_vol)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.3) for _ in range(n_assets))
        initial = np.ones(n_assets) / n_assets

        result = minimize(neg_sharpe, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def _optimize_min_variance(self, returns: np.ndarray) -> np.ndarray:
        """Minimum variance optimization."""
        from scipy.optimize import minimize

        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.3) for _ in range(n_assets))
        initial = np.ones(n_assets) / n_assets

        result = minimize(portfolio_variance, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def _optimize_risk_parity(self, returns: np.ndarray) -> np.ndarray:
        """Risk parity optimization."""
        from scipy.optimize import minimize

        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)

        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.001, 0.3) for _ in range(n_assets))
        initial = np.ones(n_assets) / n_assets

        result = minimize(risk_parity_objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def _calculate_sharpe(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        portfolio_returns = returns @ weights
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns, ddof=1)

        if std_return < 1e-10:
            return 0.0

        return (mean_return * 252) / (std_return * np.sqrt(252))

    def _get_top_holdings(self, weights: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n holdings."""
        holdings = [(self.tickers[i], weights[i])
                   for i in range(len(weights)) if weights[i] > 1e-8]
        holdings.sort(key=lambda x: x[1], reverse=True)
        return holdings[:n]

    def run_adaptive_optimization(self, test_all_methods: bool = False):
        """
        Run adaptive optimization pipeline.

        Args:
            test_all_methods: If True, test all methods for comparison
        """
        print(f"\n{'#'*70}")
        print("#" + " "*68 + "#")
        print("#" + " "*15 + "ADAPTIVE PORTFOLIO OPTIMIZATION" + " "*22 + "#")
        print("#" + " "*68 + "#")
        print(f"{'#'*70}\n")

        # Step 1: Detect regime
        regime_result = self.detect_current_regime()

        # Step 2: Get recommended method
        recommended_method, recommendation = self.get_recommended_method(regime_result)

        # Step 3: Optimize with recommended method
        self.results['recommended'] = self.optimize_portfolio(recommended_method)

        # Step 4: Optionally test all methods for comparison
        if test_all_methods:
            print(f"\n{'='*70}")
            print("TESTING ALL METHODS FOR COMPARISON")
            print(f"{'='*70}")

            all_methods = ['max_sharpe', 'min_variance', 'risk_parity',
                          'equal_weight', 'sparse_sharpe']

            for method in all_methods:
                if method != recommended_method:
                    try:
                        self.results[method] = self.optimize_portfolio(method)
                    except Exception as e:
                        print(f"Warning: Error optimizing {method}: {str(e)}")

        return self.results

    def create_comparison_report(self):
        """Generate comprehensive comparison report."""
        print(f"\n{'='*70}")
        print("PORTFOLIO COMPARISON REPORT")
        print(f"{'='*70}\n")

        # Create comparison table
        comparison_data = []
        for name, stats in self.results.items():
            comparison_data.append({
                'Method': stats['method'].replace('_', ' ').title(),
                'Recommended': 'Yes' if name == 'recommended' else '',
                'Sharpe Ratio': f"{stats['sharpe_ratio']:.4f}",
                'Annual Return': f"{stats['annual_return']:.2%}",
                'Annual Vol': f"{stats['annual_volatility']:.2%}",
                'Holdings': stats['num_holdings'],
                'Max Weight': f"{stats['max_weight']:.1%}"
            })

        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # Print top holdings for recommended portfolio
        print(f"\n{'─'*70}")
        print("RECOMMENDED PORTFOLIO - TOP HOLDINGS")
        print(f"{'─'*70}")

        recommended = self.results['recommended']
        for i, (ticker, weight) in enumerate(recommended['top_holdings'], 1):
            print(f"{i}. {ticker:6s} {weight:6.2%}")

        return df_comparison

    def visualize_results(self, save_path: str = None):
        """Create visualization of results."""
        print(f"\n{'='*70}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*70}")

        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Sharpe Ratio Comparison
        ax = axes[0, 0]
        methods = [stats['method'].replace('_', ' ').title()
                  for stats in self.results.values()]
        sharpes = [stats['sharpe_ratio'] for stats in self.results.values()]
        colors = ['green' if name == 'recommended' else 'steelblue'
                 for name in self.results.keys()]

        bars = ax.barh(methods, sharpes, color=colors)
        ax.set_xlabel('Sharpe Ratio', fontsize=12)
        ax.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Plot 2: Risk-Return Scatter
        ax = axes[0, 1]
        returns = [stats['annual_return'] * 100 for stats in self.results.values()]
        vols = [stats['annual_volatility'] * 100 for stats in self.results.values()]

        for i, (name, stats) in enumerate(self.results.items()):
            color = 'green' if name == 'recommended' else 'steelblue'
            marker = 'o' if name == 'recommended' else 's'
            size = 200 if name == 'recommended' else 100
            ax.scatter(vols[i], returns[i], c=color, marker=marker, s=size,
                      label=methods[i], alpha=0.7, edgecolors='black')

        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # Plot 3: Recommended Portfolio Weights
        ax = axes[1, 0]
        recommended = self.results['recommended']
        weights = recommended['weights']

        # Show only holdings > 1%
        significant_holdings = [(self.tickers[i], weights[i])
                               for i in range(len(weights)) if weights[i] > 0.01]
        significant_holdings.sort(key=lambda x: x[1], reverse=True)

        if significant_holdings:
            tickers_plot = [h[0] for h in significant_holdings]
            weights_plot = [h[1] * 100 for h in significant_holdings]

            ax.barh(tickers_plot, weights_plot, color='green', alpha=0.7)
            ax.set_xlabel('Weight (%)', fontsize=12)
            ax.set_title('Recommended Portfolio Allocation', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        # Plot 4: Number of Holdings Comparison
        ax = axes[1, 1]
        holdings = [stats['num_holdings'] for stats in self.results.values()]

        bars = ax.bar(methods, holdings, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Holdings', fontsize=12)
        ax.set_title('Portfolio Concentration', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = 'results/adaptive_optimization_results.png'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

        return fig


def main():
    """Main execution function."""

    # Define asset universe
    tickers = [
        # Tech leaders
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        # Financials
        'JPM', 'BAC', 'GS', 'V',
        # Healthcare
        'JNJ', 'UNH', 'PFE',
        # Consumer
        'WMT', 'HD', 'MCD',
        # Energy
        'XOM', 'CVX',
        # ETFs/Diversification
        'SPY', 'QQQ', 'TLT', 'GLD'
    ]

    try:
        # Initialize adaptive optimizer
        optimizer = AdaptivePortfolioOptimizer(
            tickers=tickers,
            lookback_days=252,
            risk_free_rate=0.03
        )

        # Fetch market data
        optimizer.fetch_data()

        # Run adaptive optimization
        # Set test_all_methods=True to compare all methods
        results = optimizer.run_adaptive_optimization(test_all_methods=True)

        # Generate comparison report
        comparison_df = optimizer.create_comparison_report()

        # Create visualizations
        fig = optimizer.visualize_results()

        # Save comparison table
        output_path = 'results/adaptive_optimization_comparison.csv'
        os.makedirs('results', exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\nComparison table saved to: {output_path}")

        print(f"\n{'#'*70}")
        print("#" + " "*68 + "#")
        print("#" + " "*18 + "OPTIMIZATION COMPLETED" + " "*27 + "#")
        print("#" + " "*68 + "#")
        print(f"{'#'*70}\n")

        print("\nKEY INSIGHTS:")
        print("1. The system automatically detected the market regime")
        print("2. Recommended the most suitable optimization method")
        print("3. Generated regime-appropriate portfolio allocation")
        print("4. Compared performance across different methods")
        print("5. All results saved to 'results/' directory")

        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
