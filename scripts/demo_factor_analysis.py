"""
Factor Analysis Demo Script

Demonstrates the integrated factor system combining:
1. 133 factors from Factor Mining System
2. Fama-French factor models
3. PCA-based factor extraction
4. Factor-tilted portfolio construction
5. Factor timing strategies

Usage:
    python scripts/demo_factor_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio
from strategy.integrated_factor_system import IntegratedFactorSystem
from strategy.backtesting_engine import BacktestingEngine

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)


class FactorAnalysisDemo:
    """
    Comprehensive demonstration of factor analysis capabilities.
    """

    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize demo.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        self.prices = None
        self.returns = None
        self.factor_system = None
        self.results = {}

    def load_data(self, scenario: str = 'balanced'):
        """
        Load market data.

        Args:
            scenario: Portfolio scenario ('aggressive', 'balanced', 'conservative')
        """
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*20 + "FACTOR ANALYSIS DEMONSTRATION" + " "*28 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        print(f"{'='*80}")
        print("STEP 1: DATA LOADING")
        print(f"{'='*80}\n")

        # Fetch data
        fetcher = MultiAssetFetcher(start_date=self.start_date, end_date=self.end_date)
        tickers = create_example_portfolio(scenario)

        self.prices, self.returns = fetcher.fetch_assets(tickers)
        fetcher.print_summary()

        # Initialize factor system
        self.factor_system = IntegratedFactorSystem(
            returns=self.returns,
            prices=self.prices,
            risk_free_rate=0.03
        )

        print(f"\n✓ Data loaded: {len(tickers)} assets, {len(self.returns)} days")

    def demo_style_factors(self):
        """Demonstrate Fama-French style factor construction."""
        print(f"\n{'='*80}")
        print("STEP 2: FAMA-FRENCH STYLE FACTORS")
        print(f"{'='*80}\n")

        # Construct factors
        if 'SPY' in self.returns.columns:
            market_proxy = self.returns['SPY']
        else:
            market_proxy = None

        factors = self.factor_system.construct_style_factors(market_proxy=market_proxy)

        # Store results
        self.results['style_factors'] = factors

        return factors

    def demo_statistical_factors(self, n_factors: int = 5):
        """Demonstrate PCA factor extraction."""
        print(f"\n{'='*80}")
        print("STEP 3: STATISTICAL FACTORS (PCA)")
        print(f"{'='*80}\n")

        factor_returns, loadings = self.factor_system.extract_statistical_factors(
            n_factors=n_factors
        )

        # Print top loadings
        print(f"\nTop Asset Loadings on PC1:")
        pc1_loadings = pd.Series(loadings[:, 0], index=self.returns.columns)
        print(pc1_loadings.abs().sort_values(ascending=False).head(10))

        # Store results
        self.results['pca_factors'] = factor_returns
        self.results['pca_loadings'] = loadings

        return factor_returns, loadings

    def demo_portfolio_factor_analysis(self):
        """Demonstrate portfolio factor analysis."""
        print(f"\n{'='*80}")
        print("STEP 4: PORTFOLIO FACTOR ANALYSIS")
        print(f"{'='*80}\n")

        # Create equal-weight portfolio
        n_assets = len(self.returns.columns)
        equal_weights = np.ones(n_assets) / n_assets
        portfolio_returns = (self.returns * equal_weights).sum(axis=1)

        # Run factor analysis
        report = self.factor_system.analyze_portfolio_factors(
            portfolio_returns=portfolio_returns,
            portfolio_weights=equal_weights,
            models=['fama_french_3', 'carhart_4', 'fama_french_5']
        )

        self.results['equal_weight_analysis'] = report

        return report

    def demo_factor_timing(self):
        """Demonstrate factor timing analysis."""
        print(f"\n{'='*80}")
        print("STEP 5: FACTOR TIMING")
        print(f"{'='*80}\n")

        timing_results = self.factor_system.run_factor_timing(
            lookback=60,
            top_n_factors=3
        )

        self.results['timing'] = timing_results

        return timing_results

    def demo_factor_tilted_portfolios(self):
        """Demonstrate factor-tilted portfolio construction."""
        print(f"\n{'='*80}")
        print("STEP 6: FACTOR-TILTED PORTFOLIOS")
        print(f"{'='*80}\n")

        # Get top momentum factors
        timing_results = self.results.get('timing')
        if timing_results is None:
            timing_results = self.demo_factor_timing()

        top_factors = timing_results['top_factors']

        # Test different base methods
        tilted_portfolios = {}

        for method in ['risk_parity', 'max_sharpe', 'min_variance']:
            print(f"\n--- Base Method: {method.upper()} ---")

            weights = self.factor_system.create_factor_tilted_portfolio(
                tilt_factors=top_factors,
                tilt_strength=1.0,
                base_method=method
            )

            tilted_portfolios[method] = weights

        self.results['tilted_portfolios'] = tilted_portfolios

        return tilted_portfolios

    def demo_factor_exposure_optimization(self):
        """Demonstrate factor exposure optimization."""
        print(f"\n{'='*80}")
        print("STEP 7: FACTOR EXPOSURE OPTIMIZATION")
        print(f"{'='*80}\n")

        # Define target exposures
        target_exposures = {
            'MKT': 0.8,   # Moderate market exposure
            'SMB': 0.2,   # Small cap tilt
            'MOM': 0.3,   # Momentum tilt
            'HML': -0.1,  # Slight growth tilt
        }

        weights = self.factor_system.optimize_factor_exposures(
            target_exposures=target_exposures,
            max_weight=0.25,
            min_weight=0.0
        )

        self.results['exposure_optimized_weights'] = weights

        return weights

    def compare_strategies(self):
        """Compare different factor-based strategies."""
        print(f"\n{'='*80}")
        print("STEP 8: STRATEGY COMPARISON")
        print(f"{'='*80}\n")

        strategies = {}

        # 1. Equal Weight
        n_assets = len(self.returns.columns)
        strategies['Equal Weight'] = np.ones(n_assets) / n_assets

        # 2. Factor-tilted portfolios
        if 'tilted_portfolios' in self.results:
            for method, weights in self.results['tilted_portfolios'].items():
                strategies[f'Factor Tilt ({method})'] = weights

        # 3. Exposure optimized
        if 'exposure_optimized_weights' in self.results:
            strategies['Exposure Optimized'] = self.results['exposure_optimized_weights']

        # Calculate performance
        comparison_results = []

        for name, weights in strategies.items():
            portfolio_returns = (self.returns * weights).sum(axis=1)

            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0

            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            comparison_results.append({
                'Strategy': name,
                'Total Return': f'{total_return:.2%}',
                'Annual Return': f'{annual_return:.2%}',
                'Volatility': f'{volatility:.2%}',
                'Sharpe Ratio': f'{sharpe:.4f}',
                'Max Drawdown': f'{max_drawdown:.2%}',
                'Active Assets': int((weights > 0.001).sum())
            })

        comparison_df = pd.DataFrame(comparison_results)
        print(f"\n{'='*80}")
        print("STRATEGY PERFORMANCE COMPARISON")
        print(f"{'='*80}\n")
        print(comparison_df.to_string(index=False))

        self.results['comparison'] = comparison_df

        # Save results
        os.makedirs('results/factor_analysis', exist_ok=True)
        comparison_df.to_csv('results/factor_analysis/strategy_comparison.csv', index=False)
        print(f"\n✓ Results saved to results/factor_analysis/strategy_comparison.csv")

        return comparison_df

    def visualize_results(self):
        """Create comprehensive visualizations."""
        print(f"\n{'='*80}")
        print("STEP 9: VISUALIZATION")
        print(f"{'='*80}\n")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Factor Returns
        if 'style_factors' in self.results:
            ax1 = fig.add_subplot(gs[0, :])
            factors = self.results['style_factors']
            cumulative_factors = (1 + factors).cumprod()

            for col in cumulative_factors.columns:
                ax1.plot(cumulative_factors.index, cumulative_factors[col],
                        label=col, linewidth=2)

            ax1.set_title('Style Factor Cumulative Returns', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend(loc='best', ncol=3)
            ax1.grid(alpha=0.3)

        # 2. Factor Correlation Heatmap
        if 'style_factors' in self.results:
            ax2 = fig.add_subplot(gs[1, 0])
            factors = self.results['style_factors']
            corr = factors.corr()

            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax2, square=True, cbar_kws={'shrink': 0.8})
            ax2.set_title('Factor Correlations', fontsize=12, fontweight='bold')

        # 3. PCA Explained Variance
        if 'pca_loadings' in self.results:
            ax3 = fig.add_subplot(gs[1, 1])
            # Calculate explained variance from factor system
            if hasattr(self.factor_system.factor_analyzer, 'pca_model'):
                pca = self.factor_system.factor_analyzer.pca_model
                explained_var = pca.explained_variance_ratio_
                cum_var = np.cumsum(explained_var)

                x = np.arange(1, len(explained_var) + 1)
                ax3.bar(x, explained_var, alpha=0.6, label='Individual')
                ax3.plot(x, cum_var, 'ro-', linewidth=2, label='Cumulative')
                ax3.set_xlabel('Principal Component')
                ax3.set_ylabel('Explained Variance Ratio')
                ax3.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(alpha=0.3)

        # 4. Factor Momentum
        if 'timing' in self.results:
            ax4 = fig.add_subplot(gs[1, 2])
            momentum_df = self.results['timing']['momentum']

            factors = momentum_df['Factor'].values
            momentum = momentum_df['Current_Momentum'].values

            colors = ['green' if x > 0 else 'red' for x in momentum]
            ax4.barh(factors, momentum, color=colors, alpha=0.7)
            ax4.set_xlabel('Momentum (Annualized)')
            ax4.set_title('Factor Momentum', fontsize=12, fontweight='bold')
            ax4.axvline(0, color='black', linewidth=0.8)
            ax4.grid(alpha=0.3, axis='x')

        # 5. Strategy Performance Comparison
        if 'tilted_portfolios' in self.results:
            ax5 = fig.add_subplot(gs[2, :])

            for method, weights in self.results['tilted_portfolios'].items():
                portfolio_returns = (self.returns * weights).sum(axis=1)
                cumulative = (1 + portfolio_returns).cumprod()
                ax5.plot(cumulative.index, cumulative.values,
                        label=f'{method}', linewidth=2)

            # Add equal weight
            n_assets = len(self.returns.columns)
            equal_returns = (self.returns * (1/n_assets)).sum(axis=1)
            equal_cumulative = (1 + equal_returns).cumprod()
            ax5.plot(equal_cumulative.index, equal_cumulative.values,
                    label='Equal Weight', linewidth=2, linestyle='--', color='black')

            ax5.set_title('Factor-Tilted Portfolio Performance', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Cumulative Return')
            ax5.legend(loc='best')
            ax5.grid(alpha=0.3)

        # 6. Portfolio Weights Heatmap (Exposure Optimized)
        if 'exposure_optimized_weights' in self.results:
            ax6 = fig.add_subplot(gs[3, 0])
            weights = self.results['exposure_optimized_weights']
            top_weights = sorted(zip(self.returns.columns, weights),
                               key=lambda x: x[1], reverse=True)[:15]

            assets = [x[0] for x in top_weights]
            weight_values = [x[1] for x in top_weights]

            ax6.barh(assets, weight_values, color='steelblue', alpha=0.7)
            ax6.set_xlabel('Weight')
            ax6.set_title('Exposure Optimized Portfolio', fontsize=12, fontweight='bold')
            ax6.grid(alpha=0.3, axis='x')

        # 7. Sharpe Ratio Comparison
        if 'comparison' in self.results:
            ax7 = fig.add_subplot(gs[3, 1])
            comparison_df = self.results['comparison']

            strategies = comparison_df['Strategy'].values
            sharpe_ratios = [float(x) for x in comparison_df['Sharpe Ratio'].values]

            colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red'
                     for x in sharpe_ratios]
            ax7.barh(strategies, sharpe_ratios, color=colors, alpha=0.7)
            ax7.set_xlabel('Sharpe Ratio')
            ax7.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
            ax7.axvline(0, color='black', linewidth=0.8)
            ax7.grid(alpha=0.3, axis='x')

        # 8. Max Drawdown Comparison
        if 'comparison' in self.results:
            ax8 = fig.add_subplot(gs[3, 2])
            comparison_df = self.results['comparison']

            strategies = comparison_df['Strategy'].values
            drawdowns = [float(x.strip('%')) / 100 for x in comparison_df['Max Drawdown'].values]

            ax8.barh(strategies, drawdowns, color='red', alpha=0.6)
            ax8.set_xlabel('Max Drawdown')
            ax8.set_title('Downside Risk', fontsize=12, fontweight='bold')
            ax8.grid(alpha=0.3, axis='x')

        plt.savefig('results/factor_analysis/comprehensive_factor_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to results/factor_analysis/comprehensive_factor_analysis.png")

        return fig

    def generate_final_report(self):
        """Generate final comprehensive report."""
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*20 + "FACTOR ANALYSIS COMPLETE" + " "*32 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        # Generate comprehensive factor report
        if 'tilted_portfolios' in self.results:
            # Use risk parity tilted portfolio
            weights = self.results['tilted_portfolios']['risk_parity']
            portfolio_returns = (self.returns * weights).sum(axis=1)

            self.factor_system.generate_comprehensive_report(
                portfolio_returns=portfolio_returns,
                portfolio_weights=weights,
                save_path='results/factor_analysis'
            )

        print(f"\n✓ All results saved to results/factor_analysis/")
        print(f"\nFiles generated:")
        print(f"  - strategy_comparison.csv")
        print(f"  - comprehensive_factor_analysis.png")
        print(f"  - integrated_factor_report.txt")


def main():
    """Main execution function."""
    # Initialize demo
    demo = FactorAnalysisDemo(
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    # Run demonstration
    demo.load_data(scenario='all_assets')
    demo.demo_style_factors()
    demo.demo_statistical_factors(n_factors=5)
    demo.demo_portfolio_factor_analysis()
    demo.demo_factor_timing()
    demo.demo_factor_tilted_portfolios()
    demo.demo_factor_exposure_optimization()
    demo.compare_strategies()
    demo.visualize_results()
    demo.generate_final_report()

    print(f"\n{'='*80}")
    print("FACTOR ANALYSIS DEMONSTRATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
