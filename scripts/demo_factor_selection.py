"""
Factor Selection Demo - Finding the Most Important Factors

This script demonstrates how to identify the most predictive factors
from the 133-factor library using multiple selection methods.

Key Questions Answered:
1. Which factors have the highest predictive power? (IC Analysis)
2. Which factors generate the best returns? (Factor Returns)
3. Which factors are selected by regularization? (LASSO/Elastic Net)
4. Which factors are most important? (Random Forest)
5. What are the consensus top factors? (Aggregated Rankings)

Usage:
    python scripts/demo_factor_selection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio
from strategy.factor_selection import FactorSelector
from strategy.integrated_factor_system import IntegratedFactorSystem

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 14)

# Add path to Factors system
FACTORS_PATH = Path(__file__).parent.parent.parent.parent / "Factors"
if FACTORS_PATH.exists():
    sys.path.insert(0, str(FACTORS_PATH))


class FactorSelectionDemo:
    """
    Comprehensive demonstration of factor selection from 133-factor library.
    """

    def __init__(self, start_date: str = None, end_date: str = None):
        self.start_date = start_date or '2020-01-01'
        self.end_date = end_date or '2024-01-01'

        self.prices = None
        self.returns = None
        self.factor_data = None
        self.selector = None
        self.results = {}

    def load_data(self, scenario: str = 'balanced'):
        """Load market data and prepare for factor mining."""
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*15 + "FACTOR SELECTION FROM 133-FACTOR LIBRARY" + " "*21 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        print(f"{'='*80}")
        print("STEP 1: DATA LOADING")
        print(f"{'='*80}\n")

        # Fetch market data
        fetcher = MultiAssetFetcher(start_date=self.start_date, end_date=self.end_date)
        tickers = create_example_portfolio(scenario)

        self.prices, self.returns = fetcher.fetch_assets(tickers)
        fetcher.print_summary()

        print(f"\n✓ Data loaded: {len(tickers)} assets, {len(self.returns)} days")

    def generate_factors(self):
        """Generate 133 factors using Factor Mining System."""
        print(f"\n{'='*80}")
        print("STEP 2: FACTOR GENERATION (133 FACTORS)")
        print(f"{'='*80}\n")

        try:
            from factors.factory import FactorFactory

            # Prepare data for factor generation
            data_list = []
            for ticker in self.prices.columns:
                ticker_prices = self.prices[ticker].dropna()

                for date, close in ticker_prices.items():
                    row = {
                        'date': date,
                        'ticker': ticker,
                        'open': close * (1 + np.random.uniform(-0.005, 0.005)),
                        'high': close * (1 + np.random.uniform(0, 0.01)),
                        'low': close * (1 - np.random.uniform(0, 0.01)),
                        'close': close,
                        'volume': 1000000
                    }
                    data_list.append(row)

            market_data = pd.DataFrame(data_list)

            # Initialize factor factory
            factory = FactorFactory(market_data)

            # Generate all factors
            print("Generating factors from:")
            print("  - Technical (49 factors)")
            print("  - Macro (20 factors)")
            print("  - ML (15 factors)")
            print("  - Beta (22 factors)")
            print("  (Skipping fundamental for speed)")

            self.factor_data = factory.generate_all_factors(
                include_technical=True,
                include_fundamental=False,  # Skip for speed
                include_macro=True,
                include_ml=True,
                include_beta=True,
                market_data=market_data
            )

            # Get summary
            summary = factory.get_factor_summary()
            print(f"\n✓ Factor Generation Complete")
            print(f"  Total Factors: {summary['total_factors']}")

            for category, info in summary['by_category'].items():
                print(f"  {category.upper()}: {info['count']} factors")

        except ImportError as e:
            print(f"\n⚠ Could not load Factor Mining System: {e}")
            print(f"  Creating simulated factors for demonstration...")

            # Create simulated factor data
            self.factor_data = self._create_simulated_factors()

    def _create_simulated_factors(self) -> pd.DataFrame:
        """Create simulated factor data for demonstration."""
        print("\nCreating simulated factors...")

        factor_list = []

        for date in self.prices.index:
            for ticker in self.prices.columns:
                row = {'date': date, 'ticker': ticker}

                # Simulate 30 factors
                for i in range(30):
                    row[f'factor_{i+1}'] = np.random.randn()

                factor_list.append(row)

        factor_df = pd.DataFrame(factor_list)
        print(f"✓ Created {len([c for c in factor_df.columns if c.startswith('factor_')])} simulated factors")

        return factor_df

    def run_factor_selection(self, top_n: int = 20):
        """Run comprehensive factor selection analysis."""
        print(f"\n{'='*80}")
        print("STEP 3: FACTOR SELECTION AND RANKING")
        print(f"{'='*80}\n")

        # Create target returns (equal-weight portfolio)
        target_returns = self.returns.mean(axis=1)

        # Initialize selector
        self.selector = FactorSelector(
            factor_data=self.factor_data,
            returns=self.returns,
            forward_periods=[1, 5, 10, 20]
        )

        # Run comprehensive analysis
        importance_report = self.selector.generate_factor_importance_report(
            target_returns=target_returns,
            top_n=top_n
        )

        self.results['importance_report'] = importance_report

        return importance_report

    def analyze_top_factors(self, top_n: int = 10):
        """Deep dive analysis of top selected factors."""
        print(f"\n{'='*80}")
        print(f"STEP 4: DEEP DIVE ON TOP {top_n} FACTORS")
        print(f"{'='*80}\n")

        importance_report = self.results.get('importance_report')
        if importance_report is None:
            print("⚠ Run factor selection first!")
            return

        top_factors = importance_report.head(top_n)['factor'].tolist()

        print(f"Analyzing top {top_n} factors:")
        for i, factor in enumerate(top_factors, 1):
            print(f"  {i}. {factor}")

        # Analyze factor correlations
        factor_cols = [f for f in top_factors if f in self.factor_data.columns]

        if len(factor_cols) > 0:
            print(f"\nFactor Correlation Matrix:")

            # Get factor values (average across assets per day)
            factor_series = {}
            for factor in factor_cols:
                factor_series[factor] = self.factor_data.groupby('date')[factor].mean()

            factor_df = pd.DataFrame(factor_series)
            corr_matrix = factor_df.corr()

            print(corr_matrix.to_string())

            self.results['top_factors'] = top_factors
            self.results['factor_correlations'] = corr_matrix

        return top_factors

    def backtest_factor_strategies(self):
        """Backtest portfolios using selected factors."""
        print(f"\n{'='*80}")
        print("STEP 5: BACKTESTING FACTOR-BASED STRATEGIES")
        print(f"{'='*80}\n")

        if 'top_factors' not in self.results:
            print("⚠ Run factor analysis first!")
            return

        top_factors = self.results['top_factors'][:5]  # Use top 5

        print(f"Creating strategies based on top factors: {', '.join(top_factors)}")

        # Strategy 1: Equal weight baseline
        equal_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        equal_returns = (self.returns * equal_weights).sum(axis=1)

        # Strategy 2: Factor-based selection
        # Select assets with favorable top factor values
        factor_selected_returns = self._factor_selection_strategy(top_factors)

        # Calculate performance
        strategies = {
            'Equal Weight': equal_returns,
            'Factor Selected': factor_selected_returns
        }

        comparison = []

        for name, returns_series in strategies.items():
            total_return = (1 + returns_series).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1
            volatility = returns_series.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0

            # Max drawdown
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            comparison.append({
                'Strategy': name,
                'Total Return': f'{total_return:.2%}',
                'Annual Return': f'{annual_return:.2%}',
                'Volatility': f'{volatility:.2%}',
                'Sharpe Ratio': f'{sharpe:.4f}',
                'Max Drawdown': f'{max_drawdown:.2%}'
            })

        comparison_df = pd.DataFrame(comparison)
        print(f"\n{'='*80}")
        print("STRATEGY PERFORMANCE COMPARISON")
        print(f"{'='*80}\n")
        print(comparison_df.to_string(index=False))

        self.results['backtest_comparison'] = comparison_df
        self.results['strategy_returns'] = strategies

        return comparison_df

    def _factor_selection_strategy(self, top_factors: List[str]) -> pd.Series:
        """
        Create strategy that selects assets based on top factor values.

        Logic: Each day, select top 5 assets with highest combined factor scores.
        """
        portfolio_returns = []

        for date in self.returns.index:
            if date not in self.factor_data['date'].values:
                portfolio_returns.append(0)
                continue

            # Get factor values for this date
            date_data = self.factor_data[self.factor_data['date'] == date].copy()

            if len(date_data) < 5:
                portfolio_returns.append(0)
                continue

            # Calculate composite score (average of top factors)
            date_data['composite_score'] = 0
            for factor in top_factors:
                if factor in date_data.columns:
                    # Normalize factor values
                    factor_values = date_data[factor].values
                    factor_mean = np.mean(factor_values)
                    factor_std = np.std(factor_values)

                    if factor_std > 0:
                        date_data['composite_score'] += (factor_values - factor_mean) / factor_std

            # Select top 5 assets
            top_assets = date_data.nlargest(5, 'composite_score')['ticker'].values

            # Equal weight among selected
            if len(top_assets) > 0:
                asset_returns = self.returns.loc[date, top_assets].values
                portfolio_return = np.mean(asset_returns)
                portfolio_returns.append(portfolio_return)
            else:
                portfolio_returns.append(0)

        return pd.Series(portfolio_returns, index=self.returns.index)

    def visualize_results(self):
        """Create comprehensive visualizations."""
        print(f"\n{'='*80}")
        print("STEP 6: VISUALIZATION")
        print(f"{'='*80}\n")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Top Factors by Different Methods
        if 'importance_report' in self.results:
            ax1 = fig.add_subplot(gs[0, :2])
            importance_report = self.results['importance_report']

            top_20 = importance_report.head(20)
            factors = top_20['factor'].values
            avg_ranks = top_20['avg_rank'].values

            colors = plt.cm.viridis(np.linspace(0, 1, len(factors)))
            ax1.barh(factors, avg_ranks, color=colors)
            ax1.set_xlabel('Average Rank (Lower is Better)')
            ax1.set_title('Top 20 Factors by Average Rank', fontsize=14, fontweight='bold')
            ax1.invert_xaxis()
            ax1.grid(alpha=0.3, axis='x')

        # 2. Selection Count
        if 'importance_report' in self.results:
            ax2 = fig.add_subplot(gs[0, 2])
            importance_report = self.results['importance_report']

            top_20 = importance_report.head(20)
            factors = top_20['factor'].values
            selection_counts = top_20['selection_count'].values

            ax2.barh(factors, selection_counts, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Number of Methods Selected')
            ax2.set_title('Factor Selection Frequency', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3, axis='x')

        # 3. IC Scores
        if self.selector and self.selector.ic_results is not None:
            ax3 = fig.add_subplot(gs[1, 0])
            ic_results = self.selector.ic_results

            # Sort by absolute IC value
            ic_results['abs_ic'] = ic_results['avg_IC'].abs()
            top_ic = ic_results.nlargest(15, 'abs_ic')
            factors = top_ic['factor'].values
            ic_values = top_ic['avg_IC'].values

            colors = ['green' if x > 0 else 'red' for x in ic_values]
            ax3.barh(factors, ic_values, color=colors, alpha=0.7)
            ax3.set_xlabel('Average IC')
            ax3.set_title('Top Factors by IC', fontsize=12, fontweight='bold')
            ax3.axvline(0, color='black', linewidth=0.8)
            ax3.grid(alpha=0.3, axis='x')

        # 4. Factor Returns Sharpe
        if 'returns' in self.selector.importance_scores:
            ax4 = fig.add_subplot(gs[1, 1])
            returns_df = self.selector.importance_scores['returns']

            top_sharpe = returns_df.head(15)
            factors = top_sharpe['factor'].values
            sharpe_values = top_sharpe['sharpe'].values

            colors = ['green' if x > 0 else 'red' for x in sharpe_values]
            ax4.barh(factors, sharpe_values, color=colors, alpha=0.7)
            ax4.set_xlabel('Sharpe Ratio')
            ax4.set_title('Top Factors by Sharpe', fontsize=12, fontweight='bold')
            ax4.axvline(0, color='black', linewidth=0.8)
            ax4.grid(alpha=0.3, axis='x')

        # 5. Random Forest Importance
        if 'random_forest' in self.selector.importance_scores:
            ax5 = fig.add_subplot(gs[1, 2])
            rf_df = self.selector.importance_scores['random_forest']

            top_rf = rf_df.head(15)
            factors = top_rf['factor'].values
            importance = top_rf['importance'].values

            ax5.barh(factors, importance, color='purple', alpha=0.7)
            ax5.set_xlabel('Importance')
            ax5.set_title('Random Forest Importance', fontsize=12, fontweight='bold')
            ax5.grid(alpha=0.3, axis='x')

        # 6. Factor Correlation Heatmap
        if 'factor_correlations' in self.results:
            ax6 = fig.add_subplot(gs[2, :])
            corr_matrix = self.results['factor_correlations']

            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax6, square=True, cbar_kws={'shrink': 0.5})
            ax6.set_title('Top Factors Correlation Matrix', fontsize=14, fontweight='bold')

        # 7. Strategy Performance
        if 'strategy_returns' in self.results:
            ax7 = fig.add_subplot(gs[3, :2])
            strategies = self.results['strategy_returns']

            for name, returns_series in strategies.items():
                cumulative = (1 + returns_series).cumprod()
                ax7.plot(cumulative.index, cumulative.values, label=name, linewidth=2)

            ax7.set_title('Cumulative Returns: Factor-Based vs Baseline', fontsize=14, fontweight='bold')
            ax7.set_ylabel('Cumulative Return')
            ax7.legend(loc='best')
            ax7.grid(alpha=0.3)

        # 8. Performance Metrics
        if 'backtest_comparison' in self.results:
            ax8 = fig.add_subplot(gs[3, 2])
            comparison_df = self.results['backtest_comparison']

            strategies = comparison_df['Strategy'].values
            sharpe_ratios = [float(x) for x in comparison_df['Sharpe Ratio'].values]

            colors = ['green' if x > 0.5 else 'orange' for x in sharpe_ratios]
            ax8.barh(strategies, sharpe_ratios, color=colors, alpha=0.7)
            ax8.set_xlabel('Sharpe Ratio')
            ax8.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
            ax8.grid(alpha=0.3, axis='x')

        plt.savefig('results/factor_analysis/factor_selection_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to results/factor_analysis/factor_selection_analysis.png")

        return fig

    def generate_report(self):
        """Generate comprehensive factor selection report."""
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*18 + "FACTOR SELECTION ANALYSIS COMPLETE" + " "*25 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        # Save importance report
        if 'importance_report' in self.results:
            importance_report = self.results['importance_report']
            importance_report.to_csv('results/factor_analysis/factor_importance_report.csv', index=False)
            print(f"✓ Factor importance report saved to results/factor_analysis/factor_importance_report.csv")

        # Save backtest comparison
        if 'backtest_comparison' in self.results:
            comparison_df = self.results['backtest_comparison']
            comparison_df.to_csv('results/factor_analysis/factor_strategy_comparison.csv', index=False)
            print(f"✓ Strategy comparison saved to results/factor_analysis/factor_strategy_comparison.csv")

        print(f"\nAll results saved to results/factor_analysis/")


def main():
    """Main execution function."""
    demo = FactorSelectionDemo(
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    # Run demonstration
    demo.load_data(scenario='balanced')
    demo.generate_factors()
    demo.run_factor_selection(top_n=20)
    demo.analyze_top_factors(top_n=10)
    demo.backtest_factor_strategies()
    demo.visualize_results()
    demo.generate_report()

    print(f"\n{'='*80}")
    print("FACTOR SELECTION DEMONSTRATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
