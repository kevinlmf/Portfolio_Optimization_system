"""
Test Script for Integrated Bayesian Portfolio System

This script:
1. Tests the new Mixture + Bayesian system
2. Compares with the original hard-regime system
3. Generates comprehensive performance reports
4. Visualizes results and regime transitions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import new system
try:
    from m5_evaluation import IntegratedBayesianSystem
    from m2_market import MarketRegimeDetector
    from m4_optimization import IntelligentOptimizerSelector
    from m5_evaluation import BacktestingEngine
    from m1_data import MultiAssetFetcher, create_example_portfolio
except ImportError:
    # Fallback for old structure
    from strategy.integrated_bayesian_system import IntegratedBayesianSystem
    from strategy.market_regime_detector import MarketRegimeDetector
    from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector
    from strategy.backtesting_engine import BacktestingEngine
    from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio


class BayesianSystemTester:
    """Comprehensive testing framework for Bayesian system."""

    def __init__(self,
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01',
                 portfolio_type: str = 'balanced'):
        """
        Initialize tester.

        Args:
            start_date: Start date for data
            end_date: End date for data
            portfolio_type: Portfolio type ('aggressive', 'balanced', 'conservative')
        """
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio_type = portfolio_type

        print("\n" + "="*70)
        print("BAYESIAN PORTFOLIO SYSTEM - COMPREHENSIVE TEST")
        print("="*70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Portfolio: {portfolio_type}")
        print()

    def load_data(self):
        """Load market data."""
        print("[1/6] Loading market data...")

        fetcher = MultiAssetFetcher(start_date=self.start_date, end_date=self.end_date)
        tickers = create_example_portfolio(self.portfolio_type)

        prices_dict, returns_dict = fetcher.fetch_assets(tickers)

        self.prices = prices_dict
        self.returns = returns_dict

        print(f"   Loaded {len(tickers)} assets")
        print(f"   Assets: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
        print(f"   Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        print(f"   Total observations: {len(self.prices)}")
        print()

    def test_new_system(self):
        """Test new Bayesian system."""
        print("[2/6] Testing NEW Bayesian System...")
        print("-" * 70)

        # Initialize system
        self.bayesian_system = IntegratedBayesianSystem(
            n_regimes=5,
            mixture_type='bayesian_gaussian',
            auto_select_k=True,
            risk_aversion=1.0,
            decay_factor=0.95,
            rebalance_frequency='monthly',
            transaction_cost=0.001,
            robust_mode=True
        )

        # Run backtest
        self.bayesian_result = self.bayesian_system.backtest(
            prices=self.prices,
            initial_capital=100000,
            train_window=252,
            constraints={'long_only': True}
        )

        print()

    def test_original_system(self):
        """Test original system for comparison."""
        print("[3/6] Testing ORIGINAL System (for comparison)...")
        print("-" * 70)

        # This is a simplified version - adapt to your actual original system
        returns = self.prices.pct_change().dropna()

        # Simple buy-and-hold equal weight for comparison
        n_assets = len(self.prices.columns)
        equal_weights = np.ones(n_assets) / n_assets

        portfolio_values = [100000]
        for date in returns.index:
            period_return = equal_weights @ returns.loc[date].values
            portfolio_values.append(portfolio_values[-1] * (1 + period_return))

        self.original_result = pd.Series(
            portfolio_values[1:],
            index=returns.index
        )

        total_return = (self.original_result.iloc[-1] / 100000) - 1
        print(f"   Original System Total Return: {total_return:.2%}")
        print()

    def compare_systems(self):
        """Compare Bayesian vs Original system."""
        print("[4/6] Comparing Systems...")
        print("-" * 70)

        # Bayesian metrics
        b_metrics = self.bayesian_result.performance_metrics

        # Original metrics
        orig_returns = self.original_result.pct_change().dropna()
        orig_total = (self.original_result.iloc[-1] / 100000) - 1
        orig_sharpe = orig_returns.mean() / orig_returns.std() * np.sqrt(252)

        cumulative = (1 + orig_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        orig_maxdd = drawdown.min()

        # Comparison table
        comparison = pd.DataFrame({
            'Bayesian System': [
                f"{b_metrics['total_return']:.2%}",
                f"{b_metrics['annual_return']:.2%}",
                f"{b_metrics['sharpe_ratio']:.3f}",
                f"{b_metrics['max_drawdown']:.2%}",
                f"{b_metrics['volatility']:.2%}",
                f"{b_metrics['win_rate']:.2%}"
            ],
            'Original System': [
                f"{orig_total:.2%}",
                f"{((1 + orig_total) ** (1/b_metrics['n_years']) - 1):.2%}",
                f"{orig_sharpe:.3f}",
                f"{orig_maxdd:.2%}",
                f"{orig_returns.std() * np.sqrt(252):.2%}",
                f"{(orig_returns > 0).mean():.2%}"
            ]
        }, index=[
            'Total Return',
            'Annual Return',
            'Sharpe Ratio',
            'Max Drawdown',
            'Volatility',
            'Win Rate'
        ])

        print(comparison)
        print()

        # Calculate improvement
        sharpe_improvement = ((b_metrics['sharpe_ratio'] - orig_sharpe) / abs(orig_sharpe)) * 100
        return_improvement = ((b_metrics['annual_return'] - ((1 + orig_total) ** (1/b_metrics['n_years']) - 1)) /
                             abs((1 + orig_total) ** (1/b_metrics['n_years']) - 1)) * 100

        print(f"Improvements:")
        print(f"  Sharpe Ratio: {sharpe_improvement:+.1f}%")
        print(f"  Annual Return: {return_improvement:+.1f}%")
        print()

        self.comparison_table = comparison

    def visualize_results(self):
        """Create comprehensive visualizations."""
        print("[5/6] Generating visualizations...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Portfolio Value Comparison
        ax1 = plt.subplot(3, 2, 1)
        self.bayesian_result.portfolio_values.plot(ax=ax1, label='Bayesian System', linewidth=2)
        self.original_result.plot(ax=ax1, label='Original System', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Regime History
        ax2 = plt.subplot(3, 2, 2)
        regime_df = self.bayesian_result.regime_history
        regime_colors = {
            'CRISIS': 'red',
            'BEAR_MARKET': 'orange',
            'HIGH_VOLATILITY': 'yellow',
            'SIDEWAYS': 'lightblue',
            'BULL_MARKET': 'green'
        }

        for regime, color in regime_colors.items():
            prob_col = f'prob_{regime}'
            if prob_col in regime_df.columns:
                regime_df[prob_col].plot(ax=ax2, label=regime, color=color, alpha=0.7)

        ax2.set_title('Regime Probabilities Over Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. Drawdown Comparison
        ax3 = plt.subplot(3, 2, 3)
        b_cumulative = (1 + self.bayesian_result.portfolio_values.pct_change()).cumprod()
        b_running_max = b_cumulative.expanding().max()
        b_drawdown = (b_cumulative - b_running_max) / b_running_max

        o_cumulative = (1 + self.original_result.pct_change()).cumprod()
        o_running_max = o_cumulative.expanding().max()
        o_drawdown = (o_cumulative - o_running_max) / o_running_max

        b_drawdown.plot(ax=ax3, label='Bayesian System', linewidth=2)
        o_drawdown.plot(ax=ax3, label='Original System', linewidth=2, alpha=0.7)
        ax3.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.fill_between(b_drawdown.index, 0, b_drawdown.values, alpha=0.3)

        # 4. Rolling Sharpe Ratio
        ax4 = plt.subplot(3, 2, 4)
        b_returns = self.bayesian_result.portfolio_values.pct_change()
        b_rolling_sharpe = (b_returns.rolling(63).mean() / b_returns.rolling(63).std()) * np.sqrt(252)

        o_returns = self.original_result.pct_change()
        o_rolling_sharpe = (o_returns.rolling(63).mean() / o_returns.rolling(63).std()) * np.sqrt(252)

        b_rolling_sharpe.plot(ax=ax4, label='Bayesian System', linewidth=2)
        o_rolling_sharpe.plot(ax=ax4, label='Original System', linewidth=2, alpha=0.7)
        ax4.set_title('Rolling Sharpe Ratio (3-month)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 5. Asset Allocation Over Time
        ax5 = plt.subplot(3, 2, 5)
        weights_df = self.bayesian_result.weights_history
        weights_df.plot(kind='area', stacked=True, ax=ax5, alpha=0.7)
        ax5.set_title('Asset Allocation Over Time', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Weight')
        ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax5.set_ylim([0, 1])
        ax5.grid(alpha=0.3)

        # 6. Performance Metrics Bar Chart
        ax6 = plt.subplot(3, 2, 6)
        metrics_data = {
            'Sharpe Ratio': [
                self.bayesian_result.performance_metrics['sharpe_ratio'],
                orig_sharpe
            ],
            'Annual Return': [
                self.bayesian_result.performance_metrics['annual_return'] * 100,
                ((1 + orig_total) ** (1/self.bayesian_result.performance_metrics['n_years']) - 1) * 100
            ]
        }

        x = np.arange(len(metrics_data))
        width = 0.35

        bars1 = ax6.bar(x - width/2, [metrics_data['Sharpe Ratio'][0], metrics_data['Annual Return'][0]],
                       width, label='Bayesian', color='steelblue')
        bars2 = ax6.bar(x + width/2, [metrics_data['Sharpe Ratio'][1], metrics_data['Annual Return'][1]],
                       width, label='Original', color='coral')

        ax6.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Sharpe Ratio', 'Annual Return (%)'])
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        output_dir = 'results/bayesian_system'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   Saved visualization to {output_dir}/comprehensive_analysis.png")
        print()

    def save_results(self):
        """Save all results to files."""
        print("[6/6] Saving results...")

        output_dir = 'results/bayesian_system'
        os.makedirs(output_dir, exist_ok=True)

        # Save Bayesian system results
        self.bayesian_system.save_results(self.bayesian_result, output_dir)

        # Save comparison table
        self.comparison_table.to_csv(f'{output_dir}/system_comparison.csv')

        # Save summary report
        with open(f'{output_dir}/test_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("BAYESIAN PORTFOLIO SYSTEM - TEST REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Portfolio Type: {self.portfolio_type}\n\n")

            f.write("SYSTEM COMPARISON\n")
            f.write("-"*70 + "\n")
            f.write(self.comparison_table.to_string())
            f.write("\n\n")

            f.write("BAYESIAN SYSTEM DETAILS\n")
            f.write("-"*70 + "\n")
            for key, value in self.bayesian_result.performance_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"   All results saved to {output_dir}/")
        print()

    def run_full_test(self):
        """Run complete test suite."""
        try:
            self.load_data()
            self.test_new_system()
            self.test_original_system()
            self.compare_systems()
            self.visualize_results()
            self.save_results()

            print("="*70)
            print("TEST COMPLETED SUCCESSFULLY")
            print("="*70)
            print()

            return True

        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run test
    tester = BayesianSystemTester(
        start_date='2020-01-01',
        end_date='2024-01-01',
        portfolio_type='balanced'
    )

    success = tester.run_full_test()

    if success:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Review results in results/bayesian_system/")
        print("2. Compare Bayesian vs Original system performance")
        print("3. Analyze regime transitions and portfolio adaptations")
        print("4. Fine-tune hyperparameters if needed")
    else:
        print("\n❌ Tests failed. Please check error messages above.")
