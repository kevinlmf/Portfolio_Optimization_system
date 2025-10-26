"""
Historical Period Analysis

This script demonstrates the portfolio optimization system across different
historical periods to show how it adapts to various market conditions.

Periods analyzed:
- 2015-2020: Pre-COVID bull market
- 2020-2025: COVID crash and recovery
- 2010-2015: Post-financial crisis recovery
- 2008-2010: Financial crisis period

Each period has different market characteristics, allowing us to see
how the regime-adaptive system performs in various conditions.

Usage:
    python scripts/historical_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.multi_asset_fetcher import MultiAssetFetcher
from strategy.market_regime_detector import MarketRegimeDetector
from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector
from strategy.backtesting_engine import BacktestingEngine, MonteCarloSimulator

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 14)


class HistoricalAnalysis:
    """
    Analyze portfolio optimization system across multiple historical periods.
    """

    def __init__(self):
        """Initialize historical analysis."""
        self.periods = [
            {
                'name': '2008-2010 Financial Crisis',
                'start': '2008-01-01',
                'end': '2010-12-31',
                'description': 'The 2008 financial crisis and recovery'
            },
            {
                'name': '2010-2015 Post-Crisis Recovery',
                'start': '2010-01-01',
                'end': '2015-12-31',
                'description': 'Steady recovery after financial crisis'
            },
            {
                'name': '2015-2020 Pre-COVID Bull Market',
                'start': '2015-01-01',
                'end': '2020-12-31',
                'description': 'Long bull market ending with COVID crash'
            },
            {
                'name': '2020-2025 COVID and Recovery',
                'start': '2020-01-01',
                'end': '2024-12-31',  # Up to present
                'description': 'COVID crash, recovery, and inflation surge'
            }
        ]

        self.results = {}
        self.regime_detector = MarketRegimeDetector()
        self.optimizer_selector = IntelligentOptimizerSelector()

    def define_portfolio(self, period_name: str):
        """
        Define portfolio based on period.
        Use more liquid assets for older periods where data may be sparse.
        """
        if '2008' in period_name:
            # Financial crisis: Focus on well-established assets
            return [
                'AAPL', 'MSFT', 'JPM', 'GE', 'XOM',  # Blue chips
                'SPY', 'QQQ',  # Equity ETFs
                'TLT', 'LQD',  # Bonds
                'GLD',  # Gold
            ]
        elif '2010' in period_name:
            # Post-crisis: Add more diversity
            return [
                'AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'GE',
                'SPY', 'QQQ', 'IWM',
                'TLT', 'IEF', 'LQD',
                'GLD', 'SLV',
            ]
        elif '2015' in period_name:
            # Pre-COVID: Full diversification
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'WMT', 'XOM',
                'SPY', 'QQQ', 'IWM', 'VEA',
                'TLT', 'IEF', 'LQD', 'HYG',
                'GLD', 'SLV', 'DBC',
            ]
        else:  # 2020+
            # Modern era: Include crypto
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'JNJ',
                'SPY', 'QQQ', 'VTI', 'VEA', 'VWO',
                'TLT', 'IEF', 'LQD', 'HYG',
                'GLD', 'SLV', 'DBC',
                'BTC-USD', 'ETH-USD',
                'VNQ'
            ]

    def analyze_period(self, period: dict):
        """
        Analyze a single historical period.

        Args:
            period: Dictionary with period information

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING: {period['name']}")
        print(f"{'='*80}")
        print(f"Period: {period['start']} to {period['end']}")
        print(f"Description: {period['description']}")

        # Define portfolio
        tickers = self.define_portfolio(period['name'])
        print(f"\nAssets: {len(tickers)} tickers")

        # Fetch data
        print("\nFetching data...")
        try:
            fetcher = MultiAssetFetcher(
                start_date=period['start'],
                end_date=period['end']
            )
            prices, returns = fetcher.fetch_assets(tickers)
            asset_info = fetcher.asset_info  # Store asset classification

            # Filter out assets with too much missing data
            min_observations = len(prices) * 0.8  # Need 80% of data
            valid_columns = []
            for col in prices.columns:
                if prices[col].notna().sum() >= min_observations:
                    valid_columns.append(col)

            if len(valid_columns) < len(prices.columns):
                print(f"   Removed {len(prices.columns) - len(valid_columns)} assets with insufficient data")
                prices = prices[valid_columns]
                returns = returns[valid_columns]

            if len(prices.columns) < 3:
                print(f"   ✗ Insufficient assets ({len(prices.columns)}), skipping period")
                return None

        except Exception as e:
            print(f"   ✗ Error fetching data: {e}")
            return None

        # Detect market regime
        print("\nDetecting market regime...")
        market_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]
        regime_result = self.regime_detector.detect_regime(market_prices)

        print(f"   Detected regime: {regime_result['regime'].upper()}")
        print(f"   Confidence: {regime_result['confidence']:.1%}")

        # Select optimization method
        print("\nSelecting optimization method...")
        recommendation = self.optimizer_selector.select_optimizer(
            prices=market_prices,
            returns=returns,
            preferences={'prefer_interpretable': True, 'prefer_robust': True}
        )

        print(f"   Recommended: {recommendation.recommended_method.upper()}")
        print(f"   Confidence: {recommendation.confidence:.1%}")

        # Run backtest with adaptive optimization
        print("\nRunning backtest...")

        def adaptive_optimization(returns_df, regime_window=60):
            """Adaptive optimization based on regime."""
            recent_prices = (1 + returns_df).cumprod().iloc[:, 0]
            if len(recent_prices) >= regime_window:
                regime_res = self.regime_detector.detect_regime(recent_prices)
                regime = regime_res['regime']
            else:
                regime = 'sideways'

            # Map regime to method
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

        try:
            backtest_engine = BacktestingEngine(
                returns=returns,
                prices=prices,
                transaction_cost=0.001,
                rebalance_frequency='monthly',
                risk_free_rate=0.03
            )

            backtest_result = backtest_engine.run_backtest(
                optimization_func=adaptive_optimization,
                lookback_window=min(252, len(returns) // 3),
                min_history=min(60, len(returns) // 5)
            )

            metrics = backtest_result['metrics']

            print(f"\n   Results:")
            print(f"   Total Return:     {metrics['total_return']:>8.2%}")
            print(f"   Annual Return:    {metrics['annual_return']:>8.2%}")
            print(f"   Sharpe Ratio:     {metrics['sharpe_ratio']:>8.4f}")
            print(f"   Max Drawdown:     {metrics['max_drawdown']:>8.2%}")
            print(f"   Win Rate:         {metrics['win_rate']:>8.2%}")

            # Compare with SPY if available
            benchmark_comparison = None
            if 'SPY' in returns.columns:
                benchmark_comparison = backtest_engine.compare_with_benchmark(returns['SPY'])
                print(f"\n   vs SPY Benchmark:")
                print(f"   Alpha:            {benchmark_comparison['alpha']:>8.2%}")
                print(f"   Beta:             {benchmark_comparison['beta']:>8.4f}")
                print(f"   Info Ratio:       {benchmark_comparison['information_ratio']:>8.4f}")

            # Monte Carlo validation (smaller for speed)
            print("\n   Running Monte Carlo validation...")
            last_weights = backtest_result['weights_history'][-1]['weights'] if backtest_result['weights_history'] else None
            if last_weights is not None:
                mc_simulator = MonteCarloSimulator(
                    returns=returns,
                    n_simulations=1000,  # Reduced for speed
                    n_days=63  # 3 months
                )
                mc_result = mc_simulator.run_simulation(last_weights, method='parametric')
                print(f"   MC Prob of Profit: {mc_result['prob_positive']:.2%}")
            else:
                mc_result = None

            # Get final portfolio weights
            final_weights = None
            weights_df = None
            if backtest_result['weights_history']:
                final_weights_dict = backtest_result['weights_history'][-1]['weights']
                tickers = returns.columns.tolist()

                # Create weights DataFrame
                weights_data = []
                for i, ticker in enumerate(tickers):
                    if i < len(final_weights_dict) and final_weights_dict[i] > 0.001:  # Only show weights > 0.1%
                        weights_data.append({
                            'ticker': ticker,
                            'weight': final_weights_dict[i],
                            'weight_pct': f"{final_weights_dict[i]*100:.2f}%"
                        })

                weights_df = pd.DataFrame(weights_data).sort_values('weight', ascending=False)

                # Print top holdings
                print(f"\n   Top 10 Holdings (Final Portfolio):")
                for idx, row in weights_df.head(10).iterrows():
                    print(f"     {row['ticker']:8s} {row['weight_pct']:>7s}")

            # Store results
            result = {
                'period': period,
                'regime': regime_result,
                'recommendation': recommendation,
                'metrics': metrics,
                'benchmark_comparison': benchmark_comparison,
                'monte_carlo': mc_result,
                'portfolio_values': backtest_result['portfolio_values'],
                'weights_history': backtest_result['weights_history'],
                'final_weights': weights_df,
                'asset_info': asset_info,
                'prices': prices,
                'returns': returns
            }

            return result

        except Exception as e:
            print(f"   ✗ Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _optimize_portfolio(self, returns_df: pd.DataFrame, method: str) -> np.ndarray:
        """Helper function to optimize portfolio."""
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
                if vol < 1e-10:
                    return 1e6
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

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            return result.x if result.success else x0
        except:
            return x0

    def run_all_periods(self):
        """Analyze all historical periods."""
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*20 + "HISTORICAL PERIOD ANALYSIS" + " "*32 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        print(f"Analyzing {len(self.periods)} historical periods...")

        for period in self.periods:
            result = self.analyze_period(period)
            if result:
                self.results[period['name']] = result

        print(f"\n✓ Analysis complete for {len(self.results)} periods")

    def create_comparison_report(self):
        """Create comprehensive comparison across periods."""
        if not self.results:
            print("No results to compare")
            return

        print(f"\n{'='*80}")
        print("CROSS-PERIOD COMPARISON")
        print(f"{'='*80}\n")

        # Create comparison table
        comparison_data = []
        for period_name, result in self.results.items():
            metrics = result['metrics']
            regime = result['regime']['regime']
            method = result['recommendation'].recommended_method

            row = {
                'Period': period_name.replace(' ', '\n', 1),  # Line break for better display
                'Regime': regime.replace('_', ' ').title(),
                'Method': method.replace('_', ' ').title(),
                'Total\nReturn': f"{metrics['total_return']:.1%}",
                'Annual\nReturn': f"{metrics['annual_return']:.1%}",
                'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                'Max\nDrawdown': f"{metrics['max_drawdown']:.1%}",
                'Win\nRate': f"{metrics['win_rate']:.1%}"
            }

            if result['benchmark_comparison']:
                row['Alpha'] = f"{result['benchmark_comparison']['alpha']:.1%}"
                row['Beta'] = f"{result['benchmark_comparison']['beta']:.2f}"

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Save to CSV
        os.makedirs('results/historical_analysis', exist_ok=True)
        comparison_df.to_csv('results/historical_analysis/period_comparison.csv', index=False)
        print(f"\n✓ Comparison saved to results/historical_analysis/period_comparison.csv")

        # Print asset allocation details
        print(f"\n{'='*80}")
        print("ASSET ALLOCATION DETAILS (FINAL PORTFOLIO)")
        print(f"{'='*80}\n")

        for period_name, result in self.results.items():
            if result['final_weights'] is not None and not result['final_weights'].empty:
                print(f"\n{period_name}:")
                print(f"{'─'*50}")

                weights_df = result['final_weights']
                n_holdings = len(weights_df)
                print(f"Number of holdings: {n_holdings}")
                print(f"\nTop Holdings:")

                for idx, row in weights_df.head(10).iterrows():
                    print(f"  {row['ticker']:8s} {row['weight_pct']:>7s}")

                # Asset class breakdown if available
                if 'asset_info' in result and result['asset_info'] is not None:
                    asset_info = result['asset_info']
                    class_allocation = {}

                    for idx, row in weights_df.iterrows():
                        ticker = row['ticker']
                        weight = row['weight']
                        asset_class = asset_info[asset_info['ticker'] == ticker]['asset_class'].iloc[0] if ticker in asset_info['ticker'].values else 'unknown'
                        class_allocation[asset_class] = class_allocation.get(asset_class, 0) + weight

                    print(f"\nAsset Class Breakdown:")
                    for asset_class, weight in sorted(class_allocation.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {asset_class:15s} {weight*100:>6.2f}%")

        # Save detailed allocation
        print(f"\n{'='*80}")
        for period_name, result in self.results.items():
            if result['final_weights'] is not None and not result['final_weights'].empty:
                filename = period_name.replace(' ', '_').replace('-', '_').lower()
                filepath = f'results/historical_analysis/allocation_{filename}.csv'
                result['final_weights'].to_csv(filepath, index=False)

        print(f"\n✓ Detailed allocations saved to results/historical_analysis/allocation_*.csv")

    def visualize_results(self):
        """Create comprehensive visualization."""
        if not self.results:
            print("No results to visualize")
            return

        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")

        n_periods = len(self.results)
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

        colors = plt.cm.Set3(range(n_periods))

        # 1. Portfolio Performance Over Time (spanning top row)
        ax1 = fig.add_subplot(gs[0, :])
        for i, (period_name, result) in enumerate(self.results.items()):
            portfolio_values = result['portfolio_values']
            ax1.plot(portfolio_values.index, portfolio_values.values,
                    linewidth=2.5, color=colors[i],
                    label=period_name.split()[0], alpha=0.8)

        ax1.set_title('Portfolio Performance Across Historical Periods',
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=14)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(alpha=0.3)
        ax1.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 2. Annual Returns Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        periods_short = [list(self.results.keys())[i].split()[0] for i in range(n_periods)]
        annual_returns = [self.results[p]['metrics']['annual_return'] * 100
                         for p in self.results.keys()]
        bars = ax2.bar(periods_short, annual_returns, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Annual Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(0, color='black', linewidth=1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Sharpe Ratios Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        sharpe_ratios = [self.results[p]['metrics']['sharpe_ratio']
                        for p in self.results.keys()]
        bars = ax3.bar(periods_short, sharpe_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(0, color='black', linewidth=1)
        ax3.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
        ax3.legend(fontsize=10)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. Max Drawdowns Comparison
        ax4 = fig.add_subplot(gs[1, 2])
        drawdowns = [self.results[p]['metrics']['max_drawdown'] * 100
                    for p in self.results.keys()]
        bars = ax4.bar(periods_short, drawdowns, color='red', alpha=0.6, edgecolor='black')
        ax4.set_title('Maximum Drawdowns', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)', fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Market Regime Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        regimes = [self.results[p]['regime']['regime'].replace('_', ' ').title()
                  for p in self.results.keys()]
        regime_colors = {'Bull Market': 'green', 'Bear Market': 'red',
                        'Sideways': 'blue', 'High Volatility': 'orange',
                        'Crisis': 'darkred'}
        bar_colors = [regime_colors.get(r, 'gray') for r in regimes]
        ax5.barh(periods_short, [1]*n_periods, color=bar_colors, alpha=0.7, edgecolor='black')
        ax5.set_title('Detected Market Regimes', fontsize=14, fontweight='bold')
        ax5.set_xlim(0, 1.2)
        ax5.set_xticks([])
        for i, regime in enumerate(regimes):
            ax5.text(0.5, i, regime, ha='center', va='center', fontweight='bold')

        # 6. Optimization Methods Used
        ax6 = fig.add_subplot(gs[2, 1])
        methods = [self.results[p]['recommendation'].recommended_method.replace('_', ' ').title()
                  for p in self.results.keys()]
        method_colors = {'Max Sharpe': 'green', 'Min Variance': 'blue',
                        'Risk Parity': 'purple', 'Equal Weight': 'gray'}
        bar_colors = [method_colors.get(m, 'gray') for m in methods]
        ax6.barh(periods_short, [1]*n_periods, color=bar_colors, alpha=0.7, edgecolor='black')
        ax6.set_title('Recommended Methods', fontsize=14, fontweight='bold')
        ax6.set_xlim(0, 1.2)
        ax6.set_xticks([])
        for i, method in enumerate(methods):
            ax6.text(0.5, i, method, ha='center', va='center', fontweight='bold')

        # 7. Alpha vs Benchmark (if available)
        ax7 = fig.add_subplot(gs[2, 2])
        alphas = []
        periods_with_alpha = []
        for period_name, result in self.results.items():
            if result['benchmark_comparison']:
                alphas.append(result['benchmark_comparison']['alpha'] * 100)
                periods_with_alpha.append(period_name.split()[0])

        if alphas:
            colors_alpha = ['green' if a > 0 else 'red' for a in alphas]
            ax7.bar(periods_with_alpha, alphas, color=colors_alpha, alpha=0.7, edgecolor='black')
            ax7.set_title('Alpha vs SPY', fontsize=14, fontweight='bold')
            ax7.set_ylabel('Alpha (%)', fontsize=12)
            ax7.grid(axis='y', alpha=0.3)
            ax7.axhline(0, color='black', linewidth=1)
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 8. Win Rate Comparison
        ax8 = fig.add_subplot(gs[3, 0])
        win_rates = [self.results[p]['metrics']['win_rate'] * 100
                    for p in self.results.keys()]
        ax8.bar(periods_short, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax8.set_title('Win Rates', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Win Rate (%)', fontsize=12)
        ax8.grid(axis='y', alpha=0.3)
        ax8.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 9. Risk-Return Scatter
        ax9 = fig.add_subplot(gs[3, 1])
        vols = [self.results[p]['metrics']['annual_volatility'] * 100
               for p in self.results.keys()]
        rets = [self.results[p]['metrics']['annual_return'] * 100
               for p in self.results.keys()]

        for i, period_name in enumerate(self.results.keys()):
            ax9.scatter(vols[i], rets[i], s=300, color=colors[i],
                       alpha=0.7, edgecolors='black', linewidth=2)
            ax9.annotate(periods_short[i], (vols[i], rets[i]),
                        fontsize=10, ha='center', va='center', fontweight='bold')

        ax9.set_xlabel('Volatility (%)', fontsize=12)
        ax9.set_ylabel('Return (%)', fontsize=12)
        ax9.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax9.grid(alpha=0.3)

        # 10. Summary Statistics Table
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('tight')
        ax10.axis('off')

        # Calculate average metrics
        avg_return = np.mean([self.results[p]['metrics']['annual_return']
                             for p in self.results.keys()])
        avg_sharpe = np.mean([self.results[p]['metrics']['sharpe_ratio']
                             for p in self.results.keys()])
        avg_drawdown = np.mean([self.results[p]['metrics']['max_drawdown']
                               for p in self.results.keys()])

        summary_text = f"""
        SUMMARY STATISTICS

        Periods Analyzed: {n_periods}

        Average Annual Return: {avg_return:.2%}
        Average Sharpe Ratio: {avg_sharpe:.3f}
        Average Max Drawdown: {avg_drawdown:.2%}

        Best Period: {periods_short[np.argmax(annual_returns)]}
        Worst Period: {periods_short[np.argmin(annual_returns)]}
        """

        ax10.text(0.1, 0.5, summary_text, fontsize=12,
                 verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save figure
        plt.savefig('results/historical_analysis/comprehensive_comparison.png',
                   dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to results/historical_analysis/comprehensive_comparison.png")

        return fig


def main():
    """Main execution function."""
    analyzer = HistoricalAnalysis()

    # Run analysis for all periods
    analyzer.run_all_periods()

    # Create comparison report
    analyzer.create_comparison_report()

    # Visualize results
    analyzer.visualize_results()

    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "ANALYSIS COMPLETE" + " "*35 + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}\n")

    print("All results saved to results/historical_analysis/")
    print("\nKey insights:")
    print("1. Compare how the system adapts to different market regimes")
    print("2. Evaluate performance consistency across economic cycles")
    print("3. Assess risk management during crisis periods")
    print("4. Review optimization method selection effectiveness")


if __name__ == '__main__':
    main()
