"""
Demonstration Script: Intelligent Optimizer Selector

This script demonstrates the AI-driven intelligent optimizer selection system
that automatically recommends the best portfolio optimization method based on:
- Current market regime
- Asset configuration characteristics
- Historical performance patterns

Usage:
    python scripts/demo_intelligent_selector.py
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

from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector
from strategy.market_regime_detector import MarketRegimeDetector
from strategy.sparse_sharpe_optimizer import SparseSharpeOptimizer

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


def download_market_data(tickers, start_date, end_date):
    """Download historical data for analysis."""
    print(f"\n{'='*70}")
    print("DOWNLOADING MARKET DATA")
    print(f"{'='*70}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if 'Adj Close' in data.columns.levels[0]:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # Remove tickers with insufficient data
    prices = prices.dropna(axis=1, how='all')
    prices = prices.fillna(method='ffill').fillna(method='bfill')

    print(f"Downloaded data for {len(prices.columns)} assets")
    print(f"{len(prices)} trading days")

    return prices


def demo_regime_detection(market_index_prices):
    """Demonstrate market regime detection."""
    print(f"\n{'='*70}")
    print("STEP 1: MARKET REGIME DETECTION")
    print(f"{'='*70}")

    detector = MarketRegimeDetector(
        lookback_short=20,
        lookback_long=60,
        vol_window=20
    )

    regime_result = detector.detect_regime(market_index_prices)

    print(f"\nDetected Regime: {regime_result['regime'].replace('_', ' ').upper()}")
    print(f"Confidence: {regime_result['confidence']:.1%}")

    print("\nRegime Probabilities:")
    for regime, score in sorted(regime_result['regime_scores'].items(),
                               key=lambda x: x[1], reverse=True):
        bar = '█' * int(score * 50)
        print(f"  {regime:20s} {score:5.1%} {bar}")

    print("\n" + regime_result['recommendation'])

    return detector, regime_result


def demo_intelligent_selection(prices, market_index_prices):
    """Demonstrate intelligent optimizer selection."""
    print(f"\n{'='*70}")
    print("STEP 2: INTELLIGENT OPTIMIZER SELECTION")
    print(f"{'='*70}")

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Create selector
    selector = IntelligentOptimizerSelector()

    # Get recommendation
    recommendation = selector.select_optimizer(
        prices=market_index_prices,
        returns=returns,
        preferences={
            'prefer_sparse': False,
            'prefer_interpretable': True,
            'prefer_robust': True
        }
    )

    # Print detailed explanation
    print(selector.explain_recommendation())

    # Show comparison table
    print("\nMETHOD COMPARISON TABLE:")
    comparison = selector.get_method_comparison()
    print(comparison.to_string(index=False))

    return selector, recommendation


def demo_regime_scenarios():
    """Demonstrate recommendations across different market scenarios."""
    print(f"\n{'='*70}")
    print("STEP 3: RECOMMENDATIONS ACROSS MARKET SCENARIOS")
    print(f"{'='*70}")

    # Create synthetic scenarios
    scenarios = {
        'Bull Market - Low Vol': {
            'trend_slope': 0.05,
            'volatility': 0.12,
            'return_short': 0.08,
            'ma_trend': 1
        },
        'Bear Market - High Vol': {
            'trend_slope': -0.04,
            'volatility': 0.35,
            'return_short': -0.12,
            'ma_trend': -1
        },
        'Sideways - Medium Vol': {
            'trend_slope': 0.00,
            'volatility': 0.18,
            'return_short': 0.01,
            'ma_trend': 0
        },
        'Crisis - Extreme Vol': {
            'trend_slope': -0.08,
            'volatility': 0.50,
            'return_short': -0.20,
            'ma_trend': -1
        }
    }

    results = []

    for scenario_name, indicators in scenarios.items():
        # Simulate scenario scoring
        regime_scores = {}

        if indicators['ma_trend'] > 0 and indicators['volatility'] < 0.20:
            regime_scores = {'bull_market': 0.7, 'sideways': 0.2, 'high_volatility': 0.1}
        elif indicators['ma_trend'] < 0 and indicators['volatility'] > 0.30:
            if indicators['return_short'] < -0.15:
                regime_scores = {'crisis': 0.6, 'bear_market': 0.3, 'high_volatility': 0.1}
            else:
                regime_scores = {'bear_market': 0.6, 'high_volatility': 0.3, 'sideways': 0.1}
        elif abs(indicators['trend_slope']) < 0.01:
            regime_scores = {'sideways': 0.6, 'bull_market': 0.2, 'bear_market': 0.2}
        else:
            regime_scores = {'high_volatility': 0.5, 'sideways': 0.3, 'bear_market': 0.2}

        primary_regime = max(regime_scores, key=regime_scores.get)

        # Determine best method based on regime
        method_map = {
            'bull_market': 'sparse_sharpe',
            'bear_market': 'min_variance',
            'sideways': 'risk_parity',
            'high_volatility': 'min_variance',
            'crisis': 'equal_weight'
        }

        recommended = method_map.get(primary_regime, 'risk_parity')

        results.append({
            'Scenario': scenario_name,
            'Detected Regime': primary_regime.replace('_', ' ').title(),
            'Volatility': f"{indicators['volatility']:.1%}",
            'Return': f"{indicators['return_short']:.1%}",
            'Recommended Method': recommended,
            'Rationale': get_rationale(recommended, primary_regime)
        })

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    return df_results


def get_rationale(method, regime):
    """Get brief rationale for method selection."""
    rationales = {
        ('sparse_sharpe', 'bull_market'): "Concentrate on winners",
        ('min_variance', 'bear_market'): "Minimize losses",
        ('min_variance', 'high_volatility'): "Stabilize returns",
        ('risk_parity', 'sideways'): "Balanced exposure",
        ('equal_weight', 'crisis'): "Avoid estimation error"
    }
    return rationales.get((method, regime), "Robust approach")


def demo_sparse_sharpe_optimization(returns):
    """Demonstrate the mSSRM-PGA sparse Sharpe optimizer."""
    print(f"\n{'='*70}")
    print("STEP 4: SPARSE SHARPE RATIO OPTIMIZATION (mSSRM-PGA)")
    print(f"{'='*70}")

    # Try different sparsity levels
    sparsity_levels = [10, 15, 20]

    results = []
    for m in sparsity_levels:
        print(f"\n{'─'*70}")
        print(f"Optimizing with sparsity m = {m}")
        print(f"{'─'*70}")

        optimizer = SparseSharpeOptimizer(
            epsilon=1e-3,
            max_iter=5000,
            verbose=True
        )

        # Convert returns to numpy array
        returns_array = returns.values

        result = optimizer.optimize(returns_array, m=m)

        results.append({
            'Sparsity (m)': m,
            'Actual Sparsity': result['sparsity'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Converged': 'Yes' if result['converged'] else 'No',
            'Iterations': result['iterations']
        })

    df_results = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("SPARSE OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))

    return df_results


def create_visualization(comparison_df, scenario_df):
    """Create visualization of results."""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Method scores
    ax = axes[0, 0]
    comparison_df_plot = comparison_df.copy()
    comparison_df_plot['Score_num'] = comparison_df_plot['Score'].str.rstrip('%').astype(float) / 100

    comparison_df_plot.plot(
        x='Method', y='Score_num', kind='barh', ax=ax,
        color='steelblue', legend=False
    )
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Optimization Method', fontsize=12)
    ax.set_title('Method Scores for Current Market Conditions', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Regime-Method Matrix
    ax = axes[0, 1]
    regimes = ['Bull\nMarket', 'Bear\nMarket', 'Sideways', 'High\nVolatility', 'Crisis']
    methods = ['Max\nSharpe', 'Min\nVariance', 'Risk\nParity', 'Equal\nWeight', 'Sparse\nSharpe']

    # Suitability matrix (example)
    suitability = np.array([
        [0.9, 0.3, 0.7, 0.2, 0.9],  # Bull
        [0.3, 0.9, 0.7, 0.5, 0.2],  # Bear
        [0.7, 0.5, 0.9, 0.6, 0.7],  # Sideways
        [0.2, 0.9, 0.7, 0.8, 0.3],  # High Vol
        [0.1, 0.8, 0.6, 0.9, 0.2],  # Crisis
    ])

    im = ax.imshow(suitability, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(regimes)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_yticklabels(regimes, fontsize=10)
    ax.set_title('Method Suitability by Market Regime', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(regimes)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{suitability[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='Suitability Score')

    # Plot 3: Scenario recommendations
    ax = axes[1, 0]
    scenario_methods = scenario_df['Recommended Method'].value_counts()
    scenario_methods.plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('Optimization Method', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Recommended Methods Across Scenarios', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Method characteristics radar
    ax = axes[1, 1]
    ax.axis('off')
    info_text = """
    METHOD CHARACTERISTICS SUMMARY

    Max Sharpe:
    • High concentration
    • Best for bull markets
    • Moderate computational cost

    Min Variance:
    • Low concentration
    • Best for bear/crisis markets
    • High robustness

    Sparse Sharpe (mSSRM-PGA):
    • Very high concentration
    • Global optimality guarantee
    • Best for large universes (50+ assets)
    • Bull/sideways markets

    Risk Parity:
    • Equal risk contribution
    • Good for all regimes
    • High interpretability

    Equal Weight:
    • No optimization
    • Crisis/extreme uncertainty
    • Maximum robustness
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
           family='monospace')

    plt.tight_layout()

    # Save figure
    output_path = 'results/intelligent_selector_demo.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    return fig


def main():
    """Main demonstration function."""
    print(f"\n{'#'*70}")
    print("#" + " "*68 + "#")
    print("#" + " "*10 + "INTELLIGENT OPTIMIZER SELECTOR DEMONSTRATION" + " "*13 + "#")
    print("#" + " "*68 + "#")
    print(f"{'#'*70}\n")

    # Configuration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data

    # Asset universe (example: diversified portfolio)
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        # Finance
        'JPM', 'BAC', 'GS', 'MS',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV',
        # Consumer
        'WMT', 'HD', 'NKE', 'MCD',
        # Energy
        'XOM', 'CVX',
        # ETFs for diversity
        'SPY', 'QQQ', 'TLT', 'GLD'
    ]

    market_index = 'SPY'

    try:
        # Download data
        prices = download_market_data(
            tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Get market index prices
        if market_index in prices.columns:
            market_index_prices = prices[market_index]
        else:
            market_index_prices = prices.iloc[:, 0]  # Use first column as fallback

        # Run demonstrations
        detector, regime_result = demo_regime_detection(market_index_prices)

        selector, recommendation = demo_intelligent_selection(
            prices,
            market_index_prices
        )

        scenario_results = demo_regime_scenarios()

        # Demonstrate sparse optimization if recommended
        returns = prices.pct_change().dropna()
        if recommendation.recommended_method == 'sparse_sharpe':
            sparse_results = demo_sparse_sharpe_optimization(returns)

        # Get comparison for visualization
        comparison_df = selector.get_method_comparison()

        # Create visualizations
        fig = create_visualization(comparison_df, scenario_results)

        print(f"\n{'#'*70}")
        print("#" + " "*68 + "#")
        print("#" + " "*18 + "DEMONSTRATION COMPLETED" + " "*27 + "#")
        print("#" + " "*68 + "#")
        print(f"{'#'*70}\n")

        print("\nKEY TAKEAWAYS:")
        print("1. The intelligent selector adapts to market conditions")
        print("2. Different methods excel in different regimes")
        print("3. Sparse Sharpe (mSSRM-PGA) is ideal for large bull market universes")
        print("4. Conservative methods (Min Variance, Equal Weight) for crises")
        print("5. System provides transparent reasoning for all recommendations")

        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
