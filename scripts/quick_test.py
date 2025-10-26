"""
Quick Test Script for Comprehensive Portfolio System

This script runs a simplified version of the system to verify everything works.
Perfect for testing or quick demonstrations.

Usage:
    python scripts/quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("\n" + "="*80)
print("QUICK TEST: Comprehensive Portfolio Optimization System")
print("="*80)

# Test 1: Data Fetcher
print("\n[Test 1/5] Testing Multi-Asset Data Fetcher...")
try:
    from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio

    fetcher = MultiAssetFetcher(
        start_date='2022-01-01',
        end_date='2023-12-31'
    )

    # Use a small balanced portfolio for testing
    test_tickers = ['AAPL', 'MSFT', 'JPM', 'SPY', 'TLT', 'GLD', 'BTC-USD']
    print(f"   Fetching {len(test_tickers)} assets...")

    prices, returns = fetcher.fetch_assets(test_tickers)

    print(f"   ✓ Data fetched: {len(prices)} days, {len(prices.columns)} assets")
    print(f"   ✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Market Regime Detector
print("\n[Test 2/5] Testing Market Regime Detection...")
try:
    from strategy.market_regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()
    market_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]

    regime_result = detector.detect_regime(market_prices)

    print(f"   ✓ Detected regime: {regime_result['regime'].upper()}")
    print(f"   ✓ Confidence: {regime_result['confidence']:.1%}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Intelligent Optimizer Selector
print("\n[Test 3/5] Testing Intelligent Optimizer Selector...")
try:
    from strategy.intelligent_optimizer_selector import IntelligentOptimizerSelector

    selector = IntelligentOptimizerSelector()

    recommendation = selector.select_optimizer(
        prices=market_prices,
        returns=returns,
        preferences={'prefer_interpretable': True}
    )

    print(f"   ✓ Recommended method: {recommendation.recommended_method.upper()}")
    print(f"   ✓ Confidence: {recommendation.confidence:.1%}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Backtesting Engine
print("\n[Test 4/5] Testing Backtesting Engine...")
try:
    from strategy.backtesting_engine import BacktestingEngine

    # Simple optimization function for testing
    def simple_equal_weight(returns_df):
        n_assets = returns_df.shape[1]
        return np.ones(n_assets) / n_assets

    engine = BacktestingEngine(
        returns=returns,
        prices=prices,
        transaction_cost=0.001,
        rebalance_frequency='monthly',
        risk_free_rate=0.03
    )

    result = engine.run_backtest(
        optimization_func=simple_equal_weight,
        lookback_window=126,
        min_history=60
    )

    metrics = result['metrics']
    print(f"   ✓ Backtest completed")
    print(f"   ✓ Total return: {metrics['total_return']:.2%}")
    print(f"   ✓ Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   ✓ Max drawdown: {metrics['max_drawdown']:.2%}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Monte Carlo Simulator
print("\n[Test 5/5] Testing Monte Carlo Simulator...")
try:
    from strategy.backtesting_engine import MonteCarloSimulator

    n_assets = returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets

    simulator = MonteCarloSimulator(
        returns=returns,
        n_simulations=1000,  # Small number for quick test
        n_days=63  # 3 months
    )

    mc_result = simulator.run_simulation(equal_weights, method='parametric')

    print(f"   ✓ Monte Carlo completed")
    print(f"   ✓ Mean final value: ${mc_result['mean_final_value']:.4f}")
    print(f"   ✓ Prob of profit: {mc_result['prob_positive']:.2%}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("\nThe system is ready to use. Run the full system with:")
print("  python scripts/comprehensive_portfolio_system.py")
print("\n" + "="*80 + "\n")
