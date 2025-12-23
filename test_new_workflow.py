#!/usr/bin/env python3
"""
Test script for the new workflow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Import workflow components directly
from workflow import (
    FactorMiner,
    StockSelector,
    EnsembleForecaster,
    ObjectiveType,
    ConstraintBuilder,
    DecisionSpecs,
    QPOptimizer,
    KnowledgeBase
)

print("="*80)
print("TESTING NEW WORKFLOW COMPONENTS")
print("="*80)

# Generate sample data
np.random.seed(42)
n_samples = 300
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'XOM',
          'BAC', 'GS', 'UNH', 'HD', 'V', 'MA', 'DIS', 'KO', 'PEP', 'WMT']

# Generate returns with some structure
returns_data = np.random.randn(n_samples, len(assets)) * 0.015 + 0.0005

# Add some correlation structure
market_factor = np.random.randn(n_samples) * 0.01
for i in range(len(assets)):
    beta = 0.5 + np.random.rand() * 1.0
    returns_data[:, i] += beta * market_factor

dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
returns = pd.DataFrame(returns_data, index=dates, columns=assets)

print(f"\nData: {n_samples} days, {len(assets)} assets")

# =========================================================================
# STEP 1: Factor Mining
# =========================================================================
print("\n" + "-"*60)
print("STEP 1: FACTOR MINING")
print("-"*60)

miner = FactorMiner(method='pca')
factors = miner.mine_factors(returns, n_factors=5)
print(f"✓ Extracted {len(factors.columns)} factors")
print(f"  Factor returns shape: {factors.shape}")

# =========================================================================
# STEP 2: Stock Selection
# =========================================================================
print("\n" + "-"*60)
print("STEP 2: STOCK SELECTION")
print("-"*60)

selector = StockSelector(n_stocks=10, sector_cap=0.5)
selection_result = selector.select(
    returns=returns,
    factors=factors,
    regime=0  # Bull market
)

print(f"✓ Selected {len(selection_result.selected_stocks)} stocks")
print(f"  Selected: {selection_result.selected_stocks}")
print(f"\n  Top 5 by score:")
top_5 = selection_result.scores[selection_result.selected_stocks].head(5)
for stock, score in top_5.items():
    print(f"    {stock}: {score:.3f}")

# Get selected returns
selected_returns = returns[selection_result.selected_stocks]

# =========================================================================
# STEP 3: Forecasting
# =========================================================================
print("\n" + "-"*60)
print("STEP 3: FORECASTING")
print("-"*60)

forecaster = EnsembleForecaster(
    return_methods=['factor', 'momentum'],
    cov_methods=['factor', 'shrinkage']
)

forecaster.fit(selected_returns, factors)
forecast_result = forecaster.predict(horizon=21)

print(f"✓ Forecast generated")
print(f"  Assets: {len(forecast_result.asset_names)}")
print(f"  Confidence: {forecast_result.confidence:.1%}")
print(f"\n  Expected returns (21-day):")
for i, name in enumerate(forecast_result.asset_names[:5]):
    print(f"    {name}: {forecast_result.mu[i]*100:.2f}%")

# =========================================================================
# STEP 4: Select Objective
# =========================================================================
print("\n" + "-"*60)
print("STEP 4: SELECT OBJECTIVE")
print("-"*60)

constraint_builder = ConstraintBuilder()
constraint_builder.long_only(True)
constraint_builder.leverage(1.0)
constraint_builder.max_weight(0.20)

decisions = DecisionSpecs(
    objective=ObjectiveType.SHARPE,
    constraints=constraint_builder.build(),
    method='qp'
)

print(f"✓ Objective: {decisions.objective.value}")
print(f"  Constraints: long_only, max_weight=20%")

# =========================================================================
# STEP 5: Optimization
# =========================================================================
print("\n" + "-"*60)
print("STEP 5: OPTIMIZATION")
print("-"*60)

# Create KnowledgeBase from forecast
knowledge = KnowledgeBase(
    mu=forecast_result.mu,
    Sigma=forecast_result.Sigma,
    asset_names=forecast_result.asset_names
)

optimizer = QPOptimizer()
weights = optimizer.optimize(knowledge, decisions)

print(f"✓ Portfolio optimized")
print(f"\n  Optimal weights:")
for i, name in enumerate(forecast_result.asset_names):
    if weights[i] > 0.01:
        print(f"    {name}: {weights[i]:.2%}")

# Calculate portfolio metrics
portfolio_return = np.dot(weights, forecast_result.mu)
portfolio_risk = np.sqrt(np.dot(weights, np.dot(forecast_result.Sigma, weights)))
sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

# Annualize
ann_factor = 252 / 21
print(f"\n  Portfolio Metrics (annualized):")
print(f"    Expected Return: {portfolio_return * ann_factor:.2%}")
print(f"    Volatility: {portfolio_risk * np.sqrt(ann_factor):.2%}")
print(f"    Sharpe Ratio: {sharpe * np.sqrt(ann_factor):.2f}")

# =========================================================================
# BACKTEST
# =========================================================================
print("\n" + "-"*60)
print("BACKTEST RESULTS")
print("-"*60)

# Calculate historical portfolio returns
portfolio_returns = (selected_returns * weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns).cumprod()
total_return = cumulative_returns.iloc[-1] - 1
ann_return = (cumulative_returns.iloc[-1] ** (252 / len(returns)) - 1)
ann_vol = portfolio_returns.std() * np.sqrt(252)
hist_sharpe = ann_return / ann_vol if ann_vol > 0 else 0

print(f"  Total Return: {total_return*100:.2f}%")
print(f"  Annualized Return: {ann_return*100:.2f}%")
print(f"  Annualized Volatility: {ann_vol*100:.2f}%")
print(f"  Sharpe Ratio: {hist_sharpe:.2f}")

print("\n" + "="*80)
print("NEW WORKFLOW TEST COMPLETE!")
print("="*80)

