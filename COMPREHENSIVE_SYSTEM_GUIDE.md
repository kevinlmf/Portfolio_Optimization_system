# Comprehensive Multi-Asset Portfolio Optimization System

##  Overview

This is an advanced portfolio optimization system that automatically:
1. **Fetches multi-asset class data** (stocks, bonds, commodities, crypto, ETFs)
2. **Detects market regimes** (bull, bear, sideways, high volatility, crisis)
3. **Selects optimal optimization method** based on current conditions
4. **Runs comprehensive backtests** with rolling window optimization
5. **Validates with Monte Carlo simulation** (10,000+ paths)
6. **Provides detailed performance analytics** and visualizations

---

##  Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the System

```bash
# Run comprehensive portfolio optimization
python scripts/comprehensive_portfolio_system.py
```

---

##  System Components

### 1. Multi-Asset Data Fetcher (`data/multi_asset_fetcher.py`)

Fetches and processes data for multiple asset classes:

**Supported Assets:**
- **Equities**: AAPL, MSFT, GOOGL, JPM, JNJ, etc.
- **ETFs**: SPY, QQQ, VTI, VEA (international), VWO (emerging markets)
- **Bonds**: TLT (20Y Treasury), IEF (7-10Y), LQD (corporate), HYG (high yield)
- **Commodities**: GLD (gold), SLV (silver), USO (oil), DBC (diversified)
- **Crypto**: BTC-USD, ETH-USD, SOL-USD, ADA-USD
- **Real Estate**: VNQ, IYR (REIT ETFs)
- **Currency**: UUP (USD index), FXE (Euro), FXY (Yen)

**Example Usage:**
```python
from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio

# Create fetcher
fetcher = MultiAssetFetcher(start_date='2020-01-01')

# Get example portfolio (aggressive/balanced/conservative/all_assets)
tickers = create_example_portfolio('all_assets')

# Fetch data
prices, returns = fetcher.fetch_assets(tickers)

# Get statistics
stats = fetcher.get_asset_statistics()
```

### 2. Market Regime Detector (`strategy/market_regime_detector.py`)

Identifies 5 market conditions using multiple indicators:

**Regimes:**
-  **Bull Market**: Sustained upward trend + low volatility
-  **Bear Market**: Sustained downward trend
- ↔ **Sideways**: No clear trend, range-bound
-  **High Volatility**: Large price swings
-  **Crisis**: Extreme volatility + negative returns

**Indicators Used:**
- Trend: Moving averages, linear regression slope
- Volatility: Rolling std, volatility ratio, extreme moves
- Momentum: Win/loss ratio, streaks, consistency
- Distribution: Skewness, kurtosis, CVaR, tail risk

**Example Output:**
```
Detected Regime: BULL MARKET
Confidence: 72.3%

Recommendations:
- Focus on momentum and growth strategies
- Can increase portfolio concentration
- Suitable for aggressive optimization (Max Sharpe)
- Sparse portfolios may capture best performers
```

### 3. Intelligent Optimizer Selector (`strategy/intelligent_optimizer_selector.py`)

AI-driven system that recommends best optimization method based on:
- Current market regime
- Asset universe size and characteristics
- Correlation structure
- Volatility environment
- User preferences

**Supported Methods:**

| Method | Best For | Typical Holdings |
|--------|----------|-----------------|
| **Max Sharpe** | Bull markets | 10-15 |
| **Min Variance** | Bear/high volatility | 15-20 |
| **Risk Parity** | Sideways markets | All assets |
| **Equal Weight** | Crisis mode | All assets |
| **Sparse Sharpe** | Bull + large universe | 5-10 |

### 4. Backtesting Engine (`strategy/backtesting_engine.py`)

Professional-grade backtesting with:
- Rolling window optimization
- Transaction cost modeling (default: 0.1%)
- Regime-adaptive strategy switching
- Multiple rebalancing frequencies

**Performance Metrics:**
- Total/Annual Return
- Sharpe/Sortino/Calmar Ratios
- Maximum Drawdown
- VaR and CVaR (95%)
- Win Rate, Skewness, Kurtosis
- Alpha, Beta, Information Ratio vs. benchmark
- Transaction costs and turnover

### 5. Monte Carlo Simulator (`strategy/backtesting_engine.py`)

Validation through simulation:
- 10,000+ simulated paths
- Parametric (multivariate normal) method
- Bootstrap (historical resampling) method
- Confidence intervals (95%)
- Probability distributions

**Output:**
- Expected final value
- 95% VaR and CVaR
- Probability of profit
- Probability of >20% gain
- Probability of >10% loss

---

##  Complete Workflow Example

```python
from scripts.comprehensive_portfolio_system import ComprehensivePortfolioSystem

# Initialize system
system = ComprehensivePortfolioSystem(
    start_date='2020-01-01',
    end_date='2024-01-01',
    risk_free_rate=0.03,
    transaction_cost=0.001,
    rebalance_frequency='monthly'
)

# Step 1: Setup portfolio
system.setup_portfolio(scenario='all_assets')
# Options: 'aggressive', 'balanced', 'conservative', 'all_assets'
# Or use custom_tickers=['AAPL', 'BTC-USD', 'GLD', ...]

# Step 2: Detect market regime
regime_result = system.detect_market_regime(market_proxy='SPY')

# Step 3: Select optimization method
recommendation = system.select_optimization_method(
    preferences={
        'prefer_sparse': False,
        'prefer_interpretable': True,
        'prefer_robust': True
    }
)

# Step 4: Run backtest (tests multiple methods)
backtest_results = system.run_backtest(test_multiple_methods=True)

# Step 5: Monte Carlo validation
mc_results = system.run_monte_carlo_validation(
    n_simulations=10000,
    n_days=252  # 1 year
)

# Step 6: Visualize results
system.visualize_results()

# Step 7: Generate final report
system.generate_final_report()
```

---

##  Output Files

All results are saved to `results/comprehensive_backtest/`:

```
results/comprehensive_backtest/
 portfolio_values.csv              # Portfolio value over time
 weights_history.csv               # Asset weights at each rebalance
 performance_metrics.csv           # All performance metrics
 method_comparison.csv             # Comparison of different methods
 comprehensive_results.png         # Visualization dashboard
 final_report.txt                  # Comprehensive text report
```

---

##  Visualization Dashboard

The system generates a comprehensive dashboard with:

1. **Portfolio Performance**: Value over time vs. benchmark
2. **Drawdown Chart**: Maximum drawdown visualization
3. **Asset Class Distribution**: Pie chart of asset allocation
4. **Monte Carlo Distribution**: Histogram of simulated outcomes
5. **Performance Metrics**: Bar chart of key metrics

---

##  Customization

### Create Custom Portfolio

```python
# Define your own asset universe
custom_tickers = [
    # Your favorites
    'AAPL', 'MSFT', 'GOOGL',
    # Bonds for stability
    'TLT', 'LQD',
    # Commodities
    'GLD', 'DBC',
    # Crypto (optional)
    'BTC-USD', 'ETH-USD',
    # International exposure
    'VEA', 'VWO'
]

system.setup_portfolio(custom_tickers=custom_tickers)
```

### Adjust Rebalancing

```python
# Change rebalancing frequency
system = ComprehensivePortfolioSystem(
    rebalance_frequency='quarterly'  # or 'weekly', 'monthly', 'quarterly'
)
```

### Modify Transaction Costs

```python
# Adjust transaction costs (default: 0.1%)
system = ComprehensivePortfolioSystem(
    transaction_cost=0.002  # 0.2% transaction cost
)
```

---

##  Example Output

```
================================================================================
PERFORMANCE METRICS
================================================================================

Return Metrics:
  Total Return:              47.32%
  Annual Return:             10.23%
  Annual Volatility:         14.56%

Risk-Adjusted Returns:
  Sharpe Ratio:              0.6234
  Sortino Ratio:             0.8912
  Calmar Ratio:              0.7845

Risk Metrics:
  Maximum Drawdown:         -13.04%
  VaR (95%):                 -2.34%
  CVaR (95%):                -3.12%

Distribution:
  Skewness:                  -0.2341
  Kurtosis:                   3.4521
  Win Rate:                  58.23%

Trading Statistics:
  Number of Rebalances:          48
  Average Turnover:           12.34%
  Transaction Costs:           0.0234

================================================================================
BENCHMARK COMPARISON (SPY)
================================================================================

Alpha & Beta:
  Alpha:                     2.34%
  Beta:                      0.8234
  Information Ratio:         0.4521

Excess Performance:
  Excess Return:             2.34%
  Win Rate vs Benchmark:    61.23%
```

---

##  Advanced Features

### 1. Regime-Adaptive Optimization

The system automatically switches between optimization methods based on detected market regime:

```python
Bull Market → Max Sharpe (maximize returns)
Bear Market → Min Variance (preserve capital)
Sideways → Risk Parity (balanced allocation)
High Volatility → Min Variance (risk management)
Crisis → Equal Weight (simplicity & robustness)
```

### 2. Multiple Method Comparison

Test all methods simultaneously:

```python
system.run_backtest(test_multiple_methods=True)
```

Output:
```
Method Comparison:
  Method            Total Return  Sharpe  Max Drawdown
  Max Sharpe        52.34%        0.6823  -15.23%
  Min Variance      38.21%        0.7234  -8.45%
  Risk Parity       43.12%        0.6534  -11.23%
  Equal Weight      35.67%        0.5123  -12.34%
  Adaptive (Ours)   47.32%        0.6234  -13.04%
```

### 3. Monte Carlo Validation

Two simulation methods:

**Parametric (Multivariate Normal):**
- Assumes normal distribution
- Fast computation
- Good for short-term projections

**Bootstrap (Historical Resampling):**
- Uses actual historical returns
- No distribution assumptions
- Better captures fat tails and skewness

---

##  Key Concepts

### Market Regime Detection

Uses multiple indicators to classify market conditions:
- **Trend indicators**: MA crossovers, regression slope
- **Volatility indicators**: Rolling std, volatility ratio
- **Momentum indicators**: Win/loss streaks, consistency
- **Distribution indicators**: Skewness, kurtosis, tail risk

### Optimization Methods

**Max Sharpe**: Maximizes return per unit of risk
- Best in bull markets
- Can be concentrated (high risk)

**Min Variance**: Minimizes portfolio volatility
- Best in bear markets or high volatility
- Emphasizes defensive assets

**Risk Parity**: Equalizes risk contribution across assets
- Good for sideways markets
- Balanced, diversified allocation

**Equal Weight**: 1/N allocation
- Simplest, most robust
- Good in crisis when models break down

**Sparse Sharpe**: Optimal sparse portfolio (NeurIPS 2024)
- Selects subset of assets
- Global optimality guarantee
- Good for large asset universes

### Monte Carlo Simulation

Generates thousands of possible future scenarios:
- **Parametric**: Assumes returns follow multivariate normal distribution
- **Bootstrap**: Randomly samples from historical returns
- Provides probabilistic view of outcomes
- Helps assess downside risk

---

##  Troubleshooting

### Data Download Issues

If yfinance fails:
```python
# Try with specific date range
fetcher = MultiAssetFetcher(
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### Optimization Failures

If optimization fails:
- Check for NaN values in returns
- Ensure sufficient history (>60 days)
- Reduce asset universe size
- Use fallback to equal weight

### Memory Issues

For large simulations:
```python
# Reduce simulation count
mc_results = system.run_monte_carlo_validation(
    n_simulations=1000,  # Instead of 10000
    n_days=126  # 6 months instead of 1 year
)
```

---

##  References

1. Lin, Y., Lai, Z.-R., & Li, C. (2024). "A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization." NeurIPS 2024.

2. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." Journal of Portfolio Management, 36(4), 60-70.

3. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" Review of Financial Studies, 22(5), 1915-1953.

---

##  Support

For issues or questions, please refer to the documentation or create an issue in the repository.

---

##  Performance Tips

1. **Use smaller lookback windows** for faster backtests (e.g., 126 days instead of 252)
2. **Reduce rebalancing frequency** to minimize transaction costs
3. **Start with balanced portfolio** before trying all_assets
4. **Run Monte Carlo with 1000 simulations** first, then increase if needed

---

##  Next Steps

1. Run the system with default settings
2. Explore different portfolio scenarios
3. Adjust parameters based on your risk tolerance
4. Compare methods and choose the best one
5. Monitor live performance (future feature)

**Happy Optimizing! **
