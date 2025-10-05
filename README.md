# Adaptive Portfolio Optimization System

> **An intelligent, market-adaptive portfolio optimization system that automatically selects the best optimization method based on current market conditions.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

## Overview

This system provides **AI-driven portfolio optimization** that adapts to market conditions. Instead of manually choosing optimization methods, the system automatically detects the market regime and selects the optimal approach - delivering regime-appropriate asset allocations without requiring factor mining or complex data pipelines.

### Key Features

- **AI-Driven Method Selection**: Automatically recommends optimal optimization method based on market regime
- **Market Regime Detection**: Identifies 5 market conditions (bull, bear, sideways, high volatility, crisis)
- **Sparse Sharpe Optimization**: Globally optimal m-sparse portfolio with theoretical guarantees (NeurIPS 2024)
- **Multi-Method Optimization**: Implements 5 optimization approaches (Max Sharpe, Min Variance, Risk Parity, Equal Weight, Sparse Sharpe)
- **Adaptive Asset Allocation**: Different portfolios for different market conditions
- **Production-Ready**: Clean, lightweight architecture optimized for real-world use
- **Comprehensive Analysis**: Automated comparison reports and visualizations
- **Fast & Simple**: No factor mining required - just price data and market conditions

## Architecture

**Simple, focused, and production-ready:**

```
Portfolio_Optimization_System/
├── README.md                                    # System documentation
├── ADAPTIVE_OPTIMIZATION_GUIDE.md              # Detailed usage guide (中文)
├── setup.py                                    # Installation script
├── requirements.txt                            # Dependencies
├── 2410.21100v1.pdf                            # NeurIPS 2024 paper (Sparse Sharpe)
│
├── scripts/                                    # Execution scripts
│   ├── adaptive_portfolio_optimizer.py            # MAIN ENTRY POINT
│   └── demo_intelligent_selector.py               # System demonstration
│
└── strategy/                                   # Core optimization algorithms
    ├── market_regime_detector.py                  # Detect market conditions
    ├── intelligent_optimizer_selector.py          # AI-driven method selection
    └── sparse_sharpe_optimizer.py                 # mSSRM-PGA algorithm (NeurIPS 2024)
```

**That's it! Just 7 core files - no bloat, no complexity.**

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Portfolio_Optimization_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set up FRED API key for enhanced macro data
export FRED_API_KEY="your_fred_api_key_here"
```

### System Requirements

- **Python**: 3.8+ (recommended: 3.9+)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB for data and results
- **Network**: Internet connection for data fetching
- **Optional**: FRED API key for comprehensive macroeconomic data

### Option 1: Adaptive Portfolio Optimizer (Main Entry Point)

**One-click execution - automatically selects the best method based on market conditions:**

```bash
python scripts/adaptive_portfolio_optimizer.py
```

This will automatically:
1. **Fetch market data** for 22 diversified assets (tech, financial, healthcare, ETFs)
2. **Detect market regime** - Identify current market conditions with confidence scores
3. **Select optimal method** - AI recommends the best optimization approach
4. **Generate portfolio** - Create regime-appropriate asset allocation
5. **Compare all methods** - Test Max Sharpe, Min Variance, Risk Parity, Equal Weight, Sparse Sharpe
6. **Create visualizations** - Automatic charts and comparison reports

### Option 2: System Demonstration

Experience the underlying intelligent selection system:

```bash
python scripts/demo_intelligent_selector.py
```

This demonstration showcases the AI decision-making process:
1. **Market Regime Detection** - See how the system identifies market conditions
2. **Method Comparison Matrix** - Understand which methods work best in each regime
3. **Scenario Analysis** - Compare recommendations across different market conditions
4. **Sparse Sharpe Optimization** - See the NeurIPS 2024 algorithm in action
5. **Comprehensive Visualizations** - Method suitability heatmaps and performance charts

## Core Capabilities

### 1. Market Regime Detection

Automatically identifies the current market condition:

| Regime | Characteristics | Detection Indicators |
|--------|----------------|---------------------|
| **Bull Market** | Uptrend, moderate volatility | MA trends, positive momentum, low VIX |
| **Bear Market** | Downtrend, rising volatility | Negative returns, bearish MA crossover |
| **Sideways** | Range-bound, no clear trend | Flat MA, mean-reverting behavior |
| **High Volatility** | Large price swings | High standard deviation, extreme CVaR |
| **Crisis** | Extreme decline, panic | Sharp drawdown, negative skew, tail events |

### 2. Optimization Methods

**5 optimization approaches for different market conditions:**

| Method | Best For | Portfolio Style | Typical Holdings |
|--------|----------|----------------|------------------|
| **Max Sharpe** | Bull markets | Concentrated | 10-15 assets |
| **Sparse Sharpe** | Bull + large universe | Very concentrated | 5-10 assets |
| **Min Variance** | Bear/high volatility | Defensive | 15-20 assets |
| **Risk Parity** | Sideways/uncertain | Balanced | All assets |
| **Equal Weight** | Crisis | Maximum robust | All assets |

### 3. Intelligent Selection Engine

The AI recommendation system evaluates methods based on:

- **40% Market Regime Suitability** - How well the method performs in detected regime
- **25% Asset Universe Fit** - Number of assets and diversification requirements
- **20% Volatility Handling** - Ability to manage current market volatility
- **10% Constraint Compatibility** - Alignment with portfolio constraints
- **5% User Preferences** - Customizable preferences (sparsity, robustness, etc.)

## Example Output

Running `python scripts/adaptive_portfolio_optimizer.py` produces:

**Terminal Output:**
```
Detected Regime: BULL MARKET
   Confidence: 78.5%

Recommended Method: SPARSE_SHARPE
   Confidence: 85.2%

Optimization Complete:
  Sharpe Ratio: 1.2543
  Annual Return: 18.45%
  Annual Volatility: 14.71%
  Number of Holdings: 8

PORTFOLIO COMPARISON:
Method           Sharpe Ratio  Annual Return  Annual Vol  Holdings
Sparse Sharpe    1.2543        18.45%         14.71%      8
Max Sharpe       1.1987        17.23%         14.38%      12
Risk Parity      0.9876        13.67%         13.85%      22
Min Variance     0.8234        10.45%         12.69%      15
Equal Weight     0.7543        11.23%         14.89%      22
```

**Visualizations:**
- Sharpe ratio comparison bar chart
- Risk-return scatter plot
- Portfolio weights allocation
- Holdings concentration analysis

All saved to `results/adaptive_optimization_results.png`

## Advanced Usage

### Sparse Sharpe Optimization (mSSRM-PGA)

Our implementation of the **globally optimal m-sparse Sharpe ratio maximization** algorithm from:

> Lin, Y., Lai, Z.-R., & Li, C. (2024). "A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization." *NeurIPS 2024*.

**Key Features:**
- **Exact Sparsity Control**: Select exactly m assets (not ℓ1 approximation)
- **Global Optimality**: Theoretical guarantee under certain conditions
- **Direct Optimization**: Maximizes actual Sharpe ratio (not a proxy)
- **Fast Convergence**: O(1/√k) for iterates, O(1/k) for function values

**Usage:**

```python
from strategy.sparse_sharpe_optimizer import SparseSharpeOptimizer, MultiSparsityOptimizer

# Single sparsity level
optimizer = SparseSharpeOptimizer(epsilon=1e-3, max_iter=5000)
result = optimizer.optimize(returns, m=10)  # Select 10 assets

print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
print(f"Selected Assets: {result['sparsity']}")
print(f"Converged: {result['converged']}")

# Try multiple sparsity levels
multi_optimizer = MultiSparsityOptimizer()
results_df = multi_optimizer.optimize_grid(
    returns=asset_returns,
    sparsity_levels=[5, 10, 15, 20],
    asset_names=ticker_list
)

# Get best portfolio
best_m, best_result = multi_optimizer.get_best_portfolio(metric='sharpe_ratio')
```

**When to Use:**
- Large asset universes (50+ assets)
- Bull market conditions
- Need for concentrated portfolios
- Desire for interpretability (few holdings)
- Have theoretical rigor requirements

### Adaptive Portfolio Optimization

**Automatically switches optimization methods based on market conditions** - This is the recommended approach for dynamic market conditions:

```python
from scripts.adaptive_portfolio_optimizer import AdaptivePortfolioOptimizer

# Initialize adaptive optimizer
optimizer = AdaptivePortfolioOptimizer(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'SPY', 'TLT', 'GLD'],
    lookback_days=252,
    risk_free_rate=0.03
)

# Fetch market data
optimizer.fetch_data()

# Run adaptive optimization
# The system will:
# 1. Detect current market regime
# 2. Select optimal optimization method
# 3. Generate portfolio allocation
# 4. Compare with other methods
results = optimizer.run_adaptive_optimization(test_all_methods=True)

# Generate comparison report
comparison_df = optimizer.create_comparison_report()

# Visualize results
fig = optimizer.visualize_results()
```

**Key Benefits:**
- **Automatically adapts to markets**: No need to manually decide which optimization method to use
- **Multi-method comparison**: Test all methods simultaneously to find the optimal solution
- **Visual analysis**: Automatically generate comparison charts and reports
- **Production-ready**: Can be directly used for live trading decisions

**Different Methods for Different Markets:**

| Market Condition | Recommended Method | Portfolio Characteristics |
|-----------------|-------------------|--------------------------|
| **Bull Market** | Sparse Sharpe / Max Sharpe | Concentrated, high Sharpe |
| **Bear Market** | Min Variance | Defensive, low volatility |
| **Sideways** | Risk Parity | Balanced, diversified |
| **High Volatility** | Min Variance / Risk Parity | Stable, risk-controlled |
| **Crisis** | Equal Weight | Maximum robustness |

### Custom Factor Development

```python
from strategy.factor.alpha.technical_alpha_factors import TechnicalAlphaFactors

class MyCustomFactors(TechnicalAlphaFactors):
    def custom_momentum_factor(self, data, window=20):
        """Example custom factor implementation"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(window).mean()
        volatility = returns.rolling(window).std()
        return momentum / (volatility + 1e-8)  # Risk-adjusted momentum
```

### Production Integration

```python
from scripts.smart_portfolio_optimizer import SmartPortfolioOptimizer

# Initialize system
optimizer = SmartPortfolioOptimizer(
    start_date="2020-01-01",
    risk_free_rate=0.03
)

# Run full pipeline
optimizer.fetch_market_data(['AAPL', 'MSFT', 'GOOGL'])
optimizer.mine_alpha_factors(min_ic_threshold=0.02)
optimizer.estimate_risk_models()
result = optimizer.optimize_portfolio(objective='max_sharpe')
optimizer.backtest_strategy(rebalance_frequency='monthly')

# Generate comprehensive report
report = optimizer.generate_report(save_plots=True)
```

## Latest Enhancements

### AI-Driven Optimizer Selection (NEW!)

Revolutionary intelligent system that adapts to market conditions:

- **Market Regime Detection**: Automatically identifies bull, bear, sideways, high volatility, and crisis regimes
- **Intelligent Method Selection**: AI-powered recommendation engine evaluates 8 optimization methods
- **Comprehensive Analysis**: Considers regime, asset configuration, volatility, constraints, and preferences
- **Confident Recommendations**: Provides ranked alternatives with detailed reasoning
- **Scenario Testing**: Compare method performance across different market conditions

### Sparse Sharpe Optimization (NEW!)

Implementation of globally optimal m-sparse portfolio (NeurIPS 2024):

- **Global Optimality**: Theoretical guarantee for m-sparse Sharpe ratio maximization
- **Exact Sparsity**: Direct ℓ0 constraint (select exactly m assets), not ℓ1 approximation
- **Fast Convergence**: O(1/√k) convergence rate with proximal gradient algorithm
- **Rigorous Theory**: Based on peer-reviewed NeurIPS 2024 paper by Lin et al.
- **Production Ready**: Fully integrated with multi-sparsity grid search

## Output Files

All results are saved to `results/` directory:

```
results/
├── adaptive_optimization_results.png      # 4-panel visualization
├── adaptive_optimization_comparison.csv   # Method comparison table
└── intelligent_selector_demo.png          # Demo visualizations
```

## Configuration

### Key Parameters

Edit parameters in `adaptive_portfolio_optimizer.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_days` | 252 | Historical data window (1 year) |
| `risk_free_rate` | 0.03 | Annual risk-free rate (3%) |
| `max_weight` | 0.30 | Maximum single asset weight (30%) |
| `sparsity_m` | N/3 | Number of assets for Sparse Sharpe |

### Asset Universe Examples

```python
# Tech-focused portfolio
tech_portfolio = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# Balanced multi-sector
balanced = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'WMT', 'HD', 'V']

# Multi-asset with ETFs
diversified = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'SPY', 'QQQ', 'TLT', 'GLD']
```

## Testing

Quick verification:

```bash
# Test the adaptive optimizer
python scripts/adaptive_portfolio_optimizer.py

# Test the demo system
python scripts/demo_intelligent_selector.py

# Test imports
python -c "from strategy.market_regime_detector import MarketRegimeDetector; print('All imports successful')"
```

## Changelog

### Version 3.0 (Latest) - October 2024

**Complete System Redesign:**
- **Simplified Focus**: Removed Alpha factor mining complexity - focused purely on adaptive portfolio optimization
- **AI-Driven Method Selection**: Intelligent system that automatically selects optimal optimization method based on market regime
- **Production Ready**: Streamlined from 50+ files to just 7 core files

**Core Features:**
- `adaptive_portfolio_optimizer.py`: Main adaptive optimization system
- `market_regime_detector.py`: Detects 5 market regimes with confidence scores
- `intelligent_optimizer_selector.py`: AI-powered recommendation engine
- `sparse_sharpe_optimizer.py`: mSSRM-PGA algorithm from NeurIPS 2024 paper
- `demo_intelligent_selector.py`: Interactive system demonstration

**Architecture:**
- Clean, modular design with clear separation of concerns
- No external data dependencies (uses yfinance for market data)
- Comprehensive Chinese documentation (ADAPTIVE_OPTIMIZATION_GUIDE.md)
- Automated visualization and reporting

**References:**
- Lin, Y., Lai, Z.-R., & Li, C. (2024). "A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization." *NeurIPS 2024*.

## Support & Documentation

**Primary Documentation:**
- [README.md](README.md) - This file (system overview)

**Quick Help:**
```bash
# Run the main system
python scripts/adaptive_portfolio_optimizer.py

# See demonstration
python scripts/demo_intelligent_selector.py
```

**For Issues:**
- Check inline code documentation
- Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Built for intelligent, adaptive portfolio management**

