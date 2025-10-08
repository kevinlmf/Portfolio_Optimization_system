# Portfolio Optimization System

This repository combines AI-driven market regime detection with modern portfolio optimization for adaptive multi-asset allocation.

The system demonstrates a production-grade quantitative architecture with:
- 5 market regime detection and adaptive strategy selection
- 6 optimization methods including globally optimal sparse Sharpe portfolio
- Factor analysis and selection from 133-factor library
- Professional backtesting with transaction costs and Monte Carlo validation
- 16-year historical analysis (2008-2024) delivering 12.8% annual return with 0.677 Sharpe ratio

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/kevinlmf/Portfolio_Optimization_system
cd Portfolio_Optimization_system

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run system
python scripts/quick_test.py                        # Validation
python scripts/historical_analysis.py               # 16-year backtest
python scripts/comprehensive_portfolio_system.py    # Full optimization
python scripts/demo_factor_selection.py             # Factor selection
python scripts/demo_factor_analysis.py              # Factor analysis
```

---

## Project Structure

```
Portfolio_Optimization_system/
├── data/
│   └── multi_asset_fetcher.py          # Multi-asset data loader
├── strategy/
│   ├── market_regime_detector.py       # 5-regime classifier
│   ├── intelligent_optimizer_selector.py   # AI method selection
│   ├── sparse_sharpe_optimizer.py      # Sparse portfolio optimizer
│   ├── backtesting_engine.py           # Backtesting framework
│   ├── factor_analyzer.py              # Factor analysis engine
│   ├── factor_selection.py             # Factor importance ranking
│   └── integrated_factor_system.py     # Factor integration
├── scripts/
│   ├── quick_test.py                   # System validation
│   ├── historical_analysis.py          # Historical backtest
│   ├── comprehensive_portfolio_system.py   # Full pipeline
│   ├── demo_factor_selection.py        # Factor selection demo
│   └── demo_factor_analysis.py         # Factor analysis demo
└── results/                            # Output directory
```

---

## Historical Performance (2008-2024)

**Overall Results:**
- Annual Return: 12.8%
- Sharpe Ratio: 0.677
- Alpha vs SPY: 6.6%
- Max Drawdown: -20.0%
- Beta: 0.57

**Period Breakdown:**

| Period | Total Return | Sharpe | Max DD | Alpha |
|--------|--------------|--------|--------|-------|
| 2008-2010 (Crisis) | 39.2% | 0.514 | -21.1% | 11.1% |
| 2010-2015 (Recovery) | 59.4% | 0.566 | -8.8% | 2.7% |
| 2015-2020 (Pre-COVID) | 98.9% | 0.777 | -23.2% | 4.6% |
| 2020-2025 (COVID+) | 201.5% | 0.852 | -27.0% | 7.8% |

---

## Key Features

### Factor Analysis System

Comprehensive multi-factor analysis framework:

| Feature | Description | Methods |
|---------|-------------|---------|
| **Factor Selection** | Identify most important factors from 133-factor library | IC, LASSO, Random Forest, Mutual Information |
| **Style Factors** | Fama-French factor models | FF3, FF5, Carhart 4-factor |
| **Statistical Factors** | PCA-based factor extraction | Principal components |
| **Factor Timing** | Momentum-based factor rotation | Factor momentum signals |
| **Factor Tilting** | Factor-tilted portfolio construction | Custom exposure targets |
| **Risk Attribution** | Factor-based risk decomposition | Systematic vs idiosyncratic |
| **Return Attribution** | Performance attribution by factor | Alpha generation analysis |

**Integration with 133-Factor Library**: Seamlessly connects with the Factor Mining System (technical, fundamental, macro, ML, beta factors)

**Factor Selection Methods**:
- **IC Analysis**: Information Coefficient (correlation with future returns)
- **LASSO Regularization**: Sparse factor selection with L1 penalty
- **Random Forest**: Tree-based feature importance
- **Mutual Information**: Non-linear dependency detection
- **Forward Selection**: Stepwise R-squared improvement
- **Factor Returns**: Long-short portfolio Sharpe ratios

### Market Regime Detection

Automatically identifies 5 market conditions and adapts strategy:

| Regime | Strategy | Holdings |
|--------|----------|----------|
| Bull Market | Max Sharpe | 10-15 |
| Bear Market | Min Variance | 15-20 |
| Sideways | Risk Parity | All |
| High Volatility | Min Variance | 15-20 |
| Crisis | Equal Weight | All |

### Optimization Methods

1. **Max Sharpe**: Maximize risk-adjusted returns
2. **Min Variance**: Minimize portfolio volatility
3. **Risk Parity**: Equalize risk contribution
4. **Equal Weight**: 1/N allocation (robust fallback)
5. **Sparse Sharpe**: Globally optimal subset selection (NeurIPS 2024)
6. **Factor-Tilted**: Target specific factor exposures

### Asset Coverage

- Equities: AAPL, MSFT, GOOGL, JPM, JNJ
- ETFs: SPY, QQQ, VTI, VEA, VWO
- Bonds: TLT, IEF, LQD, HYG
- Commodities: GLD, SLV, DBC
- Crypto: BTC-USD, ETH-USD

---

## Usage

### Basic Pipeline

```python
from scripts.comprehensive_portfolio_system import ComprehensivePortfolioSystem

# Initialize
system = ComprehensivePortfolioSystem(
    start_date='2020-01-01',
    transaction_cost=0.001,
    rebalance_frequency='monthly'
)

# Run pipeline
system.setup_portfolio(scenario='all_assets')
system.detect_market_regime()
system.run_backtest(test_multiple_methods=True)
system.run_monte_carlo_validation(n_simulations=10000)
system.generate_final_report()
```

### Factor Selection

```python
from strategy.factor_selection import FactorSelector

# Initialize
selector = FactorSelector(factor_data, returns, forward_periods=[1, 5, 10, 20])

# Run comprehensive analysis (6 methods)
importance_report = selector.generate_factor_importance_report(
    target_returns=portfolio_returns,
    top_n=20
)

# Get top factors
top_factors = importance_report.head(10)['factor'].tolist()
```

### Factor Analysis

```python
from strategy.integrated_factor_system import IntegratedFactorSystem

# Initialize factor system
factor_system = IntegratedFactorSystem(returns, prices, risk_free_rate=0.03)

# Construct Fama-French style factors
factors = factor_system.construct_style_factors(market_proxy=spy_returns)

# Run factor timing analysis
timing_results = factor_system.run_factor_timing(lookback=60, top_n_factors=3)

# Create factor-tilted portfolio
weights = factor_system.create_factor_tilted_portfolio(
    tilt_factors=['MKT', 'MOM'],
    tilt_strength=1.0,
    base_method='risk_parity'
)

# Analyze portfolio factor exposures
portfolio_returns = (returns * weights).sum(axis=1)
report = factor_system.analyze_portfolio_factors(
    portfolio_returns,
    portfolio_weights=weights,
    models=['fama_french_3', 'carhart_4']
)

# Optimize for target factor exposures
weights = factor_system.optimize_factor_exposures(
    target_exposures={'MKT': 0.8, 'SMB': 0.2, 'MOM': 0.3}
)
```

---

## Output Files

Results saved to `results/`:

```
results/
├── historical_analysis/
│   ├── period_comparison.csv
│   └── comprehensive_comparison.png
├── comprehensive_backtest/
│   ├── portfolio_values.csv
│   ├── weights_history.csv
│   ├── performance_metrics.csv
│   └── final_report.txt
└── factor_analysis/
    ├── factor_importance_report.csv        # Factor rankings (KEY FILE)
    ├── factor_strategy_comparison.csv      # Strategy performance
    ├── integrated_factor_report.txt        # Detailed analysis
    ├── factor_selection_analysis.png       # 8-panel visualization
    └── comprehensive_factor_analysis.png   # Factor dashboard
```

**Key Files to Review**:
1. `factor_importance_report.csv` - Factor rankings by avg_rank, IC, Sharpe, selection_count
2. `integrated_factor_report.txt` - Alpha, beta, R-squared, return attribution
3. `factor_selection_analysis.png` - Visual factor importance analysis

See `RESULTS_README.md` for detailed interpretation guide.

---

## Performance Metrics

**Return Metrics:** Total/Annual Return, Volatility, Excess Return

**Risk-Adjusted:** Sharpe, Sortino, Calmar, Information Ratio

**Risk Metrics:** Max Drawdown, VaR, CVaR, Downside Deviation

**Benchmark:** Alpha, Beta, Correlation, Tracking Error

---

## Advanced Features

- **Rolling Window Optimization**: 252-day lookback, 60-day minimum
- **Transaction Cost Modeling**: Bid-ask spread and market impact
- **Monte Carlo Simulation**: Parametric and bootstrap methods, 10,000+ paths
- **Multi-Period Analysis**: Performance attribution across market cycles
- **Factor Selection**: 6 methods to identify top factors from 133-factor library
- **Factor Timing**: Dynamic factor rotation based on momentum

---

## Documentation

- **RESULTS_README.md**: Complete guide to interpreting all output files
- **FACTOR_ANALYSIS_GUIDE.md**: Detailed factor analysis reference
- **COMPREHENSIVE_SYSTEM_GUIDE.md**: System architecture
- **HISTORICAL_ANALYSIS_SUMMARY.md**: 16-year backtest results

---

## References

1. Lin, Y., Lai, Z.-R., & Li, C. (2024). "A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization." NeurIPS 2024.
2. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." Journal of Portfolio Management.
3. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification." Review of Financial Studies.
4. Fama, E., & French, K. (1993). "Common risk factors in the returns on stocks and bonds." Journal of Financial Economics.
5. Carhart, M. (1997). "On Persistence in Mutual Fund Performance." Journal of Finance.

---

**Disclaimer**: Educational and research purposes only. Past performance does not guarantee future results.
