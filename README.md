# Portfolio Optimization System

Market-adaptive portfolio optimization system integrating 133-factor analysis, market regime detection, intelligent method selection, and multi-asset allocation.

**Key Highlights:**
- 5 market regimes with adaptive strategy selection
- 6 optimization methods including globally optimal sparse Sharpe portfolio
- 133-factor library with 6 selection methods (IC, LASSO, Random Forest, etc.)
- 16-year backtest (2008-2024): 12.8% annual return, 0.677 Sharpe ratio, 6.6% alpha

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

## Core Features

### 1. Factor Analysis
- **133-factor library** from Factor Mining System (technical, fundamental, macro, ML, beta)
- **6 selection methods**: IC Analysis, LASSO, Random Forest, Mutual Information, Forward Selection, Factor Returns
- **Style factors**: Fama-French 3/5-factor, Carhart 4-factor models
- **Factor timing**: Momentum-based rotation and tilting
- **Attribution**: Risk/return decomposition by factor

### 2. Market Regime Detection
Adaptive strategy selection across 5 regimes:

| Regime | Strategy | Holdings |
|--------|----------|----------|
| Bull Market | Max Sharpe | 10-15 |
| Bear Market | Min Variance | 15-20 |
| Sideways | Risk Parity | All |
| High Volatility | Min Variance | 15-20 |
| Crisis | Equal Weight | All |

### 3. Optimization Methods
Max Sharpe • Min Variance • Risk Parity • Equal Weight • Sparse Sharpe (NeurIPS 2024) • Factor-Tilted

### 4. Asset Universe
**Equities:** AAPL, MSFT, GOOGL, JPM, JNJ • **ETFs:** SPY, QQQ, VTI, VEA, VWO • **Bonds:** TLT, IEF, LQD, HYG • **Commodities:** GLD, SLV, DBC • **Crypto:** BTC-USD, ETH-USD

### 5. Performance (2008-2024)
12.8% annual return • 0.677 Sharpe • 6.6% alpha • -20% max drawdown • 0.57 beta

*See [HISTORICAL_ANALYSIS_SUMMARY.md](HISTORICAL_ANALYSIS_SUMMARY.md) for period-by-period breakdown*

---

## Usage

```python
from scripts.comprehensive_portfolio_system import ComprehensivePortfolioSystem

# Initialize and run full pipeline
system = ComprehensivePortfolioSystem(
    start_date='2020-01-01',
    transaction_cost=0.001,
    rebalance_frequency='monthly'
)

system.setup_portfolio(scenario='all_assets')
system.detect_market_regime()
system.run_backtest(test_multiple_methods=True)
system.run_monte_carlo_validation(n_simulations=10000)
system.generate_final_report()
```

*See [COMPREHENSIVE_SYSTEM_GUIDE.md](COMPREHENSIVE_SYSTEM_GUIDE.md) and [FACTOR_ANALYSIS_GUIDE.md](FACTOR_ANALYSIS_GUIDE.md) for detailed API documentation*

---

## Output & Documentation

**Key Output Files** (saved to `results/`):
- `factor_importance_report.csv` - Factor rankings by IC, Sharpe, selection count
- `integrated_factor_report.txt` - Alpha, beta, R², return attribution
- `performance_metrics.csv` - Sharpe, Sortino, Calmar, max drawdown, alpha, beta
- `portfolio_values.csv` & `weights_history.csv` - Time series results

**Documentation:**
- [RESULTS_README.md](RESULTS_README.md) - Output interpretation guide
- [FACTOR_ANALYSIS_GUIDE.md](FACTOR_ANALYSIS_GUIDE.md) - Factor analysis reference
- [COMPREHENSIVE_SYSTEM_GUIDE.md](COMPREHENSIVE_SYSTEM_GUIDE.md) - System architecture
- [HISTORICAL_ANALYSIS_SUMMARY.md](HISTORICAL_ANALYSIS_SUMMARY.md) - 16-year backtest results

---

## References

1. Lin, Y., Lai, Z.-R., & Li, C. (2024). "A Globally Optimal Portfolio for m-Sparse Sharpe Ratio Maximization." NeurIPS 2024.
2. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." Journal of Portfolio Management.
3. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification." Review of Financial Studies.
4. Fama, E., & French, K. (1993). "Common risk factors in the returns on stocks and bonds." Journal of Financial Economics.
5. Carhart, M. (1997). "On Persistence in Mutual Fund Performance." Journal of Finance.

---

**Disclaimer**: Educational and research purposes only. Past performance does not guarantee future results.
---
Code a lot, Smile a lot. 😄
