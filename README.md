# Portfolio Optimization System

Market-adaptive portfolio optimization system integrating 133-factor analysis, market regime detection, intelligent method selection, and multi-asset allocation.

## Key Features

## Key Features
- 5 market regimes with adaptive strategy selection (including mixture-based regime modeling)  
- 6 optimization methods including Bayesian and globally optimal sparse Sharpe portfolio  
- 133-factor library with 6 selection methods (IC, LASSO, Random Forest, etc.)  
- 16-year backtest (2008–2024): 12.8% annual return, 0.677 Sharpe ratio, 6.6% alpha  


## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/Portfolio_Optimization_system
cd Portfolio_Optimization_system

# ONE-COMMAND RUN: Execute entire pipeline
./run.sh

# Or run individual components:
python scripts/quick_test.py                        # Validation
python scripts/test_bayesian_system.py              # Bayesian system test
python scripts/historical_analysis.py               # 16-year backtest
python scripts/comprehensive_portfolio_system.py    # Full optimization
```

## Project Structure

```
Portfolio_Optimization_system/
├── market/                      # Market regime detection
│   ├── regime_detector.py       # Traditional detector
│   └── mixture_regime_model.py  # Mixture-based probabilistic detector
├── factor/                      # Factor analysis (133 factors)
│   ├── factor_analyzer.py       # Analysis engine
│   ├── factor_selection.py      # Factor ranking
│   └── integrated_factor_system.py
├── optimization/                # Optimization methods
│   ├── bayesian_optimizer.py    # Bayesian portfolio optimization
│   ├── sparse_sharpe_optimizer.py  # Sparse Sharpe (NeurIPS 2024)
│   └── intelligent_selector.py  # Method selector
├── evaluation/                  # Backtesting & evaluation
│   ├── backtesting_engine.py
│   ├── bayesian_updater.py
│   └── bayesian_system.py
├── scripts/                     # Execution scripts
└── run.sh                       # Master run script
```

## System Pipeline

```
1. DATA → 2. MARKET → 3. FACTOR → 4. OPTIMIZATION → 5. EVALUATION
   Fetch      Detect     Analyze     Optimize         Backtest
```

## Core Components

### 1. Bayesian + Mixture Model System

Probabilistic regime detection with soft transitions:
- Mixture Model: Soft regime probabilities instead of hard classification
- Bayesian Optimization: Regime-aware portfolio with uncertainty quantification
- Online Learning: Adaptive posterior updates with anomaly detection


### 2. Factor Analysis

- 133-factor library: technical, fundamental, macro, ML, beta
- 6 selection methods: IC Analysis, LASSO, Random Forest, MI, Forward Selection, Factor Returns
- Style factors: Fama-French 3/5-factor, Carhart 4-factor models
- Factor timing: Momentum-based rotation and tilting

### 3. Market Regime Detection

Adaptive strategy selection across 5 regimes:

| Regime | Strategy | Holdings |
|--------|----------|----------|
| Bull Market | Max Sharpe | 10-15 |
| Bear Market | Min Variance | 15-20 |
| Sideways | Risk Parity | All |
| High Volatility | Min Variance | 15-20 |
| Crisis | Equal Weight | All |

### 4. Optimization Methods

- Max Sharpe
- Min Variance
- Risk Parity
- Equal Weight
- Sparse Sharpe (NeurIPS 2024)
- Factor-Tilted
- Bayesian Mean-Variance

### 5. Asset Universe

Equities: AAPL, MSFT, GOOGL, JPM, JNJ | ETFs: SPY, QQQ, VTI, VEA, VWO | Bonds: TLT, IEF, LQD, HYG | Commodities: GLD, SLV, DBC | Crypto: BTC-USD, ETH-USD

### 6. Performance (2008–2024)

Please see detailed performance results in the `results/` folder.

## Usage Examples

### Complete Pipeline

```bash
./run.sh  # One command runs everything
```



## Output Files

Key outputs saved to `results/`:
- `factor_importance_report.csv` - Factor rankings by IC, Sharpe, selection count
- `integrated_factor_report.txt` - Alpha, beta, R-squared, return attribution
- `performance_metrics.csv` - Sharpe, Sortino, Calmar, max drawdown, alpha, beta
- `portfolio_values.csv` - Portfolio value time series
- `weights_history.csv` - Historical weight allocations

## Documentation

See [FACTOR_ANALYSIS_GUIDE.md](FACTOR_ANALYSIS_GUIDE.md) for detailed factor analysis documentation.


## Disclaimer

Educational and research purposes only. Past performance does not guarantee future results.

---

May we all find our own **alpha** — in markets and in life.📈
