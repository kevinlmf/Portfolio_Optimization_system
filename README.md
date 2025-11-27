# Portfolio Optimization System

A complete portfolio optimization workflow system organized by the natural process of portfolio construction.

## Workflow Overview

The system follows a six-step workflow:

1. **Data** - `data/` - Data acquisition and API integration
2. **Factor Mining** - `workflow/1_factor_mining/` - Extract common factors driving stock returns
3. **Build Matrix** - `workflow/2_build_matrix/` - Construct stock-factor relationship matrices
4. **Select Objective** - `workflow/3_select_objective/` - Choose optimization objectives and constraints
5. **Parameter Estimation** - `workflow/4_estimate_parameters/` - Estimate parameters (μ, F, D, Σ)
6. **Evaluation** - `workflow/5_evaluation/` - Evaluate portfolio performance and backtesting

## Project Structure

```
Portfolio_Optimization_system/
├── data/                       # Step 0: Data acquisition and API interfaces
├── workflow/                   # Workflow modules
│   ├── 1_factor_mining/       # Step 1: Factor mining (PCA, Factor Analysis)
│   ├── 2_build_matrix/         # Step 2: Build stock-factor matrices (OLS, Ridge)
│   ├── 3_select_objective/    # Step 3: Select objectives (Sharpe, CVaR, etc.)
│   ├── 4_estimate_parameters/ # Step 4: Parameter estimation (μ, F, D, Σ)
│   └── 5_evaluation/         # Step 5: Evaluation and backtesting
└── workflow.py                # Main workflow script
```

## Quick Start

### Clone the Repository
git clone https://github.com/kevinlmf/Portfolio_Optimization_system
cd Portfolio_Optimization_system

```bash
cd Portfolio_Optimization_system
```

### Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn yfinance
```

### Run Complete Workflow

```bash
python workflow.py
```

## Complete Example

### Step 1: Factor Mining

**Current Implementation**: **PCA (Principal Component Analysis)** extracts common factors from stock returns.

```python
from workflow import PortfolioWorkflow, ObjectiveType

workflow = PortfolioWorkflow(returns)
factors = workflow.step1_factor_mining(top_n=5, method='pca')
```

**How it works**: PCA decomposes the covariance matrix to find principal components explaining the most variance. These represent common factors driving stock movements.

**Alternatives**: `'factor_analysis'`, `'statistical'`

### Step 2: Build Correlation Matrix

**Current Implementation**: **OLS regression** estimates factor loadings, then builds correlation matrices.

```python
factor_loadings = workflow.step2_build_matrix()
# Regresses each stock's returns on factors: R_i = α + β * F + ε
# Extracts factor loadings (β) to form B matrix (N x K)
# Builds correlation matrices between stocks and factors
```

**How it works**: For each stock, regress returns on factor returns to get factor loadings (B matrix), showing each stock's exposure to factors.

**Alternatives**: `'ridge'` for Ridge regression (handles multicollinearity)

### Step 3: Select Objective

**Current Implementation**: Optimize **Sharpe Ratio** using **Quadratic Programming (QP)**.

```python
decisions = workflow.step3_select_objective(
    objective=ObjectiveType.SHARPE,  # Maximize Sharpe ratio
    constraints={
        'long_only': True,    # Long-only portfolio
        'leverage': 1.0,      # No leverage
        'max_weight': 0.3     # Maximum 30% per asset
    }
)
```

**How it works**: Maximizes Sharpe ratio (expected return / volatility) subject to constraints using QP solver.

**Available objectives**: `SHARPE`, `CVAR`, `RISK_PARITY`, `MIN_VARIANCE`, `MEAN_VARIANCE`

**Available methods**: `'qp'`, `'sparse_sharpe'`

### Step 4: Parameter Estimation

**Current Implementation**: **Sample Estimator** estimates all parameters from historical data.

```python
knowledge = workflow.step4_estimate_parameters(factors)
# Contains: μ (expected returns), Σ (covariance), B (factor loadings),
#           F (factor covariance), D (idiosyncratic risk)
```

**How it works**: 
- **μ**: Sample mean of returns
- **B**: Factor loadings from Step 2
- **F**: Covariance of factor returns
- **D**: Residual variances from factor regression
- **Σ**: Full covariance = B * F * B' + D

**Alternatives**: Bayesian estimators, Shrinkage estimators (extensible)

### Step 5: Evaluation

**Current Implementation**: Evaluates portfolio performance and calculates optimal weights.

```python
weights = workflow.step5_evaluation(decisions)
# Calculates: expected return, risk, Sharpe ratio, annualized metrics,
#             max drawdown, and other performance metrics
```

**How it works**: Optimizer uses estimated parameters (μ, Σ) and objective to find optimal weights, then evaluates expected performance.

**Metrics**: Annualized return/volatility, Sharpe/Sortino/Calmar ratios, max drawdown, win rate, skewness, kurtosis

### Complete Workflow

```python
from workflow import PortfolioWorkflow, ObjectiveType
from data import APIClient

# Get data
client = APIClient(source='yahoo')
returns = client.fetch_returns(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Run complete workflow
workflow = PortfolioWorkflow(returns)
weights = workflow.run_complete_workflow(
    objective=ObjectiveType.SHARPE,
    constraints={'long_only': True, 'max_weight': 0.3},
    n_factors=5,
    factor_method='pca'
)
```

## Future Extensions

The system is designed to be extensible. Potential enhancements include:

- **Factor Mining**: Fama-French factors, dynamic factor selection, nonlinear extraction (autoencoders, ICA)
- **Correlation Matrix**: Dynamic correlations (DCC-GARCH), copula models, sparse correlation estimation
- **Objectives**: Multi-objective optimization, robust optimization, regime-aware objectives, transaction cost integration
- **Parameter Estimation**: Bayesian estimation, shrinkage methods (Ledoit-Wolf), time-varying parameters (Kalman filter), ML-based return prediction
- **Evaluation**: Out-of-sample backtesting, Monte Carlo simulation, risk/performance attribution, realistic transaction cost modeling
- **Additional Features**: Multi-period optimization, risk budgeting, ESG integration, alternative data sources, real-time optimization

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- yfinance >= 0.2.0

---
Disclaimer
Educational and research purposes only. Past performance does not guarantee future results.
---
May we all find our own alpha — in markets and in life.📈
