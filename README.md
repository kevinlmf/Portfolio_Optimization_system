# Portfolio Optimization System

A complete portfolio optimization workflow system organized by the natural process of portfolio construction.

## Workflow Overview

The system follows a seven-step workflow:

1. **Data** - `data/` - Data acquisition and API integration
2. **Factor Mining** - `workflow/1_factor_mining/` - Extract common factors driving stock returns
3. **Build Matrix** - `workflow/2_build_matrix/` - Construct stock-factor relationship matrices
4. **Select Objective** - `workflow/3_select_objective/` - Choose optimization objectives and constraints
5. **Parameter Estimation** - `workflow/4_estimate_parameters/` - Estimate parameters (μ, F, D, Σ)
6. **Evaluation** - `workflow/5_evaluation/` - Evaluate portfolio performance and backtesting
7. **Options Hedging** - `workflow/6_options_hedging/` - Predict volatility from portfolio, calculate Greeks, and hedge risks

## Project Structure

```
Portfolio_Optimization_system/
├── data/                       # Step 1: Data acquisition and API interfaces
├── workflow/                   # Workflow modules
│   ├── 1_factor_mining/       # Step 2: Factor mining (PCA, Factor Analysis)
│   ├── 2_build_matrix/         # Step 3: Build stock-factor matrices (OLS, Ridge)
│   ├── 3_select_objective/    # Step 4: Select objectives (Sharpe, CVaR, etc.)
│   ├── 4_estimate_parameters/ # Step 5: Parameter estimation (μ, F, D, Σ)
│   ├── 5_evaluation/         # Step 6: Evaluation and backtesting
│   └── 6_options_hedging/    # Step 7: Options hedging and risk management
└── workflow.py                # Main workflow script
```

## Quick Start

### Clone the Repository
```bash
git clone https://github.com/kevinlmf/Portfolio_Optimization_system
cd Portfolio_Optimization_system
```

### Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn yfinance
```

### Run Complete Workflow

```bash
# Basic workflow (Steps 1-5)
python workflow.py

# Or use the run script for more options
./run.sh workflow              # Basic workflow with absolute returns
./run.sh workflow-options      # Complete workflow with options hedging
./run.sh options-demo          # Options hedging demo only
./run.sh help                  # Show all available commands
```

## Complete Example

### Step 1: Factor Mining

**Current Implementation**: **PCA (Principal Component Analysis)** extracts common factors from stock returns.

**How it works**: PCA decomposes the covariance matrix to find principal components explaining the most variance. These represent common factors driving stock movements.

**Alternatives**: `'factor_analysis'`, `'statistical'`

### Step 2: Build Correlation Matrix

**Current Implementation**: **OLS regression** estimates factor loadings, then builds correlation matrices.

**How it works**: For each stock, regress returns on factor returns to get factor loadings (B matrix), showing each stock's exposure to factors.

**Alternatives**: `'ridge'` for Ridge regression (handles multicollinearity)

### Step 3: Select Objective

**Current Implementation**: Optimize **Sharpe Ratio** using **Quadratic Programming (QP)**.

**How it works**: Maximizes Sharpe ratio (expected return / volatility) subject to constraints using QP solver.

**Available objectives**: `SHARPE`, `CVAR`, `RISK_PARITY`, `MIN_VARIANCE`, `MEAN_VARIANCE`

**Available methods**: `'qp'`, `'sparse_sharpe'`

### Step 4: Parameter Estimation

**Current Implementation**: **Sample Estimator** estimates all parameters from historical data.


**How it works**: 
- **μ**: Sample mean of returns
- **B**: Factor loadings from Step 2
- **F**: Covariance of factor returns
- **D**: Residual variances from factor regression
- **Σ**: Full covariance = B * F * B' + D

**Alternatives**: Bayesian estimators, Shrinkage estimators (extensible)

### Step 5: Evaluation

**Current Implementation**: Evaluates portfolio performance and calculates optimal weights.

**How it works**: Optimizer uses estimated parameters (μ, Σ) and objective to find optimal weights, then evaluates expected performance.

**Metrics**: Annualized return/volatility, Sharpe/Sortino/Calmar ratios, max drawdown, win rate, skewness, kurtosis

```

## Options Hedging Module（NEW!)

The system includes an optional **Options Hedging** module (`workflow/6_options_hedging/`) that extends portfolio optimization with sophisticated risk management capabilities using options.

### Purpose of Options
By using the options hedging module, you can:

1. **Eliminate Directional Risk**: Delta-neutral hedging (Delta = 0) removes price direction bias
2. **Control Convexity**: Gamma hedging ensures stable portfolio behavior across price ranges
3. **Manage Multiple Risks**: Multi-Greeks hedging strategy addresses all risk dimensions simultaneously
4. **Extract Pure Volatility Exposure**: Trade volatility independently of price direction
5. **Reduce Portfolio Volatility**: Options can reduce overall portfolio risk while maintaining returns

The output will include:
- **Absolute returns**: Total return, annualized return, final portfolio value
- **Forecasted volatility**: Predicted volatility from portfolio risk structure
- **All Greeks**: Complete risk sensitivity metrics
- **Hedging solution**: Optimal hedging positions and values




### Understanding the Greeks

The system calculates all five option Greeks to help you understand and manage risk:

#### Δ (Delta) - Price Sensitivity
- **What it measures**: How much the option price changes for a $1 change in the underlying asset price
- **Range**: Call options (0 to 1), Put options (-1 to 0)
- **Purpose**: Controls **directional exposure**
- **Hedging goal**: Delta = 0 creates a delta-neutral portfolio with no directional risk

#### Γ (Gamma) - Convexity Management
- **What it measures**: The rate of change of Delta as the underlying price moves
- **Purpose**: Manages **convexity (nonlinearity)** in the portfolio
- **Hedging goal**: Control convexity to manage hedging frequency and transaction costs
- **Impact**: High gamma requires frequent rebalancing for delta neutrality

#### Θ (Theta) - Time Decay
- **What it measures**: How much option value decreases per day as expiration approaches
- **Purpose**: Captures **time value erosion**
- **Hedging goal**: Understand the time cost of holding options
- **Strategy**: Balance time decay against other Greeks when constructing hedges

#### V (Vega) - Volatility Sensitivity
- **What it measures**: How much the option price changes for a 1% change in volatility
- **Purpose**: **Betting on volatility direction** or hedging volatility risk
- **Hedging goal**: Explicitly trade volatility expectations
- **Application**: Useful for volatility trading strategies

#### ρ (Rho) - Interest Rate Sensitivity
- **What it measures**: Sensitivity to changes in the risk-free interest rate
- **Purpose**: Manages **interest rate exposure**
- **Hedging goal**: Control interest rate risk in long-term options
- **Relevance**: More important for longer-dated options



### Module Structure

The options hedging module (`workflow/6_options_hedging/`) contains:

- **`volatility_forecast.py`**: Volatility prediction from portfolio or time series
- **`option_pricing.py`**: Black-Scholes and extensible pricing models
- **`greeks_calculator.py`**: Calculation of all option Greeks
- **`hedging_strategy.py`**: Delta and multi-Greeks hedging strategies
- **`example_usage.py`**: Complete usage examples

For detailed documentation, see `workflow/6_options_hedging/README.md`.

## Future Extensions

The system is designed to be extensible. Potential enhancements include:

- **Factor Mining**: Fama-French factors, dynamic factor selection, nonlinear extraction (autoencoders, ICA)
- **Correlation Matrix**: Dynamic correlations (DCC-GARCH), copula models, sparse correlation estimation
- **Objectives**: Multi-objective optimization, robust optimization, regime-aware objectives, transaction cost integration
- **Parameter Estimation**: Bayesian estimation, shrinkage methods (Ledoit-Wolf), time-varying parameters (Kalman filter), ML-based return prediction
- **Evaluation**: Out-of-sample backtesting, Monte Carlo simulation, risk/performance attribution, realistic transaction cost modeling
- **Options Hedging**: Volatility surface modeling, dynamic hedging and exploring multi-agent options pricing(https://github.com/kevinlmf/Options_Pricing)
- **Additional Features**: Multi-period and cross-regional portfolio optimization under Kondratieff Cycle regimes

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

