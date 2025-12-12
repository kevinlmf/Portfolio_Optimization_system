# Portfolio Optimization System

A market-regime-aware portfolio optimizer built on multi-factor modeling and risk budgeting.

## Workflow Overview

The system follows a workflow with optional regime detection:

0. **Regime Detection** (Optional) - `regime_layer/` - HMM-based market regime identification
1. **Data** - `data/` - Data acquisition and API integration
2. **Factor Mining** - `workflow/1_factor_mining/` - Extract common factors driving stock returns
3. **Build Matrix** - `workflow/2_build_matrix/` - Construct stock-factor relationship matrices
4. **Select Objective** - `workflow/3_select_objective/` - Choose optimization objectives and constraints
5. **Parameter Estimation** - `workflow/4_estimate_parameters/` - Estimate parameters (μ, F, D, Σ), supports regime-dependent estimation
6. **Evaluation** - `workflow/5_evaluation/` - Evaluate portfolio performance, supports regime-aware optimization
7. **Options Hedging** - `workflow/6_options_hedging/` - Predict volatility from portfolio, calculate Greeks, and hedge risks

## Project Structure

```
Portfolio_Optimization_system/
├── regime_layer/               # Step 0: Regime detection (HMM, optional)
│   ├── regime_detector.py     # HMM-based regime detection
│   ├── regime_knowledge_base.py # State-dependent parameters
│   ├── regime_estimator.py    # Regime parameter estimation
│   ├── regime_optimizer.py    # Regime-aware optimization
│   └── example_usage.py       # Usage examples
├── data/                       # Step 1: Data acquisition and API interfaces
├── workflow/                   # Workflow modules
│   ├── 1_factor_mining/       # Step 2: Factor mining (PCA, Factor Analysis)
│   ├── 2_build_matrix/        # Step 3: Build stock-factor matrices (OLS, Ridge)
│   ├── 3_select_objective/    # Step 4: Select objectives (Sharpe, CVaR, etc.)
│   ├── 4_estimate_parameters/ # Step 5: Parameter estimation (μ, F, D, Σ)
│   ├── 5_evaluation/          # Step 6: Evaluation and backtesting
│   └── 6_options_hedging/     # Step 7: Options hedging and risk management
└── workflow.py                 # Main workflow script
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

```
### Regime Detection (NEW)

The system includes a **Regime Layer** (`regime_layer/`) for HMM-based market regime detection and regime-aware optimization.

### Why Regime Detection?

Traditional optimization assumes constant parameters. Markets have different states (Bull/Bear) with distinct characteristics. Regime-aware optimization:
- **Identifies market states** automatically using Hidden Markov Models
- **Estimates state-dependent parameters**: μ(s), Σ(s), B(s) for each regime
- **Robust optimization**: Considers worst-case scenarios across regimes

## HMM Mathematical Foundation

**Hidden Markov Model (HMM)** assumes the system transitions between unobservable hidden states, and we only observe data generated by these states.

### Model Components

1. **State Transition**: $s_t \sim \text{Markov}(P)$ where $P_{ij} = P(s_{t+1}=j \mid s_t=i)$

2. **Emission Model**: $r_t \mid s_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$ — each regime has its own mean and covariance

3. **Parameter Estimation**: Baum-Welch algorithm (EM)
   - E-Step: Forward-Backward algorithm computes $\gamma_t(k) = P(s_t=k \mid r_1,...,r_T)$
   - M-Step: Update $\hat{\mu}_k$ and $\hat{\Sigma}_k$ using weighted samples

4. **State Decoding**: Viterbi algorithm finds optimal state sequence

### Why HMM for Finance?

| Traditional | HMM-based |
|-------------|-----------|
| Assumes constant parameters | Parameters vary by state |
| Single covariance matrix | Regime-specific covariance |
| Ignores structural changes | Captures regime switches |

### Key Algorithms

| Algorithm | Purpose | Complexity |
|-----------|---------|------------|
| Forward-Backward | State posterior | O(T × K²) |
| Baum-Welch (EM) | Parameter estimation | O(iter × T × K²) |
| Viterbi | Optimal state sequence | O(T × K²) |

---
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

