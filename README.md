# Factor-Based Portfolio Optimization Platform

**Multi-Layer Portfolio System** powered by Hidden Markov Models and Factor Analysis, combining regime-adaptive optimization, stock selection, return forecasting, and Greek-based hedging to deliver robust portfolio construction for quantitative finance.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)

## Installation

```bash
# Clone the repository
git clone https://github.com/kevinlmf/Portfolio_Optimization_system
cd Portfolio_Optimization_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Interactive Menu

Launch the interactive interface:

```bash
bash run.sh
```

### Run NEW Workflow (Recommended)

Run the complete new workflow with stock selection and forecasting:

```bash
./run.sh workflow-new              # 15 stocks, 5 factors, 21-day forecast
./run.sh workflow-new 20 8 42      # 20 stocks, 8 factors, 42-day forecast
```

### Run NEW Workflow with Options Hedging

Complete workflow including Greek-based hedging:

```bash
./run.sh workflow-new-full
```

### Quick Test

Test the new workflow components:

```bash
./run.sh test
```

### Legacy Commands

```bash
./run.sh workflow              # Legacy optimization workflow
./run.sh workflow-regime       # Legacy regime-aware workflow
./run.sh workflow-options      # Legacy with options hedging
```

## Unified Generative Model

The financial world can be written as a hierarchical generative process:

$$
\begin{aligned}
s_t &\sim \text{Markov}(P) & \text{(market regime)} \\
F_t | s_t &\sim \mathcal{D}_F(s_t) & \text{(factor dynamics)} \\
r_t | F_t, s_t &= B(s_t)F_t + \varepsilon_t(s_t) & \text{(return model)} \\
\varepsilon_t(s_t) &\sim \mathcal{N}(0, \Sigma_\varepsilon(s_t)) & \text{(idiosyncratic risk)}
\end{aligned}
$$

| Layer | Object | Description |
|-------|--------|-------------|
| Hidden State | HMM / Regime | Market state (Bull/Bear/Crisis) |
| Intermediate Structure | Factor | Common factors driving returns |
| Selection | Stock Selection | Filter assets by factor exposure |
| Forecasting | μ̂, Σ̂ Prediction | Predict future returns & covariance |
| Decision | Goal + Optimization | Objectives and constraints |

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Step 0: Regime Detection (市场状态检测)                      │
│  Input: Historical Returns r_t                               │
│  Output: Regime State s_t, Transition Matrix P               │
│  Method: Hidden Markov Model (HMM)                           │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Step 1: Factor Mining (因子挖掘)                             │
│  Input: Asset Returns (T × N), Regime State s_t              │
│  Output: Factor Returns F_t (T × K)                          │
│  Methods: PCA, Factor Analysis, Regime-Conditional           │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Step 2: Stock Selection (股票筛选) ★NEW                      │
│  Input: Factor Loadings B, Factor Returns F_t, Regime s_t    │
│  Output: Selected Stocks (N_selected assets)                 │
│  Methods: Factor Exposure Scoring, Liquidity, Sector Filters │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Step 3: Forecasting (收益与协方差预测) ★NEW                  │
│  Input: Selected Stock Returns, Factor Returns               │
│  Output: Forecasted μ̂, Σ̂, Confidence Score                   │
│  Methods: Factor Model (B·F̂), Momentum, Shrinkage            │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Step 4: Select Objective (选择优化目标)                      │
│  Objectives: SHARPE | CVaR | RISK_PARITY | MIN_VARIANCE      │
│  Constraints: Long-only, Leverage, Max/Min Weight            │
│  Methods: Adaptive based on Forecast Confidence & Regime     │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Step 5: Optimization (组合优化)                              │
│  Input: Forecasted μ̂, Σ̂, Objective, Constraints              │
│  Output: Optimal Weights w* - 最优组合权重                    │
│  Strategies: Expected, Robust (Minimax), Adaptive            │
└──────────┬───────────────────────────────────────────────────┘
           │ (Optimal Weights w*)
           ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 6: Options Hedging (期权对冲)                           │
│  Input: Optimal Weights w*, Forecasted Σ̂                     │
│  Output: Hedge Positions & Risk-Neutralized Portfolio        │
│  Methods: Volatility Forecast, Greeks Calc, Delta/Gamma Hedge│
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Portfolio_Optimization_system/
├── regime_layer/                # Step 0: Regime Detection (HMM)
│   ├── regime_detector.py      # HMM-based regime detection
│   ├── regime_knowledge_base.py # State-dependent knowledge storage
│   ├── regime_estimator.py     # Regime parameter estimation
│   └── regime_optimizer.py     # Regime-aware optimization
├── data/                        # Data acquisition and API interfaces
│   ├── api_client.py           # Financial data API client
│   └── multi_asset_fetcher.py  # Multi-asset data retrieval
├── factor_layer/                # Factor analysis modules
│   ├── factor_mining/          # PCA, Factor Analysis
│   ├── factor_regression/      # Factor loadings estimation
│   └── factor_risk_models/     # Factor-based risk models
├── portfolio_layer/             # Portfolio construction
│   ├── build_matrix/           # Correlation matrix construction
│   ├── constraints/            # Constraint definitions
│   ├── objectives/             # Objective functions
│   ├── optimizer/              # Optimization algorithms
│   ├── parameter_estimation/   # μ, Σ estimation
│   └── solvers/                # QP and other solvers
├── workflow/                    # Workflow modules
│   ├── 1_factor_mining/        # Step 1: Factor mining
│   ├── 2_stock_selection/      # Step 2: Stock selection ★NEW
│   │   └── stock_selector.py   # Factor-based stock scoring
│   ├── 3_forecasting/          # Step 3: Forecasting ★NEW
│   │   ├── return_forecaster.py    # Return prediction
│   │   ├── covariance_forecaster.py # Covariance prediction
│   │   └── ensemble_forecaster.py  # Ensemble methods
│   ├── 4_select_objective/     # Step 4: Select objectives
│   ├── 5_optimization/         # Step 5: Portfolio optimization
│   └── 6_options_hedging/      # Step 6: Options hedging
│       ├── volatility_forecast.py  # Volatility prediction
│       ├── option_pricing.py       # Black-Scholes, Binomial Tree
│       ├── greeks_calculator.py    # Greeks computation
│       └── hedging_strategy.py     # Hedging strategies
├── risk_layer/                  # Risk management
│   └── backtesting/            # Backtesting framework
├── result/                      # Output directory
├── workflow.py                  # Main workflow script
├── test_new_workflow.py         # Test script
└── run.sh                       # CLI runner
```

## Complete Example: NEW Workflow (2022-2024)

Below is a complete walkthrough of the **NEW** optimization pipeline with stock selection and forecasting.

### Data Setup

```python
import pandas as pd
import numpy as np
from workflow import PortfolioWorkflow

# 20-stock universe across sectors
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'XOM',
          'BAC', 'GS', 'UNH', 'HD', 'V', 'MA', 'DIS', 'NFLX', 'ADBE', 'CRM']

# Period: 500 trading days
# Target: Select 15 stocks, forecast 21 days
returns = fetch_returns(assets, n_days=500)  # Shape: (500, 20)
```

---

### Step 0: Regime Detection (HMM)

**Goal**: Identify hidden market states (Bull/Bear) from observed returns.

**Mathematical Model**:

$$
\begin{aligned}
s_t &\sim \text{Markov}(P), \quad s_t \in \{0, 1\} \quad \text{(Bull=0, Bear=1)} \\
r_t | s_t = k &\sim \mathcal{N}(\mu_k, \Sigma_k)
\end{aligned}
$$

**Algorithm**: Baum-Welch (EM) + Viterbi Decoding

```
E-Step (Forward-Backward):
  α_t(k) = P(r_{1:t}, s_t=k)           # Forward probability
  β_t(k) = P(r_{t+1:T} | s_t=k)        # Backward probability
  γ_t(k) = P(s_t=k | r_{1:T})          # State posterior
  
M-Step (Parameter Update):
  μ̂_k = Σ_t γ_t(k) r_t / Σ_t γ_t(k)
  Σ̂_k = Σ_t γ_t(k) (r_t - μ̂_k)(r_t - μ̂_k)' / Σ_t γ_t(k)
  P̂_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)    # Transition matrix
```

**Output**:

```
Training HMM with 2 regimes...

✓ Current regime: 1 (Bear)
✓ Regime probabilities: [0.030, 0.970]
✓ Regime distribution: [279, 221]

✓ Transition matrix:
                Bull      Bear
      Bull    63.16%    36.84%
      Bear    46.08%    53.92%

✓ Regime statistics:
  Bull: Return=2.21%, Vol=24.00%, Sharpe=0.09
  Bear: Return=23.84%, Vol=23.41%, Sharpe=1.02
```

---

### Step 1: Factor Mining (Regime-Conditional PCA)

**Goal**: Extract K common factors **conditional on current regime**.

**Mathematical Model**:

$$
\text{Cov}(r | s_t) = \Sigma(s_t) = V(s_t) \Lambda(s_t) V(s_t)^T
$$

**Algorithm**: SVD on regime-filtered data

```
Input: R ∈ ℝ^(T×N), Regime sequence s_{1:T}

1. Filter data by current regime:
   R_s = {r_t : s_t = s_current}

2. Compute SVD: R_s = U S V^T

3. Factor returns: F = R @ V[:, :K]  ∈ ℝ^(T×K)

4. Explained variance per factor
```

**Output** (K=5 factors, Bear regime):

```
✓ Extracted 5 factors
✓ Explained variance: 31.32%
  Factor 1: 6.76%
  Factor 2: 6.38%
  Factor 3: 6.28%
  Factor 4: 6.03%
  Factor 5: 5.87%
```

---

### Step 2: Stock Selection (Factor-Based Scoring) ★NEW

**Goal**: Select top N stocks based on factor exposure scores.

**Mathematical Model**:

$$
\text{Score}_i = \sum_{k=1}^{K} w_k \cdot \beta_{i,k}
$$

where $w_k$ is the factor importance weight (based on explained variance) and $\beta_{i,k}$ is stock $i$'s loading on factor $k$.

**Algorithm**:

```
1. Compute factor loadings: B = (F^T F)^{-1} F^T R

2. Compute factor weights (by explained variance):
   w_k = λ_k / Σ_j λ_j

3. Score each stock:
   Score_i = Σ_k w_k × |β_{i,k}|

4. Apply regime-specific adjustments:
   - Bull: Favor high-beta stocks
   - Bear: Favor defensive stocks (lower loading on volatile factors)

5. Apply filters (liquidity, sector constraints)

6. Select top N stocks by score
```

**Output** (Select 15 from 20):

```
[Stock Selection - BEAR Market]
  Computed factor loadings: (20, 5)
  Factor weights: {Factor_1: 0.438, Factor_2: 0.219, Factor_3: 0.146, ...}
  Score range: [-1.025, 0.943]

✓ Selected 15 stocks from 20 universe

Top 10 by Score:
  V       : Score=+0.943, Top Factor=Factor_3(+0.01)
  GS      : Score=+0.604, Top Factor=Factor_1(+0.01)
  NFLX    : Score=+0.544, Top Factor=Factor_3(+0.00)
  GOOGL   : Score=+0.527, Top Factor=Factor_1(+0.01)
  AAPL    : Score=+0.491, Top Factor=Factor_5(+0.01)
  ADBE    : Score=+0.373, Top Factor=Factor_3(+0.01)
  XOM     : Score=+0.291, Top Factor=Factor_4(+0.00)
  MSFT    : Score=+0.253, Top Factor=Factor_1(+0.00)
  PG      : Score=+0.246, Top Factor=Factor_2(+0.01)
  CRM     : Score=+0.056, Top Factor=Factor_4(-0.00)
```

---

### Step 3: Forecasting (Return & Covariance Prediction) ★NEW

**Goal**: Predict expected returns μ̂ and covariance Σ̂ for selected stocks.

**Mathematical Model**:

$$
\begin{aligned}
\hat{F}_{t+h} &= \text{ARMA}(F_t) \quad \text{or} \quad \text{Momentum}(F_t) \\
\hat{r}_{t+h} &= B \hat{F}_{t+h} + \alpha \\
\hat{\mu} &= \mathbb{E}[\hat{r}_{t+h}] \\
\hat{\Sigma} &= B \hat{\Sigma}_F B^T + \hat{D} \quad \text{(Factor Model)} \\
\hat{\Sigma}_{\text{shrink}} &= \delta \hat{\Sigma}_{\text{sample}} + (1-\delta) \text{diag}(\hat{\Sigma}_{\text{sample}})
\end{aligned}
$$

**Ensemble Methods**:
- **Return**: Factor model + Momentum
- **Covariance**: Factor model + Ledoit-Wolf shrinkage

**Confidence Score**:

$$
\text{Confidence} = \frac{\text{Consistency}(\hat{\mu}_{\text{factor}}, \hat{\mu}_{\text{momentum}})}{2} + 0.5
$$

**Output** (Horizon: 21 days):

```
Fitting forecasting models...
  Return methods: ['factor', 'momentum']
  Covariance methods: ['factor', 'shrinkage']

✓ Factor forecast (next 21 days):
  Factor 1: -0.00%
  Factor 2: 0.00%
  Factor 3: -0.00%
  Factor 4: 0.00%
  Factor 5: 0.00%

✓ Forecast generated for 15 assets
✓ Horizon: 21 days
✓ Confidence: 71.0%

Expected Returns (annualized):
  V       : μ=-16.1%, σ=23.2%, SR=-0.69
  GS      : μ=-3.3%, σ=23.6%, SR=-0.14
  NFLX    : μ=+5.9%, σ=26.0%, SR=0.23
  GOOGL   : μ=+2.9%, σ=22.9%, SR=0.12
  AAPL    : μ=+22.2%, σ=25.4%, SR=0.88
  ADBE    : μ=-11.1%, σ=24.8%, SR=-0.45
  XOM     : μ=-11.0%, σ=21.9%, SR=-0.50
  MSFT    : μ=-15.1%, σ=22.5%, SR=-0.67
  PG      : μ=+8.0%, σ=24.1%, SR=0.33
  CRM     : μ=+7.2%, σ=23.0%, SR=0.31
```

---

### Step 4: Select Objective (Adaptive)

**Goal**: Choose optimization objective based on regime and forecast confidence.

**Adaptive Selection Logic**:

| Regime | Confidence | Objective |
|--------|------------|-----------|
| Bull   | High (>70%) | SHARPE |
| Bull   | Low (<70%) | RISK_PARITY |
| Bear   | High (>70%) | MIN_VARIANCE |
| Bear   | Low (<70%) | MIN_VARIANCE |

**Objectives**:

$$
\begin{aligned}
\text{Sharpe:} \quad & \max_w \frac{w^T \hat{\mu}}{\sqrt{w^T \hat{\Sigma} w}} \\
\text{Min Variance:} \quad & \min_w \quad w^T \hat{\Sigma} w \\
\text{Risk Parity:} \quad & \min_w \sum_{i,j} \left( \frac{w_i (\hat{\Sigma} w)_i}{\sum_k w_k (\hat{\Sigma} w)_k} - \frac{1}{N} \right)^2 \\
\text{CVaR:} \quad & \min_w \quad \text{VaR}_\alpha + \frac{1}{1-\alpha} \mathbb{E}[(L - \text{VaR}_\alpha)^+]
\end{aligned}
$$

**Output** (Bear regime, 71% confidence → MIN_VARIANCE):

```
✓ Objective: min_variance
✓ Method: qp
✓ Constraints: {'long_only': True, 'max_weight': 0.15}

  [Auto-selected based on regime and forecast confidence]
```

---

### Step 5: Optimization

**Goal**: Solve for optimal weights w*.

**QP Formulation** (Min Variance):

$$
\begin{aligned}
\min_w \quad & \frac{1}{2} w^T \hat{\Sigma} w \\
\text{s.t.} \quad & \mathbf{1}^T w = 1 \\
& 0 \leq w_i \leq 0.15
\end{aligned}
$$

**Algorithm**: Quadratic Programming (cvxopt/scipy)

```
1. Build objective: P = Σ̂, q = 0
2. Add constraints:
   - Equality: Σ w_i = 1
   - Inequality: 0 ≤ w_i ≤ 0.15
3. Solve: w* = QP(P, q, G, h, A, b)
```

**Output**:

```
[Standard Optimization]

✓ Portfolio optimized for 15 assets

  Portfolio Metrics (annualized):
    Expected Return: 0.30%
    Volatility: 6.75%
    Sharpe Ratio: 0.04

  Top Holdings:
    MA: 6.67%
    AMZN: 6.67%
    HD: 6.67%
    DIS: 6.67%
    META: 6.67%
```

---

### Step 6: Options Hedging

**Goal**: Hedge portfolio risk using options and Greek neutralization.

**6.1 Volatility Forecasting**:

$$
\sigma_{\text{portfolio}} = \sqrt{w^{*T} \hat{\Sigma} w^*} = 6.75\% \text{ (annualized)}
$$

**6.2 Greek Calculation** (Black-Scholes):

For a protective put (S=100, K=95, T=0.25, r=5%):

$$
\begin{aligned}
d_1 &= \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \\
d_2 &= d_1 - \sigma\sqrt{T}
\end{aligned}
$$

$$
\begin{aligned}
\Delta &= N(d_1) - 1 \quad \text{(put delta)} \\
\Gamma &= \frac{N'(d_1)}{S\sigma\sqrt{T}} \\
\Theta &= -\frac{S N'(d_1) \sigma}{2\sqrt{T}} + rK e^{-rT} N(-d_2) \\
\mathcal{V} &= S\sqrt{T} N'(d_1) \\
\rho &= -KT e^{-rT} N(-d_2)
\end{aligned}
$$

**6.3 Delta Hedging**:

$$
\text{Hedge Quantity} = -\Delta \times \text{Portfolio Value}
$$

**Output**:

```
6.1: Computing portfolio volatility...
✓ Portfolio volatility: 6.75% (annualized)

6.2: Calculating Greeks...
✓ Greeks calculated:
  Δ (Delta): -0.0282
  Γ (Gamma): 0.0192
  Θ (Theta): -0.0008
  ν (Vega): 0.0323
  ρ (Rho): -0.0071

6.3: Computing hedge strategy...
✓ Hedge solution (protective_put):
  Strike: $95.00 (95% of spot)
  Hedge quantity: 2.82 shares
  Hedge value: $281.90
```

---

## Complete Workflow Summary

```
================================================================================
WORKFLOW COMPLETE
================================================================================

Step 0: Regime Detection  → Bear market (97% probability)
Step 1: Factor Mining     → 5 factors, 31.32% explained variance
Step 2: Stock Selection   → 15 stocks from 20 universe (factor-based)
Step 3: Forecasting       → 21-day horizon, 71% confidence
Step 4: Select Objective  → min_variance (auto-selected for Bear + high confidence)
Step 5: Optimization      → Portfolio: 6.75% vol, 0.30% return
Step 6: Options Hedging   → Protective put at 95% strike

Final Portfolio: 15 stocks
Optimization: min_variance
```

## Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
yfinance >= 0.2.0
```

## Future Extensions

The system is designed to be extensible:

- **Factor Mining**: Fama-French factors, dynamic selection, autoencoders, ICA
- **Stock Selection**: Machine learning scoring, fundamental filters, ESG constraints
- **Forecasting**: LSTM, Transformer, DCC-GARCH, Bayesian methods
- **Correlation Matrix**: Copula models, sparse estimation
- **Objectives**: Multi-objective optimization, transaction cost integration
- **Parameter Estimation**: Kalman filter, regime-switching GARCH
- **Options Hedging**: Volatility surface modeling, Heston/SABR models, dynamic hedging
- **Advanced**: Multi-period optimization, cross-regional portfolios, Kondratieff Cycle regimes

## Research & Innovation

- **Unified Generative Model**: Hierarchical probabilistic framework for financial markets
- **Regime-Adaptive Systems**: HMM-based automatic market state detection
- **Factor-Based Stock Selection**: Regime-conditional factor scoring for asset filtering
- **Ensemble Forecasting**: Multi-method return and covariance prediction with confidence
- **Adaptive Objective Selection**: Automatic objective choice based on market conditions
- **Multi-Strategy Optimization**: Expected, robust, adaptive, worst-case strategies
- **Integrated Hedging**: Portfolio-to-volatility-to-Greeks pipeline

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: This project is provided **for educational, academic research, and learning purposes only**. Past performance does not guarantee future results. The authors are not responsible for any financial decisions made based on this software.

---

*May we all find our own alpha — in markets and in life.* 📈
