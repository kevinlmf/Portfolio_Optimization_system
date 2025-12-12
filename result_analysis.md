# Portfolio Optimization Results Analysis

## Executive Summary

This analysis examines the factor structure driving portfolio covariance, with emphasis on statistical and economic interpretation of the covariance matrix's **trace** and **eigenvalues**.

**Analysis Date:** 20251204_141540
**Number of Assets:** 10
**Number of Factors:** 5

---

## 1. Covariance Matrix Overview

### 1.1 Matrix Structure

The portfolio covariance matrix **Σ** is decomposed using the factor model:

```
Σ = B × F × B' + D
```

Where:
- **B** (N × K): Factor loadings matrix - measures asset sensitivity to factors
- **F** (K × K): Factor covariance matrix - captures factor-to-factor relationships
- **D** (N × N): Idiosyncratic risk matrix - asset-specific risk not explained by factors

**Matrix Dimensions:**
- Covariance Matrix (Σ): 10 × 10
- Factor Loadings (B): 10 × 5
- Factor Covariance (F): 5 × 5

---

## 2. Trace Analysis: Statistical & Economic Interpretation

### 2.1 What is the Trace?

The **trace** of the covariance matrix is the sum of its diagonal elements:

```
Trace(Σ) = Σᵢ σᵢᵢ = Σᵢ Var(Rᵢ)
```

**Calculated Trace:** `0.001037`

### 2.2 Statistical Meaning

1. **Total Portfolio Variance**
   - Trace represents the **sum of all individual asset variances**
   - It's the total "variance budget" across all assets
   - Dimension: variance (or standard deviation squared)

2. **Invariance Property**
   - Trace is **invariant under orthogonal transformations**
   - Useful for comparing different factor decompositions
   - Preserves total variance regardless of coordinate system

3. **Variance Decomposition**
   - Trace can be decomposed into factor-driven and idiosyncratic components:
     - **Factor-driven variance:** `0.000565` (54.52%)
     - **Idiosyncratic variance:** `0.000471` (45.48%)

### 2.3 Economic Meaning

1. **Total Risk Exposure**
   - Trace quantifies **total systematic risk** in the portfolio
   - Higher trace → more volatile portfolio
   - Lower trace → more stable portfolio

2. **Diversification Efficiency**
   - **Well-diversified portfolio:** Idiosyncratic variance should be small relative to trace
   - **Poor diversification:** High trace indicates concentrated risk
   - Current ratio: 45.48% idiosyncratic (lower is better for diversification)

3. **Risk Budgeting**
   - Trace serves as a **risk budget constraint**
   - Portfolio managers can allocate this budget across factors
   - Each factor consumes a portion of the total variance budget

---

## 3. Eigenvalue Analysis: Statistical & Economic Interpretation

### 3.1 Eigenvalue Decomposition

The covariance matrix can be decomposed as:

```
Σ = Q × Λ × Q'
```

Where:
- **Λ**: Diagonal matrix of eigenvalues (λ₁ ≥ λ₂ ≥ ... ≥ λₙ)
- **Q**: Matrix of eigenvectors (orthogonal)

### 3.2 Eigenvalues

**Top 5 Eigenvalues:**

1. **λ1** = `0.000187` (18.01% of trace)

2. **λ2** = `0.000163` (15.70% of trace)

3. **λ3** = `0.000153` (14.76% of trace)

4. **λ4** = `0.000149` (14.40% of trace)

5. **λ5** = `0.000142` (13.72% of trace)

**Eigenvalue Summary:**
- **Total variance explained by top 3 eigenvalues:** 48.46%
- **Total variance explained by top 5 eigenvalues:** 76.58%
- **Largest eigenvalue (λ₁):** 0.000187 (18.01% of trace)
- **Smallest eigenvalue (λₙ):** 0.000039 (3.7607% of trace)
- **Condition number (λ₁/λₙ):** 4.79

### 3.3 Statistical Meaning of Eigenvalues

1. **Principal Component Variance**
   - Each eigenvalue λᵢ represents the **variance explained by the i-th principal component**
   - Eigenvalues are sorted: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
   - The sum of all eigenvalues equals the trace: Σᵢ λᵢ = Trace(Σ) = `0.001037`

2. **Dimensionality Reduction**
   - Large eigenvalues indicate **dominant risk factors**
   - Small eigenvalues indicate **redundant dimensions**
   - **Effective dimensionality:** Number of eigenvalues explaining > 95% of variance

3. **Condition Number**
   - Ratio of largest to smallest eigenvalue: `4.79`
   - **Low condition number (< 100):** Well-conditioned, stable matrix
   - **High condition number (> 1000):** Ill-conditioned, sensitive to perturbations
   - **Interpretation:** Well-conditioned matrix - stable risk structure

### 3.4 Economic Meaning of Eigenvalues

1. **Risk Factor Hierarchy**
   - **Largest eigenvalue (λ₁):** Represents the **dominant systematic risk factor**
     - Often corresponds to market-wide movements
     - Explains the most portfolio variance
     - Economic interpretation: "Market factor" or "First principal risk"
   
   - **Subsequent eigenvalues:** Represent **secondary risk factors**
     - Industry-specific risks
     - Style factors (growth vs. value, size, etc.)
     - Sector rotation patterns

2. **Diversification Metrics**
   - **Eigenvalue concentration:** 
     - High concentration (few large eigenvalues) → Low diversification
     - Low concentration (many similar eigenvalues) → High diversification
     - **Concentration index:** 0.6085
     - Values close to 1 indicate high concentration (poor diversification)
   
   - **Effective number of risk sources:**
     - If all eigenvalues were equal: n_eff = n
     - With varying eigenvalues: n_eff = (Σλᵢ)² / Σλᵢ²
     - **Effective dimensions:** 7.72
     - Higher is better for diversification

3. **Risk Decomposition**
   - Each eigenvalue represents a **distinct risk dimension**
   - Portfolio exposure to each dimension is captured by corresponding eigenvector
   - **Risk budgeting:** Allocate portfolio risk across these dimensions

---

## 4. Factor-Driven Covariance Analysis

### 4.1 Factor Importance Ranking

Based on factor explanatory power (contribution to total variance):


**Factor 1** (Factor_1):
- **Variance Explained:** 23.54% of total factor-driven variance
- **Factor Variance (Annualized):** 289.165807
- **Relative Importance:** ★★★★

**Factor 2** (Factor_2):
- **Variance Explained:** 20.54% of total factor-driven variance
- **Factor Variance (Annualized):** 270.898122
- **Relative Importance:** ★★★★

**Factor 3** (Factor_3):
- **Variance Explained:** 19.05% of total factor-driven variance
- **Factor Variance (Annualized):** 261.150896
- **Relative Importance:** ★★★

**Factor 4** (Factor_4):
- **Variance Explained:** 18.96% of total factor-driven variance
- **Factor Variance (Annualized):** 259.741512
- **Relative Importance:** ★★★

**Factor 5** (Factor_5):
- **Variance Explained:** 17.91% of total factor-driven variance
- **Factor Variance (Annualized):** 252.595143
- **Relative Importance:** ★★★


### 4.2 Factor Loadings Analysis

**Factor Loadings Matrix (B) Summary:**
- **Average absolute loading per factor:**
  - Factor 1: 0.0027
  - Factor 2: 0.0030
  - Factor 3: 0.0025
  - Factor 4: 0.0026
  - Factor 5: 0.0028

- **Assets with highest factor sensitivity:**

  **Factor 1:**
    - 6: 0.0066
    - 3: -0.0050
    - 8: -0.0047

  **Factor 2:**
    - 0: 0.0049
    - 2: 0.0047
    - 7: -0.0040

  **Factor 3:**
    - 4: 0.0061
    - 1: 0.0049
    - 2: 0.0045

  **Factor 4:**
    - 0: 0.0056
    - 4: -0.0045
    - 1: 0.0043

  **Factor 5:**
    - 7: 0.0053
    - 8: -0.0046
    - 5: -0.0040

### 4.3 Factor Covariance Structure

The factor covariance matrix **F** captures relationships between factors:

**Factor Correlation Matrix:**

| Factor | Factor_1 | Factor_2 | Factor_3 | Factor_4 | Factor_5 |
|--------|--------|--------|--------|--------|--------|--------|
| Factor_1 | 1.000 | -0.250 | -0.250 | -0.250 | -0.250 |
| Factor_2 | -0.250 | 1.000 | -0.250 | -0.250 | -0.250 |
| Factor_3 | -0.250 | -0.250 | 1.000 | -0.250 | -0.250 |
| Factor_4 | -0.250 | -0.250 | -0.250 | 1.000 | -0.250 |
| Factor_5 | -0.250 | -0.250 | -0.250 | -0.250 | 1.000 |

**Interpretation:**
- Values close to ±1 indicate strong factor relationships
- Values close to 0 indicate independent factors
- **Ideal structure:** Low correlations between factors (orthogonal factors)

---

## 5. Key Insights

### 5.1 Statistical Insights

1. **Variance Decomposition:**
   - Factor-driven variance: 54.52%
   - Idiosyncratic variance: 45.48%
   - **Conclusion:** Mixed structure with significant asset-specific risks

2. **Dimensionality:**
   - Effective dimensions: 7.72
   - Top 3 eigenvalues explain: 48.46% of variance
   - **Conclusion:** Higher-dimensional structure - requires more factors

3. **Conditioning:**
   - Condition number: 4.79
   - **Conclusion:** Matrix is well-conditioned - stable numerical properties

### 5.2 Economic Insights

1. **Risk Structure:**
   - The covariance matrix is driven primarily by multiple factors
   - Factor diversification: 63.1% explained by top 3 factors
   - **Portfolio risk is** well-diversified across factors

2. **Factor Interpretation:**
   - Largest eigenvalue (λ₁ = 0.000187) likely represents: **Market-wide risk**
   - Subsequent eigenvalues represent: **Sector/industry factors, style factors**
   - Idiosyncratic risk accounts for 45.48% - indicates moderate asset-level diversification

3. **Portfolio Implications:**
   - **For risk management:** Focus on hedging the top 3 risk dimensions
   - **For optimization:** Factor model provides good approximation (R² ≈ 54.5%)
   - **For diversification:** Current adding assets that load differently on dominant factors

---

## 6. Mathematical Formulations

### 6.1 Trace Decomposition

```
Trace(Σ) = Trace(B × F × B') + Trace(D)
         = 0.000565 + 0.000471
         = 0.001037
```

### 6.2 Eigenvalue Sum

```
Σᵢ λᵢ = Trace(Σ) = 0.001037
```

### 6.3 Variance Explained by Top k Eigenvalues

```
Variance Explained(k) = (Σᵢ₌₁ᵏ λᵢ) / Trace(Σ) × 100%
```

### 6.4 Factor Contribution

For factor i, its contribution to total variance:

```
Variance_i = Trace(B[:,i] × F[i,i] × B[:,i]')
```

---

## 7. Recommendations

### 7.1 For Portfolio Construction

1. **Factor Exposure Management:**
   - Monitor exposure to Factor 1 (explains 23.5% of factor variance)
   - Diversify across factors with low correlations
   - Rebalance when factor loadings drift significantly

2. **Risk Budgeting:**
   - Allocate risk budget based on eigenvalue proportions
   - Reserve 45.5% for idiosyncratic risk
   - Allocate remaining 54.5% across factor dimensions

### 7.2 For Risk Management

1. **Hedging Strategy:**
   - Hedge the top 3 risk dimensions (largest eigenvalues)
   - Use factor hedging to neutralize systematic exposures
   - Monitor condition number for stability

2. **Stress Testing:**
   - Focus stress scenarios on dominant risk factors
   - Test factor correlation breakdown scenarios
   - Evaluate impact of eigenvalue shifts

---

## Appendix: Data Sources

- **Factors CSV:** `factors_20251204_141540.csv`
- **Factor Loadings:** `matrices_20251204_141540/factor_loadings_B.csv`
- **Factor Covariance:** `matrices_20251204_141540/factor_covariance_F.csv`
- **Full Covariance:** `matrices_20251204_141540/covariance_Sigma.csv`
- **Idiosyncratic Risk:** `matrices_20251204_141540/idiosyncratic_risk_D.csv`

---

**Generated:** 2025-12-04 14:19:42

---

## 8. Regime-Aware Optimization Results (NEW)

This section presents results from **HMM Regime Detection** and **Regime-Aware Optimization**.

### 8.1 Configuration

| Parameter | Value |
|-----------|-------|
| Command | `./run.sh workflow-regime` |
| Regimes | 2 (Bull/Bear) |
| Strategy | robust (Minimax) |
| Data | 500 days, 10 assets |

### 8.2 Step 0: Regime Detection (HMM)

**Transition Matrix:**

|  | → Bull | → Bear |
|--|--------|--------|
| **Bull →** | 88.94% | 11.06% |
| **Bear →** | 6.71% | 93.29% |

**Regime Statistics:**

| Regime | Ann. Return | Ann. Vol | Sharpe |
|--------|-------------|----------|--------|
| Bull | -7.14% | 38.72% | -0.18 |
| Bear | 10.73% | 23.71% | 0.45 |

> Note: Labels may seem counter-intuitive with simulated data. HMM identifies statistical regimes.

### 8.3 Regime-Dependent Parameters

| Regime | Probability | Ann. Return | Ann. Vol |
|--------|-------------|-------------|----------|
| Bull | 95.38% | -8.75% | 39.07% |
| Bear | 4.62% | 11.36% | 23.69% |

### 8.4 Portfolio Performance by Regime

| Regime | Probability | Return | Risk | Sharpe |
|--------|-------------|--------|------|--------|
| Bull | 95.4% | 22.59% | 19.11% | 1.18 |
| Bear | 4.6% | 13.77% | 11.65% | 1.18 |

**Expected Performance:** Return=22.18%, Risk=18.83%, **Sharpe=1.18**

### 8.5 Optimal Weights

| Asset | Weight | Asset | Weight |
|-------|--------|-------|--------|
| PG | 24.74% | NVDA | 5.00% |
| DIS | 22.45% | AAPL | 0.00% |
| MA | 19.11% | JPM | 0.00% |
| GOOGL | 16.68% | JNJ | 0.00% |
| MSFT | 12.02% | V | 0.00% |

### 8.6 Backtest Results

| Metric | Value |
|--------|-------|
| Total Return | **37.37%** |
| Annualized Return | **17.35%** |
| Max Drawdown | **-12.41%** |
| Final Value | **$1.3737** |

### 8.7 Comparison with Standard Optimization

| Method | Sharpe | Max DD | Volatility |
|--------|--------|--------|------------|
| **Regime-Aware (Robust)** | **1.18** | **-12.41%** | **13.58%** |
| Standard | ~0.74 | Higher | Higher |

### 8.8 Key Insights

1. **High state persistence**: Bull→Bull 88.94%, Bear→Bear 93.29%
2. **Robust strategy achieves same Sharpe (1.18) in both regimes**
3. **Concentrated portfolio**: Only 6 assets with non-zero weights
4. **Better risk control**: Lower actual volatility (13.58%) than expected (18.83%)

---

**Run Date:** 2025-12-12 | **Command:** `./run.sh workflow-regime`
