# Factor Analysis System Guide

## Overview

The Factor Analysis System is a comprehensive framework that integrates advanced factor models with portfolio optimization. It combines:

1. **Fama-French Factor Models** (3-factor, 5-factor, Carhart 4-factor)
2. **Statistical Factor Extraction** (PCA-based)
3. **Factor Timing Strategies** (Momentum-based rotation)
4. **Factor-Tilted Portfolio Construction**
5. **Factor Exposure Optimization**
6. **Risk and Return Attribution**

### Integration with 133-Factor Library

The system seamlessly connects with the Factor Mining System located in `/Quant/Factors`, which provides:
- **49 Technical Factors**: Momentum, volatility, trend, volume indicators
- **27 Fundamental Factors**: Valuation, quality, growth metrics
- **20 Macro Factors**: Interest rates, market sentiment, economic indicators
- **15 ML Factors**: Predictions and feature engineering
- **22 Beta Factors**: Market beta, downside beta, style betas

---

## Architecture

```
Portfolio_Optimization_system/
├── strategy/
│   ├── factor_analyzer.py           # Core factor analysis engine
│   │   ├── FactorAnalyzer           # Main analysis class
│   │   │   ├── construct_market_factors()
│   │   │   ├── extract_pca_factors()
│   │   │   ├── run_factor_regression()
│   │   │   ├── calculate_factor_exposures()
│   │   │   ├── attribute_returns()
│   │   │   └── calculate_factor_risk()
│   │   └── FactorTimingAnalyzer     # Factor timing
│   │       ├── analyze_factor_momentum()
│   │       ├── calculate_factor_correlations()
│   │       └── identify_factor_regimes()
│   │
│   └── integrated_factor_system.py  # Integration layer
│       └── IntegratedFactorSystem   # Unified interface
│           ├── load_factor_library()
│           ├── construct_style_factors()
│           ├── extract_statistical_factors()
│           ├── analyze_portfolio_factors()
│           ├── optimize_factor_exposures()
│           ├── create_factor_tilted_portfolio()
│           └── run_factor_timing()
│
└── scripts/
    └── demo_factor_analysis.py      # Comprehensive demo
```

---

## Key Features

### 1. Style Factor Construction (Fama-French)

Construct market-based style factors:

**MKT (Market Factor)**
- Market return minus risk-free rate
- Systematic market risk

**SMB (Size Factor)**
- Small Minus Big
- Small cap vs large cap premium
- Uses volatility as size proxy

**HML (Value Factor)**
- High Minus Low (Book-to-Market)
- Value vs growth premium
- Uses price momentum as proxy

**MOM (Momentum Factor)**
- Winner Minus Loser
- Past return continuation

**RMW (Quality Factor)**
- Robust Minus Weak
- Profitability factor
- Uses Sharpe ratio as quality proxy

**CMA (Investment Factor)**
- Conservative Minus Aggressive
- Investment style factor
- Uses volatility trends as proxy

```python
factors = factor_system.construct_style_factors(
    market_proxy=spy_returns,
    include_size=True,
    include_value=True,
    include_momentum=True,
    include_quality=True,
    include_investment=True
)
```

### 2. Statistical Factor Extraction (PCA)

Extract principal components from return covariance:

```python
factor_returns, loadings = factor_system.extract_statistical_factors(
    n_factors=5,
    standardize=True
)

# View explained variance
print(factor_system.factor_analyzer.pca_model.explained_variance_ratio_)
```

### 3. Factor Regression Analysis

Run factor regressions on portfolio returns:

**Fama-French 3-Factor Model**
```
R_p - R_f = α + β_MKT(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + ε
```

**Carhart 4-Factor Model**
```
R_p - R_f = α + β_MKT(MKT) + β_SMB(SMB) + β_HML(HML) + β_MOM(MOM) + ε
```

**Fama-French 5-Factor Model**
```
R_p - R_f = α + β_MKT(MKT) + β_SMB(SMB) + β_HML(HML) + β_RMW(RMW) + β_CMA(CMA) + ε
```

```python
results = factor_system.factor_analyzer.run_factor_regression(
    portfolio_returns=portfolio_returns,
    model='carhart_4'
)

print(f"Alpha: {results['alpha_annualized']:.4f}")
print(f"R²: {results['r_squared']:.4f}")
print(f"Factor Betas: {results['betas']}")
```

### 4. Factor Exposure Analysis

Calculate portfolio's exposures to each factor:

```python
exposure_df = factor_system.factor_analyzer.calculate_factor_exposures(
    portfolio_weights=weights,
    factors=factors
)

# Results show exposure and significance for each factor
```

### 5. Return Attribution

Decompose returns into factor contributions:

```python
attribution = factor_system.factor_analyzer.attribute_returns(
    portfolio_returns=portfolio_returns,
    model='carhart_4'
)

print(f"Total Return: {attribution['total_return']:.2%}")
print(f"Explained by Factors: {attribution['explained_return']:.2%}")
print(f"Contributions: {attribution['contributions']}")
```

Example output:
```
Total Return: 15.88%
Explained by Factors: 17.70% (82.6%)
Factor Contributions:
  MKT     :   7.11% ( 40.2%)
  SMB     :   2.07% ( 11.7%)
  HML     :   0.27% (  1.5%)
  MOM     :   5.12% ( 29.0%)
  Alpha   :   3.13% ( 17.7%)
  Residual:  -0.00% ( -0.0%)
```

### 6. Factor Risk Decomposition

Break down portfolio risk into systematic and idiosyncratic components:

```python
risk_results = factor_system.factor_analyzer.calculate_factor_risk(
    portfolio_weights=weights,
    factors=factors
)

print(f"Total Volatility: {risk_results['total_volatility']:.2%}")
print(f"Factor Volatility: {risk_results['factor_volatility']:.2%}")
print(f"Idiosyncratic Volatility: {risk_results['idiosyncratic_volatility']:.2%}")
```

### 7. Factor Timing

Identify momentum in factor returns:

```python
timing_results = factor_system.run_factor_timing(
    lookback=60,
    top_n_factors=3
)

momentum_df = timing_results['momentum']
top_factors = timing_results['top_factors']
```

### 8. Factor-Tilted Portfolio Construction

Create portfolios with tilts toward specific factors:

```python
weights = factor_system.create_factor_tilted_portfolio(
    tilt_factors=['MKT', 'MOM', 'RMW'],
    tilt_strength=1.0,  # 0=no tilt, 1=moderate, 2=strong
    base_method='risk_parity'  # or 'max_sharpe', 'min_variance'
)
```

**Tilt Strength Guide:**
- `0.0`: No tilt, pure base optimization
- `0.5`: Slight tilt toward factors
- `1.0`: Moderate tilt (recommended)
- `2.0`: Strong tilt (may sacrifice diversification)

### 9. Factor Exposure Optimization

Optimize to achieve target factor exposures:

```python
target_exposures = {
    'MKT': 0.8,   # 80% market exposure
    'SMB': 0.2,   # 20% small cap tilt
    'MOM': 0.3,   # 30% momentum exposure
    'HML': -0.1,  # -10% growth tilt
}

weights = factor_system.optimize_factor_exposures(
    target_exposures=target_exposures,
    max_weight=0.25,
    min_weight=0.0
)
```

---

## Complete Workflow Example

```python
from data.multi_asset_fetcher import MultiAssetFetcher, create_example_portfolio
from strategy.integrated_factor_system import IntegratedFactorSystem

# 1. Load Data
fetcher = MultiAssetFetcher(start_date='2020-01-01', end_date='2024-01-01')
tickers = create_example_portfolio('all_assets')
prices, returns = fetcher.fetch_assets(tickers)

# 2. Initialize Factor System
factor_system = IntegratedFactorSystem(
    returns=returns,
    prices=prices,
    risk_free_rate=0.03
)

# 3. Construct Style Factors
factors = factor_system.construct_style_factors(
    market_proxy=returns['SPY'] if 'SPY' in returns.columns else None
)

# 4. Extract Statistical Factors (Optional)
pca_factors, loadings = factor_system.extract_statistical_factors(n_factors=5)

# 5. Run Factor Timing
timing_results = factor_system.run_factor_timing(lookback=60, top_n_factors=3)
top_factors = timing_results['top_factors']

# 6. Create Factor-Tilted Portfolio
weights = factor_system.create_factor_tilted_portfolio(
    tilt_factors=top_factors,
    tilt_strength=1.0,
    base_method='risk_parity'
)

# 7. Analyze Portfolio
portfolio_returns = (returns * weights).sum(axis=1)
report = factor_system.analyze_portfolio_factors(
    portfolio_returns=portfolio_returns,
    portfolio_weights=weights,
    models=['fama_french_3', 'carhart_4', 'fama_french_5']
)

# 8. Generate Comprehensive Report
factor_system.generate_comprehensive_report(
    portfolio_returns=portfolio_returns,
    portfolio_weights=weights,
    save_path='results/factor_analysis'
)
```

---

## Output Files

After running the demo script, check `results/factor_analysis/`:

```
results/factor_analysis/
├── strategy_comparison.csv              # Performance metrics for all strategies
├── comprehensive_factor_analysis.png    # Visualization dashboard
└── integrated_factor_report.txt         # Detailed factor analysis report
```

---

## Performance Metrics

The demo script compares multiple strategies:

| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Active Assets |
|----------|--------------|--------------|--------------|---------------|
| Equal Weight | 11.95% | 0.84 | -24.51% | 22 |
| Factor Tilt (Risk Parity) | 15.88% | 0.65 | -33.65% | 4 |
| Exposure Optimized | 13.68% | 0.68 | -29.25% | 7 |

---

## Advanced Topics

### Custom Factor Construction

Create your own factors by subclassing:

```python
class CustomFactorAnalyzer(FactorAnalyzer):
    def construct_custom_factor(self, params):
        # Your custom factor logic
        pass
```

### Multi-Period Analysis

Analyze factor performance over different periods:

```python
# Split data into periods
periods = {
    'Pre-COVID': returns.loc['2020-01-01':'2020-03-01'],
    'COVID': returns.loc['2020-03-01':'2021-01-01'],
    'Post-COVID': returns.loc['2021-01-01':'2024-01-01']
}

for name, period_returns in periods.items():
    # Run analysis for each period
    pass
```

### Factor Regime Identification

Identify different market factor regimes:

```python
timing_analyzer = FactorTimingAnalyzer(factors)
regime_info = timing_analyzer.identify_factor_regimes(n_regimes=3)
```

---

## Best Practices

### 1. Factor Selection
- Start with 3-factor model for simplicity
- Add factors incrementally (4-factor, 5-factor)
- Monitor R² improvement to assess value-add

### 2. Rebalancing Frequency
- **Monthly**: Good for most strategies
- **Quarterly**: For lower turnover
- **Weekly**: For high-frequency factor timing

### 3. Exposure Constraints
- Keep absolute exposures < 1.5 for stability
- Monitor correlation between target factors
- Use moderate tilt strength (1.0) initially

### 4. Risk Management
- Track idiosyncratic risk (keep < 30% of total)
- Monitor factor concentration
- Use stop-loss on factor momentum signals

### 5. Backtesting
- Test across multiple market regimes
- Account for transaction costs
- Validate with out-of-sample data

---

## References

1. **Fama, E., & French, K. (1993)**. "Common risk factors in the returns on stocks and bonds." Journal of Financial Economics.

2. **Carhart, M. (1997)**. "On Persistence in Mutual Fund Performance." Journal of Finance.

3. **Fama, E., & French, K. (2015)**. "A five-factor asset pricing model." Journal of Financial Economics.

4. **Jegadeesh, N., & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers." Journal of Finance.

---

## Troubleshooting

### Issue: Low R² in factor regressions
- **Solution**: Check data quality, increase lookback period, try different factors

### Issue: Factor timing shows inconsistent signals
- **Solution**: Increase lookback period, smooth with moving averages, use ensemble signals

### Issue: Factor-tilted portfolio has high concentration
- **Solution**: Reduce tilt_strength, add diversification constraints, use multiple tilt factors

### Issue: Attribution shows negative alpha
- **Solution**: Review factor exposures, consider different base optimization, adjust rebalancing frequency

---

## Support

For questions or issues:
1. Check demo script: `scripts/demo_factor_analysis.py`
2. Review test results in `results/factor_analysis/`
3. Consult factor analyzer source: `strategy/factor_analyzer.py`

---

**Note**: This factor analysis system is for educational and research purposes. Past performance does not guarantee future results. Always validate strategies with proper backtesting before live deployment.
