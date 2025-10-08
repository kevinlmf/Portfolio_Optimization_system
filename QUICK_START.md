#  (Quick Start Guide)

##  5

### 1. 
```bash
cd ~/Downloads/System/Quant/Portfolio_Optimization_system
python scripts/quick_test.py
```

****:
```
 ALL TESTS PASSED!
```

---

### 2. 
```bash
python scripts/historical_analysis.py
```

**4**:
- 2008-2010: 
- 2010-2015: 
- 2015-2020: 
- 2020-2025: COVID

****:
- `results/historical_analysis/period_comparison.csv` - 
- `results/historical_analysis/comprehensive_comparison.png` - 

---

### 3. 
```bash
python scripts/comprehensive_portfolio_system.py
```

****:
1. 
2. 
3. 
4. 
5. 
6. 

****:
- `results/comprehensive_backtest/` 

---

##  

### 
- `COMPREHENSIVE_SYSTEM_GUIDE.md` - 
- `HISTORICAL_ANALYSIS_SUMMARY.md` - 
- `README.md` - 

### 
- `scripts/quick_test.py` - 
- `scripts/historical_analysis.py` - 
- `scripts/comprehensive_portfolio_system.py` - 

### 
- `data/multi_asset_fetcher.py` - 
- `strategy/market_regime_detector.py` - 
- `strategy/intelligent_optimizer_selector.py` - 
- `strategy/backtesting_engine.py` - +
- `strategy/sparse_sharpe_optimizer.py` - Sharpe

---

##  

### 1: 
```bash
python scripts/historical_analysis.py
```
: `results/historical_analysis/`

### 2: 
 `scripts/comprehensive_portfolio_system.py`:
```python
#  main() 
custom_tickers = [
    'AAPL', 'MSFT', 'GOOGL',  # 
    'TLT', 'GLD',             # 
    'BTC-USD'                 # 
]

system.setup_portfolio(custom_tickers=custom_tickers)
```

### 3: 
```python
system = ComprehensivePortfolioSystem(
    start_date='2015-01-01',  # 
    end_date='2024-12-31',    # 
    rebalance_frequency='monthly'  #  'weekly', 'quarterly'
)
```

---

##  

2008-202416

|  |  |
|------|--------|
| **** | 12.8% |
| **Sharpe** | 0.677 |
| **** | -20.0% |
| **Alphavs SPY** | 6.6% |
| **Beta** | 0.57 |
| **** | 53.6% |

****: 2020-2025 (17.1%, Sharpe 0.852)

: `HISTORICAL_ANALYSIS_SUMMARY.md`

---

##  

### 
```python
system = ComprehensivePortfolioSystem(
    transaction_cost=0.002  # 0.2% (: 0.1%)
)
```

### 
```python
system = ComprehensivePortfolioSystem(
    risk_free_rate=0.05  # 5% (: 3%)
)
```

### 
```python
system.run_monte_carlo_validation(
    n_simulations=5000,  # : 10000
    n_days=126          # : 252
)
```

---

##  

### 
- : AAPL, MSFT, GOOGL, AMZN, NVDA, META
- : JPM, BAC, GS, V, WFC
- : JNJ, UNH, PFE, ABBV
- : WMT, HD, MCD, PG, KO
- : XOM, CVX
- : CAT, BA, GE

### ETF
- ETF: SPY, QQQ, VTI, IWM, VOO
- ETF: VEA (), VWO ()
- ETF: TLT, IEF, SHY, LQD, HYG, AGG
- ETF: GLD, SLV, USO, DBC
- REIT: VNQ, IYR

### 
- BTC-USD, ETH-USD, SOL-USD, ADA-USD

### 
- : VXX
- : UUP (), FXE (), FXY ()

---

##  



1. ****
2. ****
3. ****
4. ****
5. ****
6. ****

---

##  

### 
```python
# 
tickers = ['AAPL', 'SPY', 'TLT', 'GLD']

# 
start_date='2022-01-01'

# 
n_simulations=1000
```

### 
```python
# 
tickers = create_example_portfolio('all_assets')

# 
start_date='2015-01-01'

# 
n_simulations=10000
```

---

##  

### Q: 
A: 

### Q: 
A: 

### Q: 
A: 60

### Q: 
A: 

---

##  

1. ****: `COMPREHENSIVE_SYSTEM_GUIDE.md`
2. ****: `HISTORICAL_ANALYSIS_SUMMARY.md`
3. **NeurIPS 2024**: `2410.21100v1.pdf` (Sparse Sharpe)
4. **README**: `README.md`

---

##  


1.  `COMPREHENSIVE_SYSTEM_GUIDE.md` 
2.  `python scripts/quick_test.py` 
3.  `results/` 

****:
```bash
python scripts/historical_analysis.py
```

16
