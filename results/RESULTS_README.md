# Results Directory

This directory contains analysis outputs from the **Portfolio Optimization System**.

---

## 📊 Historical Analysis (Most Important)

### `historical_analysis/`
**16-Year Backtest Results (2008–2024)**

- `period_comparison.csv` – Performance comparison across four major market periods  
- `comprehensive_comparison.png` – Comprehensive visualization of performance metrics  
- `allocation_2008_2010_financial_crisis.csv` – Asset allocation during the 2008–2010 Financial Crisis  
- `allocation_2010_2015_post_crisis_recovery.csv` – Allocation during the 2010–2015 post-crisis recovery  
- `allocation_2015_2020_pre_covid_bull_market.csv` – Allocation during the 2015–2020 pre-COVID bull market  
- `allocation_2020_2025_covid_and_recovery.csv` – Allocation during the 2020–2025 COVID and recovery period  

**Key Findings**  
- **Average annualized return:** 12.8%  
- **Average Sharpe ratio:** 0.677  
- **Average Alpha:** 6.6%  
- Consistently **outperformed the SPY benchmark** across all market periods  

---

## 🗑️ Deprecated Files (Can Be Removed)

The following early test files have been replaced by the `historical_analysis` module:

- `adaptive_optimization_comparison.csv`  
- `adaptive_optimization_results.png`  
- `intelligent_selector_demo.png`  
- `smart_optimizer/` *(entire directory)*  

**To clean up old files:**
```bash
cd results
rm -f adaptive_optimization_comparison.csv
rm -f adaptive_optimization_results.png
rm -f intelligent_selector_demo.png
rm -f .DS_Store
rm -rf smart_optimizer/
```

---

## 📝 How to Regenerate Results

**To re-run the historical analysis:**
```bash
cd ~/Downloads/System/Quant/Portfolio_Optimization_system
python scripts/historical_analysis.py
```

**To run the full portfolio optimization system:**
```bash
python scripts/comprehensive_portfolio_system.py
```

---

**Last updated:** October 2025  
**Author:** Mengfan Long (kevinlmf)

