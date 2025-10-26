# Results Directory

This directory contains analysis results from the portfolio optimization system.

## 📊 Historical Analysis (最重要！)

### historical_analysis/
**16年历史回测分析结果 (2008-2024)**

- `period_comparison.csv` - 4个时期的性能对比表格
- `comprehensive_comparison.png` - 综合可视化图表
- `allocation_2008_2010_financial_crisis.csv` - 2008-2010期资产配置
- `allocation_2010_2015_post_crisis_recovery.csv` - 2010-2015期资产配置
- `allocation_2015_2020_pre_covid_bull_market.csv` - 2015-2020期资产配置
- `allocation_2020_2025_covid_and_recovery.csv` - 2020-2025期资产配置

**关键发现**：
- 平均年化收益: 12.8%
- 平均Sharpe比率: 0.677
- 平均Alpha: 6.6%
- 所有时期均跑赢SPY基准

---

## 🗑️ 旧文件（可以删除）

以下是早期测试文件，已被historical_analysis取代：

- `adaptive_optimization_comparison.csv`
- `adaptive_optimization_results.png`
- `intelligent_selector_demo.png`
- `smart_optimizer/` (整个目录)

执行清理：
\`\`\`bash
cd results
rm -f adaptive_optimization_comparison.csv
rm -f adaptive_optimization_results.png
rm -f intelligent_selector_demo.png
rm -f .DS_Store
rm -rf smart_optimizer/
\`\`\`

---

## 📝 如何重新生成

如果需要重新生成历史分析结果：
\`\`\`bash
cd ~/Downloads/System/Quant/Portfolio_Optimization_system
python scripts/historical_analysis.py
\`\`\`

如果需要运行完整系统：
\`\`\`bash
python scripts/comprehensive_portfolio_system.py
\`\`\`
