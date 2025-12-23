"""
Step 2: Stock Selection - 因子选股

根据因子暴露选择最适合当前市场状态的股票。

核心思想：
1. 计算每只股票的因子暴露 (factor loadings)
2. 根据当前 regime 确定因子权重
3. 计算综合得分并排名选股
4. 应用约束条件（行业集中度、流动性等）
"""

from .stock_selector import StockSelector, SelectionResult

__all__ = ['StockSelector', 'SelectionResult']

