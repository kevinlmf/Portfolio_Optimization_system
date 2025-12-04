"""
Portfolio Evaluator - 投资组合评估器

评估投资组合的表现
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .performance_metrics import PerformanceMetrics


class PortfolioEvaluator:
    """
    Portfolio Evaluator - 投资组合评估器
    
    评估投资组合的表现：
    1. 收益率指标
    2. 风险指标
    3. 风险调整收益指标
    4. 回撤分析
    """
    
    def __init__(self):
        """Initialize portfolio evaluator"""
        self.metrics_calculator = PerformanceMetrics()
    
    def evaluate(self,
                weights: np.ndarray,
                returns: pd.DataFrame,
                knowledge=None) -> Dict:
        """
        评估投资组合
        
        Args:
            weights: 投资组合权重 (N,)
            returns: 历史收益率 (T x N)
            knowledge: 知识库对象（可选）
        
        Returns:
            评估结果字典
        """
        # 计算组合收益率
        portfolio_returns = (returns.values @ weights)
        portfolio_returns_series = pd.Series(
            portfolio_returns,
            index=returns.index
        )
        
        # 计算各种指标
        metrics = self.metrics_calculator.calculate_all(portfolio_returns_series)
        
        # 如果有知识库，添加风险分解
        if knowledge is not None:
            risk_decomp = knowledge.decompose_risk(weights)
            metrics.update(risk_decomp)
        
        return metrics
    
    def compare_portfolios(self,
                          weights_list: Dict[str, np.ndarray],
                          returns: pd.DataFrame,
                          knowledge=None) -> pd.DataFrame:
        """
        比较多个投资组合
        
        Args:
            weights_list: 字典 {名称: 权重}
            returns: 历史收益率 (T x N)
            knowledge: 知识库对象（可选）
        
        Returns:
            比较结果 DataFrame
        """
        results = {}
        
        for name, weights in weights_list.items():
            metrics = self.evaluate(weights, returns, knowledge)
            results[name] = metrics
        
        return pd.DataFrame(results).T


