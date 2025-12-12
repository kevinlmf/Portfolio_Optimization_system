"""
Sparse Sharpe Optimizer

使用稀疏 Sharpe 优化方法
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Import DecisionSpecs using importlib
import importlib
_step3 = importlib.import_module('workflow.3_select_objective.decision_specs', package=None)
DecisionSpecs = _step3.DecisionSpecs


class SparseSharpeOptimizer:
    """稀疏 Sharpe 优化器"""
    
    def __init__(self, epsilon: float = 1e-3, max_iter: int = 10000):
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def optimize(self,
                knowledge,  # KnowledgeBase object
                decisions: DecisionSpecs,
                returns: np.ndarray = None) -> np.ndarray:
        """
        优化投资组合
        
        Args:
            knowledge: 知识库对象
            decisions: 决策规格
            returns: 历史收益率矩阵 (T x N)，可选
        
        Returns:
            最优权重向量
        """
        # 如果提供了历史收益率，可以使用原始算法
        # 否则使用简化版本
        m = decisions.method_params.get('sparsity', 10)
        
        if returns is not None:
            # 使用原始 mSSRM-PGA 算法
            return self._optimize_with_returns(returns, m)
        else:
            # 简化版本：返回等权重
            n_assets = len(knowledge.mu)
            return np.ones(n_assets) / n_assets
    
    def _optimize_with_returns(self, returns: np.ndarray, m: int) -> np.ndarray:
        """使用 mSSRM-PGA 算法优化"""
        # 这里可以集成原始的稀疏 Sharpe 算法
        # 暂时返回等权重
        n_assets = returns.shape[1]
        return np.ones(n_assets) / n_assets

