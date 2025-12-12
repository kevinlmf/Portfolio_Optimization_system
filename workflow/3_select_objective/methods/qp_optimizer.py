"""
Quadratic Programming Optimizer

使用 QP 方法求解投资组合优化问题
"""

import numpy as np
from scipy.optimize import minimize
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Import from correct locations using importlib
import importlib

# Import DecisionSpecs from workflow.3_select_objective
_step3 = importlib.import_module('workflow.3_select_objective.decision_specs', package=None)
DecisionSpecs = _step3.DecisionSpecs

# Import objectives
from portfolio_layer.objectives.objectives import (
    ObjectiveType,
    SharpeObjective, CVaRObjective, RiskParityObjective,
    MinVarianceObjective, MeanVarianceObjective
)


class QPOptimizer:
    """二次规划优化器"""
    
    def __init__(self):
        self.objective_map = {
            'sharpe': SharpeObjective(),
            'cvar': CVaRObjective(),
            'risk_parity': RiskParityObjective(),
            'min_variance': MinVarianceObjective(),
            'mean_variance': MeanVarianceObjective()
        }
    
    def optimize(self,
                knowledge,  # KnowledgeBase object
                decisions: DecisionSpecs) -> np.ndarray:
        """
        优化投资组合
        
        Args:
            knowledge: 知识库对象（包含 mu, Sigma, B 等）
            decisions: 决策规格
        
        Returns:
            最优权重向量
        """
        n_assets = len(knowledge.mu)
        mu = knowledge.mu
        Sigma = knowledge.get_covariance()
        
        # 选择目标函数
        obj_type = decisions.objective.value
        if obj_type not in self.objective_map:
            obj_type = 'sharpe'  # 默认
        
        objective_func = self.objective_map[obj_type]
        
        # 如果是 mean_variance，设置风险厌恶系数
        if obj_type == 'mean_variance':
            objective_func = MeanVarianceObjective(decisions.risk_aversion)
        
        # 如果是 cvar，设置 alpha
        if obj_type == 'cvar':
            objective_func = CVaRObjective(decisions.cvar_alpha)
        
        # 目标函数
        def objective(w):
            return objective_func.evaluate(w, mu, Sigma)
        
        # 约束条件
        constraints = decisions.constraints.get_constraint_functions(
            n_assets,
            knowledge.B if hasattr(knowledge, 'B') and knowledge.B is not None else None,
            knowledge.factor_names if hasattr(knowledge, 'factor_names') else None
        )
        
        # 边界
        bounds = decisions.constraints.get_bounds(n_assets)
        
        # 初始猜测
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
            return x0
        
        return result.x

