"""
Constraints - 约束条件

定义投资组合优化的约束条件
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from scipy.optimize import NonlinearConstraint


@dataclass
class Constraints:
    """投资组合约束条件"""
    long_only: bool = True
    leverage: float = 1.0  # 杠杆率
    max_weight: Optional[float] = None
    min_weight: Optional[float] = None
    max_turnover: Optional[float] = None
    previous_weights: Optional[np.ndarray] = None
    factor_bounds: Optional[Dict[str, tuple]] = None  # 因子暴露上下界
    
    def get_bounds(self, n_assets: int) -> List[tuple]:
        """获取权重边界"""
        if self.long_only:
            lower = [0.0] * n_assets
        else:
            lower = [-self.leverage] * n_assets
        
        upper = [self.leverage] * n_assets
        
        if self.max_weight is not None:
            upper = [min(u, self.max_weight) for u in upper]
        
        if self.min_weight is not None:
            lower = [max(l, self.min_weight) for l in lower]
        
        return list(zip(lower, upper))
    
    def get_constraint_functions(self,
                                n_assets: int,
                                B: Optional[np.ndarray] = None,
                                factor_names: Optional[List[str]] = None) -> List[dict]:
        """获取约束函数列表"""
        constraints = []
        
        # 权重和为 leverage
        def weight_sum_constraint(w):
            return np.sum(w) - self.leverage
        
        constraints.append({
            'type': 'eq',
            'fun': weight_sum_constraint
        })
        
        # 换手率约束
        if self.max_turnover is not None and self.previous_weights is not None:
            def turnover_constraint(w):
                turnover = np.sum(np.abs(w - self.previous_weights))
                return self.max_turnover - turnover
            
            constraints.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        # 因子暴露约束
        if self.factor_bounds is not None and B is not None and factor_names is not None:
            for factor_name, (lower, upper) in self.factor_bounds.items():
                if factor_name in factor_names:
                    factor_idx = factor_names.index(factor_name)
                    
                    def factor_exposure_constraint(w, idx=factor_idx, lb=lower, ub=upper):
                        exposure = np.dot(w, B[:, idx])
                        return [exposure - lb, ub - exposure]
                    
                    constraints.append({
                        'type': 'ineq',
                        'fun': factor_exposure_constraint
                    })
        
        return constraints


class ConstraintBuilder:
    """约束条件构建器"""
    
    def __init__(self):
        self.constraints = Constraints()
    
    def long_only(self, value: bool = True):
        """设置是否只做多"""
        self.constraints.long_only = value
        return self
    
    def leverage(self, value: float):
        """设置杠杆率"""
        self.constraints.leverage = value
        return self
    
    def max_weight(self, value: float):
        """设置最大权重"""
        self.constraints.max_weight = value
        return self
    
    def min_weight(self, value: float):
        """设置最小权重"""
        self.constraints.min_weight = value
        return self
    
    def max_turnover(self, value: float, previous_weights: Optional[np.ndarray] = None):
        """设置最大换手率"""
        self.constraints.max_turnover = value
        self.constraints.previous_weights = previous_weights
        return self
    
    def factor_bounds(self, factor_name: str, lower: float, upper: float):
        """设置因子暴露上下界"""
        if self.constraints.factor_bounds is None:
            self.constraints.factor_bounds = {}
        self.constraints.factor_bounds[factor_name] = (lower, upper)
        return self
    
    def build(self) -> Constraints:
        """构建约束条件"""
        return self.constraints


