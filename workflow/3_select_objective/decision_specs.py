"""
Decision Specs - 决策规格

封装优化决策的所有参数
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from portfolio_layer.objectives.objectives import ObjectiveType
from portfolio_layer.constraints.constraints import Constraints


@dataclass
class DecisionSpecs:
    """决策规格"""
    objective: ObjectiveType
    constraints: Constraints
    method: str = 'qp'  # 'qp', 'sparse_sharpe'
    risk_aversion: float = 1.0  # 用于 Mean-Variance
    cvar_alpha: float = 0.05  # 用于 CVaR
    method_params: Dict = field(default_factory=dict)  # 方法特定参数



