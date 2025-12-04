"""
Sample Estimator - 样本估计器

使用样本统计量估计参数
"""

import numpy as np
import pandas as pd
from typing import Optional
from .parameter_estimator import ParameterEstimator
from .knowledge_base import KnowledgeBase
from .expected_returns import SampleMeanEstimator
from .risk_structure import RiskStructureEstimator
from .dependency_structure import DependencyStructureEstimator

# Import from step 2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from workflow import FactorLoadingsEstimator


class SampleEstimator(ParameterEstimator):
    """样本估计器"""
    
    def estimate(self,
                returns: pd.DataFrame,
                factors: Optional[pd.DataFrame] = None) -> KnowledgeBase:
        """使用样本统计量估计"""
        
        # 1. 估计 μ
        mu_estimator = SampleMeanEstimator()
        mu = mu_estimator.estimate(returns)
        
        # 2. 估计 Σ（如果有因子，使用因子模型；否则直接估计）
        if factors is not None:
            # 使用因子模型
            B_estimator = FactorLoadingsEstimator()
            B = B_estimator.estimate(returns, factors)
            
            F_D_estimator = RiskStructureEstimator()
            F, D = F_D_estimator.estimate(returns, factors, B)
            
            # 构建 Σ
            Sigma = F_D_estimator.construct_covariance(B.values, F, D)
            
            return KnowledgeBase(
                mu=mu,
                Sigma=Sigma,
                B=B.values,
                F=F,
                D=D,
                asset_names=returns.columns.tolist(),
                factor_names=factors.columns.tolist()
            )
        else:
            # 直接估计协方差
            sigma_estimator = DependencyStructureEstimator()
            Sigma = sigma_estimator.estimate_covariance(returns)
            
            return KnowledgeBase(
                mu=mu,
                Sigma=Sigma,
                asset_names=returns.columns.tolist()
            )
