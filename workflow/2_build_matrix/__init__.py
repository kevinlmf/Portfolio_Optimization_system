"""
Step 2: Build Matrix - 构建股票-因子关系矩阵

Construct stock-factor relationship matrix (B)
构建股票和因子之间的相关性矩阵
"""

from .factor_loadings import FactorLoadingsEstimator
from .correlation_matrix import CorrelationMatrixBuilder
from .factor_risk_model import FactorRiskModel

__all__ = ['FactorLoadingsEstimator', 'CorrelationMatrixBuilder', 'FactorRiskModel']
