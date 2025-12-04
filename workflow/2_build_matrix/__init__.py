"""
Step 2: Build Matrix - 构建股票-因子关系矩阵

Construct stock-factor relationship matrix (B)
构建股票和因子之间的相关性矩阵
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import from actual file locations
from factor_layer.factor_regression.factor_loadings import FactorLoadingsEstimator
from portfolio_layer.build_matrix.correlation_matrix import CorrelationMatrixBuilder

# FactorRiskModel may have import issues, import it carefully
try:
    from factor_layer.factor_risk_models.factor_risk_model import FactorRiskModel
except ImportError:
    # Create a simple placeholder if import fails
    class FactorRiskModel:
        pass

__all__ = ['FactorLoadingsEstimator', 'CorrelationMatrixBuilder', 'FactorRiskModel']
