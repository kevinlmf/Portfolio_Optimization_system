"""
Step 3: Forecasting - 收益与风险预测

预测选中股票的未来收益和协方差。

核心思想：
1. Return Forecasting: 使用因子模型、时间序列模型、深度学习
2. Covariance Forecasting: 使用 DCC-GARCH、因子模型
3. Ensemble: 多模型融合提高预测稳健性
"""

from .return_forecaster import ReturnForecaster
from .covariance_forecaster import CovarianceForecaster
from .ensemble_forecaster import EnsembleForecaster, ForecastResult

__all__ = [
    'ReturnForecaster',
    'CovarianceForecaster', 
    'EnsembleForecaster',
    'ForecastResult'
]

