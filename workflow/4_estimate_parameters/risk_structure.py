"""
Risk Structure Estimation - 风险结构估计

估计 F (因子协方差) 和 D (特质风险)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Import from step 2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from workflow import FactorLoadingsEstimator


class RiskStructureEstimator:
    """风险结构估计器"""
    
    def __init__(self, lookback: int = 252):
        """
        Args:
            lookback: 回看期（默认252个交易日）
        """
        self.lookback = lookback
    
    def estimate(self,
                returns: pd.DataFrame,
                factors: pd.DataFrame,
                B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计因子协方差 F 和特质风险 D
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
            B: 因子载荷矩阵 (N x K)
        
        Returns:
            (F, D) 其中：
            - F: 因子协方差矩阵 (K x K)
            - D: 特质风险对角矩阵 (N x N)
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx]
        factors_aligned = factors.loc[common_idx]
        
        # 估计因子协方差 F
        F = np.cov(factors_aligned.values.T)
        
        # 估计特质风险 D
        # 计算残差
        factor_returns = factors_aligned.values
        B_values = B.values if isinstance(B, pd.DataFrame) else B
        predicted_returns = factor_returns @ B_values.T
        residuals = returns_aligned.values - predicted_returns
        
        # 特质风险是对角矩阵
        D = np.diag(np.var(residuals, axis=0))
        
        return F, D
    
    def construct_covariance(self,
                            B: np.ndarray,
                            F: np.ndarray,
                            D: np.ndarray) -> np.ndarray:
        """
        构建完整协方差矩阵
        
        Σ = B * F * B^T + D
        
        Args:
            B: 因子载荷矩阵 (N x K)
            F: 因子协方差矩阵 (K x K)
            D: 特质风险对角矩阵 (N x N)
        
        Returns:
            完整协方差矩阵 Σ (N x N)
        """
        return B @ F @ B.T + D
