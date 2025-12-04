"""
Factor Risk Model - 因子风险模型

基于因子模型构建风险结构
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import sys
import os

# Import FactorLoadingsEstimator from correct location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from factor_layer.factor_regression.factor_loadings import FactorLoadingsEstimator


class FactorRiskModel:
    """
    Factor Risk Model - 因子风险模型
    
    基于因子载荷矩阵 B 构建风险模型：
    Σ = B * F * B^T + D
    
    其中：
    - B: 因子载荷矩阵 (N x K)
    - F: 因子协方差矩阵 (K x K)
    - D: 特质风险对角矩阵 (N x N)
    """
    
    def __init__(self):
        """Initialize factor risk model"""
        self.B = None
        self.F = None
        self.D = None
        self.Sigma = None
    
    def build(self,
             returns: pd.DataFrame,
             factors: pd.DataFrame,
             B: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建因子风险模型
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
            B: 因子载荷矩阵 (N x K)，如果为 None 则自动估计
        
        Returns:
            (F, D, Sigma) 其中：
            - F: 因子协方差矩阵 (K x K)
            - D: 特质风险对角矩阵 (N x N)
            - Sigma: 完整协方差矩阵 (N x N)
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx]
        factors_aligned = factors.loc[common_idx]
        
        # 估计因子载荷（如果未提供）
        if B is None:
            estimator = FactorLoadingsEstimator(method='ols')
            B = estimator.estimate(returns_aligned, factors_aligned)
        
        self.B = B.values if isinstance(B, pd.DataFrame) else B
        
        # 估计因子协方差 F
        F = np.cov(factors_aligned.values.T)
        self.F = F
        
        # 估计特质风险 D
        # 计算残差
        factor_returns = factors_aligned.values
        predicted_returns = factor_returns @ self.B.T
        residuals = returns_aligned.values - predicted_returns
        
        # 特质风险是对角矩阵
        D = np.diag(np.var(residuals, axis=0))
        self.D = D
        
        # 构建完整协方差矩阵
        # Σ = B * F * B^T + D
        self.Sigma = self.B @ self.F @ self.B.T + self.D
        
        return self.F, self.D, self.Sigma
    
    def get_factor_covariance(self) -> np.ndarray:
        """Get factor covariance matrix F"""
        if self.F is None:
            raise ValueError("Model not built yet. Call build() first.")
        return self.F
    
    def get_idiosyncratic_risk(self) -> np.ndarray:
        """Get idiosyncratic risk matrix D"""
        if self.D is None:
            raise ValueError("Model not built yet. Call build() first.")
        return self.D
    
    def get_full_covariance(self) -> np.ndarray:
        """Get full covariance matrix Σ"""
        if self.Sigma is None:
            raise ValueError("Model not built yet. Call build() first.")
        return self.Sigma
    
    def decompose_risk(self, weights: np.ndarray) -> dict:
        """
        分解投资组合风险
        
        Args:
            weights: 投资组合权重 (N,)
        
        Returns:
            Dictionary with risk decomposition
        """
        if self.Sigma is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        # 总风险
        total_risk = np.sqrt(weights @ self.Sigma @ weights)
        
        # 因子风险
        factor_risk = np.sqrt(weights @ (self.B @ self.F @ self.B.T) @ weights)
        
        # 特质风险
        idiosyncratic_risk = np.sqrt(weights @ self.D @ weights)
        
        return {
            'total_risk': float(total_risk),
            'factor_risk': float(factor_risk),
            'idiosyncratic_risk': float(idiosyncratic_risk),
            'factor_risk_pct': float(factor_risk / total_risk * 100) if total_risk > 0 else 0,
            'idiosyncratic_risk_pct': float(idiosyncratic_risk / total_risk * 100) if total_risk > 0 else 0
        }


