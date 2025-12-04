"""
Knowledge Base - 知识库

统一存储所有估计的参数：μ, B, F, D, Σ
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class KnowledgeBase:
    """
    Knowledge Base - 知识库
    
    存储所有估计的参数：
    - μ: 预期收益率 (N,)
    - B: 因子载荷矩阵 (N x K)，可选
    - F: 因子协方差矩阵 (K x K)，可选
    - D: 特质风险对角矩阵 (N x N)，可选
    - Σ: 完整协方差矩阵 (N x N)
    """
    mu: np.ndarray
    Sigma: np.ndarray
    B: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    asset_names: Optional[List[str]] = None
    factor_names: Optional[List[str]] = None
    copula: Optional[Dict] = None  # Copula 模型（可选）
    
    def get_covariance(self) -> np.ndarray:
        """获取协方差矩阵"""
        return self.Sigma
    
    def get_expected_returns(self) -> np.ndarray:
        """获取预期收益率"""
        return self.mu
    
    def decompose_risk(self, weights: np.ndarray) -> Dict:
        """
        分解投资组合风险
        
        Args:
            weights: 投资组合权重 (N,)
        
        Returns:
            风险分解字典
        """
        total_risk = np.sqrt(np.dot(weights, np.dot(self.Sigma, weights)))
        
        result = {
            'total_risk': float(total_risk),
            'total_variance': float(np.dot(weights, np.dot(self.Sigma, weights)))
        }
        
        # 如果有因子模型，进行因子风险分解
        if self.B is not None and self.F is not None:
            factor_risk = np.sqrt(
                np.dot(weights, np.dot(self.B @ self.F @ self.B.T, weights))
            )
            result['factor_risk'] = float(factor_risk)
            result['factor_risk_pct'] = float(factor_risk / total_risk * 100) if total_risk > 0 else 0
        
        if self.D is not None:
            idiosyncratic_risk = np.sqrt(
                np.dot(weights, np.dot(self.D, weights))
            )
            result['idiosyncratic_risk'] = float(idiosyncratic_risk)
            result['idiosyncratic_risk_pct'] = float(idiosyncratic_risk / total_risk * 100) if total_risk > 0 else 0
        
        return result


