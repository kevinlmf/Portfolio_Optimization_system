"""
Expected Returns Estimation - 预期收益率估计

估计 μ (预期收益率)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class ExpectedReturnsEstimator(ABC):
    """预期收益率估计器基类"""
    
    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        估计预期收益率
        
        Args:
            returns: 资产收益率 DataFrame (T x N)
        
        Returns:
            预期收益率向量 μ (N,)
        """
        pass


class SampleMeanEstimator(ExpectedReturnsEstimator):
    """样本均值估计器"""
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """使用样本均值估计"""
        return returns.mean().values


class BayesianEstimator(ExpectedReturnsEstimator):
    """贝叶斯估计器"""
    
    def __init__(self, prior_mean: Optional[np.ndarray] = None, shrinkage: float = 0.5):
        """
        Args:
            prior_mean: 先验均值
            shrinkage: 收缩系数
        """
        self.prior_mean = prior_mean
        self.shrinkage = shrinkage
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """贝叶斯估计"""
        sample_mean = returns.mean().values
        
        if self.prior_mean is None:
            # 使用全局均值作为先验
            self.prior_mean = np.mean(sample_mean) * np.ones(len(sample_mean))
        
        # 收缩估计
        mu = (1 - self.shrinkage) * sample_mean + self.shrinkage * self.prior_mean
        
        return mu


class ShrinkageEstimator(ExpectedReturnsEstimator):
    """收缩估计器（Ledoit-Wolf 风格）"""
    
    def __init__(self, shrinkage_target: str = 'mean'):
        """
        Args:
            shrinkage_target: 收缩目标 ('mean', 'zero')
        """
        self.shrinkage_target = shrinkage_target
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """收缩估计"""
        sample_mean = returns.mean().values
        
        if self.shrinkage_target == 'mean':
            target = np.mean(sample_mean) * np.ones(len(sample_mean))
        else:
            target = np.zeros(len(sample_mean))
        
        # 计算最优收缩系数（简化版本）
        n = len(returns)
        shrinkage = 1.0 / (1.0 + n)
        
        mu = (1 - shrinkage) * sample_mean + shrinkage * target
        
        return mu


