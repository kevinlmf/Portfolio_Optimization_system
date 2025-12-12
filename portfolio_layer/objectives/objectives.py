"""
Objectives - 优化目标

定义各种投资组合优化目标函数
"""

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class ObjectiveType(Enum):
    """优化目标类型"""
    SHARPE = 'sharpe'
    CVAR = 'cvar'
    RISK_PARITY = 'risk_parity'
    MIN_VARIANCE = 'min_variance'
    MEAN_VARIANCE = 'mean_variance'


class ObjectiveFunction(ABC):
    """优化目标函数基类"""
    
    @abstractmethod
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """
        评估目标函数值
        
        Args:
            weights: 投资组合权重 (N,)
            mu: 预期收益率 (N,)
            Sigma: 协方差矩阵 (N x N)
        
        Returns:
            目标函数值（越小越好，对于最大化问题返回负值）
        """
        pass


class SharpeObjective(ObjectiveFunction):
    """Sharpe Ratio 最大化（等价于最小化负 Sharpe）"""
    
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        
        if portfolio_risk == 0:
            return 1e10  # 惩罚零风险
        
        sharpe_ratio = portfolio_return / portfolio_risk
        return -sharpe_ratio  # 最小化负 Sharpe = 最大化 Sharpe


class CVaRObjective(ObjectiveFunction):
    """Conditional Value at Risk (CVaR) 最小化"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 置信水平（默认 5%）
        """
        self.alpha = alpha
    
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        # 简化版本：使用正态分布假设
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        
        # CVaR 近似（正态分布）
        from scipy.stats import norm
        z_alpha = norm.ppf(self.alpha)
        cvar = portfolio_return - (norm.pdf(z_alpha) / self.alpha) * portfolio_risk
        
        return -cvar  # 最小化负 CVaR = 最大化 CVaR


class RiskParityObjective(ObjectiveFunction):
    """Risk Parity - 风险平价"""
    
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        # 计算每个资产对组合风险的贡献
        portfolio_variance = np.dot(weights, np.dot(Sigma, weights))
        marginal_contrib = np.dot(Sigma, weights)
        risk_contrib = weights * marginal_contrib
        
        # 目标：所有资产的风险贡献相等
        target_contrib = portfolio_variance / len(weights)
        deviation = np.sum((risk_contrib - target_contrib) ** 2)
        
        return deviation


class MinVarianceObjective(ObjectiveFunction):
    """Minimum Variance - 最小方差"""
    
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        portfolio_variance = np.dot(weights, np.dot(Sigma, weights))
        return portfolio_variance


class MeanVarianceObjective(ObjectiveFunction):
    """Mean-Variance - 均值方差"""
    
    def __init__(self, risk_aversion: float = 1.0):
        """
        Args:
            risk_aversion: 风险厌恶系数
        """
        self.risk_aversion = risk_aversion
    
    def evaluate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        portfolio_return = np.dot(weights, mu)
        portfolio_variance = np.dot(weights, np.dot(Sigma, weights))
        
        # 最大化：return - risk_aversion * variance
        # 等价于最小化：-(return - risk_aversion * variance)
        return -(portfolio_return - self.risk_aversion * portfolio_variance)


