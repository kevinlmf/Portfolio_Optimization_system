"""
Regime Parameter Estimator - 状态依赖参数估计器

为每个regime估计独立的参数：μ(s), B(s), F(s), D(s), Σ(s)

数学模型：
    对于每个regime s:
    r_t | s_t = s ~ N(μ(s), Σ(s))
    
    如果使用因子模型：
    r_t | F_t, s_t = s = B(s) * F_t + ε_t(s)
    ε_t(s) ~ N(0, D(s))
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from .regime_knowledge_base import RegimeKnowledgeBase, RegimeParameters
from .regime_detector import RegimeState


class RegimeParameterEstimator:
    """
    状态依赖参数估计器
    
    为每个regime分别估计：
    - μ(s): 预期收益
    - Σ(s): 协方差矩阵
    - B(s), F(s), D(s): 因子模型参数（可选）
    """
    
    def __init__(self, 
                 use_factor_model: bool = True,
                 shrinkage: bool = True,
                 min_samples_per_regime: int = 30):
        """
        初始化估计器
        
        Args:
            use_factor_model: 是否使用因子模型
            shrinkage: 是否使用收缩估计
            min_samples_per_regime: 每个regime的最小样本数
        """
        self.use_factor_model = use_factor_model
        self.shrinkage = shrinkage
        self.min_samples_per_regime = min_samples_per_regime
    
    def estimate(self,
                returns: pd.DataFrame,
                regime_state: RegimeState,
                factors: Optional[pd.DataFrame] = None) -> RegimeKnowledgeBase:
        """
        估计所有regime的参数
        
        Args:
            returns: 收益率数据 (T x N)
            regime_state: 从RegimeDetector得到的状态信息
            factors: 因子收益率 (T x K)，可选
        
        Returns:
            RegimeKnowledgeBase
        """
        n_regimes = regime_state.n_regimes
        regime_sequence = regime_state.regime_sequence
        
        # 确保数据对齐
        returns_clean = returns.dropna()
        if len(returns_clean) != len(regime_sequence):
            # 截断到相同长度
            min_len = min(len(returns_clean), len(regime_sequence))
            returns_clean = returns_clean.iloc[:min_len]
            regime_sequence = regime_sequence[:min_len]
        
        # 为每个regime估计参数
        regime_params = {}
        
        for k in range(n_regimes):
            # 获取该regime的数据
            mask = regime_sequence == k
            regime_returns = returns_clean.loc[mask.astype(bool) if isinstance(mask, pd.Series) else mask]
            
            if len(regime_returns) < self.min_samples_per_regime:
                print(f"Warning: Regime {k} has only {len(regime_returns)} samples. "
                      f"Using pooled estimates.")
                # 使用全样本估计作为后备
                regime_returns = returns_clean
            
            if factors is not None and self.use_factor_model:
                # 使用因子模型估计
                factors_clean = factors.dropna()
                if len(factors_clean) != len(regime_sequence):
                    min_len = min(len(factors_clean), len(regime_sequence))
                    factors_clean = factors_clean.iloc[:min_len]
                
                regime_factors = factors_clean.loc[mask.astype(bool) if isinstance(mask, pd.Series) else mask]
                
                params = self._estimate_with_factors(regime_returns, regime_factors)
            else:
                # 直接估计
                params = self._estimate_simple(regime_returns)
            
            regime_params[k] = params
        
        return RegimeKnowledgeBase(
            regime_params=regime_params,
            transition_matrix=regime_state.transition_matrix,
            current_regime=regime_state.current_regime,
            regime_probabilities=regime_state.regime_probabilities,
            asset_names=returns.columns.tolist(),
            factor_names=factors.columns.tolist() if factors is not None else None
        )
    
    def _estimate_simple(self, returns: pd.DataFrame) -> RegimeParameters:
        """
        简单估计（不使用因子模型）
        
        Args:
            returns: 收益率数据
        
        Returns:
            RegimeParameters
        """
        mu = returns.mean().values
        
        if self.shrinkage:
            Sigma = self._ledoit_wolf_shrinkage(returns.values)
        else:
            Sigma = returns.cov().values
        
        return RegimeParameters(mu=mu, Sigma=Sigma)
    
    def _estimate_with_factors(self,
                               returns: pd.DataFrame,
                               factors: pd.DataFrame) -> RegimeParameters:
        """
        使用因子模型估计
        
        r_t = B * F_t + ε_t
        
        Args:
            returns: 收益率数据 (T x N)
            factors: 因子收益率 (T x K)
        
        Returns:
            RegimeParameters
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx]
        factors_aligned = factors.loc[common_idx]
        
        N = returns_aligned.shape[1]
        K = factors_aligned.shape[1]
        
        # 估计因子载荷 B (N x K)
        B = np.zeros((N, K))
        residuals = np.zeros_like(returns_aligned.values)
        
        for i in range(N):
            y = returns_aligned.iloc[:, i].values
            X = factors_aligned.values
            
            # OLS回归
            X_with_const = np.column_stack([np.ones(len(X)), X])
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                B[i, :] = beta[1:]  # 不包含截距
                residuals[:, i] = y - X @ beta[1:] - beta[0]
            except np.linalg.LinAlgError:
                B[i, :] = 0
                residuals[:, i] = y - y.mean()
        
        # 估计因子协方差 F (K x K)
        if self.shrinkage:
            F = self._ledoit_wolf_shrinkage(factors_aligned.values)
        else:
            F = factors_aligned.cov().values
        
        # 估计特质风险 D (N x N 对角矩阵)
        D = np.diag(np.var(residuals, axis=0))
        
        # 构建完整协方差矩阵
        # Σ = B * F * B' + D
        Sigma = B @ F @ B.T + D
        
        # 确保正定
        Sigma = self._ensure_positive_definite(Sigma)
        
        # 估计预期收益
        mu = returns_aligned.mean().values
        
        return RegimeParameters(mu=mu, Sigma=Sigma, B=B, F=F, D=D)
    
    def _ledoit_wolf_shrinkage(self, X: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf收缩估计
        
        Σ_shrunk = (1-α) * S + α * F
        
        其中S是样本协方差，F是结构化目标（单位矩阵的缩放）
        """
        T, N = X.shape
        
        # 样本协方差
        S = np.cov(X.T)
        
        # 目标矩阵（缩放的单位矩阵）
        mu = np.trace(S) / N
        F = mu * np.eye(N)
        
        # 计算最优收缩强度
        # 简化版Ledoit-Wolf
        X_centered = X - X.mean(axis=0)
        
        # 计算收缩强度的分子分母
        delta = S - F
        delta_sq_sum = np.sum(delta ** 2)
        
        # 计算样本协方差估计的方差
        sum_sq = 0
        for t in range(T):
            outer = np.outer(X_centered[t], X_centered[t])
            sum_sq += np.sum((outer - S) ** 2)
        
        kappa = sum_sq / T ** 2
        
        # 最优收缩强度
        if delta_sq_sum > 0:
            shrinkage_intensity = min(1, kappa / delta_sq_sum)
        else:
            shrinkage_intensity = 1
        
        # 收缩估计
        Sigma_shrunk = (1 - shrinkage_intensity) * S + shrinkage_intensity * F
        
        return Sigma_shrunk
    
    def _ensure_positive_definite(self, 
                                  Sigma: np.ndarray, 
                                  min_eigenvalue: float = 1e-6) -> np.ndarray:
        """确保矩阵正定"""
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        
        # 将负特征值和过小的特征值调整为min_eigenvalue
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # 重建矩阵
        Sigma_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # 确保对称
        Sigma_pd = (Sigma_pd + Sigma_pd.T) / 2
        
        return Sigma_pd
    
    def estimate_with_weighted_samples(self,
                                       returns: pd.DataFrame,
                                       regime_state: RegimeState,
                                       factors: Optional[pd.DataFrame] = None) -> RegimeKnowledgeBase:
        """
        使用平滑概率加权估计参数
        
        这种方法使用所有样本，但根据属于各regime的概率进行加权
        可以避免样本量不足的问题
        
        Args:
            returns: 收益率数据 (T x N)
            regime_state: 状态信息（包含平滑概率）
            factors: 因子收益率，可选
        
        Returns:
            RegimeKnowledgeBase
        """
        n_regimes = regime_state.n_regimes
        gamma = regime_state.smoothed_probabilities  # (T x K)
        
        # 确保数据对齐
        returns_clean = returns.dropna()
        if len(returns_clean) != len(gamma):
            min_len = min(len(returns_clean), len(gamma))
            returns_clean = returns_clean.iloc[:min_len]
            gamma = gamma[:min_len]
        
        X = returns_clean.values
        T, N = X.shape
        
        regime_params = {}
        
        for k in range(n_regimes):
            weights = gamma[:, k]
            weights_sum = weights.sum()
            
            if weights_sum < 1e-10:
                # 该regime权重过小，使用全样本
                weights = np.ones(T) / T
                weights_sum = 1.0
            
            # 加权均值
            mu = np.average(X, axis=0, weights=weights)
            
            # 加权协方差
            X_centered = X - mu
            Sigma = np.zeros((N, N))
            for t in range(T):
                Sigma += weights[t] * np.outer(X_centered[t], X_centered[t])
            Sigma /= weights_sum
            
            # 确保正定
            Sigma = self._ensure_positive_definite(Sigma)
            
            # 如果使用因子模型
            if factors is not None and self.use_factor_model:
                factors_clean = factors.dropna()
                if len(factors_clean) != len(gamma):
                    min_len = min(len(factors_clean), len(gamma))
                    factors_clean = factors_clean.iloc[:min_len]
                
                B, F, D = self._weighted_factor_regression(
                    returns_clean, factors_clean, weights
                )
                regime_params[k] = RegimeParameters(mu=mu, Sigma=Sigma, B=B, F=F, D=D)
            else:
                regime_params[k] = RegimeParameters(mu=mu, Sigma=Sigma)
        
        return RegimeKnowledgeBase(
            regime_params=regime_params,
            transition_matrix=regime_state.transition_matrix,
            current_regime=regime_state.current_regime,
            regime_probabilities=regime_state.regime_probabilities,
            asset_names=returns.columns.tolist(),
            factor_names=factors.columns.tolist() if factors is not None else None
        )
    
    def _weighted_factor_regression(self,
                                    returns: pd.DataFrame,
                                    factors: pd.DataFrame,
                                    weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加权因子回归
        
        Args:
            returns: 收益率数据
            factors: 因子数据
            weights: 样本权重
        
        Returns:
            B, F, D
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx].values
        factors_aligned = factors.loc[common_idx].values
        
        T, N = returns_aligned.shape
        K = factors_aligned.shape[1]
        
        # 确保权重长度匹配
        if len(weights) != T:
            weights = weights[:T]
        
        W = np.diag(weights)
        
        # 加权OLS: B = (X'WX)^(-1) X'Wy
        B = np.zeros((N, K))
        residuals = np.zeros((T, N))
        
        X = factors_aligned
        XtWX = X.T @ W @ X
        
        try:
            XtWX_inv = np.linalg.inv(XtWX + 1e-6 * np.eye(K))
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        
        for i in range(N):
            y = returns_aligned[:, i]
            B[i, :] = XtWX_inv @ X.T @ W @ y
            residuals[:, i] = y - X @ B[i, :]
        
        # 加权因子协方差
        weights_sum = weights.sum()
        factor_mean = np.average(factors_aligned, axis=0, weights=weights)
        factors_centered = factors_aligned - factor_mean
        F = np.zeros((K, K))
        for t in range(T):
            F += weights[t] * np.outer(factors_centered[t], factors_centered[t])
        F /= weights_sum
        
        # 特质风险
        D = np.diag(np.average(residuals ** 2, axis=0, weights=weights))
        
        return B, F, D

