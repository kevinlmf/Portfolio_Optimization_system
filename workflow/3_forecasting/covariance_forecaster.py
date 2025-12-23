"""
Covariance Forecaster - 协方差预测器

预测股票收益的协方差矩阵。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class CovarianceForecaster:
    """
    协方差预测器
    
    支持多种预测方法：
    1. Sample: 样本协方差
    2. Factor Model: Σ = BFB' + D
    3. Shrinkage: Ledoit-Wolf 收缩估计
    4. EWMA: 指数加权移动平均
    5. DCC-like: 简化的动态条件相关
    """
    
    def __init__(self, method: str = 'factor'):
        """
        初始化预测器
        
        Args:
            method: 预测方法 ('sample', 'factor', 'shrinkage', 'ewma', 'dcc')
        """
        self.method = method
        self.fitted = False
        
        # 模型参数
        self.sample_cov = None
        self.factor_cov = None  # F
        self.factor_loadings = None  # B
        self.idiosyncratic = None  # D
        self.shrinkage_target = None
        self.shrinkage_intensity = None
        self.ewma_cov = None
        self.ewma_lambda = 0.94
    
    def fit(self,
            returns: pd.DataFrame,
            factors: Optional[pd.DataFrame] = None,
            lookback: int = 252) -> 'CovarianceForecaster':
        """
        拟合协方差预测模型
        
        Args:
            returns: 股票收益率 (T x N)
            factors: 因子收益率 (T x K)
            lookback: 回看窗口
        """
        returns = returns.iloc[-lookback:] if len(returns) > lookback else returns
        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        # 样本协方差
        self.sample_cov = returns.cov().values
        
        if self.method == 'factor' and factors is not None:
            factors = factors.iloc[-lookback:] if len(factors) > lookback else factors
            self._fit_factor_model(returns, factors)
        
        if self.method == 'shrinkage':
            self._fit_shrinkage(returns)
        
        if self.method == 'ewma':
            self._fit_ewma(returns)
        
        if self.method == 'dcc':
            self._fit_dcc(returns)
        
        self.fitted = True
        return self
    
    def _fit_factor_model(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """
        拟合因子模型: Σ = BFB' + D
        """
        common_idx = returns.index.intersection(factors.index)
        R = returns.loc[common_idx].values  # T x N
        F = factors.loc[common_idx].values  # T x K
        
        T, N = R.shape
        K = F.shape[1]
        
        # 因子载荷 B (N x K): OLS
        F_with_intercept = np.column_stack([np.ones(T), F])
        try:
            coeffs = np.linalg.lstsq(F_with_intercept, R, rcond=None)[0]
            self.factor_loadings = coeffs[1:, :].T  # N x K
        except:
            # Ridge fallback
            FtF = F_with_intercept.T @ F_with_intercept + 0.01 * np.eye(K + 1)
            FtR = F_with_intercept.T @ R
            coeffs = np.linalg.solve(FtF, FtR)
            self.factor_loadings = coeffs[1:, :].T
        
        # 因子协方差 F (K x K)
        F_centered = F - F.mean(axis=0)
        self.factor_cov = (F_centered.T @ F_centered) / (T - 1)
        
        # 残差
        alpha = coeffs[0, :]
        residuals = R - (np.ones((T, 1)) @ alpha.reshape(1, -1) + F @ self.factor_loadings.T)
        
        # 特质风险 D (对角矩阵)
        self.idiosyncratic = np.diag(np.var(residuals, axis=0))
    
    def _fit_shrinkage(self, returns: pd.DataFrame):
        """
        Ledoit-Wolf 收缩估计
        
        Σ_shrunk = δ * Target + (1-δ) * Sample
        Target = diag(sample variances) (对角阵)
        """
        R = returns.values
        T, N = R.shape
        
        # 样本协方差
        sample = self.sample_cov
        
        # 收缩目标：对角阵
        self.shrinkage_target = np.diag(np.diag(sample))
        
        # 计算最优收缩强度 (Ledoit-Wolf 公式的简化版)
        # 这里使用简化的固定值，实际应用中可以用完整公式
        self.shrinkage_intensity = min(1.0, max(0.0, (N + 1) / (T + N + 1)))
    
    def _fit_ewma(self, returns: pd.DataFrame):
        """
        指数加权移动平均协方差
        
        Σ_t = λ * Σ_{t-1} + (1-λ) * r_t * r_t'
        """
        R = returns.values
        T, N = R.shape
        lam = self.ewma_lambda
        
        # 初始化为样本协方差
        self.ewma_cov = self.sample_cov.copy()
        
        # EWMA 更新 (从最旧到最新)
        R_centered = R - R.mean(axis=0)
        for t in range(T):
            r_t = R_centered[t:t+1, :]
            self.ewma_cov = lam * self.ewma_cov + (1 - lam) * (r_t.T @ r_t)
    
    def _fit_dcc(self, returns: pd.DataFrame):
        """
        简化的 DCC (Dynamic Conditional Correlation)
        
        实际 DCC 需要两步估计，这里使用简化版本：
        1. 估计每个资产的 GARCH 波动率
        2. 使用 EWMA 估计相关矩阵
        """
        R = returns.values
        T, N = R.shape
        
        # Step 1: 估计每个资产的波动率 (EWMA)
        lam = 0.94
        variances = np.var(R, axis=0)
        ewma_var = variances.copy()
        
        for t in range(T):
            ewma_var = lam * ewma_var + (1 - lam) * R[t, :] ** 2
        
        self.dcc_volatilities = np.sqrt(ewma_var)
        
        # Step 2: 估计相关矩阵
        # 标准化收益
        R_std = R / R.std(axis=0)
        self.dcc_correlation = np.corrcoef(R_std.T)
        
        # 确保正定
        self.dcc_correlation = self._ensure_positive_definite(self.dcc_correlation)
    
    def _ensure_positive_definite(self, cov: np.ndarray) -> np.ndarray:
        """确保协方差矩阵正定"""
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 将负特征值设为小正数
        min_eigenvalue = 1e-8
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # 重构矩阵
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def predict(self, horizon: int = 21) -> np.ndarray:
        """
        预测协方差矩阵
        
        Args:
            horizon: 预测周期（天）
            
        Returns:
            预测的协方差矩阵 (N x N)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.method == 'sample':
            cov = self.sample_cov
        elif self.method == 'factor':
            cov = self._predict_factor()
        elif self.method == 'shrinkage':
            cov = self._predict_shrinkage()
        elif self.method == 'ewma':
            cov = self.ewma_cov
        elif self.method == 'dcc':
            cov = self._predict_dcc()
        else:
            cov = self.sample_cov
        
        # 调整到预测周期
        cov_horizon = cov * horizon
        
        # 确保正定
        cov_horizon = self._ensure_positive_definite(cov_horizon)
        
        return cov_horizon
    
    def _predict_factor(self) -> np.ndarray:
        """使用因子模型预测: Σ = BFB' + D"""
        B = self.factor_loadings
        F = self.factor_cov
        D = self.idiosyncratic
        
        return B @ F @ B.T + D
    
    def _predict_shrinkage(self) -> np.ndarray:
        """使用收缩估计预测"""
        delta = self.shrinkage_intensity
        return delta * self.shrinkage_target + (1 - delta) * self.sample_cov
    
    def _predict_dcc(self) -> np.ndarray:
        """使用 DCC 模型预测"""
        vol = self.dcc_volatilities
        corr = self.dcc_correlation
        
        # Σ = D * R * D, where D = diag(volatilities)
        D = np.diag(vol)
        return D @ corr @ D
    
    def get_covariance_dataframe(self, horizon: int = 21) -> pd.DataFrame:
        """返回协方差矩阵 DataFrame"""
        cov = self.predict(horizon)
        return pd.DataFrame(cov, index=self.asset_names, columns=self.asset_names)
    
    def get_correlation_matrix(self, horizon: int = 21) -> np.ndarray:
        """返回相关系数矩阵"""
        cov = self.predict(horizon)
        vol = np.sqrt(np.diag(cov))
        D_inv = np.diag(1.0 / vol)
        return D_inv @ cov @ D_inv

