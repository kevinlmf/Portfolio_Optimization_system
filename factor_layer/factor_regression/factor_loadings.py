"""
Factor Loadings Estimator - 因子载荷估计器

估计股票收益受哪些 factor 驱动
使用回归方法估计因子载荷矩阵 B
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


class FactorLoadingsEstimator:
    """
    Factor Loadings Estimator - 因子载荷估计器
    
    估计股票收益率与因子收益率之间的关系
    构建股票-因子关系矩阵 B (N x K)
    """
    
    def __init__(self, method: str = 'ols', regularization: float = 0.0):
        """
        Initialize factor loadings estimator
        
        Args:
            method: 'ols' or 'ridge'
            regularization: Ridge 正则化强度
        """
        self.method = method
        self.regularization = regularization
        self.factor_loadings = None
    
    def estimate(self, 
                returns: pd.DataFrame,
                factors: pd.DataFrame) -> pd.DataFrame:
        """
        估计因子载荷矩阵 B
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
        
        Returns:
            因子载荷矩阵 B (N x K)
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        Y = returns.loc[common_idx]
        X = factors.loc[common_idx]
        
        # 标准化因子（可选）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # 选择回归方法
        if self.method == 'ols':
            model = LinearRegression()
        elif self.method == 'ridge':
            model = Ridge(alpha=self.regularization)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 对每个资产回归
        model.fit(X_scaled, Y)
        
        # B 矩阵：每行是一个资产的因子载荷
        # model.coef_ 的形状是 (K, N)，其中 K 是因子数，N 是资产数
        # 需要转置为 (N, K)
        coef = model.coef_
        
        # 处理 coef 的形状
        if coef.ndim == 1:
            # 只有一个因子时，coef_ 是 (N,)，需要 reshape 为 (N, 1)
            coef = coef.reshape(-1, 1)
        else:
            # coef 是 (K, N)，转置为 (N, K)
            coef = coef.T
        
        # 确保形状正确
        n_assets = len(returns.columns)
        n_factors = len(factors.columns)
        if coef.shape != (n_assets, n_factors):
            # 如果形状不匹配，尝试转置
            if coef.shape == (n_factors, n_assets):
                coef = coef.T
        
        B = pd.DataFrame(
            coef,  # (N x K)
            index=returns.columns,
            columns=factors.columns
        )
        
        self.factor_loadings = B
        return B
    
    def estimate_with_alpha(self,
                            returns: pd.DataFrame,
                            factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        估计因子载荷和 alpha
        
        Returns:
            (B, alpha) 其中 alpha 是 (N,) 向量
        """
        common_idx = returns.index.intersection(factors.index)
        Y = returns.loc[common_idx]
        X = factors.loc[common_idx]
        
        # 标准化因子
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        if self.method == 'ols':
            model = LinearRegression()
        else:
            model = Ridge(alpha=self.regularization)
        
        model.fit(X_scaled, Y)
        
        # 处理 coef_ 的形状
        coef = model.coef_
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        else:
            # coef 是 (K, N) 或 (N, K)，需要检查
            n_assets = len(returns.columns)
            n_factors = len(factors.columns)
            if coef.shape == (n_factors, n_assets):
                coef = coef.T  # 转置为 (N, K)
        
        B = pd.DataFrame(
            coef,
            index=returns.columns,
            columns=factors.columns
        )
        
        alpha = pd.Series(
            model.intercept_,
            index=returns.columns
        )
        
        return B, alpha
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """Get factor loadings matrix"""
        if self.factor_loadings is None:
            raise ValueError("Factor loadings not estimated yet. Call estimate() first.")
        return self.factor_loadings

