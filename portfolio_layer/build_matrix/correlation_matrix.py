"""
Correlation Matrix Builder - 相关性矩阵构建器

构建股票和因子之间的相关性矩阵
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import pearsonr, spearmanr


class CorrelationMatrixBuilder:
    """
    Correlation Matrix Builder - 相关性矩阵构建器
    
    构建股票收益率与因子收益率之间的相关性矩阵
    """
    
    def __init__(self, method: str = 'pearson'):
        """
        Initialize correlation matrix builder
        
        Args:
            method: 'pearson' or 'spearman'
        """
        self.method = method.lower()
        self.correlation_matrix = None
    
    def build(self,
             returns: pd.DataFrame,
             factors: pd.DataFrame) -> pd.DataFrame:
        """
        构建股票-因子相关性矩阵
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
        
        Returns:
            相关性矩阵 (N x K)
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx]
        factors_aligned = factors.loc[common_idx]
        
        # 计算相关性矩阵
        if self.method == 'pearson':
            corr_matrix = returns_aligned.corrwith(factors_aligned, axis=0)
        elif self.method == 'spearman':
            # Spearman correlation
            corr_matrix = returns_aligned.corrwith(
                factors_aligned, 
                axis=0, 
                method='spearman'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 转换为 DataFrame (N x K)
        if isinstance(corr_matrix, pd.Series):
            # 如果只有一个因子
            corr_matrix = corr_matrix.to_frame(name=factors_aligned.columns[0])
        else:
            corr_matrix = corr_matrix.T  # (N x K)
        
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def build_pairwise(self,
                      returns: pd.DataFrame,
                      factors: pd.DataFrame) -> pd.DataFrame:
        """
        构建逐对相关性矩阵（使用 scipy）
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
        
        Returns:
            相关性矩阵 (N x K)
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx].values
        factors_aligned = factors.loc[common_idx].values
        
        n_assets = returns_aligned.shape[1]
        n_factors = factors_aligned.shape[1]
        
        corr_matrix = np.zeros((n_assets, n_factors))
        
        for i in range(n_assets):
            for j in range(n_factors):
                asset_returns = returns_aligned[:, i]
                factor_returns = factors_aligned[:, j]
                
                # 移除 NaN
                mask = ~(np.isnan(asset_returns) | np.isnan(factor_returns))
                if np.sum(mask) < 2:
                    corr_matrix[i, j] = 0.0
                    continue
                
                asset_clean = asset_returns[mask]
                factor_clean = factor_returns[mask]
                
                if self.method == 'pearson':
                    corr, _ = pearsonr(asset_clean, factor_clean)
                elif self.method == 'spearman':
                    corr, _ = spearmanr(asset_clean, factor_clean)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        # 转换为 DataFrame
        corr_df = pd.DataFrame(
            corr_matrix,
            index=returns.columns,
            columns=factors.columns
        )
        
        self.correlation_matrix = corr_df
        return corr_df
    
    def build_full_correlation(self,
                              returns: pd.DataFrame,
                              factors: pd.DataFrame) -> pd.DataFrame:
        """
        构建完整的相关性矩阵（包含股票和因子）
        
        Args:
            returns: 资产收益率 (T x N)
            factors: 因子收益率 (T x K)
        
        Returns:
            完整相关性矩阵 ((N+K) x (N+K))
        """
        # 对齐数据
        common_idx = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_idx]
        factors_aligned = factors.loc[common_idx]
        
        # 合并数据
        combined = pd.concat([returns_aligned, factors_aligned], axis=1)
        
        # 计算相关性矩阵
        if self.method == 'pearson':
            corr_matrix = combined.corr()
        elif self.method == 'spearman':
            corr_matrix = combined.corr(method='spearman')
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return corr_matrix
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix"""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not built yet. Call build() first.")
        return self.correlation_matrix
    
    def analyze_correlation_strength(self) -> pd.DataFrame:
        """
        分析相关性强度
        
        Returns:
            DataFrame with correlation strength analysis
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not built yet. Call build() first.")
        
        corr = self.correlation_matrix
        
        analysis = pd.DataFrame({
            'Mean': corr.mean(axis=1),
            'Std': corr.std(axis=1),
            'Max': corr.max(axis=1),
            'Min': corr.min(axis=1),
            'Abs_Mean': corr.abs().mean(axis=1),
            'Abs_Max': corr.abs().max(axis=1)
        })
        
        return analysis


