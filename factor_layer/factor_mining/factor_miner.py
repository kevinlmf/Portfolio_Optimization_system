"""
Factor Miner - 因子挖掘器

从股票收益率数据中挖掘驱动股票的共同因子
使用主成分分析（PCA）、因子分析等方法
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FactorMiner:
    """
    Factor Miner - 从股票数据中挖掘共同因子
    
    使用多种方法挖掘驱动股票的共同因子：
    1. PCA (主成分分析)
    2. Factor Analysis (因子分析)
    3. Statistical Factor Models (统计因子模型)
    """
    
    def __init__(self, method: str = 'pca', n_factors: Optional[int] = None):
        """
        Initialize factor miner
        
        Args:
            method: Mining method ('pca', 'factor_analysis', 'statistical')
            n_factors: Number of factors to extract (None for auto)
        """
        self.method = method.lower()
        self.n_factors = n_factors
        self.factors = None
        self.factor_loadings = None
        self.explained_variance = None
        
    def mine_factors(self,
                    returns: pd.DataFrame,
                    n_factors: Optional[int] = None) -> pd.DataFrame:
        """
        Mine common factors from stock returns
        
        从股票收益率中挖掘共同因子
        
        Args:
            returns: Stock returns DataFrame (T x N)
            n_factors: Number of factors to extract (uses self.n_factors if None)
        
        Returns:
            Factor returns DataFrame (T x K)
        """
        if n_factors is None:
            n_factors = self.n_factors
        
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)
        
        if self.method == 'pca':
            return self._mine_pca(returns_scaled, returns.index, n_factors)
        elif self.method == 'factor_analysis':
            return self._mine_factor_analysis(returns_scaled, returns.index, n_factors)
        elif self.method == 'statistical':
            return self._mine_statistical(returns_scaled, returns.index, n_factors)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mine_pca(self,
                  returns_scaled: np.ndarray,
                  dates: pd.DatetimeIndex,
                  n_factors: Optional[int]) -> pd.DataFrame:
        """Extract factors using PCA"""
        # Determine number of factors
        if n_factors is None:
            # Use number of factors that explain 80% variance
            pca_temp = PCA()
            pca_temp.fit(returns_scaled)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_factors = np.argmax(cumsum_var >= 0.8) + 1
            n_factors = min(n_factors, returns_scaled.shape[1] - 1)
        
        # Fit PCA
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(returns_scaled)
        
        # Store results
        self.factor_loadings = pd.DataFrame(
            pca.components_.T,
            index=range(returns_scaled.shape[1]),
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        self.explained_variance = pca.explained_variance_ratio_
        
        # Create factor returns DataFrame
        factors_df = pd.DataFrame(
            factors,
            index=dates,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        self.factors = factors_df
        return factors_df
    
    def _mine_factor_analysis(self,
                              returns_scaled: np.ndarray,
                              dates: pd.DatetimeIndex,
                              n_factors: Optional[int]) -> pd.DataFrame:
        """Extract factors using Factor Analysis"""
        if n_factors is None:
            n_factors = min(5, returns_scaled.shape[1] - 1)
        
        # Fit Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, max_iter=1000)
        factors = fa.fit_transform(returns_scaled)
        
        # Store results
        self.factor_loadings = pd.DataFrame(
            fa.components_.T,
            index=range(returns_scaled.shape[1]),
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Create factor returns DataFrame
        factors_df = pd.DataFrame(
            factors,
            index=dates,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        self.factors = factors_df
        return factors_df
    
    def _mine_statistical(self,
                         returns_scaled: np.ndarray,
                         dates: pd.DatetimeIndex,
                         n_factors: Optional[int]) -> pd.DataFrame:
        """Extract factors using statistical factor model"""
        # Use eigenvalue decomposition of covariance matrix
        cov_matrix = np.cov(returns_scaled.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of factors
        if n_factors is None:
            # Use factors that explain significant variance
            total_var = np.sum(eigenvalues)
            cumsum_var = np.cumsum(eigenvalues) / total_var
            n_factors = np.argmax(cumsum_var >= 0.8) + 1
            n_factors = min(n_factors, returns_scaled.shape[1] - 1)
        
        # Extract factors
        factors = returns_scaled @ eigenvectors[:, :n_factors]
        
        # Store results
        self.factor_loadings = pd.DataFrame(
            eigenvectors[:, :n_factors],
            index=range(returns_scaled.shape[1]),
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        self.explained_variance = eigenvalues[:n_factors] / np.sum(eigenvalues)
        
        # Create factor returns DataFrame
        factors_df = pd.DataFrame(
            factors,
            index=dates,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        self.factors = factors_df
        return factors_df
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """Get factor loadings matrix"""
        if self.factor_loadings is None:
            raise ValueError("Factors not mined yet. Call mine_factors() first.")
        return self.factor_loadings
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio"""
        if self.explained_variance is None:
            raise ValueError("Factors not mined yet. Call mine_factors() first.")
        return self.explained_variance
    
    def summarize(self) -> Dict:
        """Summarize mining results"""
        if self.factors is None:
            return {"status": "No factors mined yet"}
        
        summary = {
            "method": self.method,
            "n_factors": len(self.factors.columns),
            "n_observations": len(self.factors),
            "explained_variance": self.explained_variance.tolist() if self.explained_variance is not None else None,
            "total_explained_variance": float(np.sum(self.explained_variance)) if self.explained_variance is not None else None
        }
        
        return summary


