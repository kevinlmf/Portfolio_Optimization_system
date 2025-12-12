"""
Factor Analyzer - 因子分析器

分析因子的统计特性、相关性、稳定性等
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class FactorAnalyzer:
    """
    Factor Analyzer - 分析因子的统计特性
    
    分析因子：
    1. 统计特性（均值、波动率、夏普比率）
    2. 相关性分析
    3. 稳定性分析
    """
    
    def __init__(self):
        """Initialize factor analyzer"""
        pass
    
    def analyze_statistics(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze statistical properties of factors
        
        分析因子的统计特性
        
        Args:
            factors: Factor returns DataFrame (T x K)
        
        Returns:
            Statistics DataFrame
        """
        stats_dict = {
            'Mean': factors.mean(),
            'Std': factors.std(),
            'Sharpe': factors.mean() / factors.std(),
            'Skewness': factors.skew(),
            'Kurtosis': factors.kurtosis(),
            'Min': factors.min(),
            'Max': factors.max()
        }
        
        return pd.DataFrame(stats_dict)
    
    def analyze_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlation between factors
        
        分析因子之间的相关性
        
        Args:
            factors: Factor returns DataFrame (T x K)
        
        Returns:
            Correlation matrix
        """
        return factors.corr()
    
    def analyze_stability(self,
                         factors: pd.DataFrame,
                         window: int = 252) -> pd.DataFrame:
        """
        Analyze factor stability over time
        
        分析因子的时间稳定性
        
        Args:
            factors: Factor returns DataFrame (T x K)
            window: Rolling window size
        
        Returns:
            Stability metrics DataFrame
        """
        stability_metrics = {}
        
        for factor_name in factors.columns:
            factor_returns = factors[factor_name]
            
            # Rolling volatility
            rolling_vol = factor_returns.rolling(window=window).std()
            vol_stability = 1 / (1 + rolling_vol.std())  # Higher is more stable
            
            # Rolling Sharpe
            rolling_sharpe = (factor_returns.rolling(window=window).mean() / 
                            factor_returns.rolling(window=window).std())
            sharpe_stability = 1 / (1 + rolling_sharpe.std())
            
            stability_metrics[factor_name] = {
                'Volatility_Stability': vol_stability,
                'Sharpe_Stability': sharpe_stability
            }
        
        return pd.DataFrame(stability_metrics).T
    
    def comprehensive_analysis(self, factors: pd.DataFrame) -> Dict:
        """
        Comprehensive factor analysis
        
        综合因子分析
        
        Args:
            factors: Factor returns DataFrame (T x K)
        
        Returns:
            Analysis results dictionary
        """
        results = {
            'statistics': self.analyze_statistics(factors),
            'correlation': self.analyze_correlation(factors),
            'stability': self.analyze_stability(factors)
        }
        
        return results


