"""
Factor Selector - 因子选择器

从大量候选因子中选择最重要的因子
使用 IC 分析、回归方法、机器学习等方法
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FactorSelector:
    """
    Factor Selector - 选择最重要的因子
    
    使用多种方法选择因子：
    1. IC Analysis (信息系数分析)
    2. LASSO/Elastic Net
    3. Random Forest
    4. F-test
    """
    
    def __init__(self):
        """Initialize factor selector"""
        self.selected_factors = None
        self.importance_scores = {}
        
    def select_by_ic(self,
                    factors: pd.DataFrame,
                    returns: pd.DataFrame,
                    top_n: int = 10,
                    method: str = 'spearman') -> List[str]:
        """
        Select factors by Information Coefficient (IC)
        
        使用信息系数选择因子
        
        Args:
            factors: Factor returns DataFrame (T x K)
            returns: Stock returns DataFrame (T x N)
            top_n: Number of top factors to select
            method: Correlation method ('pearson', 'spearman')
        
        Returns:
            List of selected factor names
        """
        # Calculate IC for each factor
        ic_scores = {}
        
        for factor_name in factors.columns:
            factor_returns = factors[factor_name]
            # Calculate IC as correlation with portfolio return
            portfolio_return = returns.mean(axis=1)
            
            if method == 'spearman':
                ic, _ = stats.spearmanr(factor_returns, portfolio_return)
            else:
                ic, _ = stats.pearsonr(factor_returns, portfolio_return)
            
            ic_scores[factor_name] = abs(ic) if not np.isnan(ic) else 0
        
        # Sort by IC
        sorted_factors = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_factors = [f[0] for f in sorted_factors[:top_n]]
        self.importance_scores['ic'] = dict(sorted_factors)
        
        return self.selected_factors
    
    def select_by_lasso(self,
                       factors: pd.DataFrame,
                       returns: pd.DataFrame,
                       top_n: int = 10) -> List[str]:
        """
        Select factors using LASSO regression
        
        使用 LASSO 回归选择因子
        
        Args:
            factors: Factor returns DataFrame (T x K)
            returns: Stock returns DataFrame (T x N)
            top_n: Number of top factors to select
        
        Returns:
            List of selected factor names
        """
        # Use portfolio return as target
        target = returns.mean(axis=1).values
        X = factors.values
        
        # Fit LASSO
        lasso = LassoCV(cv=5, max_iter=1000)
        lasso.fit(X, target)
        
        # Get coefficients
        coefficients = np.abs(lasso.coef_)
        
        # Select top factors
        top_indices = np.argsort(coefficients)[::-1][:top_n]
        self.selected_factors = [factors.columns[i] for i in top_indices]
        self.importance_scores['lasso'] = {
            factors.columns[i]: float(coefficients[i]) 
            for i in top_indices
        }
        
        return self.selected_factors
    
    def select_by_random_forest(self,
                               factors: pd.DataFrame,
                               returns: pd.DataFrame,
                               top_n: int = 10,
                               n_estimators: int = 100) -> List[str]:
        """
        Select factors using Random Forest
        
        使用随机森林选择因子
        
        Args:
            factors: Factor returns DataFrame (T x K)
            returns: Stock returns DataFrame (T x N)
            top_n: Number of top factors to select
            n_estimators: Number of trees
        
        Returns:
            List of selected factor names
        """
        # Use portfolio return as target
        target = returns.mean(axis=1).values
        X = factors.values
        
        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X, target)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Select top factors
        top_indices = np.argsort(importance)[::-1][:top_n]
        self.selected_factors = [factors.columns[i] for i in top_indices]
        self.importance_scores['random_forest'] = {
            factors.columns[i]: float(importance[i]) 
            for i in top_indices
        }
        
        return self.selected_factors
    
    def select_by_f_test(self,
                        factors: pd.DataFrame,
                        returns: pd.DataFrame,
                        top_n: int = 10) -> List[str]:
        """
        Select factors using F-test
        
        使用 F 检验选择因子
        
        Args:
            factors: Factor returns DataFrame (T x K)
            returns: Stock returns DataFrame (T x N)
            top_n: Number of top factors to select
        
        Returns:
            List of selected factor names
        """
        # Use portfolio return as target
        target = returns.mean(axis=1).values
        X = factors.values
        
        # F-test
        selector = SelectKBest(score_func=f_regression, k=top_n)
        selector.fit(X, target)
        
        # Get scores
        scores = selector.scores_
        
        # Select top factors
        top_indices = np.argsort(scores)[::-1][:top_n]
        self.selected_factors = [factors.columns[i] for i in top_indices]
        self.importance_scores['f_test'] = {
            factors.columns[i]: float(scores[i]) 
            for i in top_indices
        }
        
        return self.selected_factors
    
    def comprehensive_selection(self,
                               factors: pd.DataFrame,
                               returns: pd.DataFrame,
                               top_n: int = 10) -> List[str]:
        """
        Comprehensive factor selection using multiple methods
        
        综合使用多种方法选择因子
        
        Args:
            factors: Factor returns DataFrame (T x K)
            returns: Stock returns DataFrame (T x N)
            top_n: Number of top factors to select
        
        Returns:
            List of selected factor names
        """
        # Run all selection methods
        ic_factors = self.select_by_ic(factors, returns, top_n)
        lasso_factors = self.select_by_lasso(factors, returns, top_n)
        rf_factors = self.select_by_random_forest(factors, returns, top_n)
        f_test_factors = self.select_by_f_test(factors, returns, top_n)
        
        # Combine results (voting)
        all_factors = ic_factors + lasso_factors + rf_factors + f_test_factors
        factor_votes = {}
        
        for factor in all_factors:
            factor_votes[factor] = factor_votes.get(factor, 0) + 1
        
        # Select factors with most votes
        sorted_factors = sorted(factor_votes.items(), key=lambda x: x[1], reverse=True)
        self.selected_factors = [f[0] for f in sorted_factors[:top_n]]
        
        return self.selected_factors
    
    def get_selected_factors(self) -> List[str]:
        """Get selected factors"""
        if self.selected_factors is None:
            raise ValueError("No factors selected yet. Call selection method first.")
        return self.selected_factors
    
    def get_importance_scores(self) -> Dict:
        """Get importance scores from all methods"""
        return self.importance_scores


