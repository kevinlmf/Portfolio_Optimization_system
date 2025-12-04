"""
Performance Metrics - 表现指标

计算投资组合的各种表现指标
"""

import numpy as np
import pandas as pd
from typing import Dict


class PerformanceMetrics:
    """表现指标计算器"""
    
    def calculate_all(self, returns: pd.Series) -> Dict:
        """
        计算所有指标
        
        Args:
            returns: 投资组合收益率序列
        
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['total_return'] = self.total_return(returns)
        metrics['annualized_return'] = self.annualized_return(returns)
        metrics['volatility'] = self.volatility(returns)
        metrics['annualized_volatility'] = self.annualized_volatility(returns)
        
        # 风险调整收益
        metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calmar_ratio(returns)
        
        # 回撤
        metrics['max_drawdown'] = self.max_drawdown(returns)
        metrics['avg_drawdown'] = self.avg_drawdown(returns)
        
        # 其他
        metrics['win_rate'] = self.win_rate(returns)
        metrics['skewness'] = self.skewness(returns)
        metrics['kurtosis'] = self.kurtosis(returns)
        
        return metrics
    
    def total_return(self, returns: pd.Series) -> float:
        """总收益率"""
        return float((1 + returns).prod() - 1)
    
    def annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """年化收益率"""
        total_return = self.total_return(returns)
        n_years = len(returns) / periods_per_year
        return float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    
    def volatility(self, returns: pd.Series) -> float:
        """波动率"""
        return float(returns.std())
    
    def annualized_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """年化波动率"""
        return float(returns.std() * np.sqrt(periods_per_year))
    
    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Sharpe 比率"""
        excess_return = self.annualized_return(returns, periods_per_year) - risk_free_rate
        vol = self.annualized_volatility(returns, periods_per_year)
        return float(excess_return / vol) if vol > 0 else 0.0
    
    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Sortino 比率（只考虑下行风险）"""
        excess_return = self.annualized_return(returns, periods_per_year) - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0.0
        return float(excess_return / downside_std) if downside_std > 0 else 0.0
    
    def calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calmar 比率（年化收益 / 最大回撤）"""
        annual_return = self.annualized_return(returns, periods_per_year)
        max_dd = self.max_drawdown(returns)
        return float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0
    
    def max_drawdown(self, returns: pd.Series) -> float:
        """最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def avg_drawdown(self, returns: pd.Series) -> float:
        """平均回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown_negative = drawdown[drawdown < 0]
        return float(drawdown_negative.mean()) if len(drawdown_negative) > 0 else 0.0
    
    def win_rate(self, returns: pd.Series) -> float:
        """胜率"""
        return float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0
    
    def skewness(self, returns: pd.Series) -> float:
        """偏度"""
        return float(returns.skew())
    
    def kurtosis(self, returns: pd.Series) -> float:
        """峰度"""
        return float(returns.kurtosis())


