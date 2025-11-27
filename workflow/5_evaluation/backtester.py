"""
Backtester - 回测器

对投资组合进行回测
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Callable
from .performance_metrics import PerformanceMetrics


class Backtester:
    """
    Backtester - 回测器
    
    对投资组合策略进行回测
    """
    
    def __init__(self,
                 rebalance_frequency: str = 'monthly',
                 transaction_cost: float = 0.001):
        """
        Args:
            rebalance_frequency: 再平衡频率 ('daily', 'weekly', 'monthly', 'quarterly')
            transaction_cost: 交易成本（每次交易的百分比）
        """
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.metrics_calculator = PerformanceMetrics()
    
    def backtest(self,
                returns: pd.DataFrame,
                weight_generator: Callable,
                **kwargs) -> pd.DataFrame:
        """
        回测投资组合策略
        
        Args:
            returns: 历史收益率 (T x N)
            weight_generator: 权重生成函数，接受 (returns, **kwargs) 返回权重
            **kwargs: 传递给权重生成函数的参数
        
        Returns:
            回测结果 DataFrame（包含组合收益率、累计收益等）
        """
        # 确定再平衡日期
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        # 初始化
        portfolio_returns = []
        portfolio_values = [1.0]
        current_weights = None
        
        for i, date in enumerate(returns.index):
            # 检查是否需要再平衡
            if date in rebalance_dates or i == 0:
                # 使用历史数据到当前日期
                historical_returns = returns.loc[:date]
                
                # 生成新权重
                try:
                    new_weights = weight_generator(historical_returns, **kwargs)
                except:
                    # 如果生成失败，使用等权重
                    new_weights = np.ones(len(returns.columns)) / len(returns.columns)
                
                # 计算换手成本
                if current_weights is not None:
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    cost = turnover * self.transaction_cost
                else:
                    cost = 0.0
                
                current_weights = new_weights
            else:
                cost = 0.0
            
            # 计算当日组合收益率
            if i == 0:
                portfolio_returns.append(0.0)
            else:
                daily_return = np.dot(current_weights, returns.loc[date].values) - cost
                portfolio_returns.append(daily_return)
                portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
        
        # 构建结果 DataFrame
        results = pd.DataFrame({
            'portfolio_return': portfolio_returns,
            'cumulative_value': portfolio_values
        }, index=returns.index)
        
        return results
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取再平衡日期"""
        if self.rebalance_frequency == 'daily':
            return dates
        elif self.rebalance_frequency == 'weekly':
            # 每周第一个交易日
            return dates[dates.weekday == 0] if hasattr(dates, 'weekday') else dates[::5]
        elif self.rebalance_frequency == 'monthly':
            # 每月第一个交易日
            df = pd.DataFrame({'date': dates}, index=dates)
            monthly_first = df.groupby([df.index.year, df.index.month]).first()['date']
            return pd.DatetimeIndex(monthly_first.values)
        elif self.rebalance_frequency == 'quarterly':
            # 每季度第一个交易日
            df = pd.DataFrame({'date': dates}, index=dates)
            quarterly_first = df.groupby([df.index.year, df.index.quarter]).first()['date']
            return pd.DatetimeIndex(quarterly_first.values)
        else:
            return dates
    
    def evaluate_backtest(self, backtest_results: pd.DataFrame) -> Dict:
        """
        评估回测结果
        
        Args:
            backtest_results: 回测结果 DataFrame
        
        Returns:
            评估指标字典
        """
        returns_series = backtest_results['portfolio_return']
        return self.metrics_calculator.calculate_all(returns_series)

