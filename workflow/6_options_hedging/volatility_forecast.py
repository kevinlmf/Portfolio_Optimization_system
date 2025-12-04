"""
Volatility Forecast - 波动率预测

基于资产组合构建过程中的风险结构预测波动率
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class VolatilityForecaster:
    """
    波动率预测器
    
    基于组合优化过程中的风险结构（Σ, F, D）预测未来波动率
    """
    
    def __init__(self, method: str = 'garch'):
        """
        Args:
            method: 预测方法
                - 'portfolio_risk': 基于组合风险结构
                - 'garch': GARCH模型
                - 'realized': 已实现波动率
                - 'implied': 隐含波动率
        """
        self.method = method
        self.historical_vol = None
        self.forecast_vol = None
    
    def forecast_from_portfolio(self,
                               portfolio_weights: np.ndarray,
                               covariance_matrix: np.ndarray,
                               horizon: int = 21,
                               annualization: int = 252) -> float:
        """
        从投资组合风险结构预测波动率
        
        Args:
            portfolio_weights: 组合权重 (N,)
            covariance_matrix: 协方差矩阵 Σ (N x N)
            horizon: 预测期限（天数）
            annualization: 年化天数（默认252交易日）
        
        Returns:
            预测波动率（年化）
        """
        # 计算组合方差
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 年化
        annualized_vol = portfolio_std * np.sqrt(annualization)
        
        # 调整到预测期限
        horizon_vol = annualized_vol * np.sqrt(horizon / annualization)
        
        self.forecast_vol = annualized_vol
        return float(annualized_vol)
    
    def forecast_garch(self,
                      returns: pd.Series,
                      horizon: int = 21,
                      p: int = 1,
                      q: int = 1) -> float:
        """
        使用GARCH模型预测波动率
        
        Args:
            returns: 收益率序列
            horizon: 预测期限
            p: GARCH滞后阶数
            q: ARCH滞后阶数
        
        Returns:
            预测波动率（年化）
        """
        # 简化的GARCH(1,1)模型
        # 实际应用中可以使用arch库
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 50:
            # 如果数据不足，使用简单方法
            vol = returns_clean.std() * np.sqrt(252)
            return float(vol)
        
        # 计算初始方差
        variance = returns_clean.var()
        
        # 简化的GARCH(1,1)参数（可以用MLE估计）
        alpha = 0.1  # ARCH参数
        beta = 0.85  # GARCH参数
        omega = variance * (1 - alpha - beta)  # 常数项
        
        # 迭代计算条件方差
        conditional_var = [variance]
        for i in range(1, len(returns_clean)):
            var_t = omega + alpha * returns_clean.iloc[i-1]**2 + beta * conditional_var[-1]
            conditional_var.append(var_t)
        
        # 预测未来方差
        last_var = conditional_var[-1]
        forecast_var = omega / (1 - alpha - beta) + (alpha + beta)**horizon * (last_var - omega / (1 - alpha - beta))
        
        # 年化
        annualized_vol = np.sqrt(forecast_var * 252)
        
        self.forecast_vol = annualized_vol
        return float(annualized_vol)
    
    def forecast_realized_volatility(self,
                                    returns: pd.Series,
                                    window: int = 21,
                                    annualization: int = 252) -> float:
        """
        使用已实现波动率（滚动窗口）
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            annualization: 年化天数
        
        Returns:
            预测波动率（年化）
        """
        realized_vol = returns.rolling(window=window).std() * np.sqrt(annualization)
        
        if len(realized_vol.dropna()) == 0:
            vol = returns.std() * np.sqrt(annualization)
        else:
            vol = realized_vol.dropna().iloc[-1]
        
        self.forecast_vol = float(vol)
        return float(vol)
    
    def forecast_implied_volatility(self,
                                   option_prices: pd.Series,
                                   spot_price: float,
                                   strike: float,
                                   risk_free_rate: float,
                                   time_to_expiry: float) -> float:
        """
        从期权价格反推隐含波动率
        
        Args:
            option_prices: 期权价格序列
            spot_price: 现货价格
            strike: 执行价格
            risk_free_rate: 无风险利率
            time_to_expiry: 到期时间（年）
        
        Returns:
            隐含波动率
        """
        # 使用Black-Scholes反推隐含波动率
        from .option_pricing import BlackScholesPricer
        
        pricer = BlackScholesPricer()
        
        # 使用最新期权价格
        market_price = option_prices.iloc[-1] if len(option_prices) > 0 else option_prices
        
        # 二分法求解隐含波动率
        vol_low, vol_high = 0.01, 5.0
        tolerance = 1e-6
        
        for _ in range(100):  # 最多迭代100次
            vol_mid = (vol_low + vol_high) / 2
            price_mid = pricer.price(
                spot=spot_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=vol_mid,
                option_type='call'
            )
            
            if abs(price_mid - market_price) < tolerance:
                break
            
            if price_mid < market_price:
                vol_low = vol_mid
            else:
                vol_high = vol_mid
        
        implied_vol = (vol_low + vol_high) / 2
        self.forecast_vol = float(implied_vol)
        return float(implied_vol)
    
    def forecast(self,
                portfolio_weights: Optional[np.ndarray] = None,
                covariance_matrix: Optional[np.ndarray] = None,
                returns: Optional[pd.Series] = None,
                **kwargs) -> float:
        """
        通用预测接口
        
        Args:
            portfolio_weights: 组合权重
            covariance_matrix: 协方差矩阵
            returns: 收益率序列
            **kwargs: 其他参数
        
        Returns:
            预测波动率
        """
        if self.method == 'portfolio_risk':
            if portfolio_weights is None or covariance_matrix is None:
                raise ValueError("portfolio_weights and covariance_matrix required for portfolio_risk method")
            return self.forecast_from_portfolio(
                portfolio_weights, 
                covariance_matrix,
                horizon=kwargs.get('horizon', 21),
                annualization=kwargs.get('annualization', 252)
            )
        
        elif self.method == 'garch':
            if returns is None:
                raise ValueError("returns required for garch method")
            return self.forecast_garch(returns, horizon=kwargs.get('horizon', 21))
        
        elif self.method == 'realized':
            if returns is None:
                raise ValueError("returns required for realized method")
            return self.forecast_realized_volatility(
                returns,
                window=kwargs.get('window', 21),
                annualization=kwargs.get('annualization', 252)
            )
        
        elif self.method == 'implied':
            required = ['option_prices', 'spot_price', 'strike', 'risk_free_rate', 'time_to_expiry']
            if not all(k in kwargs for k in required):
                raise ValueError(f"Required kwargs for implied method: {required}")
            return self.forecast_implied_volatility(**{k: kwargs[k] for k in required})
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

