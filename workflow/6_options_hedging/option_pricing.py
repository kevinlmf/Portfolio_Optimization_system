"""
Option Pricing Models - 期权定价模型

实现Black-Scholes等期权定价模型
"""

import numpy as np
from scipy.stats import norm
from typing import Literal
from abc import ABC, abstractmethod


class OptionPricingModel(ABC):
    """期权定价模型基类"""
    
    @abstractmethod
    def price(self, **kwargs) -> float:
        """计算期权价格"""
        pass


class BlackScholesPricer(OptionPricingModel):
    """
    Black-Scholes期权定价模型
    
    用于计算欧式期权的理论价格
    """
    
    def price(self,
             spot: float,
             strike: float,
             time_to_expiry: float,
             risk_free_rate: float,
             volatility: float,
             option_type: Literal['call', 'put'] = 'call',
             dividend_yield: float = 0.0) -> float:
        """
        计算Black-Scholes期权价格
        
        Args:
            spot: 现货价格 S
            strike: 执行价格 K
            time_to_expiry: 到期时间 T（年）
            risk_free_rate: 无风险利率 r
            volatility: 波动率 σ（年化）
            option_type: 期权类型 'call' 或 'put'
            dividend_yield: 股息率 q（默认0）
        
        Returns:
            期权理论价格
        """
        if time_to_expiry <= 0:
            # 到期时的内在价值
            if option_type == 'call':
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)
        
        if volatility <= 0:
            # 无波动率时的价格
            forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
            if option_type == 'call':
                return max(forward - strike, 0) * np.exp(-risk_free_rate * time_to_expiry)
            else:
                return max(strike - forward, 0) * np.exp(-risk_free_rate * time_to_expiry)
        
        # 计算d1和d2
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # 计算价格
        if option_type == 'call':
            price = spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # put
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        
        return float(max(price, 0))
    
    def price_multi(self,
                   spot: float,
                   strikes: np.ndarray,
                   time_to_expiry: float,
                   risk_free_rate: float,
                   volatility: float,
                   option_type: Literal['call', 'put'] = 'call',
                   dividend_yield: float = 0.0) -> np.ndarray:
        """
        批量计算期权价格
        
        Args:
            spot: 现货价格
            strikes: 执行价格数组
            其他参数同price方法
        
        Returns:
            期权价格数组
        """
        prices = np.array([
            self.price(
                spot=spot,
                strike=k,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type,
                dividend_yield=dividend_yield
            )
            for k in strikes
        ])
        return prices


class BinomialPricer(OptionPricingModel):
    """
    二叉树期权定价模型
    
    适用于美式期权和路径依赖期权
    """
    
    def price(self,
             spot: float,
             strike: float,
             time_to_expiry: float,
             risk_free_rate: float,
             volatility: float,
             option_type: Literal['call', 'put'] = 'call',
             n_steps: int = 100,
             american: bool = True,
             dividend_yield: float = 0.0) -> float:
        """
        计算二叉树期权价格
        
        Args:
            spot: 现货价格
            strike: 执行价格
            time_to_expiry: 到期时间（年）
            risk_free_rate: 无风险利率
            volatility: 波动率
            option_type: 期权类型
            n_steps: 二叉树步数
            american: 是否为美式期权
            dividend_yield: 股息率
        
        Returns:
            期权价格
        """
        dt = time_to_expiry / n_steps
        u = np.exp(volatility * np.sqrt(dt))  # 上涨因子
        d = 1 / u  # 下跌因子
        p = (np.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)  # 风险中性概率
        
        # 构建价格树
        stock_prices = np.zeros(n_steps + 1)
        option_values = np.zeros(n_steps + 1)
        
        # 计算到期时的期权价值
        for j in range(n_steps + 1):
            stock_price = spot * (u ** (n_steps - j)) * (d ** j)
            if option_type == 'call':
                option_values[j] = max(stock_price - strike, 0)
            else:
                option_values[j] = max(strike - stock_price, 0)
        
        # 反向递推计算期权价格
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                stock_price = spot * (u ** (i - j)) * (d ** j)
                
                # 风险中性定价
                option_value = np.exp(-risk_free_rate * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
                
                # 美式期权考虑提前执行
                if american:
                    if option_type == 'call':
                        intrinsic_value = max(stock_price - strike, 0)
                    else:
                        intrinsic_value = max(strike - stock_price, 0)
                    option_value = max(option_value, intrinsic_value)
                
                option_values[j] = option_value
        
        return float(option_values[0])

