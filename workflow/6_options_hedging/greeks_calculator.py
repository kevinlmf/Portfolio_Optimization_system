"""
Greeks Calculator - 希腊字母计算器

计算期权的希腊字母：
- Δ (Delta): 对价格敏感度，控制方向性敞口
- Γ (Gamma): 对Δ的变化率，管理凸性（非线性）
- Θ (Theta): 时间衰减，捕获时间价值
- V (Vega): 对波动率敏感度，赌波动率高低
- ρ (Rho): 对利率敏感度，利率敞口管理
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal
from .option_pricing import BlackScholesPricer


class GreeksCalculator:
    """
    希腊字母计算器
    
    计算期权价格对各种风险因素的敏感度
    """
    
    def __init__(self):
        self.pricer = BlackScholesPricer()
    
    def calculate_delta(self,
                       spot: float,
                       strike: float,
                       time_to_expiry: float,
                       risk_free_rate: float,
                       volatility: float,
                       option_type: Literal['call', 'put'] = 'call',
                       dividend_yield: float = 0.0) -> float:
        """
        计算 Δ (Delta) - 对价格敏感度
        
        Delta = ∂C/∂S，衡量标的资产价格变动对期权价格的影响
        控制方向性敞口（Directional Exposure）
        
        Args:
            spot: 现货价格 S
            strike: 执行价格 K
            time_to_expiry: 到期时间 T
            risk_free_rate: 无风险利率 r
            volatility: 波动率 σ
            option_type: 期权类型
            dividend_yield: 股息率 q
        
        Returns:
            Delta值（Call: 0到1, Put: -1到0）
        """
        if time_to_expiry <= 0:
            # 到期时Delta是阶梯函数
            if option_type == 'call':
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0
        
        if volatility <= 0:
            # 无波动率时
            forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
            if option_type == 'call':
                return np.exp(-dividend_yield * time_to_expiry) if forward > strike else 0.0
            else:
                return -np.exp(-dividend_yield * time_to_expiry) if forward < strike else 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        if option_type == 'call':
            delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:  # put
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        
        return float(delta)
    
    def calculate_gamma(self,
                       spot: float,
                       strike: float,
                       time_to_expiry: float,
                       risk_free_rate: float,
                       volatility: float,
                       dividend_yield: float = 0.0) -> float:
        """
        计算 Γ (Gamma) - 对Δ的变化率
        
        Gamma = ∂²C/∂S² = ∂Δ/∂S，衡量Delta对价格变动的敏感度
        管理凸性（非线性），对冲Delta风险的重要指标
        
        Args:
            参数同calculate_delta
        
        Returns:
            Gamma值（通常为正）
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        gamma = np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        
        return float(gamma)
    
    def calculate_theta(self,
                       spot: float,
                       strike: float,
                       time_to_expiry: float,
                       risk_free_rate: float,
                       volatility: float,
                       option_type: Literal['call', 'put'] = 'call',
                       dividend_yield: float = 0.0) -> float:
        """
        计算 Θ (Theta) - 时间衰减
        
        Theta = -∂C/∂T，衡量时间流逝对期权价值的影响
        捕获时间价值衰减
        
        Args:
            参数同calculate_delta
        
        Returns:
            Theta值（通常为负，表示时间衰减）
        """
        if time_to_expiry <= 0:
            return 0.0
        
        if volatility <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'call':
            theta = (
                -spot * norm.pdf(d1) * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * np.sqrt(time_to_expiry))
                - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                + dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
            )
        else:  # put
            theta = (
                -spot * norm.pdf(d1) * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * np.sqrt(time_to_expiry))
                + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                - dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            )
        
        # Theta通常按天计算
        return float(theta / 365.0)
    
    def calculate_vega(self,
                      spot: float,
                      strike: float,
                      time_to_expiry: float,
                      risk_free_rate: float,
                      volatility: float,
                      dividend_yield: float = 0.0) -> float:
        """
        计算 V (Vega) - 对波动率敏感度
        
        Vega = ∂C/∂σ，衡量波动率变动对期权价格的影响
        赌波动率高低的关键指标
        
        Args:
            参数同calculate_delta
        
        Returns:
            Vega值（通常为正，表示波动率增加对期权有利）
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        vega = spot * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry)
        
        # Vega通常按波动率变动1%计算
        return float(vega / 100.0)
    
    def calculate_rho(self,
                     spot: float,
                     strike: float,
                     time_to_expiry: float,
                     risk_free_rate: float,
                     volatility: float,
                     option_type: Literal['call', 'put'] = 'call',
                     dividend_yield: float = 0.0) -> float:
        """
        计算 ρ (Rho) - 对利率敏感度
        
        Rho = ∂C/∂r，衡量利率变动对期权价格的影响
        利率敞口管理
        
        Args:
            参数同calculate_delta
        
        Returns:
            Rho值（Call通常为正，Put通常为负）
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d2 = (np.log(spot / strike) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        if option_type == 'call':
            rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # put
            rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        
        # Rho通常按利率变动1%计算
        return float(rho / 100.0)
    
    def calculate_all(self,
                     spot: float,
                     strike: float,
                     time_to_expiry: float,
                     risk_free_rate: float,
                     volatility: float,
                     option_type: Literal['call', 'put'] = 'call',
                     dividend_yield: float = 0.0) -> Dict[str, float]:
        """
        计算所有希腊字母
        
        Returns:
            包含所有希腊字母的字典
        """
        greeks = {
            'delta': self.calculate_delta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield),
            'gamma': self.calculate_gamma(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield),
            'theta': self.calculate_theta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield),
            'vega': self.calculate_vega(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield),
            'rho': self.calculate_rho(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield),
        }
        
        return greeks
    
    def calculate_portfolio_greeks(self,
                                  positions: Dict[str, Dict]) -> Dict[str, float]:
        """
        计算投资组合的总希腊字母
        
        Args:
            positions: 持仓字典，格式为：
                {
                    'option_id_1': {
                        'spot': float,
                        'strike': float,
                        'time_to_expiry': float,
                        'risk_free_rate': float,
                        'volatility': float,
                        'option_type': 'call' or 'put',
                        'quantity': float,  # 持仓数量
                        'dividend_yield': float (optional)
                    },
                    ...
                }
        
        Returns:
            组合总希腊字母
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for option_id, params in positions.items():
            quantity = params.get('quantity', 1.0)
            dividend_yield = params.get('dividend_yield', 0.0)
            
            greeks = self.calculate_all(
                spot=params['spot'],
                strike=params['strike'],
                time_to_expiry=params['time_to_expiry'],
                risk_free_rate=params['risk_free_rate'],
                volatility=params['volatility'],
                option_type=params.get('option_type', 'call'),
                dividend_yield=dividend_yield
            )
            
            total_delta += greeks['delta'] * quantity
            total_gamma += greeks['gamma'] * quantity
            total_theta += greeks['theta'] * quantity
            total_vega += greeks['vega'] * quantity
            total_rho += greeks['rho'] * quantity
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho,
        }

