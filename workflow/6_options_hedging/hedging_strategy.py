"""
Hedging Strategy - 对冲策略

基于希腊字母进行风险对冲
"""

import numpy as np
from typing import Dict, List, Optional, Literal
from abc import ABC, abstractmethod
from .greeks_calculator import GreeksCalculator


class HedgingStrategy(ABC):
    """对冲策略基类"""
    
    def __init__(self):
        self.greeks_calc = GreeksCalculator()
    
    @abstractmethod
    def calculate_hedge(self, **kwargs) -> Dict:
        """计算对冲方案"""
        pass


class DeltaHedgingStrategy(HedgingStrategy):
    """
    Delta对冲策略
    
    通过买卖标的资产使组合Delta为0，消除方向性风险
    """
    
    def calculate_hedge(self,
                       portfolio_delta: float,
                       spot_price: float,
                       contract_size: float = 100.0) -> Dict:
        """
        计算Delta对冲需要的标的资产数量
        
        Args:
            portfolio_delta: 组合总Delta
            spot_price: 标的资产现价
            contract_size: 合约规模（默认100股/份）
        
        Returns:
            对冲方案字典
        """
        # 需要卖出（负Delta）或买入（正Delta）的标的资产数量
        # 如果组合Delta为正，需要卖出标的资产来对冲
        hedge_quantity = -portfolio_delta * contract_size
        
        hedge_value = abs(hedge_quantity) * spot_price
        
        return {
            'strategy': 'delta_hedging',
            'hedge_quantity': float(hedge_quantity),  # 负值表示卖出，正值表示买入
            'hedge_value': float(hedge_value),
            'target_delta': 0.0,
            'current_delta': float(portfolio_delta),
            'hedge_ratio': float(abs(hedge_quantity) / contract_size) if contract_size > 0 else 0.0,
        }
    
    def hedge_portfolio(self,
                       option_positions: Dict[str, Dict],
                       spot_price: float,
                       contract_size: float = 100.0) -> Dict:
        """
        对冲整个期权组合
        
        Args:
            option_positions: 期权持仓（格式同GreeksCalculator.calculate_portfolio_greeks）
            spot_price: 标的资产现价
            contract_size: 合约规模
        
        Returns:
            对冲方案
        """
        portfolio_greeks = self.greeks_calc.calculate_portfolio_greeks(option_positions)
        portfolio_delta = portfolio_greeks['delta']
        
        return self.calculate_hedge(portfolio_delta, spot_price, contract_size)


class GammaHedgingStrategy(HedgingStrategy):
    """
    Gamma对冲策略
    
    通过交易其他期权来管理Gamma风险
    """
    
    def calculate_hedge(self,
                       portfolio_gamma: float,
                       hedge_option_gamma: float,
                       portfolio_delta: float = 0.0,
                       target_gamma: float = 0.0) -> Dict:
        """
        计算Gamma对冲需要的期权数量
        
        Args:
            portfolio_gamma: 组合总Gamma
            hedge_option_gamma: 对冲期权 Gamma
            portfolio_delta: 组合当前Delta（用于调整）
            target_gamma: 目标Gamma（默认0）
        
        Returns:
            对冲方案
        """
        if abs(hedge_option_gamma) < 1e-10:
            return {
                'strategy': 'gamma_hedging',
                'error': 'Hedge option gamma is too small',
                'hedge_quantity': 0.0,
            }
        
        # 需要的对冲期权数量
        hedge_quantity = -(portfolio_gamma - target_gamma) / hedge_option_gamma
        
        return {
            'strategy': 'gamma_hedging',
            'hedge_quantity': float(hedge_quantity),
            'target_gamma': float(target_gamma),
            'current_gamma': float(portfolio_gamma),
            'hedge_option_gamma': float(hedge_option_gamma),
            'adjustment_delta': float(hedge_quantity * self._estimate_delta_from_gamma(hedge_option_gamma, portfolio_delta)),
        }
    
    def _estimate_delta_from_gamma(self, gamma: float, current_delta: float) -> float:
        """从Gamma估算Delta变化（简化）"""
        # 这是一个简化估算，实际中需要更多信息
        return gamma * 0.5  # 假设的价格变化幅度


class GreeksHedgingStrategy(HedgingStrategy):
    """
    多希腊字母对冲策略
    
    同时管理Delta、Gamma、Vega等多个风险因子
    """
    
    def calculate_hedge(self,
                       portfolio_greeks: Dict[str, float],
                       hedge_options: List[Dict],
                       target_greeks: Optional[Dict[str, float]] = None) -> Dict:
        """
        使用多个期权同时对冲多个希腊字母
        
        Args:
            portfolio_greeks: 组合的希腊字母 {'delta': float, 'gamma': float, ...}
            hedge_options: 可用对冲期权列表，每个包含所有希腊字母
            target_greeks: 目标希腊字母（默认全部为0）
        
        Returns:
            对冲方案
        """
        if target_greeks is None:
            target_greeks = {key: 0.0 for key in portfolio_greeks.keys()}
        
        # 需要对冲的希腊字母差异
        hedge_needs = {
            key: target_greeks.get(key, 0.0) - portfolio_greeks.get(key, 0.0)
            for key in portfolio_greeks.keys()
        }
        
        # 简化方案：使用线性组合
        # 实际应用中可以使用优化方法求解
        n_options = len(hedge_options)
        n_greeks = len(hedge_needs)
        
        if n_options < n_greeks:
            return {
                'strategy': 'multi_greeks_hedging',
                'error': f'Need at least {n_greeks} hedge options, but only {n_options} provided',
                'hedge_quantities': {},
            }
        
        # 构建线性方程组 A * x = b
        # A: 对冲期权的希腊字母矩阵
        # x: 对冲期权数量（待求解）
        # b: 需要对冲的希腊字母
        A = np.zeros((n_greeks, n_options))
        b = np.zeros(n_greeks)
        
        greek_names = list(hedge_needs.keys())
        for i, greek_name in enumerate(greek_names):
            b[i] = hedge_needs[greek_name]
            for j, hedge_option in enumerate(hedge_options):
                A[i, j] = hedge_option.get('greeks', {}).get(greek_name, 0.0)
        
        # 求解线性方程组
        try:
            hedge_quantities = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用最小二乘法
            hedge_quantities = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # 计算对冲后的剩余风险
        residual_greeks = {}
        for i, greek_name in enumerate(greek_names):
            residual = b[i] - np.dot(A[i, :], hedge_quantities)
            residual_greeks[greek_name] = float(residual)
        
        return {
            'strategy': 'multi_greeks_hedging',
            'hedge_quantities': {f'option_{i}': float(qty) for i, qty in enumerate(hedge_quantities)},
            'target_greeks': target_greeks,
            'current_greeks': portfolio_greeks,
            'residual_greeks': residual_greeks,
        }
    
    def delta_gamma_hedge(self,
                         portfolio_delta: float,
                         portfolio_gamma: float,
                         spot_price: float,
                         hedge_call_option: Dict,
                         hedge_put_option: Optional[Dict] = None) -> Dict:
        """
        Delta-Gamma对冲（使用标的资产和期权）
        
        Args:
            portfolio_delta: 组合Delta
            portfolio_gamma: 组合Gamma
            spot_price: 标的资产价格
            hedge_call_option: 对冲用看涨期权信息（包含greeks）
            hedge_put_option: 对冲用看跌期权信息（可选）
        
        Returns:
            对冲方案
        """
        hedge_call_gamma = hedge_call_option.get('greeks', {}).get('gamma', 0.0)
        hedge_call_delta = hedge_call_option.get('greeks', {}).get('delta', 0.0)
        
        if hedge_put_option is not None:
            # 使用两个期权对冲
            hedge_put_gamma = hedge_put_option.get('greeks', {}).get('gamma', 0.0)
            hedge_put_delta = hedge_put_option.get('greeks', {}).get('delta', 0.0)
            
            # 解方程组：对冲Delta和Gamma
            A = np.array([
                [hedge_call_delta, hedge_put_delta],
                [hedge_call_gamma, hedge_put_gamma]
            ])
            b = np.array([-portfolio_delta, -portfolio_gamma])
            
            try:
                quantities = np.linalg.solve(A, b)
                call_qty, put_qty = quantities[0], quantities[1]
            except np.linalg.LinAlgError:
                return {'error': 'Cannot solve delta-gamma hedge with given options'}
        else:
            # 只用一个期权，还需要用标的资产
            # 先用期权对冲Gamma
            if abs(hedge_call_gamma) < 1e-10:
                return {'error': 'Hedge option gamma is too small'}
            
            call_qty = -portfolio_gamma / hedge_call_gamma
            
            # 再用标的资产对冲剩余Delta
            remaining_delta = portfolio_delta + call_qty * hedge_call_delta
            spot_quantity = -remaining_delta
            
            return {
                'strategy': 'delta_gamma_hedging',
                'call_option_quantity': float(call_qty),
                'spot_quantity': float(spot_quantity),
                'target_delta': 0.0,
                'target_gamma': 0.0,
            }
        
        return {
            'strategy': 'delta_gamma_hedging',
            'call_option_quantity': float(call_qty),
            'put_option_quantity': float(put_qty),
            'target_delta': 0.0,
            'target_gamma': 0.0,
        }

