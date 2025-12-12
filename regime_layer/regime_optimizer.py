"""
Regime-Aware Optimizer - 状态感知优化器

考虑市场状态的投资组合优化

三种策略：
1. Expected: 使用概率加权的期望参数
2. Robust: 考虑最坏情况的稳健优化
3. Adaptive: 根据当前regime动态调整
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from scipy.optimize import minimize
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from .regime_knowledge_base import RegimeKnowledgeBase, RegimeParameters


class RegimeAwareOptimizer:
    """
    状态感知优化器
    
    提供多种考虑market regime的优化策略
    """
    
    def __init__(self, 
                 strategy: str = 'expected',
                 risk_aversion: float = 1.0):
        """
        初始化优化器
        
        Args:
            strategy: 优化策略
                - 'expected': 期望参数优化
                - 'robust': 稳健优化（minimax）
                - 'adaptive': 自适应优化
                - 'worst_case': 最坏情况优化
                - 'multi_regime': 多regime联合优化
            risk_aversion: 风险厌恶系数
        """
        self.strategy = strategy
        self.risk_aversion = risk_aversion
    
    def optimize(self,
                regime_knowledge: RegimeKnowledgeBase,
                constraints: Optional[Dict] = None,
                objective: str = 'sharpe') -> np.ndarray:
        """
        优化投资组合
        
        Args:
            regime_knowledge: 状态依赖知识库
            constraints: 约束条件
            objective: 目标函数 ('sharpe', 'mean_variance', 'min_variance')
        
        Returns:
            最优权重向量
        """
        if constraints is None:
            constraints = {'long_only': True, 'leverage': 1.0}
        
        if self.strategy == 'expected':
            return self._optimize_expected(regime_knowledge, constraints, objective)
        elif self.strategy == 'robust':
            return self._optimize_robust(regime_knowledge, constraints, objective)
        elif self.strategy == 'adaptive':
            return self._optimize_adaptive(regime_knowledge, constraints, objective)
        elif self.strategy == 'worst_case':
            return self._optimize_worst_case(regime_knowledge, constraints, objective)
        elif self.strategy == 'multi_regime':
            return self._optimize_multi_regime(regime_knowledge, constraints, objective)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _optimize_expected(self,
                          regime_knowledge: RegimeKnowledgeBase,
                          constraints: Dict,
                          objective: str) -> np.ndarray:
        """
        期望参数优化
        
        使用概率加权的期望参数进行标准优化
        
        E[μ] = Σ_s P(s) * μ(s)
        E[Σ] = Σ_s P(s) * Σ(s)
        """
        params = regime_knowledge.get_expected_parameters()
        return self._solve_optimization(params.mu, params.Sigma, constraints, objective)
    
    def _optimize_robust(self,
                        regime_knowledge: RegimeKnowledgeBase,
                        constraints: Dict,
                        objective: str) -> np.ndarray:
        """
        稳健优化
        
        考虑参数不确定性的优化：
        max_w min_s { utility(w, μ(s), Σ(s)) }
        """
        n_assets = regime_knowledge.n_assets
        n_regimes = regime_knowledge.n_regimes
        
        def robust_objective(w):
            """最大化最坏情况下的效用"""
            utilities = []
            for k in range(n_regimes):
                params = regime_knowledge.regime_params[k]
                util = self._calculate_utility(w, params.mu, params.Sigma, objective)
                utilities.append(util)
            # 返回最坏情况（最小效用）的负值（因为我们要最小化）
            return -min(utilities)
        
        # 设置约束和边界
        bounds, cons = self._setup_constraints(n_assets, constraints)
        
        # 初始猜测
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            robust_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_adaptive(self,
                          regime_knowledge: RegimeKnowledgeBase,
                          constraints: Dict,
                          objective: str) -> np.ndarray:
        """
        自适应优化
        
        根据当前regime和转换概率动态调整
        
        考虑：
        1. 当前regime的参数（权重更高）
        2. 可能转换到的regime的参数
        """
        n_assets = regime_knowledge.n_assets
        current_regime = regime_knowledge.current_regime
        current_prob = regime_knowledge.regime_probabilities[current_regime]
        
        # 计算调整后的概率（当前regime权重更高）
        adjustment_factor = 1.5  # 当前regime的权重倍数
        adjusted_probs = regime_knowledge.regime_probabilities.copy()
        adjusted_probs[current_regime] *= adjustment_factor
        adjusted_probs /= adjusted_probs.sum()
        
        # 使用调整后的概率计算加权参数
        expected_mu = np.zeros(n_assets)
        expected_Sigma = np.zeros((n_assets, n_assets))
        
        for k, prob in enumerate(adjusted_probs):
            params = regime_knowledge.regime_params[k]
            expected_mu += prob * params.mu
            expected_Sigma += prob * params.Sigma
        
        return self._solve_optimization(expected_mu, expected_Sigma, constraints, objective)
    
    def _optimize_worst_case(self,
                            regime_knowledge: RegimeKnowledgeBase,
                            constraints: Dict,
                            objective: str) -> np.ndarray:
        """
        最坏情况优化
        
        只考虑最差regime的参数
        """
        n_assets = regime_knowledge.n_assets
        
        # 找出最差的regime（最低期望收益或最高波动率）
        worst_regime = None
        worst_score = float('inf')
        
        for k in range(regime_knowledge.n_regimes):
            params = regime_knowledge.regime_params[k]
            mean_return = np.mean(params.mu)
            mean_vol = np.mean(np.sqrt(np.diag(params.Sigma)))
            score = mean_return / mean_vol if mean_vol > 0 else mean_return
            
            if score < worst_score:
                worst_score = score
                worst_regime = k
        
        params = regime_knowledge.regime_params[worst_regime]
        return self._solve_optimization(params.mu, params.Sigma, constraints, objective)
    
    def _optimize_multi_regime(self,
                              regime_knowledge: RegimeKnowledgeBase,
                              constraints: Dict,
                              objective: str) -> np.ndarray:
        """
        多regime联合优化
        
        同时优化所有regime下的表现：
        max_w Σ_s P(s) * utility(w, μ(s), Σ(s))
        
        subject to: utility(w, μ(s), Σ(s)) >= threshold for all s
        """
        n_assets = regime_knowledge.n_assets
        n_regimes = regime_knowledge.n_regimes
        
        def multi_regime_objective(w):
            """加权效用和"""
            total_utility = 0
            for k in range(n_regimes):
                params = regime_knowledge.regime_params[k]
                prob = regime_knowledge.regime_probabilities[k]
                util = self._calculate_utility(w, params.mu, params.Sigma, objective)
                total_utility += prob * util
            return -total_utility  # 最小化负效用
        
        # 设置约束和边界
        bounds, cons = self._setup_constraints(n_assets, constraints)
        
        # 添加每个regime的最小效用约束
        min_utility_threshold = -0.5  # 避免在任何regime下表现太差
        
        for k in range(n_regimes):
            params = regime_knowledge.regime_params[k]
            
            def regime_constraint(w, params=params, threshold=min_utility_threshold):
                util = self._calculate_utility(w, params.mu, params.Sigma, objective)
                return util - threshold
            
            cons.append({'type': 'ineq', 'fun': regime_constraint})
        
        # 初始猜测
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            multi_regime_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _calculate_utility(self,
                          w: np.ndarray,
                          mu: np.ndarray,
                          Sigma: np.ndarray,
                          objective: str) -> float:
        """计算效用函数值"""
        portfolio_return = np.dot(w, mu)
        portfolio_var = np.dot(w, np.dot(Sigma, w))
        portfolio_risk = np.sqrt(portfolio_var) if portfolio_var > 0 else 1e-10
        
        if objective == 'sharpe':
            return portfolio_return / portfolio_risk
        elif objective == 'mean_variance':
            return portfolio_return - 0.5 * self.risk_aversion * portfolio_var
        elif objective == 'min_variance':
            return -portfolio_var
        else:
            return portfolio_return / portfolio_risk
    
    def _solve_optimization(self,
                           mu: np.ndarray,
                           Sigma: np.ndarray,
                           constraints: Dict,
                           objective: str) -> np.ndarray:
        """
        求解标准优化问题
        """
        n_assets = len(mu)
        
        def objective_func(w):
            return -self._calculate_utility(w, mu, Sigma, objective)
        
        bounds, cons = self._setup_constraints(n_assets, constraints)
        
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _setup_constraints(self, n_assets: int, constraints: Dict) -> Tuple[List, List]:
        """设置优化约束"""
        long_only = constraints.get('long_only', True)
        leverage = constraints.get('leverage', 1.0)
        max_weight = constraints.get('max_weight', None)
        min_weight = constraints.get('min_weight', None)
        
        # 边界
        if long_only:
            lower = min_weight if min_weight is not None else 0.0
            upper = max_weight if max_weight is not None else 1.0
            bounds = [(lower, upper) for _ in range(n_assets)]
        else:
            lower = min_weight if min_weight is not None else -1.0
            upper = max_weight if max_weight is not None else 1.0
            bounds = [(lower, upper) for _ in range(n_assets)]
        
        # 约束
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - leverage}]
        
        return bounds, cons
    
    def optimize_with_regime_scenarios(self,
                                       regime_knowledge: RegimeKnowledgeBase,
                                       constraints: Optional[Dict] = None,
                                       objective: str = 'sharpe',
                                       n_periods: int = 12,
                                       rebalance_freq: int = 1) -> Dict:
        """
        基于regime场景的动态优化
        
        模拟未来regime演化，计算各情景下的最优权重
        
        Args:
            regime_knowledge: 状态依赖知识库
            constraints: 约束条件
            objective: 目标函数
            n_periods: 预测期数
            rebalance_freq: 再平衡频率
        
        Returns:
            包含各情景权重和预期表现的字典
        """
        if constraints is None:
            constraints = {'long_only': True, 'leverage': 1.0}
        
        # 预测regime演化
        regime_evolution = regime_knowledge.predict_regime_evolution(n_periods)
        
        # 计算各时期的最优权重
        results = {
            'period_weights': [],
            'period_params': [],
            'expected_returns': [],
            'expected_risks': []
        }
        
        for t in range(0, n_periods, rebalance_freq):
            # 该时期的regime概率
            period_probs = regime_evolution[t]
            
            # 创建临时知识库
            temp_knowledge = RegimeKnowledgeBase(
                regime_params=regime_knowledge.regime_params,
                transition_matrix=regime_knowledge.transition_matrix,
                current_regime=np.argmax(period_probs),
                regime_probabilities=period_probs,
                asset_names=regime_knowledge.asset_names,
                factor_names=regime_knowledge.factor_names
            )
            
            # 优化
            weights = self.optimize(temp_knowledge, constraints, objective)
            
            # 计算期望收益和风险
            params = temp_knowledge.get_expected_parameters()
            exp_ret = np.dot(weights, params.mu)
            exp_risk = np.sqrt(np.dot(weights, np.dot(params.Sigma, weights)))
            
            results['period_weights'].append(weights)
            results['period_params'].append(params)
            results['expected_returns'].append(exp_ret)
            results['expected_risks'].append(exp_risk)
        
        # 计算整体预期表现
        results['avg_return'] = np.mean(results['expected_returns'])
        results['avg_risk'] = np.mean(results['expected_risks'])
        results['regime_evolution'] = regime_evolution
        
        return results
    
    def compute_regime_contribution(self,
                                   weights: np.ndarray,
                                   regime_knowledge: RegimeKnowledgeBase) -> Dict:
        """
        计算各regime对组合表现的贡献
        
        Args:
            weights: 投资组合权重
            regime_knowledge: 状态依赖知识库
        
        Returns:
            各regime的贡献分析
        """
        contribution = {
            'by_regime': {},
            'expected': {}
        }
        
        total_return = 0
        total_risk_sq = 0
        
        for k in range(regime_knowledge.n_regimes):
            params = regime_knowledge.regime_params[k]
            prob = regime_knowledge.regime_probabilities[k]
            
            ret = np.dot(weights, params.mu)
            risk = np.sqrt(np.dot(weights, np.dot(params.Sigma, weights)))
            sharpe = ret / risk if risk > 0 else 0
            
            contribution['by_regime'][k] = {
                'name': regime_knowledge.get_regime_name(k),
                'probability': float(prob),
                'return': float(ret),
                'risk': float(risk),
                'sharpe': float(sharpe),
                'return_contribution': float(prob * ret),
                'risk_contribution': float(prob * risk)
            }
            
            total_return += prob * ret
            total_risk_sq += prob * (risk ** 2 + ret ** 2)  # 方差分解
        
        contribution['expected']['return'] = float(total_return)
        contribution['expected']['risk'] = float(np.sqrt(total_risk_sq - total_return ** 2))
        contribution['expected']['sharpe'] = (
            float(total_return / contribution['expected']['risk'])
            if contribution['expected']['risk'] > 0 else 0
        )
        
        return contribution

