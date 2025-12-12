"""
Regime Knowledge Base - 状态依赖知识库

存储每个regime下的参数：μ(s), B(s), F(s), D(s), Σ(s)

数学模型：
    r_t | F_t, s_t = B(s_t) * F_t + ε_t(s_t)
    ε_t(s_t) ~ N(0, Σ_ε(s_t))
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RegimeParameters:
    """
    单个Regime的参数集
    
    Attributes:
        mu: 预期收益率 (N,)
        Sigma: 协方差矩阵 (N x N)
        B: 因子载荷矩阵 (N x K)，可选
        F: 因子协方差矩阵 (K x K)，可选
        D: 特质风险对角矩阵 (N x N)，可选
    """
    mu: np.ndarray
    Sigma: np.ndarray
    B: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    
    def get_covariance(self) -> np.ndarray:
        """获取协方差矩阵"""
        return self.Sigma
    
    def get_expected_returns(self) -> np.ndarray:
        """获取预期收益率"""
        return self.mu


@dataclass
class RegimeKnowledgeBase:
    """
    状态依赖知识库
    
    存储所有regime的参数和转移概率
    
    Attributes:
        regime_params: Dict[int, RegimeParameters] - 各regime的参数
        transition_matrix: 状态转移矩阵 (K x K)
        current_regime: 当前regime
        regime_probabilities: 当前各regime的概率
    """
    regime_params: Dict[int, RegimeParameters]
    transition_matrix: np.ndarray
    current_regime: int
    regime_probabilities: np.ndarray
    asset_names: Optional[List[str]] = None
    factor_names: Optional[List[str]] = None
    regime_names: Optional[List[str]] = None
    
    @property
    def n_regimes(self) -> int:
        return len(self.regime_params)
    
    @property
    def n_assets(self) -> int:
        return len(self.regime_params[0].mu)
    
    def get_regime_name(self, regime_id: int) -> str:
        """获取regime名称"""
        if self.regime_names is not None and regime_id < len(self.regime_names):
            return self.regime_names[regime_id]
        default_names = ['Bull', 'Bear', 'Sideways', 'Crisis']
        if regime_id < len(default_names):
            return default_names[regime_id]
        return f"Regime_{regime_id}"
    
    def get_current_parameters(self) -> RegimeParameters:
        """获取当前regime的参数"""
        return self.regime_params[self.current_regime]
    
    def get_expected_parameters(self) -> RegimeParameters:
        """
        获取概率加权的期望参数
        
        E[θ] = Σ_s P(s) * θ(s)
        """
        expected_mu = np.zeros(self.n_assets)
        expected_Sigma = np.zeros((self.n_assets, self.n_assets))
        
        for regime_id, prob in enumerate(self.regime_probabilities):
            params = self.regime_params[regime_id]
            expected_mu += prob * params.mu
            expected_Sigma += prob * params.Sigma
        
        # 检查是否有因子模型
        has_factor_model = all(
            self.regime_params[k].B is not None 
            for k in range(self.n_regimes)
        )
        
        if has_factor_model:
            n_factors = self.regime_params[0].B.shape[1]
            expected_B = np.zeros((self.n_assets, n_factors))
            expected_F = np.zeros((n_factors, n_factors))
            expected_D = np.zeros((self.n_assets, self.n_assets))
            
            for regime_id, prob in enumerate(self.regime_probabilities):
                params = self.regime_params[regime_id]
                expected_B += prob * params.B
                expected_F += prob * params.F
                expected_D += prob * params.D
            
            return RegimeParameters(
                mu=expected_mu,
                Sigma=expected_Sigma,
                B=expected_B,
                F=expected_F,
                D=expected_D
            )
        
        return RegimeParameters(mu=expected_mu, Sigma=expected_Sigma)
    
    def get_robust_parameters(self, alpha: float = 0.95) -> RegimeParameters:
        """
        获取稳健参数（考虑最坏情况）
        
        对于风险：使用最大波动率
        对于收益：使用加权期望
        
        Args:
            alpha: 置信水平
        """
        expected_mu = np.zeros(self.n_assets)
        max_Sigma = np.zeros((self.n_assets, self.n_assets))
        
        for regime_id, prob in enumerate(self.regime_probabilities):
            params = self.regime_params[regime_id]
            expected_mu += prob * params.mu
            
            # 取各regime中较大的协方差元素
            max_Sigma = np.maximum(max_Sigma, params.Sigma)
        
        # 对预期收益进行保守调整
        # 使用较低的收益估计
        min_mu = np.min([self.regime_params[k].mu for k in range(self.n_regimes)], axis=0)
        robust_mu = alpha * expected_mu + (1 - alpha) * min_mu
        
        return RegimeParameters(mu=robust_mu, Sigma=max_Sigma)
    
    def get_conditional_parameters(self, 
                                   next_regime_probs: Optional[np.ndarray] = None) -> RegimeParameters:
        """
        获取条件期望参数（考虑regime转换）
        
        E[θ_{t+1} | s_t] = Σ_s' P(s'|s_t) * θ(s')
        
        Args:
            next_regime_probs: 下一时刻的regime概率（如果为None，使用转移矩阵计算）
        """
        if next_regime_probs is None:
            # 使用当前概率和转移矩阵计算下一时刻概率
            next_regime_probs = self.regime_probabilities @ self.transition_matrix
        
        expected_mu = np.zeros(self.n_assets)
        expected_Sigma = np.zeros((self.n_assets, self.n_assets))
        
        for regime_id, prob in enumerate(next_regime_probs):
            params = self.regime_params[regime_id]
            expected_mu += prob * params.mu
            expected_Sigma += prob * params.Sigma
        
        return RegimeParameters(mu=expected_mu, Sigma=expected_Sigma)
    
    def decompose_risk_by_regime(self, weights: np.ndarray) -> Dict:
        """
        按regime分解投资组合风险
        
        Args:
            weights: 投资组合权重 (N,)
        
        Returns:
            风险分解结果
        """
        result = {
            'total': {},
            'by_regime': {}
        }
        
        # 计算期望参数下的总风险
        expected_params = self.get_expected_parameters()
        total_var = np.dot(weights, np.dot(expected_params.Sigma, weights))
        total_risk = np.sqrt(total_var)
        
        result['total'] = {
            'variance': float(total_var),
            'risk': float(total_risk)
        }
        
        # 计算各regime下的风险
        for regime_id in range(self.n_regimes):
            params = self.regime_params[regime_id]
            regime_var = np.dot(weights, np.dot(params.Sigma, weights))
            regime_risk = np.sqrt(regime_var)
            
            result['by_regime'][regime_id] = {
                'name': self.get_regime_name(regime_id),
                'probability': float(self.regime_probabilities[regime_id]),
                'variance': float(regime_var),
                'risk': float(regime_risk),
                'expected_return': float(np.dot(weights, params.mu)),
                'sharpe': float(np.dot(weights, params.mu) / regime_risk) if regime_risk > 0 else 0
            }
        
        return result
    
    def to_simple_knowledge_base(self, method: str = 'expected'):
        """
        转换为简单的KnowledgeBase对象（用于向后兼容）
        
        Args:
            method: 参数聚合方法
                - 'expected': 概率加权期望
                - 'current': 当前regime参数
                - 'robust': 稳健参数
                - 'conditional': 条件期望
        """
        if method == 'expected':
            params = self.get_expected_parameters()
        elif method == 'current':
            params = self.get_current_parameters()
        elif method == 'robust':
            params = self.get_robust_parameters()
        elif method == 'conditional':
            params = self.get_conditional_parameters()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Import KnowledgeBase dynamically to avoid circular import
        # Use importlib.util to load the specific file
        import importlib.util
        import os
        
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'portfolio_layer', 'parameter_estimation', 'knowledge_base.py'
        )
        
        spec = importlib.util.spec_from_file_location("knowledge_base_module", kb_path)
        kb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kb_module)
        KnowledgeBase = kb_module.KnowledgeBase
        
        return KnowledgeBase(
            mu=params.mu,
            Sigma=params.Sigma,
            B=params.B,
            F=params.F,
            D=params.D,
            asset_names=self.asset_names,
            factor_names=self.factor_names
        )
    
    def predict_regime_evolution(self, n_steps: int = 10) -> np.ndarray:
        """
        预测未来regime概率演化
        
        Args:
            n_steps: 预测步数
        
        Returns:
            regime概率演化矩阵 (n_steps x K)
        """
        probs = np.zeros((n_steps, self.n_regimes))
        current_probs = self.regime_probabilities.copy()
        
        for t in range(n_steps):
            probs[t] = current_probs
            current_probs = current_probs @ self.transition_matrix
        
        return probs
    
    def summary(self) -> str:
        """生成摘要报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("REGIME KNOWLEDGE BASE SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"\nNumber of regimes: {self.n_regimes}")
        lines.append(f"Number of assets: {self.n_assets}")
        lines.append(f"Current regime: {self.get_regime_name(self.current_regime)} "
                    f"(prob: {self.regime_probabilities[self.current_regime]:.2%})")
        
        lines.append("\n" + "-" * 40)
        lines.append("REGIME PARAMETERS")
        lines.append("-" * 40)
        
        for regime_id in range(self.n_regimes):
            params = self.regime_params[regime_id]
            name = self.get_regime_name(regime_id)
            prob = self.regime_probabilities[regime_id]
            
            lines.append(f"\n{name} (P = {prob:.2%}):")
            lines.append(f"  Expected return (ann.): {np.mean(params.mu) * 252:.2%}")
            lines.append(f"  Volatility (ann.): {np.mean(np.sqrt(np.diag(params.Sigma))) * np.sqrt(252):.2%}")
        
        lines.append("\n" + "-" * 40)
        lines.append("TRANSITION MATRIX")
        lines.append("-" * 40)
        
        header = "From\\To  " + "  ".join([f"{self.get_regime_name(j)[:6]:>8}" 
                                           for j in range(self.n_regimes)])
        lines.append(header)
        
        for i in range(self.n_regimes):
            row = f"{self.get_regime_name(i)[:6]:>8}"
            for j in range(self.n_regimes):
                row += f"  {self.transition_matrix[i, j]:>8.2%}"
            lines.append(row)
        
        return "\n".join(lines)

