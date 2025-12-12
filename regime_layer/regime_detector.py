"""
Regime Detector - 市场状态检测器

使用隐马尔可夫模型(HMM)识别市场状态

数学模型：
    s_t ~ Markov(P)
    
其中：
- s_t ∈ {1, 2, ..., K} 是隐藏状态（市场regime）
- P 是状态转移概率矩阵 (K x K)
- P[i,j] = P(s_{t+1} = j | s_t = i)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RegimeState:
    """当前状态信息"""
    current_regime: int                    # 当前最可能的regime
    regime_probabilities: np.ndarray       # 当前各regime的概率
    transition_matrix: np.ndarray          # 状态转移矩阵
    regime_sequence: np.ndarray            # 历史regime序列
    smoothed_probabilities: np.ndarray     # 平滑后的历史regime概率
    
    @property
    def n_regimes(self) -> int:
        return len(self.regime_probabilities)
    
    def get_regime_name(self, regime_id: int, names: Optional[List[str]] = None) -> str:
        """获取regime名称"""
        if names is not None and regime_id < len(names):
            return names[regime_id]
        default_names = ['Bull', 'Bear', 'Sideways', 'Crisis']
        if regime_id < len(default_names):
            return default_names[regime_id]
        return f"Regime_{regime_id}"


class RegimeDetector(ABC):
    """市场状态检测器基类"""
    
    @abstractmethod
    def fit(self, returns: pd.DataFrame, n_regimes: int = 2) -> 'RegimeDetector':
        """训练模型"""
        pass
    
    @abstractmethod
    def detect(self, returns: pd.DataFrame) -> RegimeState:
        """检测当前市场状态"""
        pass
    
    @abstractmethod
    def predict_next_regime(self, current_state: RegimeState) -> np.ndarray:
        """预测下一时刻的regime概率"""
        pass


class HMMRegimeDetector(RegimeDetector):
    """
    基于隐马尔可夫模型的市场状态检测器
    
    使用高斯HMM (Gaussian Hidden Markov Model)
    
    观测模型：
        r_t | s_t ~ N(μ_{s_t}, Σ_{s_t})
    
    其中每个regime有自己的均值和协方差
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 covariance_type: str = 'full',
                 n_iter: int = 100,
                 random_state: int = 42):
        """
        初始化HMM检测器
        
        Args:
            n_regimes: 状态数量（默认2：牛市/熊市）
            covariance_type: 协方差类型 ('full', 'diag', 'spherical')
            n_iter: EM算法最大迭代次数
            random_state: 随机种子
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # 模型参数（训练后填充）
        self.means_: Optional[np.ndarray] = None          # (K, N) 各regime的均值
        self.covars_: Optional[np.ndarray] = None         # (K, N, N) 各regime的协方差
        self.transmat_: Optional[np.ndarray] = None       # (K, K) 转移矩阵
        self.startprob_: Optional[np.ndarray] = None      # (K,) 初始状态概率
        self._fitted = False
        
    def fit(self, returns: pd.DataFrame, n_regimes: Optional[int] = None) -> 'HMMRegimeDetector':
        """
        训练HMM模型
        
        Args:
            returns: 收益率数据 (T x N)
            n_regimes: 状态数量（可覆盖初始化设置）
        
        Returns:
            self
        """
        if n_regimes is not None:
            self.n_regimes = n_regimes
        
        # 数据预处理
        X = returns.dropna().values
        T, N = X.shape
        K = self.n_regimes
        
        # 使用EM算法训练HMM
        self._fit_em(X)
        self._fitted = True
        
        return self
    
    def _fit_em(self, X: np.ndarray, tol: float = 1e-4):
        """
        使用EM算法训练HMM
        
        Args:
            X: 观测数据 (T x N)
            tol: 收敛阈值
        """
        T, N = X.shape
        K = self.n_regimes
        
        np.random.seed(self.random_state)
        
        # 初始化参数
        # 使用K-means初始化
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step: 前向-后向算法
            log_alpha, log_beta, log_likelihood = self._forward_backward(X)
            
            # 计算gamma和xi
            gamma = self._compute_gamma(log_alpha, log_beta)
            xi = self._compute_xi(X, log_alpha, log_beta)
            
            # M-step: 更新参数
            self._update_parameters(X, gamma, xi)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
    
    def _initialize_parameters(self, X: np.ndarray):
        """使用K-means初始化HMM参数"""
        T, N = X.shape
        K = self.n_regimes
        
        # 简单初始化：按收益率排序分组
        sorted_indices = np.argsort(X.mean(axis=1))
        segment_size = T // K
        
        self.means_ = np.zeros((K, N))
        self.covars_ = np.zeros((K, N, N))
        
        for k in range(K):
            start_idx = k * segment_size
            end_idx = (k + 1) * segment_size if k < K - 1 else T
            segment_data = X[sorted_indices[start_idx:end_idx]]
            
            self.means_[k] = segment_data.mean(axis=0)
            self.covars_[k] = np.cov(segment_data.T) + 1e-6 * np.eye(N)
        
        # 初始化转移矩阵（倾向于停留在当前状态）
        self.transmat_ = np.full((K, K), 0.1 / (K - 1)) if K > 1 else np.array([[1.0]])
        np.fill_diagonal(self.transmat_, 0.9)
        
        # 初始状态概率（均匀分布）
        self.startprob_ = np.ones(K) / K
    
    def _forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        前向-后向算法（对数空间以避免数值下溢）
        
        Returns:
            log_alpha: 前向概率 (T x K)
            log_beta: 后向概率 (T x K)
            log_likelihood: 对数似然
        """
        T = X.shape[0]
        K = self.n_regimes
        
        # 计算观测概率
        log_B = self._compute_log_emission(X)  # (T x K)
        
        # 前向算法
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_B[0]
        
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = self._logsumexp(
                    log_alpha[t-1] + np.log(self.transmat_[:, k] + 1e-300)
                ) + log_B[t, k]
        
        # 后向算法
        log_beta = np.zeros((T, K))
        log_beta[-1] = 0  # log(1) = 0
        
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = self._logsumexp(
                    np.log(self.transmat_[k, :] + 1e-300) + log_B[t+1] + log_beta[t+1]
                )
        
        log_likelihood = self._logsumexp(log_alpha[-1])
        
        return log_alpha, log_beta, log_likelihood
    
    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """计算观测概率的对数"""
        T, N = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        
        for k in range(K):
            diff = X - self.means_[k]  # (T x N)
            try:
                cov_inv = np.linalg.inv(self.covars_[k])
                sign, log_det = np.linalg.slogdet(self.covars_[k])
                
                # 多元高斯对数概率
                mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
                log_B[:, k] = -0.5 * (N * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                # 如果协方差矩阵奇异，使用对角近似
                var = np.diag(self.covars_[k])
                log_B[:, k] = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + diff**2 / var, axis=1
                )
        
        return log_B
    
    def _compute_gamma(self, log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
        """计算gamma (后验状态概率)"""
        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)
    
    def _compute_xi(self, X: np.ndarray, 
                    log_alpha: np.ndarray, 
                    log_beta: np.ndarray) -> np.ndarray:
        """计算xi (状态转移后验概率)"""
        T = X.shape[0]
        K = self.n_regimes
        log_B = self._compute_log_emission(X)
        
        xi = np.zeros((T - 1, K, K))
        
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (log_alpha[t, i] + 
                                  np.log(self.transmat_[i, j] + 1e-300) +
                                  log_B[t+1, j] + 
                                  log_beta[t+1, j])
            xi[t] -= self._logsumexp(xi[t].flatten())
        
        return np.exp(xi)
    
    def _update_parameters(self, X: np.ndarray, 
                          gamma: np.ndarray, 
                          xi: np.ndarray):
        """M-step: 更新参数"""
        T, N = X.shape
        K = self.n_regimes
        
        # 更新初始状态概率
        self.startprob_ = gamma[0] / gamma[0].sum()
        
        # 更新转移矩阵
        for i in range(K):
            for j in range(K):
                self.transmat_[i, j] = xi[:, i, j].sum() / gamma[:-1, i].sum()
        
        # 归一化转移矩阵
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
        
        # 更新均值和协方差
        for k in range(K):
            gamma_sum = gamma[:, k].sum()
            
            # 更新均值
            self.means_[k] = (gamma[:, k:k+1] * X).sum(axis=0) / gamma_sum
            
            # 更新协方差
            diff = X - self.means_[k]
            if self.covariance_type == 'full':
                self.covars_[k] = (gamma[:, k:k+1] * diff).T @ diff / gamma_sum
                self.covars_[k] += 1e-6 * np.eye(N)  # 正则化
            elif self.covariance_type == 'diag':
                self.covars_[k] = np.diag(
                    (gamma[:, k:k+1] * diff**2).sum(axis=0) / gamma_sum + 1e-6
                )
    
    @staticmethod
    def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        """稳定的logsumexp实现"""
        a_max = np.max(a, axis=axis, keepdims=True)
        result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
        if not keepdims and axis is not None:
            result = np.squeeze(result, axis=axis)
        return result
    
    def detect(self, returns: pd.DataFrame) -> RegimeState:
        """
        检测市场状态
        
        Args:
            returns: 收益率数据 (T x N)
        
        Returns:
            RegimeState 对象
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = returns.dropna().values
        
        # 前向-后向算法
        log_alpha, log_beta, _ = self._forward_backward(X)
        
        # 计算平滑概率
        gamma = self._compute_gamma(log_alpha, log_beta)
        
        # Viterbi算法找最优路径
        regime_sequence = self._viterbi(X)
        
        # 当前状态
        current_probs = gamma[-1]
        current_regime = np.argmax(current_probs)
        
        return RegimeState(
            current_regime=current_regime,
            regime_probabilities=current_probs,
            transition_matrix=self.transmat_.copy(),
            regime_sequence=regime_sequence,
            smoothed_probabilities=gamma
        )
    
    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi算法找最优状态序列"""
        T = X.shape[0]
        K = self.n_regimes
        log_B = self._compute_log_emission(X)
        
        # Viterbi
        V = np.zeros((T, K))
        backtrack = np.zeros((T, K), dtype=int)
        
        V[0] = np.log(self.startprob_ + 1e-300) + log_B[0]
        
        for t in range(1, T):
            for k in range(K):
                probs = V[t-1] + np.log(self.transmat_[:, k] + 1e-300)
                backtrack[t, k] = np.argmax(probs)
                V[t, k] = probs[backtrack[t, k]] + log_B[t, k]
        
        # 回溯
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(V[-1])
        
        for t in range(T - 2, -1, -1):
            path[t] = backtrack[t + 1, path[t + 1]]
        
        return path
    
    def predict_next_regime(self, current_state: RegimeState) -> np.ndarray:
        """
        预测下一时刻的regime概率
        
        Args:
            current_state: 当前状态
        
        Returns:
            下一时刻各regime的概率
        """
        return current_state.regime_probabilities @ self.transmat_
    
    def get_regime_statistics(self) -> Dict:
        """获取各regime的统计信息"""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        stats = {}
        for k in range(self.n_regimes):
            stats[k] = {
                'mean': self.means_[k].copy(),
                'covariance': self.covars_[k].copy(),
                'volatility': np.sqrt(np.diag(self.covars_[k]))
            }
        
        stats['transition_matrix'] = self.transmat_.copy()
        stats['stationary_distribution'] = self._get_stationary_distribution()
        
        return stats
    
    def _get_stationary_distribution(self) -> np.ndarray:
        """计算稳态分布"""
        # 解 π = π * P，即 (P' - I)' * π = 0
        eigenvalues, eigenvectors = np.linalg.eig(self.transmat_.T)
        
        # 找特征值为1对应的特征向量
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        return stationary


class MultiFactorRegimeDetector(HMMRegimeDetector):
    """
    基于因子的市场状态检测器
    
    使用因子收益而非股票收益来识别regime
    更加稳健且计算效率更高
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 use_factors: bool = True,
                 **kwargs):
        super().__init__(n_regimes=n_regimes, **kwargs)
        self.use_factors = use_factors
    
    def fit_with_factors(self, 
                         returns: pd.DataFrame,
                         factors: pd.DataFrame,
                         n_regimes: Optional[int] = None) -> 'MultiFactorRegimeDetector':
        """
        使用因子数据训练
        
        Args:
            returns: 股票收益率 (T x N)
            factors: 因子收益率 (T x K)
            n_regimes: 状态数量
        """
        if self.use_factors:
            return self.fit(factors, n_regimes)
        else:
            return self.fit(returns, n_regimes)
    
    def detect_with_factors(self, 
                           returns: pd.DataFrame,
                           factors: pd.DataFrame) -> RegimeState:
        """使用因子数据检测状态"""
        if self.use_factors:
            return self.detect(factors)
        else:
            return self.detect(returns)

