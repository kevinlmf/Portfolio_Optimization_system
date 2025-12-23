"""
Return Forecaster - 收益预测器

预测股票未来收益的多种方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class ReturnForecaster:
    """
    收益预测器
    
    支持多种预测方法：
    1. Factor Model: μ̂ = α + B @ F̂
    2. Historical Mean: μ̂ = mean(r)
    3. ARMA: 时间序列模型
    4. Momentum/Reversal: 动量和均值回归信号
    """
    
    def __init__(self, method: str = 'factor'):
        """
        初始化预测器
        
        Args:
            method: 预测方法 ('factor', 'historical', 'arma', 'momentum', 'ensemble')
        """
        self.method = method
        self.fitted = False
        
        # 模型参数
        self.alpha = None  # 截距项
        self.beta = None   # 因子载荷
        self.historical_mean = None
        self.arma_params = None
    
    def fit(self,
            returns: pd.DataFrame,
            factors: Optional[pd.DataFrame] = None,
            lookback: int = 252) -> 'ReturnForecaster':
        """
        拟合预测模型
        
        Args:
            returns: 股票收益率 (T x N)
            factors: 因子收益率 (T x K)，factor 方法必需
            lookback: 回看窗口（天）
        """
        # 使用最近 lookback 天的数据
        returns = returns.iloc[-lookback:] if len(returns) > lookback else returns
        
        if self.method == 'factor' and factors is not None:
            factors = factors.iloc[-lookback:] if len(factors) > lookback else factors
            self._fit_factor_model(returns, factors)
        
        if self.method in ['historical', 'ensemble']:
            self.historical_mean = returns.mean()
        
        if self.method in ['arma', 'ensemble']:
            self._fit_arma(returns)
        
        if self.method in ['momentum', 'ensemble']:
            self._fit_momentum(returns)
        
        self.fitted = True
        return self
    
    def _fit_factor_model(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """拟合因子模型: r = α + B @ F + ε"""
        common_idx = returns.index.intersection(factors.index)
        R = returns.loc[common_idx].values  # T x N
        F = factors.loc[common_idx].values  # T x K
        
        # 添加截距项
        F_with_intercept = np.column_stack([np.ones(len(F)), F])
        
        # OLS
        try:
            coeffs = np.linalg.lstsq(F_with_intercept, R, rcond=None)[0]
            self.alpha = coeffs[0, :]  # N
            self.beta = coeffs[1:, :].T  # N x K
        except np.linalg.LinAlgError:
            # Ridge regression fallback
            ridge_alpha = 0.01
            FtF = F_with_intercept.T @ F_with_intercept
            FtF += ridge_alpha * np.eye(FtF.shape[0])
            FtR = F_with_intercept.T @ R
            coeffs = np.linalg.solve(FtF, FtR)
            self.alpha = coeffs[0, :]
            self.beta = coeffs[1:, :].T
        
        # 保存因子历史用于预测
        self.factor_history = factors.copy()
        self.asset_names = returns.columns.tolist()
        self.factor_names = factors.columns.tolist()
    
    def _fit_arma(self, returns: pd.DataFrame, order: Tuple[int, int] = (2, 1)):
        """
        拟合简化的 ARMA 模型
        
        使用 AR(p) 近似: r_t = c + Σ φ_i * r_{t-i} + ε
        """
        p, q = order
        self.arma_params = {}
        
        for col in returns.columns:
            r = returns[col].dropna().values
            if len(r) < p + 10:
                self.arma_params[col] = {'c': r.mean(), 'phi': np.zeros(p)}
                continue
            
            # 构建 AR 设计矩阵
            T = len(r)
            X = np.column_stack([np.ones(T - p)] + [r[p-i-1:T-i-1] for i in range(p)])
            y = r[p:]
            
            try:
                phi = np.linalg.lstsq(X, y, rcond=None)[0]
                self.arma_params[col] = {'c': phi[0], 'phi': phi[1:]}
            except:
                self.arma_params[col] = {'c': r.mean(), 'phi': np.zeros(p)}
    
    def _fit_momentum(self, returns: pd.DataFrame, windows: List[int] = [21, 63, 126]):
        """
        计算动量/反转信号
        
        动量: 过去收益的 t-统计量
        反转: 使用 Ornstein-Uhlenbeck 半衰期
        """
        self.momentum_signals = {}
        
        for col in returns.columns:
            r = returns[col].dropna()
            if len(r) < max(windows):
                self.momentum_signals[col] = {'signal': 0, 'type': 'neutral'}
                continue
            
            # 计算不同窗口的动量
            mom_scores = []
            for w in windows:
                window_ret = r.iloc[-w:]
                t_stat = window_ret.mean() / (window_ret.std() / np.sqrt(w))
                mom_scores.append(t_stat)
            
            avg_t_stat = np.mean(mom_scores)
            
            # 计算均值回归半衰期 (AR(1) 近似)
            if len(r) > 20:
                r_lag = r.shift(1).dropna()
                r_curr = r.iloc[1:]
                rho = np.corrcoef(r_lag, r_curr)[0, 1]
                half_life = -np.log(2) / np.log(abs(rho)) if abs(rho) < 1 and rho != 0 else np.inf
            else:
                half_life = np.inf
            
            # 判断动量还是反转
            if abs(avg_t_stat) > 1.5 and half_life > 20:
                signal_type = 'momentum'
                signal = avg_t_stat
            elif half_life < 15:
                signal_type = 'reversal'
                signal = -avg_t_stat * 0.5  # 反转信号取反并降权
            else:
                signal_type = 'neutral'
                signal = 0
            
            self.momentum_signals[col] = {
                'signal': signal,
                'type': signal_type,
                't_stat': avg_t_stat,
                'half_life': half_life
            }
    
    def predict(self,
                horizon: int = 21,
                factor_forecast: Optional[np.ndarray] = None) -> pd.Series:
        """
        预测未来收益
        
        Args:
            horizon: 预测周期（天）
            factor_forecast: 因子收益预测 (K,)，factor 方法必需
            
        Returns:
            预测的期望收益 (N,)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.method == 'factor':
            return self._predict_factor(horizon, factor_forecast)
        elif self.method == 'historical':
            return self._predict_historical(horizon)
        elif self.method == 'arma':
            return self._predict_arma(horizon)
        elif self.method == 'momentum':
            return self._predict_momentum(horizon)
        elif self.method == 'ensemble':
            return self._predict_ensemble(horizon, factor_forecast)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _predict_factor(self, horizon: int, factor_forecast: Optional[np.ndarray]) -> pd.Series:
        """使用因子模型预测"""
        if factor_forecast is None:
            # 如果没有提供因子预测，使用历史均值
            factor_forecast = self.factor_history.mean().values
        
        # μ̂ = α + B @ F̂
        mu = self.alpha + self.beta @ factor_forecast
        
        # 调整到预测周期
        mu_horizon = mu * horizon
        
        return pd.Series(mu_horizon, index=self.asset_names)
    
    def _predict_historical(self, horizon: int) -> pd.Series:
        """使用历史均值预测"""
        return self.historical_mean * horizon
    
    def _predict_arma(self, horizon: int) -> pd.Series:
        """使用 ARMA 模型预测"""
        predictions = {}
        
        for col, params in self.arma_params.items():
            # 简单预测：使用无条件均值
            # 对于 AR(p): E[r] = c / (1 - Σφ_i)
            c = params['c']
            phi = params['phi']
            phi_sum = np.sum(phi)
            
            if abs(1 - phi_sum) > 0.01:
                unconditional_mean = c / (1 - phi_sum)
            else:
                unconditional_mean = c
            
            predictions[col] = unconditional_mean * horizon
        
        return pd.Series(predictions)
    
    def _predict_momentum(self, horizon: int) -> pd.Series:
        """使用动量/反转信号预测"""
        predictions = {}
        
        for col, info in self.momentum_signals.items():
            # 信号强度转换为收益预测
            # 假设：1个标准差的 t-stat 对应 0.5% 的日超额收益
            signal = info['signal']
            daily_return = signal * 0.005 / 1.96  # 归一化
            predictions[col] = daily_return * horizon
        
        return pd.Series(predictions)
    
    def _predict_ensemble(self,
                          horizon: int,
                          factor_forecast: Optional[np.ndarray]) -> pd.Series:
        """集成多种方法的预测"""
        predictions = []
        weights = []
        
        # Factor model prediction (if available)
        if self.beta is not None and factor_forecast is not None:
            pred_factor = self._predict_factor(horizon, factor_forecast)
            predictions.append(pred_factor)
            weights.append(0.4)
        
        # Historical mean
        if self.historical_mean is not None:
            pred_hist = self._predict_historical(horizon)
            predictions.append(pred_hist)
            weights.append(0.3 if predictions else 0.5)
        
        # Momentum/Reversal
        if self.momentum_signals:
            pred_mom = self._predict_momentum(horizon)
            predictions.append(pred_mom)
            weights.append(0.3)
        
        # 加权平均
        if not predictions:
            raise ValueError("No prediction method available")
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 对齐 index
        common_idx = predictions[0].index
        for p in predictions[1:]:
            common_idx = common_idx.intersection(p.index)
        
        ensemble = pd.Series(0.0, index=common_idx)
        for pred, w in zip(predictions, weights):
            ensemble += w * pred[common_idx]
        
        return ensemble
    
    def get_forecast_confidence(self) -> pd.Series:
        """
        计算预测置信度
        
        基于：
        1. 因子模型 R²
        2. 动量信号强度
        3. 历史波动率
        """
        if not self.fitted:
            return None
        
        confidence = pd.Series(0.5, index=self.asset_names)  # 默认中等置信度
        
        # 动量信号增加置信度
        if hasattr(self, 'momentum_signals'):
            for col, info in self.momentum_signals.items():
                if col in confidence.index:
                    if info['type'] == 'momentum':
                        # 强动量信号 → 高置信度
                        confidence[col] += min(0.3, abs(info['t_stat']) * 0.1)
                    elif info['type'] == 'reversal':
                        # 反转信号 → 中等置信度
                        confidence[col] += 0.1
        
        return confidence.clip(0, 1)

