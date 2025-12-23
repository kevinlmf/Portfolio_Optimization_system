"""
Ensemble Forecaster - 集成预测器

整合收益和协方差预测，提供统一的预测接口。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .return_forecaster import ReturnForecaster
from .covariance_forecaster import CovarianceForecaster


@dataclass
class ForecastResult:
    """预测结果"""
    mu: np.ndarray  # 预期收益 (N,)
    Sigma: np.ndarray  # 协方差矩阵 (N x N)
    asset_names: List[str]  # 资产名称
    horizon: int  # 预测周期
    confidence: float  # 预测置信度 (0-1)
    return_method: str  # 收益预测方法
    cov_method: str  # 协方差预测方法
    details: Dict  # 详细信息
    
    def to_dict(self) -> Dict:
        return {
            'mu': self.mu,
            'Sigma': self.Sigma,
            'asset_names': self.asset_names,
            'horizon': self.horizon,
            'confidence': self.confidence,
            'return_method': self.return_method,
            'cov_method': self.cov_method,
        }
    
    def get_volatility(self) -> np.ndarray:
        """获取各资产的波动率"""
        return np.sqrt(np.diag(self.Sigma))
    
    def get_sharpe_estimates(self, rf: float = 0.0) -> np.ndarray:
        """获取各资产的夏普比率估计"""
        vol = self.get_volatility()
        return (self.mu - rf) / vol


class EnsembleForecaster:
    """
    集成预测器
    
    整合多种预测方法，提供统一接口。
    """
    
    def __init__(self,
                 return_methods: List[str] = ['factor', 'momentum'],
                 cov_methods: List[str] = ['factor', 'shrinkage'],
                 return_weights: Optional[List[float]] = None,
                 cov_weights: Optional[List[float]] = None):
        """
        初始化集成预测器
        
        Args:
            return_methods: 收益预测方法列表
            cov_methods: 协方差预测方法列表
            return_weights: 收益预测方法权重（默认等权重）
            cov_weights: 协方差预测方法权重（默认等权重）
        """
        self.return_methods = return_methods
        self.cov_methods = cov_methods
        
        # 设置权重
        self.return_weights = return_weights or [1/len(return_methods)] * len(return_methods)
        self.cov_weights = cov_weights or [1/len(cov_methods)] * len(cov_methods)
        
        # 预测器
        self.return_forecasters = {}
        self.cov_forecasters = {}
        
        self.fitted = False
    
    def fit(self,
            returns: pd.DataFrame,
            factors: Optional[pd.DataFrame] = None,
            lookback: int = 252) -> 'EnsembleForecaster':
        """
        拟合所有预测模型
        
        Args:
            returns: 股票收益率 (T x N)
            factors: 因子收益率 (T x K)
            lookback: 回看窗口
        """
        self.asset_names = returns.columns.tolist()
        
        # 拟合收益预测器
        for method in self.return_methods:
            forecaster = ReturnForecaster(method=method)
            forecaster.fit(returns, factors, lookback)
            self.return_forecasters[method] = forecaster
        
        # 拟合协方差预测器
        for method in self.cov_methods:
            forecaster = CovarianceForecaster(method=method)
            forecaster.fit(returns, factors, lookback)
            self.cov_forecasters[method] = forecaster
        
        # 保存因子用于预测
        self.factors = factors
        self.returns = returns
        
        self.fitted = True
        return self
    
    def predict(self,
                horizon: int = 21,
                factor_forecast: Optional[np.ndarray] = None) -> ForecastResult:
        """
        生成集成预测
        
        Args:
            horizon: 预测周期（天）
            factor_forecast: 因子收益预测 (K,)
            
        Returns:
            ForecastResult 对象
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # 收益预测
        mu, return_details = self._ensemble_return_forecast(horizon, factor_forecast)
        
        # 协方差预测
        Sigma, cov_details = self._ensemble_cov_forecast(horizon)
        
        # 计算置信度
        confidence = self._compute_confidence(mu, Sigma)
        
        return ForecastResult(
            mu=mu,
            Sigma=Sigma,
            asset_names=self.asset_names,
            horizon=horizon,
            confidence=confidence,
            return_method='+'.join(self.return_methods),
            cov_method='+'.join(self.cov_methods),
            details={
                'return': return_details,
                'covariance': cov_details
            }
        )
    
    def _ensemble_return_forecast(self,
                                   horizon: int,
                                   factor_forecast: Optional[np.ndarray]
                                   ) -> tuple:
        """集成收益预测"""
        predictions = []
        details = {}
        
        for method, weight in zip(self.return_methods, self.return_weights):
            forecaster = self.return_forecasters[method]
            
            # 生成预测
            pred = forecaster.predict(horizon, factor_forecast)
            predictions.append((pred, weight))
            
            # 保存详情
            details[method] = {
                'prediction': pred.to_dict() if isinstance(pred, pd.Series) else pred,
                'weight': weight
            }
        
        # 加权平均
        # 对齐所有预测的 index
        common_idx = predictions[0][0].index
        for pred, _ in predictions[1:]:
            common_idx = common_idx.intersection(pred.index)
        
        mu = np.zeros(len(common_idx))
        for pred, weight in predictions:
            mu += weight * pred[common_idx].values
        
        # 更新 asset_names 为共同的
        self.asset_names = common_idx.tolist()
        
        return mu, details
    
    def _ensemble_cov_forecast(self, horizon: int) -> tuple:
        """集成协方差预测"""
        predictions = []
        details = {}
        
        for method, weight in zip(self.cov_methods, self.cov_weights):
            forecaster = self.cov_forecasters[method]
            
            # 生成预测
            pred = forecaster.predict(horizon)
            
            # 只取共同资产的子矩阵
            n = len(self.asset_names)
            if pred.shape[0] > n:
                # 获取资产索引
                all_assets = forecaster.asset_names
                indices = [all_assets.index(a) for a in self.asset_names if a in all_assets]
                pred = pred[np.ix_(indices, indices)]
            
            predictions.append((pred, weight))
            
            # 保存详情
            details[method] = {
                'shape': pred.shape,
                'weight': weight,
                'avg_vol': np.sqrt(np.diag(pred)).mean()
            }
        
        # 加权平均
        Sigma = np.zeros_like(predictions[0][0])
        for pred, weight in predictions:
            Sigma += weight * pred
        
        return Sigma, details
    
    def _compute_confidence(self, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """
        计算预测置信度
        
        基于：
        1. 收益信号强度
        2. 协方差矩阵条件数
        3. 样本量
        """
        confidence = 0.5  # 基础置信度
        
        # 收益信号强度
        vol = np.sqrt(np.diag(Sigma))
        sharpe_estimates = np.abs(mu) / (vol + 1e-8)
        avg_sharpe = np.mean(sharpe_estimates)
        confidence += min(0.2, avg_sharpe * 0.1)
        
        # 协方差矩阵条件数 (越小越好)
        try:
            cond_number = np.linalg.cond(Sigma)
            if cond_number < 100:
                confidence += 0.15
            elif cond_number < 500:
                confidence += 0.05
        except:
            pass
        
        # 样本量
        if hasattr(self, 'returns') and self.returns is not None:
            T = len(self.returns)
            if T > 500:
                confidence += 0.1
            elif T > 250:
                confidence += 0.05
        
        return min(1.0, confidence)
    
    def forecast_factors(self, horizon: int = 21) -> np.ndarray:
        """
        预测因子收益
        
        使用简单的动量/均值回归方法
        """
        if self.factors is None:
            return None
        
        F = self.factors.values
        K = F.shape[1]
        
        factor_forecast = np.zeros(K)
        
        for k in range(K):
            f_k = F[:, k]
            
            # 计算动量信号
            lookback = min(63, len(f_k) - 1)
            recent_mean = f_k[-lookback:].mean()
            overall_mean = f_k.mean()
            
            # AR(1) 系数估计半衰期
            if len(f_k) > 20:
                rho = np.corrcoef(f_k[:-1], f_k[1:])[0, 1]
                half_life = -np.log(2) / np.log(abs(rho)) if abs(rho) < 1 and rho != 0 else np.inf
            else:
                half_life = np.inf
            
            # 预测
            if half_life > 20:
                # 动量：预测继续趋势
                factor_forecast[k] = recent_mean * horizon
            else:
                # 均值回归：预测回到均值
                mean_revert_speed = 1 - 0.5 ** (horizon / half_life)
                factor_forecast[k] = (1 - mean_revert_speed) * f_k[-1] + mean_revert_speed * overall_mean
                factor_forecast[k] *= horizon
        
        return factor_forecast
    
    def get_forecast_summary(self, result: ForecastResult) -> str:
        """生成预测摘要"""
        lines = [
            f"\n{'='*60}",
            f"FORECAST SUMMARY (Horizon: {result.horizon} days)",
            f"{'='*60}",
            f"Methods: Return={result.return_method}, Cov={result.cov_method}",
            f"Confidence: {result.confidence:.1%}",
            "",
            "Expected Returns (annualized):",
        ]
        
        # 年化收益
        ann_factor = 252 / result.horizon
        for i, name in enumerate(result.asset_names[:10]):
            ann_ret = result.mu[i] * ann_factor
            ann_vol = np.sqrt(result.Sigma[i, i]) * np.sqrt(ann_factor)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            lines.append(f"  {name:8s}: μ={ann_ret:+.1%}, σ={ann_vol:.1%}, SR={sharpe:.2f}")
        
        if len(result.asset_names) > 10:
            lines.append(f"  ... and {len(result.asset_names) - 10} more assets")
        
        lines.append(f"\n{'='*60}")
        return '\n'.join(lines)

