"""
Stock Selector - 因子选股器

根据因子暴露和市场状态选择股票进入投资组合。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class SelectionResult:
    """选股结果"""
    selected_stocks: List[str]  # 选中的股票列表
    selected_indices: np.ndarray  # 选中股票在原始数据中的索引
    scores: pd.Series  # 所有股票的得分
    factor_loadings: pd.DataFrame  # 因子载荷矩阵
    regime: int  # 当前市场状态
    regime_name: str  # 市场状态名称
    selection_details: Dict  # 选股详细信息


class StockSelector:
    """
    因子选股器
    
    根据因子暴露和市场状态选择股票。
    
    选股逻辑：
    1. 计算每只股票对各因子的暴露 (β)
    2. 根据当前 regime 确定因子权重
    3. 计算综合得分: score_i = Σ_k w_k * zscore(β_i,k)
    4. 选择得分最高的 N 只股票
    """
    
    # 不同市场状态下的因子权重配置
    REGIME_FACTOR_WEIGHTS = {
        'bull': {
            'momentum': 0.30,
            'growth': 0.25,
            'high_beta': 0.25,
            'quality': 0.15,
            'size': 0.05,
        },
        'bear': {
            'low_volatility': 0.35,
            'value': 0.25,
            'dividend': 0.20,
            'quality': 0.15,
            'defensive': 0.05,
        },
        'crisis': {
            'liquidity': 0.40,
            'quality': 0.30,
            'low_leverage': 0.20,
            'cash_flow': 0.10,
        },
        'sideways': {
            'value': 0.30,
            'quality': 0.25,
            'momentum': 0.20,
            'dividend': 0.15,
            'size': 0.10,
        }
    }
    
    REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'crisis', 3: 'sideways'}
    
    def __init__(self,
                 n_stocks: int = 15,
                 sector_cap: float = 0.40,
                 min_liquidity: Optional[float] = None,
                 custom_weights: Optional[Dict[str, Dict[str, float]]] = None):
        """
        初始化选股器
        
        Args:
            n_stocks: 选择的股票数量
            sector_cap: 单一行业最大占比
            min_liquidity: 最小流动性要求（日均成交额）
            custom_weights: 自定义因子权重 {regime: {factor: weight}}
        """
        self.n_stocks = n_stocks
        self.sector_cap = sector_cap
        self.min_liquidity = min_liquidity
        
        # 合并自定义权重
        self.factor_weights = self.REGIME_FACTOR_WEIGHTS.copy()
        if custom_weights:
            for regime, weights in custom_weights.items():
                if regime in self.factor_weights:
                    self.factor_weights[regime].update(weights)
                else:
                    self.factor_weights[regime] = weights
    
    def select(self,
               returns: pd.DataFrame,
               factors: pd.DataFrame,
               regime: int = 0,
               sector_info: Optional[pd.Series] = None,
               liquidity_info: Optional[pd.Series] = None) -> SelectionResult:
        """
        执行选股
        
        Args:
            returns: 股票收益率 DataFrame (T x N)
            factors: 因子收益率 DataFrame (T x K)
            regime: 当前市场状态 (0=bull, 1=bear, 2=crisis, 3=sideways)
            sector_info: 股票所属行业 Series
            liquidity_info: 股票日均成交额 Series
            
        Returns:
            SelectionResult 对象
        """
        regime_name = self.REGIME_NAMES.get(regime, 'bull')
        
        print(f"\n[Stock Selection - {regime_name.upper()} Market]")
        
        # Step 1: 计算因子载荷
        factor_loadings = self._compute_factor_loadings(returns, factors)
        print(f"  Computed factor loadings: {factor_loadings.shape}")
        
        # Step 2: 获取当前 regime 的因子权重
        weights = self._get_factor_weights(regime_name, factors.columns.tolist())
        print(f"  Factor weights: {weights}")
        
        # Step 3: 计算综合得分
        scores = self._compute_scores(factor_loadings, weights)
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Step 4: 应用筛选条件
        eligible_stocks = self._apply_filters(
            scores, 
            sector_info, 
            liquidity_info
        )
        print(f"  Eligible stocks after filtering: {len(eligible_stocks)}")
        
        # Step 5: 选择得分最高的 N 只股票
        selected_stocks = self._select_top_stocks(
            scores,
            eligible_stocks,
            sector_info
        )
        print(f"  Selected {len(selected_stocks)} stocks")
        
        # 获取选中股票的索引
        all_stocks = returns.columns.tolist()
        selected_indices = np.array([all_stocks.index(s) for s in selected_stocks])
        
        # 构建详细信息
        details = {
            'regime': regime_name,
            'factor_weights': weights,
            'n_eligible': len(eligible_stocks),
            'n_selected': len(selected_stocks),
            'top_scores': scores[selected_stocks].to_dict(),
        }
        
        if sector_info is not None:
            sector_dist = sector_info[selected_stocks].value_counts(normalize=True)
            details['sector_distribution'] = sector_dist.to_dict()
        
        return SelectionResult(
            selected_stocks=selected_stocks,
            selected_indices=selected_indices,
            scores=scores,
            factor_loadings=factor_loadings,
            regime=regime,
            regime_name=regime_name,
            selection_details=details
        )
    
    def _compute_factor_loadings(self,
                                  returns: pd.DataFrame,
                                  factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子载荷矩阵 B (N x K)
        
        使用 OLS 回归: r_i = α_i + Σ_k β_ik * F_k + ε_i
        """
        # 对齐时间索引
        common_idx = returns.index.intersection(factors.index)
        R = returns.loc[common_idx].values  # T x N
        F = factors.loc[common_idx].values  # T x K
        
        # 添加截距项
        F_with_intercept = np.column_stack([np.ones(len(F)), F])
        
        # OLS: β = (F'F)^(-1) F'R
        try:
            coeffs = np.linalg.lstsq(F_with_intercept, R, rcond=None)[0]
            # 去掉截距项，保留因子载荷
            loadings = coeffs[1:, :].T  # N x K
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用 Ridge 回归
            ridge_alpha = 0.01
            FtF = F_with_intercept.T @ F_with_intercept
            FtF += ridge_alpha * np.eye(FtF.shape[0])
            FtR = F_with_intercept.T @ R
            coeffs = np.linalg.solve(FtF, FtR)
            loadings = coeffs[1:, :].T
        
        return pd.DataFrame(
            loadings,
            index=returns.columns,
            columns=factors.columns
        )
    
    def _get_factor_weights(self,
                            regime_name: str,
                            factor_names: List[str]) -> Dict[str, float]:
        """
        获取当前 regime 的因子权重
        
        如果因子名称不在预设权重中，使用等权重
        """
        preset_weights = self.factor_weights.get(regime_name, {})
        
        # 映射因子名称到权重
        weights = {}
        total_weight = 0.0
        
        for i, fname in enumerate(factor_names):
            # 尝试匹配预设权重
            matched = False
            for key, w in preset_weights.items():
                if key.lower() in fname.lower() or fname.lower() in key.lower():
                    weights[fname] = w
                    total_weight += w
                    matched = True
                    break
            
            if not matched:
                # 对于 PCA 因子，根据解释方差分配权重
                # 假设因子按重要性排序
                base_weight = 1.0 / (i + 1)
                weights[fname] = base_weight
                total_weight += base_weight
        
        # 归一化权重
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _compute_scores(self,
                        factor_loadings: pd.DataFrame,
                        weights: Dict[str, float]) -> pd.Series:
        """
        计算综合得分
        
        score_i = Σ_k w_k * zscore(β_ik)
        """
        # Z-score 标准化因子载荷
        loadings_zscore = factor_loadings.apply(stats.zscore, axis=0)
        loadings_zscore = loadings_zscore.fillna(0)
        
        # 加权求和
        scores = pd.Series(0.0, index=factor_loadings.index)
        for factor_name, weight in weights.items():
            if factor_name in loadings_zscore.columns:
                scores += weight * loadings_zscore[factor_name]
        
        return scores
    
    def _apply_filters(self,
                       scores: pd.Series,
                       sector_info: Optional[pd.Series],
                       liquidity_info: Optional[pd.Series]) -> List[str]:
        """
        应用筛选条件
        """
        eligible = set(scores.index.tolist())
        
        # 流动性筛选
        if liquidity_info is not None and self.min_liquidity is not None:
            liquid_stocks = set(liquidity_info[liquidity_info >= self.min_liquidity].index)
            eligible = eligible.intersection(liquid_stocks)
        
        # 排除得分过低的股票（负2个标准差以下）
        score_threshold = scores.mean() - 2 * scores.std()
        high_score_stocks = set(scores[scores >= score_threshold].index)
        eligible = eligible.intersection(high_score_stocks)
        
        return list(eligible)
    
    def _select_top_stocks(self,
                           scores: pd.Series,
                           eligible_stocks: List[str],
                           sector_info: Optional[pd.Series]) -> List[str]:
        """
        选择得分最高的 N 只股票，同时控制行业集中度
        """
        # 只考虑合格的股票
        eligible_scores = scores[eligible_stocks].sort_values(ascending=False)
        
        if sector_info is None:
            # 没有行业信息，直接选择 top N
            return eligible_scores.head(self.n_stocks).index.tolist()
        
        # 有行业信息，控制行业集中度
        selected = []
        sector_counts = {}
        max_per_sector = int(np.ceil(self.n_stocks * self.sector_cap))
        
        for stock in eligible_scores.index:
            if len(selected) >= self.n_stocks:
                break
            
            sector = sector_info.get(stock, 'Unknown')
            current_count = sector_counts.get(sector, 0)
            
            if current_count < max_per_sector:
                selected.append(stock)
                sector_counts[sector] = current_count + 1
        
        return selected
    
    def get_selection_summary(self, result: SelectionResult) -> str:
        """生成选股摘要"""
        lines = [
            f"\n{'='*60}",
            f"STOCK SELECTION SUMMARY ({result.regime_name.upper()} Market)",
            f"{'='*60}",
            f"Selected: {len(result.selected_stocks)} stocks",
            f"",
            "Top 10 by Score:",
        ]
        
        top_10 = result.scores[result.selected_stocks].head(10)
        for stock, score in top_10.items():
            loadings = result.factor_loadings.loc[stock]
            top_factor = loadings.abs().idxmax()
            top_loading = loadings[top_factor]
            lines.append(f"  {stock:8s}: Score={score:+.3f}, Top Factor={top_factor}({top_loading:+.2f})")
        
        if 'sector_distribution' in result.selection_details:
            lines.append("\nSector Distribution:")
            for sector, pct in result.selection_details['sector_distribution'].items():
                lines.append(f"  {sector}: {pct:.1%}")
        
        lines.append(f"{'='*60}")
        return '\n'.join(lines)

