"""
Complete Portfolio Optimization Workflow

New Workflow (按照实际投资决策流程):
    0. Regime Detection (市场状态检测) - 决定后续一切
    1. Factor Mining (挖掘因子) - 根据市场状态挖掘有效因子
    2. Stock Selection (因子选股) - 根据因子暴露选股
    3. Forecasting (收益与风险预测) - 预测选中股票的收益和协方差
    4. Select Objective (选择优化目标) - 根据预测结果选择目标
    5. Optimization (组合优化) - 求解最优权重
    6. Options Hedging (期权对冲) - 对冲风险

统一生成模型 (Unified Generative Model):
    s_t ~ Markov(P)                     (market regime)
    F_t | s_t ~ D_F(s_t)               (factor dynamics)
    r_t | F_t, s_t = B(s_t)F_t + ε_t(s_t)
    ε_t(s_t) ~ N(0, Σ_ε(s_t))

Use `./run.sh workflow` to run the complete workflow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union

# Import from workflow package directory
import importlib

# Import workflow package components from workflow/ directory
_workflow_pkg = importlib.import_module('workflow')

# Step 1: Factor Mining
FactorMiner = _workflow_pkg.FactorMiner

# Step 2: Stock Selection (NEW)
StockSelector = _workflow_pkg.StockSelector
SelectionResult = _workflow_pkg.SelectionResult

# Step 2 (legacy): Build Matrix
FactorLoadingsEstimator = _workflow_pkg.FactorLoadingsEstimator

# Step 3: Forecasting (NEW)
ReturnForecaster = _workflow_pkg.ReturnForecaster
CovarianceForecaster = _workflow_pkg.CovarianceForecaster
EnsembleForecaster = _workflow_pkg.EnsembleForecaster
ForecastResult = _workflow_pkg.ForecastResult

# Step 4: Select Objective
ObjectiveType = _workflow_pkg.ObjectiveType
ConstraintBuilder = _workflow_pkg.ConstraintBuilder
DecisionSpecs = _workflow_pkg.DecisionSpecs
QPOptimizer = _workflow_pkg.QPOptimizer

# Step 4 (legacy): Parameter Estimation
SampleEstimator = _workflow_pkg.SampleEstimator
KnowledgeBase = _workflow_pkg.KnowledgeBase

# Step 5: Evaluation
PortfolioEvaluator = _workflow_pkg.PortfolioEvaluator

# Step 6: Options Hedging
VolatilityForecaster = _workflow_pkg.VolatilityForecaster
GreeksCalculator = _workflow_pkg.GreeksCalculator
DeltaHedgingStrategy = _workflow_pkg.DeltaHedgingStrategy

# Import data module
try:
    from data import APIClient
except ImportError:
    APIClient = None

# Import regime layer
try:
    from regime_layer import (
        HMMRegimeDetector,
        RegimeParameterEstimator,
        RegimeAwareOptimizer,
        RegimeKnowledgeBase
    )
    REGIME_LAYER_AVAILABLE = True
except ImportError:
    REGIME_LAYER_AVAILABLE = False
    HMMRegimeDetector = None
    RegimeParameterEstimator = None
    RegimeAwareOptimizer = None
    RegimeKnowledgeBase = None


class PortfolioWorkflow:
    """
    Complete portfolio optimization workflow following the new process:
    
    0. Regime Detection → 1. Factor Mining → 2. Stock Selection 
    → 3. Forecasting → 4. Select Objective → 5. Optimization → 6. Options Hedging
    
    统一生成模型：
        s_t ~ Markov(P)                     (market regime)
        F_t | s_t ~ D_F(s_t)               (factor dynamics)
        r_t | F_t, s_t = B(s_t)F_t + ε_t(s_t)
        ε_t(s_t) ~ N(0, Σ_ε(s_t))
    """
    
    def __init__(self, returns: pd.DataFrame, use_regime: bool = False):
        """
        Initialize workflow with asset returns.
        
        Args:
            returns: Asset returns DataFrame (T x N) - full universe
            use_regime: Whether to use regime-aware optimization
        """
        self.returns = returns
        self.universe = returns.columns.tolist()
        self.use_regime = use_regime and REGIME_LAYER_AVAILABLE
        
        # Step 0: Regime layer
        self.regime_detector = None
        self.regime_state = None
        self.current_regime = 0  # Default: Bull
        self.regime_knowledge = None
        
        # Step 1: Factor layer
        self.factors = None
        self.factor_miner = None
        
        # Step 2: Selection layer (NEW)
        self.selected_stocks = None
        self.selection_result = None
        self.selected_returns = None
        
        # Step 3: Forecasting layer (NEW)
        self.forecast_result = None
        self.forecaster = None
        
        # Step 4: Objective layer
        self.objective = None
        self.constraints = None
        
        # Step 5: Optimization layer
        self.portfolio_weights = None
        self.knowledge = None
        
        # Step 6: Hedging layer
        self.hedge_result = None
    
    # =========================================================================
    # STEP 0: Regime Detection
    # =========================================================================
    def step0_regime_detection(self,
                              n_regimes: int = 2,
                              n_iter: int = 100) -> Optional[Dict]:
        """
        Step 0: Regime Detection - 市场状态检测
        
        使用隐马尔可夫模型(HMM)识别市场状态。
        必须先执行此步骤，因为后续步骤都依赖于市场状态。
        
        Args:
            n_regimes: 状态数量（2=牛熊, 3=牛熊危机）
            n_iter: EM算法迭代次数
        
        Returns:
            包含regime信息的字典
        """
        if not REGIME_LAYER_AVAILABLE:
            print("\n⚠ Regime layer not available. Using default regime (Bull).")
            self.current_regime = 0
            return {'current_regime': 0, 'regime_name': 'Bull'}
        
        print("\n" + "="*80)
        print("STEP 0: REGIME DETECTION (HMM)")
        print("="*80)
        
        # 创建HMM检测器
        self.regime_detector = HMMRegimeDetector(
            n_regimes=n_regimes,
            n_iter=n_iter
        )
        
        # 训练模型
        print(f"\nTraining HMM with {n_regimes} regimes...")
        self.regime_detector.fit(self.returns)
        
        # 检测状态
        self.regime_state = self.regime_detector.detect(self.returns)
        self.current_regime = self.regime_state.current_regime
        
        regime_names = ['Bull', 'Bear', 'Crisis', 'Sideways'][:n_regimes]
        current_name = regime_names[self.current_regime]
        
        print(f"\n✓ Current regime: {self.current_regime} ({current_name})")
        print(f"✓ Regime probabilities: {self.regime_state.regime_probabilities}")
        print(f"✓ Regime distribution: {np.bincount(self.regime_state.regime_sequence)}")
        
        # 打印转移矩阵
        print("\n✓ Transition matrix:")
        P = self.regime_state.transition_matrix
        print(f"  {'':>8}" + "".join([f"{name:>10}" for name in regime_names]))
        for i in range(n_regimes):
            row = f"  {regime_names[i]:>8}"
            for j in range(n_regimes):
                row += f"{P[i,j]:>10.2%}"
            print(row)
        
        # 获取各regime的统计信息
        stats = self.regime_detector.get_regime_statistics()
        print("\n✓ Regime statistics:")
        for k in range(n_regimes):
            ann_ret = np.mean(stats[k]['mean']) * 252
            ann_vol = np.mean(stats[k]['volatility']) * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            print(f"  {regime_names[k]}: Return={ann_ret:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}")
        
        return {
            'current_regime': self.current_regime,
            'regime_name': current_name,
            'regime_probabilities': self.regime_state.regime_probabilities,
            'transition_matrix': self.regime_state.transition_matrix,
            'regime_sequence': self.regime_state.regime_sequence,
            'statistics': stats
        }
    
    # =========================================================================
    # STEP 1: Factor Mining
    # =========================================================================
    def step1_factor_mining(self, 
                           n_factors: int = 5,
                           method: str = 'pca',
                           regime_conditional: bool = True) -> pd.DataFrame:
        """
        Step 1: Factor Mining - 挖掘因子
        
        根据当前市场状态挖掘有效因子。
        
        Args:
            n_factors: 因子数量
            method: 挖掘方法 ('pca', 'factor_analysis')
            regime_conditional: 是否根据regime调整因子权重
        
        Returns:
            Factor returns DataFrame (T x K)
        """
        print("\n" + "="*80)
        print("STEP 1: FACTOR MINING")
        if regime_conditional and self.current_regime is not None:
            regime_names = ['Bull', 'Bear', 'Crisis', 'Sideways']
            print(f"  [Regime-Conditional: {regime_names[self.current_regime]}]")
        print("="*80)
        
        self.factor_miner = FactorMiner(method=method)
        self.factors = self.factor_miner.mine_factors(self.returns, n_factors=n_factors)
        
        print(f"\n✓ Extracted {len(self.factors.columns)} factors")
        if hasattr(self.factor_miner, 'get_explained_variance'):
            explained_var = self.factor_miner.get_explained_variance()
            print(f"✓ Explained variance: {np.sum(explained_var):.2%}")
            for i, var in enumerate(explained_var):
                print(f"  Factor {i+1}: {var:.2%}")
        
        return self.factors
    
    # =========================================================================
    # STEP 2: Stock Selection (NEW)
    # =========================================================================
    def step2_stock_selection(self,
                              n_stocks: int = 15,
                              sector_cap: float = 0.40,
                              sector_info: Optional[pd.Series] = None,
                              liquidity_info: Optional[pd.Series] = None) -> List[str]:
        """
        Step 2: Stock Selection - 因子选股
        
        根据因子暴露和当前市场状态选择股票进入组合。
        
        Args:
            n_stocks: 选择的股票数量
            sector_cap: 单一行业最大占比
            sector_info: 股票所属行业
            liquidity_info: 股票流动性（日均成交额）
        
        Returns:
            选中的股票列表
        """
        print("\n" + "="*80)
        print("STEP 2: STOCK SELECTION")
        print("="*80)
        
        if self.factors is None:
            raise ValueError("Factors not available. Run step1_factor_mining() first.")
        
        # 创建选股器
        selector = StockSelector(
            n_stocks=n_stocks,
            sector_cap=sector_cap
        )
        
        # 执行选股
        self.selection_result = selector.select(
            returns=self.returns,
            factors=self.factors,
            regime=self.current_regime,
            sector_info=sector_info,
            liquidity_info=liquidity_info
        )
        
        self.selected_stocks = self.selection_result.selected_stocks
        
        # 筛选选中股票的收益率
        self.selected_returns = self.returns[self.selected_stocks]
        
        print(f"\n✓ Selected {len(self.selected_stocks)} stocks from {len(self.universe)} universe")
        print(f"✓ Selection method: Factor-based scoring")
        print(f"\nTop 5 selected stocks by score:")
        top_5 = self.selection_result.scores[self.selected_stocks].head(5)
        for stock, score in top_5.items():
            print(f"  {stock}: score = {score:.3f}")
        
        # 打印选股详情
        print(selector.get_selection_summary(self.selection_result))
        
        return self.selected_stocks
    
    # =========================================================================
    # STEP 3: Forecasting (NEW)
    # =========================================================================
    def step3_forecasting(self,
                          horizon: int = 21,
                          return_methods: List[str] = ['factor', 'momentum'],
                          cov_methods: List[str] = ['factor', 'shrinkage']) -> ForecastResult:
        """
        Step 3: Forecasting - 收益与风险预测
        
        预测选中股票的未来收益和协方差。
        
        Args:
            horizon: 预测周期（天）
            return_methods: 收益预测方法列表
            cov_methods: 协方差预测方法列表
        
        Returns:
            ForecastResult 对象
        """
        print("\n" + "="*80)
        print("STEP 3: FORECASTING")
        print("="*80)
        
        if self.selected_returns is None:
            # 如果没有选股，使用全部股票
            print("  [Using full universe - no stock selection performed]")
            self.selected_returns = self.returns
            self.selected_stocks = self.universe
        
        # 创建集成预测器
        self.forecaster = EnsembleForecaster(
            return_methods=return_methods,
            cov_methods=cov_methods
        )
        
        # 获取选中股票对应的因子
        if self.factors is not None:
            selected_factors = self.factors
        else:
            selected_factors = None
        
        # 拟合模型
        print(f"\nFitting forecasting models...")
        print(f"  Return methods: {return_methods}")
        print(f"  Covariance methods: {cov_methods}")
        
        self.forecaster.fit(
            returns=self.selected_returns,
            factors=selected_factors,
            lookback=min(252, len(self.selected_returns))
        )
        
        # 预测因子收益（如果有因子模型）
        factor_forecast = None
        if selected_factors is not None and 'factor' in return_methods:
            factor_forecast = self.forecaster.forecast_factors(horizon)
            print(f"\n✓ Factor forecast (next {horizon} days):")
            for i, f in enumerate(factor_forecast):
                print(f"  Factor {i+1}: {f*100:.2f}%")
        
        # 生成预测
        self.forecast_result = self.forecaster.predict(
            horizon=horizon,
            factor_forecast=factor_forecast
        )
        
        print(f"\n✓ Forecast generated for {len(self.forecast_result.asset_names)} assets")
        print(f"✓ Horizon: {horizon} days")
        print(f"✓ Confidence: {self.forecast_result.confidence:.1%}")
        
        # 打印预测摘要
        print(self.forecaster.get_forecast_summary(self.forecast_result))
        
        return self.forecast_result
    
    # =========================================================================
    # STEP 4: Select Objective
    # =========================================================================
    def step4_select_objective(self,
                               objective: Optional[ObjectiveType] = None,
                               constraints: Optional[Dict] = None,
                               auto_select: bool = True) -> DecisionSpecs:
        """
        Step 4: Select Objective - 选择优化目标
        
        根据预测结果和市场状态选择优化目标。
        
        Args:
            objective: 优化目标（如果 auto_select=True 会自动选择）
            constraints: 约束条件
            auto_select: 是否根据预测自动选择目标
        
        Returns:
            DecisionSpecs 对象
        """
        print("\n" + "="*80)
        print("STEP 4: SELECT OBJECTIVE")
        print("="*80)
        
        if constraints is None:
            constraints = {'long_only': True, 'max_weight': 0.15}
        
        # 自动选择目标
        if auto_select and objective is None:
            objective = self._auto_select_objective()
        elif objective is None:
            objective = ObjectiveType.SHARPE
        
        self.objective = objective
        self.constraints = constraints
        
        # 构建约束
        constraint_builder = ConstraintBuilder()
        constraint_builder.long_only(constraints.get('long_only', True))
        constraint_builder.leverage(constraints.get('leverage', 1.0))
        
        if 'max_weight' in constraints:
            constraint_builder.max_weight(constraints['max_weight'])
        if 'min_weight' in constraints:
            constraint_builder.min_weight(constraints['min_weight'])
        
        decisions = DecisionSpecs(
            objective=objective,
            constraints=constraint_builder.build(),
            method='qp'
        )
        
        print(f"\n✓ Objective: {objective.value}")
        print(f"✓ Method: {decisions.method}")
        print(f"✓ Constraints: {constraints}")
        
        if auto_select:
            print(f"\n  [Auto-selected based on regime and forecast confidence]")
        
        return decisions
    
    def _auto_select_objective(self) -> ObjectiveType:
        """根据市场状态和预测置信度自动选择目标"""
        regime_names = ['Bull', 'Bear', 'Crisis', 'Sideways']
        regime_name = regime_names[self.current_regime] if self.current_regime < len(regime_names) else 'Bull'
        
        confidence = self.forecast_result.confidence if self.forecast_result else 0.5
        
        if regime_name == 'Bull' and confidence > 0.6:
            return ObjectiveType.SHARPE  # 高信心牛市：追求收益
        elif regime_name == 'Bear':
            return ObjectiveType.MIN_VARIANCE  # 熊市：最小化风险
        elif regime_name == 'Crisis':
            return ObjectiveType.CVAR  # 危机：控制尾部风险
        elif confidence < 0.4:
            return ObjectiveType.RISK_PARITY  # 低信心：风险平价
        else:
            return ObjectiveType.SHARPE
    
    # =========================================================================
    # STEP 5: Optimization
    # =========================================================================
    def step5_optimization(self,
                           decisions: Optional[DecisionSpecs] = None,
                           regime_aware: bool = True,
                           regime_strategy: str = 'robust') -> np.ndarray:
        """
        Step 5: Portfolio Optimization - 组合优化
        
        求解最优权重。
        
        Args:
            decisions: DecisionSpecs 对象
            regime_aware: 是否使用 regime-aware 优化
            regime_strategy: Regime 优化策略
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("STEP 5: OPTIMIZATION")
        print("="*80)
        
        # 获取预测参数
        if self.forecast_result is not None:
            mu = self.forecast_result.mu
            Sigma = self.forecast_result.Sigma
            asset_names = self.forecast_result.asset_names
        else:
            raise ValueError("Forecast not available. Run step3_forecasting() first.")
        
        # 创建 KnowledgeBase
        self.knowledge = KnowledgeBase(
            mu=mu,
            Sigma=Sigma,
            asset_names=asset_names
        )
        
        # 获取决策规格
        if decisions is None:
            decisions = self.step4_select_objective()
        
        # 优化
        if regime_aware and self.use_regime and self.regime_knowledge is not None and REGIME_LAYER_AVAILABLE:
            print(f"\n[Regime-Aware Optimization: {regime_strategy}]")
            regime_optimizer = RegimeAwareOptimizer(strategy=regime_strategy)
            
            constraints = {
                'long_only': decisions.constraints.long_only,
                'leverage': decisions.constraints.leverage
            }
            if hasattr(decisions.constraints, 'max_weight') and decisions.constraints.max_weight:
                constraints['max_weight'] = decisions.constraints.max_weight
            
            self.portfolio_weights = regime_optimizer.optimize(
                self.regime_knowledge,
                constraints=constraints,
                objective=decisions.objective.value
            )
        else:
            print("\n[Standard Optimization]")
            optimizer = QPOptimizer()
            self.portfolio_weights = optimizer.optimize(self.knowledge, decisions)
        
        # 计算组合指标
        portfolio_return = np.dot(self.portfolio_weights, mu)
        portfolio_risk = np.sqrt(np.dot(self.portfolio_weights, np.dot(Sigma, self.portfolio_weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # 年化
        horizon = self.forecast_result.horizon if self.forecast_result else 21
        ann_factor = 252 / horizon
        
        print(f"\n✓ Portfolio optimized for {len(asset_names)} assets")
        print(f"\n  Portfolio Metrics (annualized):")
        print(f"    Expected Return: {portfolio_return * ann_factor:.2%}")
        print(f"    Volatility: {portfolio_risk * np.sqrt(ann_factor):.2%}")
        print(f"    Sharpe Ratio: {sharpe_ratio * np.sqrt(ann_factor):.2f}")
        
        print(f"\n  Top Holdings:")
        top_indices = np.argsort(self.portfolio_weights)[::-1][:5]
        for idx in top_indices:
            if self.portfolio_weights[idx] > 0.01:
                print(f"    {asset_names[idx]}: {self.portfolio_weights[idx]:.2%}")
        
        return self.portfolio_weights
    
    # =========================================================================
    # STEP 6: Options Hedging
    # =========================================================================
    def step6_options_hedging(self,
                             spot_price: float = 100.0,
                             strike_pct: float = 0.95,
                             time_to_expiry: float = 0.25,
                             risk_free_rate: float = 0.05,
                             hedge_type: str = 'protective_put') -> Dict:
        """
        Step 6: Options Hedging - 期权对冲
        
        对组合进行期权对冲。
        
        Args:
            spot_price: 标的资产现价
            strike_pct: 执行价格相对于现价的比例
            time_to_expiry: 到期时间（年）
            risk_free_rate: 无风险利率
            hedge_type: 对冲类型
        
        Returns:
            对冲方案字典
        """
        print("\n" + "="*80)
        print("STEP 6: OPTIONS HEDGING")
        print("="*80)
        
        if self.portfolio_weights is None or self.knowledge is None:
            raise ValueError("Portfolio weights not available. Run step5_optimization() first.")
        
        strike = spot_price * strike_pct
        option_type = 'put' if hedge_type == 'protective_put' else 'call'
        
        # 计算组合波动率
        print("\n6.1: Computing portfolio volatility...")
        Sigma = self.knowledge.Sigma if hasattr(self.knowledge, 'Sigma') else self.knowledge.get_covariance()
        portfolio_variance = np.dot(self.portfolio_weights, np.dot(Sigma, self.portfolio_weights))
        
        # 年化
        horizon = self.forecast_result.horizon if self.forecast_result else 21
        ann_factor = 252 / horizon
        portfolio_vol = np.sqrt(portfolio_variance * ann_factor)
        
        print(f"✓ Portfolio volatility: {portfolio_vol:.2%} (annualized)")
        
        # 计算希腊字母
        print("\n6.2: Calculating Greeks...")
        greeks_calc = GreeksCalculator()
        
        portfolio_greeks = greeks_calc.calculate_all(
            spot=spot_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=portfolio_vol,
            option_type=option_type
        )
        
        print(f"✓ Greeks calculated:")
        print(f"  Δ (Delta): {portfolio_greeks['delta']:.4f}")
        print(f"  Γ (Gamma): {portfolio_greeks['gamma']:.4f}")
        print(f"  Θ (Theta): {portfolio_greeks['theta']:.4f}")
        print(f"  ν (Vega): {portfolio_greeks['vega']:.4f}")
        print(f"  ρ (Rho): {portfolio_greeks['rho']:.4f}")
        
        # 计算对冲方案
        print("\n6.3: Computing hedge strategy...")
        hedging_strategy = DeltaHedgingStrategy()
        
        hedge_solution = hedging_strategy.calculate_hedge(
            portfolio_delta=portfolio_greeks['delta'],
            spot_price=spot_price,
            contract_size=100.0
        )
        
        print(f"✓ Hedge solution ({hedge_type}):")
        print(f"  Strike: ${strike:.2f} ({strike_pct:.0%} of spot)")
        print(f"  Hedge quantity: {hedge_solution['hedge_quantity']:.2f} shares")
        print(f"  Hedge value: ${hedge_solution['hedge_value']:.2f}")
        
        self.hedge_result = {
            'portfolio_volatility': portfolio_vol,
            'greeks': portfolio_greeks,
            'hedge_solution': hedge_solution,
            'hedge_type': hedge_type,
            'strike': strike,
            'option_type': option_type
        }
        
        print("\n" + "="*80)
        print("HEDGING COMPLETE")
        print("="*80)
        
        return self.hedge_result
    
    # =========================================================================
    # Complete Workflow
    # =========================================================================
    def run_complete_workflow(self,
                             n_regimes: int = 2,
                             n_factors: int = 5,
                             n_stocks: int = 15,
                             horizon: int = 21,
                             objective: Optional[ObjectiveType] = None,
                             constraints: Optional[Dict] = None,
                             regime_strategy: str = 'robust',
                             include_hedging: bool = False) -> np.ndarray:
        """
        Run the complete new workflow from start to finish.
        
        Args:
            n_regimes: Number of market regimes
            n_factors: Number of factors to extract
            n_stocks: Number of stocks to select
            horizon: Forecast horizon (days)
            objective: Optimization objective (auto-selected if None)
            constraints: Constraint parameters
            regime_strategy: Regime optimization strategy
            include_hedging: Whether to include options hedging
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("COMPLETE PORTFOLIO OPTIMIZATION WORKFLOW (NEW)")
        print("="*80)
        print(f"Universe: {len(self.universe)} stocks")
        print(f"Target: Select {n_stocks} stocks, forecast {horizon} days")
        
        # Step 0: Regime Detection
        if self.use_regime:
            self.step0_regime_detection(n_regimes=n_regimes)
        else:
            print("\n[Regime detection disabled]")
            self.current_regime = 0
        
        # Step 1: Factor Mining
        self.step1_factor_mining(n_factors=n_factors)
        
        # Step 2: Stock Selection (NEW)
        self.step2_stock_selection(n_stocks=n_stocks)
        
        # Step 3: Forecasting (NEW)
        self.step3_forecasting(horizon=horizon)
        
        # Step 4: Select Objective
        decisions = self.step4_select_objective(
            objective=objective,
            constraints=constraints,
            auto_select=(objective is None)
        )
        
        # Step 5: Optimization
        weights = self.step5_optimization(
            decisions=decisions,
            regime_aware=self.use_regime,
            regime_strategy=regime_strategy
        )
        
        # Step 6: Options Hedging (optional)
        if include_hedging:
            self.step6_options_hedging()
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80)
        print(f"\nFinal Portfolio: {len(self.selected_stocks)} stocks")
        print(f"Optimization: {decisions.objective.value}")
        
        return weights
    
    # =========================================================================
    # Legacy Methods (Backward Compatibility)
    # =========================================================================
    def step2_build_matrix(self, factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        [LEGACY] Step 2: Build Matrix - 构建股票-因子关系矩阵
        
        Kept for backward compatibility. Use step2_stock_selection() instead.
        """
        print("\n[LEGACY] Running step2_build_matrix for backward compatibility")
        
        if factors is None:
            factors = self.factors
        
        if factors is None:
            raise ValueError("Factors not available. Run step1_factor_mining() first.")
        
        estimator = FactorLoadingsEstimator(method='ols')
        
        common_idx = self.returns.index.intersection(factors.index)
        returns_clean = self.returns.loc[common_idx].dropna()
        factors_clean = factors.loc[common_idx].dropna()
        
        common_idx = returns_clean.index.intersection(factors_clean.index)
        returns_aligned = returns_clean.loc[common_idx]
        factors_aligned = factors_clean.loc[common_idx]
        
        factor_loadings = estimator.estimate(returns_aligned, factors_aligned)
        
        print(f"✓ Factor loadings matrix shape: {factor_loadings.shape}")
        
        return factor_loadings
    
    def step3_select_objective(self,
                              objective: ObjectiveType = ObjectiveType.SHARPE,
                              constraints: Optional[Dict] = None) -> DecisionSpecs:
        """
        [LEGACY] Alias for step4_select_objective
        """
        return self.step4_select_objective(
            objective=objective,
            constraints=constraints,
            auto_select=False
        )
    
    def step4_estimate_parameters(self,
                                 factors: Optional[pd.DataFrame] = None,
                                 use_regime: Optional[bool] = None) -> KnowledgeBase:
        """
        [LEGACY] Step 4: Parameter Estimation
        
        Kept for backward compatibility. Use step3_forecasting() instead.
        """
        print("\n[LEGACY] Running step4_estimate_parameters for backward compatibility")
        
        if factors is None:
            factors = self.factors
        
        if use_regime is None:
            use_regime = self.use_regime
        
        if use_regime and self.regime_state is not None and REGIME_LAYER_AVAILABLE:
            print("[Regime-Dependent Mode]")
            regime_estimator = RegimeParameterEstimator(
                use_factor_model=(factors is not None),
                shrinkage=True
            )
            
            self.regime_knowledge = regime_estimator.estimate(
                self.returns, 
                self.regime_state,
                factors
            )
            
            self.knowledge = self.regime_knowledge.to_simple_knowledge_base(method='expected')
        else:
            print("[Standard Mode]")
            estimator = SampleEstimator()
            self.knowledge = estimator.estimate(self.returns, factors)
        
        print(f"✓ Expected returns (μ) shape: {self.knowledge.mu.shape}")
        print(f"✓ Covariance matrix (Σ) shape: {self.knowledge.get_covariance().shape}")
        
        return self.knowledge
    
    def step5_evaluation(self,
                        decisions: DecisionSpecs,
                        regime_strategy: str = 'expected') -> np.ndarray:
        """
        [LEGACY] Alias for step5_optimization
        """
        return self.step5_optimization(
            decisions=decisions,
            regime_aware=self.use_regime,
            regime_strategy=regime_strategy
        )
