"""
Complete Portfolio Optimization Workflow

按照实际工作流程：
0. Regime Detection (市场状态检测) - HMM识别市场regime [可选]
1. Factor Mining (挖掘因子)
2. Build Matrix (构建股票-因子关系矩阵 B)
3. Select Objective (选择优化目标)
4. Parameter Estimation (参数估计: μ, F, D, Σ) - 支持状态依赖参数
5. Evaluation (评估/回测)
6. Options Hedging (期权对冲) - 通过组合预测波动率，计算希腊字母，进行对冲

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
from typing import Optional, Dict

# Import from workflow package directory
# Use importlib to avoid naming conflict with this file
import importlib

# Import workflow package components from workflow/ directory
_workflow_pkg = importlib.import_module('workflow')

FactorMiner = _workflow_pkg.FactorMiner
FactorLoadingsEstimator = _workflow_pkg.FactorLoadingsEstimator
ObjectiveType = _workflow_pkg.ObjectiveType
ConstraintBuilder = _workflow_pkg.ConstraintBuilder
DecisionSpecs = _workflow_pkg.DecisionSpecs
QPOptimizer = _workflow_pkg.QPOptimizer
SampleEstimator = _workflow_pkg.SampleEstimator
KnowledgeBase = _workflow_pkg.KnowledgeBase
PortfolioEvaluator = _workflow_pkg.PortfolioEvaluator
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
    Complete portfolio optimization workflow following the natural process:
    
    0. Regime Detection (Optional) → 1. Factor Mining → 2. Build Matrix 
    → 3. Select Objective → 4. Estimate Parameters → 5. Evaluate → 6. Options Hedging
    
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
            returns: Asset returns DataFrame (T x N)
            use_regime: Whether to use regime-aware optimization
        """
        self.returns = returns
        self.use_regime = use_regime and REGIME_LAYER_AVAILABLE
        
        # Regime layer
        self.regime_detector = None
        self.regime_state = None
        self.regime_knowledge = None
        
        # Factor layer
        self.factors = None
        self.selected_factors = None
        self.factor_loadings = None
        
        # Parameter layer
        self.knowledge = None
        
        # Portfolio layer
        self.portfolio_weights = None
        
        # Options layer
        self.forecasted_volatility = None
        self.portfolio_greeks = None
        self.hedge_solution = None
    
    def step0_regime_detection(self,
                              n_regimes: int = 2,
                              n_iter: int = 100) -> Optional[Dict]:
        """
        Step 0: Regime Detection - 市场状态检测 (可选)
        
        使用隐马尔可夫模型(HMM)识别市场状态
        
        数学模型：
            s_t ~ Markov(P)
            r_t | s_t ~ N(μ(s_t), Σ(s_t))
        
        Args:
            n_regimes: 状态数量（默认2：牛市/熊市）
            n_iter: EM算法迭代次数
        
        Returns:
            包含regime信息的字典
        """
        if not REGIME_LAYER_AVAILABLE:
            print("\n⚠ Regime layer not available. Skipping regime detection.")
            return None
        
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
        
        print(f"\n✓ Current regime: {self.regime_state.current_regime} "
              f"({self.regime_state.get_regime_name(self.regime_state.current_regime)})")
        print(f"✓ Regime probabilities: {self.regime_state.regime_probabilities}")
        print(f"✓ Regime distribution: {np.bincount(self.regime_state.regime_sequence)}")
        
        # 打印转移矩阵
        print("\n✓ Transition matrix:")
        P = self.regime_state.transition_matrix
        regime_names = ['Bull', 'Bear', 'Sideways', 'Crisis'][:n_regimes]
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
            'current_regime': self.regime_state.current_regime,
            'regime_probabilities': self.regime_state.regime_probabilities,
            'transition_matrix': self.regime_state.transition_matrix,
            'regime_sequence': self.regime_state.regime_sequence,
            'statistics': stats
        }
    
    def step1_factor_mining(self, 
                           top_n: int = 5,
                           method: str = 'pca') -> pd.DataFrame:
        """
        Step 1: Factor Mining - 挖掘因子
        
        Extract/select factors from stocks
        
        Args:
            top_n: Number of factors to extract
            method: Mining method ('pca', 'factor_analysis', 'statistical')
        
        Returns:
            Factor returns DataFrame (T x K)
        """
        print("\n" + "="*80)
        print("STEP 1: FACTOR MINING")
        print("="*80)
        
        miner = FactorMiner(method=method)
        self.factors = miner.mine_factors(self.returns, n_factors=top_n)
        
        print(f"\n✓ Extracted {len(self.factors.columns)} factors")
        if hasattr(miner, 'get_explained_variance'):
            print(f"✓ Explained variance: {np.sum(miner.get_explained_variance()):.2%}")
        
        return self.factors
    
    def step2_build_matrix(self, 
                           factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Step 2: Build Matrix - 构建股票-因子关系矩阵
        
        Construct stock-factor relationship matrix (B)
        
        Args:
            factors: Factor returns DataFrame (T x K)，如果为 None 则使用 step1 的结果
        
        Returns:
            Factor loadings matrix B (N x K)
        """
        print("\n" + "="*80)
        print("STEP 2: BUILD MATRIX")
        print("="*80)
        
        if factors is None:
            factors = self.factors
        
        if factors is None:
            raise ValueError("Factors not available. Run step1_factor_mining() first.")
        
        estimator = FactorLoadingsEstimator(method='ols')
        
        # Drop NaN values before estimation
        common_idx = self.returns.index.intersection(factors.index)
        returns_clean = self.returns.loc[common_idx].dropna()
        factors_clean = factors.loc[common_idx].dropna()
        
        # Align indices
        common_idx = returns_clean.index.intersection(factors_clean.index)
        returns_aligned = returns_clean.loc[common_idx]
        factors_aligned = factors_clean.loc[common_idx]
        
        self.factor_loadings = estimator.estimate(returns_aligned, factors_aligned)
        
        print(f"✓ Factor loadings matrix shape: {self.factor_loadings.shape}")
        print(f"  Assets: {self.factor_loadings.shape[0]}, Factors: {self.factor_loadings.shape[1]}")
        
        return self.factor_loadings
    
    def step3_select_objective(self,
                              objective: ObjectiveType = ObjectiveType.SHARPE,
                              constraints: Optional[Dict] = None) -> DecisionSpecs:
        """
        Step 3: Select Objective - 选择优化目标
        
        Choose optimization objective, constraints, and methods
        
        Args:
            objective: Optimization objective type
            constraints: Constraint parameters dict
        
        Returns:
            DecisionSpecs object
        """
        print("\n" + "="*80)
        print("STEP 3: SELECT OBJECTIVE")
        print("="*80)
        
        if constraints is None:
            constraints = {}
        
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
        
        print(f"✓ Objective: {objective.value}")
        print(f"✓ Method: {decisions.method}")
        
        return decisions
    
    def step4_estimate_parameters(self,
                                 factors: Optional[pd.DataFrame] = None,
                                 use_regime: Optional[bool] = None) -> KnowledgeBase:
        """
        Step 4: Parameter Estimation - 参数估计
        
        Estimate μ, F, D, Σ
        支持状态依赖参数估计 (Regime-Dependent)
        
        数学模型（如果使用regime）：
            μ(s), B(s), F(s), D(s), Σ(s) for each regime s
        
        Args:
            factors: Factor returns DataFrame (optional)
            use_regime: Whether to use regime-dependent estimation (default: self.use_regime)
        
        Returns:
            KnowledgeBase object (or RegimeKnowledgeBase if using regime)
        """
        print("\n" + "="*80)
        print("STEP 4: PARAMETER ESTIMATION")
        print("="*80)
        
        if use_regime is None:
            use_regime = self.use_regime
        
        # 检查是否使用regime模式
        if use_regime and self.regime_state is not None and REGIME_LAYER_AVAILABLE:
            print("\n[Regime-Dependent Mode]")
            
            # 使用状态依赖参数估计
            regime_estimator = RegimeParameterEstimator(
                use_factor_model=(factors is not None),
                shrinkage=True
            )
            
            self.regime_knowledge = regime_estimator.estimate(
                self.returns, 
                self.regime_state,
                factors
            )
            
            print(f"✓ Number of regimes: {self.regime_knowledge.n_regimes}")
            print(f"✓ Current regime: {self.regime_knowledge.get_regime_name(self.regime_knowledge.current_regime)}")
            
            # 显示各regime参数
            for k in range(self.regime_knowledge.n_regimes):
                params = self.regime_knowledge.regime_params[k]
                name = self.regime_knowledge.get_regime_name(k)
                prob = self.regime_knowledge.regime_probabilities[k]
                
                ann_ret = np.mean(params.mu) * 252
                ann_vol = np.mean(np.sqrt(np.diag(params.Sigma))) * np.sqrt(252)
                
                print(f"\n  {name} (P={prob:.2%}):")
                print(f"    Expected return (ann.): {ann_ret:.2%}")
                print(f"    Volatility (ann.): {ann_vol:.2%}")
                if params.B is not None:
                    print(f"    Factor loadings shape: {params.B.shape}")
            
            # 转换为标准KnowledgeBase以保持向后兼容
            self.knowledge = self.regime_knowledge.to_simple_knowledge_base(method='expected')
            print(f"\n✓ Converted to standard KnowledgeBase (expected parameters)")
        else:
            # 标准模式
            print("\n[Standard Mode]")
            estimator = SampleEstimator()
            self.knowledge = estimator.estimate(self.returns, factors)
        
        print(f"\n✓ Expected returns (μ) shape: {self.knowledge.mu.shape}")
        print(f"✓ Covariance matrix (Σ) shape: {self.knowledge.get_covariance().shape}")
        
        if self.knowledge.B is not None:
            print(f"✓ Factor loadings (B) shape: {self.knowledge.B.shape}")
            print(f"✓ Factor covariance (F) shape: {self.knowledge.F.shape}")
            print(f"✓ Idiosyncratic risk (D) shape: {self.knowledge.D.shape}")
        
        return self.knowledge
    
    def step5_evaluation(self,
                        decisions: DecisionSpecs,
                        regime_strategy: str = 'expected') -> np.ndarray:
        """
        Step 5: Evaluation - 评估/回测
        
        Evaluate portfolio performance
        支持状态感知优化 (Regime-Aware Optimization)
        
        Args:
            decisions: DecisionSpecs object
            regime_strategy: Regime optimization strategy (if using regime)
                - 'expected': 期望参数优化
                - 'robust': 稳健优化（minimax）
                - 'adaptive': 自适应优化
                - 'worst_case': 最坏情况优化
                - 'multi_regime': 多regime联合优化
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("STEP 5: EVALUATION")
        print("="*80)
        
        # 检查是否使用regime感知优化
        if self.use_regime and self.regime_knowledge is not None and REGIME_LAYER_AVAILABLE:
            print(f"\n[Regime-Aware Optimization: {regime_strategy}]")
            
            # 使用状态感知优化器
            regime_optimizer = RegimeAwareOptimizer(strategy=regime_strategy)
            
            # 构建约束
            constraints = {
                'long_only': decisions.constraints.long_only,
                'leverage': decisions.constraints.leverage
            }
            if hasattr(decisions.constraints, 'max_weight') and decisions.constraints.max_weight:
                constraints['max_weight'] = decisions.constraints.max_weight
            
            # 优化
            self.portfolio_weights = regime_optimizer.optimize(
                self.regime_knowledge,
                constraints=constraints,
                objective=decisions.objective.value
            )
            
            # 计算各regime下的表现
            contribution = regime_optimizer.compute_regime_contribution(
                self.portfolio_weights, 
                self.regime_knowledge
            )
            
            print(f"\n✓ Portfolio weights calculated (Regime-Aware)")
            print(f"\n  Performance by Regime:")
            for k, info in contribution['by_regime'].items():
                print(f"    {info['name']} (P={info['probability']:.1%}): "
                      f"Return={info['return']*252:.2%}, "
                      f"Risk={info['risk']*np.sqrt(252):.2%}, "
                      f"Sharpe={info['sharpe']*np.sqrt(252):.2f}")
            
            print(f"\n  Expected Performance:")
            print(f"    Return (ann.): {contribution['expected']['return']*252:.2%}")
            print(f"    Risk (ann.): {contribution['expected']['risk']*np.sqrt(252):.2%}")
            print(f"    Sharpe (ann.): {contribution['expected']['sharpe']*np.sqrt(252):.2f}")
        else:
            # 标准优化
            print("\n[Standard Optimization]")
            optimizer = QPOptimizer()
            self.portfolio_weights = optimizer.optimize(self.knowledge, decisions)
        
        # 评估组合
        evaluator = PortfolioEvaluator()
        metrics = evaluator.evaluate(self.portfolio_weights, self.returns, self.knowledge)
        
        # 计算组合指标
        mu = self.knowledge.mu
        Sigma = self.knowledge.get_covariance()
        
        portfolio_return = np.dot(self.portfolio_weights, mu)
        portfolio_risk = np.sqrt(np.dot(self.portfolio_weights, np.dot(Sigma, self.portfolio_weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        print(f"\n✓ Portfolio metrics (from standard KnowledgeBase):")
        print(f"  Expected return: {portfolio_return:.4f}")
        print(f"  Risk (std): {portfolio_risk:.4f}")
        print(f"  Sharpe ratio: {sharpe_ratio:.4f}")
        print(f"\n  Performance Metrics:")
        print(f"    Annualized Return: {metrics.get('annualized_return', 0):.4f}")
        print(f"    Annualized Volatility: {metrics.get('annualized_volatility', 0):.4f}")
        print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
        print(f"\nTop 5 holdings:")
        top_indices = np.argsort(self.portfolio_weights)[::-1][:5]
        for idx in top_indices:
            asset_name = self.knowledge.asset_names[idx] if self.knowledge.asset_names else f"Asset {idx}"
            print(f"  {asset_name}: {self.portfolio_weights[idx]:.4f}")
        
        return self.portfolio_weights
    
    def step6_options_hedging(self,
                             spot_price: float,
                             strike: float,
                             time_to_expiry: float = 0.25,
                             risk_free_rate: float = 0.05,
                             option_type: str = 'call',
                             hedge_method: str = 'delta',
                             volatility_method: str = 'portfolio_risk') -> Dict:
        """
        Step 6: Options Hedging - 期权对冲
        
        通过资产组合构建过程预测波动率，计算希腊字母，进行对冲
        
        Args:
            spot_price: 标的资产现价
            strike: 期权执行价格
            time_to_expiry: 到期时间（年）
            risk_free_rate: 无风险利率
            option_type: 期权类型 'call' 或 'put'
            hedge_method: 对冲方法 'delta' 或 'multi_greeks'
            volatility_method: 波动率预测方法
        
        Returns:
            对冲方案字典
        """
        print("\n" + "="*80)
        print("STEP 6: OPTIONS HEDGING")
        print("="*80)
        
        if self.knowledge is None or self.portfolio_weights is None:
            raise ValueError("Knowledge base and portfolio weights required. Run steps 1-5 first.")
        
        # Step 6.1: 从组合风险结构预测波动率
        print("\n6.1: Forecasting volatility from portfolio risk structure...")
        forecaster = VolatilityForecaster(method=volatility_method)
        
        if volatility_method == 'portfolio_risk':
            self.forecasted_volatility = forecaster.forecast_from_portfolio(
                portfolio_weights=self.portfolio_weights,
                covariance_matrix=self.knowledge.get_covariance(),
                horizon=21,
                annualization=252
            )
        else:
            # 使用收益率序列
            portfolio_returns = (self.returns * self.portfolio_weights).sum(axis=1)
            self.forecasted_volatility = forecaster.forecast(
                returns=portfolio_returns,
                method=volatility_method
            )
        
        print(f"✓ Forecasted volatility: {self.forecasted_volatility:.4f} ({self.forecasted_volatility*100:.2f}%)")
        
        # Step 6.2: 计算希腊字母
        print("\n6.2: Calculating Greeks...")
        greeks_calc = GreeksCalculator()
        
        # 假设持有1份期权
        option_quantity = 1.0
        
        self.portfolio_greeks = greeks_calc.calculate_all(
            spot=spot_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=self.forecasted_volatility,
            option_type=option_type
        )
        
        print(f"✓ Greeks calculated:")
        print(f"  Δ (Delta): {self.portfolio_greeks['delta']:.4f} - Price sensitivity (Directional exposure)")
        print(f"  Γ (Gamma): {self.portfolio_greeks['gamma']:.4f} - Delta change rate (Convexity)")
        print(f"  Θ (Theta): {self.portfolio_greeks['theta']:.4f} - Time decay (Time value)")
        print(f"  V (Vega): {self.portfolio_greeks['vega']:.4f} - Volatility sensitivity")
        print(f"  ρ (Rho): {self.portfolio_greeks['rho']:.4f} - Interest rate sensitivity")
        
        # Step 6.3: 计算对冲方案
        print("\n6.3: Calculating hedging strategy...")
        
        if hedge_method == 'delta':
            hedging_strategy = DeltaHedgingStrategy()
            portfolio_delta = self.portfolio_greeks['delta'] * option_quantity
            
            self.hedge_solution = hedging_strategy.calculate_hedge(
                portfolio_delta=portfolio_delta,
                spot_price=spot_price,
                contract_size=100.0
            )
            
            print(f"✓ Delta hedging solution:")
            print(f"  Hedge quantity: {self.hedge_solution['hedge_quantity']:.2f} shares")
            print(f"  Hedge value: ${self.hedge_solution['hedge_value']:.2f}")
            print(f"  Target Delta: {self.hedge_solution['target_delta']:.4f}")
        
        else:
            print(f"  Multi-Greeks hedging not yet implemented in workflow")
            self.hedge_solution = {'strategy': 'multi_greeks', 'status': 'not_implemented'}
        
        print("\n" + "="*80)
        print("OPTIONS HEDGING COMPLETE")
        print("="*80)
        
        return {
            'forecasted_volatility': self.forecasted_volatility,
            'greeks': self.portfolio_greeks,
            'hedge_solution': self.hedge_solution
        }
    
    def run_complete_workflow(self,
                             objective: ObjectiveType = ObjectiveType.SHARPE,
                             constraints: Optional[Dict] = None,
                             n_factors: int = 5,
                             factor_method: str = 'pca',
                             n_regimes: int = 2,
                             regime_strategy: str = 'expected') -> np.ndarray:
        """
        Run the complete workflow from start to finish.
        
        Args:
            objective: Optimization objective (Step 3)
            constraints: Constraint parameters (Step 3)
            n_factors: Number of factors to extract (Step 1)
            factor_method: Factor mining method (Step 1)
            n_regimes: Number of market regimes (Step 0)
            regime_strategy: Regime optimization strategy (Step 5)
                - 'expected': 期望参数优化
                - 'robust': 稳健优化
                - 'adaptive': 自适应优化
                - 'worst_case': 最坏情况优化
                - 'multi_regime': 多regime联合优化
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("COMPLETE PORTFOLIO OPTIMIZATION WORKFLOW")
        if self.use_regime:
            print("(REGIME-AWARE MODE)")
        print("="*80)
        
        # Step 0: Regime Detection (if enabled)
        if self.use_regime:
            self.step0_regime_detection(n_regimes=n_regimes)
        
        # Step 1: Factor Mining
        self.step1_factor_mining(top_n=n_factors, method=factor_method)
        
        # Step 2: Build Matrix
        self.step2_build_matrix()
        
        # Step 3: Select Objective
        decisions = self.step3_select_objective(objective, constraints)
        
        # Step 4: Estimate Parameters
        self.step4_estimate_parameters(self.factors)
        
        # Step 5: Evaluation
        weights = self.step5_evaluation(decisions, regime_strategy=regime_strategy)
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80)
        
        return weights
