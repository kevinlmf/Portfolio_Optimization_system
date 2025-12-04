"""
Complete Portfolio Optimization Workflow

按照实际工作流程：
1. Factor Mining (挖掘因子)
2. Build Matrix (构建股票-因子关系矩阵 B)
3. Select Objective (选择优化目标)
4. Parameter Estimation (参数估计: μ, F, D, Σ)
5. Evaluation (评估/回测)
6. Options Hedging (期权对冲) - 通过组合预测波动率，计算希腊字母，进行对冲

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


class PortfolioWorkflow:
    """
    Complete portfolio optimization workflow following the natural process:
    
    1. Factor Mining → 2. Build Matrix → 3. Select Objective 
    → 4. Estimate Parameters → 5. Evaluate → 6. Options Hedging
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize workflow with asset returns.
        
        Args:
            returns: Asset returns DataFrame (T x N)
        """
        self.returns = returns
        self.factors = None
        self.selected_factors = None
        self.factor_loadings = None
        self.knowledge = None
        self.portfolio_weights = None
        self.forecasted_volatility = None
        self.portfolio_greeks = None
        self.hedge_solution = None
    
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
                                 factors: Optional[pd.DataFrame] = None) -> KnowledgeBase:
        """
        Step 4: Parameter Estimation - 参数估计
        
        Estimate μ, F, D, Σ
        
        Args:
            factors: Factor returns DataFrame (optional)
        
        Returns:
            KnowledgeBase object
        """
        print("\n" + "="*80)
        print("STEP 4: PARAMETER ESTIMATION")
        print("="*80)
        
        estimator = SampleEstimator()
        self.knowledge = estimator.estimate(self.returns, factors)
        
        print(f"✓ Expected returns (μ) shape: {self.knowledge.mu.shape}")
        print(f"✓ Covariance matrix (Σ) shape: {self.knowledge.get_covariance().shape}")
        
        if self.knowledge.B is not None:
            print(f"✓ Factor loadings (B) shape: {self.knowledge.B.shape}")
            print(f"✓ Factor covariance (F) shape: {self.knowledge.F.shape}")
            print(f"✓ Idiosyncratic risk (D) shape: {self.knowledge.D.shape}")
        
        return self.knowledge
    
    def step5_evaluation(self,
                        decisions: DecisionSpecs) -> np.ndarray:
        """
        Step 5: Evaluation - 评估/回测
        
        Evaluate portfolio performance
        
        Args:
            decisions: DecisionSpecs object
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("STEP 5: EVALUATION")
        print("="*80)
        
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
        
        print(f"\n✓ Portfolio weights calculated")
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
                             factor_method: str = 'pca') -> np.ndarray:
        """
        Run the complete workflow from start to finish.
        
        Args:
            objective: Optimization objective (Step 3)
            constraints: Constraint parameters (Step 3)
            n_factors: Number of factors to extract (Step 1)
            factor_method: Factor mining method (Step 1)
        
        Returns:
            Optimal portfolio weights
        """
        print("\n" + "="*80)
        print("COMPLETE PORTFOLIO OPTIMIZATION WORKFLOW")
        print("="*80)
        
        # Step 1: Factor Mining
        self.step1_factor_mining(top_n=n_factors, method=factor_method)
        
        # Step 2: Build Matrix
        self.step2_build_matrix()
        
        # Step 3: Select Objective
        decisions = self.step3_select_objective(objective, constraints)
        
        # Step 4: Estimate Parameters
        self.step4_estimate_parameters(self.factors)
        
        # Step 5: Evaluation
        weights = self.step5_evaluation(decisions)
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80)
        
        return weights
