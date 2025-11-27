"""
Complete Portfolio Optimization Workflow

按照实际工作流程：
1. Factor Mining (挖掘因子)
2. Build Matrix (构建股票-因子关系矩阵 B)
3. Select Objective (选择优化目标)
4. Parameter Estimation (参数估计: μ, F, D, Σ)
5. Evaluation (评估/回测)

Run `python workflow.py` to see the complete workflow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

# Import from workflow package
from workflow import (
    # Step 1
    FactorMiner, FactorSelector,
    # Step 2
    FactorLoadingsEstimator, CorrelationMatrixBuilder,
    # Step 3
    ObjectiveType, ConstraintBuilder, DecisionSpecs, QPOptimizer,
    # Step 4
    SampleEstimator, KnowledgeBase,
    # Step 5
    PortfolioEvaluator
)
from data import APIClient


class PortfolioWorkflow:
    """
    Complete portfolio optimization workflow following the natural process:
    
    1. Factor Mining → 2. Build Matrix → 3. Select Objective 
    → 4. Estimate Parameters → 5. Evaluate
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
        self.factor_loadings = estimator.estimate(self.returns, factors)
        
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


def main():
    """Demo of the complete workflow"""
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION WORKFLOW DEMO")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'DIS', 'NVDA']
    
    # Generate returns
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(assets)) * 0.01,
        index=dates,
        columns=assets
    )
    
    # Initialize workflow
    workflow = PortfolioWorkflow(returns)
    
    # Run complete workflow
    weights = workflow.run_complete_workflow(
        objective=ObjectiveType.SHARPE,
        constraints={'long_only': True, 'max_weight': 0.3},
        n_factors=5
    )
    
    print(f"\nFinal portfolio weights:\n{weights}")


if __name__ == '__main__':
    main()
