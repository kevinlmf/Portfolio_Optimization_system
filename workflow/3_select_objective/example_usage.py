"""
Select Objective Usage Example

选择优化目标使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from workflow import (
    ObjectiveType, ObjectiveFunction,
    Constraints, ConstraintBuilder,
    DecisionSpecs, QPOptimizer
)
# Import objective classes directly from module
import importlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
obj_module = importlib.import_module('workflow.3_select_objective.objectives')
SharpeObjective = obj_module.SharpeObjective
CVaRObjective = obj_module.CVaRObjective
RiskParityObjective = obj_module.RiskParityObjective
MinVarianceObjective = obj_module.MinVarianceObjective
MeanVarianceObjective = obj_module.MeanVarianceObjective
from data import APIClient


def example_objective_types():
    """优化目标类型示例"""
    print("=" * 80)
    print("EXAMPLE 1: Objective Types")
    print("=" * 80)
    
    # 创建示例数据
    n_assets = 5
    mu = np.random.randn(n_assets) * 0.01
    Sigma = np.random.randn(n_assets, n_assets)
    Sigma = Sigma @ Sigma.T  # 确保正定
    weights = np.ones(n_assets) / n_assets
    
    # 测试不同目标函数
    objectives = {
        'Sharpe': ObjectiveType.SHARPE,
        'CVaR': ObjectiveType.CVAR,
        'Risk Parity': ObjectiveType.RISK_PARITY,
        'Min Variance': ObjectiveType.MIN_VARIANCE,
        'Mean-Variance': ObjectiveType.MEAN_VARIANCE
    }
    
    print("\nObjective function values:")
    for name, obj_type in objectives.items():
        if obj_type == ObjectiveType.SHARPE:
            obj_func = SharpeObjective()
        elif obj_type == ObjectiveType.CVAR:
            obj_func = CVaRObjective()
        elif obj_type == ObjectiveType.RISK_PARITY:
            obj_func = RiskParityObjective()
        elif obj_type == ObjectiveType.MIN_VARIANCE:
            obj_func = MinVarianceObjective()
        else:
            obj_func = MeanVarianceObjective()
        
        value = obj_func.evaluate(weights, mu, Sigma)
        print(f"  {name}: {value:.6f}")


def example_constraints():
    """约束条件示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Constraints")
    print("=" * 80)
    
    # 构建约束
    builder = ConstraintBuilder()
    builder.long_only(True)
    builder.leverage(1.0)
    builder.max_weight(0.3)
    builder.min_weight(0.05)
    
    constraints = builder.build()
    
    print("\nConstraint settings:")
    print(f"  Long only: {constraints.long_only}")
    print(f"  Leverage: {constraints.leverage}")
    print(f"  Max weight: {constraints.max_weight}")
    print(f"  Min weight: {constraints.min_weight}")
    
    # 获取边界
    bounds = constraints.get_bounds(n_assets=5)
    print(f"\nWeight bounds (5 assets):")
    for i, (lower, upper) in enumerate(bounds):
        print(f"  Asset {i}: [{lower:.2f}, {upper:.2f}]")


def example_optimization():
    """优化示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Portfolio Optimization")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 创建简单的知识库对象
    class SimpleKnowledge:
        def __init__(self, returns):
            self.mu = returns.mean().values
            self.returns = returns
            self.B = None
            self.factor_names = None
        
        def get_covariance(self):
            return returns.cov().values
    
    knowledge = SimpleKnowledge(returns)
    
    # 构建约束
    builder = ConstraintBuilder()
    builder.long_only(True)
    builder.leverage(1.0)
    builder.max_weight(0.3)
    constraints = builder.build()
    
    # 创建决策规格
    decisions = DecisionSpecs(
        objective=ObjectiveType.SHARPE,
        constraints=constraints,
        method='qp'
    )
    
    # 优化
    print("\n1. Optimizing with Sharpe objective...")
    optimizer = QPOptimizer()
    weights = optimizer.optimize(knowledge, decisions)
    
    print(f"\n   Optimal weights:")
    for symbol, weight in zip(symbols, weights):
        print(f"     {symbol}: {weight:.4f}")
    
    # 计算组合指标
    portfolio_return = np.dot(weights, knowledge.mu)
    portfolio_risk = np.sqrt(np.dot(weights, knowledge.get_covariance() @ weights))
    sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    print(f"\n   Portfolio metrics:")
    print(f"     Expected return: {portfolio_return:.4f}")
    print(f"     Risk (std): {portfolio_risk:.4f}")
    print(f"     Sharpe ratio: {sharpe:.4f}")


def example_different_objectives():
    """不同优化目标对比"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comparing Different Objectives")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    class SimpleKnowledge:
        def __init__(self, returns):
            self.mu = returns.mean().values
            self.returns = returns
            self.B = None
            self.factor_names = None
        
        def get_covariance(self):
            return returns.cov().values
    
    knowledge = SimpleKnowledge(returns)
    
    # 构建约束
    builder = ConstraintBuilder()
    builder.long_only(True)
    builder.leverage(1.0)
    constraints = builder.build()
    
    # 测试不同目标
    objectives = [
        ObjectiveType.SHARPE,
        ObjectiveType.MIN_VARIANCE,
        ObjectiveType.MEAN_VARIANCE
    ]
    
    optimizer = QPOptimizer()
    
    print("\nComparing optimization results:")
    print(f"{'Objective':<20} {'Return':<12} {'Risk':<12} {'Sharpe':<12}")
    print("-" * 60)
    
    for obj_type in objectives:
        decisions = DecisionSpecs(
            objective=obj_type,
            constraints=constraints,
            method='qp',
            risk_aversion=1.0
        )
        
        weights = optimizer.optimize(knowledge, decisions)
        
        portfolio_return = np.dot(weights, knowledge.mu)
        portfolio_risk = np.sqrt(np.dot(weights, knowledge.get_covariance() @ weights))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        print(f"{obj_type.value:<20} {portfolio_return:>10.4f}  {portfolio_risk:>10.4f}  {sharpe:>10.4f}")


if __name__ == '__main__':
    try:
        example_objective_types()
        example_constraints()
        example_optimization()
        example_different_objectives()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

