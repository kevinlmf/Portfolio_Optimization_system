"""
Evaluation Usage Example

评估使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from workflow import PortfolioEvaluator, Backtester, PerformanceMetrics
from workflow import SampleEstimator, KnowledgeBase
from workflow import ObjectiveType, ConstraintBuilder, DecisionSpecs, QPOptimizer
from data import APIClient


def example_portfolio_evaluation():
    """投资组合评估示例"""
    print("=" * 80)
    print("EXAMPLE 1: Portfolio Evaluation")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 估计参数
    estimator = SampleEstimator()
    knowledge = estimator.estimate(returns)
    
    # 优化
    builder = ConstraintBuilder()
    builder.long_only(True)
    builder.leverage(1.0)
    constraints = builder.build()
    
    decisions = DecisionSpecs(
        objective=ObjectiveType.SHARPE,
        constraints=constraints,
        method='qp'
    )
    
    optimizer = QPOptimizer()
    weights = optimizer.optimize(knowledge, decisions)
    
    # 评估
    print("\n1. Evaluating portfolio...")
    evaluator = PortfolioEvaluator()
    metrics = evaluator.evaluate(weights, returns, knowledge)
    
    print("\n   Portfolio Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")


def example_performance_metrics():
    """表现指标示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Performance Metrics")
    print("=" * 80)
    
    # 创建示例收益率
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    
    # 计算指标
    print("\n1. Calculating performance metrics...")
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all(returns)
    
    print("\n   Performance Metrics:")
    for key, value in metrics.items():
        print(f"     {key}: {value:.4f}")


def example_backtesting():
    """回测示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Backtesting")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 定义权重生成函数（等权重策略）
    def equal_weight_strategy(historical_returns):
        n = len(historical_returns.columns)
        return np.ones(n) / n
    
    # 回测
    print("\n1. Running backtest...")
    backtester = Backtester(rebalance_frequency='monthly', transaction_cost=0.001)
    backtest_results = backtester.backtest(returns, equal_weight_strategy)
    
    print(f"   Backtest results shape: {backtest_results.shape}")
    print(f"\n   First few rows:")
    print(backtest_results.head())
    
    # 评估回测结果
    print("\n2. Evaluating backtest results...")
    backtest_metrics = backtester.evaluate_backtest(backtest_results)
    
    print("\n   Backtest Metrics:")
    for key, value in backtest_metrics.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")


def example_complete_evaluation():
    """完整评估流程示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete Evaluation Workflow")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Step 1: 估计参数
    print("\nStep 1: Estimating parameters...")
    estimator = SampleEstimator()
    knowledge = estimator.estimate(returns)
    print(f"   ✓ Estimated μ shape: {knowledge.mu.shape}")
    print(f"   ✓ Estimated Σ shape: {knowledge.Sigma.shape}")
    
    # Step 2: 优化
    print("\nStep 2: Optimizing portfolio...")
    builder = ConstraintBuilder()
    builder.long_only(True)
    builder.leverage(1.0)
    constraints = builder.build()
    
    decisions = DecisionSpecs(
        objective=ObjectiveType.SHARPE,
        constraints=constraints,
        method='qp'
    )
    
    optimizer = QPOptimizer()
    weights = optimizer.optimize(knowledge, decisions)
    print(f"   ✓ Optimized weights calculated")
    
    # Step 3: 评估
    print("\nStep 3: Evaluating portfolio...")
    evaluator = PortfolioEvaluator()
    metrics = evaluator.evaluate(weights, returns, knowledge)
    
    print(f"\n   Key Metrics:")
    print(f"     Annualized Return: {metrics.get('annualized_return', 0):.4f}")
    print(f"     Annualized Volatility: {metrics.get('annualized_volatility', 0):.4f}")
    print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"     Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
    print(f"     Total Risk: {metrics.get('total_risk', 0):.4f}")
    
    if 'factor_risk_pct' in metrics:
        print(f"     Factor Risk: {metrics.get('factor_risk_pct', 0):.2f}%")
    if 'idiosyncratic_risk_pct' in metrics:
        print(f"     Idiosyncratic Risk: {metrics.get('idiosyncratic_risk_pct', 0):.2f}%")


if __name__ == '__main__':
    try:
        example_portfolio_evaluation()
        example_performance_metrics()
        example_backtesting()
        example_complete_evaluation()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


