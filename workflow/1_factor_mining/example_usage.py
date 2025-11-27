"""
Factor Mining Usage Example

因子挖掘使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from workflow import FactorMiner, FactorSelector, FactorAnalyzer
from data import APIClient


def example_factor_mining():
    """因子挖掘示例"""
    print("=" * 80)
    print("EXAMPLE 1: Factor Mining")
    print("=" * 80)
    
    # 1. 获取股票数据
    print("\n1. Fetching stock data...")
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'MA']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   Returns shape: {returns.shape}")
    
    # 2. 挖掘因子
    print("\n2. Mining factors using PCA...")
    miner = FactorMiner(method='pca')
    factors = miner.mine_factors(returns, n_factors=5)
    print(f"   Factors shape: {factors.shape}")
    print(f"   Explained variance: {miner.get_explained_variance()}")
    
    # 3. 显示因子摘要
    print("\n3. Factor summary:")
    summary = miner.summarize()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    return factors, returns


def example_factor_selection():
    """因子选择示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Factor Selection")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 挖掘因子
    miner = FactorMiner(method='pca')
    factors = miner.mine_factors(returns, n_factors=10)
    
    # 选择因子
    print("\n1. Selecting factors by IC...")
    selector = FactorSelector()
    selected_ic = selector.select_by_ic(factors, returns, top_n=5)
    print(f"   Selected factors (IC): {selected_ic}")
    
    print("\n2. Selecting factors by LASSO...")
    selected_lasso = selector.select_by_lasso(factors, returns, top_n=5)
    print(f"   Selected factors (LASSO): {selected_lasso}")
    
    print("\n3. Comprehensive selection...")
    selected_comprehensive = selector.comprehensive_selection(factors, returns, top_n=5)
    print(f"   Selected factors (Comprehensive): {selected_comprehensive}")
    
    # 显示重要性分数
    print("\n4. Importance scores:")
    importance = selector.get_importance_scores()
    for method, scores in importance.items():
        print(f"   {method}: {list(scores.keys())[:3]}...")


def example_factor_analysis():
    """因子分析示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Factor Analysis")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 挖掘因子
    miner = FactorMiner(method='pca')
    factors = miner.mine_factors(returns, n_factors=5)
    
    # 分析因子
    print("\n1. Statistical analysis:")
    analyzer = FactorAnalyzer()
    stats = analyzer.analyze_statistics(factors)
    print(stats)
    
    print("\n2. Correlation analysis:")
    correlation = analyzer.analyze_correlation(factors)
    print(correlation)
    
    print("\n3. Stability analysis:")
    stability = analyzer.analyze_stability(factors)
    print(stability)
    
    print("\n4. Comprehensive analysis:")
    comprehensive = analyzer.comprehensive_analysis(factors)
    print(f"   Statistics shape: {comprehensive['statistics'].shape}")
    print(f"   Correlation shape: {comprehensive['correlation'].shape}")
    print(f"   Stability shape: {comprehensive['stability'].shape}")


def example_complete_workflow():
    """完整工作流程示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete Factor Mining Workflow")
    print("=" * 80)
    
    # Step 1: 获取数据
    print("\nStep 1: Fetching data...")
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'DIS', 'NVDA']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   ✓ Fetched {len(symbols)} assets, {len(returns)} observations")
    
    # Step 2: 挖掘因子
    print("\nStep 2: Mining factors...")
    miner = FactorMiner(method='pca')
    factors = miner.mine_factors(returns, n_factors=10)
    print(f"   ✓ Extracted {len(factors.columns)} factors")
    print(f"   ✓ Explained variance: {np.sum(miner.get_explained_variance()):.2%}")
    
    # Step 3: 选择因子
    print("\nStep 3: Selecting factors...")
    selector = FactorSelector()
    selected_factors = selector.comprehensive_selection(factors, returns, top_n=5)
    print(f"   ✓ Selected {len(selected_factors)} factors: {selected_factors}")
    
    # Step 4: 分析因子
    print("\nStep 4: Analyzing factors...")
    analyzer = FactorAnalyzer()
    analysis = analyzer.comprehensive_analysis(factors[selected_factors])
    print(f"   ✓ Analysis complete")
    print(f"\n   Factor Statistics:")
    print(analysis['statistics'])
    
    return factors[selected_factors], returns


if __name__ == '__main__':
    try:
        example_factor_mining()
        example_factor_selection()
        example_factor_analysis()
        example_complete_workflow()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


