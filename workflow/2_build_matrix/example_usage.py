"""
Build Matrix Usage Example

构建矩阵使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from workflow import FactorLoadingsEstimator, CorrelationMatrixBuilder, FactorRiskModel
from data import APIClient


def example_factor_loadings():
    """因子载荷估计示例"""
    print("=" * 80)
    print("EXAMPLE 1: Factor Loadings Estimation")
    print("=" * 80)
    
    # 1. 获取数据
    print("\n1. Fetching data...")
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   Returns shape: {returns.shape}")
    
    # 2. 获取因子数据（使用市场因子）
    factor_symbols = ['SPY']  # 市场因子
    factors = client.fetch_factors(
        factor_names=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   Factors shape: {factors.shape}")
    
    # 3. 估计因子载荷
    print("\n2. Estimating factor loadings...")
    estimator = FactorLoadingsEstimator(method='ols')
    B = estimator.estimate(returns, factors)
    print(f"   Factor loadings shape: {B.shape}")
    print(f"\n   Factor Loadings Matrix B:")
    print(B)
    
    # 4. 估计带 alpha 的模型
    print("\n3. Estimating with alpha...")
    B_alpha, alpha = estimator.estimate_with_alpha(returns, factors)
    print(f"   Alpha (intercept):")
    print(alpha)


def example_correlation_matrix():
    """相关性矩阵构建示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Correlation Matrix")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 获取多个因子
    factor_symbols = ['SPY', 'QQQ', 'DIA']  # 市场、科技、道指
    factors = client.fetch_factors(
        factor_names=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 构建相关性矩阵
    print("\n1. Building correlation matrix (Pearson)...")
    builder = CorrelationMatrixBuilder(method='pearson')
    corr_matrix = builder.build(returns, factors)
    print(f"   Correlation matrix shape: {corr_matrix.shape}")
    print(f"\n   Stock-Factor Correlation Matrix:")
    print(corr_matrix)
    
    # 分析相关性强度
    print("\n2. Analyzing correlation strength...")
    analysis = builder.analyze_correlation_strength()
    print(analysis)
    
    # 构建完整相关性矩阵
    print("\n3. Building full correlation matrix...")
    full_corr = builder.build_full_correlation(returns, factors)
    print(f"   Full correlation matrix shape: {full_corr.shape}")
    print(f"\n   First few rows/columns:")
    print(full_corr.iloc[:5, :5])


def example_factor_risk_model():
    """因子风险模型示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Factor Risk Model")
    print("=" * 80)
    
    # 获取数据
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 获取因子
    factor_symbols = ['SPY', 'QQQ']
    factors = client.fetch_factors(
        factor_names=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 构建因子风险模型
    print("\n1. Building factor risk model...")
    model = FactorRiskModel()
    F, D, Sigma = model.build(returns, factors)
    
    print(f"   Factor covariance F shape: {F.shape}")
    print(f"   Idiosyncratic risk D shape: {D.shape}")
    print(f"   Full covariance Σ shape: {Sigma.shape}")
    
    print(f"\n   Factor Covariance Matrix F:")
    print(pd.DataFrame(F, index=factors.columns, columns=factors.columns))
    
    print(f"\n   Idiosyncratic Risk (diagonal):")
    print(pd.Series(np.diag(D), index=returns.columns))
    
    # 风险分解
    print("\n2. Risk decomposition for equal-weight portfolio...")
    weights = np.ones(len(symbols)) / len(symbols)
    risk_decomp = model.decompose_risk(weights)
    
    print(f"   Total Risk: {risk_decomp['total_risk']:.4f}")
    print(f"   Factor Risk: {risk_decomp['factor_risk']:.4f} ({risk_decomp['factor_risk_pct']:.2f}%)")
    print(f"   Idiosyncratic Risk: {risk_decomp['idiosyncratic_risk']:.4f} ({risk_decomp['idiosyncratic_risk_pct']:.2f}%)")


def example_complete_workflow():
    """完整工作流程示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete Build Matrix Workflow")
    print("=" * 80)
    
    # Step 1: 获取数据
    print("\nStep 1: Fetching data...")
    client = APIClient(source='yahoo')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   ✓ Fetched {len(symbols)} assets")
    
    # Step 2: 获取因子
    factor_symbols = ['SPY', 'QQQ']
    factors = client.fetch_factors(
        factor_names=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"   ✓ Fetched {len(factor_symbols)} factors")
    
    # Step 3: 构建相关性矩阵
    print("\nStep 2: Building correlation matrix...")
    corr_builder = CorrelationMatrixBuilder(method='pearson')
    corr_matrix = corr_builder.build(returns, factors)
    print(f"   ✓ Correlation matrix shape: {corr_matrix.shape}")
    
    # Step 4: 估计因子载荷
    print("\nStep 3: Estimating factor loadings...")
    loader = FactorLoadingsEstimator(method='ols')
    B = loader.estimate(returns, factors)
    print(f"   ✓ Factor loadings shape: {B.shape}")
    print(f"\n   Factor Loadings:")
    print(B)
    
    # Step 5: 构建风险模型
    print("\nStep 4: Building factor risk model...")
    risk_model = FactorRiskModel()
    F, D, Sigma = risk_model.build(returns, factors, B)
    print(f"   ✓ Risk model built")
    print(f"   ✓ Full covariance matrix shape: {Sigma.shape}")
    
    return B, corr_matrix, Sigma


if __name__ == '__main__':
    try:
        example_factor_loadings()
        example_correlation_matrix()
        example_factor_risk_model()
        example_complete_workflow()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



