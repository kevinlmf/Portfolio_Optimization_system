"""
Data Module Usage Example

数据模块使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import APIClient, MultiAssetFetcher
import pandas as pd


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("EXAMPLE 1: Basic API Usage")
    print("=" * 80)
    
    # 创建 API 客户端
    client = APIClient(source='yahoo')
    
    # 获取股票收益率
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    returns = client.fetch_returns(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31',
        frequency='daily'
    )
    
    print(f"\nReturns shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(returns.head())
    
    # 验证数据
    is_valid = client.validate_data(returns)
    print(f"\nData validation: {'✓ Valid' if is_valid else '✗ Invalid'}")


def example_multi_asset_fetcher():
    """多资产获取器示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Asset Fetcher")
    print("=" * 80)
    
    # 创建多资产获取器
    fetcher = MultiAssetFetcher()
    
    # 获取投资组合数据
    portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL']
    returns = fetcher.fetch_portfolio_data(
        symbols=portfolio_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    print(f"\nPortfolio returns shape: {returns.shape}")
    
    # 获取因子数据
    factor_symbols = ['SPY', 'VIX']  # 市场因子和波动率因子
    factors = fetcher.fetch_factor_data(
        factor_symbols=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    print(f"Factor data shape: {factors.shape}")
    
    # 对齐数据
    aligned_returns, aligned_factors = fetcher.align_data(returns, factors)
    print(f"\nAligned returns shape: {aligned_returns.shape}")
    print(f"Aligned factors shape: {aligned_factors.shape if aligned_factors is not None else 'N/A'}")


def example_factor_data():
    """因子数据示例"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Factor Data")
    print("=" * 80)
    
    client = APIClient(source='yahoo')
    
    # 获取常见因子
    factor_symbols = ['SPY', 'VIX', 'DXY', 'GLD']  # 市场、波动率、美元、黄金
    factors = client.fetch_factors(
        factor_names=factor_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    print(f"\nFactor data shape: {factors.shape}")
    print(f"\nFactor statistics:")
    print(factors.describe())


if __name__ == '__main__':
    # Run examples
    try:
        example_basic_usage()
        example_multi_asset_fetcher()
        example_factor_data()
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

