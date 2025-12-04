# Data Module

数据获取和 API 接口模块

## 功能

- **统一 API 接口**: 支持多种数据源（Yahoo Finance, Alpha Vantage 等）
- **数据获取**: 获取股票收益率、因子数据、市场数据
- **数据验证**: 自动验证数据质量

## 使用方法

### 基本用法

```python
from data import APIClient

# 初始化客户端
client = APIClient(source='yahoo')

# 获取股票收益率
returns = client.fetch_returns(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    frequency='daily'
)

# 获取因子数据
factors = client.fetch_factors(
    factor_names=['SPY', 'VIX', 'DXY'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### 数据验证

```python
# 验证数据质量
is_valid = client.validate_data(returns)
```

## 支持的数据源

1. **Yahoo Finance** (默认)
   - 免费，无需 API key
   - 支持股票、ETF、指数

2. **Alpha Vantage** (需要 API key)
   - 提供更多技术指标
   - 需要注册获取 API key

3. **自定义数据源**
   - 可以扩展支持其他数据源

## 文件结构

```
data/
├── __init__.py              # 模块初始化
├── api_client.py            # API 客户端
├── multi_asset_fetcher.py  # 多资产数据获取器
└── README.md               # 说明文档
```



