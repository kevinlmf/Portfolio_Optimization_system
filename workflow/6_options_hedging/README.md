# Options Hedging Module - 期权对冲模块

通过资产组合构建过程预测波动率，进行对冲，分析希腊字母。

## 核心功能

### 1. 波动率预测 (Volatility Forecast)

从投资组合的风险结构（协方差矩阵 Σ）预测未来波动率。

**关键思想**：
- 投资组合的波动率 = √(w^T Σ w)，其中 w 是组合权重，Σ 是协方差矩阵
- 可以从组合优化的过程中直接提取波动率预测

**方法**：
- `portfolio_risk`: 基于组合风险结构
- `garch`: GARCH模型
- `realized`: 已实现波动率（滚动窗口）
- `implied`: 隐含波动率（从期权价格反推）

### 2. 期权定价 (Option Pricing)

实现 Black-Scholes 和二叉树模型。

**Black-Scholes 模型**：
- 欧式期权定价
- 支持 Call 和 Put
- 支持股息率

### 3. 希腊字母计算 (Greeks Calculator)

计算期权价格对各种风险因素的敏感度：

- **Δ (Delta)**: 对价格敏感度，控制方向性敞口
  - Call: 0 到 1
  - Put: -1 到 0
  - 组合Delta = 0 → 无方向性风险

- **Γ (Gamma)**: 对Δ的变化率，管理凸性（非线性）
  - 衡量Delta的稳定性
  - 高Gamma → Delta变化快，需要频繁对冲

- **Θ (Theta)**: 时间衰减，捕获时间价值
  - 通常为负（时间流逝导致价值损失）
  - 衡量时间价值衰减速度

- **V (Vega)**: 对波动率敏感度，赌波动率高低
  - 衡量波动率变动对期权价格的影响
  - 波动率交易的核心指标

- **ρ (Rho)**: 对利率敏感度，利率敞口管理
  - Call通常为正，Put通常为负
  - 长期期权对利率更敏感

### 4. 对冲策略 (Hedging Strategy)

基于希腊字母进行风险对冲：

- **Delta对冲**: 通过买卖标的资产使组合Delta = 0
- **Gamma对冲**: 使用其他期权管理Gamma风险
- **多希腊字母对冲**: 同时对冲Delta、Gamma、Vega等

## 使用示例

### 从组合预测波动率并计算对冲

```python
from workflow import PortfolioWorkflow, ObjectiveType
from data import APIClient

# 获取数据
client = APIClient(source='yahoo')
returns = client.fetch_returns(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 运行完整工作流（Step 1-5）
workflow = PortfolioWorkflow(returns)
weights = workflow.run_complete_workflow(
    objective=ObjectiveType.SHARPE,
    constraints={'long_only': True}
)

# Step 6: 期权对冲
hedge_result = workflow.step6_options_hedging(
    spot_price=100.0,
    strike=105.0,
    time_to_expiry=0.25,  # 3个月
    risk_free_rate=0.05,
    option_type='call',
    hedge_method='delta',
    volatility_method='portfolio_risk'  # 从组合风险结构预测
)

print(f"Forecasted volatility: {hedge_result['forecasted_volatility']*100:.2f}%")
print(f"Delta: {hedge_result['greeks']['delta']:.4f}")
print(f"Hedge quantity: {hedge_result['hedge_solution']['hedge_quantity']:.2f} shares")
```

### 独立使用各模块

```python
from workflow import (
    VolatilityForecaster,
    BlackScholesPricer,
    GreeksCalculator,
    DeltaHedgingStrategy
)

# 1. 预测波动率
forecaster = VolatilityForecaster(method='portfolio_risk')
vol = forecaster.forecast_from_portfolio(
    portfolio_weights=weights,
    covariance_matrix=cov_matrix,
    horizon=21
)

# 2. 定价期权
pricer = BlackScholesPricer()
option_price = pricer.price(
    spot=100, strike=105,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    volatility=vol,
    option_type='call'
)

# 3. 计算希腊字母
greeks_calc = GreeksCalculator()
greeks = greeks_calc.calculate_all(
    spot=100, strike=105,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    volatility=vol,
    option_type='call'
)

# 4. 对冲
hedging = DeltaHedgingStrategy()
hedge = hedging.calculate_hedge(
    portfolio_delta=greeks['delta'] * 10,  # 10份期权
    spot_price=100,
    contract_size=100
)
```

## 关键思想

**期权的本质是对冲风险**：
- 通过组合优化构建资产组合 → 得到风险结构（Σ）
- 从风险结构预测波动率 → 为期权定价
- 计算希腊字母 → 量化各种风险敞口
- 设计对冲策略 → 消除或管理风险

**Delta对冲的核心**：
- 如果组合有正的Delta（价格上升时获利），卖出标的资产对冲
- 如果组合有负的Delta（价格下降时获利），买入标的资产对冲
- 目标是使组合对价格变动不敏感

## 未来扩展

- [ ] 支持更多期权类型（美式、亚式、障碍等）
- [ ] 波动率曲面建模
- [ ] 动态对冲（频繁调整）
- [ ] 交易成本考虑
- [ ] 多资产组合的希腊字母聚合
- [ ] 压力测试和场景分析

