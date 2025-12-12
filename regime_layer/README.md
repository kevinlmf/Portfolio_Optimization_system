# Regime Layer - 市场状态层

基于隐马尔可夫模型(HMM)的市场状态识别与状态感知投资组合优化。

## 统一生成模型 (Unified Generative Model)

金融世界可以被写成：

$$
\begin{aligned}
s_t &\sim \text{Markov}(P) & \text{(market regime)} \\
F_t | s_t &\sim \mathcal{D}_F(s_t) & \text{(factor dynamics)} \\
r_t | F_t, s_t &= B(s_t)F_t + \varepsilon_t(s_t) & \text{(return model)} \\
\varepsilon_t(s_t) &\sim \mathcal{N}(0, \Sigma_\varepsilon(s_t)) & \text{(idiosyncratic risk)}
\end{aligned}
$$

### 层级结构

| 层级 | 对应对象 | 说明 |
|------|----------|------|
| 隐状态 | HMM / Regime | 市场状态（牛市/熊市等） |
| 中间结构 | Factor | 驱动收益的公共因子 |
| 依赖结构 | Correlation / Covariance | 资产间的依赖关系 |
| 决策 | Goal + Optimization | 优化目标和约束 |

## 核心组件

### 1. RegimeDetector - 市场状态检测器

使用隐马尔可夫模型(HMM)识别市场状态。

```python
from regime_layer import HMMRegimeDetector

# 创建检测器（2个状态：牛市/熊市）
detector = HMMRegimeDetector(n_regimes=2, n_iter=100)

# 训练
detector.fit(returns)

# 检测当前状态
regime_state = detector.detect(returns)

print(f"当前状态: {regime_state.current_regime}")
print(f"状态概率: {regime_state.regime_probabilities}")
print(f"转移矩阵: {regime_state.transition_matrix}")
```

### 2. RegimeParameterEstimator - 状态依赖参数估计器

为每个regime估计独立的参数：μ(s), B(s), F(s), D(s), Σ(s)

```python
from regime_layer import RegimeParameterEstimator

# 创建估计器
estimator = RegimeParameterEstimator(
    use_factor_model=True,  # 使用因子模型
    shrinkage=True          # 使用收缩估计
)

# 估计各regime参数
regime_knowledge = estimator.estimate(returns, regime_state, factors)

# 查看各regime的参数
for k in range(regime_knowledge.n_regimes):
    params = regime_knowledge.regime_params[k]
    print(f"Regime {k}: μ={params.mu.mean():.4f}, σ={np.sqrt(np.diag(params.Sigma)).mean():.4f}")
```

### 3. RegimeAwareOptimizer - 状态感知优化器

提供多种考虑market regime的优化策略。

```python
from regime_layer import RegimeAwareOptimizer

# 创建优化器
optimizer = RegimeAwareOptimizer(strategy='robust')

# 优化
weights = optimizer.optimize(
    regime_knowledge,
    constraints={'long_only': True, 'leverage': 1.0},
    objective='sharpe'
)
```

#### 优化策略

| 策略 | 说明 |
|------|------|
| `expected` | 使用概率加权的期望参数 |
| `robust` | Minimax优化，考虑最坏情况 |
| `adaptive` | 根据当前regime动态调整权重 |
| `worst_case` | 只考虑最差regime的参数 |
| `multi_regime` | 多regime联合优化 |

### 4. RegimeKnowledgeBase - 状态依赖知识库

存储所有regime的参数和转移概率。

```python
# 获取当前regime参数
current_params = regime_knowledge.get_current_parameters()

# 获取期望参数（概率加权）
expected_params = regime_knowledge.get_expected_parameters()

# 获取稳健参数（保守估计）
robust_params = regime_knowledge.get_robust_parameters()

# 按regime分解风险
risk_decomp = regime_knowledge.decompose_risk_by_regime(weights)

# 预测regime演化
evolution = regime_knowledge.predict_regime_evolution(n_steps=12)

# 转换为标准KnowledgeBase（向后兼容）
simple_kb = regime_knowledge.to_simple_knowledge_base(method='expected')
```

## 完整工作流示例

```python
import pandas as pd
from workflow import PortfolioWorkflow, ObjectiveType

# 加载数据
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# 创建工作流（启用regime模式）
workflow = PortfolioWorkflow(returns, use_regime=True)

# 运行完整工作流
weights = workflow.run_complete_workflow(
    objective=ObjectiveType.SHARPE,
    constraints={'long_only': True, 'max_weight': 0.2},
    n_factors=5,
    n_regimes=2,
    regime_strategy='robust'
)

# 或者分步执行
workflow.step0_regime_detection(n_regimes=2)
workflow.step1_factor_mining(top_n=5)
workflow.step2_build_matrix()
decisions = workflow.step3_select_objective(ObjectiveType.SHARPE)
workflow.step4_estimate_parameters(workflow.factors)
weights = workflow.step5_evaluation(decisions, regime_strategy='robust')
```

## 数学细节

### HMM参数

- **状态数 K**: 通常2-4个（牛市、熊市、震荡、危机）
- **转移矩阵 P**: K×K矩阵，P[i,j] = P(s_{t+1}=j | s_t=i)
- **初始分布 π**: 初始状态概率

### 参数估计

对于每个regime s，估计：
- **μ(s)**: 预期收益率向量
- **Σ(s)**: 协方差矩阵
- **B(s)**: 因子载荷矩阵（可选）
- **F(s)**: 因子协方差（可选）
- **D(s)**: 特质风险（可选）

### 优化目标

**期望优化**:
$$\max_w \sum_s P(s) \cdot u(w; \mu(s), \Sigma(s))$$

**稳健优化**:
$$\max_w \min_s u(w; \mu(s), \Sigma(s))$$

其中 u 是效用函数（如Sharpe比率）。

## 文件结构

```
regime_layer/
├── __init__.py              # 模块导出
├── regime_detector.py       # HMM状态检测器
├── regime_knowledge_base.py # 状态依赖知识库
├── regime_estimator.py      # 状态依赖参数估计器
├── regime_optimizer.py      # 状态感知优化器
├── example_usage.py         # 使用示例
└── README.md               # 本文档
```

## 运行示例

```bash
cd Portfolio_Optimization_system
python -m regime_layer.example_usage
```

## 依赖

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## 参考文献

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
- Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates"
- Guidolin, M. & Timmermann, A. (2007). "Asset Allocation Under Multivariate Regime Switching"

