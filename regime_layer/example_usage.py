"""
Regime Layer Example Usage - 市场状态层使用示例

展示如何使用HMM进行市场状态识别和状态感知优化

数学模型：
    s_t ~ Markov(P)                     (market regime)
    F_t | s_t ~ D_F(s_t)               (factor dynamics)
    r_t | F_t, s_t = B(s_t)F_t + ε_t(s_t)
    ε_t(s_t) ~ N(0, Σ_ε(s_t))
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from regime_layer import (
    HMMRegimeDetector,
    RegimeParameterEstimator,
    RegimeAwareOptimizer,
    RegimeKnowledgeBase
)


def generate_regime_data(n_samples: int = 500, 
                         n_assets: int = 10,
                         n_regimes: int = 2,
                         seed: int = 42) -> tuple:
    """
    生成具有regime结构的模拟数据
    
    模拟牛市和熊市两种状态
    """
    np.random.seed(seed)
    
    # 状态转移矩阵
    P = np.array([
        [0.95, 0.05],   # 牛市：95%停留，5%转熊市
        [0.10, 0.90]    # 熊市：10%转牛市，90%停留
    ])
    
    # 生成regime序列
    regimes = np.zeros(n_samples, dtype=int)
    regimes[0] = 0  # 从牛市开始
    
    for t in range(1, n_samples):
        regimes[t] = np.random.choice([0, 1], p=P[regimes[t-1]])
    
    # 各regime的参数
    # 牛市：高收益，低波动
    mu_bull = np.random.uniform(0.0005, 0.001, n_assets)  # 日均0.05%-0.1%
    sigma_bull = np.random.uniform(0.01, 0.015, n_assets)  # 日波动1%-1.5%
    
    # 熊市：低/负收益，高波动
    mu_bear = np.random.uniform(-0.001, 0.0002, n_assets)  # 日均-0.1%-0.02%
    sigma_bear = np.random.uniform(0.02, 0.03, n_assets)    # 日波动2%-3%
    
    # 生成收益率
    returns = np.zeros((n_samples, n_assets))
    
    for t in range(n_samples):
        if regimes[t] == 0:  # 牛市
            returns[t] = np.random.normal(mu_bull, sigma_bull)
        else:  # 熊市
            returns[t] = np.random.normal(mu_bear, sigma_bear)
    
    # 添加相关性（使用随机正定矩阵）
    # 生成随机矩阵并通过AA'构造正定矩阵
    A = np.random.randn(n_assets, n_assets) * 0.3
    corr_matrix = A @ A.T
    # 归一化为相关矩阵
    D = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D
    
    # Cholesky分解添加相关性
    try:
        L = np.linalg.cholesky(corr_matrix)
        returns = returns @ L.T
    except np.linalg.LinAlgError:
        # 如果Cholesky失败，使用特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        returns = returns @ L.T
    
    # 创建DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    return returns_df, regimes


def example_basic_regime_detection():
    """基础示例：市场状态检测"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC REGIME DETECTION")
    print("="*70)
    
    # 生成数据
    returns, true_regimes = generate_regime_data(n_samples=500, n_assets=10)
    
    print(f"\nData shape: {returns.shape}")
    print(f"True regime distribution: {np.bincount(true_regimes)}")
    
    # 创建HMM检测器
    detector = HMMRegimeDetector(n_regimes=2, n_iter=100)
    
    # 训练模型
    print("\nTraining HMM...")
    detector.fit(returns)
    
    # 检测状态
    regime_state = detector.detect(returns)
    
    print(f"\n✓ Current regime: {regime_state.current_regime}")
    print(f"✓ Regime probabilities: {regime_state.regime_probabilities}")
    print(f"✓ Detected regime distribution: {np.bincount(regime_state.regime_sequence)}")
    
    # 计算检测准确率
    accuracy = np.mean(regime_state.regime_sequence == true_regimes)
    # HMM可能会反转标签，所以也检查反转情况
    accuracy_flipped = np.mean(regime_state.regime_sequence == (1 - true_regimes))
    accuracy = max(accuracy, accuracy_flipped)
    print(f"\n✓ Detection accuracy: {accuracy:.2%}")
    
    # 打印转移矩阵
    print("\n✓ Estimated transition matrix:")
    print(regime_state.transition_matrix)
    
    # 获取统计信息
    stats = detector.get_regime_statistics()
    print("\n✓ Regime statistics:")
    for k in range(2):
        print(f"  Regime {k}:")
        print(f"    Mean return: {np.mean(stats[k]['mean'])*252:.2%} (annualized)")
        print(f"    Volatility: {np.mean(stats[k]['volatility'])*np.sqrt(252):.2%} (annualized)")
    
    return detector, regime_state


def example_regime_parameter_estimation():
    """示例：状态依赖参数估计"""
    print("\n" + "="*70)
    print("EXAMPLE 2: REGIME-DEPENDENT PARAMETER ESTIMATION")
    print("="*70)
    
    # 生成数据
    returns, _ = generate_regime_data(n_samples=500, n_assets=10)
    
    # 检测状态
    detector = HMMRegimeDetector(n_regimes=2)
    detector.fit(returns)
    regime_state = detector.detect(returns)
    
    # 估计参数
    estimator = RegimeParameterEstimator(
        use_factor_model=False,
        shrinkage=True
    )
    
    regime_knowledge = estimator.estimate(returns, regime_state)
    
    print("\n" + regime_knowledge.summary())
    
    # 比较各regime参数
    print("\n" + "-"*40)
    print("REGIME COMPARISON")
    print("-"*40)
    
    for k in range(2):
        params = regime_knowledge.regime_params[k]
        name = regime_knowledge.get_regime_name(k)
        print(f"\n{name}:")
        print(f"  Mean return: {np.mean(params.mu)*252:.4f} (annualized)")
        print(f"  Mean volatility: {np.mean(np.sqrt(np.diag(params.Sigma)))*np.sqrt(252):.4f} (annualized)")
        print(f"  Sharpe ratio: {np.mean(params.mu)/np.mean(np.sqrt(np.diag(params.Sigma)))*np.sqrt(252):.4f}")
    
    return regime_knowledge


def example_regime_aware_optimization():
    """示例：状态感知优化"""
    print("\n" + "="*70)
    print("EXAMPLE 3: REGIME-AWARE OPTIMIZATION")
    print("="*70)
    
    # 生成数据
    returns, _ = generate_regime_data(n_samples=500, n_assets=10)
    
    # 检测状态
    detector = HMMRegimeDetector(n_regimes=2)
    detector.fit(returns)
    regime_state = detector.detect(returns)
    
    # 估计参数
    estimator = RegimeParameterEstimator(use_factor_model=False)
    regime_knowledge = estimator.estimate(returns, regime_state)
    
    # 约束条件
    constraints = {
        'long_only': True,
        'leverage': 1.0,
        'max_weight': 0.3
    }
    
    # 测试不同策略
    strategies = ['expected', 'robust', 'adaptive', 'worst_case', 'multi_regime']
    results = {}
    
    print("\nOptimizing with different strategies...")
    
    for strategy in strategies:
        optimizer = RegimeAwareOptimizer(strategy=strategy)
        weights = optimizer.optimize(regime_knowledge, constraints, objective='sharpe')
        
        # 计算各regime下的表现
        contribution = optimizer.compute_regime_contribution(weights, regime_knowledge)
        
        results[strategy] = {
            'weights': weights,
            'expected_return': contribution['expected']['return'],
            'expected_risk': contribution['expected']['risk'],
            'expected_sharpe': contribution['expected']['sharpe']
        }
        
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Expected Return: {contribution['expected']['return']*252:.4f} (annualized)")
        print(f"  Expected Risk: {contribution['expected']['risk']*np.sqrt(252):.4f} (annualized)")
        print(f"  Expected Sharpe: {contribution['expected']['sharpe']*np.sqrt(252):.4f}")
        print(f"  Top 3 weights: {sorted(weights, reverse=True)[:3]}")
    
    # 比较策略
    print("\n" + "-"*40)
    print("STRATEGY COMPARISON")
    print("-"*40)
    
    print(f"\n{'Strategy':<15} {'Return':>10} {'Risk':>10} {'Sharpe':>10}")
    print("-" * 45)
    for strategy, res in results.items():
        print(f"{strategy:<15} {res['expected_return']*252:>10.4f} "
              f"{res['expected_risk']*np.sqrt(252):>10.4f} "
              f"{res['expected_sharpe']*np.sqrt(252):>10.4f}")
    
    return results


def example_regime_scenario_analysis():
    """示例：Regime场景分析"""
    print("\n" + "="*70)
    print("EXAMPLE 4: REGIME SCENARIO ANALYSIS")
    print("="*70)
    
    # 生成数据
    returns, _ = generate_regime_data(n_samples=500, n_assets=10)
    
    # 检测状态
    detector = HMMRegimeDetector(n_regimes=2)
    detector.fit(returns)
    regime_state = detector.detect(returns)
    
    # 估计参数
    estimator = RegimeParameterEstimator(use_factor_model=False)
    regime_knowledge = estimator.estimate(returns, regime_state)
    
    # 场景分析
    optimizer = RegimeAwareOptimizer(strategy='expected')
    scenario_results = optimizer.optimize_with_regime_scenarios(
        regime_knowledge,
        constraints={'long_only': True, 'leverage': 1.0},
        n_periods=12,
        rebalance_freq=1
    )
    
    print("\nRegime Evolution Forecast:")
    print("-" * 40)
    
    evolution = scenario_results['regime_evolution']
    for t in range(0, 12, 3):
        print(f"Period {t+1}: Bull={evolution[t, 0]:.2%}, Bear={evolution[t, 1]:.2%}")
    
    print(f"\nAverage Expected Return: {scenario_results['avg_return']*252:.4f} (annualized)")
    print(f"Average Expected Risk: {scenario_results['avg_risk']*np.sqrt(252):.4f} (annualized)")
    
    return scenario_results


def example_complete_workflow():
    """完整工作流示例"""
    print("\n" + "="*70)
    print("EXAMPLE 5: COMPLETE REGIME-AWARE WORKFLOW")
    print("="*70)
    
    # 1. 数据准备
    print("\n[1] Generating data...")
    returns, true_regimes = generate_regime_data(n_samples=500, n_assets=10)
    print(f"    Data shape: {returns.shape}")
    
    # 2. Regime检测
    print("\n[2] Detecting market regimes...")
    detector = HMMRegimeDetector(n_regimes=2, n_iter=100)
    detector.fit(returns)
    regime_state = detector.detect(returns)
    print(f"    Current regime: {regime_state.current_regime} "
          f"({regime_state.get_regime_name(regime_state.current_regime)})")
    print(f"    Regime probabilities: {regime_state.regime_probabilities}")
    
    # 3. 状态依赖参数估计
    print("\n[3] Estimating regime-dependent parameters...")
    estimator = RegimeParameterEstimator(use_factor_model=False, shrinkage=True)
    regime_knowledge = estimator.estimate(returns, regime_state)
    
    for k in range(2):
        params = regime_knowledge.regime_params[k]
        name = regime_knowledge.get_regime_name(k)
        ann_ret = np.mean(params.mu) * 252
        ann_vol = np.mean(np.sqrt(np.diag(params.Sigma))) * np.sqrt(252)
        print(f"    {name}: Return={ann_ret:.2%}, Vol={ann_vol:.2%}")
    
    # 4. 状态感知优化
    print("\n[4] Running regime-aware optimization...")
    optimizer = RegimeAwareOptimizer(strategy='robust')
    weights = optimizer.optimize(
        regime_knowledge,
        constraints={'long_only': True, 'leverage': 1.0, 'max_weight': 0.25},
        objective='sharpe'
    )
    
    print(f"    Optimal weights (top 5):")
    sorted_idx = np.argsort(weights)[::-1][:5]
    for idx in sorted_idx:
        print(f"      {regime_knowledge.asset_names[idx]}: {weights[idx]:.4f}")
    
    # 5. 风险分解
    print("\n[5] Decomposing risk by regime...")
    risk_decomp = regime_knowledge.decompose_risk_by_regime(weights)
    
    print(f"    Total risk: {risk_decomp['total']['risk']*np.sqrt(252):.4f} (annualized)")
    for k, info in risk_decomp['by_regime'].items():
        print(f"    {info['name']}: Risk={info['risk']*np.sqrt(252):.4f}, "
              f"Sharpe={info['sharpe']*np.sqrt(252):.2f}, "
              f"P={info['probability']:.2%}")
    
    # 6. 转换为标准KnowledgeBase（向后兼容）
    print("\n[6] Converting to standard KnowledgeBase...")
    simple_kb = regime_knowledge.to_simple_knowledge_base(method='expected')
    print(f"    mu shape: {simple_kb.mu.shape}")
    print(f"    Sigma shape: {simple_kb.Sigma.shape}")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    
    return {
        'detector': detector,
        'regime_state': regime_state,
        'regime_knowledge': regime_knowledge,
        'weights': weights,
        'risk_decomposition': risk_decomp
    }


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# REGIME LAYER EXAMPLES")
    print("# 市场状态层使用示例")
    print("#"*70)
    
    # 运行所有示例
    example_basic_regime_detection()
    example_regime_parameter_estimation()
    example_regime_aware_optimization()
    example_regime_scenario_analysis()
    result = example_complete_workflow()
    
    print("\n\nAll examples completed successfully!")

