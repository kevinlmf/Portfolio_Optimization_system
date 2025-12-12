"""
Regime Layer - 市场状态层

实现隐马尔可夫模型(HMM)驱动的市场状态识别与状态依赖参数估计

核心组件：
1. RegimeDetector - 市场状态检测器 (HMM)
2. RegimeKnowledgeBase - 状态依赖知识库
3. RegimeAwareOptimizer - 状态感知优化器

数学模型：
    s_t ~ Markov(P)                     (market regime)
    F_t | s_t ~ D_F(s_t)               (factor dynamics)
    r_t | F_t, s_t = B(s_t)F_t + ε_t(s_t)  (return model)
    ε_t(s_t) ~ N(0, Σ_ε(s_t))          (idiosyncratic risk)
"""

from .regime_detector import RegimeDetector, HMMRegimeDetector
from .regime_knowledge_base import RegimeKnowledgeBase, RegimeParameters
from .regime_optimizer import RegimeAwareOptimizer
from .regime_estimator import RegimeParameterEstimator

__all__ = [
    'RegimeDetector',
    'HMMRegimeDetector',
    'RegimeKnowledgeBase',
    'RegimeParameters',
    'RegimeAwareOptimizer',
    'RegimeParameterEstimator',
]

