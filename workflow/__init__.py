"""
Complete Portfolio Optimization Workflow

按照实际工作流程：
1. Factor Mining (挖掘因子)
2. Build Matrix (构建股票-因子关系矩阵 B)
3. Select Objective (选择优化目标)
4. Parameter Estimation (参数估计: μ, F, D, Σ)
5. Evaluation (评估/回测)
"""

# Use importlib to avoid circular imports with numeric module names
import importlib

# Step 1: Factor Mining
_step1 = importlib.import_module('.1_factor_mining', __package__)
FactorMiner = _step1.FactorMiner
FactorSelector = _step1.FactorSelector
FactorAnalyzer = _step1.FactorAnalyzer

# Step 2: Build Matrix
_step2 = importlib.import_module('.2_build_matrix', __package__)
FactorLoadingsEstimator = _step2.FactorLoadingsEstimator
CorrelationMatrixBuilder = _step2.CorrelationMatrixBuilder
FactorRiskModel = _step2.FactorRiskModel

# Step 3: Select Objective
_step3 = importlib.import_module('.3_select_objective', __package__)
ObjectiveType = _step3.ObjectiveType
ObjectiveFunction = _step3.ObjectiveFunction
Constraints = _step3.Constraints
ConstraintBuilder = _step3.ConstraintBuilder
DecisionSpecs = _step3.DecisionSpecs
QPOptimizer = _step3.QPOptimizer
SparseSharpeOptimizer = _step3.SparseSharpeOptimizer

# Step 4: Parameter Estimation
_step4 = importlib.import_module('.4_estimate_parameters', __package__)
ExpectedReturnsEstimator = _step4.ExpectedReturnsEstimator
SampleMeanEstimator = _step4.SampleMeanEstimator
RiskStructureEstimator = _step4.RiskStructureEstimator
DependencyStructureEstimator = _step4.DependencyStructureEstimator
KnowledgeBase = _step4.KnowledgeBase
ParameterEstimator = _step4.ParameterEstimator
SampleEstimator = _step4.SampleEstimator

# Step 5: Evaluation
_step5 = importlib.import_module('.5_evaluation', __package__)
PortfolioEvaluator = _step5.PortfolioEvaluator
Backtester = _step5.Backtester
PerformanceMetrics = _step5.PerformanceMetrics

# Step 6: Options Hedging
_step6 = importlib.import_module('.6_options_hedging', __package__)
VolatilityForecaster = _step6.VolatilityForecaster
BlackScholesPricer = _step6.BlackScholesPricer
GreeksCalculator = _step6.GreeksCalculator
DeltaHedgingStrategy = _step6.DeltaHedgingStrategy
GreeksHedgingStrategy = _step6.GreeksHedgingStrategy

# Step 0: Regime Detection (Optional)
try:
    from regime_layer import (
        HMMRegimeDetector,
        RegimeParameterEstimator,
        RegimeAwareOptimizer,
        RegimeKnowledgeBase,
        RegimeParameters
    )
    REGIME_LAYER_AVAILABLE = True
except ImportError:
    HMMRegimeDetector = None
    RegimeParameterEstimator = None
    RegimeAwareOptimizer = None
    RegimeKnowledgeBase = None
    RegimeParameters = None
    REGIME_LAYER_AVAILABLE = False

__all__ = [
    # Step 1: Factor Mining
    'FactorMiner',
    'FactorSelector',
    'FactorAnalyzer',
    # Step 2: Build Matrix
    'FactorLoadingsEstimator',
    'CorrelationMatrixBuilder',
    'FactorRiskModel',
    # Step 3: Select Objective
    'ObjectiveType',
    'ObjectiveFunction',
    'Constraints',
    'ConstraintBuilder',
    'DecisionSpecs',
    'QPOptimizer',
    'SparseSharpeOptimizer',
    # Step 4: Parameter Estimation
    'ExpectedReturnsEstimator',
    'SampleMeanEstimator',
    'RiskStructureEstimator',
    'DependencyStructureEstimator',
    'KnowledgeBase',
    'ParameterEstimator',
    'SampleEstimator',
    # Step 5: Evaluation
    'PortfolioEvaluator',
    'Backtester',
    'PerformanceMetrics',
    # Step 6: Options Hedging
    'VolatilityForecaster',
    'BlackScholesPricer',
    'GreeksCalculator',
    'DeltaHedgingStrategy',
    'GreeksHedgingStrategy',
    # Step 0: Regime Detection (Optional)
    'HMMRegimeDetector',
    'RegimeParameterEstimator',
    'RegimeAwareOptimizer',
    'RegimeKnowledgeBase',
    'RegimeParameters',
    'REGIME_LAYER_AVAILABLE',
]

# Import PortfolioWorkflow from workflow.py (avoiding circular import)
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import PortfolioWorkflow from workflow.py file
import importlib.util
_workflow_file = os.path.join(_project_root, 'workflow.py')
if os.path.exists(_workflow_file):
    spec = importlib.util.spec_from_file_location("workflow_module", _workflow_file)
    workflow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_module)
    PortfolioWorkflow = workflow_module.PortfolioWorkflow
    __all__.append('PortfolioWorkflow')
else:
    PortfolioWorkflow = None
