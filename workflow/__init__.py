"""
Complete Portfolio Optimization Workflow

New Workflow (按照实际投资决策流程):
0. Regime Detection (市场状态检测) - 决定后续一切
1. Factor Mining (挖掘因子) - 根据市场状态挖掘有效因子
2. Stock Selection (因子选股) - 根据因子暴露选股
3. Forecasting (收益与风险预测) - 预测选中股票的收益和协方差
4. Select Objective (选择优化目标) - 根据预测结果选择目标
5. Optimization (组合优化) - 求解最优权重
6. Options Hedging (期权对冲) - 对冲风险
"""

# Use importlib to avoid circular imports with numeric module names
import importlib

# Step 1: Factor Mining
_step1 = importlib.import_module('.1_factor_mining', __package__)
FactorMiner = _step1.FactorMiner
FactorSelector = _step1.FactorSelector
FactorAnalyzer = _step1.FactorAnalyzer

# Step 2: Stock Selection (NEW)
_step2_selection = importlib.import_module('.2_stock_selection', __package__)
StockSelector = _step2_selection.StockSelector
SelectionResult = _step2_selection.SelectionResult

# Step 2 (old): Build Matrix - kept for backward compatibility
_step2_matrix = importlib.import_module('.2_build_matrix', __package__)
FactorLoadingsEstimator = _step2_matrix.FactorLoadingsEstimator
CorrelationMatrixBuilder = _step2_matrix.CorrelationMatrixBuilder
FactorRiskModel = _step2_matrix.FactorRiskModel

# Step 3: Forecasting (NEW)
_step3_forecast = importlib.import_module('.3_forecasting', __package__)
ReturnForecaster = _step3_forecast.ReturnForecaster
CovarianceForecaster = _step3_forecast.CovarianceForecaster
EnsembleForecaster = _step3_forecast.EnsembleForecaster
ForecastResult = _step3_forecast.ForecastResult

# Step 4: Select Objective (was Step 3)
_step4_objective = importlib.import_module('.3_select_objective', __package__)
ObjectiveType = _step4_objective.ObjectiveType
ObjectiveFunction = _step4_objective.ObjectiveFunction
Constraints = _step4_objective.Constraints
ConstraintBuilder = _step4_objective.ConstraintBuilder
DecisionSpecs = _step4_objective.DecisionSpecs
QPOptimizer = _step4_objective.QPOptimizer
SparseSharpeOptimizer = _step4_objective.SparseSharpeOptimizer

# Step 4 (old): Parameter Estimation - kept for backward compatibility
_step4_params = importlib.import_module('.4_estimate_parameters', __package__)
ExpectedReturnsEstimator = _step4_params.ExpectedReturnsEstimator
SampleMeanEstimator = _step4_params.SampleMeanEstimator
RiskStructureEstimator = _step4_params.RiskStructureEstimator
DependencyStructureEstimator = _step4_params.DependencyStructureEstimator
KnowledgeBase = _step4_params.KnowledgeBase
ParameterEstimator = _step4_params.ParameterEstimator
SampleEstimator = _step4_params.SampleEstimator

# Step 5: Evaluation/Optimization
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
    # Step 2: Stock Selection (NEW)
    'StockSelector',
    'SelectionResult',
    # Step 2 (legacy): Build Matrix
    'FactorLoadingsEstimator',
    'CorrelationMatrixBuilder',
    'FactorRiskModel',
    # Step 3: Forecasting (NEW)
    'ReturnForecaster',
    'CovarianceForecaster',
    'EnsembleForecaster',
    'ForecastResult',
    # Step 4: Select Objective
    'ObjectiveType',
    'ObjectiveFunction',
    'Constraints',
    'ConstraintBuilder',
    'DecisionSpecs',
    'QPOptimizer',
    'SparseSharpeOptimizer',
    # Step 4 (legacy): Parameter Estimation
    'ExpectedReturnsEstimator',
    'SampleMeanEstimator',
    'RiskStructureEstimator',
    'DependencyStructureEstimator',
    'KnowledgeBase',
    'ParameterEstimator',
    'SampleEstimator',
    # Step 5: Evaluation/Optimization
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
