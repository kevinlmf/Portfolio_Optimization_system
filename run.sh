#!/bin/bash

# Portfolio Optimization System - Run Script
# Usage: ./run.sh [command]

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Show help message
show_help() {
    echo -e "${BLUE}Portfolio Optimization System - Run Script${NC}"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  workflow              Run complete workflow (Steps 1-5)"
    echo "  workflow-regime       Run regime-aware workflow (Steps 0-5) [NEW]"
    echo "  workflow-options      Run complete workflow with options hedging (Steps 1-6)"
    echo "  regime-demo           Run regime detection demo only [NEW]"
    echo "  options-demo          Run options hedging demo only"
    echo "  clean                 Clean cache files (__pycache__)"
    echo "  install               Install dependencies"
    echo "  help                  Show this help message"
    echo ""
    echo "Regime Strategies (for workflow-regime):"
    echo "  expected     - Probability-weighted expected parameters (default)"
    echo "  robust       - Minimax optimization (worst-case)"
    echo "  adaptive     - Emphasize current regime"
    echo "  worst_case   - Only consider worst regime"
    echo "  multi_regime - Joint optimization across all regimes"
    echo ""
    echo "Examples:"
    echo "  ./run.sh workflow-regime              # Default: 2 regimes, robust strategy"
    echo "  ./run.sh workflow-regime 3 adaptive   # 3 regimes, adaptive strategy"
    echo ""
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Warning: python3 not found. Trying python...${NC}"
        if ! command -v python &> /dev/null; then
            echo -e "${RED}Error: Python is not installed. Please install Python 3.7+${NC}"
            exit 1
        fi
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"
    fi
}

# Main script
check_python

case "$1" in
    workflow-regime)
        N_REGIMES=${2:-2}
        REGIME_STRATEGY=${3:-robust}
        echo -e "${BLUE}Running regime-aware workflow (Steps 0-5)...${NC}"
        echo -e "${GREEN}  Regimes: $N_REGIMES, Strategy: $REGIME_STRATEGY${NC}"
        $PYTHON_CMD -c "
import sys
import os
import importlib.util

# Get the script directory
script_dir = os.getcwd()
sys.path.insert(0, script_dir)

# Import workflow.py directly to avoid conflict with workflow/ package
workflow_file = os.path.join(script_dir, 'workflow.py')
spec = importlib.util.spec_from_file_location('workflow_module', workflow_file)
workflow_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_mod)
PortfolioWorkflow = workflow_mod.PortfolioWorkflow

from portfolio_layer.objectives.objectives import ObjectiveType
import numpy as np
import pandas as pd

print('=' * 80)
print('REGIME-AWARE PORTFOLIO OPTIMIZATION WORKFLOW')
print('=' * 80)

# Generate sample data with regime structure
np.random.seed(42)
n_samples = 500
assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'DIS', 'NVDA']

# Simulate regime-switching returns
P = np.array([[0.95, 0.05], [0.10, 0.90]])  # Transition matrix
regimes = np.zeros(n_samples, dtype=int)
for t in range(1, n_samples):
    regimes[t] = np.random.choice([0, 1], p=P[regimes[t-1]])

# Bull: high return, low vol; Bear: low return, high vol
returns_data = np.zeros((n_samples, len(assets)))
for t in range(n_samples):
    if regimes[t] == 0:  # Bull
        returns_data[t] = np.random.randn(len(assets)) * 0.015 + 0.0008
    else:  # Bear
        returns_data[t] = np.random.randn(len(assets)) * 0.025 - 0.0003

dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
returns = pd.DataFrame(returns_data, index=dates, columns=assets)

print(f'Data: {n_samples} days, {len(assets)} assets')
print(f'True regime distribution: Bull={sum(regimes==0)}, Bear={sum(regimes==1)}')

# Initialize workflow with regime mode
workflow = PortfolioWorkflow(returns, use_regime=True)

# Run complete workflow
weights = workflow.run_complete_workflow(
    objective=ObjectiveType.SHARPE,
    constraints={'long_only': True, 'max_weight': 0.25},
    n_factors=5,
    n_regimes=$N_REGIMES,
    regime_strategy='$REGIME_STRATEGY'
)

print(f'\nFinal portfolio weights:')
for i, asset in enumerate(assets):
    if weights[i] > 0.01:
        print(f'  {asset}: {weights[i]:.4f}')

# Calculate returns
portfolio_returns_series = (returns * weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns_series).cumprod()
total_return = cumulative_returns.iloc[-1] - 1
annualized_return = (cumulative_returns.iloc[-1] ** (252 / len(returns)) - 1)

print(f'\n' + '='*80)
print('ABSOLUTE RETURN SUMMARY')
print('='*80)
print(f'  Total Return: {total_return*100:.2f}%')
print(f'  Annualized Return: {annualized_return*100:.2f}%')
print(f'  Final Portfolio Value: \${cumulative_returns.iloc[-1]:.4f} (from \$1.00)')
"
        ;;
    regime-demo)
        echo -e "${BLUE}Running regime detection demo...${NC}"
        $PYTHON_CMD -m regime_layer.example_usage 2>/dev/null || \
        $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from regime_layer import HMMRegimeDetector, RegimeParameterEstimator, RegimeAwareOptimizer
import numpy as np
import pandas as pd

print('=' * 80)
print('REGIME DETECTION DEMO')
print('=' * 80)

# Generate data with regime structure
np.random.seed(42)
n_samples = 300
n_assets = 5

# Simulate regime-switching
P = np.array([[0.95, 0.05], [0.10, 0.90]])
regimes = np.zeros(n_samples, dtype=int)
for t in range(1, n_samples):
    regimes[t] = np.random.choice([0, 1], p=P[regimes[t-1]])

returns_data = np.zeros((n_samples, n_assets))
for t in range(n_samples):
    if regimes[t] == 0:
        returns_data[t] = np.random.randn(n_assets) * 0.012 + 0.0006
    else:
        returns_data[t] = np.random.randn(n_assets) * 0.022 - 0.0002

dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
returns = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i}' for i in range(n_assets)])

print(f'Data: {n_samples} days, {n_assets} assets')
print(f'True regime: Bull={sum(regimes==0)}, Bear={sum(regimes==1)}')

# 1. Regime Detection
print('\n' + '-'*40)
print('1. REGIME DETECTION (HMM)')
print('-'*40)

detector = HMMRegimeDetector(n_regimes=2, n_iter=100)
detector.fit(returns)
regime_state = detector.detect(returns)

print(f'Current regime: {regime_state.current_regime} ({regime_state.get_regime_name(regime_state.current_regime)})')
print(f'Regime probabilities: {regime_state.regime_probabilities}')
print(f'Detected distribution: {np.bincount(regime_state.regime_sequence)}')

# Accuracy
acc = max(
    np.mean(regime_state.regime_sequence == regimes),
    np.mean(regime_state.regime_sequence == (1 - regimes))
)
print(f'Detection accuracy: {acc:.2%}')

print('\nTransition Matrix:')
print(regime_state.transition_matrix)

# 2. Parameter Estimation
print('\n' + '-'*40)
print('2. REGIME-DEPENDENT PARAMETERS')
print('-'*40)

estimator = RegimeParameterEstimator(use_factor_model=False, shrinkage=True)
regime_knowledge = estimator.estimate(returns, regime_state)

for k in range(2):
    params = regime_knowledge.regime_params[k]
    name = regime_knowledge.get_regime_name(k)
    ann_ret = np.mean(params.mu) * 252
    ann_vol = np.mean(np.sqrt(np.diag(params.Sigma))) * np.sqrt(252)
    print(f'{name}: Return={ann_ret:.2%}, Vol={ann_vol:.2%}')

# 3. Optimization
print('\n' + '-'*40)
print('3. REGIME-AWARE OPTIMIZATION')
print('-'*40)

strategies = ['expected', 'robust', 'adaptive']
for strategy in strategies:
    optimizer = RegimeAwareOptimizer(strategy=strategy)
    weights = optimizer.optimize(regime_knowledge, {'long_only': True, 'leverage': 1.0})
    contrib = optimizer.compute_regime_contribution(weights, regime_knowledge)
    print(f'{strategy:12s}: Sharpe={contrib[\"expected\"][\"sharpe\"]*np.sqrt(252):.2f}, Risk={contrib[\"expected\"][\"risk\"]*np.sqrt(252):.2%}')

print('\n' + '='*80)
print('DEMO COMPLETE')
print('='*80)
"
        ;;
    workflow)
        echo -e "${BLUE}Running complete workflow (Steps 1-5)...${NC}"
        $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from workflow import PortfolioWorkflow, ObjectiveType
import numpy as np
import pandas as pd

print('=' * 80)
print('PORTFOLIO OPTIMIZATION WORKFLOW DEMO')
print('=' * 80)

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'DIS', 'NVDA']

# Generate returns
returns = pd.DataFrame(
    np.random.randn(len(dates), len(assets)) * 0.01,
    index=dates,
    columns=assets
)

# Initialize workflow
workflow = PortfolioWorkflow(returns)

# Run complete workflow
weights = workflow.run_complete_workflow(
    objective=ObjectiveType.SHARPE,
    constraints={'long_only': True, 'max_weight': 0.3},
    n_factors=5
)

print(f'\nFinal portfolio weights:\n{weights}')

# Calculate absolute returns
portfolio_returns_series = (returns * weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns_series).cumprod()
total_return = cumulative_returns.iloc[-1] - 1
annualized_return = (cumulative_returns.iloc[-1] ** (252 / len(returns)) - 1)

print(f'\n' + '='*80)
print('ABSOLUTE RETURN SUMMARY')
print('='*80)
print(f'  Total Return: {total_return*100:.2f}%')
print(f'  Annualized Return: {annualized_return*100:.2f}%')
print(f'  Final Portfolio Value: \${cumulative_returns.iloc[-1]:.4f} (starting from \$1.00)')
"
        ;;
    workflow-options)
        echo -e "${BLUE}Running complete workflow with options hedging (Steps 1-6)...${NC}"
        $PYTHON_CMD run_workflow_options.py
        ;;
    options-demo)
        echo -e "${BLUE}Running options hedging demo...${NC}"
        $PYTHON_CMD -m workflow.6_options_hedging.example_usage 2>/dev/null || \
        $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'workflow', '6_options_hedging'))

# Try to run the example
try:
    exec(open('workflow/6_options_hedging/example_usage.py').read())
except Exception as e:
    print(f'Note: Example file has relative imports. Creating standalone demo...')
    from workflow import VolatilityForecaster, BlackScholesPricer, GreeksCalculator, DeltaHedgingStrategy
    import numpy as np
    
    print('=' * 80)
    print('OPTIONS HEDGING DEMO')
    print('=' * 80)
    
    # Simple demo
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    cov = np.eye(5) * 0.02
    forecaster = VolatilityForecaster(method='portfolio_risk')
    vol = forecaster.forecast_from_portfolio(weights, cov)
    print(f'Forecasted volatility: {vol*100:.2f}%')
    
    pricer = BlackScholesPricer()
    price = pricer.price(spot=100, strike=105, time_to_expiry=0.25, risk_free_rate=0.05, volatility=vol, option_type='call')
    print(f'Option price: \${price:.4f}')
    
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_all(spot=100, strike=105, time_to_expiry=0.25, risk_free_rate=0.05, volatility=vol, option_type='call')
    print(f'Delta: {greeks[\"delta\"]:.4f}')
"
        ;;
    clean)
        echo -e "${BLUE}Cleaning cache files...${NC}"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
        find . -type f -name "*.pyc" -delete 2>/dev/null
        find . -type f -name "*.pyo" -delete 2>/dev/null
        echo -e "${GREEN}✓ Cleaned __pycache__ directories and .pyc files.${NC}"
        ;;
    install)
        echo -e "${BLUE}Installing dependencies...${NC}"
        $PYTHON_CMD -m pip install numpy pandas scipy scikit-learn yfinance
        echo -e "${GREEN}✓ Dependencies installed.${NC}"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
