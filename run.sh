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
    echo "  workflow          Run complete workflow (Steps 1-5)"
    echo "  workflow-options  Run complete workflow with options hedging (Steps 1-6)"
    echo "  options-demo      Run options hedging demo only"
    echo "  clean             Clean cache files (__pycache__)"
    echo "  install           Install dependencies"
    echo "  help              Show this help message"
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
