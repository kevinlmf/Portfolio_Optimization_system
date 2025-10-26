#!/bin/bash

################################################################################
# Portfolio Optimization System - Master Run Script
#
# This script runs the complete portfolio optimization pipeline:
# 1. Data Acquisition
# 2. Market Regime Detection
# 3. Factor Analysis
# 4. Portfolio Optimization
# 5. Backtesting and Evaluation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

################################################################################
# Configuration
################################################################################

START_DATE="2020-01-01"
END_DATE="2024-01-01"
PORTFOLIO_TYPE="balanced"  # Options: aggressive, balanced, conservative, all_assets
INITIAL_CAPITAL=100000
REBALANCE_FREQ="monthly"   # Options: daily, weekly, monthly, quarterly

################################################################################
# Main Pipeline
################################################################################

print_header "PORTFOLIO OPTIMIZATION SYSTEM"
echo "Configuration:"
echo "  Period: $START_DATE to $END_DATE"
echo "  Portfolio Type: $PORTFOLIO_TYPE"
echo "  Initial Capital: \$$INITIAL_CAPITAL"
echo "  Rebalance Frequency: $REBALANCE_FREQ"
echo ""

# Step 0: Check environment
print_header "Step 0: Environment Check"
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python 3.7+"
    exit 1
fi
print_success "Python3 found: $(python3 --version)"

if [ ! -d "venv" ]; then
    print_info "Virtual environment not found. Creating..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

print_info "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
print_success "Dependencies installed"

# Step 1: Quick validation
print_header "Step 1: System Validation"
print_info "Running quick validation test..."
python3 scripts/quick_test.py
if [ $? -eq 0 ]; then
    print_success "System validation passed"
else
    print_error "System validation failed"
    exit 1
fi

# Step 2: Run Bayesian System Test
print_header "Step 2: Bayesian System Test"
print_info "Testing new Mixture + Bayesian system..."
python3 scripts/test_bayesian_system.py
if [ $? -eq 0 ]; then
    print_success "Bayesian system test completed"
    print_info "Results saved to: results/bayesian_system/"
else
    print_error "Bayesian system test failed"
fi

# Step 3: Historical Analysis
print_header "Step 3: Historical Analysis (2008-2024)"
print_info "Running 16-year backtest..."
python3 scripts/historical_analysis.py
if [ $? -eq 0 ]; then
    print_success "Historical analysis completed"
    print_info "Results saved to: results/historical_analysis/"
else
    print_error "Historical analysis failed"
fi

# Step 4: Factor Analysis
print_header "Step 4: Factor Analysis"
print_info "Running factor selection and analysis..."
python3 scripts/demo_factor_selection.py
python3 scripts/demo_factor_analysis.py
if [ $? -eq 0 ]; then
    print_success "Factor analysis completed"
    print_info "Results saved to: results/factor_analysis/"
else
    print_error "Factor analysis failed"
fi

# Step 5: Comprehensive System
print_header "Step 5: Comprehensive Portfolio System"
print_info "Running full optimization pipeline..."
python3 scripts/comprehensive_portfolio_system.py
if [ $? -eq 0 ]; then
    print_success "Comprehensive system completed"
    print_info "Results saved to: results/comprehensive_backtest/"
else
    print_error "Comprehensive system failed"
fi

################################################################################
# Summary
################################################################################

print_header "EXECUTION SUMMARY"

echo "All results have been saved to the 'results/' directory:"
echo ""
echo "  results/bayesian_system/         - Bayesian + Mixture model results"
echo "  results/historical_analysis/     - 16-year backtest (2008-2024)"
echo "  results/factor_analysis/         - Factor selection and analysis"
echo "  results/comprehensive_backtest/  - Full system backtest"
echo ""

if [ -f "results/bayesian_system/performance_metrics.csv" ]; then
    print_success "Bayesian System Results:"
    python3 << 'EOF'
import pandas as pd
try:
    df = pd.read_csv('results/bayesian_system/performance_metrics.csv')
    print(f"  Total Return:     {df['total_return'].iloc[0]:.2%}")
    print(f"  Annual Return:    {df['annual_return'].iloc[0]:.2%}")
    print(f"  Sharpe Ratio:     {df['sharpe_ratio'].iloc[0]:.3f}")
    print(f"  Max Drawdown:     {df['max_drawdown'].iloc[0]:.2%}")
except:
    pass
EOF
fi

echo ""
print_success "Pipeline execution complete!"
echo ""
echo "To view visualizations:"
echo "  open results/bayesian_system/comprehensive_analysis.png"
echo ""
echo "To view detailed reports:"
echo "  cat results/bayesian_system/test_report.txt"
echo ""

################################################################################
# Optional: Open results browser
################################################################################

read -p "Open results directory in file browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open results/
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open results/
    else
        print_info "Please manually open: results/"
    fi
fi

print_header "Thank you for using Portfolio Optimization System!"
