#!/usr/bin/env python3
"""
Run complete workflow with options hedging and save results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow import PortfolioWorkflow, ObjectiveType
import numpy as np
import pandas as pd
import json
from datetime import datetime
import importlib

print('=' * 80)
print('PORTFOLIO OPTIMIZATION WORKFLOW WITH OPTIONS HEDGING')
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

# Run workflow Steps 1-5
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
print(f'  Final Portfolio Value: ${cumulative_returns.iloc[-1]:.4f} (starting from $1.00)')

# Step 6: Options hedging
print('\n' + '='*80)
print('ADDING OPTIONS HEDGING (STEP 6)')
print('='*80)

hedge_result = workflow.step6_options_hedging(
    spot_price=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    option_type='call',
    volatility_method='portfolio_risk'
)

print(f'\n✓ Hedging complete!')
print(f'\nOptions Hedging Summary:')
print(f'  Forecasted volatility: {hedge_result["forecasted_volatility"]*100:.2f}%')
print(f'  Delta: {hedge_result["greeks"]["delta"]:.4f}')
print(f'  Gamma: {hedge_result["greeks"]["gamma"]:.4f}')
print(f'  Theta: {hedge_result["greeks"]["theta"]:.4f}/day')
print(f'  Vega: {hedge_result["greeks"]["vega"]:.4f}')
print(f'  Rho: {hedge_result["greeks"]["rho"]:.4f}')

if 'hedge_quantity' in hedge_result['hedge_solution']:
    print(f'  Hedge quantity: {hedge_result["hedge_solution"]["hedge_quantity"]:.2f} shares')
    print(f'  Hedge value: ${hedge_result["hedge_solution"]["hedge_value"]:.2f}')

# Calculate hedged portfolio returns
print(f'\n' + '='*80)
print('')
print('ABSOLUTE RETURN SUMMARY (After Options Hedging)')
print('')
print('='*80)

# Get option price for cost calculation
_step6 = importlib.import_module('workflow.6_options_hedging.option_pricing')
BlackScholesPricer = _step6.BlackScholesPricer
pricer = BlackScholesPricer()
option_price = pricer.price(
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    volatility=hedge_result['forecasted_volatility'],
    option_type='call'
)

# Calculate hedging cost impact
portfolio_value_final = cumulative_returns.iloc[-1]
option_cost_pct = option_price / 100.0

# Simplified calculation: hedge cost reduces returns
hedged_total_return = max(0, total_return - option_cost_pct * 0.1)
hedged_final_value = max(0.1, portfolio_value_final - option_cost_pct * 0.1)
hedged_annualized_return = (hedged_final_value ** (252 / len(returns)) - 1) if hedged_final_value > 0 else 0

print(f'  Total Return: {hedged_total_return*100:.2f}%')
print(f'    (Original: {total_return*100:.2f}%, Hedging Cost: -{option_cost_pct*0.1*100:.3f}%)')
print(f'')
print(f'  Annualized Return: {hedged_annualized_return*100:.2f}%')
print(f'    (Original: {annualized_return*100:.2f}%)')
print(f'')
print(f'  Final Portfolio Value: ${hedged_final_value:.4f} (starting from $1.00)')
print(f'    (Original: ${portfolio_value_final:.4f})')
print(f'')
print(f'  ⚠️  Note: This is a simplified calculation. In reality, hedging')
print(f'        primarily REDUCES VOLATILITY/RISK, not expected returns.')
print(f'        The cost shown is minimal; real benefit is risk reduction.')

# Save results to result/ folder
print(f'\n' + '='*80)
print('SAVING RESULTS TO result/ FOLDER')
print('='*80)

# Create result directory
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. Save selected factors
if workflow.factors is not None:
    factor_file = os.path.join(result_dir, f'factors_{timestamp}.csv')
    workflow.factors.to_csv(factor_file)
    print(f'✓ Saved factors to: {factor_file}')
    print(f'  Number of factors: {workflow.factors.shape[1]}')
    print(f'  Factor names: {list(workflow.factors.columns)}')

# 2. Save matrices (B, F, D, Σ)
if workflow.knowledge is not None:
    matrices_dir = os.path.join(result_dir, f'matrices_{timestamp}')
    os.makedirs(matrices_dir, exist_ok=True)
    
    # Save factor loadings matrix (B)
    if workflow.knowledge.B is not None:
        B_file = os.path.join(matrices_dir, 'factor_loadings_B.csv')
        pd.DataFrame(workflow.knowledge.B).to_csv(B_file)
        print(f'✓ Saved factor loadings matrix (B) to: {B_file}')
        print(f'  Shape: {workflow.knowledge.B.shape}')
    
    # Save factor covariance matrix (F)
    if workflow.knowledge.F is not None:
        F_file = os.path.join(matrices_dir, 'factor_covariance_F.csv')
        pd.DataFrame(workflow.knowledge.F).to_csv(F_file)
        print(f'✓ Saved factor covariance matrix (F) to: {F_file}')
        print(f'  Shape: {workflow.knowledge.F.shape}')
    
    # Save idiosyncratic risk matrix (D)
    if workflow.knowledge.D is not None:
        D_file = os.path.join(matrices_dir, 'idiosyncratic_risk_D.csv')
        pd.DataFrame(workflow.knowledge.D).to_csv(D_file)
        print(f'✓ Saved idiosyncratic risk matrix (D) to: {D_file}')
        print(f'  Shape: {workflow.knowledge.D.shape}')
    
    # Save full covariance matrix (Σ)
    Sigma = workflow.knowledge.get_covariance()
    Sigma_file = os.path.join(matrices_dir, 'covariance_Sigma.csv')
    pd.DataFrame(Sigma).to_csv(Sigma_file)
    print(f'✓ Saved covariance matrix (Σ) to: {Sigma_file}')
    print(f'  Shape: {Sigma.shape}')

# 3. Save evaluation metrics
evaluation_file = os.path.join(result_dir, f'evaluation_{timestamp}.json')
evaluation_metrics = {
    'portfolio_weights': weights.tolist(),
    'asset_names': list(returns.columns),
    'total_return': float(total_return),
    'annualized_return': float(annualized_return),
    'final_portfolio_value': float(cumulative_returns.iloc[-1]),
    'hedged_total_return': float(hedged_total_return),
    'hedged_annualized_return': float(hedged_annualized_return),
    'hedged_final_value': float(hedged_final_value),
    'top_5_holdings': {}
}
# Add top 5 holdings
if workflow.knowledge is not None:
    asset_names = workflow.knowledge.asset_names if workflow.knowledge.asset_names else list(returns.columns)
    top_indices = np.argsort(weights)[::-1][:5]
    for idx in top_indices:
        evaluation_metrics['top_5_holdings'][asset_names[idx]] = float(weights[idx])

with open(evaluation_file, 'w') as f:
    json.dump(evaluation_metrics, f, indent=2)
print(f'✓ Saved evaluation metrics to: {evaluation_file}')

# 4. Save options hedging details
options_file = os.path.join(result_dir, f'options_hedging_{timestamp}.json')
options_data = {
    'option_parameters': {
        'spot_price': 100.0,
        'strike': 105.0,
        'time_to_expiry': 0.25,
        'risk_free_rate': 0.05,
        'option_type': 'call',
        'option_price': float(option_price)
    },
    'forecasted_volatility': float(hedge_result['forecasted_volatility']),
    'greeks': {
        'delta': float(hedge_result['greeks']['delta']),
        'gamma': float(hedge_result['greeks']['gamma']),
        'theta': float(hedge_result['greeks']['theta']),
        'vega': float(hedge_result['greeks']['vega']),
        'rho': float(hedge_result['greeks']['rho'])
    },
    'hedge_solution': {
        'hedge_quantity': float(hedge_result['hedge_solution'].get('hedge_quantity', 0)),
        'hedge_value': float(hedge_result['hedge_solution'].get('hedge_value', 0)),
        'target_delta': float(hedge_result['hedge_solution'].get('target_delta', 0))
    }
}
with open(options_file, 'w') as f:
    json.dump(options_data, f, indent=2)
print(f'✓ Saved options hedging details to: {options_file}')
print(f'  Option quantity: 1 contract (100 shares)')
print(f'  Hedge quantity: {hedge_result["hedge_solution"].get("hedge_quantity", 0):.2f} shares')

# 5. Save summary report
summary_file = os.path.join(result_dir, f'summary_{timestamp}.txt')
with open(summary_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('PORTFOLIO OPTIMIZATION RESULTS SUMMARY\n')
    f.write('='*80 + '\n\n')
    f.write(f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    
    f.write('1. SELECTED FACTORS:\n')
    f.write('-'*80 + '\n')
    if workflow.factors is not None:
        f.write(f'   Number of factors: {workflow.factors.shape[1]}\n')
        f.write(f'   Factor names: {list(workflow.factors.columns)}\n')
    f.write('\n')
    
    f.write('2. MATRICES:\n')
    f.write('-'*80 + '\n')
    if workflow.knowledge is not None:
        if workflow.knowledge.B is not None:
            f.write(f'   Factor Loadings (B): {workflow.knowledge.B.shape}\n')
        if workflow.knowledge.F is not None:
            f.write(f'   Factor Covariance (F): {workflow.knowledge.F.shape}\n')
        if workflow.knowledge.D is not None:
            f.write(f'   Idiosyncratic Risk (D): {workflow.knowledge.D.shape}\n')
        Sigma = workflow.knowledge.get_covariance()
        f.write(f'   Full Covariance (Σ): {Sigma.shape}\n')
    f.write('\n')
    
    f.write('3. EVALUATION METRICS:\n')
    f.write('-'*80 + '\n')
    f.write(f'   Total Return: {total_return*100:.2f}%\n')
    f.write(f'   Annualized Return: {annualized_return*100:.2f}%\n')
    f.write(f'   Final Portfolio Value: ${cumulative_returns.iloc[-1]:.4f}\n')
    f.write('\n')
    
    f.write('4. OPTIONS HEDGING:\n')
    f.write('-'*80 + '\n')
    f.write(f'   Option Type: Call\n')
    f.write(f'   Spot Price: $100.0\n')
    f.write(f'   Strike Price: $105.0\n')
    f.write(f'   Option Price: ${option_price:.4f}\n')
    f.write(f'   Forecasted Volatility: {hedge_result["forecasted_volatility"]*100:.2f}%\n')
    f.write(f'   Hedge Quantity: {hedge_result["hedge_solution"].get("hedge_quantity", 0):.2f} shares\n')
    f.write('\n')
    
    f.write('5. ABSOLUTE RETURNS (After Hedging):\n')
    f.write('-'*80 + '\n')
    f.write(f'   Total Return: {hedged_total_return*100:.2f}%\n')
    f.write(f'   Annualized Return: {hedged_annualized_return*100:.2f}%\n')
    f.write(f'   Final Portfolio Value: ${hedged_final_value:.4f}\n')
    f.write('\n')

print(f'✓ Saved summary report to: {summary_file}')
print(f'\n✅ All results saved to {result_dir}/ directory!')

