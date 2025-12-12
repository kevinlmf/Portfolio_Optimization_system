"""
Example Usage - 期权对冲示例

展示如何使用波动率预测、期权定价、希腊字母计算和对冲策略
"""

import numpy as np
import pandas as pd
from .volatility_forecast import VolatilityForecaster
from .option_pricing import BlackScholesPricer
from .greeks_calculator import GreeksCalculator
from .hedging_strategy import DeltaHedgingStrategy, GreeksHedgingStrategy


def example_volatility_forecast():
    """示例：从投资组合预测波动率"""
    print("=" * 80)
    print("EXAMPLE 1: Volatility Forecast from Portfolio")
    print("=" * 80)
    
    # 模拟组合权重和协方差矩阵
    n_assets = 5
    portfolio_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # 生成协方差矩阵
    np.random.seed(42)
    returns_sample = np.random.randn(252, n_assets) * 0.02
    covariance_matrix = np.cov(returns_sample.T)
    
    # 预测波动率
    forecaster = VolatilityForecaster(method='portfolio_risk')
    forecasted_vol = forecaster.forecast_from_portfolio(
        portfolio_weights=portfolio_weights,
        covariance_matrix=covariance_matrix,
        horizon=21,
        annualization=252
    )
    
    print(f"\nPortfolio weights: {portfolio_weights}")
    print(f"Forecasted annualized volatility: {forecasted_vol:.4f} ({forecasted_vol*100:.2f}%)")
    print(f"21-day horizon volatility: {forecasted_vol * np.sqrt(21/252):.4f}")
    
    return forecasted_vol


def example_option_pricing():
    """示例：期权定价"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Option Pricing")
    print("=" * 80)
    
    pricer = BlackScholesPricer()
    
    # 参数
    spot = 100.0
    strike = 105.0
    time_to_expiry = 0.25  # 3个月
    risk_free_rate = 0.05
    volatility = 0.20
    
    # 计算Call和Put价格
    call_price = pricer.price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type='call'
    )
    
    put_price = pricer.price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type='put'
    )
    
    print(f"\nSpot Price: ${spot}")
    print(f"Strike Price: ${strike}")
    print(f"Time to Expiry: {time_to_expiry*365:.0f} days")
    print(f"Risk-free Rate: {risk_free_rate*100:.2f}%")
    print(f"Volatility: {volatility*100:.2f}%")
    print(f"\nCall Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    
    return call_price, put_price


def example_greeks():
    """示例：计算希腊字母"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Greeks Calculation")
    print("=" * 80)
    
    greeks_calc = GreeksCalculator()
    
    # 参数
    spot = 100.0
    strike = 105.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    
    # 计算所有希腊字母
    greeks_call = greeks_calc.calculate_all(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type='call'
    )
    
    greeks_put = greeks_calc.calculate_all(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type='put'
    )
    
    print(f"\n{'Greek':<10} {'Call':<15} {'Put':<15} {'Description':<40}")
    print("-" * 80)
    print(f"{'Δ (Delta)':<10} {greeks_call['delta']:<15.4f} {greeks_put['delta']:<15.4f} {'Price sensitivity (Directional exposure)'}")
    print(f"{'Γ (Gamma)':<10} {greeks_call['gamma']:<15.4f} {greeks_put['gamma']:<15.4f} {'Delta change rate (Convexity)'}")
    print(f"{'Θ (Theta)':<10} {greeks_call['theta']:<15.4f} {greeks_put['theta']:<15.4f} {'Time decay (Time value)'}")
    print(f"{'V (Vega)':<10} {greeks_call['vega']:<15.4f} {greeks_put['vega']:<15.4f} {'Volatility sensitivity'}")
    print(f"{'ρ (Rho)':<10} {greeks_call['rho']:<15.4f} {greeks_put['rho']:<15.4f} {'Interest rate sensitivity'}")
    
    return greeks_call, greeks_put


def example_portfolio_greeks():
    """示例：计算组合希腊字母"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Portfolio Greeks")
    print("=" * 80)
    
    greeks_calc = GreeksCalculator()
    
    # 创建期权组合
    positions = {
        'call_1': {
            'spot': 100.0,
            'strike': 105.0,
            'time_to_expiry': 0.25,
            'risk_free_rate': 0.05,
            'volatility': 0.20,
            'option_type': 'call',
            'quantity': 10  # 持有10份看涨期权
        },
        'put_1': {
            'spot': 100.0,
            'strike': 95.0,
            'time_to_expiry': 0.25,
            'risk_free_rate': 0.05,
            'volatility': 0.20,
            'option_type': 'put',
            'quantity': 5  # 持有5份看跌期权
        },
    }
    
    portfolio_greeks = greeks_calc.calculate_portfolio_greeks(positions)
    
    print(f"\nPortfolio Positions:")
    for pos_id, params in positions.items():
        print(f"  {pos_id}: {params['quantity']} {params['option_type']}s, Strike=${params['strike']}")
    
    print(f"\nPortfolio Greeks:")
    print(f"  Total Δ (Delta): {portfolio_greeks['delta']:.4f}")
    print(f"  Total Γ (Gamma): {portfolio_greeks['gamma']:.4f}")
    print(f"  Total Θ (Theta): {portfolio_greeks['theta']:.4f} (per day)")
    print(f"  Total V (Vega): {portfolio_greeks['vega']:.4f}")
    print(f"  Total ρ (Rho): {portfolio_greeks['rho']:.4f}")
    
    return portfolio_greeks


def example_delta_hedging():
    """示例：Delta对冲"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Delta Hedging")
    print("=" * 80)
    
    greeks_calc = GreeksCalculator()
    hedging_strategy = DeltaHedgingStrategy()
    
    # 期权持仓
    positions = {
        'call_1': {
            'spot': 100.0,
            'strike': 105.0,
            'time_to_expiry': 0.25,
            'risk_free_rate': 0.05,
            'volatility': 0.20,
            'option_type': 'call',
            'quantity': 10
        },
    }
    
    # 计算组合Delta
    portfolio_greeks = greeks_calc.calculate_portfolio_greeks(positions)
    portfolio_delta = portfolio_greeks['delta']
    
    print(f"\nPortfolio Delta: {portfolio_delta:.4f}")
    print(f"  (If Delta > 0, portfolio gains when price rises)")
    
    # 计算对冲方案
    hedge_solution = hedging_strategy.calculate_hedge(
        portfolio_delta=portfolio_delta,
        spot_price=100.0,
        contract_size=100.0  # 每份期权对应100股
    )
    
    print(f"\nHedging Solution:")
    print(f"  Strategy: {hedge_solution['strategy']}")
    print(f"  Hedge Quantity: {hedge_solution['hedge_quantity']:.2f} shares")
    print(f"    (Negative = sell, Positive = buy)")
    print(f"  Hedge Value: ${hedge_solution['hedge_value']:.2f}")
    print(f"  Hedge Ratio: {hedge_solution['hedge_ratio']:.2f} contracts")
    
    return hedge_solution


def example_complete_workflow():
    """完整工作流示例：从组合构建到对冲"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complete Workflow (Portfolio → Volatility → Greeks → Hedging)")
    print("=" * 80)
    
    # Step 1: 从组合预测波动率
    portfolio_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    np.random.seed(42)
    returns_sample = np.random.randn(252, 5) * 0.02
    covariance_matrix = np.cov(returns_sample.T)
    
    forecaster = VolatilityForecaster(method='portfolio_risk')
    forecasted_vol = forecaster.forecast_from_portfolio(
        portfolio_weights=portfolio_weights,
        covariance_matrix=covariance_matrix,
        horizon=21
    )
    
    print(f"\nStep 1: Forecasted Volatility from Portfolio = {forecasted_vol*100:.2f}%")
    
    # Step 2: 使用预测波动率定价期权
    pricer = BlackScholesPricer()
    spot = 100.0
    strike = 105.0
    
    option_price = pricer.price(
        spot=spot,
        strike=strike,
        time_to_expiry=0.25,
        risk_free_rate=0.05,
        volatility=forecasted_vol,
        option_type='call'
    )
    
    print(f"\nStep 2: Option Price (using forecasted vol) = ${option_price:.4f}")
    
    # Step 3: 计算希腊字母
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_all(
        spot=spot,
        strike=strike,
        time_to_expiry=0.25,
        risk_free_rate=0.05,
        volatility=forecasted_vol,
        option_type='call'
    )
    
    print(f"\nStep 3: Greeks (using forecasted vol):")
    print(f"  Δ = {greeks['delta']:.4f}")
    print(f"  Γ = {greeks['gamma']:.4f}")
    print(f"  Θ = {greeks['theta']:.4f}/day")
    print(f"  V = {greeks['vega']:.4f}")
    print(f"  ρ = {greeks['rho']:.4f}")
    
    # Step 4: 对冲
    hedging_strategy = DeltaHedgingStrategy()
    hedge_solution = hedging_strategy.calculate_hedge(
        portfolio_delta=greeks['delta'] * 10,  # 假设持有10份期权
        spot_price=spot,
        contract_size=100.0
    )
    
    print(f"\nStep 4: Hedging Solution:")
    print(f"  Need to trade {abs(hedge_solution['hedge_quantity']):.2f} shares")
    print(f"  To neutralize directional exposure")


if __name__ == '__main__':
    # 运行所有示例
    example_volatility_forecast()
    example_option_pricing()
    example_greeks()
    example_portfolio_greeks()
    example_delta_hedging()
    example_complete_workflow()

