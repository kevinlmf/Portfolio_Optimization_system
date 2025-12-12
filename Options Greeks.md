### Understanding the Greeks

The system calculates all five option Greeks to help you understand and manage risk:

#### Δ (Delta) - Price Sensitivity
- **What it measures**: How much the option price changes for a $1 change in the underlying asset price
- **Range**: Call options (0 to 1), Put options (-1 to 0)
- **Purpose**: Controls **directional exposure**
- **Hedging goal**: Delta = 0 creates a delta-neutral portfolio with no directional risk

#### Γ (Gamma) - Convexity Management
- **What it measures**: The rate of change of Delta as the underlying price moves
- **Purpose**: Manages **convexity (nonlinearity)** in the portfolio
- **Hedging goal**: Control convexity to manage hedging frequency and transaction costs
- **Impact**: High gamma requires frequent rebalancing for delta neutrality

#### Θ (Theta) - Time Decay
- **What it measures**: How much option value decreases per day as expiration approaches
- **Purpose**: Captures **time value erosion**
- **Hedging goal**: Understand the time cost of holding options
- **Strategy**: Balance time decay against other Greeks when constructing hedges

#### V (Vega) - Volatility Sensitivity
- **What it measures**: How much the option price changes for a 1% change in volatility
- **Purpose**: **Betting on volatility direction** or hedging volatility risk
- **Hedging goal**: Explicitly trade volatility expectations
- **Application**: Useful for volatility trading strategies

#### ρ (Rho) - Interest Rate Sensitivity
- **What it measures**: Sensitivity to changes in the risk-free interest rate
- **Purpose**: Manages **interest rate exposure**
- **Hedging goal**: Control interest rate risk in long-term options
- **Relevance**: More important for longer-dated options
