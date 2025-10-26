"""
Integrated Factor System

This module bridges the Factor Mining System (133 factors) with the Portfolio
Optimization System, enabling:
1. Factor-based portfolio construction
2. Factor exposure optimization
3. Factor timing strategies
4. Multi-factor alpha generation

Combines:
- 133 factors from /Quant/Factors (technical, fundamental, macro, ML, beta)
- Factor analysis framework (Fama-French, PCA)
- Portfolio optimization (max Sharpe, min variance, risk parity)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add path to Factors system
FACTORS_PATH = Path(__file__).parent.parent.parent.parent / "Factors"
if FACTORS_PATH.exists():
    sys.path.insert(0, str(FACTORS_PATH))

from strategy.factor_analyzer import FactorAnalyzer, FactorTimingAnalyzer


class IntegratedFactorSystem:
    """
    Integrated system combining factor mining with portfolio optimization.

    Features:
    1. Import 133 factors from Factor Mining System
    2. Run factor analysis (Fama-French, PCA)
    3. Factor-based portfolio construction
    4. Factor exposure optimization
    5. Factor timing and regime adaptation
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 prices: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.03):
        """
        Initialize integrated factor system.

        Args:
            returns: Asset returns DataFrame (assets as columns)
            prices: Asset prices DataFrame (optional, for factor mining)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.prices = prices if prices is not None else (1 + returns).cumprod()
        self.risk_free_rate = risk_free_rate

        # Initialize factor analyzer
        self.factor_analyzer = FactorAnalyzer(returns, risk_free_rate)

        # Storage for results
        self.alpha_factors = None
        self.beta_factors = None
        self.combined_factors = None
        self.factor_scores = None
        self.optimal_weights = None

    def load_factor_library(self,
                           include_technical: bool = True,
                           include_fundamental: bool = False,
                           include_macro: bool = True,
                           include_ml: bool = True,
                           include_beta: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load factors from Factor Mining System.

        Args:
            include_technical: Include 49 technical factors
            include_fundamental: Include 27 fundamental factors
            include_macro: Include 20 macro factors
            include_ml: Include 15 ML factors
            include_beta: Include 22 beta factors

        Returns:
            Dictionary of factor DataFrames by category
        """
        print(f"\n{'='*80}")
        print("LOADING FACTOR LIBRARY (133 FACTORS)")
        print(f"{'='*80}\n")

        try:
            from factors.factory import FactorFactory

            # Prepare data for factor generation
            data = self._prepare_data_for_factors()

            # Initialize factory
            factory = FactorFactory(data)

            # Generate factors
            all_factors = factory.generate_all_factors(
                include_technical=include_technical,
                include_fundamental=include_fundamental,
                include_macro=include_macro,
                include_ml=include_ml,
                include_beta=include_beta,
                market_data=data
            )

            # Get summary
            summary = factory.get_factor_summary()
            print(f"\n✓ Factor Library Loaded Successfully")
            print(f"  Total Factors: {summary['total_factors']}")
            for category, info in summary['by_category'].items():
                print(f"  {category.upper()}: {info['count']} factors")

            # Store results
            self.alpha_factors = all_factors

            return factory.factors_cache

        except ImportError as e:
            print(f"⚠ Could not load Factor Mining System: {e}")
            print(f"  Expected path: {FACTORS_PATH}")
            print(f"  Falling back to built-in factor models...")
            return {}

    def _prepare_data_for_factors(self) -> pd.DataFrame:
        """
        Prepare data in format expected by Factor Mining System.

        Expected format:
        - Columns: date, ticker, open, high, low, close, volume
        """
        data_list = []

        for ticker in self.prices.columns:
            ticker_prices = self.prices[ticker].dropna()

            for date, close in ticker_prices.items():
                # Estimate OHLC from close if not available
                row = {
                    'date': date,
                    'ticker': ticker,
                    'open': close * (1 + np.random.uniform(-0.005, 0.005)),
                    'high': close * (1 + np.random.uniform(0, 0.01)),
                    'low': close * (1 - np.random.uniform(0, 0.01)),
                    'close': close,
                    'volume': 1000000  # Placeholder
                }
                data_list.append(row)

        return pd.DataFrame(data_list)

    def construct_style_factors(self,
                                market_proxy: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Construct Fama-French style factors.

        Args:
            market_proxy: Market return series (if None, uses equal-weighted)

        Returns:
            DataFrame of factor returns
        """
        print(f"\n{'='*80}")
        print("CONSTRUCTING FAMA-FRENCH STYLE FACTORS")
        print(f"{'='*80}\n")

        factors = self.factor_analyzer.construct_market_factors(
            market_proxy=market_proxy,
            include_size=True,
            include_value=True,
            include_momentum=True,
            include_quality=True,
            include_investment=True
        )

        print(f"✓ Style Factors Constructed:")
        for factor in factors.columns:
            mean_ret = factors[factor].mean() * 252
            vol = factors[factor].std() * np.sqrt(252)
            sharpe = mean_ret / vol if vol > 0 else 0
            print(f"  {factor:8s}: Return={mean_ret:>7.2%}, Vol={vol:>6.2%}, Sharpe={sharpe:>6.3f}")

        self.beta_factors = factors
        return factors

    def extract_statistical_factors(self, n_factors: int = 5) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract statistical factors using PCA.

        Args:
            n_factors: Number of principal components

        Returns:
            Tuple of (factor_returns, loadings)
        """
        print(f"\n{'='*80}")
        print("EXTRACTING STATISTICAL FACTORS (PCA)")
        print(f"{'='*80}")

        factor_returns, loadings = self.factor_analyzer.extract_pca_factors(
            n_factors=n_factors,
            standardize=True
        )

        return factor_returns, loadings

    def analyze_portfolio_factors(self,
                                  portfolio_returns: pd.Series,
                                  portfolio_weights: Optional[np.ndarray] = None,
                                  models: List[str] = None) -> pd.DataFrame:
        """
        Comprehensive factor analysis of portfolio.

        Args:
            portfolio_returns: Portfolio return series
            portfolio_weights: Portfolio weights (optional)
            models: Factor models to test

        Returns:
            DataFrame with factor analysis report
        """
        print(f"\n{'='*80}")
        print("PORTFOLIO FACTOR ANALYSIS")
        print(f"{'='*80}")

        # Ensure factors are constructed
        if self.beta_factors is None:
            self.construct_style_factors()

        # Run comprehensive analysis
        report = self.factor_analyzer.generate_factor_report(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            models=models or ['fama_french_3', 'fama_french_5', 'carhart_4']
        )

        return report

    def optimize_factor_exposures(self,
                                 target_exposures: Dict[str, float],
                                 max_weight: float = 0.30,
                                 min_weight: float = 0.0) -> np.ndarray:
        """
        Optimize portfolio to achieve target factor exposures.

        Args:
            target_exposures: Dictionary of target factor exposures
            max_weight: Maximum asset weight
            min_weight: Minimum asset weight

        Returns:
            Optimal portfolio weights
        """
        from scipy.optimize import minimize

        print(f"\n{'='*80}")
        print("FACTOR EXPOSURE OPTIMIZATION")
        print(f"{'='*80}\n")

        print(f"Target Factor Exposures:")
        for factor, exposure in target_exposures.items():
            print(f"  {factor:8s}: {exposure:>7.4f}")

        # Calculate factor exposures for each asset
        n_assets = len(self.returns.columns)

        if self.beta_factors is None:
            self.construct_style_factors()

        # Run regression for each asset to get betas
        asset_betas = np.zeros((n_assets, len(self.beta_factors.columns)))

        for i, asset in enumerate(self.returns.columns):
            asset_returns = self.returns[asset]
            common_idx = asset_returns.index.intersection(self.beta_factors.index)

            if len(common_idx) < 60:  # Need minimum data
                continue

            y = asset_returns.loc[common_idx].values
            X = self.beta_factors.loc[common_idx].values
            X_with_const = np.column_stack([np.ones(len(X)), X])

            try:
                betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                asset_betas[i, :] = betas[1:]  # Exclude alpha
            except:
                pass

        # Create target vector
        factor_names = self.beta_factors.columns.tolist()
        target_vector = np.array([target_exposures.get(f, 0) for f in factor_names])

        # Optimization objective: minimize distance to target exposures
        def objective(w):
            portfolio_betas = w @ asset_betas
            return np.sum((portfolio_betas - target_vector) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]

        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
            achieved_exposures = optimal_weights @ asset_betas

            print(f"\n✓ Optimization Successful")
            print(f"\nAchieved Factor Exposures:")
            for i, factor in enumerate(factor_names):
                target = target_exposures.get(factor, 0)
                achieved = achieved_exposures[i]
                diff = achieved - target
                print(f"  {factor:8s}: Target={target:>7.4f}, Achieved={achieved:>7.4f}, Diff={diff:>7.4f}")

            # Print top holdings
            top_holdings = sorted(zip(self.returns.columns, optimal_weights),
                                 key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop 10 Holdings:")
            for asset, weight in top_holdings:
                if weight > 0.001:
                    print(f"  {asset:10s}: {weight:>6.2%}")

            self.optimal_weights = optimal_weights
            return optimal_weights
        else:
            print(f"⚠ Optimization failed: {result.message}")
            return np.ones(n_assets) / n_assets

    def create_factor_tilted_portfolio(self,
                                      tilt_factors: List[str],
                                      tilt_strength: float = 1.0,
                                      base_method: str = 'risk_parity') -> np.ndarray:
        """
        Create portfolio with factor tilts on top of base optimization.

        Args:
            tilt_factors: List of factors to tilt towards
            tilt_strength: Strength of tilt (0=no tilt, 1=moderate, 2=strong)
            base_method: Base optimization method

        Returns:
            Portfolio weights
        """
        from scipy.optimize import minimize

        print(f"\n{'='*80}")
        print(f"FACTOR-TILTED PORTFOLIO: {base_method.upper()}")
        print(f"{'='*80}\n")

        print(f"Tilting towards factors: {', '.join(tilt_factors)}")
        print(f"Tilt strength: {tilt_strength}")

        n_assets = len(self.returns.columns)
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values

        # Base optimization objective
        if base_method == 'max_sharpe':
            def base_objective(w):
                ret = np.dot(w, mean_returns)
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                return -(ret / vol) if vol > 0 else 1e6

        elif base_method == 'min_variance':
            def base_objective(w):
                return np.dot(w.T, np.dot(cov_matrix, w))

        elif base_method == 'risk_parity':
            def base_objective(w):
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                marginal_contrib = np.dot(cov_matrix, w) / vol
                risk_contrib = w * marginal_contrib
                target_risk = vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)
        else:
            def base_objective(w):
                return 0

        # Factor tilt component
        if self.beta_factors is None:
            self.construct_style_factors()

        # Calculate asset factor loadings
        asset_factors = np.zeros((n_assets, len(self.beta_factors.columns)))
        for i, asset in enumerate(self.returns.columns):
            asset_returns = self.returns[asset]
            common_idx = asset_returns.index.intersection(self.beta_factors.index)

            if len(common_idx) >= 60:
                y = asset_returns.loc[common_idx].values
                X = self.beta_factors.loc[common_idx].values
                X_with_const = np.column_stack([np.ones(len(X)), X])

                try:
                    betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                    asset_factors[i, :] = betas[1:]
                except:
                    pass

        # Identify tilt factor indices
        factor_names = self.beta_factors.columns.tolist()
        tilt_indices = [i for i, name in enumerate(factor_names) if name in tilt_factors]

        # Combined objective
        def combined_objective(w):
            base_score = base_objective(w)

            # Factor tilt score (maximize exposure to tilt factors)
            tilt_score = 0
            for idx in tilt_indices:
                factor_exposure = np.dot(w, asset_factors[:, idx])
                tilt_score -= factor_exposure  # Negative to maximize

            return base_score + tilt_strength * tilt_score

        # Optimize
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 0.30) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        result = minimize(combined_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x

            # Calculate factor exposures
            exposures = weights @ asset_factors

            print(f"\n✓ Factor-Tilted Portfolio Created")
            print(f"\nFactor Exposures:")
            for i, factor in enumerate(factor_names):
                marker = " ◀" if factor in tilt_factors else ""
                print(f"  {factor:8s}: {exposures[i]:>7.4f}{marker}")

            # Print top holdings
            top_holdings = sorted(zip(self.returns.columns, weights),
                                 key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop 10 Holdings:")
            for asset, weight in top_holdings:
                if weight > 0.001:
                    print(f"  {asset:10s}: {weight:>6.2%}")

            return weights
        else:
            print(f"⚠ Optimization failed")
            return np.ones(n_assets) / n_assets

    def run_factor_timing(self,
                         lookback: int = 60,
                         top_n_factors: int = 3) -> Dict:
        """
        Analyze factor timing opportunities.

        Args:
            lookback: Lookback period for momentum calculation
            top_n_factors: Number of top factors to identify

        Returns:
            Dictionary with timing analysis
        """
        print(f"\n{'='*80}")
        print("FACTOR TIMING ANALYSIS")
        print(f"{'='*80}")

        if self.beta_factors is None:
            self.construct_style_factors()

        timing_analyzer = FactorTimingAnalyzer(self.beta_factors)

        # Analyze factor momentum
        momentum_df = timing_analyzer.analyze_factor_momentum(lookback=lookback)

        print(f"\nFactor Momentum Rankings:")
        print(momentum_df.to_string(index=False))

        # Top factors
        top_factors = momentum_df.head(top_n_factors)['Factor'].tolist()
        print(f"\n✓ Top {top_n_factors} Momentum Factors: {', '.join(top_factors)}")

        # Factor correlations
        corr_matrix = timing_analyzer.calculate_factor_correlations()

        return {
            'momentum': momentum_df,
            'top_factors': top_factors,
            'correlations': corr_matrix
        }

    def generate_comprehensive_report(self,
                                     portfolio_returns: pd.Series,
                                     portfolio_weights: Optional[np.ndarray] = None,
                                     save_path: str = 'results/factor_analysis') -> str:
        """
        Generate comprehensive factor analysis report.

        Args:
            portfolio_returns: Portfolio return series
            portfolio_weights: Portfolio weights
            save_path: Path to save report

        Returns:
            Report text
        """
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE FACTOR REPORT")
        print(f"{'='*80}\n")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INTEGRATED FACTOR ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 1. Factor Model Results
        if self.beta_factors is None:
            self.construct_style_factors()

        report_lines.append("1. FACTOR MODEL ANALYSIS")
        report_lines.append("-" * 80)

        for model in ['fama_french_3', 'carhart_4']:
            reg_results = self.factor_analyzer.run_factor_regression(
                portfolio_returns, model=model
            )

            report_lines.append(f"\n{model.upper().replace('_', ' ')} MODEL:")
            report_lines.append(f"  Alpha (annualized): {reg_results['alpha_annualized']:.4f}")
            report_lines.append(f"  Alpha t-stat: {reg_results['alpha_tstat']:.2f}")
            report_lines.append(f"  R²: {reg_results['r_squared']:.4f}")
            report_lines.append(f"  Adj. R²: {reg_results['adj_r_squared']:.4f}")
            report_lines.append(f"  Factor Loadings:")
            for factor in reg_results['factors']:
                beta = reg_results['betas'][factor]
                tstat = reg_results['tstat'][factor]
                pval = reg_results['pvalues'][factor]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                report_lines.append(f"    {factor:8s}: {beta:>7.4f} (t={tstat:>6.2f}) {sig}")

        # 2. Return Attribution
        report_lines.append("\n2. RETURN ATTRIBUTION")
        report_lines.append("-" * 80)

        attribution = self.factor_analyzer.attribute_returns(
            portfolio_returns, model='carhart_4'
        )

        report_lines.append(f"\n  Total Return: {attribution['total_return']:.2%}")
        report_lines.append(f"  Explained by Factors: {attribution['explained_return']:.2%}")
        report_lines.append(f"  R²: {attribution['r_squared']:.1%}")
        report_lines.append(f"\n  Factor Contributions:")
        for factor, contrib in attribution['contributions'].items():
            pct = attribution['contribution_pct'][factor]
            report_lines.append(f"    {factor:8s}: {contrib:>7.2%} ({pct:>6.1%})")

        # 3. Risk Decomposition
        if portfolio_weights is not None:
            report_lines.append("\n3. FACTOR RISK DECOMPOSITION")
            report_lines.append("-" * 80)

            risk_results = self.factor_analyzer.calculate_factor_risk(portfolio_weights)

            report_lines.append(f"\n  Total Volatility: {risk_results['total_volatility']:.2%}")
            report_lines.append(f"  Factor Volatility: {risk_results['factor_volatility']:.2%} ({risk_results['factor_contribution_pct']:.1%})")
            report_lines.append(f"  Idiosyncratic Vol: {risk_results['idiosyncratic_volatility']:.2%}")
            report_lines.append(f"\n  Factor Risk Contributions:")
            for factor, risk_info in risk_results['factor_risks'].items():
                report_lines.append(f"    {factor:8s}: {risk_info['pct_contribution']:>6.1%}")

        report_text = "\n".join(report_lines)
        print(report_text)

        # Save report
        os.makedirs(save_path, exist_ok=True)
        report_file = os.path.join(save_path, 'integrated_factor_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Report saved to {report_file}")

        return report_text


def main():
    """Demo of Integrated Factor System."""
    print("=" * 80)
    print("INTEGRATED FACTOR SYSTEM DEMO")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'TLT', 'GLD']

    returns = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.01,
        index=dates,
        columns=tickers
    )

    # Initialize system
    system = IntegratedFactorSystem(returns)

    # 1. Construct style factors
    system.construct_style_factors()

    # 2. Run factor timing
    timing_results = system.run_factor_timing()

    # 3. Create factor-tilted portfolio
    top_factors = timing_results['top_factors']
    weights = system.create_factor_tilted_portfolio(
        tilt_factors=top_factors,
        tilt_strength=1.0,
        base_method='risk_parity'
    )

    # 4. Analyze portfolio
    portfolio_returns = (returns * weights).sum(axis=1)
    system.analyze_portfolio_factors(portfolio_returns, weights)

    # 5. Generate report
    system.generate_comprehensive_report(portfolio_returns, weights)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
