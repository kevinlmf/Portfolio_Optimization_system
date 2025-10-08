"""
Factor Analysis Module

This module provides comprehensive factor analysis for portfolio returns including:
1. Fama-French factor models (3-factor, 5-factor, 6-factor)
2. PCA-based factor extraction
3. Custom factor models
4. Factor exposure analysis
5. Return attribution analysis
6. Risk decomposition

References:
- Fama, E., & French, K. (1993). "Common risk factors in the returns on stocks and bonds"
- Carhart, M. (1997). "On Persistence in Mutual Fund Performance"
- Fama, E., & French, K. (2015). "A five-factor asset pricing model"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FactorAnalyzer:
    """
    Comprehensive factor analysis for portfolio returns.

    Supports multiple factor models:
    - Fama-French 3-factor (Market, SMB, HML)
    - Fama-French 5-factor (adds RMW, CMA)
    - Carhart 4-factor (adds Momentum)
    - PCA-based factors
    - Custom factors
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.03):
        """
        Initialize factor analyzer.

        Args:
            returns: DataFrame of asset returns (assets as columns)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Computed factors
        self.factors = None
        self.factor_loadings = None
        self.factor_returns = None
        self.pca_model = None

        # Analysis results
        self.regression_results = {}
        self.attribution_results = {}

    def construct_market_factors(self,
                                 market_proxy: Optional[pd.Series] = None,
                                 include_size: bool = True,
                                 include_value: bool = True,
                                 include_momentum: bool = True,
                                 include_quality: bool = True,
                                 include_investment: bool = True) -> pd.DataFrame:
        """
        Construct market-based factors (Fama-French style).

        Args:
            market_proxy: Market return series (if None, uses equal-weighted portfolio)
            include_size: Include SMB (Small Minus Big) factor
            include_value: Include HML (High Minus Low) factor
            include_momentum: Include MOM (Momentum) factor
            include_quality: Include RMW (Robust Minus Weak) factor
            include_investment: Include CMA (Conservative Minus Aggressive) factor

        Returns:
            DataFrame of factor returns
        """
        factors_df = pd.DataFrame(index=self.returns.index)

        # 1. Market Factor (MKT-RF)
        if market_proxy is not None:
            factors_df['MKT'] = market_proxy - self.daily_rf
        else:
            # Use equal-weighted market portfolio
            factors_df['MKT'] = self.returns.mean(axis=1) - self.daily_rf

        # 2. SMB Factor (Size)
        if include_size:
            factors_df['SMB'] = self._construct_smb_factor()

        # 3. HML Factor (Value)
        if include_value:
            factors_df['HML'] = self._construct_hml_factor()

        # 4. Momentum Factor
        if include_momentum:
            factors_df['MOM'] = self._construct_momentum_factor()

        # 5. Quality Factor (RMW - Robust Minus Weak)
        if include_quality:
            factors_df['RMW'] = self._construct_quality_factor()

        # 6. Investment Factor (CMA - Conservative Minus Aggressive)
        if include_investment:
            factors_df['CMA'] = self._construct_investment_factor()

        self.factors = factors_df
        return factors_df

    def _construct_smb_factor(self) -> pd.Series:
        """
        Construct SMB (Small Minus Big) factor using volatility as proxy for size.
        Higher volatility assets treated as "smaller" stocks.
        """
        # Use rolling volatility as size proxy (inverse relationship)
        rolling_vol = self.returns.rolling(60).std()

        smb_returns = []
        for date in self.returns.index:
            if date not in rolling_vol.index:
                smb_returns.append(0)
                continue

            vols = rolling_vol.loc[date].dropna()
            if len(vols) < 2:
                smb_returns.append(0)
                continue

            # Split into small (high vol) and big (low vol)
            median_vol = vols.median()
            small_stocks = vols[vols >= median_vol].index
            big_stocks = vols[vols < median_vol].index

            if len(small_stocks) > 0 and len(big_stocks) > 0:
                small_return = self.returns.loc[date, small_stocks].mean()
                big_return = self.returns.loc[date, big_stocks].mean()
                smb_returns.append(small_return - big_return)
            else:
                smb_returns.append(0)

        return pd.Series(smb_returns, index=self.returns.index)

    def _construct_hml_factor(self) -> pd.Series:
        """
        Construct HML (High Minus Low) factor using momentum as value proxy.
        Negative momentum (losers) = value, positive momentum (winners) = growth.
        """
        # Use past returns as value proxy (contrarian)
        past_returns = self.returns.rolling(120).mean()

        hml_returns = []
        for date in self.returns.index:
            if date not in past_returns.index:
                hml_returns.append(0)
                continue

            past_rets = past_returns.loc[date].dropna()
            if len(past_rets) < 2:
                hml_returns.append(0)
                continue

            # Split into value (low past returns) and growth (high past returns)
            median_ret = past_rets.median()
            value_stocks = past_rets[past_rets <= median_ret].index
            growth_stocks = past_rets[past_rets > median_ret].index

            if len(value_stocks) > 0 and len(growth_stocks) > 0:
                value_return = self.returns.loc[date, value_stocks].mean()
                growth_return = self.returns.loc[date, growth_stocks].mean()
                hml_returns.append(value_return - growth_return)
            else:
                hml_returns.append(0)

        return pd.Series(hml_returns, index=self.returns.index)

    def _construct_momentum_factor(self, lookback: int = 60) -> pd.Series:
        """
        Construct momentum factor (winners minus losers).
        """
        past_returns = self.returns.rolling(lookback).mean()

        mom_returns = []
        for date in self.returns.index:
            if date not in past_returns.index:
                mom_returns.append(0)
                continue

            past_rets = past_returns.loc[date].dropna()
            if len(past_rets) < 2:
                mom_returns.append(0)
                continue

            # Split into winners and losers
            median_ret = past_rets.median()
            winners = past_rets[past_rets >= median_ret].index
            losers = past_rets[past_rets < median_ret].index

            if len(winners) > 0 and len(losers) > 0:
                winner_return = self.returns.loc[date, winners].mean()
                loser_return = self.returns.loc[date, losers].mean()
                mom_returns.append(winner_return - loser_return)
            else:
                mom_returns.append(0)

        return pd.Series(mom_returns, index=self.returns.index)

    def _construct_quality_factor(self) -> pd.Series:
        """
        Construct quality factor using Sharpe ratio as quality proxy.
        High Sharpe = robust, low Sharpe = weak.
        """
        rolling_sharpe = self.returns.rolling(120).apply(
            lambda x: x.mean() / x.std() if x.std() > 0 else 0
        )

        quality_returns = []
        for date in self.returns.index:
            if date not in rolling_sharpe.index:
                quality_returns.append(0)
                continue

            sharpes = rolling_sharpe.loc[date].dropna()
            if len(sharpes) < 2:
                quality_returns.append(0)
                continue

            # Split into robust (high Sharpe) and weak (low Sharpe)
            median_sharpe = sharpes.median()
            robust_stocks = sharpes[sharpes >= median_sharpe].index
            weak_stocks = sharpes[sharpes < median_sharpe].index

            if len(robust_stocks) > 0 and len(weak_stocks) > 0:
                robust_return = self.returns.loc[date, robust_stocks].mean()
                weak_return = self.returns.loc[date, weak_stocks].mean()
                quality_returns.append(robust_return - weak_return)
            else:
                quality_returns.append(0)

        return pd.Series(quality_returns, index=self.returns.index)

    def _construct_investment_factor(self) -> pd.Series:
        """
        Construct investment factor using volatility trends.
        Decreasing vol = conservative, increasing vol = aggressive.
        """
        rolling_vol = self.returns.rolling(60).std()
        vol_change = rolling_vol.pct_change(20)

        investment_returns = []
        for date in self.returns.index:
            if date not in vol_change.index:
                investment_returns.append(0)
                continue

            vol_changes = vol_change.loc[date].dropna()
            if len(vol_changes) < 2:
                investment_returns.append(0)
                continue

            # Split into conservative (decreasing vol) and aggressive (increasing vol)
            median_change = vol_changes.median()
            conservative_stocks = vol_changes[vol_changes <= median_change].index
            aggressive_stocks = vol_changes[vol_changes > median_change].index

            if len(conservative_stocks) > 0 and len(aggressive_stocks) > 0:
                conservative_return = self.returns.loc[date, conservative_stocks].mean()
                aggressive_return = self.returns.loc[date, aggressive_stocks].mean()
                investment_returns.append(conservative_return - aggressive_return)
            else:
                investment_returns.append(0)

        return pd.Series(investment_returns, index=self.returns.index)

    def extract_pca_factors(self,
                           n_factors: int = 5,
                           standardize: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract statistical factors using PCA.

        Args:
            n_factors: Number of principal components to extract
            standardize: Whether to standardize returns before PCA

        Returns:
            Tuple of (factor_returns DataFrame, factor_loadings array)
        """
        # Prepare data
        returns_clean = self.returns.dropna()

        if standardize:
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean)
        else:
            returns_scaled = returns_clean.values

        # Fit PCA
        pca = PCA(n_components=n_factors)
        factor_scores = pca.fit_transform(returns_scaled)

        # Create factor returns DataFrame
        factor_names = [f'PC{i+1}' for i in range(n_factors)]
        factor_returns = pd.DataFrame(
            factor_scores,
            index=returns_clean.index,
            columns=factor_names
        )

        # Store results
        self.pca_model = pca
        self.factor_loadings = pca.components_.T  # Assets x Factors
        self.factor_returns = factor_returns

        # Print explained variance
        print(f"\nPCA Factor Analysis:")
        print(f"  Total factors: {n_factors}")
        print(f"  Cumulative explained variance:")
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        for i, var in enumerate(cum_var):
            print(f"    PC{i+1}: {var:.2%}")

        return factor_returns, self.factor_loadings

    def run_factor_regression(self,
                             portfolio_returns: pd.Series,
                             factors: Optional[pd.DataFrame] = None,
                             model: str = 'fama_french_3') -> Dict:
        """
        Run factor regression analysis.

        Args:
            portfolio_returns: Portfolio return series
            factors: Factor returns DataFrame (if None, uses self.factors)
            model: Factor model to use ('fama_french_3', 'fama_french_5',
                   'carhart_4', 'pca', 'custom')

        Returns:
            Dictionary with regression results
        """
        if factors is None:
            if self.factors is None:
                # Construct default 3-factor model
                self.construct_market_factors(
                    include_size=True,
                    include_value=True,
                    include_momentum=False,
                    include_quality=False,
                    include_investment=False
                )
            factors = self.factors

        # Select factors based on model
        if model == 'fama_french_3':
            selected_factors = ['MKT', 'SMB', 'HML']
        elif model == 'fama_french_5':
            selected_factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        elif model == 'carhart_4':
            selected_factors = ['MKT', 'SMB', 'HML', 'MOM']
        elif model == 'pca':
            selected_factors = [col for col in factors.columns if col.startswith('PC')]
        else:
            selected_factors = factors.columns.tolist()

        # Keep only available factors
        available_factors = [f for f in selected_factors if f in factors.columns]

        if len(available_factors) == 0:
            raise ValueError(f"No factors available for model {model}")

        # Align data
        common_idx = portfolio_returns.index.intersection(factors.index)
        y = portfolio_returns.loc[common_idx].values
        X = factors.loc[common_idx, available_factors].values

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # Run regression
        betas, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

        # Calculate statistics
        alpha = betas[0]
        factor_betas = betas[1:]

        # Predicted returns
        y_pred = X_with_const @ betas

        # R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # Adjusted R-squared
        n = len(y)
        k = len(available_factors)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors
        mse = ss_residual / (n - k - 1)
        var_betas = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
        se_betas = np.sqrt(var_betas)

        # T-statistics
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

        # Store results
        results = {
            'model': model,
            'factors': available_factors,
            'alpha': alpha,
            'alpha_annualized': alpha * 252,
            'alpha_tstat': t_stats[0],
            'alpha_pvalue': p_values[0],
            'betas': dict(zip(available_factors, factor_betas)),
            'tstat': dict(zip(available_factors, t_stats[1:])),
            'pvalues': dict(zip(available_factors, p_values[1:])),
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'residual_volatility': np.std(y - y_pred) * np.sqrt(252),
            'observations': n
        }

        self.regression_results[model] = results
        return results

    def calculate_factor_exposures(self,
                                   portfolio_weights: np.ndarray,
                                   factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate portfolio's factor exposures given weights.

        Args:
            portfolio_weights: Array of portfolio weights
            factors: Factor returns (if None, uses self.factors)

        Returns:
            DataFrame of factor exposures
        """
        if factors is None:
            factors = self.factors

        if factors is None:
            raise ValueError("Must construct factors first")

        # Run regression for each asset
        exposures = []

        for i, asset in enumerate(self.returns.columns):
            asset_returns = self.returns[asset]

            # Align data
            common_idx = asset_returns.index.intersection(factors.index)
            y = asset_returns.loc[common_idx].values
            X = factors.loc[common_idx].values

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # Regression
            betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

            exposures.append(betas[1:])  # Exclude alpha

        exposures = np.array(exposures)

        # Portfolio exposures = weighted average
        portfolio_exposures = portfolio_weights @ exposures

        # Create results DataFrame
        exposure_df = pd.DataFrame({
            'Factor': factors.columns,
            'Exposure': portfolio_exposures,
            'Absolute_Exposure': np.abs(portfolio_exposures)
        })

        return exposure_df.sort_values('Absolute_Exposure', ascending=False)

    def attribute_returns(self,
                         portfolio_returns: pd.Series,
                         model: str = 'fama_french_3') -> Dict:
        """
        Decompose portfolio returns into factor contributions.

        Args:
            portfolio_returns: Portfolio return series
            model: Factor model to use

        Returns:
            Dictionary with attribution results
        """
        # Run regression if not already done
        if model not in self.regression_results:
            self.run_factor_regression(portfolio_returns, model=model)

        results = self.regression_results[model]

        # Calculate factor contributions
        factors = self.factors if self.factors is not None else self.factor_returns
        available_factors = results['factors']

        # Align data
        common_idx = portfolio_returns.index.intersection(factors.index)
        factor_values = factors.loc[common_idx, available_factors]

        # Calculate contributions
        contributions = {}
        for factor in available_factors:
            beta = results['betas'][factor]
            factor_return = factor_values[factor].mean() * 252  # Annualized
            contribution = beta * factor_return
            contributions[factor] = contribution

        # Alpha contribution
        contributions['Alpha'] = results['alpha_annualized']

        # Residual (unexplained)
        total_return = portfolio_returns.loc[common_idx].mean() * 252
        explained_return = sum(contributions.values())
        contributions['Residual'] = total_return - explained_return

        # Create summary
        attribution = {
            'total_return': total_return,
            'explained_return': explained_return,
            'r_squared': results['r_squared'],
            'contributions': contributions,
            'contribution_pct': {
                k: v / total_return if total_return != 0 else 0
                for k, v in contributions.items()
            }
        }

        self.attribution_results[model] = attribution
        return attribution

    def calculate_factor_risk(self,
                             portfolio_weights: np.ndarray,
                             factors: Optional[pd.DataFrame] = None) -> Dict:
        """
        Decompose portfolio risk into factor contributions.

        Args:
            portfolio_weights: Array of portfolio weights
            factors: Factor returns

        Returns:
            Dictionary with risk decomposition
        """
        if factors is None:
            factors = self.factors

        # Calculate factor exposures
        exposure_df = self.calculate_factor_exposures(portfolio_weights, factors)
        exposures = exposure_df['Exposure'].values

        # Factor covariance matrix
        factor_cov = factors.cov() * 252  # Annualized

        # Factor risk contribution
        factor_variance = exposures @ factor_cov @ exposures
        factor_vol = np.sqrt(factor_variance)

        # Individual factor risks
        factor_risks = {}
        for i, factor in enumerate(factors.columns):
            # Marginal contribution to risk
            marginal = (factor_cov @ exposures)[i]
            contribution = exposures[i] * marginal
            factor_risks[factor] = {
                'exposure': exposures[i],
                'volatility': np.sqrt(factor_cov.iloc[i, i]),
                'contribution': contribution,
                'pct_contribution': contribution / factor_variance if factor_variance > 0 else 0
            }

        # Portfolio total risk
        portfolio_returns = self.returns @ portfolio_weights
        total_variance = np.var(portfolio_returns) * 252
        total_vol = np.sqrt(total_variance)

        # Idiosyncratic risk
        idiosyncratic_variance = total_variance - factor_variance
        idiosyncratic_vol = np.sqrt(max(0, idiosyncratic_variance))

        return {
            'total_volatility': total_vol,
            'factor_volatility': factor_vol,
            'idiosyncratic_volatility': idiosyncratic_vol,
            'factor_contribution_pct': factor_variance / total_variance if total_variance > 0 else 0,
            'factor_risks': factor_risks
        }

    def generate_factor_report(self,
                              portfolio_returns: pd.Series,
                              portfolio_weights: Optional[np.ndarray] = None,
                              models: List[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive factor analysis report.

        Args:
            portfolio_returns: Portfolio return series
            portfolio_weights: Portfolio weights (optional)
            models: List of models to test (if None, tests all standard models)

        Returns:
            DataFrame with comprehensive factor analysis
        """
        if models is None:
            models = ['fama_french_3', 'carhart_4']

        print(f"\n{'='*80}")
        print("COMPREHENSIVE FACTOR ANALYSIS REPORT")
        print(f"{'='*80}\n")

        report_data = []

        for model in models:
            print(f"\n--- {model.upper().replace('_', ' ')} MODEL ---")

            try:
                # Run regression
                reg_results = self.run_factor_regression(portfolio_returns, model=model)

                print(f"\nAlpha: {reg_results['alpha_annualized']:.4f} (t={reg_results['alpha_tstat']:.2f}, p={reg_results['alpha_pvalue']:.4f})")
                print(f"R²: {reg_results['r_squared']:.4f}, Adj. R²: {reg_results['adj_r_squared']:.4f}")
                print(f"Residual Vol: {reg_results['residual_volatility']:.2%}")

                print(f"\nFactor Loadings:")
                for factor in reg_results['factors']:
                    beta = reg_results['betas'][factor]
                    t = reg_results['tstat'][factor]
                    p = reg_results['pvalues'][factor]
                    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
                    print(f"  {factor:8s}: {beta:>7.4f}  (t={t:>6.2f}) {sig}")

                # Attribution
                attribution = self.attribute_returns(portfolio_returns, model=model)
                print(f"\nReturn Attribution:")
                print(f"  Total Return: {attribution['total_return']:.2%}")
                print(f"  Explained:    {attribution['explained_return']:.2%} ({attribution['r_squared']:.1%})")
                for factor, contrib in attribution['contributions'].items():
                    pct = attribution['contribution_pct'][factor]
                    print(f"    {factor:8s}: {contrib:>7.2%} ({pct:>6.1%})")

                # Add to report
                report_data.append({
                    'Model': model,
                    'Alpha (ann.)': f"{reg_results['alpha_annualized']:.4f}",
                    'Alpha t-stat': f"{reg_results['alpha_tstat']:.2f}",
                    'R²': f"{reg_results['r_squared']:.4f}",
                    'Adj. R²': f"{reg_results['adj_r_squared']:.4f}",
                    'Residual Vol': f"{reg_results['residual_volatility']:.2%}"
                })

            except Exception as e:
                print(f"Error analyzing {model}: {e}")

        # Risk decomposition if weights provided
        if portfolio_weights is not None and self.factors is not None:
            print(f"\n{'='*80}")
            print("FACTOR RISK DECOMPOSITION")
            print(f"{'='*80}\n")

            risk_results = self.calculate_factor_risk(portfolio_weights)

            print(f"Total Portfolio Volatility: {risk_results['total_volatility']:.2%}")
            print(f"  Factor Volatility:        {risk_results['factor_volatility']:.2%} ({risk_results['factor_contribution_pct']:.1%})")
            print(f"  Idiosyncratic Volatility: {risk_results['idiosyncratic_volatility']:.2%}")

            print(f"\nFactor Risk Contributions:")
            for factor, risk_info in risk_results['factor_risks'].items():
                print(f"  {factor:8s}: Exposure={risk_info['exposure']:>7.4f}, "
                      f"Risk Contrib={risk_info['pct_contribution']:>6.1%}")

        return pd.DataFrame(report_data)


class FactorTimingAnalyzer:
    """
    Analyze factor timing and regime-dependent factor performance.
    """

    def __init__(self, factors: pd.DataFrame):
        """
        Initialize factor timing analyzer.

        Args:
            factors: DataFrame of factor returns
        """
        self.factors = factors

    def analyze_factor_momentum(self, lookback: int = 60) -> pd.DataFrame:
        """
        Analyze momentum in factor returns.

        Args:
            lookback: Lookback period for momentum calculation

        Returns:
            DataFrame with factor momentum statistics
        """
        momentum = self.factors.rolling(lookback).mean()

        results = []
        for factor in self.factors.columns:
            current_momentum = momentum[factor].iloc[-1]
            avg_momentum = momentum[factor].mean()

            results.append({
                'Factor': factor,
                'Current_Momentum': current_momentum * 252,
                'Avg_Momentum': avg_momentum * 252,
                'Momentum_Percentile': stats.percentileofscore(momentum[factor].dropna(), current_momentum) / 100
            })

        return pd.DataFrame(results).sort_values('Current_Momentum', ascending=False)

    def calculate_factor_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.

        Returns:
            Correlation matrix
        """
        return self.factors.corr()

    def identify_factor_regimes(self, n_regimes: int = 3) -> Dict:
        """
        Identify different factor regimes using clustering.

        Args:
            n_regimes: Number of regimes to identify

        Returns:
            Dictionary with regime information
        """
        from sklearn.cluster import KMeans

        # Standardize factors
        scaler = StandardScaler()
        factors_scaled = scaler.fit_transform(self.factors.dropna())

        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(factors_scaled)

        # Analyze each regime
        regime_profiles = []
        for i in range(n_regimes):
            regime_mask = regimes == i
            regime_factors = self.factors.iloc[regime_mask]

            profile = {
                'Regime': f'Regime {i+1}',
                'Frequency': regime_mask.sum() / len(regimes),
                'Avg_Returns': regime_factors.mean().to_dict()
            }
            regime_profiles.append(profile)

        return {
            'regimes': regimes,
            'profiles': regime_profiles,
            'model': kmeans
        }
