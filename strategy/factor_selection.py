"""
Factor Selection and Importance Analysis

This module identifies the most important and predictive factors from a large
factor library (133 factors) using multiple selection methods:

1. Information Coefficient (IC) Analysis
2. Factor Returns and Sharpe Ratios
3. LASSO/Elastic Net Regularization
4. Random Forest Feature Importance
5. Principal Component Analysis
6. Forward/Backward Selection
7. Factor Decay Analysis
8. Multi-factor IC Analysis

Integrates with:
- Factor Mining System (133 factors)
- Portfolio optimization framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class FactorSelector:
    """
    Comprehensive factor selection and importance ranking.

    Methods:
    1. IC Analysis - Information Coefficient (correlation with future returns)
    2. Return-based - Factor returns and Sharpe ratios
    3. Regression-based - LASSO, Elastic Net, Ridge
    4. Tree-based - Random Forest feature importance
    5. Statistical - F-test, Mutual Information
    6. Sequential - Forward/Backward selection
    """

    def __init__(self,
                 factor_data: pd.DataFrame,
                 returns: pd.DataFrame,
                 forward_periods: List[int] = [1, 5, 10, 20]):
        """
        Initialize factor selector.

        Args:
            factor_data: DataFrame with factors (columns) for each asset-date
                        Expected columns: ['date', 'ticker', 'factor1', 'factor2', ...]
            returns: DataFrame of asset returns (assets as columns, dates as index)
            forward_periods: List of forward-looking periods for IC calculation
        """
        self.factor_data = factor_data
        self.returns = returns
        self.forward_periods = forward_periods

        # Results storage
        self.ic_results = None
        self.importance_scores = {}
        self.selected_factors = {}

    def calculate_ic_analysis(self,
                              method: str = 'spearman') -> pd.DataFrame:
        """
        Calculate Information Coefficient (IC) for all factors.

        IC measures correlation between factor values and future returns.
        High |IC| indicates predictive power.

        Args:
            method: 'spearman' (rank) or 'pearson' (linear)

        Returns:
            DataFrame with IC statistics for each factor
        """
        print(f"\n{'='*80}")
        print("INFORMATION COEFFICIENT (IC) ANALYSIS")
        print(f"{'='*80}\n")

        # Get factor columns (exclude date, ticker, etc.)
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        if len(factor_cols) == 0:
            print("⚠ No factor columns found!")
            return pd.DataFrame()

        print(f"Analyzing {len(factor_cols)} factors...")
        print(f"Forward periods: {self.forward_periods} days")
        print(f"Method: {method}")

        results = []

        for factor in factor_cols:
            factor_stats = {'factor': factor}

            # Calculate IC for each forward period
            for period in self.forward_periods:
                ic_values = []

                # Group by ticker and calculate IC
                for ticker in self.factor_data['ticker'].unique():
                    ticker_data = self.factor_data[self.factor_data['ticker'] == ticker].copy()

                    if ticker not in self.returns.columns:
                        continue

                    # Align factor values with forward returns
                    ticker_data = ticker_data.sort_values('date')
                    ticker_data['forward_return'] = self.returns[ticker].shift(-period)

                    # Drop NaN
                    valid_data = ticker_data[[factor, 'forward_return']].dropna()

                    if len(valid_data) < 30:  # Need minimum observations
                        continue

                    # Calculate correlation
                    if method == 'spearman':
                        ic, _ = stats.spearmanr(valid_data[factor], valid_data['forward_return'])
                    else:
                        ic, _ = stats.pearsonr(valid_data[factor], valid_data['forward_return'])

                    if not np.isnan(ic):
                        ic_values.append(ic)

                if len(ic_values) > 0:
                    # IC statistics
                    mean_ic = np.mean(ic_values)
                    std_ic = np.std(ic_values)
                    ic_ir = mean_ic / std_ic if std_ic > 0 else 0
                    t_stat = np.sqrt(len(ic_values)) * ic_ir

                    factor_stats[f'IC_{period}d'] = mean_ic
                    factor_stats[f'IC_std_{period}d'] = std_ic
                    factor_stats[f'IC_IR_{period}d'] = ic_ir
                    factor_stats[f't_stat_{period}d'] = t_stat
                else:
                    factor_stats[f'IC_{period}d'] = 0
                    factor_stats[f'IC_std_{period}d'] = 0
                    factor_stats[f'IC_IR_{period}d'] = 0
                    factor_stats[f't_stat_{period}d'] = 0

            # Overall IC score (average across periods)
            ic_cols = [f'IC_{p}d' for p in self.forward_periods]
            factor_stats['avg_IC'] = np.mean([factor_stats.get(col, 0) for col in ic_cols])
            factor_stats['avg_IC_IR'] = np.mean([factor_stats.get(f'IC_IR_{p}d', 0) for p in self.forward_periods])

            results.append(factor_stats)

        ic_df = pd.DataFrame(results)
        ic_df = ic_df.sort_values('avg_IC', ascending=False, key=abs)

        self.ic_results = ic_df

        # Print top factors
        print(f"\n✓ IC Analysis Complete")
        print(f"\nTop 10 Factors by |IC|:")
        print(ic_df[['factor', 'avg_IC', 'avg_IC_IR', 't_stat_20d']].head(10).to_string(index=False))

        return ic_df

    def calculate_factor_returns(self,
                                quantile_split: int = 5) -> pd.DataFrame:
        """
        Calculate returns of quantile portfolios for each factor.

        Creates long-short portfolios based on factor values.

        Args:
            quantile_split: Number of quantiles (default 5 = quintiles)

        Returns:
            DataFrame with factor return statistics
        """
        print(f"\n{'='*80}")
        print("FACTOR RETURN ANALYSIS")
        print(f"{'='*80}\n")

        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        results = []

        for factor in factor_cols:
            # Create quantile portfolios
            long_short_returns = []

            for date in self.factor_data['date'].unique():
                date_data = self.factor_data[self.factor_data['date'] == date].copy()

                if len(date_data) < quantile_split:
                    continue

                # Rank by factor value
                date_data['quantile'] = pd.qcut(date_data[factor],
                                                quantile_split,
                                                labels=False,
                                                duplicates='drop')

                # Get returns for next period
                tickers = date_data['ticker'].values
                if date not in self.returns.index:
                    continue

                next_returns = self.returns.loc[date, tickers].values
                date_data['next_return'] = next_returns

                # Long top quantile, short bottom quantile
                top_quantile = date_data[date_data['quantile'] == quantile_split - 1]
                bottom_quantile = date_data[date_data['quantile'] == 0]

                if len(top_quantile) > 0 and len(bottom_quantile) > 0:
                    long_return = top_quantile['next_return'].mean()
                    short_return = bottom_quantile['next_return'].mean()
                    long_short_returns.append(long_return - short_return)

            if len(long_short_returns) > 0:
                returns_series = pd.Series(long_short_returns)

                results.append({
                    'factor': factor,
                    'mean_return': returns_series.mean() * 252,  # Annualized
                    'volatility': returns_series.std() * np.sqrt(252),
                    'sharpe': (returns_series.mean() / returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0,
                    'win_rate': (returns_series > 0).mean(),
                    't_stat': stats.ttest_1samp(returns_series, 0)[0]
                })

        returns_df = pd.DataFrame(results)
        returns_df = returns_df.sort_values('sharpe', ascending=False)

        print(f"\n✓ Factor Return Analysis Complete")
        print(f"\nTop 10 Factors by Sharpe Ratio:")
        print(returns_df.head(10).to_string(index=False))

        self.importance_scores['returns'] = returns_df

        return returns_df

    def lasso_selection(self,
                       target_returns: pd.Series,
                       n_folds: int = 5) -> Dict:
        """
        Use LASSO regression for factor selection.

        LASSO (L1 regularization) automatically selects sparse factors.

        Args:
            target_returns: Target return series to predict
            n_folds: Number of CV folds

        Returns:
            Dictionary with selected factors and coefficients
        """
        print(f"\n{'='*80}")
        print("LASSO FACTOR SELECTION")
        print(f"{'='*80}\n")

        # Prepare data
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        # Create feature matrix
        X_list = []
        y_list = []

        for date in self.factor_data['date'].unique():
            if date not in target_returns.index:
                continue

            date_factors = self.factor_data[self.factor_data['date'] == date][factor_cols].mean()
            X_list.append(date_factors.values)
            y_list.append(target_returns[date])

        X = np.array(X_list)
        y = np.array(y_list)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LASSO with cross-validation
        lasso = LassoCV(cv=n_folds, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)

        # Get non-zero coefficients
        coefs = lasso.coef_
        selected_indices = np.where(np.abs(coefs) > 1e-10)[0]

        selected_factors = [factor_cols[i] for i in selected_indices]
        selected_coefs = coefs[selected_indices]

        # Sort by absolute coefficient
        sorted_idx = np.argsort(np.abs(selected_coefs))[::-1]
        selected_factors = [selected_factors[i] for i in sorted_idx]
        selected_coefs = [selected_coefs[i] for i in sorted_idx]

        print(f"✓ LASSO selected {len(selected_factors)} / {len(factor_cols)} factors")
        print(f"  R² score: {lasso.score(X_scaled, y):.4f}")
        print(f"  Optimal alpha: {lasso.alpha_:.6f}")

        print(f"\nTop 10 Selected Factors:")
        for factor, coef in zip(selected_factors[:10], selected_coefs[:10]):
            print(f"  {factor:30s}: {coef:>8.4f}")

        results = {
            'selected_factors': selected_factors,
            'coefficients': selected_coefs,
            'model': lasso,
            'r_squared': lasso.score(X_scaled, y),
            'alpha': lasso.alpha_
        }

        self.selected_factors['lasso'] = selected_factors
        self.importance_scores['lasso'] = results

        return results

    def random_forest_importance(self,
                                 target_returns: pd.Series,
                                 n_estimators: int = 100,
                                 top_n: int = 20) -> pd.DataFrame:
        """
        Use Random Forest to rank feature importance.

        Args:
            target_returns: Target return series
            n_estimators: Number of trees
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        print(f"\n{'='*80}")
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print(f"{'='*80}\n")

        # Prepare data
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        X_list = []
        y_list = []

        for date in self.factor_data['date'].unique():
            if date not in target_returns.index:
                continue

            date_factors = self.factor_data[self.factor_data['date'] == date][factor_cols].mean()
            X_list.append(date_factors.values)
            y_list.append(target_returns[date])

        X = np.array(X_list)
        y = np.array(y_list)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   random_state=42,
                                   max_depth=5,
                                   min_samples_split=20)
        rf.fit(X, y)

        # Get importances
        importances = rf.feature_importances_

        importance_df = pd.DataFrame({
            'factor': factor_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"✓ Random Forest R² score: {rf.score(X, y):.4f}")
        print(f"\nTop {min(top_n, len(importance_df))} Features by Importance:")
        print(importance_df.head(top_n).to_string(index=False))

        self.selected_factors['random_forest'] = importance_df.head(top_n)['factor'].tolist()
        self.importance_scores['random_forest'] = importance_df

        return importance_df

    def mutual_information_selection(self,
                                    target_returns: pd.Series,
                                    top_n: int = 20) -> pd.DataFrame:
        """
        Calculate mutual information between factors and returns.

        Captures non-linear relationships.

        Args:
            target_returns: Target return series
            top_n: Number of top features

        Returns:
            DataFrame with MI scores
        """
        print(f"\n{'='*80}")
        print("MUTUAL INFORMATION ANALYSIS")
        print(f"{'='*80}\n")

        # Prepare data
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        X_list = []
        y_list = []

        for date in self.factor_data['date'].unique():
            if date not in target_returns.index:
                continue

            date_factors = self.factor_data[self.factor_data['date'] == date][factor_cols].mean()
            X_list.append(date_factors.values)
            y_list.append(target_returns[date])

        X = np.array(X_list)
        y = np.array(y_list)

        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_df = pd.DataFrame({
            'factor': factor_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        print(f"✓ Mutual Information calculated")
        print(f"\nTop {min(top_n, len(mi_df))} Factors by MI Score:")
        print(mi_df.head(top_n).to_string(index=False))

        self.selected_factors['mutual_info'] = mi_df.head(top_n)['factor'].tolist()
        self.importance_scores['mutual_info'] = mi_df

        return mi_df

    def forward_selection(self,
                         target_returns: pd.Series,
                         max_features: int = 10,
                         threshold: float = 0.01) -> List[str]:
        """
        Forward stepwise selection based on R² improvement.

        Args:
            target_returns: Target returns
            max_features: Maximum features to select
            threshold: Minimum R² improvement to add feature

        Returns:
            List of selected factors
        """
        print(f"\n{'='*80}")
        print("FORWARD STEPWISE SELECTION")
        print(f"{'='*80}\n")

        # Prepare data
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in self.factor_data.columns if col not in meta_cols]

        X_list = []
        y_list = []

        for date in self.factor_data['date'].unique():
            if date not in target_returns.index:
                continue

            date_factors = self.factor_data[self.factor_data['date'] == date][factor_cols].mean()
            X_list.append(date_factors.values)
            y_list.append(target_returns[date])

        X = np.array(X_list)
        y = np.array(y_list)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        selected_indices = []
        remaining_indices = list(range(len(factor_cols)))
        current_r2 = 0

        print("Selection progress:")

        for step in range(max_features):
            best_r2 = current_r2
            best_idx = None

            for idx in remaining_indices:
                test_indices = selected_indices + [idx]
                X_subset = X_scaled[:, test_indices]

                model = Ridge(alpha=1.0)
                model.fit(X_subset, y)
                r2 = model.score(X_subset, y)

                if r2 > best_r2:
                    best_r2 = r2
                    best_idx = idx

            if best_idx is None or (best_r2 - current_r2) < threshold:
                break

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            current_r2 = best_r2

            print(f"  Step {step+1}: Added {factor_cols[best_idx]:30s}, R²={current_r2:.4f} (+{best_r2-current_r2:.4f})")

        selected_factors = [factor_cols[i] for i in selected_indices]

        print(f"\n✓ Selected {len(selected_factors)} factors")
        print(f"  Final R²: {current_r2:.4f}")

        self.selected_factors['forward'] = selected_factors

        return selected_factors

    def generate_factor_importance_report(self,
                                         target_returns: pd.Series,
                                         top_n: int = 20) -> pd.DataFrame:
        """
        Comprehensive factor importance analysis using all methods.

        Args:
            target_returns: Target return series (e.g., equal-weight portfolio)
            top_n: Number of top factors to highlight

        Returns:
            DataFrame with aggregated importance rankings
        """
        print(f"\n{'#'*80}")
        print("#" + " "*78 + "#")
        print("#" + " "*18 + "COMPREHENSIVE FACTOR IMPORTANCE ANALYSIS" + " "*19 + "#")
        print("#" + " "*78 + "#")
        print(f"{'#'*80}\n")

        # Run all methods
        ic_df = self.calculate_ic_analysis()
        returns_df = self.calculate_factor_returns()
        lasso_results = self.lasso_selection(target_returns)
        rf_importance = self.random_forest_importance(target_returns, top_n=top_n)
        mi_scores = self.mutual_information_selection(target_returns, top_n=top_n)
        forward_factors = self.forward_selection(target_returns, max_features=top_n)

        # Aggregate rankings
        meta_cols = ['date', 'ticker', 'tic', 'open', 'high', 'low', 'close', 'volume']
        all_factors = [col for col in self.factor_data.columns if col not in meta_cols]

        aggregated = []

        for factor in all_factors:
            scores = {'factor': factor}

            # IC rank
            if ic_df is not None and len(ic_df) > 0:
                ic_rank = ic_df[ic_df['factor'] == factor].index[0] + 1 if factor in ic_df['factor'].values else len(ic_df)
                scores['ic_rank'] = ic_rank
                scores['ic_score'] = ic_df[ic_df['factor'] == factor]['avg_IC'].values[0] if factor in ic_df['factor'].values else 0

            # Returns rank
            if returns_df is not None and len(returns_df) > 0:
                ret_rank = returns_df[returns_df['factor'] == factor].index[0] + 1 if factor in returns_df['factor'].values else len(returns_df)
                scores['returns_rank'] = ret_rank
                scores['sharpe'] = returns_df[returns_df['factor'] == factor]['sharpe'].values[0] if factor in returns_df['factor'].values else 0

            # LASSO
            scores['lasso_selected'] = 1 if factor in lasso_results['selected_factors'] else 0

            # Random Forest rank
            rf_rank = rf_importance[rf_importance['factor'] == factor].index[0] + 1 if factor in rf_importance['factor'].values else len(rf_importance)
            scores['rf_rank'] = rf_rank

            # Mutual Information rank
            mi_rank = mi_scores[mi_scores['factor'] == factor].index[0] + 1 if factor in mi_scores['factor'].values else len(mi_scores)
            scores['mi_rank'] = mi_rank

            # Forward selection
            scores['forward_selected'] = 1 if factor in forward_factors else 0

            # Average rank (lower is better)
            ranks = [scores.get('ic_rank', 999), scores.get('returns_rank', 999),
                    scores.get('rf_rank', 999), scores.get('mi_rank', 999)]
            scores['avg_rank'] = np.mean(ranks)

            # Selection count
            scores['selection_count'] = scores['lasso_selected'] + scores['forward_selected']

            aggregated.append(scores)

        final_df = pd.DataFrame(aggregated).sort_values('avg_rank')

        print(f"\n{'='*80}")
        print("FINAL FACTOR IMPORTANCE RANKINGS")
        print(f"{'='*80}\n")
        print(f"Top {top_n} Most Important Factors:")
        print(final_df[['factor', 'avg_rank', 'ic_score', 'sharpe',
                       'lasso_selected', 'forward_selected', 'selection_count']].head(top_n).to_string(index=False))

        return final_df


def main():
    """Demo of factor selection."""
    print("=" * 80)
    print("FACTOR SELECTION DEMO")
    print("=" * 80)

    # This would be replaced with real factor data from Factor Mining System
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']

    # Simulate factor data
    factor_data_list = []
    for date in dates:
        for ticker in tickers:
            row = {
                'date': date,
                'ticker': ticker,
                'momentum_5d': np.random.randn(),
                'momentum_20d': np.random.randn(),
                'volatility': np.random.rand(),
                'rsi': np.random.rand() * 100,
                'macd': np.random.randn(),
            }
            factor_data_list.append(row)

    factor_data = pd.DataFrame(factor_data_list)

    # Simulate returns
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.01,
        index=dates,
        columns=tickers
    )

    # Initialize selector
    selector = FactorSelector(factor_data, returns)

    # Target returns (equal-weight portfolio)
    target_returns = returns.mean(axis=1)

    # Run comprehensive analysis
    importance_report = selector.generate_factor_importance_report(target_returns, top_n=10)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
