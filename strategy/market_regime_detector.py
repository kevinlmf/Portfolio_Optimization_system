"""
Market Regime Detection Module

This module identifies different market regimes and conditions to help select
the most appropriate portfolio optimization method.

Market Regimes:
- Bull Market: Sustained upward trend with low volatility
- Bear Market: Sustained downward trend
- Sideways/Ranging: No clear trend, oscillating
- High Volatility: Large price swings regardless of direction
- Crisis: Extreme volatility with negative returns
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    """
    Detects current market regime using multiple indicators and methods.

    Methods:
    1. Trend-based detection (moving averages, linear regression)
    2. Volatility-based detection (rolling std, GARCH-like)
    3. Hidden Markov Model / Gaussian Mixture Model
    4. Multi-indicator scoring system
    """

    def __init__(self,
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 vol_window: int = 20):
        """
        Initialize the market regime detector.

        Args:
            lookback_short: Short-term lookback period (days/periods)
            lookback_long: Long-term lookback period (days/periods)
            vol_window: Window for volatility calculation
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.vol_window = vol_window

        self.current_regime = None
        self.regime_scores = {}
        self.indicators = {}

    def detect_regime(self,
                      prices: pd.Series,
                      volume: Optional[pd.Series] = None) -> Dict:
        """
        Detect current market regime using multiple methods.

        Args:
            prices: Price series (can be index or portfolio values)
            volume: Optional volume series

        Returns:
            Dictionary containing:
                - regime: Detected regime name
                - confidence: Confidence score (0-1)
                - indicators: Dictionary of individual indicators
                - regime_scores: Scores for each possible regime
        """
        if len(prices) < self.lookback_long:
            raise ValueError(f"Need at least {self.lookback_long} periods of data")

        # Calculate returns
        returns = prices.pct_change().dropna()

        # 1. Trend indicators
        trend_indicators = self._calculate_trend_indicators(prices, returns)

        # 2. Volatility indicators
        vol_indicators = self._calculate_volatility_indicators(returns)

        # 3. Momentum indicators
        momentum_indicators = self._calculate_momentum_indicators(prices, returns)

        # 4. Distribution indicators
        dist_indicators = self._calculate_distribution_indicators(returns)

        # Combine all indicators
        self.indicators = {
            **trend_indicators,
            **vol_indicators,
            **momentum_indicators,
            **dist_indicators
        }

        # Score each regime
        regime_scores = self._score_regimes(self.indicators)
        self.regime_scores = regime_scores

        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[primary_regime]

        self.current_regime = primary_regime

        return {
            'regime': primary_regime,
            'confidence': confidence,
            'indicators': self.indicators,
            'regime_scores': regime_scores,
            'recommendation': self._get_regime_recommendation(primary_regime, confidence)
        }

    def _calculate_trend_indicators(self,
                                   prices: pd.Series,
                                   returns: pd.Series) -> Dict:
        """Calculate trend-based indicators."""
        indicators = {}

        # Moving averages
        ma_short = prices.rolling(self.lookback_short).mean().iloc[-1]
        ma_long = prices.rolling(self.lookback_long).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # Trend direction
        indicators['ma_trend'] = 1 if ma_short > ma_long else -1
        indicators['price_vs_ma_short'] = (current_price - ma_short) / ma_short
        indicators['price_vs_ma_long'] = (current_price - ma_long) / ma_long

        # Linear regression trend
        recent_prices = prices.iloc[-self.lookback_short:].values
        x = np.arange(len(recent_prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, recent_prices)

        indicators['trend_slope'] = slope / recent_prices[0]  # Normalized slope
        indicators['trend_r_squared'] = r_value ** 2

        # Recent performance
        indicators['return_short'] = prices.iloc[-1] / prices.iloc[-self.lookback_short] - 1
        indicators['return_long'] = prices.iloc[-1] / prices.iloc[-self.lookback_long] - 1

        return indicators

    def _calculate_volatility_indicators(self, returns: pd.Series) -> Dict:
        """Calculate volatility-based indicators."""
        indicators = {}

        # Rolling volatility
        vol_short = returns.iloc[-self.vol_window:].std()
        vol_long = returns.iloc[-self.lookback_long:].std()

        indicators['volatility_current'] = vol_short * np.sqrt(252)  # Annualized
        indicators['volatility_long'] = vol_long * np.sqrt(252)
        indicators['volatility_ratio'] = vol_short / vol_long if vol_long > 0 else 1.0

        # Volatility trend
        vol_series = returns.rolling(self.vol_window).std()
        vol_trend = vol_series.iloc[-self.lookback_short:].mean() / vol_series.iloc[-self.lookback_long:-self.lookback_short].mean()
        indicators['volatility_trend'] = vol_trend

        # Extreme moves
        recent_returns = returns.iloc[-self.lookback_short:]
        indicators['extreme_moves_pct'] = (np.abs(recent_returns) > 2 * vol_short).sum() / len(recent_returns)

        # Volatility clustering (GARCH-like)
        squared_returns = returns.iloc[-self.lookback_long:] ** 2
        acf_lag1 = squared_returns.autocorr(lag=1)
        indicators['volatility_clustering'] = acf_lag1 if not np.isnan(acf_lag1) else 0

        return indicators

    def _calculate_momentum_indicators(self,
                                      prices: pd.Series,
                                      returns: pd.Series) -> Dict:
        """Calculate momentum-based indicators."""
        indicators = {}

        # RSI-like indicator
        recent_returns = returns.iloc[-self.lookback_short:]
        up_days = (recent_returns > 0).sum()
        down_days = (recent_returns < 0).sum()
        indicators['momentum_direction'] = (up_days - down_days) / len(recent_returns)

        # Winning/Losing streaks
        signs = np.sign(recent_returns)
        changes = np.diff(np.concatenate([[0], signs]))
        streaks = np.split(signs, np.where(changes != 0)[0])
        max_streak = max([len(s) for s in streaks]) if streaks else 0
        indicators['max_streak'] = max_streak / len(recent_returns)

        # Return consistency
        indicators['return_consistency'] = 1 - returns.iloc[-self.lookback_short:].std() / (np.abs(returns.iloc[-self.lookback_short:].mean()) + 1e-10)

        return indicators

    def _calculate_distribution_indicators(self, returns: pd.Series) -> Dict:
        """Calculate distribution-based indicators."""
        indicators = {}

        recent_returns = returns.iloc[-self.lookback_long:]

        # Skewness and Kurtosis
        indicators['skewness'] = stats.skew(recent_returns)
        indicators['kurtosis'] = stats.kurtosis(recent_returns)

        # Tail risk
        var_95 = np.percentile(recent_returns, 5)
        cvar_95 = recent_returns[recent_returns <= var_95].mean()
        indicators['cvar_95'] = cvar_95

        # Normality test
        _, p_value = stats.normaltest(recent_returns)
        indicators['normality_p_value'] = p_value

        return indicators

    def _score_regimes(self, indicators: Dict) -> Dict[str, float]:
        """
        Score each possible regime based on indicators.
        Returns normalized scores (sum to 1).
        """
        scores = {
            'bull_market': 0.0,
            'bear_market': 0.0,
            'sideways': 0.0,
            'high_volatility': 0.0,
            'crisis': 0.0
        }

        # Bull Market indicators
        if indicators['ma_trend'] > 0:
            scores['bull_market'] += 2.0
        if indicators['return_short'] > 0.05:
            scores['bull_market'] += 2.0
        if indicators['trend_slope'] > 0:
            scores['bull_market'] += 1.5
        if indicators['volatility_current'] < 0.20:  # Low volatility
            scores['bull_market'] += 1.5
        if indicators['momentum_direction'] > 0.2:
            scores['bull_market'] += 1.0

        # Bear Market indicators
        if indicators['ma_trend'] < 0:
            scores['bear_market'] += 2.0
        if indicators['return_short'] < -0.05:
            scores['bear_market'] += 2.0
        if indicators['trend_slope'] < 0:
            scores['bear_market'] += 1.5
        if indicators['momentum_direction'] < -0.2:
            scores['bear_market'] += 1.0
        if indicators['cvar_95'] < -0.02:
            scores['bear_market'] += 1.5

        # Sideways Market indicators
        if abs(indicators['return_short']) < 0.03:
            scores['sideways'] += 2.0
        if indicators['trend_r_squared'] < 0.3:
            scores['sideways'] += 2.0
        if 0.8 < indicators['volatility_ratio'] < 1.2:
            scores['sideways'] += 1.5
        if abs(indicators['momentum_direction']) < 0.1:
            scores['sideways'] += 1.5

        # High Volatility indicators
        if indicators['volatility_current'] > 0.25:
            scores['high_volatility'] += 2.5
        if indicators['volatility_ratio'] > 1.3:
            scores['high_volatility'] += 2.0
        if indicators['extreme_moves_pct'] > 0.15:
            scores['high_volatility'] += 2.0
        if indicators['volatility_clustering'] > 0.3:
            scores['high_volatility'] += 1.5

        # Crisis indicators
        if indicators['volatility_current'] > 0.35:
            scores['crisis'] += 2.5
        if indicators['return_short'] < -0.10:
            scores['crisis'] += 2.5
        if indicators['cvar_95'] < -0.03:
            scores['crisis'] += 2.0
        if indicators['extreme_moves_pct'] > 0.25:
            scores['crisis'] += 2.0
        if indicators['kurtosis'] > 3:  # Fat tails
            scores['crisis'] += 1.5

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            # Default to equal probabilities
            scores = {k: 0.2 for k in scores.keys()}

        return scores

    def _get_regime_recommendation(self, regime: str, confidence: float) -> str:
        """Get recommendation text based on detected regime."""
        recommendations = {
            'bull_market': (
                "Bull Market Detected - Recommendations:\n"
                "- Focus on momentum and growth strategies\n"
                "- Can increase portfolio concentration\n"
                "- Suitable for aggressive optimization (Max Sharpe)\n"
                "- Sparse portfolios (mSSRM-PGA) may capture best performers"
            ),
            'bear_market': (
                "Bear Market Detected - Recommendations:\n"
                "- Emphasize risk management and preservation\n"
                "- Increase diversification\n"
                "- Consider defensive assets and hedging\n"
                "- Minimum Variance or Risk Parity methods recommended\n"
                "- Avoid highly concentrated sparse portfolios"
            ),
            'sideways': (
                "Sideways Market Detected - Recommendations:\n"
                "- Range-bound strategies may work well\n"
                "- Mean reversion and factor rotation\n"
                "- Balanced risk-return optimization\n"
                "- Risk Parity for stable allocation\n"
                "- Moderate sparsity levels (15-20 assets)"
            ),
            'high_volatility': (
                "High Volatility Detected - Recommendations:\n"
                "- Reduce position sizes and increase cash\n"
                "- Strong diversification essential\n"
                "- CVaR or robust optimization methods\n"
                "- Avoid sparse portfolios - need broad diversification\n"
                "- Frequent rebalancing may be needed"
            ),
            'crisis': (
                "Crisis Mode Detected - Recommendations:\n"
                "- Capital preservation is priority\n"
                "- Maximum diversification\n"
                "- Consider defensive assets (bonds, gold)\n"
                "- Minimum Variance strongly recommended\n"
                "- Avoid optimization - equal weight may be safer\n"
                "- Monitor positions frequently"
            )
        }

        rec = recommendations.get(regime, "Unknown regime")
        return f"{rec}\n\nConfidence: {confidence:.1%}"

    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of regime scores and indicators."""
        if not self.regime_scores:
            raise ValueError("Must call detect_regime() first")

        summary = []
        summary.append({
            'Metric': 'Current Regime',
            'Value': self.current_regime,
            'Description': 'Primary detected market regime'
        })

        for regime, score in sorted(self.regime_scores.items(),
                                   key=lambda x: x[1], reverse=True):
            summary.append({
                'Metric': f'Score: {regime}',
                'Value': f'{score:.1%}',
                'Description': 'Probability/confidence of regime'
            })

        # Key indicators
        key_indicators = [
            ('return_short', 'Short-term return'),
            ('volatility_current', 'Current volatility (annualized)'),
            ('trend_slope', 'Trend strength'),
            ('ma_trend', 'MA trend direction')
        ]

        for key, desc in key_indicators:
            if key in self.indicators:
                val = self.indicators[key]
                if isinstance(val, float):
                    val = f'{val:.4f}'
                summary.append({
                    'Metric': desc,
                    'Value': val,
                    'Description': f'Indicator: {key}'
                })

        return pd.DataFrame(summary)


class AssetConfigurationAnalyzer:
    """
    Analyzes portfolio asset configuration characteristics.

    Analyzes:
    - Number of assets
    - Correlation structure
    - Sector/industry concentration
    - Asset type diversity
    - Risk characteristics
    """

    def __init__(self):
        """Initialize the asset configuration analyzer."""
        self.config_features = {}

    def analyze(self,
                returns: pd.DataFrame,
                asset_info: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze asset configuration characteristics.

        Args:
            returns: DataFrame of asset returns (assets as columns)
            asset_info: Optional DataFrame with asset metadata
                       (sectors, asset_types, etc.)

        Returns:
            Dictionary of configuration features
        """
        n_assets = returns.shape[1]

        features = {
            'n_assets': n_assets,
            'asset_size_category': self._categorize_size(n_assets)
        }

        # Correlation analysis
        corr_matrix = returns.corr()
        features.update(self._analyze_correlations(corr_matrix))

        # Diversity analysis
        if asset_info is not None:
            features.update(self._analyze_diversity(asset_info))

        # Risk characteristics
        features.update(self._analyze_risk_chars(returns))

        self.config_features = features
        return features

    def _categorize_size(self, n: int) -> str:
        """Categorize portfolio size."""
        if n <= 10:
            return 'small'
        elif n <= 30:
            return 'medium'
        elif n <= 75:
            return 'large'
        else:
            return 'very_large'

    def _analyze_correlations(self, corr_matrix: pd.DataFrame) -> Dict:
        """Analyze correlation structure."""
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack().values

        return {
            'avg_correlation': np.mean(correlations),
            'max_correlation': np.max(correlations),
            'min_correlation': np.min(correlations),
            'correlation_std': np.std(correlations),
            'high_corr_pairs_pct': np.sum(correlations > 0.7) / len(correlations),
            'low_corr_pairs_pct': np.sum(correlations < 0.3) / len(correlations)
        }

    def _analyze_diversity(self, asset_info: pd.DataFrame) -> Dict:
        """Analyze asset diversity."""
        features = {}

        if 'sector' in asset_info.columns:
            sector_counts = asset_info['sector'].value_counts()
            features['n_sectors'] = len(sector_counts)
            features['max_sector_concentration'] = sector_counts.iloc[0] / len(asset_info)
            features['sector_herfindahl'] = np.sum((sector_counts / len(asset_info)) ** 2)

        if 'asset_type' in asset_info.columns:
            type_counts = asset_info['asset_type'].value_counts()
            features['n_asset_types'] = len(type_counts)
            features['asset_type_diversity'] = 1 - np.sum((type_counts / len(asset_info)) ** 2)

        return features

    def _analyze_risk_chars(self, returns: pd.DataFrame) -> Dict:
        """Analyze risk characteristics."""
        volatilities = returns.std() * np.sqrt(252)

        return {
            'avg_volatility': volatilities.mean(),
            'max_volatility': volatilities.max(),
            'min_volatility': volatilities.min(),
            'volatility_dispersion': volatilities.std() / volatilities.mean(),
            'high_vol_assets_pct': (volatilities > 0.3).sum() / len(volatilities)
        }

    def get_recommendation(self) -> str:
        """Get optimization recommendations based on configuration."""
        if not self.config_features:
            raise ValueError("Must call analyze() first")

        features = self.config_features
        recommendations = []

        # Size-based recommendations
        size = features['asset_size_category']
        if size == 'small':
            recommendations.append("Small universe - sparse methods less necessary")
        elif size in ['large', 'very_large']:
            recommendations.append("Large universe - sparse methods (mSSRM-PGA) highly beneficial")

        # Correlation-based
        if features['avg_correlation'] > 0.6:
            recommendations.append("High correlation - diversification limited, focus on factor exposure")
        elif features['avg_correlation'] < 0.3:
            recommendations.append("Low correlation - good diversification opportunity")

        # Risk-based
        if features.get('high_vol_assets_pct', 0) > 0.5:
            recommendations.append("Many high-volatility assets - risk management critical")

        return "\n- ".join(["Asset Configuration Analysis:"] + recommendations)
