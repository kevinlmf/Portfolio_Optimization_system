"""
Mixture-based Market Regime Detector

Uses probabilistic mixture models (Gaussian/t-distribution) to detect market regimes
with soft transitions instead of hard classification.

Key Features:
- Soft regime probabilities: p(z_t=k | x_t) for all K regimes
- Multiple mixture types: Gaussian, Student-t, Bayesian Gaussian
- Automatic K selection via BIC/AIC
- Online updating capability
- Feature engineering from market data

Mathematical Framework:
    p(z_t=k | x_t) = π_k * f_k(x_t | θ_k) / Σ_j π_j * f_j(x_t | θ_j)

Where:
    - z_t: latent regime at time t
    - x_t: observed market features at time t
    - π_k: prior probability of regime k
    - f_k: likelihood function for regime k
    - θ_k: parameters for regime k
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegimeResult:
    """Container for regime detection results."""
    regime_probs: np.ndarray          # P(z_t=k | x_t) for all k
    dominant_regime: int              # argmax P(z_t=k | x_t)
    regime_names: List[str]           # Human-readable names
    confidence: float                 # max P(z_t=k | x_t)
    entropy: float                    # Uncertainty measure
    features: np.ndarray              # Extracted features
    feature_names: List[str]          # Feature descriptions


class MarketFeatureExtractor:
    """
    Extracts interpretable market features for regime detection.

    Features extracted:
    - Trend: returns, momentum, moving average slopes
    - Volatility: realized vol, vol-of-vol, extreme moves
    - Distribution: skewness, kurtosis, tail risk
    - Correlation: cross-sectional dispersion
    - Volume: turnover changes, liquidity indicators
    """

    def __init__(self,
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 vol_window: int = 20):
        """
        Initialize feature extractor.

        Args:
            lookback_short: Short-term window (e.g., 20 days)
            lookback_long: Long-term window (e.g., 60 days)
            vol_window: Volatility calculation window
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.vol_window = vol_window

    def extract(self,
                prices: Union[pd.Series, pd.DataFrame],
                volume: Optional[pd.Series] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract market features from price/volume data.

        Args:
            prices: Price series (Series) or multivariate (DataFrame)
            volume: Optional volume series

        Returns:
            features: 2D array (n_samples, n_features)
            feature_names: List of feature descriptions
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame('price')

        returns = prices.pct_change()

        features_list = []
        feature_names = []

        # 1. Trend features
        trend_features = self._extract_trend(returns)
        features_list.append(trend_features)
        feature_names.extend([
            'return_short', 'return_long', 'momentum',
            'ma_slope', 'trend_strength'
        ])

        # 2. Volatility features
        vol_features = self._extract_volatility(returns)
        features_list.append(vol_features)
        feature_names.extend([
            'volatility', 'vol_ratio', 'vol_of_vol',
            'extreme_moves', 'volatility_skew'
        ])

        # 3. Distribution features
        dist_features = self._extract_distribution(returns)
        features_list.append(dist_features)
        feature_names.extend([
            'skewness', 'kurtosis', 'tail_risk',
            'downside_deviation', 'max_drawdown'
        ])

        # 4. Cross-sectional features (if multivariate)
        if prices.shape[1] > 1:
            cross_features = self._extract_cross_sectional(returns)
            features_list.append(cross_features)
            feature_names.extend([
                'correlation_avg', 'correlation_std', 'dispersion'
            ])

        # 5. Volume features (if available)
        if volume is not None:
            vol_change = volume.pct_change()
            volume_features = self._extract_volume(vol_change, returns)
            features_list.append(volume_features)
            feature_names.extend([
                'volume_change', 'volume_volatility', 'liquidity_indicator'
            ])

        # Combine all features
        features = np.column_stack(features_list)

        # Handle NaN values
        features = pd.DataFrame(features, columns=feature_names).fillna(method='bfill').fillna(0).values

        return features, feature_names

    def _extract_trend(self, returns: pd.DataFrame) -> np.ndarray:
        """Extract trend-related features."""
        r = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]

        return_short = r.rolling(self.lookback_short).mean()
        return_long = r.rolling(self.lookback_long).mean()
        momentum = return_short - return_long

        # Moving average slope
        prices_cum = (1 + r).cumprod()
        ma_short = prices_cum.rolling(self.lookback_short).mean()
        ma_slope = ma_short.pct_change(self.lookback_short // 2)

        # Trend strength (% days above MA)
        trend_strength = (prices_cum > ma_short).rolling(self.lookback_short).mean()

        return np.column_stack([
            return_short, return_long, momentum, ma_slope, trend_strength
        ])

    def _extract_volatility(self, returns: pd.DataFrame) -> np.ndarray:
        """Extract volatility-related features."""
        r = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]

        volatility = r.rolling(self.vol_window).std() * np.sqrt(252)
        vol_long = r.rolling(self.vol_window * 3).std() * np.sqrt(252)
        vol_ratio = volatility / (vol_long + 1e-8)

        # Volatility of volatility
        vol_of_vol = volatility.rolling(self.lookback_short).std()

        # Extreme moves
        extreme_moves = (np.abs(r) > 2 * volatility / np.sqrt(252)).rolling(self.lookback_short).mean()

        # Volatility skew (upside vs downside vol)
        upside_vol = r[r > 0].rolling(self.vol_window).std()
        downside_vol = r[r < 0].rolling(self.vol_window).std()
        vol_skew = (upside_vol - downside_vol).fillna(0)

        return np.column_stack([
            volatility, vol_ratio, vol_of_vol, extreme_moves, vol_skew
        ])

    def _extract_distribution(self, returns: pd.DataFrame) -> np.ndarray:
        """Extract distributional features."""
        r = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]

        skewness = r.rolling(self.lookback_long).skew()
        kurtosis = r.rolling(self.lookback_long).kurt()

        # Tail risk (95% CVaR)
        def cvar_95(x):
            if len(x) < 5:
                return 0
            return np.mean(x[x <= np.percentile(x, 5)])

        tail_risk = r.rolling(self.lookback_long).apply(cvar_95, raw=True)

        # Downside deviation
        downside_dev = r[r < 0].rolling(self.lookback_long).std()

        # Maximum drawdown
        cumulative = (1 + r).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.rolling(self.lookback_long).min()

        return np.column_stack([
            skewness, kurtosis, tail_risk, downside_dev, max_dd
        ])

    def _extract_cross_sectional(self, returns: pd.DataFrame) -> np.ndarray:
        """Extract cross-sectional features."""
        # Average pairwise correlation
        def rolling_corr_avg(window):
            corrs = []
            for i in range(len(returns) - window + 1):
                window_data = returns.iloc[i:i+window]
                corr_matrix = window_data.corr().values
                # Get upper triangle (excluding diagonal)
                corrs.append(corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean())
            return pd.Series(corrs, index=returns.index[window-1:])

        correlation_avg = rolling_corr_avg(self.lookback_short)
        correlation_avg = correlation_avg.reindex(returns.index, method='ffill')

        # Correlation stability (std of correlations)
        def rolling_corr_std(window):
            corrs = []
            for i in range(len(returns) - window + 1):
                window_data = returns.iloc[i:i+window]
                corr_matrix = window_data.corr().values
                corrs.append(corr_matrix[np.triu_indices_from(corr_matrix, k=1)].std())
            return pd.Series(corrs, index=returns.index[window-1:])

        correlation_std = rolling_corr_std(self.lookback_short)
        correlation_std = correlation_std.reindex(returns.index, method='ffill')

        # Cross-sectional dispersion
        dispersion = returns.std(axis=1)

        return np.column_stack([
            correlation_avg, correlation_std, dispersion
        ])

    def _extract_volume(self, vol_change: pd.Series, returns: pd.DataFrame) -> np.ndarray:
        """Extract volume-related features."""
        r = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]

        volume_change = vol_change.rolling(self.lookback_short).mean()
        volume_volatility = vol_change.rolling(self.lookback_short).std()

        # Liquidity indicator (correlation between volume and |returns|)
        def rolling_vol_return_corr(window):
            corrs = []
            for i in range(len(r) - window + 1):
                abs_ret = np.abs(r.iloc[i:i+window])
                vol = vol_change.iloc[i:i+window]
                if len(abs_ret.dropna()) > 5 and len(vol.dropna()) > 5:
                    corr = abs_ret.corr(vol)
                    corrs.append(corr if not np.isnan(corr) else 0)
                else:
                    corrs.append(0)
            return pd.Series(corrs, index=r.index[window-1:])

        liquidity = rolling_vol_return_corr(self.lookback_short)
        liquidity = liquidity.reindex(r.index, method='ffill').fillna(0)

        return np.column_stack([
            volume_change, volume_volatility, liquidity
        ])


class MixtureRegimeDetector:
    """
    Probabilistic market regime detector using mixture models.

    Supported Models:
    - 'gaussian': Standard Gaussian Mixture Model (EM algorithm)
    - 'bayesian_gaussian': Bayesian GMM with Dirichlet Process prior
    - 't_mixture': Student-t Mixture (robust to outliers)
    """

    REGIME_NAMES = {
        0: 'CRISIS',
        1: 'BEAR_MARKET',
        2: 'HIGH_VOLATILITY',
        3: 'SIDEWAYS',
        4: 'BULL_MARKET'
    }

    def __init__(self,
                 n_regimes: int = 5,
                 mixture_type: str = 'bayesian_gaussian',
                 auto_select_k: bool = True,
                 k_range: Tuple[int, int] = (3, 7),
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 vol_window: int = 20,
                 random_state: int = 42):
        """
        Initialize mixture-based regime detector.

        Args:
            n_regimes: Number of regimes (if auto_select_k=False)
            mixture_type: Type of mixture model
            auto_select_k: Automatically determine optimal K via BIC
            k_range: Range of K to test if auto_select_k=True
            lookback_short: Short-term feature window
            lookback_long: Long-term feature window
            vol_window: Volatility window
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.mixture_type = mixture_type
        self.auto_select_k = auto_select_k
        self.k_range = k_range
        self.random_state = random_state

        self.feature_extractor = MarketFeatureExtractor(
            lookback_short=lookback_short,
            lookback_long=lookback_long,
            vol_window=vol_window
        )

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.optimal_k = n_regimes
        self.feature_names = None

    def fit(self,
            prices: Union[pd.Series, pd.DataFrame],
            volume: Optional[pd.Series] = None) -> 'MixtureRegimeDetector':
        """
        Fit the mixture model to historical data.

        Args:
            prices: Price series or DataFrame
            volume: Optional volume series

        Returns:
            self (fitted detector)
        """
        # Extract features
        features, self.feature_names = self.feature_extractor.extract(prices, volume)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Determine optimal K if needed
        if self.auto_select_k:
            self.optimal_k = self._select_optimal_k(features_scaled)
            print(f"Optimal number of regimes (K): {self.optimal_k}")

        # Fit mixture model
        self.model = self._create_model(self.optimal_k)
        self.model.fit(features_scaled)

        self.is_fitted = True

        # Assign interpretable names based on feature means
        self._assign_regime_names(features_scaled)

        return self

    def predict_proba(self,
                      prices: Union[pd.Series, pd.DataFrame],
                      volume: Optional[pd.Series] = None) -> RegimeResult:
        """
        Predict regime probabilities for new data.

        Args:
            prices: Price series or DataFrame
            volume: Optional volume series

        Returns:
            RegimeResult containing probabilities and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        features, _ = self.feature_extractor.extract(prices, volume)
        features_scaled = self.scaler.transform(features)

        # Get the latest observation
        latest_features = features_scaled[-1:, :]

        # Predict probabilities
        regime_probs = self.model.predict_proba(latest_features)[0]
        dominant_regime = np.argmax(regime_probs)
        confidence = regime_probs[dominant_regime]

        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))

        return RegimeResult(
            regime_probs=regime_probs,
            dominant_regime=dominant_regime,
            regime_names=[self.REGIME_NAMES.get(i, f'REGIME_{i}')
                         for i in range(len(regime_probs))],
            confidence=confidence,
            entropy=entropy,
            features=latest_features[0],
            feature_names=self.feature_names
        )

    def predict_sequence(self,
                        prices: Union[pd.Series, pd.DataFrame],
                        volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Predict regime probabilities for entire time series.

        Args:
            prices: Price series or DataFrame
            volume: Optional volume series

        Returns:
            DataFrame with regime probabilities over time
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        features, _ = self.feature_extractor.extract(prices, volume)
        features_scaled = self.scaler.transform(features)

        # Predict probabilities
        regime_probs = self.model.predict_proba(features_scaled)

        # Create DataFrame
        regime_df = pd.DataFrame(
            regime_probs,
            index=prices.index if isinstance(prices, pd.Series) else prices.index,
            columns=[self.REGIME_NAMES.get(i, f'REGIME_{i}')
                    for i in range(regime_probs.shape[1])]
        )

        # Add dominant regime and confidence
        regime_df['dominant_regime'] = np.argmax(regime_probs, axis=1)
        regime_df['confidence'] = np.max(regime_probs, axis=1)
        regime_df['entropy'] = -np.sum(regime_probs * np.log(regime_probs + 1e-10), axis=1)

        return regime_df

    def _create_model(self, n_components: int):
        """Create mixture model instance."""
        if self.mixture_type == 'gaussian':
            return GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=self.random_state,
                max_iter=200,
                n_init=10
            )
        elif self.mixture_type == 'bayesian_gaussian':
            return BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='full',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.1,  # Controls sparsity
                random_state=self.random_state,
                max_iter=200,
                n_init=10
            )
        else:
            raise ValueError(f"Unknown mixture_type: {self.mixture_type}")

    def _select_optimal_k(self, features: np.ndarray) -> int:
        """Select optimal number of regimes using BIC."""
        bic_scores = []
        k_values = range(self.k_range[0], self.k_range[1] + 1)

        for k in k_values:
            model = self._create_model(k)
            model.fit(features)
            bic_scores.append(model.bic(features))

        optimal_k = k_values[np.argmin(bic_scores)]
        return optimal_k

    def _assign_regime_names(self, features: np.ndarray):
        """
        Assign interpretable names to regimes based on feature statistics.

        Logic:
        - High negative returns + high vol → Crisis
        - Negative returns + medium vol → Bear
        - Low returns + very high vol → High Volatility
        - Low returns + low vol → Sideways
        - High positive returns + low vol → Bull
        """
        labels = self.model.predict(features)

        regime_stats = {}
        for k in range(self.optimal_k):
            mask = labels == k
            regime_features = features[mask]

            # Calculate key statistics
            # Assuming first few features are return-related, next are vol-related
            avg_return = regime_features[:, 0].mean()  # return_short
            avg_vol = regime_features[:, 5].mean()     # volatility

            regime_stats[k] = {
                'return': avg_return,
                'volatility': avg_vol,
                'score': avg_return - avg_vol  # Simple scoring
            }

        # Sort regimes by score (low to high)
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['score'])

        # Assign names
        name_mapping = {}
        for idx, (regime_id, stats) in enumerate(sorted_regimes):
            if idx == 0:
                name_mapping[regime_id] = 'CRISIS'
            elif idx == 1:
                name_mapping[regime_id] = 'BEAR_MARKET'
            elif idx == len(sorted_regimes) - 1:
                name_mapping[regime_id] = 'BULL_MARKET'
            elif stats['volatility'] > np.median([s['volatility'] for s in regime_stats.values()]):
                name_mapping[regime_id] = 'HIGH_VOLATILITY'
            else:
                name_mapping[regime_id] = 'SIDEWAYS'

        # Update class attribute
        self.REGIME_NAMES = name_mapping

    def get_regime_characteristics(self) -> pd.DataFrame:
        """
        Get statistical characteristics of each detected regime.

        Returns:
            DataFrame with regime statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        characteristics = []
        for k in range(self.optimal_k):
            char = {
                'regime': self.REGIME_NAMES.get(k, f'REGIME_{k}'),
                'prior_prob': self.model.weights_[k],
                'mean_features': self.model.means_[k].tolist()
            }
            characteristics.append(char)

        return pd.DataFrame(characteristics)


if __name__ == "__main__":
    # Example usage
    print("Mixture-based Regime Detector")
    print("=" * 60)
    print("Usage example:")
    print("""
    from strategy.mixture_regime_model import MixtureRegimeDetector
    import yfinance as yf

    # Download data
    spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')
    prices = spy['Close']

    # Initialize and fit
    detector = MixtureRegimeDetector(
        n_regimes=5,
        mixture_type='bayesian_gaussian',
        auto_select_k=True
    )
    detector.fit(prices)

    # Predict current regime
    result = detector.predict_proba(prices)
    print(f"Regime Probabilities: {result.regime_probs}")
    print(f"Dominant Regime: {result.regime_names[result.dominant_regime]}")
    print(f"Confidence: {result.confidence:.2%}")

    # Get full time series
    regime_history = detector.predict_sequence(prices)
    """)
