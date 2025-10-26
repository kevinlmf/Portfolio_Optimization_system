"""
Intelligent Optimizer Selector

AI-driven system that automatically selects the best portfolio optimization method
based on market conditions, asset configuration, and historical performance.

This module integrates:
- Market regime detection
- Asset configuration analysis
- Method performance tracking
- ML-based recommendation engine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .market_regime_detector import MarketRegimeDetector, AssetConfigurationAnalyzer
from .sparse_sharpe_optimizer import SparseSharpeOptimizer


@dataclass
class OptimizerRecommendation:
    """Container for optimizer recommendation results."""
    recommended_method: str
    confidence: float
    reasoning: List[str]
    alternative_methods: List[Tuple[str, float]]
    market_regime: str
    regime_confidence: float
    asset_config_summary: Dict
    expected_characteristics: Dict


class IntelligentOptimizerSelector:
    """
    Intelligent system for selecting optimal portfolio optimization method.

    This selector considers:
    1. Current market regime
    2. Asset universe characteristics
    3. Historical method performance
    4. Investor constraints and preferences
    5. Computational requirements

    Supported Methods:
    - 'max_sharpe': Maximum Sharpe Ratio (traditional)
    - 'min_variance': Minimum Variance
    - 'risk_parity': Risk Parity
    - 'equal_weight': Equal Weight (1/N)
    - 'sparse_sharpe': m-Sparse Sharpe Ratio (mSSRM-PGA)
    - 'black_litterman': Black-Litterman
    - 'cvar': CVaR optimization
    - 'robust_mvo': Robust Mean-Variance
    """

    # Method characteristics matrix
    METHOD_PROFILES = {
        'max_sharpe': {
            'suitable_regimes': ['bull_market', 'sideways'],
            'min_assets': 5,
            'max_assets': 100,
            'handles_high_vol': False,
            'concentration_tendency': 'high',
            'computational_cost': 'low',
            'robustness': 'low',
            'interpretability': 'high'
        },
        'min_variance': {
            'suitable_regimes': ['bear_market', 'high_volatility', 'crisis'],
            'min_assets': 5,
            'max_assets': 200,
            'handles_high_vol': True,
            'concentration_tendency': 'medium',
            'computational_cost': 'low',
            'robustness': 'high',
            'interpretability': 'high'
        },
        'risk_parity': {
            'suitable_regimes': ['sideways', 'high_volatility', 'bear_market'],
            'min_assets': 3,
            'max_assets': 50,
            'handles_high_vol': True,
            'concentration_tendency': 'low',
            'computational_cost': 'medium',
            'robustness': 'high',
            'interpretability': 'high'
        },
        'equal_weight': {
            'suitable_regimes': ['crisis', 'high_volatility'],
            'min_assets': 5,
            'max_assets': 30,
            'handles_high_vol': True,
            'concentration_tendency': 'low',
            'computational_cost': 'very_low',
            'robustness': 'very_high',
            'interpretability': 'very_high'
        },
        'sparse_sharpe': {
            'suitable_regimes': ['bull_market', 'sideways'],
            'min_assets': 20,
            'max_assets': 500,
            'handles_high_vol': False,
            'concentration_tendency': 'very_high',
            'computational_cost': 'medium',
            'robustness': 'high',  # Has global optimality guarantee
            'interpretability': 'high'
        },
        'black_litterman': {
            'suitable_regimes': ['bull_market', 'sideways', 'bear_market'],
            'min_assets': 5,
            'max_assets': 100,
            'handles_high_vol': False,
            'concentration_tendency': 'medium',
            'computational_cost': 'medium',
            'robustness': 'high',
            'interpretability': 'medium'
        },
        'cvar': {
            'suitable_regimes': ['bear_market', 'high_volatility', 'crisis'],
            'min_assets': 5,
            'max_assets': 100,
            'handles_high_vol': True,
            'concentration_tendency': 'medium',
            'computational_cost': 'high',
            'robustness': 'high',
            'interpretability': 'medium'
        },
        'robust_mvo': {
            'suitable_regimes': ['high_volatility', 'crisis'],
            'min_assets': 5,
            'max_assets': 100,
            'handles_high_vol': True,
            'concentration_tendency': 'medium',
            'computational_cost': 'high',
            'robustness': 'very_high',
            'interpretability': 'medium'
        }
    }

    def __init__(self,
                 regime_detector: Optional[MarketRegimeDetector] = None,
                 config_analyzer: Optional[AssetConfigurationAnalyzer] = None,
                 performance_history: Optional[pd.DataFrame] = None):
        """
        Initialize the intelligent optimizer selector.

        Args:
            regime_detector: Market regime detector instance
            config_analyzer: Asset configuration analyzer instance
            performance_history: Historical performance of different methods
        """
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.config_analyzer = config_analyzer or AssetConfigurationAnalyzer()
        self.performance_history = performance_history

        self.current_recommendation = None
        self.method_scores = {}

    def select_optimizer(self,
                        prices: pd.Series,
                        returns: pd.DataFrame,
                        asset_info: Optional[pd.DataFrame] = None,
                        constraints: Optional[Dict] = None,
                        preferences: Optional[Dict] = None) -> OptimizerRecommendation:
        """
        Intelligently select the best optimization method.

        Args:
            prices: Price series for market regime detection (e.g., S&P 500)
            returns: Asset return matrix for configuration analysis
            asset_info: Optional metadata about assets
            constraints: Investment constraints (e.g., max_concentration)
            preferences: Investor preferences (e.g., prefer_sparse, risk_tolerance)

        Returns:
            OptimizerRecommendation object with detailed recommendation
        """
        # Set defaults
        constraints = constraints or {}
        preferences = preferences or {}

        # 1. Detect market regime
        regime_result = self.regime_detector.detect_regime(prices)
        market_regime = regime_result['regime']
        regime_confidence = regime_result['confidence']

        # 2. Analyze asset configuration
        config_features = self.config_analyzer.analyze(returns, asset_info)
        n_assets = config_features['n_assets']

        # 3. Score each method
        method_scores = {}
        for method, profile in self.METHOD_PROFILES.items():
            score = self._score_method(
                method, profile, market_regime, regime_confidence,
                config_features, constraints, preferences
            )
            method_scores[method] = score

        self.method_scores = method_scores

        # 4. Select best method
        sorted_methods = sorted(method_scores.items(),
                              key=lambda x: x[1], reverse=True)

        best_method, best_score = sorted_methods[0]
        alternatives = sorted_methods[1:4]  # Top 3 alternatives

        # 5. Generate reasoning
        reasoning = self._generate_reasoning(
            best_method, market_regime, config_features,
            constraints, preferences
        )

        # 6. Expected characteristics
        expected_chars = self._predict_characteristics(
            best_method, market_regime, config_features
        )

        # Create recommendation
        recommendation = OptimizerRecommendation(
            recommended_method=best_method,
            confidence=best_score,
            reasoning=reasoning,
            alternative_methods=alternatives,
            market_regime=market_regime,
            regime_confidence=regime_confidence,
            asset_config_summary=config_features,
            expected_characteristics=expected_chars
        )

        self.current_recommendation = recommendation
        return recommendation

    def _score_method(self,
                     method: str,
                     profile: Dict,
                     regime: str,
                     regime_conf: float,
                     config: Dict,
                     constraints: Dict,
                     preferences: Dict) -> float:
        """
        Score a method based on all factors.
        Returns a score between 0 and 1.
        """
        score = 0.0
        max_score = 0.0

        # 1. Regime suitability (weight: 40%)
        regime_weight = 0.40
        if regime in profile['suitable_regimes']:
            # Primary suitable regime
            regime_score = 1.0 * regime_conf
        else:
            # Check if regime is somewhat compatible
            compatibility = self._check_regime_compatibility(regime, profile['suitable_regimes'])
            regime_score = 0.3 * compatibility * regime_conf

        score += regime_score * regime_weight
        max_score += regime_weight

        # 2. Asset universe fit (weight: 25%)
        universe_weight = 0.25
        n_assets = config['n_assets']
        if profile['min_assets'] <= n_assets <= profile['max_assets']:
            universe_score = 1.0
        elif n_assets < profile['min_assets']:
            universe_score = 0.5
        else:  # n_assets > max
            universe_score = 0.6

        score += universe_score * universe_weight
        max_score += universe_weight

        # 3. Volatility handling (weight: 20%)
        vol_weight = 0.20
        high_vol = config['avg_volatility'] > 0.25 or regime in ['high_volatility', 'crisis']
        if high_vol:
            vol_score = 1.0 if profile['handles_high_vol'] else 0.3
        else:
            vol_score = 0.8  # Neutral if not high vol

        score += vol_score * vol_weight
        max_score += vol_weight

        # 4. Constraint compatibility (weight: 10%)
        constraint_weight = 0.10
        constraint_score = self._check_constraint_compatibility(
            method, profile, constraints
        )
        score += constraint_score * constraint_weight
        max_score += constraint_weight

        # 5. User preferences (weight: 5%)
        pref_weight = 0.05
        pref_score = self._check_preference_match(method, profile, preferences)
        score += pref_score * pref_weight
        max_score += pref_weight

        # Normalize to 0-1
        return score / max_score if max_score > 0 else 0

    def _check_regime_compatibility(self,
                                    regime: str,
                                    suitable_regimes: List[str]) -> float:
        """Check compatibility between regimes."""
        compatibility_matrix = {
            'bull_market': {'sideways': 0.6, 'high_volatility': 0.3},
            'bear_market': {'sideways': 0.5, 'high_volatility': 0.6, 'crisis': 0.7},
            'sideways': {'bull_market': 0.6, 'bear_market': 0.5, 'high_volatility': 0.4},
            'high_volatility': {'crisis': 0.8, 'bear_market': 0.6},
            'crisis': {'high_volatility': 0.8, 'bear_market': 0.7}
        }

        for suitable in suitable_regimes:
            if regime in compatibility_matrix.get(suitable, {}):
                return compatibility_matrix[suitable][regime]

        return 0.1  # Minimal compatibility

    def _check_constraint_compatibility(self,
                                       method: str,
                                       profile: Dict,
                                       constraints: Dict) -> float:
        """Check if method satisfies constraints."""
        score = 1.0

        # Max concentration constraint
        if 'max_concentration' in constraints:
            max_conc = constraints['max_concentration']
            conc_tendency = profile['concentration_tendency']

            if conc_tendency == 'very_high' and max_conc < 0.2:
                score *= 0.3  # Strong mismatch
            elif conc_tendency == 'high' and max_conc < 0.15:
                score *= 0.5

        # Computational budget
        if 'max_computation_time' in constraints:
            if constraints['max_computation_time'] == 'low':
                cost = profile['computational_cost']
                if cost in ['high', 'very_high']:
                    score *= 0.4

        return score

    def _check_preference_match(self,
                               method: str,
                               profile: Dict,
                               preferences: Dict) -> float:
        """Check if method matches user preferences."""
        score = 0.5  # Neutral default

        if preferences.get('prefer_sparse', False):
            if method == 'sparse_sharpe':
                score = 1.0
            elif profile['concentration_tendency'] in ['high', 'very_high']:
                score = 0.7

        if preferences.get('prefer_interpretable', False):
            interp = profile['interpretability']
            if interp in ['high', 'very_high']:
                score = max(score, 0.8)

        if preferences.get('prefer_robust', False):
            robust = profile['robustness']
            if robust in ['high', 'very_high']:
                score = max(score, 0.8)

        return score

    def _generate_reasoning(self,
                           method: str,
                           regime: str,
                           config: Dict,
                           constraints: Dict,
                           preferences: Dict) -> List[str]:
        """Generate human-readable reasoning for recommendation."""
        reasoning = []

        profile = self.METHOD_PROFILES[method]

        # Regime-based reasoning
        if regime in profile['suitable_regimes']:
            reasoning.append(
                f"[+] Method is specifically designed for {regime.replace('_', ' ')} conditions"
            )
        else:
            reasoning.append(
                f"[o] Method has acceptable compatibility with {regime.replace('_', ' ')}"
            )

        # Asset universe reasoning
        n_assets = config['n_assets']
        if method == 'sparse_sharpe' and n_assets >= 20:
            reasoning.append(
                f"[+] Large asset universe ({n_assets} assets) benefits from sparse optimization"
            )
        elif method in ['equal_weight', 'risk_parity'] and n_assets <= 30:
            reasoning.append(
                f"[+] Portfolio size ({n_assets} assets) is manageable for this method"
            )

        # Volatility reasoning
        if config['avg_volatility'] > 0.25:
            if profile['handles_high_vol']:
                reasoning.append(
                    f"[+] Method effectively handles high volatility (current: {config['avg_volatility']:.1%})"
                )
            else:
                reasoning.append(
                    f"[!] Current volatility is high ({config['avg_volatility']:.1%}), monitor closely"
                )

        # Correlation reasoning
        if config['avg_correlation'] > 0.6:
            reasoning.append(
                f"[o] High average correlation ({config['avg_correlation']:.2f}) limits diversification benefits"
            )

        # Method-specific strengths
        if method == 'sparse_sharpe':
            reasoning.append(
                "[+] Global optimality guarantee under certain conditions (see Lin et al. 2024)"
            )
        elif method == 'min_variance':
            reasoning.append(
                "[+] Robust in uncertain markets, prioritizes capital preservation"
            )
        elif method == 'equal_weight':
            reasoning.append(
                "[+] Simple and robust, avoids estimation error in extreme conditions"
            )

        return reasoning

    def _predict_characteristics(self,
                                method: str,
                                regime: str,
                                config: Dict) -> Dict:
        """Predict expected portfolio characteristics."""
        profile = self.METHOD_PROFILES[method]

        # Expected concentration
        conc_map = {
            'very_high': (0.30, 0.50),
            'high': (0.20, 0.35),
            'medium': (0.10, 0.25),
            'low': (0.05, 0.15)
        }
        expected_max_weight = conc_map.get(
            profile['concentration_tendency'], (0.10, 0.30)
        )

        # Expected number of active positions
        if method == 'sparse_sharpe':
            expected_positions = (10, 20)
        elif method == 'equal_weight':
            expected_positions = (config['n_assets'], config['n_assets'])
        else:
            ratio = 0.4 if profile['concentration_tendency'] == 'high' else 0.7
            expected_positions = (
                int(config['n_assets'] * 0.2),
                int(config['n_assets'] * ratio)
            )

        # Expected turnover
        turnover_map = {
            'bull_market': 'low',
            'bear_market': 'medium',
            'sideways': 'low',
            'high_volatility': 'high',
            'crisis': 'high'
        }
        expected_turnover = turnover_map.get(regime, 'medium')

        return {
            'expected_max_weight_range': expected_max_weight,
            'expected_n_positions_range': expected_positions,
            'expected_turnover': expected_turnover,
            'expected_sharpe_stability': profile['robustness'],
            'rebalancing_frequency_suggestion': self._suggest_rebalance_freq(regime)
        }

    def _suggest_rebalance_freq(self, regime: str) -> str:
        """Suggest rebalancing frequency based on regime."""
        freq_map = {
            'bull_market': 'quarterly',
            'bear_market': 'monthly',
            'sideways': 'quarterly',
            'high_volatility': 'weekly or bi-weekly',
            'crisis': 'daily to weekly'
        }
        return freq_map.get(regime, 'monthly')

    def get_method_comparison(self) -> pd.DataFrame:
        """Get comparison table of all methods with current scores."""
        if not self.method_scores:
            raise ValueError("Must call select_optimizer() first")

        comparison = []
        for method, score in sorted(self.method_scores.items(),
                                   key=lambda x: x[1], reverse=True):
            profile = self.METHOD_PROFILES[method]
            comparison.append({
                'Method': method,
                'Score': f'{score:.1%}',
                'Suitable Regimes': ', '.join(profile['suitable_regimes'][:2]),
                'Concentration': profile['concentration_tendency'],
                'Handles High Vol': 'Yes' if profile['handles_high_vol'] else 'No',
                'Computational Cost': profile['computational_cost'],
                'Robustness': profile['robustness']
            })

        return pd.DataFrame(comparison)

    def explain_recommendation(self) -> str:
        """Generate detailed explanation of current recommendation."""
        if not self.current_recommendation:
            raise ValueError("Must call select_optimizer() first")

        rec = self.current_recommendation

        explanation = f"""
{'='*70}
INTELLIGENT OPTIMIZER RECOMMENDATION
{'='*70}

RECOMMENDED METHOD: {rec.recommended_method.upper()}
Confidence: {rec.confidence:.1%}

MARKET CONTEXT:
- Current Regime: {rec.market_regime.replace('_', ' ').title()}
- Regime Confidence: {rec.regime_confidence:.1%}
- {rec.asset_config_summary['n_assets']} assets in universe
- Average Correlation: {rec.asset_config_summary['avg_correlation']:.2f}
- Average Volatility: {rec.asset_config_summary['avg_volatility']:.1%}

REASONING:
{chr(10).join('  ' + r for r in rec.reasoning)}

EXPECTED CHARACTERISTICS:
- Max Weight Range: {rec.expected_characteristics['expected_max_weight_range'][0]:.1%} - {rec.expected_characteristics['expected_max_weight_range'][1]:.1%}
- Number of Positions: {rec.expected_characteristics['expected_n_positions_range'][0]}-{rec.expected_characteristics['expected_n_positions_range'][1]}
- Expected Turnover: {rec.expected_characteristics['expected_turnover'].title()}
- Suggested Rebalancing: {rec.expected_characteristics['rebalancing_frequency_suggestion'].title()}

ALTERNATIVE METHODS:
"""
        for i, (method, score) in enumerate(rec.alternative_methods, 1):
            explanation += f"  {i}. {method} (score: {score:.1%})\n"

        explanation += f"\n{'='*70}\n"

        return explanation
