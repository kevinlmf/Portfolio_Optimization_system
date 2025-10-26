"""
Module 5: Evaluation and Integrated Systems

This module handles backtesting, performance evaluation, and integrated systems.
"""

from .backtesting_engine import BacktestingEngine
from .bayesian_updater import (
    BayesianPosteriorUpdater,
    RegimeAwareUpdater,
    UpdateResult
)
from .bayesian_system import (
    IntegratedBayesianSystem,
    SystemState,
    BacktestResult
)

__all__ = [
    'BacktestingEngine',
    'BayesianPosteriorUpdater',
    'RegimeAwareUpdater',
    'UpdateResult',
    'IntegratedBayesianSystem',
    'SystemState',
    'BacktestResult'
]
