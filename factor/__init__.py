"""
Module 3: Factor Analysis System

This module provides 133-factor library and factor selection methods.
"""

from .factor_analyzer import FactorAnalyzer
from .factor_selection import FactorSelector
from .integrated_factor_system import IntegratedFactorSystem

__all__ = [
    'FactorAnalyzer',
    'FactorSelector',
    'IntegratedFactorSystem'
]
