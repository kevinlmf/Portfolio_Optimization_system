"""
Step 1: Factor Mining - 挖掘因子

Extract/select factors from stocks
从股票数据中挖掘驱动股票的共同因子
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
factor_mining_path = os.path.join(project_root, 'factor_layer', 'factor_mining')
if factor_mining_path not in sys.path:
    sys.path.insert(0, project_root)

# Import from factor_layer directory
from factor_layer.factor_mining.factor_miner import FactorMiner
from factor_layer.factor_mining.factor_selector import FactorSelector
from factor_layer.factor_mining.factor_analyzer import FactorAnalyzer

__all__ = ['FactorMiner', 'FactorSelector', 'FactorAnalyzer']

