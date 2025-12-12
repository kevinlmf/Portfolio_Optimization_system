"""
Parameter Estimator Base Class - 参数估计器基类

参数估计器基类
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from .knowledge_base import KnowledgeBase


class ParameterEstimator(ABC):
    """参数估计器基类"""
    
    @abstractmethod
    def estimate(self,
                returns: pd.DataFrame,
                factors: Optional[pd.DataFrame] = None) -> KnowledgeBase:
        """
        估计所有参数并构建知识库
        
        Args:
            returns: 资产收益率
            factors: 因子收益率（可选）
        
        Returns:
            KnowledgeBase 对象
        """
        pass
