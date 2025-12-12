"""
Multi-Asset Fetcher - 多资产数据获取器

批量获取多个资产的数据
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime
from .api_client import APIClient


class MultiAssetFetcher:
    """
    Multi-asset data fetcher
    
    批量获取多个资产的数据
    """
    
    def __init__(self, api_client: Optional[APIClient] = None):
        """
        Initialize multi-asset fetcher
        
        Args:
            api_client: APIClient instance (creates new one if None)
        """
        self.api_client = api_client or APIClient()
    
    def fetch_portfolio_data(self,
                            symbols: List[str],
                            start_date: Union[str, datetime],
                            end_date: Union[str, datetime],
                            frequency: str = 'daily') -> pd.DataFrame:
        """
        Fetch returns for a portfolio of assets
        
        Args:
            symbols: List of asset symbols
            start_date: Start date
            end_date: End date
            frequency: 'daily', 'weekly', 'monthly'
        
        Returns:
            Returns DataFrame (T x N)
        """
        return self.api_client.fetch_returns(symbols, start_date, end_date, frequency)
    
    def fetch_factor_data(self,
                         factor_symbols: List[str],
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch factor data
        
        Args:
            factor_symbols: List of factor symbols (e.g., ['SPY', 'VIX'])
            start_date: Start date
            end_date: End date
        
        Returns:
            Factor returns DataFrame (T x K)
        """
        return self.api_client.fetch_factors(factor_symbols, start_date, end_date)
    
    def align_data(self,
                  returns: pd.DataFrame,
                  factors: Optional[pd.DataFrame] = None) -> tuple:
        """
        Align returns and factors to common date index
        
        Args:
            returns: Returns DataFrame
            factors: Factors DataFrame (optional)
        
        Returns:
            (aligned_returns, aligned_factors)
        """
        if factors is None:
            return returns, None
        
        # Find common dates
        common_dates = returns.index.intersection(factors.index)
        
        aligned_returns = returns.loc[common_dates]
        aligned_factors = factors.loc[common_dates]
        
        return aligned_returns, aligned_factors



