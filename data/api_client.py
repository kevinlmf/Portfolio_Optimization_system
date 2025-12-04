"""
API Client - API 客户端

提供统一的数据 API 接口，支持多种数据源：
- Yahoo Finance
- Alpha Vantage
- IEX Cloud
- 自定义数据源
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class APIClient:
    """
    Unified API client for fetching financial data
    
    统一的数据 API 客户端，支持多种数据源
    """
    
    def __init__(self, 
                 source: str = 'yahoo',
                 api_key: Optional[str] = None):
        """
        Initialize API client
        
        Args:
            source: Data source ('yahoo', 'alpha_vantage', 'iex', 'custom')
            api_key: API key for premium data sources
        """
        self.source = source.lower()
        self.api_key = api_key
        
    def fetch_returns(self,
                     symbols: List[str],
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime],
                     frequency: str = 'daily') -> pd.DataFrame:
        """
        Fetch asset returns
        
        Args:
            symbols: List of asset symbols
            start_date: Start date
            end_date: End date
            frequency: 'daily', 'weekly', 'monthly'
        
        Returns:
            Returns DataFrame (T x N)
        """
        if self.source == 'yahoo':
            return self._fetch_yahoo_returns(symbols, start_date, end_date, frequency)
        elif self.source == 'alpha_vantage':
            return self._fetch_alpha_vantage_returns(symbols, start_date, end_date, frequency)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def _fetch_yahoo_returns(self,
                            symbols: List[str],
                            start_date: Union[str, datetime],
                            end_date: Union[str, datetime],
                            frequency: str = 'daily') -> pd.DataFrame:
        """Fetch returns from Yahoo Finance"""
        try:
            # Download data
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                # Use 'Close' prices
                if 'Close' in data.columns.levels[0]:
                    prices = data['Close']
                else:
                    prices = data.xs('Close', level=0, axis=1)
            else:
                prices = data['Close'] if 'Close' in data.columns else data
            
            # Calculate returns
            if frequency == 'daily':
                returns = prices.pct_change().dropna()
            elif frequency == 'weekly':
                returns = prices.resample('W').last().pct_change().dropna()
            elif frequency == 'monthly':
                returns = prices.resample('M').last().pct_change().dropna()
            else:
                returns = prices.pct_change().dropna()
            
            return returns
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=symbols)
    
    def _fetch_alpha_vantage_returns(self,
                                    symbols: List[str],
                                    start_date: Union[str, datetime],
                                    end_date: Union[str, datetime],
                                    frequency: str = 'daily') -> pd.DataFrame:
        """Fetch returns from Alpha Vantage (requires API key)"""
        if not self.api_key:
            raise ValueError("API key required for Alpha Vantage")
        
        # Implementation for Alpha Vantage
        # This is a placeholder - implement based on Alpha Vantage API
        print("Alpha Vantage integration not yet implemented")
        return pd.DataFrame()
    
    def fetch_factors(self,
                     factor_names: List[str],
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch factor data
        
        Args:
            factor_names: List of factor names (e.g., ['SPY', 'VIX', 'DXY'])
            start_date: Start date
            end_date: End date
        
        Returns:
            Factor returns DataFrame (T x K)
        """
        return self.fetch_returns(factor_names, start_date, end_date)
    
    def fetch_market_data(self,
                         symbol: str,
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch comprehensive market data for a single symbol
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.source == 'yahoo':
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                return data
            except Exception as e:
                print(f"Error fetching market data for {symbol}: {e}")
                return pd.DataFrame()
        else:
            raise ValueError(f"Market data fetching not implemented for {self.source}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if data is valid
        """
        if data.empty:
            return False
        
        # Check for sufficient data points
        if len(data) < 10:
            return False
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.5:
            return False
        
        return True


def main():
    """Demo of API client"""
    print("=" * 80)
    print("API CLIENT DEMO")
    print("=" * 80)
    
    # Initialize client
    client = APIClient(source='yahoo')
    
    # Fetch returns
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"\nFetching returns for {symbols}...")
    returns = client.fetch_returns(symbols, start_date, end_date)
    
    print(f"\nReturns DataFrame shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"\nFirst few rows:")
    print(returns.head())
    
    # Validate data
    is_valid = client.validate_data(returns)
    print(f"\nData validation: {'✓ Valid' if is_valid else '✗ Invalid'}")


if __name__ == '__main__':
    main()



