"""
Multi-Asset Data Fetcher

Fetches and processes data for multiple asset classes including:
- Equities (stocks, ETFs)
- Fixed Income (bonds, TLT, etc.)
- Commodities (gold, silver, oil)
- Cryptocurrencies (Bitcoin, Ethereum)
- Currency (forex)

Supports various data sources and handles different data formats.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MultiAssetFetcher:
    """
    Comprehensive multi-asset class data fetcher.

    Supports:
    - US Equities (via yfinance)
    - ETFs (equity, bond, commodity ETFs)
    - Cryptocurrencies (via yfinance crypto tickers)
    - Commodities (via ETFs and futures proxies)
    - Bonds (via bond ETFs)
    """

    # Asset class mapping
    ASSET_CLASSES = {
        # Large Cap Tech
        'AAPL': 'equity', 'MSFT': 'equity', 'GOOGL': 'equity',
        'AMZN': 'equity', 'NVDA': 'equity', 'META': 'equity', 'TSLA': 'equity',

        # Financials
        'JPM': 'equity', 'BAC': 'equity', 'GS': 'equity', 'V': 'equity',
        'WFC': 'equity', 'MA': 'equity', 'C': 'equity',

        # Healthcare
        'JNJ': 'equity', 'UNH': 'equity', 'PFE': 'equity', 'ABBV': 'equity',
        'LLY': 'equity', 'TMO': 'equity',

        # Consumer
        'WMT': 'equity', 'HD': 'equity', 'MCD': 'equity', 'NKE': 'equity',
        'COST': 'equity', 'PG': 'equity', 'KO': 'equity', 'PEP': 'equity',

        # Energy
        'XOM': 'equity', 'CVX': 'equity', 'COP': 'equity', 'SLB': 'equity',

        # Industrial
        'CAT': 'equity', 'BA': 'equity', 'GE': 'equity', 'MMM': 'equity',

        # Equity ETFs
        'SPY': 'equity_etf', 'QQQ': 'equity_etf', 'IWM': 'equity_etf',
        'DIA': 'equity_etf', 'VTI': 'equity_etf', 'VOO': 'equity_etf',
        'VEA': 'equity_etf',  # International developed
        'VWO': 'equity_etf',  # Emerging markets
        'EEM': 'equity_etf',  # Emerging markets

        # Bond ETFs
        'TLT': 'bond_etf',    # 20+ Year Treasury
        'IEF': 'bond_etf',    # 7-10 Year Treasury
        'SHY': 'bond_etf',    # 1-3 Year Treasury
        'LQD': 'bond_etf',    # Investment Grade Corporate
        'HYG': 'bond_etf',    # High Yield Corporate
        'AGG': 'bond_etf',    # Aggregate Bond
        'BND': 'bond_etf',    # Total Bond Market
        'TIP': 'bond_etf',    # TIPS
        'MUB': 'bond_etf',    # Municipal Bonds

        # Commodity ETFs
        'GLD': 'commodity',   # Gold
        'SLV': 'commodity',   # Silver
        'USO': 'commodity',   # Oil
        'UNG': 'commodity',   # Natural Gas
        'DBC': 'commodity',   # Diversified Commodities
        'PDBC': 'commodity',  # Optimum Yield Diversified Commodity

        # Crypto (via exchanges)
        'BTC-USD': 'crypto',  # Bitcoin
        'ETH-USD': 'crypto',  # Ethereum
        'BNB-USD': 'crypto',  # Binance Coin
        'SOL-USD': 'crypto',  # Solana
        'ADA-USD': 'crypto',  # Cardano

        # Real Estate
        'VNQ': 'reit',        # Vanguard Real Estate ETF
        'IYR': 'reit',        # iShares Real Estate ETF

        # Volatility
        'VXX': 'volatility',  # Short-term VIX futures

        # Currency
        'UUP': 'currency',    # US Dollar Index
        'FXE': 'currency',    # Euro
        'FXY': 'currency',    # Japanese Yen
    }

    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize multi-asset fetcher.

        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
        """
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = pd.to_datetime(end_date)

        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365 * 3)  # 3 years default
        else:
            self.start_date = pd.to_datetime(start_date)

        self.prices_df = None
        self.returns_df = None
        self.asset_info = None

    def fetch_assets(self,
                    tickers: List[str],
                    auto_classify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch prices for multiple assets.

        Args:
            tickers: List of ticker symbols
            auto_classify: Automatically classify asset types

        Returns:
            Tuple of (prices_df, returns_df)
        """
        print(f"\n{'='*70}")
        print("MULTI-ASSET DATA FETCHER")
        print(f"{'='*70}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Assets: {len(tickers)} tickers")

        # Classify assets
        if auto_classify:
            self.asset_info = self._classify_assets(tickers)
            print(f"\nAsset Classification:")
            for asset_class, count in self.asset_info['asset_class'].value_counts().items():
                print(f"  {asset_class}: {count} assets")

        # Download data
        print(f"\nDownloading data...")
        try:
            data = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by='ticker'
            )

            # Process data
            prices_dict = {}

            if len(tickers) == 1:
                # Single ticker case
                prices_dict[tickers[0]] = data['Adj Close'] if 'Adj Close' in data else data['Close']
            else:
                # Multiple tickers case
                for ticker in tickers:
                    try:
                        if ticker in data.columns.levels[0]:
                            ticker_data = data[ticker]
                            if 'Adj Close' in ticker_data.columns:
                                prices_dict[ticker] = ticker_data['Adj Close']
                            else:
                                prices_dict[ticker] = ticker_data['Close']
                    except Exception as e:
                        print(f"Warning: Could not process {ticker}: {e}")

            # Create DataFrame
            self.prices_df = pd.DataFrame(prices_dict)

            # Clean data
            self.prices_df = self.prices_df.dropna(how='all')
            self.prices_df = self.prices_df.fillna(method='ffill').fillna(method='bfill')

            # Calculate returns
            self.returns_df = self.prices_df.pct_change().dropna()

            print(f"\n✓ Successfully fetched {len(self.prices_df.columns)} assets")
            print(f"✓ Date range: {self.prices_df.index[0].date()} to {self.prices_df.index[-1].date()}")
            print(f"✓ Total observations: {len(self.prices_df)}")

            return self.prices_df, self.returns_df

        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            raise

    def _classify_assets(self, tickers: List[str]) -> pd.DataFrame:
        """
        Classify assets into categories.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with asset classifications
        """
        classifications = []

        for ticker in tickers:
            asset_class = self.ASSET_CLASSES.get(ticker, 'unknown')

            # Additional heuristic classification
            if asset_class == 'unknown':
                if '-USD' in ticker:
                    asset_class = 'crypto'
                elif ticker.startswith('^'):
                    asset_class = 'index'
                elif ticker.endswith('=X'):
                    asset_class = 'currency'

            classifications.append({
                'ticker': ticker,
                'asset_class': asset_class
            })

        return pd.DataFrame(classifications)

    def get_asset_statistics(self) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each asset.

        Returns:
            DataFrame with asset statistics
        """
        if self.returns_df is None:
            raise ValueError("Must fetch data first using fetch_assets()")

        stats_list = []

        for ticker in self.returns_df.columns:
            returns = self.returns_df[ticker].dropna()
            prices = self.prices_df[ticker].dropna()

            # Basic statistics
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe = mean_return / volatility if volatility > 0 else 0

            # Risk metrics
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            max_dd = self._calculate_max_drawdown(prices)

            # Distribution metrics
            from scipy import stats as scipy_stats
            skewness = scipy_stats.skew(returns)
            kurtosis = scipy_stats.kurtosis(returns)

            # Asset class
            asset_class = self.asset_info[
                self.asset_info['ticker'] == ticker
            ]['asset_class'].iloc[0] if self.asset_info is not None else 'unknown'

            stats_list.append({
                'ticker': ticker,
                'asset_class': asset_class,
                'annual_return': mean_return,
                'annual_volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'latest_price': prices.iloc[-1]
            })

        return pd.DataFrame(stats_list)

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def get_correlation_matrix(self,
                               by_asset_class: bool = False) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            by_asset_class: Group by asset class

        Returns:
            Correlation matrix
        """
        if self.returns_df is None:
            raise ValueError("Must fetch data first using fetch_assets()")

        corr_matrix = self.returns_df.corr()

        if by_asset_class and self.asset_info is not None:
            # Reorder by asset class
            ordered_tickers = self.asset_info.sort_values('asset_class')['ticker'].tolist()
            ordered_tickers = [t for t in ordered_tickers if t in corr_matrix.columns]
            corr_matrix = corr_matrix.loc[ordered_tickers, ordered_tickers]

        return corr_matrix

    def create_diversified_portfolio(self,
                                    n_per_class: int = 3,
                                    include_crypto: bool = True,
                                    include_bonds: bool = True) -> List[str]:
        """
        Create a diversified portfolio across asset classes.

        Args:
            n_per_class: Number of assets per class
            include_crypto: Include cryptocurrencies
            include_bonds: Include bonds

        Returns:
            List of selected tickers
        """
        selected = []

        # Equities
        equities = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'WMT', 'XOM']
        selected.extend(equities[:n_per_class])

        # Equity ETFs
        equity_etfs = ['SPY', 'QQQ', 'VTI']
        selected.extend(equity_etfs[:n_per_class])

        # Bonds
        if include_bonds:
            bonds = ['TLT', 'IEF', 'LQD']
            selected.extend(bonds[:n_per_class])

        # Commodities
        commodities = ['GLD', 'SLV', 'DBC']
        selected.extend(commodities[:n_per_class])

        # Crypto
        if include_crypto:
            crypto = ['BTC-USD', 'ETH-USD']
            selected.extend(crypto[:min(2, n_per_class)])

        # Real Estate
        reits = ['VNQ']
        selected.extend(reits[:1])

        return selected

    def print_summary(self):
        """Print summary of fetched data."""
        if self.prices_df is None:
            print("No data fetched yet.")
            return

        print(f"\n{'='*70}")
        print("DATA SUMMARY")
        print(f"{'='*70}")

        print(f"\nAssets: {len(self.prices_df.columns)}")
        print(f"Observations: {len(self.prices_df)}")
        print(f"Date Range: {self.prices_df.index[0].date()} to {self.prices_df.index[-1].date()}")

        if self.asset_info is not None:
            print(f"\nAsset Class Distribution:")
            for asset_class, count in self.asset_info['asset_class'].value_counts().items():
                pct = count / len(self.asset_info) * 100
                print(f"  {asset_class:15s}: {count:3d} ({pct:5.1f}%)")

        # Calculate statistics
        stats_df = self.get_asset_statistics()

        print(f"\nReturn Statistics (Annualized):")
        print(f"  Mean Return:    {stats_df['annual_return'].mean():6.2%}")
        print(f"  Mean Volatility: {stats_df['annual_volatility'].mean():6.2%}")
        print(f"  Mean Sharpe:    {stats_df['sharpe_ratio'].mean():6.3f}")

        print(f"\nTop 5 Performers (by Sharpe Ratio):")
        top5 = stats_df.nlargest(5, 'sharpe_ratio')
        for idx, row in top5.iterrows():
            print(f"  {row['ticker']:8s} - Sharpe: {row['sharpe_ratio']:6.3f}, "
                  f"Return: {row['annual_return']:6.2%}, Vol: {row['annual_volatility']:6.2%}")


def create_example_portfolio(scenario: str = 'balanced') -> List[str]:
    """
    Create example portfolio for different scenarios.

    Args:
        scenario: 'aggressive', 'balanced', 'conservative', 'all_assets'

    Returns:
        List of tickers
    """
    scenarios = {
        'aggressive': [
            # High growth tech
            'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'TSLA',
            # Growth ETFs
            'QQQ', 'ARKK',
            # Crypto
            'BTC-USD', 'ETH-USD',
            # Commodities
            'GLD'
        ],

        'balanced': [
            # Blue chips
            'AAPL', 'MSFT', 'JPM', 'JNJ', 'WMT',
            # Index ETFs
            'SPY', 'QQQ',
            # Bonds
            'TLT', 'LQD',
            # Commodities
            'GLD', 'DBC',
            # Crypto (small allocation)
            'BTC-USD'
        ],

        'conservative': [
            # Defensive stocks
            'JNJ', 'PG', 'KO', 'WMT',
            # Bonds (large allocation)
            'TLT', 'IEF', 'LQD', 'AGG',
            # Gold
            'GLD',
            # Low vol ETF
            'SPY'
        ],

        'all_assets': [
            # Equities
            'AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'WMT', 'XOM',
            # ETFs
            'SPY', 'QQQ', 'VEA', 'VWO',
            # Bonds
            'TLT', 'IEF', 'LQD', 'HYG',
            # Commodities
            'GLD', 'SLV', 'DBC',
            # Crypto
            'BTC-USD', 'ETH-USD',
            # Real Estate
            'VNQ',
            # Currency
            'UUP'
        ]
    }

    return scenarios.get(scenario, scenarios['balanced'])


if __name__ == '__main__':
    # Example usage
    print("Multi-Asset Data Fetcher - Example")

    fetcher = MultiAssetFetcher(start_date='2021-01-01')

    # Create balanced portfolio
    tickers = create_example_portfolio('balanced')
    print(f"\nExample portfolio: {', '.join(tickers)}")

    # Fetch data
    prices, returns = fetcher.fetch_assets(tickers)

    # Print summary
    fetcher.print_summary()

    # Get statistics
    stats = fetcher.get_asset_statistics()
    print(f"\n{stats.to_string()}")
