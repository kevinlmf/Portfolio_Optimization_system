"""
Data Module - 数据获取和 API 接口

提供数据获取、API 接口等功能
"""

from .api_client import APIClient
from .multi_asset_fetcher import MultiAssetFetcher

__all__ = ['APIClient', 'MultiAssetFetcher']

