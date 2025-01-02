"""
Yahoo Finance Data Fetcher

This module provides functionality to fetch financial data from Yahoo Finance.
"""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

class YahooFinanceFetcher:
    """Fetches financial data from Yahoo Finance."""

    INTERVALS = {
        '1m': {'days': 7, 'description': '1 minute'},
        '2m': {'days': 60, 'description': '2 minutes'},
        '5m': {'days': 60, 'description': '5 minutes'},
        '15m': {'days': 60, 'description': '15 minutes'},
        '30m': {'days': 60, 'description': '30 minutes'},
        '60m': {'days': 730, 'description': '60 minutes'},
        '90m': {'days': 60, 'description': '90 minutes'},
        '1h': {'days': 730, 'description': '1 hour'},
        '1d': {'days': None, 'description': '1 day'},
        '5d': {'days': None, 'description': '5 days'},
        '1wk': {'days': None, 'description': '1 week'},
        '1mo': {'days': None, 'description': '1 month'},
        '3mo': {'days': None, 'description': '3 months'}
    }

    def __init__(self, rate_limit_pause: float = 0.5, max_retries: int = 3, cache_dir: Optional[str] = None):
        """
        Initialize the Yahoo Finance fetcher.

        Args:
            rate_limit_pause (float): Pause between API calls to avoid rate limiting
            max_retries (int): Maximum number of retries for failed requests
            cache_dir (str, optional): Directory to cache downloaded data
        """
        self.rate_limit_pause = rate_limit_pause
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _validate_interval(self, interval: str) -> bool:
        """
        Validate if the given interval is supported.

        Args:
            interval (str): Time interval for data points

        Returns:
            bool: True if interval is valid, False otherwise
        """
        return interval in self.INTERVALS

    def _validate_date_range(self, start_date: datetime, end_date: datetime, interval: str) -> bool:
        """
        Validate if the date range is valid for the given interval.

        Args:
            start_date (datetime): Start date for data retrieval
            end_date (datetime): End date for data retrieval
            interval (str): Time interval for data points

        Returns:
            bool: True if date range is valid, False otherwise
        """
        if not self._validate_interval(interval):
            return False

        max_days = self.INTERVALS[interval]['days']
        if max_days is None:
            return True

        date_diff = (end_date - start_date).days
        return date_diff <= max_days

    @lru_cache(maxsize=100)
    def fetch_data(self, symbol: str, interval: str = '1d',
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch financial data for a given symbol.

        Args:
            symbol (str): Stock symbol to fetch data for
            interval (str): Time interval for data points
            start_date (datetime, optional): Start date for data retrieval
            end_date (datetime, optional): End date for data retrieval

        Returns:
            pd.DataFrame: DataFrame containing the fetched data
        """
        try:
            if not self._validate_interval(interval):
                logger.error(f"Invalid interval: {interval}")
                return None

            if start_date and end_date:
                if not self._validate_date_range(start_date, end_date, interval):
                    logger.error(f"Invalid date range for interval {interval}")
                    return None

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                interval=interval,
                start=start_date,
                end=end_date
            )

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None