"""
Unit tests for the Yahoo Finance fetcher module.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from ...pipeline.data_ingestion.fetchers.yahoo_finance import YahooFinanceFetcher

class TestYahooFinanceFetcher(unittest.TestCase):
    """Test cases for YahooFinanceFetcher class."""

    def setUp(self):
        """Set up test cases."""
        plt.style.use('default')
        self.fetcher = YahooFinanceFetcher(
            rate_limit_pause=0.1,
            max_retries=2,
            cache_dir='tests/data/cache'
        )
        self.symbol = 'AAPL'
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()

    def test_interval_validation(self):
        """Test interval validation."""
        self.assertTrue(self.fetcher._validate_interval('1d'))
        self.assertFalse(self.fetcher._validate_interval('invalid'))

    def test_date_range_validation(self):
        """Test date range validation."""
        # Valid range for daily data
        self.assertTrue(
            self.fetcher._validate_date_range(
                self.start_date,
                self.end_date,
                '1d'
            )
        )

        # Invalid range for minute data (> 7 days)
        self.assertFalse(
            self.fetcher._validate_date_range(
                self.start_date,
                self.end_date,
                '1m'
            )
        )

    def test_fetch_daily_data(self):
        """Test fetching daily data."""
        data = self.fetcher.fetch_data(
            symbol=self.symbol,
            interval='1d',
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_cache_functionality(self):
        """Test data caching."""
        # First fetch (should cache)
        data1 = self.fetcher.fetch_data(
            symbol=self.symbol,
            interval='1d',
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Second fetch (should use cache)
        data2 = self.fetcher.fetch_data(
            symbol=self.symbol,
            interval='1d',
            start_date=self.start_date,
            end_date=self.end_date
        )

        pd.testing.assert_frame_equal(data1, data2) 