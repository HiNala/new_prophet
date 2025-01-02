"""
Unit tests for the ingestion manager module.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
from ...pipeline.data_ingestion.ingestion_manager import IngestionManager

class TestIngestionManager(unittest.TestCase):
    """Test cases for IngestionManager class."""

    def setUp(self):
        """Set up test cases."""
        self.manager = IngestionManager()
        self.symbol = 'AAPL'
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()

    def test_fetch_single_symbol(self):
        """Test fetching data for a single symbol."""
        data = self.manager.fetch_single_symbol(
            symbol=self.symbol,
            interval='1d',
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_fetch_multiple_symbols(self):
        """Test fetching data for multiple symbols."""
        symbols = ['AAPL', 'MSFT']
        data_dict = self.manager.fetch_multiple_symbols(
            symbols=symbols,
            interval='1d',
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(len(data_dict), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, data_dict)
            self.assertIsInstance(data_dict[symbol], pd.DataFrame)
            self.assertFalse(data_dict[symbol].empty) 