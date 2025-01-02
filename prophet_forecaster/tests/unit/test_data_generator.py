"""
Unit tests for the test data generator module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from ..utils.test_data_generator import TestDataGenerator

class TestDataGeneratorTests(unittest.TestCase):
    """Test cases for TestDataGenerator class."""

    def test_generate_stock_data(self):
        """Test basic stock data generation."""
        data = TestDataGenerator.generate_stock_data()
        
        # Check structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data.columns), 5)  # OHLCV columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check data validity
        self.assertTrue(all(data['High'] >= data['Low']))
        self.assertTrue(all(data['Volume'] > 0))
        
        # Check date range
        self.assertEqual(data.index[0].strftime('%Y-%m-%d'), '2023-01-01')
        self.assertEqual(data.index[-1].strftime('%Y-%m-%d'), '2023-12-31')

    def test_generate_technical_indicator_data(self):
        """Test technical indicator data generation."""
        data, expected_values = TestDataGenerator.generate_technical_indicator_data()
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(expected_values, dict)
        
        # Check expected values
        expected_indicators = ['sma_20', 'sma_50', 'rsi_14', 'macd']
        self.assertListEqual(list(expected_values.keys()), expected_indicators)
        
        # Verify calculations
        pd.testing.assert_series_equal(
            expected_values['sma_20'],
            data['Close'].rolling(window=20).mean(),
            check_names=False
        )

    def test_generate_sorting_categories(self):
        """Test category data generation."""
        categories = TestDataGenerator.generate_sorting_categories()
        
        # Check structure
        self.assertIsInstance(categories, dict)
        expected_categories = [
            'high_consistency_low_volatility',
            'low_consistency_high_volatility',
            'trending_upward',
            'trending_downward'
        ]
        self.assertListEqual(list(categories.keys()), expected_categories)
        
        # Check characteristics
        hlv_std = categories['high_consistency_low_volatility']['Close'].std()
        lhv_std = categories['low_consistency_high_volatility']['Close'].std()
        self.assertGreater(lhv_std, hlv_std)  # High volatility should have higher std

    def test_generate_ensemble_data(self):
        """Test ensemble data generation."""
        features, target = TestDataGenerator.generate_ensemble_data()
        
        # Check structure
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        
        # Check features
        expected_features = ['sma_20', 'sma_50', 'rsi_14', 'volume_ma', 'returns', 'volatility']
        self.assertListEqual(list(features.columns), expected_features)
        
        # Check alignment
        self.assertEqual(len(features), len(target))
        self.assertTrue(all(features.index == target.index))
        
        # Check for NaN values
        self.assertFalse(features.isnull().any().any())
        self.assertFalse(target.isnull().any()) 