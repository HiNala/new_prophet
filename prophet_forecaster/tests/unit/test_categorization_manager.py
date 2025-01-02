"""
Unit tests for the categorization manager module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

from ...pipeline.data_sorting.categorization_manager import CategorizationManager
from ..utils.test_data_generator import TestDataGenerator

class TestCategorizationManager(unittest.TestCase):
    """Test cases for CategorizationManager class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test directories
        cls.test_input_dir = Path('data/test/processed')
        cls.test_output_dir = Path('data/test/sorted')
        
        cls.test_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save test data
        cls.test_data = TestDataGenerator.generate_sorting_categories()
        
        # Save test data to input directory
        for category, data in cls.test_data.items():
            symbol = f"TEST_{category.upper()}"
            data.to_csv(cls.test_input_dir / f"{symbol}.csv")
        
        # Initialize manager
        cls.manager = CategorizationManager(
            input_dir=str(cls.test_input_dir),
            output_dir=str(cls.test_output_dir),
            consistency_threshold=0.8,
            volatility_threshold=0.05
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove test directories
        if cls.test_input_dir.exists():
            shutil.rmtree(cls.test_input_dir)
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)

    def test_load_stock_data(self):
        """Test loading stock data."""
        # Load all stocks
        stock_data = self.manager.load_stock_data()
        
        # Check loaded data
        self.assertEqual(len(stock_data), len(self.test_data))
        self.assertTrue(all(isinstance(data, pd.DataFrame) for data in stock_data.values()))
        
        # Load specific symbols
        symbols = ['TEST_HIGH_CONSISTENCY_LOW_VOLATILITY']
        stock_data = self.manager.load_stock_data(symbols)
        
        # Check filtered data
        self.assertEqual(len(stock_data), len(symbols))
        self.assertTrue(all(symbol in stock_data for symbol in symbols))

    def test_categorize_stocks(self):
        """Test stock categorization process."""
        # Categorize all stocks
        categorized_stocks = self.manager.categorize_stocks()
        
        # Check categorization
        self.assertIsInstance(categorized_stocks, dict)
        self.assertTrue(all(category in categorized_stocks for category in ['HCLV', 'HCHV', 'LCLV', 'LCHV']))
        
        # Check output files
        summary_path = self.test_output_dir / 'sorting_summary.csv'
        self.assertTrue(summary_path.exists())
        
        # Check category directories
        for category in ['HCLV', 'HCHV', 'LCLV', 'LCHV']:
            category_dir = self.test_output_dir / category
            self.assertTrue(category_dir.exists())
            self.assertTrue((category_dir / 'summary').exists())

    def test_get_category_summary(self):
        """Test category summary generation."""
        # First categorize stocks
        self.manager.categorize_stocks()
        
        # Get summary
        summary = self.manager.get_category_summary()
        
        # Check summary structure
        self.assertIsInstance(summary, pd.DataFrame)
        expected_columns = [
            'stock_count',
            'avg_consistency',
            'std_consistency',
            'avg_volatility',
            'std_volatility'
        ]
        self.assertTrue(all(col in summary.columns for col in expected_columns))
        
        # Check summary content
        self.assertTrue(len(summary) > 0)
        self.assertTrue(all(summary['stock_count'] >= 0))
        self.assertTrue(all(summary['avg_consistency'] >= 0))
        self.assertTrue(all(summary['avg_volatility'] >= 0)) 