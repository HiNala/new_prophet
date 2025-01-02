"""
Unit tests for the stock sorting algorithms module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime, timedelta

from ...pipeline.data_sorting.sort_algorithms import StockSorter
from ..utils.test_data_generator import TestDataGenerator

class TestStockSorter(unittest.TestCase):
    """Test cases for StockSorter class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test output directory
        cls.test_output_dir = Path('data/test/sorted')
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sorter
        cls.sorter = StockSorter(
            consistency_threshold=0.8,
            volatility_threshold=0.05,
            output_dir=str(cls.test_output_dir)
        )
        
        # Generate test data
        cls.test_data = TestDataGenerator.generate_sorting_categories()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove test output directory
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)

    def test_calculate_consistency(self):
        """Test consistency calculation."""
        # Test with high consistency data
        high_consistency_data = self.test_data['high_consistency_low_volatility']
        consistency_metrics = self.sorter.calculate_consistency(high_consistency_data)
        
        # Check metric structure
        self.assertIsInstance(consistency_metrics, dict)
        self.assertIn('r2_score', consistency_metrics)
        self.assertIn('normalized_mad', consistency_metrics)
        
        # Score should be high for consistent data
        self.assertGreaterEqual(consistency_metrics['r2_score'], self.sorter.consistency_threshold)
        self.assertLess(consistency_metrics['normalized_mad'], 0.1)  # Expect low MAD for consistent data
        
        # Test with low consistency data
        low_consistency_data = self.test_data['low_consistency_high_volatility']
        consistency_metrics = self.sorter.calculate_consistency(low_consistency_data)
        
        # Score should be lower for inconsistent data
        self.assertLess(consistency_metrics['r2_score'], self.sorter.consistency_threshold)
        self.assertGreater(consistency_metrics['normalized_mad'], 0.1)  # Expect higher MAD for inconsistent data

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Test with low volatility data
        low_volatility_data = self.test_data['high_consistency_low_volatility']
        volatility_metrics = self.sorter.calculate_volatility(low_volatility_data)
        
        # Check metric structure
        self.assertIsInstance(volatility_metrics, dict)
        self.assertIn('returns_std', volatility_metrics)
        self.assertIn('cv', volatility_metrics)
        
        # Score should be low for stable data
        self.assertLess(volatility_metrics['returns_std'], self.sorter.volatility_threshold)
        self.assertLess(volatility_metrics['cv'], 0.1)  # Expect low CV for stable data
        
        # Test with high volatility data
        high_volatility_data = self.test_data['low_consistency_high_volatility']
        volatility_metrics = self.sorter.calculate_volatility(high_volatility_data)
        
        # Score should be higher for volatile data
        self.assertGreaterEqual(volatility_metrics['returns_std'], self.sorter.volatility_threshold)
        self.assertGreater(volatility_metrics['cv'], 0.1)  # Expect higher CV for volatile data

    def test_categorize_stock(self):
        """Test stock categorization."""
        # Test HCLV category
        data = self.test_data['high_consistency_low_volatility']
        category, metrics = self.sorter.categorize_stock(data)
        
        self.assertEqual(category, 'HCLV')
        self.assertIsInstance(metrics, dict)
        self.assertIn('r2_score', metrics)
        self.assertIn('normalized_mad', metrics)
        self.assertIn('returns_std', metrics)
        self.assertIn('cv', metrics)
        
        # Test LCHV category
        data = self.test_data['low_consistency_high_volatility']
        category, metrics = self.sorter.categorize_stock(data)
        self.assertEqual(category, 'LCHV')

    def test_sort_stocks(self):
        """Test sorting multiple stocks."""
        # Create test stock data
        stock_data = {
            'HCLV_STOCK': self.test_data['high_consistency_low_volatility'],
            'LCHV_STOCK': self.test_data['low_consistency_high_volatility'],
            'TREND_UP': self.test_data['trending_upward'],
            'TREND_DOWN': self.test_data['trending_downward']
        }
        
        # Sort stocks
        categorized_stocks = self.sorter.sort_stocks(stock_data)
        
        # Check categorization
        self.assertIsInstance(categorized_stocks, dict)
        self.assertTrue(all(category in categorized_stocks for category in self.sorter.CATEGORIES))
        
        # Check summary files
        summary_path = self.test_output_dir / 'sorting_summary.csv'
        self.assertTrue(summary_path.exists())
        
        # Read and verify summary
        summary_df = pd.read_csv(summary_path)
        self.assertEqual(len(summary_df), len(stock_data))
        expected_columns = ['symbol', 'category', 'r2_score', 'normalized_mad', 'returns_std', 'cv']
        self.assertTrue(all(col in summary_df.columns for col in expected_columns))
        
        # Check category directories and documentation
        for category in self.sorter.CATEGORIES:
            category_dir = self.test_output_dir / category
            self.assertTrue(category_dir.exists())
            self.assertTrue((category_dir / 'summary').exists())
            
            # Check README
            readme_path = category_dir / 'README.md'
            self.assertTrue(readme_path.exists())
            
            # Check statistics file
            stats_path = category_dir / 'summary' / f"{category}_statistics.csv"
            self.assertTrue(stats_path.exists())
            
            # Verify statistics content
            stats_df = pd.read_csv(stats_path, index_col=0)
            self.assertTrue(all(metric in stats_df.columns for metric in ['r2_score', 'normalized_mad', 'returns_std', 'cv'])) 