"""
Unit tests for the visualization module.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

from ...pipeline.data_visualization.visualization import DataVisualizer
from ...pipeline.data_ingestion.fetchers.yahoo_finance import YahooFinanceFetcher

class TestDataVisualizer(unittest.TestCase):
    """Test cases for DataVisualizer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test data directory
        cls.test_dir = Path('tests/data/visualizations')
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        cls.dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        cls.data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(cls.dates)),
            'High': np.random.normal(105, 10, len(cls.dates)),
            'Low': np.random.normal(95, 10, len(cls.dates)),
            'Close': np.random.normal(100, 10, len(cls.dates)),
            'Volume': np.random.randint(1000, 10000, len(cls.dates))
        }, index=cls.dates)

        # Initialize visualizer
        cls.visualizer = DataVisualizer(output_dir=str(cls.test_dir))
        cls.symbol = 'TEST'

    def test_plot_price_trends(self):
        """Test price trends visualization."""
        save_path = self.visualizer.plot_price_trends(
            data=self.data,
            symbol=self.symbol
        )
        self.assertTrue(Path(save_path).exists())
        self.assertTrue(save_path.endswith('_price_analysis.png'))

    def test_plot_distribution_analysis(self):
        """Test distribution analysis visualization."""
        save_path = self.visualizer.plot_distribution_analysis(
            data=self.data,
            symbol=self.symbol
        )
        self.assertTrue(Path(save_path).exists())
        self.assertTrue(save_path.endswith('_distribution_analysis.png'))

    def test_plot_missing_values(self):
        """Test missing values visualization."""
        # Add some missing values
        data_with_missing = self.data.copy()
        data_with_missing.iloc[10:20, 0] = np.nan
        data_with_missing.iloc[30:40, 2] = np.nan

        save_path = self.visualizer.plot_missing_values(
            data=data_with_missing,
            symbol=self.symbol
        )
        self.assertTrue(Path(save_path).exists())
        self.assertTrue(save_path.endswith('_missing_values.png'))

    def test_generate_summary_report(self):
        """Test summary report generation."""
        summary = self.visualizer.generate_summary_report(
            data=self.data,
            symbol=self.symbol
        )
        
        # Check summary structure
        self.assertIn('symbol', summary)
        self.assertIn('date_range', summary)
        self.assertIn('price_statistics', summary)
        self.assertIn('volume_statistics', summary)
        self.assertIn('missing_values', summary)

        # Check values
        self.assertEqual(summary['symbol'], self.symbol)
        self.assertEqual(summary['date_range']['trading_days'], len(self.data))

    def test_real_data_visualization(self):
        """Test visualization with real data from Yahoo Finance."""
        fetcher = YahooFinanceFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = fetcher.fetch_data(
            symbol='AAPL',
            interval='1d',
            start_date=start_date,
            end_date=end_date
        )

        if data is not None:
            # Test all visualizations with real data
            price_path = self.visualizer.plot_price_trends(data, 'AAPL')
            dist_path = self.visualizer.plot_distribution_analysis(data, 'AAPL')
            missing_path = self.visualizer.plot_missing_values(data, 'AAPL')
            summary = self.visualizer.generate_summary_report(data, 'AAPL')

            self.assertTrue(all(Path(p).exists() for p in [price_path, dist_path, missing_path]))
            self.assertIsInstance(summary, dict)
            self.assertIn('symbol', summary)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove test directory and its contents
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir) 