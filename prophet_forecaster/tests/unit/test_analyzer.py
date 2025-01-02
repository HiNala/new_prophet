"""
Unit tests for the data analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ...pipeline.data_analysis.analyzer import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    """Test cases for DataAnalyzer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test data
        cls.dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        cls.data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(cls.dates)),
            'High': np.random.normal(105, 10, len(cls.dates)),
            'Low': np.random.normal(95, 10, len(cls.dates)),
            'Close': np.random.normal(100, 10, len(cls.dates)),
            'Volume': np.random.randint(1000, 10000, len(cls.dates))
        }, index=cls.dates)

        # Add some missing values
        cls.data.iloc[10:20, 0] = np.nan
        cls.data.iloc[30:40, 2] = np.nan

        # Initialize analyzer
        cls.analyzer = DataAnalyzer()

    def test_calculate_statistics(self):
        """Test statistical calculations."""
        stats = self.analyzer.calculate_statistics(self.data)
        
        # Check structure
        self.assertIn('basic_stats', stats)
        self.assertIn('distribution_stats', stats)
        self.assertIn('range_stats', stats)

        # Check values
        self.assertIsInstance(stats['basic_stats']['mean'], dict)
        self.assertIsInstance(stats['distribution_stats']['skewness'], dict)
        self.assertIsInstance(stats['range_stats']['quantiles'], dict)

    def test_analyze_missing_values(self):
        """Test missing value analysis."""
        missing_stats = self.analyzer.analyze_missing_values(self.data)
        
        # Check structure
        self.assertIn('total_missing', missing_stats)
        self.assertIn('missing_percentage', missing_stats)
        self.assertIn('missing_patterns', missing_stats)

        # Verify missing values count
        self.assertEqual(missing_stats['total_missing']['Open'], 10)
        self.assertEqual(missing_stats['total_missing']['Low'], 10)

    def test_handle_missing_values(self):
        """Test missing value handling methods."""
        # Test forward fill
        filled_data = self.analyzer.handle_missing_values(self.data, method='ffill')
        self.assertEqual(filled_data.isnull().sum().sum(), 0)

        # Test backward fill
        filled_data = self.analyzer.handle_missing_values(self.data, method='bfill')
        self.assertEqual(filled_data.isnull().sum().sum(), 0)

        # Test interpolation
        filled_data = self.analyzer.handle_missing_values(self.data, method='interpolate')
        self.assertEqual(filled_data.isnull().sum().sum(), 0)

        # Test KNN imputation
        filled_data = self.analyzer.handle_missing_values(self.data, method='knn')
        self.assertEqual(filled_data.isnull().sum().sum(), 0)

    def test_detect_outliers(self):
        """Test outlier detection methods."""
        # Add some outliers
        data_with_outliers = self.data.copy()
        data_with_outliers.iloc[50, 0] = 1000  # Extreme value
        data_with_outliers.iloc[100, 2] = -500  # Extreme value

        # Test Z-score method
        outlier_mask, outlier_stats = self.analyzer.detect_outliers(
            data_with_outliers, method='zscore'
        )
        self.assertTrue(outlier_mask.iloc[50, 0])  # Should detect the outlier
        self.assertTrue(outlier_mask.iloc[100, 2])  # Should detect the outlier

        # Test IQR method
        outlier_mask, outlier_stats = self.analyzer.detect_outliers(
            data_with_outliers, method='iqr'
        )
        self.assertTrue(outlier_mask.iloc[50, 0])  # Should detect the outlier
        self.assertTrue(outlier_mask.iloc[100, 2])  # Should detect the outlier

    def test_normalize_data(self):
        """Test data normalization methods."""
        # Test MinMax scaling
        normalized_data = self.analyzer.normalize_data(self.data, method='minmax')
        numeric_columns = normalized_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            self.assertTrue(normalized_data[column].min() >= 0)
            self.assertTrue(normalized_data[column].max() <= 1)

        # Test Standard scaling
        normalized_data = self.analyzer.normalize_data(self.data, method='standard')
        for column in numeric_columns:
            # Allow for numerical precision issues
            self.assertTrue(abs(normalized_data[column].mean()) < 1e-6)
            self.assertTrue(abs(normalized_data[column].std() - 1) < 1e-6)

    def test_check_stationarity(self):
        """Test stationarity checks."""
        # Create non-stationary data (with trend)
        trend_data = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        stationarity_results = self.analyzer.check_stationarity(trend_data)
        
        # Check structure
        self.assertIn('adf_test', stationarity_results)
        self.assertIn('kpss_test', stationarity_results)

        # Check values
        self.assertIn('statistic', stationarity_results['adf_test'])
        self.assertIn('p_value', stationarity_results['adf_test'])
        self.assertIn('is_stationary', stationarity_results['adf_test'])

    def test_integration_with_yahoo_data(self):
        """Test analyzer with real Yahoo Finance data."""
        from ...pipeline.data_ingestion.fetchers.yahoo_finance import YahooFinanceFetcher
        
        # Fetch real data
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
            # Test full analysis pipeline
            stats = self.analyzer.calculate_statistics(data)
            missing_stats = self.analyzer.analyze_missing_values(data)
            filled_data = self.analyzer.handle_missing_values(data)
            outlier_mask, outlier_stats = self.analyzer.detect_outliers(filled_data)
            normalized_data = self.analyzer.normalize_data(filled_data)
            stationarity_results = self.analyzer.check_stationarity(filled_data['Close'])

            # Verify results
            self.assertIsInstance(stats, dict)
            self.assertIsInstance(missing_stats, dict)
            self.assertIsInstance(filled_data, pd.DataFrame)
            self.assertIsInstance(outlier_mask, pd.DataFrame)
            self.assertIsInstance(normalized_data, pd.DataFrame)
            self.assertIsInstance(stationarity_results, dict) 