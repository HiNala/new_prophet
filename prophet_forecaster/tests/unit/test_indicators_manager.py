"""
Unit tests for the Technical Indicators Manager module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from prophet_forecaster.pipeline.technical_indicators.indicators_manager import IndicatorsManager

class TestIndicatorsManager(unittest.TestCase):
    """Test cases for the IndicatorsManager class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic price data
        close_prices = np.random.normal(100, 10, len(dates)).cumsum()
        high_prices = close_prices + np.random.uniform(0, 5, len(dates))
        low_prices = close_prices - np.random.uniform(0, 5, len(dates))
        open_prices = close_prices + np.random.uniform(-3, 3, len(dates))
        volume = np.random.uniform(1000000, 5000000, len(dates))
        
        self.test_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
        
        self.manager = IndicatorsManager(self.test_data)
        
    def test_initialization(self):
        """Test IndicatorsManager initialization."""
        self.assertIsNotNone(self.manager.data)
        self.assertEqual(len(self.manager.indicators), 0)
        
    def test_data_validation(self):
        """Test data validation."""
        # Test with missing columns
        invalid_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        with self.assertRaises(ValueError):
            IndicatorsManager(invalid_data)
            
    def test_calculate_sma(self):
        """Test SMA calculation."""
        periods = [20, 50]
        sma_data = self.manager.calculate_sma(periods)
        
        # Check if SMA columns exist
        self.assertTrue(all(f'SMA_{period}' in sma_data.columns for period in periods))
        
        # Check if SMA values are correct for a known sequence
        known_prices = pd.Series([1, 2, 3, 4, 5])
        expected_sma = pd.Series([None, None, None, None, 3.0])  # 3-day SMA
        test_data = pd.DataFrame({'Close': known_prices})
        test_manager = IndicatorsManager(test_data)
        actual_sma = test_manager.calculate_sma(3)['SMA_3'].iloc[-1]
        self.assertAlmostEqual(actual_sma, expected_sma.iloc[-1])
        
    def test_calculate_ema(self):
        """Test EMA calculation."""
        periods = [12, 26]
        ema_data = self.manager.calculate_ema(periods)
        
        # Check if EMA columns exist
        self.assertTrue(all(f'EMA_{period}' in ema_data.columns for period in periods))
        
        # Verify EMA is more responsive to recent prices than SMA
        last_ema = ema_data[f'EMA_{periods[0]}'].iloc[-1]
        last_sma = self.manager.calculate_sma(periods[0])['SMA_12'].iloc[-1]
        self.assertNotEqual(last_ema, last_sma)
        
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi_data = self.manager.calculate_rsi()
        
        # Check if RSI column exists
        self.assertTrue('RSI' in rsi_data.columns)
        
        # Verify RSI values are between 0 and 100
        self.assertTrue(all(0 <= x <= 100 for x in rsi_data['RSI'].dropna()))
        
    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_data = self.manager.calculate_macd()
        
        # Check if all MACD components exist
        required_columns = ['MACD_Line', 'Signal_Line', 'MACD_Histogram']
        self.assertTrue(all(col in macd_data.columns for col in required_columns))
        
        # Verify MACD histogram is the difference between MACD and signal line
        pd.testing.assert_series_equal(
            macd_data['MACD_Histogram'],
            macd_data['MACD_Line'] - macd_data['Signal_Line']
        )
        
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_data = self.manager.calculate_bollinger_bands()
        
        # Check if all Bollinger Bands components exist
        required_columns = ['BB_Middle', 'BB_Upper', 'BB_Lower']
        self.assertTrue(all(col in bb_data.columns for col in required_columns))
        
        # Verify upper band is always higher than middle band
        self.assertTrue(all(bb_data['BB_Upper'] >= bb_data['BB_Middle']))
        
        # Verify lower band is always lower than middle band
        self.assertTrue(all(bb_data['BB_Lower'] <= bb_data['BB_Middle']))
        
    def test_calculate_atr(self):
        """Test ATR calculation."""
        atr_data = self.manager.calculate_atr()
        
        # Check if ATR column exists
        self.assertTrue('ATR' in atr_data.columns)
        
        # Verify ATR is always positive
        self.assertTrue(all(atr_data['ATR'].dropna() >= 0))
        
    def test_calculate_all_indicators(self):
        """Test calculation of all indicators."""
        all_data = self.manager.calculate_all_indicators()
        
        # Verify all indicators are calculated
        expected_indicators = [
            'SMA_20', 'SMA_50', 'SMA_200',  # SMA
            'EMA_12', 'EMA_26',             # EMA
            'RSI',                          # RSI
            'MACD_Line', 'Signal_Line',     # MACD
            'BB_Middle', 'BB_Upper', 'BB_Lower',  # Bollinger Bands
            'ATR'                           # ATR
        ]
        
        self.assertTrue(all(indicator in all_data.columns 
                          for indicator in expected_indicators))
        
    def test_get_indicator(self):
        """Test retrieving specific indicators."""
        # Calculate some indicators
        self.manager.calculate_sma([20])
        self.manager.calculate_rsi()
        
        # Test getting existing indicator
        sma_data = self.manager.get_indicator('sma')
        self.assertIsNotNone(sma_data)
        self.assertTrue('SMA_20' in sma_data.columns)
        
        # Test getting non-existent indicator
        self.assertIsNone(self.manager.get_indicator('non_existent'))
        
    def test_get_all_indicators(self):
        """Test retrieving all indicators."""
        # Calculate some indicators
        self.manager.calculate_all_indicators()
        
        # Get all indicators
        all_indicators = self.manager.get_all_indicators()
        
        # Verify we get a non-empty DataFrame
        self.assertFalse(all_indicators.empty)
        
        # Verify all calculated indicators are included
        self.assertTrue(len(all_indicators.columns) > 0)
        
if __name__ == '__main__':
    unittest.main() 