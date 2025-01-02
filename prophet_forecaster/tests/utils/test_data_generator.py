"""
Test Data Generator Module

This module provides utilities to generate test data for various components
of the Prophet Forecaster application.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class TestDataGenerator:
    """Generates test data for various components."""

    @staticmethod
    def generate_stock_data(
        start_date: str = '2023-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'D',
        with_trends: bool = True,
        with_seasonality: bool = True,
        with_noise: bool = True,
        noise_scale: float = 2.0,
        trend_scale: float = 20.0,
        seasonality_scale: float = 10.0
    ) -> pd.DataFrame:
        """
        Generate synthetic stock data with optional trends and seasonality.

        Args:
            start_date (str): Start date for the data
            end_date (str): End date for the data
            freq (str): Frequency of data points ('D' for daily)
            with_trends (bool): Include trend components
            with_seasonality (bool): Include seasonal components
            with_noise (bool): Add random noise
            noise_scale (float): Scale factor for noise
            trend_scale (float): Scale factor for trend
            seasonality_scale (float): Scale factor for seasonality

        Returns:
            pd.DataFrame: Generated stock data
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_points = len(dates)
        
        # Base price around 100
        base_price = 100
        
        # Generate components
        trend = np.linspace(0, trend_scale, n_points) if with_trends else 0
        seasonality = seasonality_scale * np.sin(2 * np.pi * np.arange(n_points) / 252) if with_seasonality else 0
        noise = np.random.normal(0, noise_scale, n_points) if with_noise else 0
        
        # Combine components
        close_prices = base_price + trend + seasonality + noise
        
        # Generate other price components
        high_prices = close_prices + np.random.uniform(0, 2, n_points)
        low_prices = close_prices - np.random.uniform(0, 2, n_points)
        open_prices = close_prices + np.random.uniform(-1, 1, n_points)
        
        # Generate volume
        volume = np.random.randint(1000, 10000, n_points)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
        
        return data

    @staticmethod
    def generate_technical_indicator_data() -> Tuple[pd.DataFrame, Dict]:
        """
        Generate data specifically for testing technical indicators.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Stock data and expected indicator values
        """
        # Generate base stock data
        data = TestDataGenerator.generate_stock_data(
            start_date='2023-01-01',
            end_date='2023-12-31',
            with_trends=True,
            with_seasonality=True
        )
        
        # Calculate expected indicator values
        expected_values = {
            'sma_20': data['Close'].rolling(window=20).mean(),
            'sma_50': data['Close'].rolling(window=50).mean(),
            'rsi_14': TestDataGenerator._calculate_rsi(data['Close'], period=14),
            'macd': TestDataGenerator._calculate_macd(data['Close'])
        }
        
        return data, expected_values

    @staticmethod
    def generate_sorting_categories() -> Dict[str, pd.DataFrame]:
        """
        Generate test data for different stock categories.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of categorized stock data
        """
        categories = {
            'high_consistency_low_volatility': TestDataGenerator.generate_stock_data(
                with_trends=True,
                with_seasonality=False,
                with_noise=True,
                noise_scale=1.0,  # Low volatility
                trend_scale=30.0  # Strong trend for high consistency
            ),
            'low_consistency_high_volatility': TestDataGenerator.generate_stock_data(
                with_trends=False,
                with_seasonality=True,
                with_noise=True,
                noise_scale=8.0,  # High volatility
                seasonality_scale=20.0  # Strong seasonality for low consistency
            ),
            'trending_upward': TestDataGenerator.generate_stock_data(
                with_trends=True,
                with_seasonality=False,
                with_noise=True,
                trend_scale=40.0  # Strong upward trend
            ),
            'trending_downward': -1 * TestDataGenerator.generate_stock_data(
                with_trends=True,
                with_seasonality=False,
                with_noise=True,
                trend_scale=40.0  # Strong downward trend
            )
        }
        return categories

    @staticmethod
    def generate_ensemble_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate test data for ensemble learning, including features and targets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target data
        """
        # Generate base data with extra day for target shift
        end_date = (datetime.strptime('2023-12-31', '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        data = TestDataGenerator.generate_stock_data(end_date=end_date)
        
        # Create features
        features = pd.DataFrame({
            'sma_20': data['Close'].rolling(window=20).mean(),
            'sma_50': data['Close'].rolling(window=50).mean(),
            'rsi_14': TestDataGenerator._calculate_rsi(data['Close'], period=14),
            'volume_ma': data['Volume'].rolling(window=20).mean(),
            'returns': data['Close'].pct_change(),
            'volatility': data['Close'].rolling(window=20).std()
        })
        
        # Create target (next day's return)
        target = data['Close'].pct_change().shift(-1)
        
        # Drop any rows with NaN values to ensure alignment
        features = features.dropna()
        target = target[features.index]
        
        # Remove the last row which will have NaN target due to shift
        features = features[:-1]
        target = target[:-1]
        
        return features, target

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for test data."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(prices: pd.Series) -> pd.Series:
        """Calculate MACD for test data."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2 