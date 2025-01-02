"""
Technical Indicators Manager Module

This module provides a centralized manager for calculating and managing various technical indicators
including SMA, EMA, RSI, MACD, Bollinger Bands, and ATR.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class IndicatorsManager:
    """
    Manages the calculation and storage of technical indicators for financial data.
    
    This class serves as the main interface for computing and accessing technical indicators,
    providing methods to calculate individual indicators and batch process multiple indicators
    for given financial data.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the IndicatorsManager.
        
        Args:
            data (Optional[pd.DataFrame]): Initial financial data with columns: 
                ['Open', 'High', 'Low', 'Close', 'Volume']. Can be None if data will be set later.
        """
        self.data = data
        self._validate_data() if data is not None else None
        self.indicators: Dict[str, pd.DataFrame] = {}
        
    def _validate_data(self) -> None:
        """Validate that the input data has the required columns."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set or update the financial data for indicator calculations.
        
        Args:
            data (pd.DataFrame): Financial data with required columns.
        """
        self.data = data
        self._validate_data()
        # Clear existing indicators as they need to be recalculated
        self.indicators.clear()
        
    def calculate_all_indicators(self, 
                               periods: Dict[str, Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Calculate all supported technical indicators.
        
        Args:
            periods (Dict[str, Union[int, List[int]]], optional): Dictionary of indicator periods.
                Example: {'sma': [20, 50, 200], 'rsi': 14, 'macd': [12, 26, 9]}
        
        Returns:
            pd.DataFrame: DataFrame containing all calculated indicators.
        """
        if periods is None:
            periods = {
                'sma': [20, 50, 200],
                'ema': [12, 26],
                'rsi': 14,
                'macd': [12, 26, 9],
                'bollinger': 20,
                'atr': 14
            }
            
        # Calculate each indicator
        self.calculate_sma(periods['sma'])
        self.calculate_ema(periods['ema'])
        self.calculate_rsi(periods['rsi'])
        self.calculate_macd(*periods['macd'])
        self.calculate_bollinger_bands(periods['bollinger'])
        self.calculate_atr(periods['atr'])
        
        # Combine all indicators
        result = pd.concat(self.indicators.values(), axis=1)
        return result
    
    def calculate_sma(self, periods: Union[int, List[int]]) -> pd.DataFrame:
        """
        Calculate Simple Moving Average for specified periods.
        
        Args:
            periods (Union[int, List[int]]): Period(s) for SMA calculation.
            
        Returns:
            pd.DataFrame: DataFrame with SMA values.
        """
        if isinstance(periods, int):
            periods = [periods]
            
        sma_data = pd.DataFrame(index=self.data.index)
        for period in periods:
            sma = self.data['Close'].rolling(window=period).mean()
            sma_data[f'SMA_{period}'] = sma
            
        self.indicators['sma'] = sma_data
        return sma_data
    
    def calculate_ema(self, periods: Union[int, List[int]]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average for specified periods.
        
        Args:
            periods (Union[int, List[int]]): Period(s) for EMA calculation.
            
        Returns:
            pd.DataFrame: DataFrame with EMA values.
        """
        if isinstance(periods, int):
            periods = [periods]
            
        ema_data = pd.DataFrame(index=self.data.index)
        for period in periods:
            ema = self.data['Close'].ewm(span=period, adjust=False).mean()
            ema_data[f'EMA_{period}'] = ema
            
        self.indicators['ema'] = ema_data
        return ema_data
    
    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            period (int): Period for RSI calculation. Default is 14.
            
        Returns:
            pd.DataFrame: DataFrame with RSI values.
        """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_data = pd.DataFrame(index=self.data.index)
        rsi_data['RSI'] = rsi
        
        self.indicators['rsi'] = rsi_data
        return rsi_data
    
    def calculate_macd(self, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            fast_period (int): Fast EMA period. Default is 12.
            slow_period (int): Slow EMA period. Default is 26.
            signal_period (int): Signal line period. Default is 9.
            
        Returns:
            pd.DataFrame: DataFrame with MACD values.
        """
        fast_ema = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        macd_data = pd.DataFrame(index=self.data.index)
        macd_data['MACD_Line'] = macd_line
        macd_data['Signal_Line'] = signal_line
        macd_data['MACD_Histogram'] = macd_histogram
        
        self.indicators['macd'] = macd_data
        return macd_data
    
    def calculate_bollinger_bands(self, 
                                period: int = 20, 
                                num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            period (int): Period for moving average. Default is 20.
            num_std (float): Number of standard deviations. Default is 2.
            
        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands values.
        """
        sma = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        
        bb_data = pd.DataFrame(index=self.data.index)
        bb_data['BB_Middle'] = sma
        bb_data['BB_Upper'] = upper_band
        bb_data['BB_Lower'] = lower_band
        
        self.indicators['bollinger'] = bb_data
        return bb_data
    
    def calculate_atr(self, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            period (int): Period for ATR calculation. Default is 14.
            
        Returns:
            pd.DataFrame: DataFrame with ATR values.
        """
        high_low = self.data['High'] - self.data['Low']
        high_close = abs(self.data['High'] - self.data['Close'].shift())
        low_close = abs(self.data['Low'] - self.data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        atr_data = pd.DataFrame(index=self.data.index)
        atr_data['ATR'] = atr
        
        self.indicators['atr'] = atr_data
        return atr_data
    
    def get_indicator(self, indicator_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve calculated indicator data.
        
        Args:
            indicator_name (str): Name of the indicator to retrieve.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing the indicator data if available.
        """
        return self.indicators.get(indicator_name)
    
    def get_all_indicators(self) -> pd.DataFrame:
        """
        Retrieve all calculated indicators.
        
        Returns:
            pd.DataFrame: DataFrame containing all calculated indicators.
        """
        if not self.indicators:
            return pd.DataFrame()
        return pd.concat(self.indicators.values(), axis=1) 