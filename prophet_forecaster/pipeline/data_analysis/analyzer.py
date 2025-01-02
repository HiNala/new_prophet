"""
Data Analysis Module

This module provides functionality for analyzing financial data, including
statistical analysis, missing value handling, and outlier detection.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Handles data analysis and preprocessing tasks."""

    def __init__(self):
        """Initialize the data analyzer."""
        self.scaler = None
        self.imputer = None

    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate descriptive statistics for the dataset.

        Args:
            data (pd.DataFrame): Input financial data

        Returns:
            Dict: Dictionary containing various statistical measures
        """
        try:
            stats_dict = {
                'basic_stats': {
                    'mean': data.mean().to_dict(),
                    'median': data.median().to_dict(),
                    'std': data.std().to_dict(),
                    'var': data.var().to_dict()
                },
                'distribution_stats': {
                    'skewness': data.skew().to_dict(),
                    'kurtosis': data.kurtosis().to_dict()
                },
                'range_stats': {
                    'min': data.min().to_dict(),
                    'max': data.max().to_dict(),
                    'quantiles': {
                        '25%': data.quantile(0.25).to_dict(),
                        '50%': data.quantile(0.50).to_dict(),
                        '75%': data.quantile(0.75).to_dict()
                    }
                }
            }
            return stats_dict

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}

    def analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """
        Analyze missing values in the dataset.

        Args:
            data (pd.DataFrame): Input financial data

        Returns:
            Dict: Missing value analysis results
        """
        try:
            missing_stats = {
                'total_missing': data.isnull().sum().to_dict(),
                'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
                'missing_patterns': {
                    'consecutive_missing': self._find_consecutive_missing(data)
                }
            }
            return missing_stats

        except Exception as e:
            logger.error(f"Error analyzing missing values: {str(e)}")
            return {}

    def _find_consecutive_missing(self, data: pd.DataFrame) -> Dict:
        """Find consecutive missing values in each column."""
        consecutive_missing = {}
        for column in data.columns:
            mask = data[column].isnull()
            if mask.any():
                consecutive_missing[column] = self._get_consecutive_ranges(mask)
        return consecutive_missing

    def _get_consecutive_ranges(self, mask: pd.Series) -> list:
        """Get ranges of consecutive True values in a boolean mask."""
        ranges = []
        start = None
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                ranges.append((start, i - 1))
                start = None
        if start is not None:
            ranges.append((start, len(mask) - 1))
        return ranges

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'ffill',
                            k_neighbors: int = 5) -> pd.DataFrame:
        """
        Handle missing values using specified method.

        Args:
            data (pd.DataFrame): Input financial data
            method (str): Method to handle missing values
                         ('ffill', 'bfill', 'interpolate', 'knn')
            k_neighbors (int): Number of neighbors for KNN imputation

        Returns:
            pd.DataFrame: Data with handled missing values
        """
        try:
            if method == 'ffill':
                return data.ffill()
            elif method == 'bfill':
                return data.bfill()
            elif method == 'interpolate':
                return data.interpolate(method='linear')
            elif method == 'knn':
                self.imputer = KNNImputer(n_neighbors=k_neighbors)
                imputed_data = self.imputer.fit_transform(data)
                return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
            else:
                logger.warning(f"Unknown method {method}, using forward fill")
                return data.ffill()

        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return data

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                       threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect outliers in the dataset.

        Args:
            data (pd.DataFrame): Input financial data
            method (str): Method to detect outliers ('zscore', 'iqr')
            threshold (float): Threshold for outlier detection

        Returns:
            Tuple[pd.DataFrame, Dict]: Mask of outliers and outlier statistics
        """
        try:
            outlier_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
            outlier_stats = {}

            for column in data.select_dtypes(include=[np.number]).columns:
                if method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[column].dropna()))
                    column_mask = z_scores > threshold
                elif method == 'iqr':
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    column_mask = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
                else:
                    logger.warning(f"Unknown method {method}, using zscore")
                    z_scores = np.abs(stats.zscore(data[column].dropna()))
                    column_mask = z_scores > threshold

                outlier_mask[column] = column_mask
                outlier_stats[column] = {
                    'num_outliers': column_mask.sum(),
                    'outlier_percentage': (column_mask.sum() / len(data) * 100)
                }

            return outlier_mask, outlier_stats

        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return pd.DataFrame(False, index=data.index, columns=data.columns), {}

    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax',
                       feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        Normalize the dataset.

        Args:
            data (pd.DataFrame): Input financial data
            method (str): Normalization method ('minmax', 'standard')
            feature_range (Tuple[float, float]): Range for MinMaxScaler

        Returns:
            pd.DataFrame: Normalized data
        """
        try:
            # Create a copy of the data to avoid modifying the original
            data_copy = data.copy()
            
            # Scale only numeric columns
            numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                for column in numeric_columns:
                    # Handle NaN values before scaling
                    column_data = data_copy[column].ffill().bfill()
                    
                    # Ensure data is finite before scaling
                    column_data = np.clip(column_data, -np.inf, np.inf)
                    
                    if method == 'minmax':
                        # MinMax scaling
                        min_val = column_data.min()
                        max_val = column_data.max()
                        if max_val > min_val:
                            scaled_data = (column_data - min_val) / (max_val - min_val)
                            scaled_data = scaled_data * (feature_range[1] - feature_range[0]) + feature_range[0]
                        else:
                            scaled_data = np.zeros_like(column_data) + feature_range[0]
                    elif method == 'standard':
                        # Standard scaling
                        mean_val = column_data.mean()
                        std_val = column_data.std()
                        if std_val > 0:
                            scaled_data = (column_data - mean_val) / std_val
                        else:
                            scaled_data = np.zeros_like(column_data)
                    else:
                        logger.warning(f"Unknown method {method}, using minmax")
                        scaled_data = column_data  # Return original data if method is unknown
                    
                    data_copy[column] = scaled_data

            return data_copy

        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data

    def check_stationarity(self, data: pd.Series) -> Dict:
        """
        Check stationarity of a time series.

        Args:
            data (pd.Series): Input time series data

        Returns:
            Dict: Stationarity test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # ADF Test
            adf_result = adfuller(data.dropna())
            
            # KPSS Test
            kpss_result = kpss(data.dropna())

            stationarity_results = {
                'adf_test': {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                },
                'kpss_test': {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'is_stationary': kpss_result[1] > 0.05
                }
            }

            return stationarity_results

        except Exception as e:
            logger.error(f"Error checking stationarity: {str(e)}")
            return {} 