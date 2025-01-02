"""
Data Visualization Module

This module provides functionality for creating exploratory visualizations
of financial data, including trends, distributions, and missing value patterns.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class DataVisualizer:
    """Handles creation and management of data visualizations."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the data visualizer.

        Args:
            output_dir (str, optional): Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

    def plot_price_trends(self, data: pd.DataFrame, symbol: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> str:
        """
        Create a comprehensive price trend visualization.

        Args:
            data (pd.DataFrame): Stock data DataFrame
            symbol (str): Stock symbol
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering

        Returns:
            str: Path to saved visualization
        """
        try:
            # Filter data if dates provided
            if start_date and end_date:
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data[mask]

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
            fig.suptitle(f'{symbol} Stock Analysis', fontsize=16)

            # Price and Volume Plot
            axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
            axes[0].fill_between(data.index, data['High'], data['Low'], alpha=0.2, color='gray')
            axes[0].set_title('Price Movement')
            axes[0].set_ylabel('Price')
            axes[0].grid(True)
            axes[0].legend()

            # Volume Plot
            axes[1].bar(data.index, data['Volume'], color='purple', alpha=0.6)
            axes[1].set_title('Trading Volume')
            axes[1].set_ylabel('Volume')
            axes[1].grid(True)

            # Returns Plot
            returns = data['Close'].pct_change()
            axes[2].plot(data.index, returns, color='green', label='Daily Returns')
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[2].set_title('Daily Returns')
            axes[2].set_ylabel('Returns (%)')
            axes[2].grid(True)
            axes[2].legend()

            # Adjust layout and save
            plt.tight_layout()
            save_path = self.output_dir / f'{symbol}_price_analysis.png'
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Price trend visualization saved to {save_path}")
            return str(save_path.absolute())

        except Exception as e:
            logger.error(f"Error creating price trend visualization: {str(e)}")
            return ""

    def plot_distribution_analysis(self, data: pd.DataFrame, symbol: str) -> str:
        """
        Create distribution analysis visualizations.

        Args:
            data (pd.DataFrame): Stock data DataFrame
            symbol (str): Stock symbol

        Returns:
            str: Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{symbol} Distribution Analysis', fontsize=16)

            # Returns Distribution
            returns = data['Close'].pct_change().dropna()
            sns.histplot(returns, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Returns Distribution')
            axes[0, 0].set_xlabel('Returns')
            
            # Volume Distribution
            sns.histplot(np.log10(data['Volume']), kde=True, ax=axes[0, 1])
            axes[0, 1].set_title('Volume Distribution (log10)')
            axes[0, 1].set_xlabel('Log10(Volume)')

            # Price Range Box Plot
            price_data = pd.DataFrame({
                'Open': data['Open'],
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            sns.boxplot(data=price_data, ax=axes[1, 0])
            axes[1, 0].set_title('Price Range Analysis')
            axes[1, 0].set_ylabel('Price')

            # QQ Plot for Returns
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Returns Q-Q Plot')

            # Adjust layout and save
            plt.tight_layout()
            save_path = self.output_dir / f'{symbol}_distribution_analysis.png'
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Distribution analysis saved to {save_path}")
            return str(save_path.absolute())

        except Exception as e:
            logger.error(f"Error creating distribution analysis: {str(e)}")
            return ""

    def plot_missing_values(self, data: pd.DataFrame, symbol: str) -> str:
        """
        Create missing values visualization.

        Args:
            data (pd.DataFrame): Stock data DataFrame
            symbol (str): Stock symbol

        Returns:
            str: Path to saved visualization
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate missing values
            missing = data.isnull().sum()
            missing_pct = (missing / len(data)) * 100

            # Create missing values plot
            sns.barplot(x=missing_pct.index, y=missing_pct.values)
            plt.title(f'{symbol} Missing Values Analysis')
            plt.xlabel('Features')
            plt.ylabel('Missing Values (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Save plot
            save_path = self.output_dir / f'{symbol}_missing_values.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            logger.info(f"Missing values analysis saved to {save_path}")
            return str(save_path.absolute())

        except Exception as e:
            logger.error(f"Error creating missing values visualization: {str(e)}")
            return ""

    def generate_summary_report(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Generate a statistical summary report.

        Args:
            data (pd.DataFrame): Stock data DataFrame
            symbol (str): Stock symbol

        Returns:
            Dict: Summary statistics and insights
        """
        try:
            summary = {
                'symbol': symbol,
                'date_range': {
                    'start': data.index.min().strftime('%Y-%m-%d'),
                    'end': data.index.max().strftime('%Y-%m-%d'),
                    'trading_days': len(data)
                },
                'price_statistics': {
                    'mean': data['Close'].mean(),
                    'std': data['Close'].std(),
                    'min': data['Close'].min(),
                    'max': data['Close'].max(),
                    'current': data['Close'].iloc[-1],
                    'daily_returns_mean': data['Close'].pct_change().mean(),
                    'daily_returns_std': data['Close'].pct_change().std()
                },
                'volume_statistics': {
                    'mean': data['Volume'].mean(),
                    'std': data['Volume'].std(),
                    'min': data['Volume'].min(),
                    'max': data['Volume'].max()
                },
                'missing_values': data.isnull().sum().to_dict()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {} 