"""
Categorization Manager Module

This module provides high-level management of the stock categorization process,
coordinating data loading, sorting, and result storage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress

from .sort_algorithms import StockSorter
from ..data_visualization.visualization import DataVisualizer

logger = logging.getLogger(__name__)
console = Console()

class CategorizationManager:
    """Manages the stock categorization workflow."""
    
    def __init__(
        self,
        input_dir: str = 'data/processed',
        output_dir: str = 'data/processed/sorted',
        consistency_threshold: float = 0.8,
        volatility_threshold: float = 0.05
    ):
        """
        Initialize the CategorizationManager.

        Args:
            input_dir (str): Directory containing processed stock data
            output_dir (str): Directory to store sorted results
            consistency_threshold (float): R² threshold for high/low consistency
            volatility_threshold (float): Standard deviation threshold for high/low volatility
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.sorter = StockSorter(
            consistency_threshold=consistency_threshold,
            volatility_threshold=volatility_threshold,
            output_dir=str(self.output_dir)
        )
        
        self.visualizer = DataVisualizer(
            output_dir=str(self.output_dir / 'visualizations')
        )

    def load_stock_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock data from the input directory.

        Args:
            symbols (List[str], optional): List of stock symbols to load

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded stock data
        """
        stock_data = {}
        
        try:
            # List all CSV files in input directory
            csv_files = list(self.input_dir.glob('*.csv'))
            
            if symbols:
                # Filter files by symbols
                csv_files = [f for f in csv_files if f.stem in symbols]
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Loading stock data...", total=len(csv_files))
                
                for file_path in csv_files:
                    try:
                        symbol = file_path.stem
                        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        # Ensure no negative values for logarithmic operations
                        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].abs()
                        stock_data[symbol] = data
                        progress.advance(task)
                        
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {str(e)}")
                        continue
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading stock data: {str(e)}")
            return {}

    def categorize_stocks(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load and categorize stocks.

        Args:
            symbols (List[str], optional): List of stock symbols to categorize

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Categorized stocks
        """
        try:
            # Load stock data
            console.print("[bold cyan]Loading stock data...")
            stock_data = self.load_stock_data(symbols)
            
            if not stock_data:
                logger.error("No stock data loaded")
                return {}
            
            # Sort stocks
            console.print("[bold cyan]Categorizing stocks...")
            categorized_stocks = self.sorter.sort_stocks(stock_data)
            
            # Generate visualizations
            console.print("[bold cyan]Generating category visualizations...")
            self._generate_category_visualizations(categorized_stocks)
            
            return categorized_stocks
            
        except Exception as e:
            logger.error(f"Error in categorization process: {str(e)}")
            return {}

    def _generate_category_visualizations(self, categorized_stocks: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Generate visualizations for each category.

        Args:
            categorized_stocks (Dict[str, Dict[str, pd.DataFrame]]): Categorized stock data
        """
        try:
            for category, stocks in categorized_stocks.items():
                if not stocks:
                    continue
                
                # Create category visualization directory
                vis_dir = self.output_dir / 'visualizations' / category
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate visualizations for each stock in category
                for symbol, data in stocks.items():
                    try:
                        # Ensure data is suitable for logarithmic operations
                        data = data.copy()
                        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].abs()
                        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].replace(0, np.nan)
                        
                        # Price trends
                        self.visualizer.plot_price_trends(
                            data=data,
                            symbol=f"{symbol}_{category}"
                        )
                        
                        # Distribution analysis
                        self.visualizer.plot_distribution_analysis(
                            data=data,
                            symbol=f"{symbol}_{category}"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error generating visualizations for {symbol}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error generating category visualizations: {str(e)}")

    def get_category_summary(self) -> pd.DataFrame:
        """
        Get summary of categorized stocks.

        Returns:
            pd.DataFrame: Summary statistics for each category
        """
        try:
            summary_path = self.output_dir / 'sorting_summary.csv'
            if not summary_path.exists():
                logger.error("Summary file not found")
                return pd.DataFrame()
            
            summary_df = pd.read_csv(summary_path)
            
            # Calculate category statistics
            category_summary = summary_df.groupby('category').agg({
                'symbol': 'count',
                'consistency': ['mean', 'std'],
                'volatility': ['mean', 'std']
            }).round(4)
            
            category_summary.columns = [
                'stock_count',
                'avg_consistency',
                'std_consistency',
                'avg_volatility',
                'std_volatility'
            ]
            
            return category_summary
            
        except Exception as e:
            logger.error(f"Error getting category summary: {str(e)}")
            return pd.DataFrame() 