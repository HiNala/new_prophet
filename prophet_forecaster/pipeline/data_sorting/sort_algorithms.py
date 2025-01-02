"""
Stock Sorting Algorithms Module

This module provides algorithms for categorizing stocks based on their
consistency and volatility characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StockSorter:
    """Sorts stocks into categories based on their characteristics."""
    
    # Category definitions
    CATEGORIES = {
        'HCLV': 'High Consistency, Low Volatility',
        'HCHV': 'High Consistency, High Volatility',
        'LCLV': 'Low Consistency, Low Volatility',
        'LCHV': 'Low Consistency, High Volatility'
    }
    
    def __init__(
        self,
        consistency_threshold: float = 0.8,
        volatility_threshold: float = 0.05,
        output_dir: str = 'data/processed/sorted'
    ):
        """
        Initialize the StockSorter.

        Args:
            consistency_threshold (float): R² threshold for high/low consistency
            volatility_threshold (float): Standard deviation threshold for high/low volatility
            output_dir (str): Directory to store sorted data
        """
        self.consistency_threshold = consistency_threshold
        self.volatility_threshold = volatility_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directories and documentation
        for category in self.CATEGORIES:
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            (category_dir / 'summary').mkdir(exist_ok=True)
            
            # Create README for category
            readme_path = category_dir / 'README.md'
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"# {self.CATEGORIES[category]}\n\n")
                    f.write("## Category Description\n")
                    f.write(self._get_category_description(category))
                    f.write("\n\n## Metrics Used\n")
                    f.write("- **Consistency Metrics:**\n")
                    f.write("  - R² of linear regression\n")
                    f.write("  - Mean Absolute Deviation (MAD)\n")
                    f.write("- **Volatility Metrics:**\n")
                    f.write("  - Standard deviation of returns\n")
                    f.write("  - Coefficient of Variation (CV)\n")

    def _get_category_description(self, category: str) -> str:
        """Get detailed description for a category."""
        descriptions = {
            'HCLV': "Stocks with high predictability and low risk. These are ideal candidates for Prophet forecasting due to their stable trends and consistent behavior.",
            'HCHV': "Stocks with clear trends but significant price swings. May require volatility preprocessing before Prophet forecasting.",
            'LCLV': "Stocks with less predictable patterns but stable prices. Consider alternative forecasting methods or feature engineering.",
            'LCHV': "Highly unpredictable stocks with large price swings. Most challenging for forecasting; requires sophisticated preprocessing."
        }
        return descriptions.get(category, "")

    def calculate_consistency(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate consistency metrics using R² and MAD.

        Args:
            data (pd.DataFrame): Stock price data

        Returns:
            Dict[str, float]: Dictionary of consistency metrics
        """
        try:
            # Calculate R² using linear regression
            X = np.arange(len(data)).reshape(-1, 1)
            y = data['Close'].values
            model = LinearRegression()
            model.fit(X, y)
            r2_score = model.score(X, y)
            
            # Calculate Mean Absolute Deviation
            price_series = data['Close']
            mad = np.mean(np.abs(price_series - price_series.mean()))
            normalized_mad = mad / price_series.mean()  # Normalize by mean price
            
            return {
                'r2_score': r2_score,
                'normalized_mad': normalized_mad
            }
            
        except Exception as e:
            logger.error(f"Error calculating consistency metrics: {str(e)}")
            return {'r2_score': 0.0, 'normalized_mad': float('inf')}

    def calculate_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volatility using standard deviation and CV.

        Args:
            data (pd.DataFrame): Stock price data

        Returns:
            Dict[str, float]: Dictionary of volatility metrics
        """
        try:
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Standard deviation of returns
            returns_std = returns.std()
            
            # Calculate Coefficient of Variation
            price_series = data['Close']
            cv = price_series.std() / price_series.mean()
            
            return {
                'returns_std': returns_std,
                'cv': cv
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {'returns_std': float('inf'), 'cv': float('inf')}

    def categorize_stock(self, data: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Categorize a stock based on its consistency and volatility.

        Args:
            data (pd.DataFrame): Stock price data

        Returns:
            Tuple[str, Dict[str, float]]: Category and metrics
        """
        # Calculate all metrics
        consistency_metrics = self.calculate_consistency(data)
        volatility_metrics = self.calculate_volatility(data)
        
        # Determine category using primary metrics
        is_high_consistency = consistency_metrics['r2_score'] >= self.consistency_threshold
        is_high_volatility = volatility_metrics['returns_std'] >= self.volatility_threshold
        
        if is_high_consistency and not is_high_volatility:
            category = 'HCLV'
        elif is_high_consistency and is_high_volatility:
            category = 'HCHV'
        elif not is_high_consistency and not is_high_volatility:
            category = 'LCLV'
        else:
            category = 'LCHV'
        
        # Combine all metrics
        metrics = {
            **consistency_metrics,
            **volatility_metrics
        }
        
        return category, metrics

    def sort_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Sort multiple stocks into categories.

        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock data by symbol

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Categorized stocks
        """
        categorized_stocks = {category: {} for category in self.CATEGORIES}
        summary_data = []
        
        for symbol, data in stock_data.items():
            try:
                # Categorize stock
                category, metrics = self.categorize_stock(data)
                
                # Store stock in category
                categorized_stocks[category][symbol] = data
                
                # Save to category directory
                output_path = self.output_dir / category / f"{symbol}_{category}.csv"
                data.to_csv(output_path)
                
                # Add to summary
                summary_data.append({
                    'symbol': symbol,
                    'category': category,
                    **metrics
                })
                
                logger.info(f"Sorted {symbol} into {category}")
                
            except Exception as e:
                logger.error(f"Error sorting {symbol}: {str(e)}")
        
        # Generate summary report
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.output_dir / 'sorting_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            # Generate category-specific summaries with enhanced statistics
            for category in self.CATEGORIES:
                category_summary = summary_df[summary_df['category'] == category]
                
                # Create empty statistics file even if no stocks in category
                stats_path = self.output_dir / category / 'summary' / f"{category}_statistics.csv"
                if category_summary.empty:
                    pd.DataFrame(columns=['r2_score', 'normalized_mad', 'returns_std', 'cv']).to_csv(stats_path)
                else:
                    # Calculate additional statistics
                    stats = category_summary.describe()
                    
                    # Save detailed summary
                    category_summary_path = self.output_dir / category / 'summary' / f"{category}_summary.csv"
                    category_summary.to_csv(category_summary_path, index=False)
                    
                    # Save statistics
                    stats.to_csv(stats_path)
        
        return categorized_stocks 