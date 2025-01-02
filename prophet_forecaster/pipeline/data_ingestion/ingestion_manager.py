"""
Data Ingestion Manager

This module orchestrates the data ingestion process, managing different data sources
and providing a command-line interface for data retrieval.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
from rich.console import Console
from rich.progress import Progress

from ...configs.config_manager import get_config
from .fetchers.yahoo_finance import YahooFinanceFetcher

# Initialize configuration
config = get_config()

# Configure logging
log_path = Path(config.get('paths.logs', 'logs')) / 'ingestion.log'
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.get('logging.level', 'INFO')),
    format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(str(log_path)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

class IngestionManager:
    """Manages the data ingestion process from various sources."""

    def __init__(self, env: Optional[str] = None):
        """
        Initialize the ingestion manager.

        Args:
            env (str, optional): Environment name for configuration
        """
        self.config = get_config(env)
        cache_dir = Path(self.config.get('paths.cache_dir', 'data/cache'))
        
        self.fetcher = YahooFinanceFetcher(
            rate_limit_pause=self.config.get('yahoo_finance.rate_limit_pause', 0.5),
            max_retries=self.config.get('yahoo_finance.max_retries', 3),
            cache_dir=str(cache_dir)
        )

    def fetch_single_symbol(self, symbol: str, interval: str = '1d',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol.

        Args:
            symbol (str): Stock symbol to fetch
            interval (str): Time interval for data points
            start_date (datetime, optional): Start date for data retrieval
            end_date (datetime, optional): End date for data retrieval

        Returns:
            pd.DataFrame: DataFrame containing the fetched data
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            data = self.fetcher.fetch_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            if data is None:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_multiple_symbols(self, symbols: List[str], interval: str = '1d',
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols (List[str]): List of stock symbols to fetch
            interval (str): Time interval for data points
            start_date (datetime, optional): Start date for data retrieval
            end_date (datetime, optional): End date for data retrieval

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        with Progress() as progress:
            task = progress.add_task("[cyan]Fetching data...", total=len(symbols))
            for symbol in symbols:
                data = self.fetch_single_symbol(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                if data is not None:
                    results[symbol] = data
                progress.update(task, advance=1)
        return results 