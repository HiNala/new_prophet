"""
Configuration Manager

This module handles loading and merging of configuration files based on the environment.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from functools import lru_cache

class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or parsing."""
    pass

class ConfigManager:
    """Manages configuration loading and environment-specific settings."""

    def __init__(self, env: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            env (str, optional): Environment name ('development', 'production', 'test').
                               If not provided, will use ENV environment variable.
        """
        self.env = env or os.getenv('ENV', 'development')
        self.config_dir = Path(__file__).parent
        self._config: Dict[str, Any] = {}
        self.load_config()

    @lru_cache(maxsize=None)
    def load_config(self) -> Dict[str, Any]:
        """
        Load and merge configuration files based on the environment.

        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        try:
            # Load base configuration
            base_config = self._load_yaml('base.yml')
            
            # Load environment-specific configuration
            env_config = self._load_yaml(f'{self.env}.yml')
            
            # Merge configurations
            self._config = self._deep_merge(base_config, env_config)
            
            # Create necessary directories
            self._create_directories()
            
            return self._config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {filename}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.

        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration

        Returns:
            Dict[str, Any]: Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def _create_directories(self):
        """Create necessary directories based on configuration."""
        if 'paths' in self._config:
            for path in self._config['paths'].values():
                Path(path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): Configuration key (dot notation supported)
            default (Any): Default value if key not found

        Returns:
            Any: Configuration value
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()

    @property
    def environment(self) -> str:
        """Get the current environment name."""
        return self.env

def get_config(env: Optional[str] = None) -> ConfigManager:
    """
    Factory function to get a ConfigManager instance.

    Args:
        env (str, optional): Environment name

    Returns:
        ConfigManager: Configuration manager instance
    """
    return ConfigManager(env) 