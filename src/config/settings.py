"""
Centralized Settings Configuration

This module provides centralized access to all configuration settings
for the Customer Analytics & Recommendation System.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Import base settings
from config.settings import *

class ConfigManager:
    """
    Configuration manager for loading and accessing YAML configurations
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._model_config = None
        self._data_config = None
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Load and return model configuration"""
        if self._model_config is None:
            self._model_config = self._load_yaml_config("model_config.yaml")
        return self._model_config
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Load and return data configuration"""
        if self._data_config is None:
            self._data_config = self._load_yaml_config("data_config.yaml")
        return self._data_config
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Configuration file {filename} not found")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filename}: {e}")
            return {}
    
    def get_model_param(self, section: str, param: str, default: Any = None) -> Any:
        """Get model parameter from configuration"""
        return self.model_config.get(section, {}).get(param, default)
    
    def get_data_param(self, section: str, param: str, default: Any = None) -> Any:
        """Get data parameter from configuration"""
        return self.data_config.get(section, {}).get(param, default)

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions for accessing configuration
def get_model_config() -> Dict[str, Any]:
    """Get complete model configuration"""
    return config_manager.model_config

def get_data_config() -> Dict[str, Any]:
    """Get complete data configuration"""
    return config_manager.data_config

def get_clustering_config() -> Dict[str, Any]:
    """Get clustering configuration"""
    return config_manager.model_config.get('clustering', {})

def get_recommendation_config() -> Dict[str, Any]:
    """Get recommendation configuration"""
    return config_manager.model_config.get('recommendations', {})

def get_rfm_config() -> Dict[str, Any]:
    """Get RFM configuration"""
    return config_manager.model_config.get('rfm', {}) 