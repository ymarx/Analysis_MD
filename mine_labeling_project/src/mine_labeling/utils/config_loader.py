"""
Configuration Loader Utility
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file
                     If None, uses default_config.yaml

    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        # Use default config
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / 'config' / 'default_config.yaml'

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(default_config: Dict[str, Any],
                  custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge custom configuration with default configuration

    Args:
        default_config: Default configuration dictionary
        custom_config: Custom configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()

    for key, value in custom_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_custom_config(custom_config_path: str) -> Dict[str, Any]:
    """
    Load custom configuration and merge with defaults

    Args:
        custom_config_path: Path to custom YAML config file

    Returns:
        Merged configuration dictionary
    """
    # Load default config
    default_config = load_config()

    # Load custom config
    custom_config_path = Path(custom_config_path)

    if not custom_config_path.exists():
        raise FileNotFoundError(f"Custom configuration file not found: {custom_config_path}")

    with open(custom_config_path, 'r', encoding='utf-8') as f:
        custom_config = yaml.safe_load(f)

    # Merge configurations
    merged_config = merge_configs(default_config, custom_config)

    return merged_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required keys
    required_keys = ['paths', 'extraction', 'coordinate_mapping', 'labeling']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate coordinate mapping
    coord_mapping = config['coordinate_mapping']

    if coord_mapping['scale_x'] <= 0 or coord_mapping['scale_y'] <= 0:
        raise ValueError("Scaling factors must be positive")

    # Validate sampling ratios
    if 'sampling' in config:
        sampling = config['sampling']

        if sampling['mine_samples'] <= 0:
            raise ValueError("mine_samples must be positive")

        if sampling['background_samples'] <= 0:
            raise ValueError("background_samples must be positive")

    return True
