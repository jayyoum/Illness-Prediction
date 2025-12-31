"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Default to configs/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths to absolute paths
    project_root = Path(__file__).parent.parent.parent
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config['paths'][key] = str(project_root / value)
    
    return config


def get_path(config: Dict[str, Any], *keys: str) -> str:
    """
    Get nested path from config dictionary.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
    
    Returns:
        Path string
    """
    value = config
    for key in keys:
        value = value[key]
    return value

