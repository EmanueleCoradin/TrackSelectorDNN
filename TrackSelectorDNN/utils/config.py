"""
Module for configuration loading.
"""

import yaml

def load_config(path):
    """
    Load a YAML configuration file.
    Args:
        path (str): Path to the YAML config file.  
    Returns:
        dict: Configuration as a nested dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
