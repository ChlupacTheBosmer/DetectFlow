import json
from configparser import ConfigParser
from typing import Dict, Any
import logging
import sys


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def load_ini_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from an INI file.

    Args:
        config_path (str): Path to the INI configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        Exception: If there is an error reading the INI file.
    """
    config = ConfigParser()
    config.read(config_path)
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    return config_dict


def load_config(config_path: str, config_format: str) -> Dict[str, Any]:
    try:
        if config_format == 'json':
            return load_json_config(config_path)
        elif config_format == 'ini':
            return load_ini_config(config_path)
        else:
            logging.error(f"Unsupported configuration format: {config_format}")
            sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def merge_configs(file_config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    merged_config = file_config.copy()
    merged_config.update(kwargs)  # kwargs takes precedence
    return merged_config