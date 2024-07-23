import json
import logging
import configparser
from abc import ABC, abstractmethod
from detectflow.utils.config import load_json_config, load_ini_config


class ConfigHandler(ABC):
    def __init__(self, config_path, format='json', defaults=None):
        """
        Initialize the configuration handler with the given path and format.

        Args:
            config_path (str): The path to the configuration file.
            format (str): The format of the configuration file ('json' or 'ini').
            defaults (dict, optional): Default values for configuration keys.
        """
        self.config_path = config_path
        self.format = format
        self.defaults = defaults or {}
        self.config = self.load_config()

    def load_config(self):
        """
        Load the configuration from the specified file.

        Returns:
            dict: The configuration dictionary.
        """
        from detectflow.validators.validator import Validator

        if Validator.is_valid_file_path(self.config_path):
            if self.format == 'json':
                config_data = load_json_config(self.config_path)
            elif self.format == 'ini':
                config_data = load_ini_config(self.config_path)
            else:
                raise ValueError("Unsupported format. Use 'json' or 'ini'.")
        else:
            logging.warning(f"Configuration file not found at path: {self.config_path}")
            config_data = {}

        # Merge with defaults before validation
        config_with_defaults = {**self.defaults, **config_data}
        self.config = config_with_defaults

        # Validate configuration
        self._validate_config()

        return self.config

    def pack_config(self, **kwargs):
        """
        Gather configuration data from various attributes or variables passed as kwargs,
        pack them into the self.config dictionary.
        """
        self.config = {key: value for key, value in kwargs.items()}
        self._validate_config()  # Validate the newly packed config

    def save_config(self):
        """
        Save the current configuration state back to the file.
        """
        if not self.config_path:
            raise ValueError("Config file path is not defined.")
        if self.format == 'json':
            with open(self.config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
        elif self.format == 'ini':
            config = configparser.ConfigParser()
            for section, settings in self.config.items():
                config.add_section(section)
                for key, value in settings.items():
                    config.set(section, key, str(value))
            with open(self.config_path, 'w') as file:
                config.write(file)

    @abstractmethod
    def _validate_config(self):
        """
        Validate the loaded configuration. Must be implemented in subclasses.
        """
        pass
