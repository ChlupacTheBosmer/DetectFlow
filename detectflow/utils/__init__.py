from typing import Union
from pathlib import Path
import os
import copy
import platform
import logging
from detectflow.utils.file import json_load, json_save
from detectflow.config import S3_CONFIG
from detectflow.validators.validator import Validator

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows']) # operating system
NUM_THREADS = min(8, max(1, os.cpu_count() - 1)) # Number of threads to use for parallel processing
LOGGING_NAME = 'detectflow' # Name of the logger

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def get_user_config_dir(sub_dir='detectflow'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    # Return the appropriate config directory for each operating system
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        logging.warning(f"User config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
                       'You can define a DF_CONFIG_DIR environment variable for this path.')
        path = Path('/tmp') / sub_dir if is_dir_writeable('/tmp') else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = Path(os.getenv('DF_CONFIG_DIR') or get_user_config_dir())  # Detectflow settings dir
SETTINGS_JSON = USER_CONFIG_DIR / 'settings.json'


class Settings(dict):

    def __init__(self, file=SETTINGS_JSON):
        root = Path()
        s3_config = S3_CONFIG

        self.file = Path(file)
        self.defaults = {
            'downloads_dir': str(root / 'downloads'),
            'checkpoints_dir': str(root / 'checkpoints'),
            'config_dir': str(root / 'configs'),
            's3_config': s3_config
        }

        super().__init__(copy.deepcopy(self.defaults))

        if not self.file.exists():
            self.save()

        self.load()
        correct_keys = self.keys() == self.defaults.keys()
        correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
        if not (correct_keys and correct_types):
            logging.warning(f'Detectflow settings reset to default values. Settings saved at {self.file}')
            self.reset()

    def load(self):
        """Loads settings from the JSON file."""
        super().update(json_load(self.file))

    def save(self):
        """Saves the current settings to the JSON file."""
        json_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        """Updates a setting value in the current settings."""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """Resets the settings to default and saves them."""
        self.clear()
        self.update(self.defaults)
        self.save()


class BaseClass:
    """Base class providing string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, BaseClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)


def install_s3_config(host_base, use_https, access_key, secret_key, host_bucket):
    """
    Install the S3 configuration file.

    Args:
        host_base (str): The base URL of the S3 host.
        use_https (bool): Whether to use HTTPS.
        access_key (str): The access key.
        secret_key (str): The secret key.
        host_bucket (str): The bucket host.
    """
    # Define the configuration content
    config_content = f"""
    [default]
    host_base = {host_base}
    use_https = {use_https}
    access_key = {access_key}
    secret_key = {secret_key}
    host_bucket = {host_bucket}
    """

    # Ensure the directory exists
    os.makedirs(os.path.dirname(S3_CONFIG), exist_ok=True)

    # Write the configuration to the file
    with open(S3_CONFIG, 'w') as config_file:
        config_file.write(config_content)

    print(f"Configuration saved to {S3_CONFIG}")
    print(f"Configuration verification status: {Validator.is_valid_file_path(S3_CONFIG)}")


# init of some global constants
SETTINGS = Settings()
DOWNLOADS_DIR = SETTINGS['downloads_dir']
CHECKPOINTS_DIR = SETTINGS['checkpoints_dir']
CONFIG_DIR = SETTINGS['config_dir']

# TODO: Create checkpoints directory and others
