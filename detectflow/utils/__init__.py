# from .hash import get_numeric_hash
# from .inspector import Inspector
# from .log_file import LogFile
# from .pbs_job_report import PBSJobReport
# from .pdf_creator import PDFCreator
# from .profile import log_function_call, profile_function_call, profile_memory, profile_cpu
# from .sampler import Sampler
# from .threads import calculate_optimal_threads, profile_threads, manage_threads
#
# __all__ = ['get_numeric_hash', 'Inspector', 'LogFile', 'PBSJobReport', 'PDFCreator', 'log_function_call',
#            'profile_function_call', 'profile_memory', 'profile_cpu', 'Sampler', 'calculate_optimal_threads',
#            'profile_threads', 'manage_threads']
from typing import Union
from pathlib import Path
import os
import copy
import json
import re
import platform
import logging

from detectflow.config import S3_CONFIG

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows']) # operating system


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


def json_save(file='data.json', data=None, header=''):
    """
    Save JSON data to a file.

    Args:
        file (str, optional): File name. Default is 'data.json'.
        data (dict): Data to save in JSON format.
        header (str, optional): JSON header to add (will be ignored in JSON).

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in JSON format
    with open(file, 'w', errors='ignore', encoding='utf-8') as f:
        if header:
            f.write(header)  # JSON doesn't support headers, but we write it as a comment
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_load(file='data.json', append_filename=False):
    """
    Load JSON data from a file.

    Args:
        file (str, optional): File name. Default is 'data.json'.
        append_filename (bool): Add the JSON filename to the JSON dictionary. Default is False.

    Returns:
        (dict): JSON data and file name.
    """
    assert Path(file).suffix == '.json', f'Attempting to load non-JSON file {file} with json_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add JSON filename to dict and return
        data = json.loads(s) or {}  # always return a dict (json.loads() may return None for empty files)
        if append_filename:
            data['json_file'] = str(file)
        return data


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


# init of some global constants
SETTINGS = Settings()
DOWNLOADS_DIR = SETTINGS['downloads_dir']
CHECKPOINTS_DIR = SETTINGS['checkpoints_dir']
CONFIG_DIR = SETTINGS['runs_dir']

