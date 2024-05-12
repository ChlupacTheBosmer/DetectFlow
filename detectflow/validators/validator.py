import os
import re
import pandas as pd

class Validator:
    def __init__(self):
        pass

    @staticmethod
    def is_valid_file_path(file_path: str) -> bool:
        """Check if a file path is valid and exists."""
        if file_path is None:
            return False
        return os.path.exists(file_path) and os.path.isfile(file_path)

    @staticmethod
    def validate_file_path(func):
        """Decorator to validate the output of a function as a file path."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_file_path(result):
                raise ValueError(f"Invalid file path: {result}")
            return result

        return wrapper

    @staticmethod
    def is_valid_directory_path(directory_path: str) -> bool:
        """Check if a directory path is valid and exists."""
        return os.path.exists(directory_path) and os.path.isdir(directory_path)

    @staticmethod
    def validate_directory_path(func):
        """Decorator to validate the output of a function as a directory path."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_directory_path(result):
                raise ValueError(f"Invalid directory path: {result}")
            return result

        return wrapper

    @staticmethod
    def is_valid_paths(paths):
        """Check if the argument is a list or a tuple of path-like strings."""
        if not isinstance(paths, (list, tuple)):
            return False
        return all(isinstance(path, str) and os.path.exists(path) for path in paths)

    @staticmethod
    def validate_paths(func):
        """Decorator to validate the output of a function as a list or tuple of path-like strings."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_paths(result):
                raise ValueError(f"Invalid paths collection: {result}")
            return result

        return wrapper

    @staticmethod
    def is_valid_dataframe(df):  # TODO: Move to InputValidator or to DataValidator or dont I don't care
        """Check if the argument is a non-empty Pandas DataFrame."""
        return isinstance(df, pd.DataFrame) and not df.empty

    @staticmethod
    def validate_dataframe(func):  # TODO: Move to InputValidator
        """Decorator to validate the output of a function as a non-empty Pandas DataFrame."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_dataframe(result):
                raise ValueError("Invalid DataFrame: None or empty")
            return result

        return wrapper

    @staticmethod  # TODO: Move to InputValidator
    def fix_kwargs(config_map, kwargs, remove_unexpected: bool = True):
        """
        Validates a kwargs dictionary against a predefined config map and attempts to fix it.
        Args:
            config_map (dict): A dictionary with key:data_type pairs.
            kwargs (dict): A dictionary to be validated and fixed.
        """
        keys_to_remove = []

        for key, value in kwargs.items():
            if key not in config_map:
                keys_to_remove.append(key)
                continue

            expected_type = config_map[key]
            if not isinstance(value, expected_type):
                try:
                    if isinstance(expected_type, tuple):
                        # Attempt conversion to the first type in the tuple
                        kwargs[key] = expected_type[0](value)
                    else:
                        kwargs[key] = expected_type(value)
                except (ValueError, TypeError):
                    keys_to_remove.append(key)

        # Remove invalid keys
        if remove_unexpected:
            for key in keys_to_remove:
                del kwargs[key]

    @staticmethod  # TODO: Move to InputValidator
    def sort_and_validate_dict(input_dict, *maps):
        """
        Sort keys from the input dictionary into multiple dictionaries based on the maps provided.
        Validate and, if possible, convert the types of the values according to the maps.
        Handles multiple possible types for a single key.

        :param input_dict: Dictionary to be sorted and validated.
        :param maps: Variable number of dictionaries specifying allowed data types for keys.
        :return: Tuple of dictionaries, each corresponding to a provided map.
        """
        sorted_dicts = [{} for _ in maps]

        for key, value in list(input_dict.items()):
            in_any_map = False
            for i, map_dict in enumerate(maps):
                if key in map_dict:
                    in_any_map = True
                    allowed_types = map_dict[key] if isinstance(map_dict[key], tuple) else (map_dict[key],)
                    if any(isinstance(value, t) for t in allowed_types) or any(
                            Validator.try_convert_type(value, t) for t in allowed_types):
                        sorted_dicts[i][key] = value
                        break  # Break after the first successful map to avoid duplicating keys
            if in_any_map:
                del input_dict[key]

        return tuple(sorted_dicts)

    @staticmethod  # TODO: Move to InputValidator
    def try_convert_type(value, target_type):
        """
        Attempt to convert the value to the target_type.

        :param value: Value to be converted.
        :param target_type: Desired type for the value.
        :return: Converted value if successful, otherwise False.
        """
        try:
            # Special handling for boolean, as bool(int/str) always returns True
            if target_type == bool and not isinstance(value, bool):
                return False
            return target_type(value)
        except (ValueError, TypeError):
            return False

    @staticmethod  # TODO: Move to InputValidator
    def is_valid_regex(regex: str) -> bool:
        """Check if a regex pattern is valid."""
        try:
            re.compile(regex)
            return True
        except re.error:
            return False

    @staticmethod  # TODO: Move to InputValidator
    def validate_regex(func):
        """Decorator to validate the output of a function as a regex pattern."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_regex(result):
                raise ValueError(f"Invalid regex pattern: {result}")
            return result

        return wrapper
