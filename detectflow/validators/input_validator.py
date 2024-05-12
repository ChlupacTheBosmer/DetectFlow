import re
from typing import Dict, Union, List, Tuple
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.validators.s3_validator import S3Validator
import os


class InputValidator(S3Validator):
    def __init__(self, cfg_file: str = "/storage/brno2/home/USER/.s3.cfg"):

        # Run the init method of S3Validator parent class
        S3Validator.__init__(self, cfg_file)

        # Init S3Manipulator
        self.s3manipulator = S3Manipulator(cfg_file)

    def validate_and_process_input(self, input_data):
        if isinstance(input_data, str):
            return self._process_single_input(input_data)
        elif isinstance(input_data, (list, tuple)):
            return self._process_multiple_inputs(input_data)
        else:
            raise ValueError("Invalid input format")

    def _process_single_input(self, input_data):
        if self.is_s3_bucket(input_data):
            # Fetch directories from the S3 bucket
            return self.s3manipulator.list_directories_s3(input_data, full_path=True), (True, False, False, False)

        elif self.is_s3_directory(input_data):
            # Fetch directories from the S3 directory
            bucket_name, prefix = self._parse_s3_path(input_data)
            return self.s3manipulator.list_directories_s3(bucket_name, prefix, full_path=True), (
            True, False, False, False)

        elif self.is_s3_file(input_data):
            return [input_data], (False, True, False, False)

        elif os.path.isdir(input_data):
            # Fetch subdirectories from the local directory
            return self.s3manipulator.list_folders(input_data, return_full_path=True), (False, False, True, False)

        elif os.path.isfile(input_data):
            return [input_data], (False, False, False, True)

        else:
            raise FileNotFoundError(f"File or directory not found: {input_data}")

    def _process_multiple_inputs(self, input_data):
        directories = []
        flags = (False, False, False, False)  # default flags
        for item in input_data:
            processed_item, item_flags = self._process_single_input(item)
            directories.extend(processed_item)
            flags = tuple(any(pair) for pair in zip(flags, item_flags))
        return directories, flags

    @staticmethod
    def determine_source_type(path):
        if path is None:
            return "array"

        # Check if the source_path is a URL
        if InputValidator.is_valid_url(path):
            return "url"

        # File extension mapping
        video_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
        image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Get file extension
        _, file_extension = os.path.splitext(path.lower())

        # Determine the type based on the extension
        if file_extension in video_formats:
            return "video"
        elif file_extension in image_formats:
            return "image"
        else:
            return "unknown"

    @staticmethod
    def is_valid_url(path):

        import urllib.parse

        try:
            result = urllib.parse.urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def is_valid_email(email):
        """Validate an email address using a regular expression."""
        # Improved pattern to handle more complex email addresses
        pattern = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+([.]\w+)*\.[a-z]{2,4}$'
        if re.match(pattern, email, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def validate_string(s, pattern):
        """
        Check if the provided string `s` matches the regular expression `pattern`.

        Example patterns:

        recording_id = r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}$'
        video_id = (r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}_' # same as pattern 1
            r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])_'  # yyyymmdd
            r'([01]\d|2[0-3])_([0-5]\d)$')

        """
        match = re.match(pattern, s)
        return match is not None

    @staticmethod
    def validate_flags(input_flags: Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]],
                       flag_map: Union[Dict[int, str], List[str], Tuple[str, ...]],
                       fix: bool = False) -> List[str]:

        # Check if flag_map is a list or tuple, convert it to a dictionary if so
        if isinstance(flag_map, (list, tuple)):
            flag_map = {i: flag for i, flag in enumerate(flag_map)}
        allowed_flags = set(flag_map.values())

        if isinstance(input_flags, str):
            if input_flags not in allowed_flags:
                if fix:
                    return []
                else:
                    raise ValueError("Invalid flag string")
            input_flags = [input_flags]
        elif isinstance(input_flags, int):
            if input_flags not in flag_map:
                if fix:
                    return []
                else:
                    raise ValueError("Invalid flag integer")
            input_flags = [flag_map[input_flags]]
        elif isinstance(input_flags, (list, tuple)):
            new_flags = []
            for flag in input_flags:
                if isinstance(flag, int):
                    if flag not in flag_map:
                        if not fix:
                            raise ValueError("Invalid flag integer")
                        continue  # Skip invalid integers if fixing
                    new_flags.append(flag_map[flag])
                elif isinstance(flag, str):
                    if flag not in allowed_flags:
                        if not fix:
                            raise ValueError("Invalid flag string")
                        continue  # Skip invalid strings if fixing
                    new_flags.append(flag)
                else:
                    raise TypeError("Each flag should be either an integer or a string")
            input_flags = new_flags
        else:
            raise TypeError("Input flags should be a string, integer, list, or tuple")

        return input_flags
