import os
from detectflow.manipulators.manipulator import Manipulator

def validate_and_process_input(input_data, s3_manipulator):

    if isinstance(input_data, str):
        return process_single_input(input_data, s3_manipulator)
    elif isinstance(input_data, (list, tuple)):
        return process_multiple_inputs(input_data, s3_manipulator)
    else:
        raise ValueError("Invalid input format")

def process_single_input(input_data, s3_manipulator):

    if s3_manipulator.is_s3_bucket(input_data):
        # Fetch directories from the S3 bucket
        return s3_manipulator.list_directories_s3(input_data, full_path=True), (True, False, False, False)

    elif s3_manipulator.is_s3_directory(input_data):
        # Fetch directories from the S3 directory
        bucket_name, prefix = s3_manipulator._parse_s3_path(input_data)
        return s3_manipulator.list_directories_s3(bucket_name, prefix, full_path=True), (
            True, False, False, False)

    elif s3_manipulator.is_s3_file(input_data):
        return [input_data], (False, True, False, False)

    elif os.path.isdir(input_data):
        # Fetch subdirectories from the local directory
        return Manipulator.list_folders(input_data, return_full_path=True), (False, False, True, False)

    elif os.path.isfile(input_data):
        return [input_data], (False, False, False, True)

    else:
        raise FileNotFoundError(f"File or directory not found: {input_data}")

def process_multiple_inputs(input_data, s3_manipulator):
    directories = []
    flags = (False, False, False, False)  # default flags
    for item in input_data:
        processed_item, item_flags = process_single_input(item, s3_manipulator)
        directories.extend(processed_item)
        flags = tuple(any(pair) for pair in zip(flags, item_flags))
    return directories, flags
