import time
import hashlib


def get_timestamp_hash():
    # Get the current time in milliseconds since the epoch
    timestamp = int(time.time() * 1000)
    return timestamp


def get_filepath_hash(file_path):
    unique_hash = hashlib.md5((file_path + str(time.time())).encode()).hexdigest()
    return unique_hash
