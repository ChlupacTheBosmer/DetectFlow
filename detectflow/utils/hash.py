import time

def get_numeric_hash():
    # Get the current time in milliseconds since the epoch
    timestamp = int(time.time() * 1000)
    return timestamp