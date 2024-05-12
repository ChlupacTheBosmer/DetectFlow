import multiprocessing
import math
import threading
import re
import logging

def calculate_optimal_threads(divisor: int = 1):
    # Get the number of available CPUs
    num_cpus = multiprocessing.cpu_count()

    # Calculate the number of threads as num_cpus divided by 20, rounded up
    num_threads = math.floor((num_cpus / 20) / divisor)

    print(f"The optimal number of threads based on the CPU count is: {num_threads}")

    return num_threads


def profile_threads():
    # Profile running threads
    for thread in threading.enumerate():
        print(f"Thread name: {thread.name}, Alive: {thread.is_alive()}")


def manage_threads(name_regex: str = r".*", action: str = "status"):
    '''
    Manages threads based on a regex pattern and action.
    :param name_regex: str - regex to match thread names
    :param action: str - "join", "daemon", or "status"
    :returns: List of thread names that were affected
    '''
    affected_threads = []
    try:
        regex = re.compile(name_regex)
        for thread in threading.enumerate():
            if regex.match(thread.name):
                if action == "join":
                    logging.info(f"Joining thread: {thread.name}")
                    thread.join()
                    affected_threads.append(thread.name)
                elif action == "daemon":
                    if not thread.is_alive():
                        logging.info(f"Setting thread as daemon: {thread.name}")
                        thread.daemon = True
                        affected_threads.append(thread.name)
                    else:
                        logging.warning(f"Thread {thread.name} is already running; cannot set as daemon.")
                elif action == "status":
                    logging.info(f"Checking status of thread: {thread.name}, Alive: {thread.is_alive()}")
                    if thread.is_alive():
                        affected_threads.append(thread.name)
    except re.error as e:
        logging.error(f"Invalid regex pattern: {e}")
        raise ValueError("Invalid regex pattern supplied, no action taken.")
    except Exception as e:
        logging.error(f"Error in managing threads: {e}")
        raise

    return affected_threads