import logging
import functools
import time
try:
    from memory_profiler import memory_usage
    import psutil
    dev_available = True
except ImportError:
    dev_available = False

# Init the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s : %(message)s')

#@log_function_call(logging.getLogger(__name__))

def log_function_call(logger):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper_log(*args, **kwargs):
            logger.info(f"Entering: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Exiting: {func.__name__}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                raise
        return wrapper_log
    return decorator_log
#@profile_function_call(logging.getLogger(__name__))

def profile_function_call(logger):
    def decorator_profile(func):
        @functools.wraps(func)
        def wrapper_profile(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time} seconds")
            return result
        return wrapper_profile
    return decorator_profile

#@profile_memory(logging.getLogger(__name__))

def profile_memory(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if dev_available:
                mem_usage_before = memory_usage(-1, interval=0.2, timeout=1)
                result = func(*args, **kwargs)
                mem_usage_after = memory_usage(-1, interval=0.2, timeout=1)
                logger.info(f"{func.__name__} - Memory Usage Before: {max(mem_usage_before)} MB - Memory Usage After: {max(mem_usage_after)} MB")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

#@profile_cpu(logging.getLogger(__name__))

def profile_cpu(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if dev_available:
                cpu_percent_before = psutil.cpu_percent()
                result = func(*args, **kwargs)
                cpu_percent_after = psutil.cpu_percent()
                logger.info(f"{func.__name__} - CPU Usage: {cpu_percent_after - cpu_percent_before}%")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator