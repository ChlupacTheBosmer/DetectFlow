import os.path

import logging
import functools
import time
import threading
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Queue, Process, Event
from queue import Empty
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


class ResourceMonitor:
    def __init__(self, interval=1, plot_interval=600, show=False, output_dir=None, email_handler=None, email_address=None, email_interval=10):
        self.interval = interval
        self.plot_interval = plot_interval
        self.show = show
        self.memory_usage = []
        self.cpu_usage = []
        self.timestamps = []
        self.events = []
        self.event_queue = Queue()
        self.stop_event = Event()
        self.total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)  # Total memory in GB
        self.cpu_count = psutil.cpu_count()  # Total number of CPU cores
        self.output_dir = output_dir or "."
        self.email_handler = email_handler
        self.email_address = email_address
        self.email_interval = email_interval

    def log_usage(self):
        error_count = 0
        while not self.stop_event.is_set():
            try:
                memory_info = psutil.virtual_memory()
                cpu_info = psutil.cpu_percent(interval=None)
                timestamp = datetime.now()
            except Exception as e:
                logging.error(f"Resource Monitor: An error occurred when logging resource usage: {e}")

                memory_info = None
                cpu_info = None
                timestamp = None

                error_count += 1
                if error_count > 5:
                    logging.error("Resource Monitor: Too many errors occurred, stopping monitoring.")
                    self.stop_event.set()

            if all([x is not None for x in [memory_info, cpu_info, timestamp]]):
                self.memory_usage.append(memory_info.percent)
                self.cpu_usage.append(cpu_info)
                self.timestamps.append(timestamp)

            # Check for events
            try:
                while True:
                    event, color = self.event_queue.get_nowait()
                    self.events.append((timestamp, event, color))
            except Empty:
                pass

            time.sleep(self.interval)

    def generate_plots(self):
        error_count = 0
        email_count = 0
        while not self.stop_event.is_set():
            time.sleep(self.plot_interval)
            try:
                plot_path = self.plot_usage()
                email_count += 1

                if self.email_handler and plot_path and email_count % self.email_interval == 0:
                    try:
                        body, image_attachment = self.email_handler.format_email_with_image("", plot_path)
                        subject = os.getenv('PBS_JOBID') or "Resource Monitoring Plot"
                        self.email_handler.send_email(self.email_address, subject, body, attachments={"plot.jpg": image_attachment})
                    except Exception as e:
                        logging.error(f"Resource Monitor: An error occurred when sending email: {e}")

            except Exception as e:
                logging.error(f"Resource Monitor: An error occurred when generating plots: {e}")
                error_count += 1
                if error_count > 5:
                    logging.error("Resource Monitor: Too many errors occurred, stopping monitoring.")
                    self.stop_event.set()

    def plot_usage(self):
        if not self.timestamps:
            return

        plt.figure(figsize=(12, 8))  # Increase figure size

        # Plot Memory Usage
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.timestamps, self.memory_usage, label=f'Memory Usage (%) - Total: {self.total_memory_gb:.2f} GB')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.set_title('Memory Usage Over Time')
        ax1.legend()

        # Plot CPU Usage
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(self.timestamps, self.cpu_usage, label=f'CPU Usage (%) - Total: {self.cpu_count} Cores', color='orange')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_title('CPU Usage Over Time')
        ax2.legend()

        # Plot events as vertical lines and labels
        for timestamp, event, color in self.events:
            ax1.axvline(x=timestamp, color=color, linestyle='--', lw=0.5)
            ax1.text(timestamp, max(self.memory_usage) * 0.95, event, rotation=90, verticalalignment='center', color=color)
            ax2.axvline(x=timestamp, color=color, linestyle='--', lw=0.5)
            ax2.text(timestamp, max(self.cpu_usage) * 0.95, event, rotation=90, verticalalignment='center', color=color)

        # Format x-axis for better readability
        plt.gcf().autofmt_xdate()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit labels
        plot_filename = f'resource_usage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_filepath = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_filepath)
        if self.show:
            plt.show()
        plt.close()
        print(f"Plot saved as {plot_filename}")
        return plot_filepath

    def start(self):
        self.monitor_thread = threading.Thread(target=self.log_usage)
        self.plot_thread = threading.Thread(target=self.generate_plots)
        self.monitor_thread.start()
        self.plot_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()
        self.plot_thread.join()

    def log_event(self, event_message, color='red'):
        self.event_queue.put((event_message, color))

