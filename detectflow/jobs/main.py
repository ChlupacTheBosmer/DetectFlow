import argparse
import logging
import os
import sys
import traceback
from typing import Optional, Dict, Any
import multiprocessing

# Now import detectflow package
from detectflow.utils.config import load_config, merge_configs
from detectflow.process.database_manager import start_db_manager, stop_db_manager
from detectflow.config import S3_CONFIG, DETECTFLOW_DIR
from detectflow.manipulators.dataloader import Dataloader
from detectflow.process.orchestrator import Orchestrator
from detectflow.utils import install_s3_config


def setup_logging(log_file: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.propagate = False
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def main(config_path: str, config_format: str, log_file: Optional[str], **kwargs):

    # Set the start method to spawn. Absolute must with torch and on linux systems.
    multiprocessing.set_start_method('spawn')

    # Load config and merge with kwargs
    file_config = load_config(config_path, config_format)
    merged_config = merge_configs(file_config, kwargs)

    if kwargs.get('verbose') == 'false':
        os.environ['YOLO_VERBOSE'] = 'false'

    if not kwargs.get('scratch_path', None):
        try:
            kwargs['scratch_path'] = os.getenv('SCRATCHDIR')
            if not kwargs['scratch_path']:
                raise Exception("SCRATCHDIR environment variable not set.")
        except Exception as e:
            kwargs['scratch_path'] = os.getcwd()

    # if not kwargs.get('log_file', None):
    #     try:
    #         log_file = os.path.join(kwargs.get('scratch_path'), 'orchestrator.log')
    #     except Exception as e:
    #         log_file = 'orchestrator.log'
    # setup_logging(log_file)

    # Install s3 config if the parameters were passed
    try:
        if all([key in merged_config for key in ['host_base', 'use_https', 'access_key', 'secret_key', 'host_bucket']]):
            install_s3_config(merged_config.get('host_base'),
                              merged_config.get('use_https'),
                              merged_config.get('access_key'),
                              merged_config.get('secret_key'),
                              merged_config.get('host_bucket'))
    except Exception as e:
        logging.error(f"An error occurred when installing S3 config: {e}")

    # Sort Kwargs
    orchestrator_kwargs_keys = {
        'batch_size',
        'max_workers',
        'force_restart',
        'input_data',
        'checkpoint_dir',
        'task_name',
        'scratch_path',
        'user_name',
        'dataloader',
        'process_task_callback',
        'parallelism'
    }

    database_manager_kwargs_keys = {
        'db_batch_size',
        'db_backup_interval'
    }

    orchestrator_kwargs = {key: value for key, value in kwargs.items() if key in orchestrator_kwargs_keys}
    database_manager_kwargs = {key.replace('db_', ''): value for key, value in merged_config.items() if key in database_manager_kwargs_keys}
    other_kwargs = {key: value for key, value in merged_config.items() if key not in orchestrator_kwargs_keys}

    # If debug is set, initialize Resource Monitor
    resource_monitor = None
    resource_monitor_queue = None
    if merged_config.get('debug'):
        logging.info("Initializing Resource Monitor...")
        try:

            if merged_config.get('email_debug', False) and merged_config.get('email_auth', None):
                from detectflow.handlers.email_handler import EmailHandler
                email_handler = EmailHandler("detectflow@gmail.com", merged_config.get('email_auth'))
            else:
                email_handler = None

            from detectflow.utils.profiling import ResourceMonitorPID
            main_pid = os.getpid()
            resource_monitor = ResourceMonitorPID(main_pid=main_pid,
                                                  interval=1,
                                                  plot_interval=merged_config.get('plot_interval', 300),
                                                  show=False,
                                                  output_dir=merged_config.get('scratch_path'),
                                                  email_handler=email_handler,
                                                  email_address=merged_config.get('email_address'),
                                                  email_interval=merged_config.get('email_interval', 2))
            resource_monitor.start()
            resource_monitor_queue = resource_monitor.event_queue
        except Exception as e:
            logging.error(f"An error occurred when initializing Resource Monitor: {e}")

    # Initialize Database Manager
    logging.info("Initializing Database Manager...")
    try:
        db_man_info = start_db_manager(**database_manager_kwargs)
    except Exception as e:
        logging.error(f"An error occurred when initializing Database Manager: {e}")
        sys.exit(1)

    # Initialize Dataloader
    logging.info("Initializing Dataloader...")
    try:
        dataloader = Dataloader(S3_CONFIG)
    except Exception as e:
        logging.error(f"An error occurred when initializing Dataloader: {e}")
        sys.exit(1)

    # Initialize Orchestrator
    logging.info("Packing Orchestrator Config...")
    try:
        callback_kwargs = {
            "db_manager_control_queue": db_man_info.get("control_queue", None),
            "db_manager_data_queue": db_man_info.get("data_queue", None),
            "resource_monitor_queue": resource_monitor_queue,
            **other_kwargs
        }
    except Exception as e:
        logging.error(f"An error occurred when packing Orchestrator Config: {e}")
        sys.exit(1)
    logging.info("Initializing Orchestrator...")
    try:
        orchestrator = Orchestrator(config_path=config_path, config_format=config_format, dataloader=dataloader, **orchestrator_kwargs, **callback_kwargs)
        orchestrator.run()
        logging.info("Orchestrator run completed successfully.")
        status = 0
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        status = 1

    # Stop the Database Manager
    try:
        stop_db_manager(db_man_info["control_queue"], db_man_info["process"])
        logging.info("Database Manager stopped.")
    except Exception as e:
        logging.error(f"An error occurred when stopping Database Manager: {e}")
        status = 1

    # Stop the Resource Monitor
    try:
        if resource_monitor:
            resource_monitor.stop()
            logging.info("Resource Monitor stopped.")
    except Exception as e:
        logging.error(f"An error occurred when stopping Resource Monitor: {e}")
        status = 1
    sys.exit(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Orchestrator with specified configuration.")
    parser.add_argument('--config_path', type=str, help="Path to the configuration file.")
    parser.add_argument('--config_format', type=str, choices=['json', 'ini'], default='json', help="Configuration file format.")
    parser.add_argument('--log_file', type=str, default="orchestrator.log", help="Path to the log file.")

    # Additional keyword arguments
    parser.add_argument('--batch_size', type=int, help="Batch size for processing.")
    parser.add_argument('--max_workers', type=int, help="Maximum number of workers.")
    parser.add_argument('--force_restart', help="Force restart the processing.")
    parser.add_argument('--input_data', type=str, help="Input data path or identifier.")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory for checkpoints.")
    parser.add_argument('--task_name', type=str, help="Name of the task.")
    parser.add_argument('--scratch_path', type=str, help="Scratch path for temporary files.")
    parser.add_argument('--user_name', type=str, help="User name.")
    parser.add_argument('--process_task_callback', type=str, help="Callback function for processing tasks.")

    # Database Manager arguments
    parser.add_argument('--db_batch_size', type=int, help="Size of batch of data to process.")
    parser.add_argument('--db_backup_interval', type=int, help="Interval for backing up the database.")

    # Debug and verbose
    parser.add_argument('--verbose', type=str, default='false', help="Prediction progress verbose settings. Logging unaffected.")
    parser.add_argument('--debug', type=bool, help="Sets logging level and resource monitoring.")
    parser.add_argument('--email_debug', type=bool, help="Sets receiving of monitoring emails.")
    parser.add_argument('--email_auth', type=str, help="Email service account authentication key.")
    parser.add_argument('--email_address', type=str, help="Email address fro receiving debug messages.")
    parser.add_argument('--plot_interval', type=int, help="Interval of plot generation in seconds.")
    parser.add_argument('--email_interval', type=int, help="How many plot intervals will pass before sending email.")

    args = parser.parse_args()

    kwargs = {key: value for key, value in vars(args).items() if value is not None}

    print("Arguments (args):", args)
    print("Keyword Arguments (kwargs):", kwargs)

    main(**kwargs)

