import logging
import json
import sys
import uuid
import os
import time
from typing import Dict, Optional
from detectflow.handlers.config_handler import ConfigHandler
from detectflow.manipulators.dataloader import Dataloader
from detectflow.utils.threads import profile_threads, manage_threads
from detectflow.utils.input import validate_and_process_input
from detectflow.utils import WINDOWS, CHECKPOINTS_DIR, DOWNLOADS_DIR
from detectflow.utils.dependencies import get_import_path, get_callable
from detectflow.manipulators.manipulator import Manipulator
import importlib

class Task:
    def __init__(self, directory: str, video_files: list, status: dict):
        """
        Initialize a Task instance for video processing.

        :param directory: The parent directory containing the video files.
        :param video_files: A list of paths to the video files to be processed.
        :param status: A dictionary mapping video file names to their processing status.
                       0 = not processed, positive int = last processed frame, -1 = processed.
        """
        self.directory = directory
        self._video_files = video_files
        self._status = status

    def get_status(self, file_path=None, file_name=None):
        """
        Get the processing status for a specific video file.

        :param file_path: Path of the video file.
        :param file_name: Name of the video file.
        :return: Processing status for the given file.
        """
        if file_path:
            return self._status.get(file_path, 0)
        if file_name:
            for key in self._status.keys():
                if os.path.basename(key) == file_name:
                    return self._status[key]
        return 0

    @property
    def files(self):
        """
        Get a list of file paths for the task.

        :return: List of file paths.
        """
        return self._video_files

    @property
    def statuses(self):
        """
        Get a list of processing statuses for the task.

        :return: List of statuses.
        """
        return [self._status.get(file) for file in self._video_files]

    @property
    def data(self):
        """
        Convert the task data to a dictionary format.

        :return: Dictionary representing the task.
        """
        return {'directory': self.directory, 'video_files': self._video_files, 'status': self._status}

    def __repr__(self):
        return f"Task(directory={self.directory}, video_files={self._video_files}, status={self._status})"


class Orchestrator(ConfigHandler):
    """
        Args:
        - config_path (str): Path to the config file. If it does not exist defaults will be used for config.
                            Manually adjusted config can be saved with save_config method.
        - format (str): Format of the config file ('json' or 'ini').
        - defaults Optional(dict): Dictionary of the default values used in case of no config file.
                                - When no defaults are provided then defaults from the DEFAULT_CONFIG map are used
                                  to fill in the missing keys in config.
                                - If defaults are provided than those custom defaults are used to fill in the
                                  missing keys in config.
        - kwargs: They override the settings loaded from the config file.

    """

    CONFIG_MAP = {
        "input_data": (str, list, tuple, type(None)),
        "checkpoint_dir": (str, type(None)),
        "task_name": (str, type(None)),
        "batch_size": int,
        "max_workers": int,
        "force_restart": bool,
        "scratch_path": str,
        "dataloader": type(None),
        "process_task_callback": (str, type(None)),
    }

    CONFIG_DEFAULTS = {
        "input_data": None,
        "checkpoint_dir": None,
        "task_name": None,
        "batch_size": 3,
        "max_workers": 3,
        "force_restart": False,
        "scratch_path": "",
        "dataloader": None,
        "process_task_callback": None
    }

    def __init__(self,
                 config_path: Optional[str] = None,
                 config_format: str = "json",
                 config_defaults: Optional[Dict] = None,
                 parallelism: Optional[str] = "process",
                 **kwargs):

        if not config_defaults:
            config_defaults = self.CONFIG_DEFAULTS

        self.callback_config = {}

        super().__init__(config_path, config_format, config_defaults)

        try:
            merged_config = {**self.config, **kwargs}

            # Start by initializing dataloader, it should be injected
            self.dataloader = merged_config.get('dataloader', Dataloader())
            self.input_data = merged_config.get('input_data')
            self.checkpoint_dir = merged_config.get('checkpoint_dir', os.getcwd())
            self.task_name = merged_config.get('task_name', str(uuid.uuid4()))
            self.batch_size = merged_config.get('batch_size', 3)
            self.max_workers = merged_config.get('max_workers', 3)
            self.force_restart = merged_config.get('force_restart', False)
            scratch_path = merged_config.get('scratch_path', None)
            self.scratch_path = scratch_path if self.dataloader.is_valid_directory_path(scratch_path) else os.getcwd()
            self.process_task_callback = merged_config.get('process_task_callback')

            self.fallback_directories = [CHECKPOINTS_DIR, DOWNLOADS_DIR, self.scratch_path]

            # Pack the callback config rewriting values loaded from config file
            if merged_config:
                for key, value in merged_config.items():
                    if key not in self.CONFIG_MAP:
                        self.callback_config[key] = value

            # Init other attributes
            self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.task_name}.json")
            self.checkpoint_data = None
            self.parallelism = None
            self._concurrent_unit = None
            self.task_queue = None
            self.control_queue = None
            self._unit_type = None
            self.concurrent_units = []

            # Determine parallelism type and set up
            self.parallelism = "process" if parallelism in ["process", "p"] else "thread"
            self._setup_parallelism()

            # Init checkpoint
            self._initialize_checkpoint()

        except Exception as e:
            logging.error(f"Failed to initialize Orchestrator: {e}")
            raise

        try:
            callback_import_path = get_import_path(self.process_task_callback)
        except Exception as e:
            logging.error(f"Error getting import path for process_task_callback: {e}")
            callback_import_path = None

        try:
            # Update the config by the values of the atr in case custom values were passed. Saving config would save the custom configuration.
            self.pack_config(
                input_data=self.input_data,
                checkpoint_dir=self.checkpoint_dir,
                task_name=self.task_name,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
                force_restart=self.force_restart,
                scratch_path=self.scratch_path,
                dataloader=None,
                process_task_callback=callback_import_path
            )
        except Exception as e:
            logging.error(f"Failed to pack config: {e}")

    def _setup_parallelism(self):
        try:
            if self.parallelism == "process":
                from multiprocessing import Process, JoinableQueue, Queue as MPQueue, set_start_method
                from queue import Empty as Queue_Empty
                self._concurrent_unit = Process
                self.task_queue = JoinableQueue()
                self.control_queue = MPQueue()
                self._unit_type = "process"
                self._queue_empty_exception = Queue_Empty
                set_start_method('spawn', force=True)
            else:
                from threading import Thread
                from queue import Queue, Empty as Queue_Empty
                self._concurrent_unit = Thread
                self.task_queue = Queue()
                self.control_queue = Queue()
                self._unit_type = "thread"
                self._queue_empty_exception = Queue_Empty
        except Exception as e:
            raise RuntimeError(f"Failed to set up parallelism: {e}")

    def load_config(self):

        self.config = super().load_config()

        # Assign attributes
        self.input_data = self.config.get('input_data', None)
        self.checkpoint_dir = self.config.get('checkpoint_dir', None)
        self.task_name = self.config.get('task_name', None)
        self.batch_size = self.config.get('batch_size', None)
        self.max_workers = self.config.get('max_workers', None)
        self.force_restart = self.config.get('force_restart', False)
        self.scratch_path = self.config.get('scratch_path', None)

        # Retrieve the callback function from the config
        callback_function_path = self.config.get('process_task_callback', None)
        self.process_task_callback = None
        if callback_function_path and isinstance(callback_function_path, str):
            try:
                self.process_task_callback = get_callable(callback_function_path)
                self.config['process_task_callback'] = self.process_task_callback
            except Exception as e:
                logging.error(f"Error loading process_task_callback from config: {e}")

        return self.config

    def _validate_config(self):
        """
        Validate config. Checks if all required keys are present.
        """
        required_keys = self.CONFIG_MAP  # Keys and their expected data types

        for key, type_ in required_keys.items():
            if key not in self.config:
                raise KeyError(f"Missing required configuration key: {key}")
            if type_ is None:
                continue
            if isinstance(type_, str):
                # Resolve the type from string when using strong literals for specifying type
                type_ = eval(type_, sys.modules[__name__].__dict__)
            if isinstance(type_, (tuple, list)):
                if not any([isinstance(self.config[key], t) for t in type_]):
                    raise TypeError(f"Expected types {type_} for key '{key}', got {type(self.config[key])}")
            if not isinstance(self.config[key], type_):
                raise TypeError(f"Expected type {type_} for key '{key}', got {type(self.config[key])}")

        # Sort out config keys that are supposed to be passed into the callback
        for key, value in self.config.items():
            if key not in required_keys:
                self.callback_config[key] = value

    def _initialize_checkpoint(self):
        def try_checkpoint_file(checkpoint_file):
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError("Checkpoint file not found in the specified location")

            with open(checkpoint_file, 'r') as file:
                checkpoint_data = json.load(file)

            # Convert 'input_type_flags' from list to tuple
            if 'input_type_flags' in checkpoint_data and isinstance(checkpoint_data['input_type_flags'], list):
                checkpoint_data['input_type_flags'] = tuple(checkpoint_data['input_type_flags'])

            if not self._validate_checkpoint(checkpoint_data):
                raise ValueError("Invalid format in checkpoint file")

            return checkpoint_data

        try:
            # Try loading the specified file
            self.checkpoint_data = try_checkpoint_file(self.checkpoint_file)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            logging.error(f"Error reading checkpoint file: {e}")

            if self.force_restart:
                logging.info("Force restart enabled. Creating a new checkpoint.")
                self._create_new_checkpoint()
                return
            else:
                # Seek and gather valdi alternatives
                for directory in self.fallback_directories:
                    files = Manipulator.find_files(directory, f"{self.task_name}.json")
                    if files is not None:
                        for file in files:
                            try:
                                self.checkpoint_data = try_checkpoint_file(file)
                                logging.info(f"Found valid checkpoint file in fallback location: {file}")
                                self.checkpoint_file = file
                                self.checkpoint_dir = os.path.dirname(file)
                                return
                            except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                                continue
                raise FileNotFoundError(f"Unable to continue. No valid checkpoint file found in fallback locations.\nEnable force restart if you wish to start from begining.")

    def _validate_checkpoint(self, data):
        required_keys = ['task_name', 'input_type_flags', 'batch_size', 'max_workers', 'tasks', 'progress']

        # Check if all required keys are present
        if not all(key in data for key in required_keys):
            raise ValueError("Checkpoint file is missing required keys.")

        # Validate the format of each key
        if not isinstance(data['task_name'], str):
            raise ValueError("Invalid format for 'task_name'.")
        if not isinstance(data['input_type_flags'], tuple) or len(data['input_type_flags']) != 4:
            raise ValueError("Invalid format for 'input_type_flags'.")
        if not isinstance(data['batch_size'], int) or data['batch_size'] <= 0:
            raise ValueError("Invalid format for 'batch_size'.")
        if not isinstance(data['max_workers'], int) or data['max_workers'] <= 0:
            raise ValueError("Invalid format for 'max_workers'.")
        if not isinstance(data['tasks'], list):
            raise ValueError("Invalid format for 'tasks'.")
        if not isinstance(data['progress'], dict):
            raise ValueError("Invalid format for 'progress'.")

        # Additional validations for the contents of 'tasks' and 'progress' can be added here

        return True

    def _create_new_checkpoint(self):
        try:
            # Attempt to validate and process input data
            directories, input_flags = validate_and_process_input(self.input_data, self.dataloader)

            # Prepare initial data for the checkpoint
            self.checkpoint_data = {
                'task_name': self.task_name,
                'input_type_flags': input_flags,
                'batch_size': self.batch_size,
                'max_workers': self.max_workers,
                'tasks': [{'directory': directory, 'status': self._prepare_initial_checkpoint(directory, input_flags)} for directory in
                          directories],
                'progress': {}
            }

            # Attempt to write initial data to checkpoint file
            self._write_checkpoint()

        except Exception as e:
            # Log the error
            logging.error(f"Failed to create new checkpoint due to invalid input data: {e}")

            # Raise an exception to stop further processing and notify the user
            raise RuntimeError(f"Checkpoint creation failed due to invalid input: {e}")

    def _prepare_initial_checkpoint(self, directory, input_flags):
        try:
            if input_flags is None:
                raise ValueError("Error during input data processing - 'None' type")

            if input_flags[0]:  # S3 bucket, directory
                bucket, prefix = self.dataloader.parse_s3_path(directory)
                print(bucket, prefix)
                file_list = self.dataloader.list_files_s3(bucket, prefix, regex=r"^(?!.*^\.).*(?<=\.mp4|\.avi|\.mkv)$",
                                                          return_full_path=True)
            elif input_flags[2]:  # Local directory
                file_list = self.dataloader.list_files(directory, regex=r"^(?!.*^\.).*(?<=\.mp4|\.avi|\.mkv)$",
                                                   return_full_path=True)
            elif input_flags[1] or input_flags[3]:  # S3 file or local file
                file_list = [directory]  # If it's a file, we typically just return a list containing it.
            else:
                raise ValueError(
                    f"Invalid input data format, type processing flags: {input_flags} - (bucket/prefix, s3_file, dir, file)")

            #print(file_list)
            return {file: 0 for file in file_list}

        except Exception as e:
            logging.error(f"Error preparing initial status for directory {directory}: {e}")
            raise

    def _write_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'w') as file:
                json.dump(self.checkpoint_data, file, indent=4)
        except Exception as e:
            logging.error(f"Failed to write checkpoint file: {e}")
            self._write_fallback_checkpoint()

    def _write_fallback_checkpoint(self):
        # Attempt to save to fallback dirs
        for directory in self.fallback_directories:
            fallback_file = os.path.join(directory, f"{self.task_name}.json")
            try:
                with open(fallback_file, 'w') as file:
                    json.dump(self.checkpoint_data, file, indent=4)
                    logging.info(f"Checkpoint successfully written to fallback location: {fallback_file}")
                    return
            except Exception as e:
                logging.error(f"Failed to write checkpoint to fallback location {directory}: {e}")

        logging.critical("All attempts to write checkpoint failed. Progress may be lost.")

    def run(self):
        # Start the workers
        self._start_workers()

        # Begin managing tasks
        self._manage_tasks()

        # Control loop to check for signals and process tasks
        while (not self.task_queue.empty()) or not (any(u.is_alive() for u in self.concurrent_units)):
            try:
                control_signal, args = self.control_queue.get_nowait()
                if control_signal == "stop":
                    logging.info("Received stop signal")
                    break
                if control_signal == "update_task":
                    self._handle_worker_update(*args)
            except self._queue_empty_exception:
                pass

            time.sleep(0.1)  # Prevent busy-waiting

        # Signal workers to stop after all tasks are queued
        self.task_queue.join()
        for _ in range(self.max_workers):
            logging.info("Signaling workers to stop")
            self.task_queue.put(None)

        # Wait for all tasks to be completed
        for unit in self.concurrent_units:
            unit.join()

        # Profile running concurrent_units
        profile_threads() if self.parallelism == "thread" else None

    def _manage_tasks(self):
        directory = None
        for task in self.checkpoint_data.get('tasks', []):
            try:
                logging.info(f"Managing task for directory {task.get('directory')}")
                directory = task.get('directory')
                status = task.get('status', {})

                if not directory or not isinstance(status, dict):
                    raise ValueError("Invalid task data")

                if all(value == -1 for value in status.values()):
                    continue  # Skip if all files in the directory are processed

                batch = []
                batch_status = {}
                for file, progress in status.items():
                    if progress != -1:  # Not completed
                        batch.append(file)
                        batch_status[file] = progress
                        if len(batch) >= self.batch_size:
                            self._queue_batch(Task(directory, batch, batch_status))
                            batch = []
                            batch_status = {}

                if batch:
                    self._queue_batch(Task(directory, batch, batch_status))

            except Exception as e:
                logging.error(f"Error managing task for directory {directory}: {e}")
                # Continue with the next task, ensuring other tasks are not interrupted

    def _queue_batch(self, task: Task):
        try:
            self.task_queue.put(task)
            logging.info(f"Queued batch for directory {task.directory}")
            # Update checkpoint file with the queued batch
            for file in task.files:
                self.checkpoint_data['progress'][file] = 0  # Mark as queued but not started
            self._write_checkpoint()

        except Exception as e:
            logging.error(f"Error queuing batch for directory {task.directory}: {e}")

    def _update_task_progress(self, directory, file, last_processed_frame):
        try:
            # Update the status within the tasks
            task = next((t for t in self.checkpoint_data['tasks'] if t['directory'] == directory), None)
            if task is None or file not in task['status']:
                raise ValueError(f"File {file} in directory {directory} not found in tasks")

            # Update the progress of a specific file in both 'tasks' and 'progress'
            task['status'][file] = last_processed_frame
            self.checkpoint_data['progress'][file] = last_processed_frame

            # Check if all files in the directory are completed
            if all(status == -1 for status in task['status'].values()):
                for f in task['status']:
                    task['status'][f] = -1

            # Write the updated data to the checkpoint file
            self._write_checkpoint()

        except Exception as e:
            logging.error(f"Error updating task progress for {file} in {directory}: {e}")

    def _handle_worker_update(self, update_info):
        file, directory = None, None
        try:
            # Validate update_info format
            if not all(key in update_info for key in ['directory', 'file', 'status']):
                raise ValueError("update_info is missing required keys")

            # Extract values from update_info
            directory = update_info['directory']
            file = update_info['file']
            last_processed_frame = update_info['status']

            # Validate data types
            if not all(isinstance(value, str) for value in [directory, file]) or not isinstance(last_processed_frame,
                                                                                                int):
                raise ValueError("Invalid data types in update_info")

            # Update the task progress
            self._update_task_progress(directory, file, last_processed_frame)

        except Exception as e:
            # Log the error and continue
            logging.error(f"Error handling worker update for {file} in {directory}: {e}")
            # Continue with the next operation, ensuring other updates are not interrupted

    def _start_workers(self):
        for i in range(self.max_workers):
            try:
                worker_name = f"Worker #{i}"
                callback_kwargs = {
                    "orchestrator_control_queue": self.control_queue,
                    "scratch_path": self.scratch_path,
                    **self.callback_config
                }
                worker_unit = self._concurrent_unit(target=self._worker, args=(worker_name,
                                                                               self.task_queue,
                                                                               self.process_task_callback,
                                                                               callback_kwargs), name=worker_name)
                self.concurrent_units.append(worker_unit)
                worker_unit.start()
            except Exception as e:
                logging.error(f"Failed to start worker {self._unit_type}: {e}")

        # Profile running concurrent_units
        manage_threads(r'Worker #\d+', 'status') if self.parallelism == "thread" else None

    @staticmethod
    def _worker(name, task_queue, process_task_callback, callback_kwargs):
        print(process_task_callback)
        while True:
            try:
                task = task_queue.get()
                logging.info(f"{name} - Processing task: {task}")
                if task is None:
                    task_queue.put(None)
                    break

                # Call the processing callback if it's set
                if process_task_callback:
                    try:
                        process_task_callback(
                            task=task,
                            name=name,
                            **callback_kwargs
                        )
                    except Exception as callback_exc:
                        logging.error(
                            f"{name} - Error during processing callback in orchestrator process task: {callback_exc}")
                        # TODO: Consider whether to continue or break the loop based on the nature of the error

                task_queue.task_done()
            except Exception as e:
                logging.error(f"{name} - Error processing task: {e}")
                task_queue.task_done()  # Ensure task_done is called even if there's an error

    # @staticmethod
    # def _process_task(task, name, process_task_callback, callback_kwargs):
    #
    #     # Call the processing callback if it's set
    #     if process_task_callback:
    #         try:
    #             process_task_callback(
    #                 task=task,
    #                 name=name,
    #                 **callback_kwargs
    #             )
    #         except Exception as callback_exc:
    #             logging.error(f"{name} - Error during processing callback in orchestrator process task: {callback_exc}")
    #             # Consider whether to continue or break the loop based on the nature of the error
    #
    #     return
        # Placeholder for task processing logic
        # This method should handle the actual processing of each task, including:
        # - Loading data (if necessary)
        # - Processing the data (e.g., running predictions)
        # - Reporting progress back to the Orchestrator

        # Example:
        # for file in task['files']:
        #     last_processed_frame = ...  # Logic to process the file
        #     self.handle_worker_update({'directory': task['directory'], 'file': file, 'last_processed_frame': last_processed_frame})

