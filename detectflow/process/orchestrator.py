import logging
from threading import Thread
import json
import sys
import uuid
import os
from queue import Queue
from typing import Dict, Optional
from detectflow.handlers.config_handler import ConfigHandler
from detectflow.manipulators.dataloader import Dataloader
from detectflow.utils.threads import profile_threads, manage_threads
from detectflow.utils.s3.input import validate_and_process_input


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

    def get_status(self, file_path):
        """
        Get the processing status for a specific video file.

        :param file_path: Path of the video file.
        :return: Processing status for the given file.
        """
        return self._status.get(file_path, 0)

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
        "user_name": str,
        "dataloader": type(None),
        "process_task_callback": type(None),
    }

    CONFIG_DEFAULTS = {
        "input_data": None,
        "checkpoint_dir": None,
        "task_name": None,
        "batch_size": 3,
        "max_workers": 3,
        "force_restart": False,
        "scratch_path": "",
        "user_name": "USER",
        "dataloader": None,
        "process_task_callback": None
    }

    def __init__(self,
                 config_path: Optional[str] = None,
                 config_format: str = "json",
                 config_defaults: Optional[Dict] = None,
                 **kwargs):

        if not config_defaults:
            config_defaults = self.CONFIG_DEFAULTS

        self.callback_config = {}

        super().__init__(config_path, config_format, config_defaults)

        try:
            # Start by initializing dataloader, it should be injected
            dataloader = self.config.get('dataloader', None) if not kwargs.get('dataloader', None) else kwargs.get('dataloader', None)
            self.dataloader = dataloader or Dataloader()

            # Assign attributes
            self.input_data = self.config.get('input_data', None) if not kwargs.get('input_data', None) else kwargs.get('input_data', None)
            self.checkpoint_dir = self.config.get('checkpoint_dir', os.getcwd()) if not kwargs.get('checkpoint_dir', None) else kwargs.get('checkpoint_dir', None)
            self.task_name = self.config.get('task_name', str(uuid.uuid4())) if not kwargs.get('task_name', None) else kwargs.get('task_name', None)
            self.batch_size = self.config.get('batch_size', 3) if not kwargs.get('batch_size', None) else kwargs.get('batch_size', None)
            self.max_workers = self.config.get('max_workers', 3) if not kwargs.get('max_workers', None) else kwargs.get('max_workers', None)
            self.force_restart = self.config.get('force_restart', False) if not kwargs.get('force_restart', None) else kwargs.get('force_restart', None)
            scratch_path = self.config.get('scratch_path', "") if not kwargs.get('scratch_path', None) else kwargs.get('scratch_path', None)
            self.scratch_path = scratch_path if self.dataloader.is_valid_directory_path(scratch_path) else ""
            self.user_name = self.config.get('user_name', None) if not kwargs.get('user_name', None) else kwargs.get('user_name', None)
            self.fallback_directories = self._generate_fallback_directories()
            self.process_task_callback = self.config.get('process_task_callback', None) if not kwargs.get('process_task_callback', None) else kwargs.get('process_task_callback', None)

            # Init other attributes
            self.task_queue = Queue()
            self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.task_name}.json")
            self._setup_logging()
            self.checkpoint_data = None
            self._initialize_checkpoint()
            self.threads = []

            # Update the config by the values of the atr in case custom values were passed. Saving config would save the custom configuration.
            self.pack_config(
                input_data=self.input_data,
                checkpoint_dir=self.checkpoint_dir,
                task_name=self.task_name,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
                force_restart=self.force_restart,
                scratch_path=self.scratch_path,
                user_name=self.user_name,
                dataloader=None,
                process_task_callback=None
            )
        except Exception as e:
            logging.error(f"Failed to initialize Orchestrator: {e}")
            raise

    def _generate_fallback_directories(self):
        # Generate dynamic fallback paths based on the instance attributes
        return [
            f'/storage/brno2/home/{self.user_name}/',
            f'/storage/plzen1/home/{self.user_name}/',
            f'/storage/brno1-cerit/home/{self.user_name}/',
            self.scratch_path
        ]

    def validate_config(self):
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

    def _setup_logging(self):
        # Initialize the logger attribute
        self.logger = logging.getLogger(__name__)

        # Setup logging with a desired format and level
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Ensure that no duplicate handlers are added
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)

    def _initialize_checkpoint(self):
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as file:
                    self.checkpoint_data = json.load(file)

                # Convert 'input_type_flags' from list to tuple
                if 'input_type_flags' in self.checkpoint_data and isinstance(self.checkpoint_data['input_type_flags'],
                                                                             list):
                    self.checkpoint_data['input_type_flags'] = tuple(self.checkpoint_data['input_type_flags'])

                if not self._validate_checkpoint_format(self.checkpoint_data):
                    raise ValueError("Invalid format in checkpoint file")
            else:
                raise FileNotFoundError(
                    "Checkpoint file not found")  # TODO: Should probably check for checkpoint in a fallback location if it faield to be created in the original one

        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            logging.error(f"Error reading checkpoint file: {e}")
            if self.force_restart:
                logging.info("Force restart enabled. Creating a new checkpoint.")
                self._create_new_checkpoint()
            else:
                raise RuntimeError(f"Unable to proceed due to checkpoint file issue: {e}")

    def _validate_checkpoint_format(self, data):
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
                'tasks': [{'directory': directory, 'status': self._prepare_initial_status(directory, input_flags)} for directory in
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

    def _prepare_initial_status(self, directory, input_flags):
        try:
            if input_flags is None:
                raise ValueError("Error during input data processing - 'None' type")

            if input_flags[0]:  # S3 bucket, directory
                bucket, prefix = self.dataloader._parse_s3_path(directory)
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

            print(file_list)
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
            self._attempt_fallback_checkpoint_write()

    def _attempt_fallback_checkpoint_write(self):
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

    def start_processing(self):
        # Start the workers
        self._start_workers()

        # Begin managing tasks
        self._manage_tasks()

        # Signal workers to stop after all tasks are queued
        for _ in range(self.max_workers):
            self.task_queue.put(None)

        # Wait for all tasks to be completed
        # self.task_queue.join() #TODO: Is queue a thread? Shouldn't the workers be joined rather than the queue?
        for thread in self.threads:
            thread.join()

        # Profile running threads
        profile_threads()

    def _manage_tasks(self):
        directory = None
        for task in self.checkpoint_data.get('tasks', []):
            try:
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

    def handle_worker_update(self, update_info):
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
                worker_thread = Thread(target=self._worker_process, args=(worker_name,), name=worker_name)
                self.threads.append(worker_thread)
                worker_thread.start()
            except Exception as e:
                logging.error(f"Failed to start worker thread: {e}")

        # Profile running threads
        manage_threads(r'Worker #\d+', 'status')

    def _worker_process(self, name):
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    self.task_queue.put(None)
                    break

                self._process_task(task, name)
                self.task_queue.task_done()
            except Exception as e:
                logging.error(f"{name} - Error processing task: {e}")
                self.task_queue.task_done()  # Ensure task_done is called even if there's an error

        # Join the thread
        logging.info(f"{name} - Joining Worker thread")
        manage_threads(name, 'join')

    def _process_task(self, task, name):

        # Call the processing callback if it's set
        if self.process_task_callback:
            try:
                self.process_task_callback(
                    task=task,
                    orchestrator=self,
                    name=name,
                    scratch_path=self.scratch_path,
                    max_workers=self.max_workers,
                    **self.callback_config  # DONE: Rename config after implementing the ocnfig funcionality
                )
            except Exception as callback_exc:
                logging.error(f"{name} - Error during processing callback in orchestrator process task: {callback_exc}")
                # Consider whether to continue or break the loop based on the nature of the error

        return
        # Placeholder for task processing logic
        # This method should handle the actual processing of each task, including:
        # - Loading data (if necessary)
        # - Processing the data (e.g., running predictions)
        # - Reporting progress back to the Orchestrator

        # Example:
        # for file in task['files']:
        #     last_processed_frame = ...  # Logic to process the file
        #     self.handle_worker_update({'directory': task['directory'], 'file': file, 'last_processed_frame': last_processed_frame})

