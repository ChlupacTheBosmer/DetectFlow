import traceback
import os
import threading
import csv
from typing import Dict, List, Optional, Type, Any
from detectflow.utils.profiling import profile_function_call
from detectflow.manipulators.dataloader import Dataloader
from detectflow.config import S3_CONFIG
import logging
from detectflow.utils import DOWNLOADS_DIR
import multiprocessing
import time
from datetime import datetime

VISITS_SQL = """
            CREATE TABLE IF NOT EXISTS visits (
                frame_number integer,
                video_time text NOT NULL,
                life_time text,
                year integer,
                month integer,
                day integer,
                recording_id text,
                video_id text,
                video_path text,
                flower_bboxes text,
                rois text,
                all_visitor_bboxes text,
                relevant_visitor_bboxes text,
                visit_ids text,
                on_flower boolean,
                flags text
            );
            """

VISITS_COLS = [
    ("frame_number", "integer", ""),
    ("video_time", "text", "NOT NULL"),
    ("life_time", "text", ""),
    ("year", "integer", ""),
    ("month", "integer", ""),
    ("day", "integer", ""),
    ("recording_id", "text", ""),
    ("video_id", "text", "NOT NULL"),
    ("video_path", "text", ""),
    ("flower_bboxes", "text", ""),
    ("rois", "text", ""),
    ("all_visitor_bboxes", "text", ""),
    ("relevant_visitor_bboxes", "text", ""),
    ("visit_ids", "text", ""),
    ("on_flower", "boolean", ""),
    ("flags", "text", "")
]

VISITS_CONSTR = 'PRIMARY KEY (frame_number, video_id)'

VIDEOS_SQL = """
            CREATE TABLE IF NOT EXISTS videos (
                recording_id text,
                video_id text PRIMARY KEY,
                s3_bucket text,
                s3_directory text,
                format text,
                start_time text,
                end_time text,
                length integer,
                total_frames integer,
                fps integer,
                focus real,
                blur real,
                contrast real,
                brightness real,
                daytime text,
                focus_regions_start text,
                flowers_start text,
                focus_acc_start text,
                focus_regions_end text,
                flowers_end text,
                focus_acc_end text,
                motion real
            );
        """

VIDEOS_COLS = [
    ("recording_id", "text", ""),
    ("video_id", "text", "PRIMARY KEY"),
    ("s3_bucket", "text", ""),
    ("s3_directory", "text", ""),
    ("format", "text", ""),
    ("start_time", "text", ""),
    ("end_time", "text", ""),
    ("length", "integer", ""),
    ("total_frames", "integer", ""),
    ("fps", "integer", ""),
    ("focus", "real", ""),
    ("blur", "real", ""),
    ("contrast", "real", ""),
    ("brightness", "real", ""),
    ("daytime", "text", ""),
    ("focus_regions_start", "text", ""),
    ("flowers_start", "text", ""),
    ("focus_acc_start", "text", ""),
    ("focus_regions_end", "text", ""),
    ("flowers_end", "text", ""),
    ("focus_acc_end", "text", ""),
    ("motion", "real", "")
]


class DatabaseManager:

    DEFAULT_STRUCTURE = {
        "visits": {'columns': VISITS_COLS,
                   'constraints': VISITS_CONSTR,
                   'batching': True},
        "videos": {'columns': VIDEOS_COLS,
                   'batching': False}
    }

    def __init__(self,
                 db_paths: Optional[Dict[str, str]] = None,
                 batch_size: int = 100,
                 backup_interval: int = 500,
                 control_queue: Optional[multiprocessing.Queue] = None,
                 data_queue: Optional[multiprocessing.Queue] = None,
                 dataloader: Optional[Dataloader] = None,
                 database_structure: Optional[Dict[str, Any]] = None):
        """
        Initialize the DatabaseManager instance.

        Args:
        db_manipulators (Optional[Dict[str, Type['DatabaseManipulator']]]): A dictionary of database manipulators.
        batch_size (int): The size of each batch for processing.
        backup_interval (int): The interval at which to back up the database to S3.
        dataloader (Optional[Type[Dataloader]]): An instance or subclass of Dataloader.
        """
        self.database_structure = database_structure if database_structure is not None else self.DEFAULT_STRUCTURE
        self.db_paths = db_paths
        self.db_manipulators = {}
        self.processed_databases = set()
        self.backup_interval = backup_interval
        self.backup_counters = {}
        self.batch_size = batch_size
        self.data_batches = {}
        self.control_queue = control_queue if control_queue is not None else multiprocessing.Queue()
        self.queue = data_queue if data_queue is not None else multiprocessing.Queue()
        self.dataloader = dataloader

    def _init_database_manipulator(self, db_file):
        """
        Initialize a new database manipulator for the given database file.
        """
        from detectflow.manipulators.database_manipulator import DatabaseManipulator

        return DatabaseManipulator(db_file, batch_size=self.batch_size, lock_type="processing")

    def _init_database(self, db_manipulator):
        """
        Initialize the database and required tables for the given recording ID.
        """
        for table_name, table_info in self.database_structure.items():
            columns = table_info['columns']
            constraints = table_info.get('constraints', '')
            if table_name not in db_manipulator.get_table_names():
                db_manipulator.create_table(table_name, columns, constraints)

    def add_database(self, recording_id: str, db_path: str):
        """
        Add a new database for the given recording ID and initialize it.
        """
        if recording_id in self.db_manipulators:
            print(f"Database for recording ID {recording_id} already exists.")
            return

        # Determine which database to use (new, local, remote)
        db_path = self._resolve_db_conflict(recording_id, db_path)

        db_manipulator = self._init_database_manipulator(db_path)
        self.db_manipulators[recording_id] = db_manipulator
        self.data_batches[recording_id] = []
        self.backup_counters[recording_id] = 0

        # Initialize the database and required tables
        self._init_database(db_manipulator)

    def get_database(self, recording_id: str) -> Optional["DatabaseManipulator"]:
        """
        Get the database for the given recording ID.
        """
        return self.db_manipulators.get(recording_id)

    def _dump_to_csv(self, data_entry: Dict[str, Any]):
        """ Dump data to a CSV file as a fallback """
        from detectflow.manipulators.manipulator import Manipulator
        from detectflow.utils.hash import get_numeric_hash

        destination_folder = Manipulator.create_folders(directories="dumps")[0]
        filepath = os.path.join(destination_folder, f"emergency_dump_{data_entry.get('recording_id', 'unknown')}_{get_numeric_hash()}.csv")
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_entry)
            print(f"Data dumped to {filepath}")

    def _check_backup_interval(self, recording_id: str):
        # Check if the backup should be performed
        self.backup_counters[recording_id] += 1
        if self.backup_interval and self.backup_counters[recording_id] >= self.backup_interval:
            self.backup_to_s3(recording_id)
            self.backup_counters[recording_id] = 0

    def _resolve_db_conflict(self, recording_id: str, db_path: str):
        """ Resolve a database conflict by merging the new database with the existing one """
        from detectflow.manipulators.input_manipulator import InputManipulator
        from detectflow.manipulators.database_manipulator import merge_databases

        logging.info(f"Resolving database conflict for recording ID {recording_id}.")

        # Specify the bucket and directory names
        bucket_name = InputManipulator.get_bucket_name_from_id(recording_id)
        directory_name = f"{InputManipulator.zero_pad_id(recording_id)}"
        print(rf'{directory_name}/({recording_id}|{directory_name}).db', bucket_name)

        # Attempt to locate the database file online and locally
        online_file = self.dataloader.locate_file_s3(rf'{directory_name}/({recording_id}|{directory_name}).db',
                                                    bucket_name, 'name')
        local_file = self.dataloader.locate_file_local(rf'({recording_id}|{directory_name}).db', os.path.dirname(db_path), 'name')

        # Get metadata of both files to resolve conflicts
        online_dict = self.dataloader.get_version_metadata_s3(online_file)
        local_dict = self.dataloader.get_version_metadata_local(local_file)

        # Resolve conflict based on the metadata
        if online_file is None:
            final_db_path = local_file if local_file else db_path
            logging.info(f"Database for recording ID {recording_id} not found online. Using local database: {os.path.basename(final_db_path)}.")
        elif local_file is None:
            # Fetch the file from S3
            download_path = self.fetch_from_s3(online_file, db_path)
            final_db_path = download_path if download_path else db_path
            logging.info(f"Database for recording ID {recording_id} not found locally. Using online database: {os.path.basename(final_db_path)}.")
        else:
            # Resolve conflicts
            if online_dict['date'] > local_dict['date'] and online_dict['size'] > local_dict['size']:
                # Fetch the file from S3
                download_path = self.fetch_from_s3(online_file, db_path)
                final_db_path = download_path if download_path else db_path
                logging.info(f"Online database for recording ID {recording_id} is newer. Using online database: {os.path.basename(final_db_path)}.")
            elif online_dict['date'] < local_dict['date'] and online_dict['size'] < local_dict['size']:
                final_db_path = local_file
                logging.info(f"Local database for recording ID {recording_id} is newer. Using local database: {os.path.basename(final_db_path)}.")
            else:
                # Fetch the file from S3
                tmp_path_online = os.path.join(os.path.dirname(db_path),
                                               f"{os.path.splitext(os.path.basename(db_path))[0]}_online.db")
                download_path = self.fetch_from_s3(online_file, tmp_path_online)

                # Rename the local file
                move_path = self.dataloader.move_file(local_file, os.path.dirname(db_path),
                                                  f"{os.path.splitext(os.path.basename(db_path))[0]}_local.db",
                                                  overwrite=True)

                # Merge databases
                try:
                    final_db_path = merge_databases(move_path, download_path, db_path)
                    logging.info(f"Databases for recording ID {recording_id} merged successfully.")
                except Exception as e:
                    logging.error(f"Error while merging databases for recording ID {recording_id}: {e}")
                    final_db_path = db_path

        print(f"Final database path: {final_db_path}")
        return final_db_path

    def backup_to_s3(self, recording_id: str, validate_upload: bool = True):
        from detectflow.manipulators.input_manipulator import InputManipulator

        if self.dataloader is None:
            logging.warning("No S3 manipulator provided. Skipping database S3 backup.")
            return

        db_manipulator = self.get_database(recording_id)
        if not db_manipulator:
            logging.error(f"Database manipulator for recording ID {recording_id} not found. Skipping S3 backup.")
            return

        # Get the local file path
        local_file_path = db_manipulator.db_file

        # Re-insert any emergency dumps
        try:
            if os.path.isdir(os.path.dirname(local_file_path)):
                db_manipulator.gather_dump_data(dumps_folder=os.path.dirname(local_file_path), delete_dumps=True, update_on_conflict=False)
                logging.info(f"Emergency dumps re-inserted for recording ID {recording_id}.")
        except Exception as e:
            logging.error(f"Error while gathering emergency dumps for recording ID {recording_id}: {e}")

        # Close db connection for safety
        db_manipulator.close_connection()

        # Specify the bucket and directory names
        bucket_name = InputManipulator.get_bucket_name_from_id(recording_id)
        directory_name = f"{InputManipulator.zero_pad_id(recording_id)}/"

        # Upload the file to S3
        try:
            self.dataloader.backup_file_s3(bucket_name, directory_name, local_file_path, validate_upload=validate_upload)
        except Exception as e:
            logging.error(f"Failed to backup database for recording ID {recording_id} to S3: {e}")
            return

    def fetch_from_s3(self, s3_path: str, local_path: str):

        # Fetch the file from S3
        recording_id = os.path.splitext(os.path.basename(local_path))[0]
        if recording_id in self.db_manipulators:
            print(f"Database for recording ID currently managed. Closing connection.")
            db = self.db_manipulators.get(recording_id, None)
            if db:
                db.close_connection()

        try:
            download_path = self.dataloader.download_file_s3(*self.dataloader.parse_s3_path(s3_path), local_path)
            logging.info(f"Database for recording ID {os.path.basename(local_path)} downloaded from S3.")
        except Exception as e:
            logging.error(f"Error while downloading database for recording ID {os.path.basename(local_path)} from S3: {e}")
            download_path = None
        return download_path


    def run(self):
        """ Process tasks from the queue """
        # TODO: Consider handling errors here to avoid crashing the whole process.
        #  Come up with a good way to fix errors. Test the script to see what errors may occur and then address them.

        # Initialize the database manipulators
        self.db_manipulators = {}
        if self.db_paths is not None:
            for recording_id, db_path in self.db_paths.items():
                db_manipulator = self._init_database_manipulator(db_path)
                self.db_manipulators[recording_id] = db_manipulator
                self._init_database(db_manipulator)
                self.backup_counters[recording_id] = 0
            self.data_batches = {recording_id: [] for recording_id in self.db_manipulators}
            logging.info(f"Database manipulators initialized: #{len(self.db_manipulators)}.")

        # Initialize S3Manipulator if not provided
        if self.dataloader is None:
            self.dataloader = Dataloader(S3_CONFIG)
            logging.warning("No Dataloader provided. Using a new instance for S3 backup.")

        # Start the processing
        logging.info(f"Process {multiprocessing.current_process().name} started.")
        mark_keys = {'id', 'status'}
        stop = False
        counter = 0
        while True:
            try:
                # Check for control messages
                if counter > 9:
                    print("Checking control queue.")
                    counter = 0
                if not self.control_queue.empty():
                    command, args = self.control_queue.get()
                    if command == 'add_database':
                        self.add_database(*args)  # Pass recording_id and db_path
                        logging.info(f"Adding database for recording ID {args[0]}.")
                    elif command == 'stop':
                        stop = True
                        logging.info("Stopping database manager.")
                    elif command == 'flush_all_batches':
                        self.flush_all_batches()
                        logging.info("Flushing all batches.")
                    elif command == 'flush_batch':
                        self.flush_batch(*args)
                        logging.info(f"Flushing batch for recording ID {args[0]}.")
                    elif command == 'mark_processed':
                        self._mark_recording_processed(*args)
                        logging.info(f"Marking recording as processed for recording ID {args[0]}.")
                    elif command == 'backup_to_s3':
                        self.backup_to_s3(*args)
                        logging.info(f"Backing up database for recording ID {args[0]} to S3.")
                    elif command == 'fetch_from_s3':
                        self.fetch_from_s3(*args)
                        logging.info(f"Fetching database for recording ID {args[0]} from S3.")
                    #self.control_queue.task_done()

                # Check if the queue is empty and stop is requested
                if self.queue.empty() and stop:
                    self.flush_all_batches()
                    break

                # Process the next data entry
                if not self.queue.empty():
                    data_entry = self.queue.get()
                    if not isinstance(data_entry, dict):
                        #self.queue.task_done()
                        raise TypeError(f"Invalid data type supplied to database manager: {type(data_entry)}")
                    elif isinstance(data_entry, dict) and set(data_entry.keys()) == mark_keys:
                        # Mark the recording as processed and flush its batch
                        self._mark_recording_processed(data_entry['id'])
                        #self.queue.task_done()
                        continue
                    else:
                        try:
                            self._process_input_data(data_entry)
                        except Exception as e:
                            raise RuntimeError(f"Error processing input data dictionary: {e} - {traceback.format_exc()}")
                        #finally:
                            #self.queue.task_done() # TODO: Test this to see if it works as expected

                # Sleep to prevent high CPU usage
                time.sleep(0.5)
                counter += 1
            except TypeError as e:
                logging.error(f"Type Error: {e} - {traceback.format_exc()}. Ignoring and continuing.")
            except Exception as e:
                logging.error(f"Error during database manager processing: {e} - {traceback.format_exc()}")

        # Clean up the database manager when processing is done
        self.clean_up()

    @profile_function_call(logging.getLogger(__name__))
    def _process_input_data(self, data_entry: Dict[str, Any]):
        """ Process a single DetectionResult object. Should be overwritten by subclasses to address specific
        structure of database tables and entries."""
        recording_id = data_entry.get("recording_id")

        # Get the appropriate db_manipulator for the recording ID
        db_manipulator = self.get_database(recording_id)

        if not db_manipulator:
            print(f"Recording ID {recording_id} not found in managed databases. Emergency data dump initiated.")
            self._dump_to_csv(data_entry)
            return

        for table_name, table_info in self.database_structure.items():
            column_names = [col[0] for col in table_info['columns']]
            if all([key in column_names for key in data_entry.keys()]):
                if table_info.get('batching', True):
                    # Add data to the manipulators batch
                    db_manipulator.add_to_batch(table_name, data_entry)
                else:
                    # Add data directly to the table
                    db_manipulator.insert(table_name, data_entry)

        # Check if the backup should be performed
        self._check_backup_interval(recording_id)

    def _mark_recording_processed(self, recording_id):
        """ Mark a recording as processed and flush its batch """
        # Add the recording to the list of processed databases
        self.processed_databases.add(recording_id)
        self.flush_batch(recording_id)
        self.backup_to_s3(recording_id)

    def flush_batch(self, recording_id):
        """ Insert remaining data from the batch of a specific video to its database """
        db_manipulator = self.get_database(recording_id)

        if not db_manipulator:
            print(f"Recording ID {recording_id} not found in managed databases.")
            return

        # Flush the batch of the recording
        try:
            db_manipulator.flush_batch()
            logging.info(f"Flushed batch for recording ID {recording_id}.")
        except Exception as e:
            logging.error(f"Error while flushing batch for recording ID {recording_id}: {e}")

    def flush_all_batches(self):
        for recording_id, _ in self.db_manipulators.items():
            self.flush_batch(recording_id)
        logging.info("Flushed all batches.")

    def clean_up(self):
        """ Clean up the database manager """
        if self.db_manipulators is not None:
            for recording_id, db_manipulator in self.db_manipulators.items():
                self.flush_batch(recording_id)
                self.backup_to_s3(recording_id)
                if recording_id not in self.processed_databases:
                    logging.warning(f"Recording ID {recording_id} was not marked as processed before cleanup.")

            recording_ids = list(self.db_manipulators.keys())
            for recording_id in recording_ids:
                self.db_manipulators.pop(recording_id)

            logging.info("Database manager cleaned up.")


def start_db_manager(db_paths: Optional[Dict[str, str]] = None,
                     batch_size: int = 100,
                     backup_interval: int = 500,
                     dataloader: Optional[Dataloader] = None,
                     database_structure: Optional[Dict[str, Any]] = None):

    control_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()

    manager = DatabaseManager(db_paths=db_paths, batch_size=batch_size, backup_interval=backup_interval,
                              control_queue=control_queue, data_queue=data_queue, dataloader=dataloader,
                              database_structure=database_structure)
    manager_process = multiprocessing.Process(target=manager.run, name="DatabaseManager")
    manager_process.start()
    return {'manager': manager,
            'process': manager_process,
            'control_queue': control_queue,
            'data_queue': data_queue
            }


def stop_db_manager(control_queue, manager_process):
    control_queue.put(('stop', []))
    manager_process.join()


def add_database_to_db_manager(control_queue, db_name, db_path):
    logging.info(f"Adding database {db_name} to control queue.")
    control_queue.put(('add_database', (db_name, db_path)))


def flush_one_db_manager(control_queue, recording_id):
    control_queue.put(('flush_batch', (recording_id,)))


def flush_all_db_manager(control_queue):
    control_queue.put(('flush_all_batches', []))


def mark_processed_db_manager(control_queue, recording_id):
    control_queue.put(('mark_processed', (recording_id,)))


def backup_file_db_manager(control_queue, recording_id):
    control_queue.put(('backup_to_s3', (recording_id, True)))


def fetch_file_db_manager(control_queue, s3_path, db_path):
    control_queue.put(('fetch_from_s3', (s3_path, db_path)))





