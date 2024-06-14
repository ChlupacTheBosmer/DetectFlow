import sqlite3
import traceback
import os
import threading
import csv
from typing import Dict, List, Optional, Type, Any
from detectflow.utils.profile import profile_function_call
from detectflow.manipulators.s3_manipulator import S3Manipulator
import time
from datetime import timedelta, datetime
from queue import Queue
import logging
from detectflow.predict.results import DetectionResults
from detectflow.utils import DOWNLOADS_DIR


class DatabaseManager:

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
        ("frame_number", "integer", "NOT NULL"),
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
                    thumbnail blob,
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
        ("thumbnail", "blob", ""),
        ("focus_regions_start", "text", ""),
        ("flowers_start", "text", ""),
        ("focus_acc_start", "text", ""),
        ("focus_regions_end", "text", ""),
        ("flowers_end", "text", ""),
        ("focus_acc_end", "text", ""),
        ("motion", "real", "")
    ]

    def __init__(self,
                 db_manipulators: Optional[Dict[str, Type['DatabaseManipulator']]] = None,
                 batch_size: int = 100,
                 backup_interval: int = 500,
                 s3_manipulator: Optional[S3Manipulator] = None):
        """
        Initialize the DatabaseManager instance.

        Args:
        db_manipulators (Optional[Dict[str, Type['DatabaseManipulator']]]): A dictionary of database manipulators.
        batch_size (int): The size of each batch for processing.
        s3_manipulator (Optional[Type[S3Manipulator]]): An instance or subclass of S3Manipulator.
        """
        self.db_manipulators = db_manipulators
        self.lock = threading.Lock()
        self.processed_databases = set()
        self.backup_interval = backup_interval
        self.backup_counters = {}
        if self.db_manipulators is not None:
            for recording_id, db_manipulator in self.db_manipulators.items():
                db_manipulator.batch_size = batch_size
                self.init_database(db_manipulator)
                self.backup_counters[recording_id] = 0
        self.batch_size = batch_size
        self.data_batches = {recording_id: [] for recording_id in db_manipulators}
        self.queue = None
        self.s3_manipulator = s3_manipulator

    def init_database(self, db_manipulator):
        """
        Initialize the database and required tables for the given recording ID.
        """
        for table_name, columns in [("visits", self.VISITS_COLS), ("videos", self.VIDEOS_COLS)]:
            db_manipulator.create_table(table_name, columns)

    def add_database(self, recording_id: str, db_manipulator: Type["DatabaseManipulator"]):
        """
        Add a new database for the given recording ID and initialize it.
        """
        if recording_id in self.db_manipulators:
            print(f"Database for recording ID {recording_id} already exists.")
            return

        self.db_manipulators[recording_id] = db_manipulator
        self.data_batches[recording_id] = []
        self.backup_counters[recording_id] = 0

        # Initialize the database and required tables
        self.init_database(db_manipulator)

    def get_database(self, recording_id: str) -> Optional["DatabaseManipulator"]:
        """
        Get the database for the given recording ID.
        """
        return self.db_manipulators.get(recording_id)

    def dump_to_csv(self, data_entry: Dict[str, Any]):
        """ Dump data to a CSV file as a fallback """
        from detectflow.manipulators.manipulator import Manipulator
        from detectflow.utils.hash import get_numeric_hash

        destination_folder = Manipulator.create_folders(directories="dumps")[0]
        filepath = os.path.join(destination_folder, f"emergency_dump_{data_entry.get('recording_id', 'unknown')}_{get_numeric_hash()}.csv")
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_entry)
            print(f"Data dumped to {filepath}")

    def check_backup_interval(self, recording_id: str):
        # Check if the backup should be performed
        self.backup_counters[recording_id] += 1
        if self.backup_interval and self.backup_counters[recording_id] >= self.backup_interval:
            self.backup_to_s3(recording_id)
            self.backup_counters[recording_id] = 0

    def backup_to_s3(self, recording_id: str, validate_upload: bool = True):
        from detectflow.manipulators.input_manipulator import InputManipulator

        if self.s3_manipulator is None:
            logging.warning("No S3 manipulator provided. Skipping database S3 backup.")
            return

        db_manipulator = self.get_database(recording_id)
        if not db_manipulator:
            logging.error(f"Database manipulator for recording ID {recording_id} not found. Skipping S3 backup.")
            return

        # Close db connection for safety
        db_manipulator.close_connection()

        # Get the local file path
        local_file_path = db_manipulator.db_path

        # Specify the bucket and directory names
        bucket_name = InputManipulator.get_bucket_name_from_id(recording_id)
        directory_name = f"{InputManipulator.zero_pad_id(recording_id)}/"

        try:
            if not self.s3_manipulator.is_s3_bucket(bucket_name):
                logging.warning(f"Bucket {bucket_name} does not exist. Attempting backup into the 'data' bucket.")
                bucket_name = 'data'
                self.s3_manipulator.create_bucket_s3(bucket_name)
        except Exception as e:
            logging.error(f"Error during bucket resolution. Attempted to create bucket: {bucket_name}: {e}")
            bucket_name = 'data'
            if self.s3_manipulator.is_s3_bucket(bucket_name):
                logging.warning(f"Bucket {bucket_name} exists. Proceeding with backup into the 'data' bucket.")

        try:
            if not self.s3_manipulator.is_s3_directory(f"{bucket_name}/{directory_name}"):
                logging.warning(f"Directory {directory_name} does not exist. File backed up into the 'data' bucket.")
                bucket_name = 'data'
                self.s3_manipulator.create_directory_s3(bucket_name, directory_name)
        except Exception as e:
            logging.error(f"Error during directory resolution. Attempted to create directory: {bucket_name}/{directory_name}: {e}")
            bucket_name = 'data'
            if self.s3_manipulator.is_s3_directory(f"{bucket_name}/{directory_name}"):
                logging.warning(f"Directory {bucket_name}/{directory_name} exists. Proceeding with backup into the 'data' bucket.")

        s3_file_path = f"{directory_name}{os.path.basename(local_file_path)}"  # Path in the bucket where the file will be uploaded
        try:
            # Upload a file to the specified directory
            self.s3_manipulator.upload_file_s3(bucket_name, local_file_path, s3_file_path)
            logging.info(f"Uploaded {local_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
        except Exception as e:
            logging.error(f"Failed to upload {local_file_path} to S3: {e}")

        if validate_upload:
            self.validate_backup_to_s3(bucket_name, s3_file_path, local_file_path)

    def validate_backup_to_s3(self, bucket_name: str, s3_file_path: str, db_file_path: str):
        from detectflow.utils.file import compare_file_sizes

        tmp_folder = os.path.join(DOWNLOADS_DIR, os.path.dirname(db_file_path))
        tmp_file_path = os.path.join(tmp_folder, os.path.basename(db_file_path))
        os.makedirs(tmp_folder, exist_ok=True)

        try:
            # Validate the upload
            if self.s3_manipulator.is_s3_file(f"s3://{bucket_name}/{s3_file_path}"):
                self.s3_manipulator.download_file_s3(bucket_name, s3_file_path, tmp_file_path)
                if compare_file_sizes(db_file_path, tmp_file_path, 0.05): # Tolerance of 5%
                    logging.info(f"Successfully validated the upload of {db_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
                    # TODO: Could add additional validation by constructing a manipulator and reading something from the database
                else:
                    logging.warning(f"Partially validated the upload of {db_file_path} to S3 bucket {bucket_name}/{s3_file_path}. Size discrepancy found.")
            else:
                logging.error(f"Failed to validate the upload of {db_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
        except Exception as e:
            logging.error(f"Failed to validate the upload of {db_file_path} to S3: {e}")

    def process_queue(self, queue: Queue):
        """ Process tasks from the queue """
        # TODO: Consider handling errors here to avoid crashing the whole process.
        #  Come up with a good way to fix errors. Test the script to see what errors may occur and then address them.
        mark_keys = {'id', 'status'}
        self.queue = queue
        while True:
            data_entry = queue.get()
            if data_entry is None:
                # Signal to stop processing
                self.flush_all_batches()
                break
            elif not isinstance(data_entry, dict):
                raise RuntimeError(f"Invalid data type supplied to database manager: {type(data_entry)}")
            else:
                if isinstance(data_entry, dict) and set(data_entry.keys()) == mark_keys:
                    # Mark the recording as processed and flush its batch
                    self.mark_recording_processed(data_entry['id'])
                    continue
                try:
                    self.process_input_data(data_entry)
                except Exception as e:
                    raise RuntimeError(f"Error processing input data dictionary: {e} - {traceback.format_exc()}")
                finally:
                    self.queue.task_done() # TODO: Test this to see if it works as expected

    @profile_function_call(logging.getLogger(__name__))
    def process_input_data(self, data_entry: Dict[str, Any]):
        """ Process a single DetectionResult object. Should be overwritten by subclasses to address specific
        structure of database tables and entries."""
        recording_id = data_entry.get("recording_id")

        # Get the appropriate db_manipulator for the recording ID
        db_manipulator = self.get_database(recording_id)

        if not db_manipulator:
            print(f"Recording ID {recording_id} not found in managed databases. Emergency data dump initiated.")
            self.dump_to_csv(data_entry)
            return

        if any(["total_frames", "fps", "focus", "blur", "contrast", "brightness"]) in data_entry.keys():
            # Add data to the manipulators batch
            db_manipulator.insert("videos", data_entry)
        else:
            # Add data to the manipulators batch
            db_manipulator.add_to_batch("visits", data_entry)

        # Check if the backup should be performed
        self.check_backup_interval(recording_id)

    def mark_recording_processed(self, recording_id):
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
        db_manipulator.flush_batch()
        print(f"Flushed batch for <{recording_id}>.")

    def flush_all_batches(self):
        for recording_id, _ in self.db_manipulators.items():
            self.flush_batch(recording_id)
        logging.info("Flushed all batches.")

    def clean_up(self):
        if self.queue is not None:
            self.queue.join()
            self.queue = None

        for recording_id, db_manipulator in self.db_manipulators.items():
            self.flush_batch(recording_id)
            self.backup_to_s3(recording_id)
            if recording_id not in self.processed_databases:
                logging.warning(f"Recording ID {recording_id} was not marked as processed before cleanup.")
            self.db_manipulators.pop(recording_id, None)
        logging.info("Database manager cleaned up.")

    def __del__(self):
        self.clean_up()





