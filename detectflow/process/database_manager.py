import sqlite3
import traceback
import os
import threading
import csv
from typing import Dict, List
from detectflow.utils.hash import get_numeric_hash
from detectflow.utils.profile import profile_function_call
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.validators.validator import Validator
import time
import re
from datetime import timedelta, datetime
from queue import Queue
import logging
from detectflow.predict.results import DetectionResults
import json


class DatabaseManager:
    def __init__(self, database_paths: Dict = {}, batch_size: int = 100):
        """
        Initialize the DatabaseManager instance.

        Args:
        database_paths (dict): A dictionary mapping video IDs to their respective database file paths.
        """
        self.database_paths = database_paths
        self.lock = threading.Lock()
        self.processed_videos = set()
        self.initialize_all_databases()
        self.batch_size = batch_size
        self.data_batches = {video_id: [] for video_id in database_paths}
        self.queue = None
        self.s3manipulator = S3Manipulator()

    def create_connection(self, db_file):
        """ Create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(f"Error connecting to database {db_file}: {e}")
            print(traceback.format_exc())
        return conn

    def create_table(self, conn, create_table_sql):
        """ Create a table from the create_table_sql statement """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except sqlite3.Error as e:
            print(f"Error creating table in database: {e}")
            print(traceback.format_exc())

    def initialize_all_databases(self):
        for video_id, db_path in self.database_paths.items():
            self.add_and_initialize_database(video_id, db_path)

    def add_and_initialize_database(self, video_id, db_path):
        """
        Add a new database for the given video ID and initialize it.
        """
        if video_id in self.database_paths:
            print(f"Database for video ID {video_id} already exists.")
            return

        self.database_paths[video_id] = db_path
        self.data_batches[video_id] = []
        if not os.path.exists(db_path):
            conn = self.create_connection(db_path)
            if conn is not None:
                self.create_visits_table(conn)
                self.create_videos_table(conn)
                conn.close()
            else:
                print(f"Failed to create database for video ID {video_id}")

    def create_visits_table(self, conn):
        """ Create the 'visits' table in the given database connection """
        sql_create_visits_table = """
            CREATE TABLE IF NOT EXISTS visits (
                frame_number integer PRIMARY KEY,
                video_time text NOT NULL,
                life_time text,
                year integer,
                month integer,
                day integer,
                recording_id text,
                video_ID text,
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
        self.create_table(conn, sql_create_visits_table)

    def create_videos_table(self, conn):
        """ Create the 'videos' table in the given database connection """
        sql_create_videos_table = """
            CREATE TABLE IF NOT EXISTS videos (
                s3_bucket text,
                s3_directory text,
                recording_id text,
                video_id text PRIMARY KEY,
                length real,
                total_frames integer,
                fps real,
                format text,
                day_night text,
                flowers_start integer,
                flowers_end integer,
                motion_all real,
                motion_rois real
            );
        """
        self.create_table(conn, sql_create_videos_table)

    def safe_execute(self, conn, sql, data, retries=3):
        """ Safely execute a SQL command with robust error handling and retry mechanism """
        for attempt in range(retries):
            try:
                with self.lock:
                    cur = conn.cursor()
                    cur.execute(sql, data)
                    conn.commit()
                    return True  # Successful execution
            except sqlite3.Error as e:
                print(f"SQLite error on attempt {attempt + 1}: {e}")
                print("Traceback:", traceback.format_exc())
                if attempt == retries - 1:
                    self.dump_to_csv(data)  # Final attempt failed, dump data to CSV
                    return False
                time.sleep(1)  # Wait before retrying

    def dump_to_csv(self, data):
        """ Dump data to a CSV file as a fallback """
        filename = f"emergency_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print(f"Data dumped to {filename}")

    def insert_visits_batch(self, video_id, visits_data_batch):
        """ Insert a batch of visits data into the database """
        db_path = self.database_paths.get(video_id)
        if not db_path:
            print(f"No database path found for video ID {video_id}")
            return False

        conn = self.create_connection(db_path)
        if conn is None:
            print(f"Failed to connect to database for video ID {video_id}")
            return False

        sql = ''' INSERT OR REPLACE INTO visits(frame_number, video_time, life_time, year, month, day, recording_id, video_ID, video_path, flower_bboxes, rois, all_visitor_bboxes, relevant_visitor_bboxes, visit_ids, on_flower, flags)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        success = True
        for visit_data in visits_data_batch:
            if not self.safe_execute(conn, sql, visit_data):
                success = False

        conn.close()
        return success

    def insert_video(self, video_id, video_data):
        """ Insert a single video data entry into the database """
        db_path = self.database_paths.get(video_id)
        if not db_path:
            print(f"No database path found for video ID {video_id}")
            return False

        conn = self.create_connection(db_path)
        if conn is None:
            print(f"Failed to connect to database for video ID {video_id}")
            return False

        sql = ''' INSERT OR REPLACE INTO videos(s3_bucket, s3_directory, recording_id, video_id, length, total_frames, fps, format, day_night, flowers_start, flowers_end, motion_all, motion_rois)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        success = self.safe_execute(conn, sql, video_data)
        conn.close()
        return success

    def backup_to_s3(self, video_id): #TODO: Implement S3 backup logic - take inspiration in JobHandler
        db_path = self.database_paths.get(video_id)
        if db_path and os.path.exists(db_path):
            try:
                # Use S3Manipualtor to upload the file
                bucket_name = "data"
                directory_name = "visits"
                s3.upload_file(db_path, s3_bucket, os.path.join(s3_directory, os.path.basename(db_path)))

                # Upload file
                self.s3manipulator.upload_file_s3(bucket_name, db_path,
                                                  os.path.join(directory_name, os.path.basename(db_path)),
                                                  max_attempts=3, delay=2)
                print(f"Database for video ID {video_id} backed up to S3.")
            except Exception as e:
                print(f"Failed to backup database for video ID {video_id} to S3: {e}")
                print(traceback.format_exc())

    def process_queue(self, queue: Queue):
        """ Process tasks from the queue """
        self.queue = queue
        while True:
            detection_result = queue.get()
            if detection_result is None:
                # Signal to stop processing
                self.flush_all_batches()
                break

            self.process_detection_result(detection_result)

    @profile_function_call(logging.getLogger(__name__))
    def process_detection_result(self, detection_result):
        """ Process a single DetectionResult object """
        video_id = detection_result.video_id
        data = self.extract_data_from_result(detection_result)

        if video_id not in self.data_batches:
            print(f"Video ID {video_id} not found in database paths. Emergency data dump initiated.")
            self.dump_to_csv(data)
            return

        self.data_batches[video_id].append(data)
        if len(self.data_batches[video_id]) >= self.batch_size:
            self.insert_visits_batch(video_id, self.data_batches[video_id])
            self.data_batches[video_id] = []

    def extract_data_from_result(self, result) -> List:
        """ Extract relevant data from a DetectionResult object """
        # Implement extraction logic based on the structure of DetectionResults
        # TODO: This function should be left not implement so it can be implemented by subclassing the class

        print(type(result))

        if isinstance(result, DetectionResults):

            # Data from the result
            frame_number = result.frame_number if result.frame_number is not None else get_numeric_hash()
            video_time = timedelta(seconds=result.video_time) if result.video_time is not None else timedelta(seconds=0)
            life_time = result.real_time
            year = life_time.year if isinstance(life_time, datetime) else 0
            month = life_time.month if isinstance(life_time, datetime) else 0
            day = life_time.day if isinstance(life_time, datetime) else 0
            recording_id = result.recording_id
            video_id = result.video_id
            video_path = result.source_path
            flower_bboxes = result.ref_boxes.to_list() if result.ref_boxes is not None else None
            all_visitor_bboxes = result.boxes.to_list() if result.boxes is not None else None
            relevant_visitor_bboxes = result.fil_boxes.to_list() if result.fil_boxes is not None else None
            visit_ids = result.boxes.id.to_list() if result.boxes is not None and result.boxes.id is not None else None
            on_flower = result.on_flowers.to_list()

            # I dont remember
            rois = ""
            flags = ""  # TODO: Data preprocessing and evaluate where there is a visitor likely and where not. Based on prob and so on

            # Validation
            self._validate_data_values(frame_number, video_time, recording_id, video_id, video_path)

            return [int(frame_number),
                    str(video_time),
                    str(life_time.time() if isinstance(life_time, datetime) else None),
                    int(year),
                    int(month),
                    int(day),
                    str(recording_id),
                    str(video_id),
                    str(video_path),
                    json.dumps(flower_bboxes),
                    rois,
                    json.dumps(all_visitor_bboxes),
                    json.dumps(relevant_visitor_bboxes),
                    json.dumps(visit_ids),
                    json.dumps(on_flower),
                    flags]

        else:
            frame_number = get_numeric_hash()
            video_time = timedelta(seconds=0)
            flags = f"Got instance of {str(type(result))} instead of DetectionResults"
            placeholders = ["" for i in range(1, 13)]
            data_values = [frame_number] + [video_time] + placeholders + [flags]

            # Validation #ERROR: Fix this flawed logic
            self._validate_data_values(frame_number, video_time, recording_id, video_id, video_path)

            return data_values

    def mark_video_processed(self, video_id):
        """ Mark a video as processed and flush its batch """
        with self.lock:
            self.processed_videos.add(video_id)
            self.flush_batch(video_id)
            self.backup_to_s3(video_id)

    def flush_batch(self, video_id):
        """ Insert remaining data from the batch of a specific video to its database """
        if video_id in self.data_batches and self.data_batches[video_id]:
            self.insert_visits_batch(video_id, self.data_batches[video_id])
            self.data_batches[video_id] = []
            print(f"Flushed batch for <{video_id}>.")

    def flush_all_batches(self):
        for video_id, db_path in self.database_paths.items():
            self.flush_batch(video_id)
        print("Flushed all batches.")

    def _validate_data_values(self,
                              frame_number,
                              video_time,
                              recording_id,
                              video_id,
                              video_path):

        # Frame number should not be larger than 30K ussually
        if not isinstance(frame_number, int):
            try:
                frame_number = int(frame_number)
            except Exception as e:
                print(f"Error when validating frame_number: {e}")

        if not frame_number < 30000:
            print(f"The frame number '{frame_number}' is larger than expected.")

        # Video time will not ussually be higher than 960 s
        try:
            if not isinstance(video_time, timedelta):
                raise TypeError(
                    f"Unexpected format in video_time. Expected <timedelta>, got <{type(video_time)}> instead.")

            if video_time <= timedelta(seconds=0):
                raise ValueError(f"Unexpected video_time value '{video_time}.' Expected value larger than 0.")
        except Exception as e:
            print(f"Error when validating video time: {e}")

        # Video IDS
        try:
            # Recording ID has a format of XX(X)0_X0_XXXXXX00
            recording_id_pattern = r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}$'

            if re.match(recording_id_pattern, recording_id) is None:
                raise ValueError(f"The string '{recording_id}' does not match the expected format.")

            # Video ID has a format of XX(X)0_X0_XXXXXX00_00000000_00_00
            video_id_pattern = (r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}_'
                                r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])_'
                                r'([01]\d|2[0-3])_([0-5]\d)$')

            if re.match(video_id_pattern, video_id) is None:
                raise ValueError(f"The string '{video_id}' does not match the expected format.")

        except Exception as e:
            print(f"Error when validating video IDs: {e}")
            # Decide what to do to handle these errors.

        # Path validation
        try:
            if not Validator.is_valid_file_path(video_path):
                raise ValueError(f"The video path '{video_path}' is not a valid file.")
        except Exception as e:
            print(f"Error when validating video filepath: {e}")
            # Decide what to do to handle this error.