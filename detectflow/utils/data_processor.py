import matplotlib.pyplot
import traceback
import os
import sqlite3
from datetime import datetime, timedelta
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from detectflow.manipulators.box_manipulator import BoxManipulator
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.config.database_structure import VISITS_COLS, VISITS_CONSTR
import logging
import ast
import numpy as np

VISITS_EXT_COLS = [
    ("frame_number", "integer", ""),
    ("video_time", "text", "NOT NULL"),
    ("real_life_time", "text", ""),
    ("recording_id", "text", ""),
    ("video_id", "text", "NOT NULL"),
    ("video_path", "text", ""),
    ("flower_bboxes", "text", ""),
    ("all_visitor_bboxes", "text", ""),
    ("visit_ids", "text", ""),
    ("relevant_visitor_bboxes", "text", ""),
    ("relevant_visit_ids", "text", ""),
    ("on_flower", "boolean", ""),
    ("flags", "text", "")
]

PERIODS_COLS = [
    ("video_id", "text", "NOT NULL"),
    ("start_time", "text", "NOT NULL"),
    ("end_time", "text", "NOT NULL"),
    ("start_frame", "integer", "NOT NULL"),
    ("end_frame", "integer", "NOT NULL"),
    ("start_real_life_time", "text", "NOT NULL"),
    ("end_real_life_time", "text", "NOT NULL"),
    ("visit_duration", "real", "NOT NULL"),
    ("flower_bboxes", "text", ""),
    ("visitor_bboxes", "text", ""),
    ("frame_numbers", "text", ""),
    ("visit_ids", "text", ""),
    ("visitor_id", "integer", "NOT NULL"),
    ("visitor_species", "text", ""),
    ("flags", "text", ""),
    # --- New Columns ---
    ("FC", "integer", "DEFAULT 0"),  # Flower Contact
    ("AC", "integer", "DEFAULT 0"),  # Anther Contact
    ("CS", "integer", "DEFAULT 0"),  # Stigma Contact
    ("F", "integer", "DEFAULT 0"),   # Feeding
    ("R", "integer", "DEFAULT 0"),   # Robbing
    ("T", "integer", "DEFAULT 0"),   # Thieving
    ("num_F", "integer", "DEFAULT 0") # Number of Visited Flowers
    # --- End New Columns ---
]

PERIODS_CONSTR = "PRIMARY KEY (video_id, visitor_id)"


def load_dataframe(db_path: str, table_name: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        raise RuntimeError(f'Error loading dataframe from {db_path}. Manual intervention required: {e}') from e
    finally:
        conn.close()


def str_to_bool_list(s: str):
    # Replace 'true' and 'false' with 'True' and 'False'
    s = s.replace('true', 'True').replace('false', 'False')
    # Use ast.literal_eval to safely evaluate the string as a list
    return ast.literal_eval(s)


def filter_int_by_bool(int_list: list, bool_list: list):
    return [int_item for int_item, bool_item in zip(int_list, bool_list) if bool_item]


def refine_visits(df: pd.DataFrame):
    try:
        cols_to_list = [
            'all_visitor_bboxes',
            'relevant_visitor_bboxes',
            'flower_bboxes',
            'visit_ids'
        ]

        for col in cols_to_list:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert lists of floats to ints, string representations of lists of bools to actual lists of bools. Create a new column 'relevant_visit_ids' by filtering 'visit_ids' by 'on_flower'
        df[f'visit_ids'] = df[f'visit_ids'].apply(lambda x: [int(visit_id) for visit_id in x])
        df[f'on_flower'] = df[f'on_flower'].apply(str_to_bool_list)
        df[f'relevant_visit_ids'] = df.apply(lambda row: filter_int_by_bool(row[f'visit_ids'], row[f'on_flower']),
                                             axis=1)
    except Exception as e:
        logging.error(
            f'Error converting string representations of lists to actual lists. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Fill missing values in the relevant columns
        cols_to_fill = [
            'visit_ids',
            'relevant_visit_ids',
            'all_visitor_bboxes',
            'relevant_visitor_bboxes',
            'flower_bboxes',
            'on_flower',
            'flags',
            'rois'
        ]

        for col in cols_to_fill:
            df[col] = df[col].apply(lambda x: [] if not isinstance(x, (list, tuple)) and pd.isna(x) else x)

    except Exception as e:
        logging.error(f'Error filling missing values in the relevant columns. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        cols_to_int = [
            'year',
            'month',
            'day'
        ]

        for col in cols_to_int:
            df[col] = df[col].astype(int)
    except Exception as e:
        logging.error(f'Error converting columns to int. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Create a new datetime column from the separate date columns
        df['real_life_time'] = pd.to_datetime(
            df['year'].astype(str) + '-' +
            df['month'].astype(str) + '-' +
            df['day'].astype(str) + ' ' +
            df['life_time']
        )

        # Drop the unnecessary columns
        df = df.drop(columns=['life_time', 'year', 'month', 'day', 'rois'])
    except Exception as e:
        logging.error(f'Error creating real_life_time column. Further exceptions possible: {e}')
        traceback.print_exc()

    return df


def refine_visits_ext(df: pd.DataFrame):
    try:
        cols_to_list = [
            'all_visitor_bboxes',
            'relevant_visitor_bboxes',
            'flower_bboxes',
            'on_flower',
            'flags',
            'visit_ids',
            'relevant_visit_ids'
        ]

        for col in cols_to_list:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        # Convert lists of floats to ints, string representations of lists of bools to actual lists of bools.
        df[f'visit_ids'] = df[f'visit_ids'].apply(lambda x: [int(visit_id) for visit_id in x])
        df[f'relevant_visit_ids'] = df[f'relevant_visit_ids'].apply(lambda x: [int(visit_id) for visit_id in x])
        df[f'on_flower'] = df[f'on_flower'].apply(lambda x: [bool(on_f) for on_f in x])
        # df[f'on_flower'] = df[f'on_flower'].apply(str_to_bool_list)
    except Exception as e:
        logging.error(f'Error converting string representations of lists to actual lists. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Fill missing values in the relevant columns
        cols_to_fill = [
            'visit_ids',
            'relevant_visit_ids',
            'all_visitor_bboxes',
            'relevant_visitor_bboxes',
            'flower_bboxes',
            'on_flower',
            'flags'
        ]

        for col in cols_to_fill:
            df[col] = df[col].apply(lambda x: [] if not isinstance(x, (list, tuple)) and pd.isna(x) else x)

    except Exception as e:
        logging.error(f'Error filling missing values in the relevant columns. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Ensure 'real_life_time' and 'video_time' are datetime types
        df['real_life_time'] = pd.to_datetime(df['real_life_time'])
        df['video_time'] = pd.to_timedelta(df['video_time'])
    except Exception as e:
        logging.error(f'Error converting time columns. Further exceptions possible: {e}')
        traceback.print_exc()

    return df


def refine_periods(df: pd.DataFrame):
    try:
        cols_to_list = [
            'flower_bboxes',
            'visitor_bboxes',
            'frame_numbers',
            'visit_ids',
            'flags'
        ]

        for col in cols_to_list:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        # Convert lists of floats to ints, string representations of lists of bools to actual lists of bools.
        df['visit_ids'] = df['visit_ids'].apply(lambda x: [int(visit_id) for visit_id in x])
        df['frame_numbers'] = df['frame_numbers'].apply(lambda x: [int(visit_id) for visit_id in x])
    except Exception as e:
        logging.error(f'Error converting string representations of lists to actual lists. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Ensure 'start_real_life_time', 'end_real_life_time', 'start_time', and 'end_time' are datetime types
        df['start_real_life_time'] = pd.to_datetime(df['start_real_life_time'])
        df['end_real_life_time'] = pd.to_datetime(df['end_real_life_time'])
        df['start_time'] = pd.to_timedelta(df['start_time'])
        df['end_time'] = pd.to_timedelta(df['end_time'])
    except Exception as e:
        logging.error(f'Error converting time columns. Further exceptions possible: {e}')
        traceback.print_exc()

    try:
        # Ensure 'visit_duration' is an integer and calculate if missing
        df['visit_duration'] = df.apply(
            lambda row: float((row['end_real_life_time'] - row['start_real_life_time']).total_seconds())
            if pd.isna(row['visit_duration']) or float(row['visit_duration']) == 0.0
            else float(row['visit_duration']),
            axis=1
        )

        # Ensure 'visitor_id' is an integer
        df['visitor_id'] = df['visitor_id'].astype(int)

        # --- Add New Columns to Type Conversion ---
        cols_to_int = [
            'FC', 'AC', 'CS', 'F', 'R', 'T', 'num_F'
        ]
        for col in cols_to_int:
            if col in df.columns:
                # Fill NaNs with 0 before converting to int
                df[col] = df[col].fillna(0).astype(int)
            else:
                # If column doesn't exist (older DB), add it with default 0
                df[col] = 0

        # --- Ensure new integer columns are integers (redundant but safe) ---
        for col in cols_to_int:
            if col in df.columns:
                df[col] = df[col].astype(int)
        # --- End New Columns ---

    except Exception as e:
        logging.error(
            f'Error ensuring column data types and calculating visit duration. Further exceptions possible: {e}')
        traceback.print_exc()

    return df


def refine_videos(df: pd.DataFrame):
    try:
        # Convert appropriate columns to lists
        cols_to_list = [
            'focus_regions_start',
            'flowers_start',
            'focus_regions_end',
            'flowers_end'
        ]

        for col in cols_to_list:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert specific columns to their respective types
        df['fps'] = df['fps'].astype(int)
        df['length'] = df['length'].astype(int)
        df['total_frames'] = df['total_frames'].astype(int)
        df['focus'] = df['focus'].astype(float)
        df['blur'] = df['blur'].astype(float)
        df['contrast'] = df['contrast'].astype(float)
        df['brightness'] = df['brightness'].astype(float)
        df['focus_acc_start'] = df['focus_acc_start'].astype(float)
        df['focus_acc_end'] = df['focus_acc_end'].astype(float)
        df['motion'] = df['motion'].astype(float)

        # Convert 'daytime' to integer
        df['daytime'] = df['daytime'].astype(int)

        # Convert 'start_time' and 'end_time' to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

    except Exception as e:
        logging.error(f'Error refining videos dataframe. Further exceptions possible: {e}')
        traceback.print_exc()

    return df


class ResultsLoader:
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f'Database file not found: {db_path}')

        self.db_path = db_path
        self.db_man = DatabaseManipulator(db_path)

        self._videos_df = None
        self._visits_df = None
        self._ground_truth_df = None

    @property
    def videos_df(self):
        if self._videos_df is None:
            try:
                self._videos_df = self.db_man.load_dataframe('videos')
                self._videos_df = refine_videos(self._videos_df)
            except Exception as e:
                logging.error(f'Error loading and refining videos dataframe. Further exceptions possible: {e}')
                traceback.print_exc()
        return self._videos_df

    @property
    def visits_df(self):
        if self._visits_df is None:
            try:
                self._visits_df = self.initialize_visits_ext()
            except Exception as e:
                logging.error(f'Error loading and refining visits dataframe. Further exceptions possible: {e}')
                traceback.print_exc()
        return self._visits_df

    @property
    def ground_truth_df(self):
        if self._ground_truth_df is None:
            try:
                self._ground_truth_df = self.load_ground_truth()
            except Exception as e:
                logging.error(f'Error loading and refining ground truth dataframe. Further exceptions possible: {e}')
                traceback.print_exc()
        return self._ground_truth_df

    def create_visits_ext_table(self):
        try:
            self.db_man.create_table('visits_ext', VISITS_EXT_COLS, VISITS_CONSTR)
        except Exception as e:
            logging.error(f'Error creating visits_ext table. Further exceptions possible: {e}')
            traceback.print_exc()

    def create_periods_table(self, table_name: str = 'periods'):
        """
        Create the periods table in the SQLite database.
        """
        try:
            self.db_man.create_table(table_name, PERIODS_COLS, PERIODS_CONSTR)
        except Exception as e:
            logging.error(f'Error creating table {table_name}. Further exceptions possible: {e}')
            traceback.print_exc()

    def save_visits(self, visits_df):
        """
        Save the refined visits DataFrame to the visits_ext table in the SQLite database.
        """
        from detectflow.utils.extract_data import safe_json, safe_str, safe_datetime, safe_int, safe_timedelta

        data = []
        for row in visits_df.iterrows():
            entry = (
                safe_int(row[1]['frame_number']),
                safe_timedelta(row[1]['video_time']).split('.')[0],
                safe_datetime(row[1]['real_life_time']),
                safe_str(row[1]['recording_id']),
                safe_str(row[1]['video_id']),
                safe_str(row[1]['video_path']),
                safe_json(row[1]['flower_bboxes']),
                safe_json(row[1]['all_visitor_bboxes']),
                safe_json(row[1]['visit_ids']),
                safe_json(row[1]['relevant_visitor_bboxes']),
                safe_json(row[1]['relevant_visit_ids']),
                safe_json(row[1]['on_flower']),
                safe_json(row[1]['flags'])
            )
            data.append(entry)

        delete_query = "DELETE FROM visits_ext"
        try:
            self.db_man.safe_execute(delete_query)
        except Exception as e:
            logging.error(f'Error deleting existing visits_ext data in SQLite. Further exceptions possible: {e}')
            traceback.print_exc()

        query = f"""
            INSERT INTO visits_ext ({','.join([col[0] for col in VISITS_EXT_COLS])}) 
            VALUES ({','.join(['?'] * len(VISITS_EXT_COLS))}) 
            ON CONFLICT(frame_number, video_id)
            DO UPDATE SET {','.join([f"{col[0]} = excluded.{col[0]}" for col in VISITS_EXT_COLS])}
        """
        try:
            self.db_man.safe_execute(query, data)
        except Exception as e:
            logging.error(f'Error saving refined visits to SQLite. Further exceptions possible: {e}')
            traceback.print_exc()

    def save_periods(self, periods_df):
        """
        Save the processed visits DataFrame to the periods table in the SQLite database.
        """

        self.save_periods_table(periods_df, 'periods')

    def save_periods_gt(self, periods_df):
        """
        Save the processed visits DataFrame to the periods_gt table in the SQLite database.
        """

        self.save_periods_table(periods_df, 'periods_gt')

    def save_periods_table(self, periods_df, table_name: str):
        """
        Save the processed visits DataFrame to a periods type table with a specified name in the SQLite database.
        """

        from detectflow.utils.extract_data import safe_json, safe_str, safe_datetime, safe_float, safe_timedelta, \
            safe_int

        self.initialize_periods_table(table_name)

        data = []
        for row in periods_df.iterrows():
            entry = (
                safe_str(row[1]['video_id']),
                safe_timedelta(row[1]['start_time']).split('.')[0],
                safe_timedelta(row[1]['end_time']).split('.')[0],
                safe_int(row[1]['start_frame']),
                safe_int(row[1]['end_frame']),
                safe_datetime(row[1]['start_real_life_time']),
                safe_datetime(row[1]['end_real_life_time']),
                safe_float(row[1]['visit_duration']),
                safe_json(row[1]['flower_bboxes']),
                safe_json(row[1]['visitor_bboxes']),
                safe_json(row[1]['frame_numbers']),
                safe_json(row[1]['visit_ids']),
                safe_int(row[1]['visitor_id']),
                safe_str(row[1]['visitor_species']),
                safe_json(row[1]['flags']),
                # --- Add Defaults for New Columns ---
                safe_int(row[1]['FC']),
                safe_int(row[1]['AC']),
                safe_int(row[1]['CS']),
                safe_int(row[1]['F']),
                safe_int(row[1]['R']),
                safe_int(row[1]['T']),
                safe_int(row[1]['num_F'])
                # --- End New Columns ---
            )
            data.append(entry)

        delete_query = f"""DELETE FROM {table_name}"""
        try:
            self.db_man.safe_execute(delete_query)
        except Exception as e:
            logging.error(f'Error deleting existing {table_name} data in SQLite. Further exceptions possible: {e}')
            traceback.print_exc()

        query = f"""
                            INSERT INTO {table_name} ({','.join([col[0] for col in PERIODS_COLS])}) 
                            VALUES ({','.join(['?'] * len(PERIODS_COLS))}) 
                            ON CONFLICT(video_id, visitor_id)
                            DO UPDATE SET {','.join([f"{col[0]} = excluded.{col[0]}" for col in PERIODS_COLS])}
                        """
        try:
            self.db_man.safe_execute(query, data)
        except Exception as e:
            logging.error(f'Error saving processed visit periods to SQLite. Further exceptions possible: {e}')
            traceback.print_exc()

    def load_visits_ext(self):
        """
        Load the visits_ext DataFrame from the SQLite database and refine it.
        """
        visits_df = self.db_man.load_dataframe('visits_ext')
        return refine_visits_ext(visits_df)

    def load_visits(self):
        """
        Load the visits DataFrame from the SQLite database and refine it.
        """
        visits_df = self.db_man.load_dataframe('visits')
        return refine_visits(visits_df)

    def load_periods(self):
        """
        Load the periods DataFrame from the SQLite database and refine it.
        """
        self.initialize_periods_table('periods')

        periods_df = self.db_man.load_dataframe('periods')
        return refine_periods(periods_df)

    def load_periods_gt(self):
        """
        Load the periods_gt DataFrame from the SQLite database and refine it.
        """
        self.initialize_periods_table('periods_gt')

        periods_df = self.db_man.load_dataframe('periods_gt')
        return refine_periods(periods_df)

    def load_ground_truth(self):
        """
        Load the ground truth DataFrame from the SQLite database and refine it.
        """
        ground_truth_df = self.db_man.load_dataframe('ground_truth')
        return refine_visits(ground_truth_df)


    def initialize_visits_ext(self):
        """
        Initialize the visits_ext table: create if not exists and load/refine data.
        """
        try:
            table_names = self.db_man.get_table_names()
        except Exception as e:
            logging.error(f'Error getting table names from SQLite. Further exceptions possible: {e}')
            traceback.print_exc()
            table_names = []

        visits_df = None
        try:
            if 'visits_ext' not in table_names:
                self.create_visits_ext_table()
                visits_df = self.load_visits()
                self.save_visits(visits_df)
            else:
                visits_df = self.load_visits_ext()
        except Exception as e:
            logging.error(f'Error initializing visits_ext table. Further exceptions possible: {e}')
            traceback.print_exc()

        return visits_df

    def initialize_periods_table(self, table_name: str):
        try:
            table_names = self.db_man.get_table_names()
        except Exception as e:
            logging.error(f'Error getting table names from SQLite. Further exceptions possible: {e}')
            traceback.print_exc()
            table_names = []

        try:
            if table_name not in table_names:
                self.create_periods_table(table_name)
        except Exception as e:
            logging.error(f'Error initializing table {table_name}. Further exceptions possible: {e}')
            traceback.print_exc()


class VisitsProcessor(ResultsLoader):
    def __init__(self, db_path: str):
        super().__init__(db_path=db_path)

        self._has_periods = False

    @property
    def has_periods(self):
        try:
            self._has_periods = "periods" in self.db_man.get_table_names()
        except Exception as e:
            logging.error(f'Error checking if periods table exists. Further exceptions possible: {e}')
            traceback.print_exc()
        return self._has_periods

    def process_visits(self, df, iou_threshold=0.3, max_missing_frames=2):
        df['video_time'] = pd.to_timedelta(df['video_time'])
        df = df.sort_values(by=['video_id', 'frame_number'])

        processed_visits = []
        unique_visit_id = 1

        for video_id, group in df.groupby('video_id'):
            visits = self.merge_detections(group, iou_threshold, max_missing_frames)

            for visit in visits:
                visit['visitor_id'] = unique_visit_id
                unique_visit_id += 1
                visit['visit_duration'] = max(0.08, float((visit['end_real_life_time'] - visit['start_real_life_time']).total_seconds()))
                visit['visitor_species'] = ''
                visit['flags'] = []
                processed_visits.append(visit)

                # --- Ensure New Columns Exist (even if start_new_visit didn't add them) ---
                visit.setdefault('FC', 0)
                visit.setdefault('AC', 0)
                visit.setdefault('CS', 0)
                visit.setdefault('F', 0)
                visit.setdefault('R', 0)
                visit.setdefault('T', 0)
                visit.setdefault('num_F', 0)
                # --- End New Columns ---

        return pd.DataFrame(processed_visits)

    def merge_detections(self, group, iou_threshold, max_missing_frames):
        visits = []
        current_visits = {}  # Dictionary to track ongoing visits by visitor ID
        unique_temp_id = 0  # Counter for generating unique temporary IDs

        for _, row in group.iterrows():
            new_visits = []
            matched_visits = set()
            frame_number = row['frame_number']

            for idx, visitor_bbox in enumerate(row['relevant_visitor_bboxes']):
                visitor_id = row['relevant_visit_ids'][idx] if idx < len(row['relevant_visit_ids']) else -1
                if visitor_id == -1:
                    temp_visitor_id = f"temp_{unique_temp_id}"
                    unique_temp_id += 1
                else:
                    temp_visitor_id = visitor_id

                matched = False

                # Check for matches with ongoing visits
                for ongoing_visit_id, ongoing_visit in list(current_visits.items()):
                    if ongoing_visit['end_frame'] < row['frame_number'] - max_missing_frames:
                        visits.append(ongoing_visit)
                        del current_visits[ongoing_visit_id]
                    else:
                        if any(BoxManipulator.get_iou(visitor_bbox, box) > iou_threshold for box
                               in ongoing_visit['visitor_bboxes'][-1]):
                            if ongoing_visit_id not in matched_visits:
                                self.extend_visit(row, ongoing_visit, visitor_bbox, visitor_id)
                                matched_visits.add(ongoing_visit_id)
                                matched = True
                                break

                if not matched:
                    new_visit = self.start_new_visit(row, visitor_bbox, visitor_id)
                    new_visits.append((temp_visitor_id, new_visit))

            current_visits.update({visitor_id: visit for visitor_id, visit in new_visits})

        visits.extend(current_visits.values())
        return visits

    def start_new_visit(self, row, visitor_bbox, visitor_id):
        return {
            'video_id': row['video_id'],
            'start_time': row['video_time'],
            'end_time': row['video_time'],
            'start_frame': row['frame_number'],
            'end_frame': row['frame_number'],
            'start_real_life_time': row['real_life_time'],
            'end_real_life_time': row['real_life_time'],
            'flower_bboxes': row['flower_bboxes'],
            'visitor_bboxes': [[visitor_bbox]],
            'frame_numbers': [row['frame_number']],
            'visit_ids': [visitor_id],
            'visitor_id': visitor_id,
            # --- Add Defaults for New Columns ---
            'FC': 0,
            'AC': 0,
            'CS': 0,
            'F': 0,
            'R': 0,
            'T': 0,
            'num_F': 0
            # --- End New Columns ---
        }

    def extend_visit(self, row, current_visit, visitor_bbox, visitor_id):
        current_visit['end_time'] = row['video_time']
        current_visit['end_frame'] = row['frame_number']
        current_visit['end_real_life_time'] = row['real_life_time']
        current_visit['visitor_bboxes'].append([visitor_bbox])
        current_visit['frame_numbers'].append(row['frame_number'])
        current_visit['visit_ids'].append(visitor_id)
        return current_visit

    def filter_by_minimum_duration(self, visits_df, min_duration):
        """
        Filter visits based on a minimum duration.

        :param visits_df: DataFrame containing processed visits.
        :param min_duration: Minimum duration (in seconds) for a visit to be considered valid.
        :return: Filtered DataFrame with visits meeting the minimum duration requirement.
        """

        visits_df = visits_df.copy()

        # Convert min_duration to a timedelta object for comparison
        min_duration_td = pd.to_timedelta(min_duration, unit='s')

        # Calculate the duration of each visit
        visits_df['visit_duration_temp'] = visits_df['end_real_life_time'] - visits_df['start_real_life_time']

        # Filter out visits shorter than the minimum duration
        filtered_visits_df = visits_df[visits_df['visit_duration_temp'] >= min_duration_td].copy()

        # Drop the temporary 'visit_duration' column
        filtered_visits_df.drop(columns=['visit_duration_temp'], inplace=True)

        return filtered_visits_df

    def filter_bboxes_by_confidence(self, visits_df, confidence_threshold):
        """
        Filter bounding boxes in visits_df based on a confidence threshold.

        :param visits_df: DataFrame containing visit detections.
        :param confidence_threshold: Minimum confidence required for a bounding box to be retained.
        :return: DataFrame with filtered bounding boxes.
        """

        def filter_bboxes(bboxes):
            return [bbox for bbox in bboxes if bbox[4] >= confidence_threshold]

        visits_df['relevant_visitor_bboxes'] = visits_df['relevant_visitor_bboxes'].apply(filter_bboxes)
        return visits_df

    def filter_visits_by_confidence(self, visits_df, confidence_threshold, method='average'):
        """
        Filter visits based on bounding box confidence within each visit.

        :param visits_df: DataFrame containing processed visits.
        :param confidence_threshold: Minimum confidence required for a visit to be retained.
        :param method: Method to use for filtering ('average', 'minimum', 'maximum').
        :return: DataFrame with filtered visits.
        """

        def get_confidences(visitor_bboxes):
            return [bbox[4] for frame_bboxes in visitor_bboxes for bbox in frame_bboxes]

        def filter_visit_confidences(row):
            confidences = get_confidences(row['visitor_bboxes'])
            if method == 'average':
                return np.mean(confidences) >= confidence_threshold
            elif method == 'minimum':
                return np.min(confidences) >= confidence_threshold
            elif method == 'maximum':
                return np.max(confidences) >= confidence_threshold
            else:
                raise ValueError("Method must be 'average', 'minimum', or 'maximum'.")

        filtered_visits_df = visits_df[visits_df.apply(filter_visit_confidences, axis=1)]
        return filtered_visits_df

    def filter_by_reference_boxes(self, visits_df, reference_boxes, overlap_threshold: float = 0.1, expansion_radius: int = 0):
        """
        Filter visits_df based on user-specified reference boxes.

        :param visits_df: DataFrame containing visit detections.
        :param reference_boxes: List of user-specified reference boxes (flower boxes).
        :param overlap_threshold: Minimum overlap required to retain a bounding box.
        :param expansion_radius: Radius to expand reference boxes for overlap checking.
        :return: DataFrame with filtered bounding boxes based on reference boxes.
        """

        def expand_box(box, expansion_radius):
            x1, y1, x2, y2 = box[:4]
            return [x1 - expansion_radius, y1 - expansion_radius, x2 + expansion_radius, y2 + expansion_radius]

        def filter_boxes_with_reference(all_visitor_bboxes, reference_boxes, overlap_threshold, expansion_radius):
            relevant_bboxes = []
            on_flower_flags = []

            expanded_reference_boxes = [expand_box(box, expansion_radius) for box in reference_boxes]

            for visitor_bbox in all_visitor_bboxes:
                on_flower = any(BoxManipulator.get_overlap(visitor_bbox, ref_box) >= overlap_threshold for ref_box in
                                expanded_reference_boxes)
                on_flower_flags.append(on_flower)
                if on_flower:
                    relevant_bboxes.append(visitor_bbox)

            return relevant_bboxes, on_flower_flags

        # Update each row in visits_df based on reference boxes
        visits_df['flower_bboxes'] = visits_df.apply(lambda row: reference_boxes, axis=1)
        visits_df['relevant_visitor_bboxes'], visits_df['on_flower'] = zip(*visits_df.apply(
            lambda row: filter_boxes_with_reference(row['all_visitor_bboxes'], reference_boxes, overlap_threshold,
                                                    expansion_radius),
            axis=1
        ))
        visits_df['relevant_visit_ids'] = visits_df.apply(
            lambda row: [visitor_id for visitor_id, on_flower in zip(row['visit_ids'], row['on_flower']) if on_flower],
            axis=1)

        return visits_df


def plot_visits_with_overlaps(visits_df: pd.DataFrame, video_start: str, video_end: str, **kwargs):
    """
    Plot a Gantt chart to illustrate visit intervals relative to a video timeline, handling overlaps.

    :param visits_df: A pandas DataFrame containing visit intervals with columns 'start_time', 'end_time', etc.
                      The 'start_time' and 'end_time' columns should be in datetime format.
    :param video_start: The start time of the video in datetime format (string).
    :param video_end: The end time of the video in datetime format (string).
    :param title: The title of the plot.
    """

    visit_color = kwargs.get('visit_color', 'green')
    video_color = kwargs.get('video_color', 'grey')
    level_height = kwargs.get('level_height', 0.25)
    title = kwargs.get('title', 'Visits')
    labels = kwargs.get('labels', False)

    # Convert video start and end times to datetime
    video_start = pd.to_datetime(video_start)
    video_end = pd.to_datetime(video_end)

    video_duration = (video_end - video_start).total_seconds() # duration of the video in seconds

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the video timeline
    ax.broken_barh([(0, video_duration)], (0, level_height + 0.05), facecolors=video_color, edgecolor='black', label='Video')

    # Prepare intervals for plotting
    def get_intervals(df):
        intervals = []
        current_y = 1
        max_y = current_y

        for idx, row in df.iterrows():
            start_num = row['start_time'].total_seconds()
            duration = (row['end_time'] - row['start_time']).total_seconds()
            overlap = False

            # Check for overlap
            for interval in intervals:
                if not (start_num + duration < interval[0] or start_num > interval[0] + interval[1]):
                    overlap = True
                    current_y += level_height
                    break

            if not overlap:
                current_y = 1  # reset to first level if no overlap

            intervals.append((start_num, duration, current_y))
            max_y = max(max_y, current_y)

        return intervals, max_y

    # Plot the visits with broken_barh
    visit_intervals, max_y = get_intervals(visits_df)
    for idx, (start, duration, y) in enumerate(visit_intervals):
        ax.broken_barh([(start, duration)], (y, level_height), facecolors=visit_color, edgecolor='black',
                       label='Visits' if y == 1 else "")
        if labels:
            ax.text(start + duration / 2, y + level_height / 2, f'{idx + 1}', ha='center', va='center', color='white')

    # Adjusting the plot
    ax.set_yticks([0.5] + list(range(1, int(max_y) + 1)))
    ax.set_yticklabels(['Video'] + ['Visits'] * int(max_y))
    ax.set_ylim(0, max_y + 1)
    ax.set_xlabel('Time')
    ax.set_title(title)

    # Display x axis ticks as HH:MM:SS of video duration instead of seconds
    ax.set_xticks(range(0, int(video_duration), 60))
    ax.set_xticklabels([str(timedelta(seconds=x)) for x in range(0, int(video_duration), 60)])
    plt.xticks(rotation=45)

    # Add legend
    legend_elements = [
        Patch(facecolor=video_color, edgecolor='black', label='Video'),
        Patch(facecolor=visit_color, edgecolor='black', label='Visits')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()





def test(processor):

    # Get the visits DataFrame
    df = processor.visits_df

    # Filter bounding boxes by confidence
    confidence_threshold = 0.1  # Example confidence threshold
    df = processor.filter_bboxes_by_confidence(df, confidence_threshold)

    # Filter bounding boxes by reference boxes
    reference_boxes = [[500, 400, 600, 700], [0, 0, 1000, 600]]  # Example reference boxes
    iou_threshold = 0.1  # Example IoU threshold
    expansion_radius = 10  # Example expansion radius
    df = processor.filter_by_reference_boxes(df, reference_boxes, iou_threshold,
                                             expansion_radius)

    # Process the visits with the defined parameters
    iou_threshold = 0.8  # IoU threshold for detecting the same visitor
    max_missing_frames = 30  # Max number of missing frames to still consider as a continuous visit
    processed_visits_df = processor.process_visits(df, iou_threshold, max_missing_frames)

    # Filter visits by confidence
    confidence_threshold = 0.1  # Example confidence threshold
    method = 'minimum'  # Method for filtering ('average', 'minimum', 'maximum')
    processed_visits_df = processor.filter_visits_by_confidence(processed_visits_df, confidence_threshold, method)

    # Get unique video IDs from the visits_df
    unique_video_ids = processed_visits_df['video_id'].unique()
    videos_df = processor.videos_df

    for video_id in unique_video_ids:
        # Filter visits_df by video_id
        filtered_visits_df = processed_visits_df[processed_visits_df['video_id'] == video_id]

        # Get start_time and end_time for this video_id from videos_df
        video_info = videos_df[videos_df['video_id'] == video_id].iloc[0]
        video_start_time = video_info['start_time']
        video_end_time = video_info['end_time']

        plot_visits_with_overlaps(filtered_visits_df, video_start=video_start_time, video_end=video_end_time,
                                  title=f'{video_id}')

    return processed_visits_df

    # Filter visits by minimum duration
    # min_duration = 1  # seconds
    # long_visits_df = processor.filter_by_minimum_duration(processed_visits_df, min_duration)
    #
    # unique_video_ids = long_visits_df['video_id'].unique()
    # videos_df = processor.videos_df
    #
    # for video_id in unique_video_ids:
    #     # Filter visits_df by video_id
    #     filtered_visits_df = long_visits_df[long_visits_df['video_id'] == video_id]
    #
    #     # Get start_time and end_time for this video_id from videos_df
    #     video_info = videos_df[videos_df['video_id'] == video_id].iloc[0]
    #     video_start_time = video_info['start_time']
    #     video_end_time = video_info['end_time']
    #
    #     plot_visits_with_overlaps(filtered_visits_df, video_start=video_start_time, video_end=video_end_time,
    #                               title=f'LONG {video_id}')



if __name__ == '__main__':

    from detectflow.config import TESTS_DIR

    # Load the database
    db_path = os.path.join(TESTS_DIR, 'scratch', 'CZ1_M1_EupCyp02.db')

    # Define processor
    processor = VisitsProcessor(db_path)

    #df = processor.load_periods()

    df = test(processor)

    processor.save_periods(df)

    #
    # # Get the visits DataFrame
    # df = processor.visits_df
    #
    # processor.save_visits(df)

