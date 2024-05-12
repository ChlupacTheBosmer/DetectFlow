import itertools
from detectflow.validators.validator import Validator
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.handlers.checkpoint_handler import CheckpointHandler
from concurrent.futures import ThreadPoolExecutor
import sqlite3


class DatasetSourceProcessor(CheckpointHandler):
    def __init__(self,
                 database_path: str,
                 working_dir: str,
                 max_workers: int = 1,
                 checkpoint_file: str = "DatasetProcessor_checkpoint.json",
                 **kwargs):
        super().__init__(checkpoint_file)

        self.database_path = database_path
        self.working_dir = working_dir
        self.max_workers = max_workers
        self.kwargs = kwargs

    def get_dataset_data(self, table=None, columns=None, condition=None):
        """
        Retrieves specified data from a dataset database.

        Args:
            dataset_database (str): The path to the dataset database.
            table (str):  The name of the table to query.
            columns (list of str): A list of column names to retrieve from the database. Defaults to all columns.
            condition (str): An optional SQL WHERE clause to apply when fetching the data.

        Returns:
            list: A list of tuples containing the requested data from the database.
        """
        if not Validator.is_valid_file_path(self.database_path):
            raise FileNotFoundError(f"Database file could not be found: {self.database_path}")
        try:
            # Create an instance of the DatabaseManipulator
            db = DatabaseManipulator(self.database_path)
            db.create_connection()

            # Determine which columns to select
            if not columns:
                columns = '*'  # Select all columns if none are specified
            else:
                columns = ', '.join(columns)  # Create a string from the list of columns

            # Construct the SQL query
            query = f"SELECT {columns} FROM {table}"
            if condition:
                query += f" WHERE {condition}"

            # Fetch all the records as per the query
            all_frames = db.fetch_all(query)

            # Close the database connection
            db.close_connection()
            del db
        except sqlite3.Error as sqle:
            raise RuntimeError(f"Error handling the dataset database: {self.database_path}.") from sqle
        except Exception as e:
            raise RuntimeError(f"Unexpected error when retrieving dataset database data: {self.database_path}.") from e

        return all_frames

    def sort_dataset_data(self, dataset_data, sort_index: int = 1):

        # Transform dataset data
        dataset_data_sorted = sorted(dataset_data, key=lambda x: x[
            sort_index])  # Sort the list by the key you want to group by, in this case, the second item in each tuple
        groups = itertools.groupby(dataset_data_sorted, key=lambda x: x[
            sort_index])  # Using groupby to sort data into groups by video_file_id
        frames_by_video = {video_id: list(frames) for video_id, frames in
                           groups}  # Convert groupby object to a dictionary where each key is a video_file_id and value is list of frames

        return frames_by_video

    def process_dataset(self, workers: int = 4, frames_by_video=None, **kwargs):
        if not frames_by_video:
            raise ValueError("No frames data provided.")

        if workers == 1:
            for video_id, frames in frames_by_video.items():
                self.process_video(video_id, frames, **kwargs)
        else:
            # Using ThreadPoolExecutor to manage concurrent execution
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.process_video, video_id, frames, **kwargs)
                           for video_id, frames in frames_by_video.items()]

                # Wait for all futures to complete (optional, only if you need to process results)
                for future in futures:
                    try:
                        result = future.result()  # This blocks until the future is complete
                        print("Video future completed.")  # Optional: process or print the result
                    except Exception as e:
                        print(f"An error occurred: {e}")

    def process_video(self, video_id, frames_data, **kwargs):

        raise NotImplementedError("Subclasses should implement this method.")

    def run(self):

        # # Get dataset data from database
        # dataset_data = self.get_dataset_data(table, columns)
        #
        # # Sort dataset data
        # frames_by_video = self.sort_dataset_data(dataset_data, sort_index=1)
        #
        # # Process dataset
        # self.process_dataset(workers=self.max_workers, frames_by_video=frames_by_video)

        # Should be properly implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")