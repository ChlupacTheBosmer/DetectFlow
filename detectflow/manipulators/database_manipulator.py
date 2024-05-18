import sqlite3
from sqlite3 import Error
import csv
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.validator import Validator
import os
import threading
import traceback
import time
from detectflow.utils.hash import get_numeric_hash
import logging


class DatabaseManipulator:
    def __init__(self, db_file: str, batch_size: int = 50):
        """
        Initialize the DatabaseManipulator object with the path to the SQLite database file.

        :param db_file: A string representing the path to the SQLite database file.
        Example: 'example.db'
        """
        self.db_file = db_file
        self._db_name = os.path.splitext(os.path.basename(self.db_file))[0]
        self.conn = None
        self.lock = threading.RLock()  # Assuming you are using threading for concurrency
        self.batch_data = []
        self.batch_table = None
        self.batch_size = batch_size

    def create_connection(self):
        """
        Create a database connection to the SQLite database specified by the db_file.
        If the database file does not exist, it will be created.
        """
        with self.lock:
            try:
                self.conn = sqlite3.connect(self.db_file)
                print(f"SQLite connection is opened to {self.db_file}")
            except Error as e:
                print(f"Error connecting to database: {e}")

    def close_connection(self):
        """
        Close the database connection.
        """
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None
                print("The SQLite connection is closed.")

    def execute_query(self, query, params=None):
        """
        Execute a SQL query optionally using parameters.

        :param query: A string containing a SQL query.
        Example: 'INSERT INTO users (name, age) VALUES (?, ?)'
        :param params: A tuple containing parameters to substitute into the query.
        Example: ('John', 30)
        :return: The cursor object if the query was executed successfully, or None if an error occurred.
        """
        with self.lock:
            try:
                if not self.conn:
                    self.create_connection()
                cur = self.conn.cursor()
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                self.conn.commit()
                return cur
            except Error as e:
                self.conn.rollback()
                print(f"An error occurred: {e}")

    def safe_execute(self, sql: str, data: tuple = None, retries: int = 3, use_transaction: bool = True):
        """
        Execute a SQL command with error handling, retry mechanism, and optional transaction control.

        :param sql: SQL command to be executed.
        :param data: Tuple of data to be used in the SQL command.
        :param retries: Number of retry attempts before failing.
        :param use_transaction: Whether to use transaction control (commit/rollback).
        :return: True if successful, False otherwise.
        """
        for attempt in range(retries):
            try:
                with self.lock:
                    if not self.conn:
                        self.create_connection()
                    cur = self.conn.cursor()
                    if use_transaction:
                        cur.execute("BEGIN;")
                    if data:
                        cur.execute(sql, data)
                    else:
                        cur.execute(sql)
                    if use_transaction:
                        self.conn.commit()
                    return True
            except sqlite3.Error as e:
                print(f"SQLite error on attempt {attempt + 1}: {e}")
                print("Traceback:", traceback.format_exc())
                if use_transaction:
                    self.conn.rollback()
                if attempt == retries - 1 and data:
                    self.dump_to_csv(data)  # Dump data to CSV on final failure
                    return False
                time.sleep(1)  # Wait before retrying

    def dump_to_csv(self, data):
        """ Dump data to a CSV file as a fallback """
        destination_folder = Manipulator.create_folders(directories="dumps")[0]
        filepath = os.path.join(destination_folder, f"emergency_dump_{self._db_name}_{get_numeric_hash()}.csv")
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print(f"Data dumped to {filepath}")

    def fetch_all(self, query, params=None):
        """
        Fetch all rows from the database following a query.

        :param query: A SQL query string to retrieve data.
        :param params: Optional parameters to pass to the query.
        :return: A list of tuples representing the fetched rows.
        """
        cur = self.execute_query(query, params)
        if cur:
            return cur.fetchall()
        return []

    def fetch_one(self, query, params=None):
        """
        Fetch a single row from the database.

        :param query: A SQL query string to retrieve data.
        :param params: Optional parameters to pass to the query.
        :return: A single tuple representing the fetched row, or None if no row was fetched.
        """
        cur = self.execute_query(query, params)
        if cur:
            return cur.fetchone()
        return None

    def create_table(self, table_name: str, columns: list):
        """
        Create a table using a list of column definitions.

        :param table_name: Name of the table to create.
        :param columns: List of column definitions in the format (name, data_type, constraints).
        """
        try:
            # Construct the column definitions string
            columns_str = ', '.join([f"{col[0]} {col[1]} {col[2]}" for col in columns])

            # Construct the CREATE TABLE SQL statement
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"

            # Execute the SQL statement
            self.safe_execute(query, use_transaction=True)
            print("Table created successfully.")
        except sqlite3.Error as e:
            print(f"Failed to create table: {e}")

    def insert(self, table, data, use_transaction=False):
        """
        Insert data into a table.

        :param table: A string specifying the table to insert data into.
        :param data: A dictionary where the keys are column names and the values are data to be inserted.
        Example: {'name': 'Alice', 'age': 25}
        """
        with self.lock:
            columns = ', '.join(data.keys())
            placeholders = ', '.join('?' for _ in data)
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            self.safe_execute(query, tuple(data.values()), use_transaction=use_transaction)

    def update(self, table, data, condition, use_transaction=False):
        """
        Update data in a table based on a condition.

        :param table: A string specifying the table to update.
        :param data: A dictionary where keys are column names to be updated, and values are the new data values.
        :param condition: A string specifying the SQL condition for updating records.
        Example: 'id = 1'
        """
        with self.lock:
            updates = ', '.join(f"{k} = ?" for k in data.keys())
            query = f"UPDATE {table} SET {updates} WHERE {condition}"
            self.safe_execute(query, tuple(data.values()), use_transaction=use_transaction)

    def delete(self, table, condition, use_transaction=False):
        """
        Delete data from a table based on a condition.

        :param table: A string specifying the table to delete from.
        :param condition: A string specifying the SQL condition for deleting records.
        Example: 'id = 1'
        """
        with self.lock:
            query = f"DELETE FROM {table} WHERE {condition}"
            self.safe_execute(query, use_transaction=use_transaction)

    def add_to_batch(self, table, data):
        """
        Adds a data dictionary to the current batch for later insertion into the specified database table.
        If the table changes, existing batch data for the previous table is flushed and a new batch is started.
        The batch is also flushed when it reaches the predefined batch size.

        Args:
            table (str): The name of the database table where the data will be inserted.
                         This should correspond to a valid table name in the database.
            data (dict): A dictionary where the keys are the column names and the values are the corresponding data values.
                         The dictionary should match the column structure of the table specified.

        Example:
            table = 'users'
            data = {'name': 'Alice', 'age': 30}
            db.add_to_batch(table, data)
        """
        with self.lock:
            if self.batch_table is not None and self.batch_table != table:
                self.flush_batch()  # Flush existing batch if table name changes
            self.batch_table = table
            self.batch_data.append(data)
            if len(self.batch_data) >= self.batch_size:
                self.flush_batch()

    def flush_batch(self):
        """
        Inserts all data currently in the batch into the database table specified by batch_table.
        This method uses the 'executemany' approach for efficient bulk inserts.
        If an error occurs during insertion, all changes are rolled back, and the data is dumped to a CSV file.

        Raises:
            sqlite3.Error: If an error occurs during the SQL execution, indicating problems with database insertion.
        """
        if not self.batch_data:
            return
        with self.lock:
            try:
                if not self.conn:
                    self.create_connection()
                cur = self.conn.cursor()
                columns = ', '.join(self.batch_data[0].keys())
                placeholders = ', '.join('?' for _ in self.batch_data[0])
                query = f"INSERT INTO {self.batch_table} ({columns}) VALUES ({placeholders})"
                cur.executemany(query, [tuple(d.values()) for d in self.batch_data])
                self.conn.commit()
                self.batch_data = []  # Clear the batch after successful insertion
                print(f"Batch data inserted into {self.batch_table}.")
            except sqlite3.Error as e:
                self.conn.rollback()
                print(f"Error inserting batch data: {e}")

                # Dump each row of the batch data to CSV individually
                for data in self.batch_data:
                    self.dump_to_csv(list(data.values()))  # Ensure data is in list format for writerow

                # Clear the batch data after dumping
                self.batch_data = []
            finally:

                # Close the connection
                self.close_connection()

    def get_table_names(self):
        """
        Fetches the names of all tables in the SQLite database.
        Returns:
            list: A list of table names, or an empty list if no tables are found or an error occurs.
        """
        try:
            # Execute the query to fetch the names of all tables
            query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            table_names = self.fetch_all(query)  # Assuming fetch_all returns a list of tuples

            # Extract table names from tuples
            table_names = [name[0] for name in table_names]

            return table_names
        except sqlite3.Error as e:
            print(f"Error accessing SQLite database: {e}")
            return []

    def get_column_names(self, table_name, exclude_autoincrement_pks: bool = True):
        """Fetches column names for a given table excluding autoincrement primary key."""
        columns = []
        print(self.fetch_all(f"PRAGMA table_info({table_name})"))
        for column in self.fetch_all(f"PRAGMA table_info({table_name})"):
            # Column format: (cid, name, type, notnull, dflt_value, pk)
            # We exclude columns that are primary key and autoincrement
            if exclude_autoincrement_pks:
                if not (column[5] == 1 and 'INTEGER' in column[2]):
                    columns.append(column[1])
            else:
                columns.append(column[1])
        return columns

    def gather_dump_data(self, table_name=None, delete_dumps=False):
        """
        Retrieve data from CSV files in the "dumps" folder and insert it into the SQLite database.
        """
        try:
            if not table_name:
                query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1"
                table_name = self.fetch_one(query)[0]
                if not table_name:
                    raise RuntimeError(
                        f"Table name not specified and cannot be extracted from the database. Table name: {table_name}")
        except Exception as e:
            logging.error(f"Error when accessing database table: {e}")

        column_names = self.get_column_names(table_name)
        columns_str = ', '.join(column_names)
        placeholders = ', '.join('?' for _ in column_names)

        dumps_folder = "dumps"
        try:
            # Check if the dumps folder exists
            if not Validator.is_valid_directory_path(dumps_folder):
                print(f"No dumps folder found in the current directory.")
                return

            # Get a list of CSV files in the dumps folder
            csv_files = Manipulator.list_files(dumps_folder, extensions=('.csv',), return_full_path=True)
            if not csv_files:
                print("No CSV files found in dumps folder.")
                return

            # Iterate over each CSV file
            for csv_file in csv_files:
                if f"_{self._db_name}_" in csv_file:
                    try:
                        # Read data from the CSV file
                        with open(csv_file, 'r', newline='') as file:
                            reader = csv.reader(file)
                            # next(reader, None)  # Skip the header if present
                            for row in reader:
                                query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                                # Using safe_execute to handle the SQL execution
                                if not self.safe_execute(query, tuple(row), use_transaction=True):
                                    raise Exception("Failed to insert all rows from the CSV.")
                        # Delete the CSV file after inserting its data into the database
                        if delete_dumps:
                            Manipulator.delete_file(csv_file)
                            print(f"Data from {csv_file} inserted into the database and file removed.")
                        else:
                            print(f"Data from {csv_file} inserted into the database.")
                    except Exception as e:
                        print(f"Error processing CSV file {csv_file}: {e}")
        except Exception as e:
            print(f"Error accessing dumps folder: {e}")

    def __del__(self):
        # Cleanup code here
        self.flush_batch()
