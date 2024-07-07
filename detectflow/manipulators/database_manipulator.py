import random
import multiprocessing
import sqlite3
from sqlite3 import Error
import re
import csv
import os
import threading
import multiprocessing
import traceback
import time
import logging
from typing import List, Optional, Tuple, Union
from detectflow.utils.hash import get_numeric_hash
from detectflow.manipulators.manipulator import Manipulator


class DatabaseManipulator:
    def __init__(self, db_file: str, batch_size: int = 50, lock_type: str = "threading", batch_update: bool = True):
        """
        Initialize the DatabaseManipulator object with the path to the SQLite database file.

        :param db_file: A string representing the path to the SQLite database file.
        :param batch_size: An integer specifying the number of rows to insert in a batch.
        :param lock_type: A string specifying the type of lock to use for concurrency control.
                            Options: 'threading' or 'multiprocessing'
        :param batch_update: A boolean specifying whether to update existing rows on primary key conflict during batching.
        """
        self.db_file = db_file
        self._db_name = os.path.splitext(os.path.basename(self.db_file))[0]
        self._conn = None
        self._lock_type = lock_type
        self._lock = None
        self.batch_data = []
        self.batch_table = None
        self.batch_size = batch_size
        self.batch_update_on_conflict = batch_update

    @property
    def lock(self):
        if self._lock is None:
            self._lock = threading.RLock() if self._lock_type == 'threading' else multiprocessing.RLock()  # Assumes using threading or multiprocesing for concurrency
        return self._lock

    @lock.setter
    def lock(self, value):
        self._lock = value

    @property
    def conn(self):
        if self._conn is None:
            self._conn = self.create_connection()
        return self._conn

    def create_connection(self):
        """
        Create a database connection to the SQLite database specified by the db_file.
        If the database file does not exist, it will be created.
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
                print(f"SQLite connection is opened to {self._db_name}")
                return conn
            except Exception as e:
                raise RuntimeError(f"Error connecting to database: {self._db_name} - {e}")

    def close_connection(self):
        """
        Close the database connection.
        """
        with self.lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                print(f"The SQLite connection is closed for {self._db_name}")

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
                cur = self.conn.cursor()
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                self.conn.commit()
                return cur
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Failed to execute SQL query: {self._db_name} - {e}")

    def safe_execute(self,
                     sql: str,
                     data: Optional[Union[Tuple, List]] = None,
                     retries: int = 3,
                     use_transaction: bool = True,
                     enable_emergency_dumps: bool = True,
                     sustain_emergency_dumps: bool = False):
        """
        Execute a SQL command with error handling, retry mechanism, and optional transaction control.

        :param sql: SQL command to be executed.
        :param data: Tuple of data to be used in the SQL command or a List of tuples to use executemany.
        :param retries: Number of retry attempts before failing.
        :param use_transaction: Whether to use transaction control (commit/rollback).
        :param enable_emergency_dumps: Whether to enable emergency dumps to CSV on final failure.
        :param sustain_emergency_dumps: Whether to sustain emergency dumps to CSV or raise an error.
        """
        try:
            for attempt in range(retries):
                try:
                    with self.lock:
                        cur = self.conn.cursor()
                        if use_transaction:
                            cur.execute("BEGIN;")
                        if data:
                            if isinstance(data, list) and all(isinstance(d, tuple) for d in data):
                                # data is a list of tuples, use executemany
                                cur.executemany(sql, data)
                            else:
                                # data is a single tuple, use execute
                                cur.execute(sql, data)
                        else:
                            cur.execute(sql)
                        self.conn.commit()
                    break  # Exit the loop if the query was successful
                except sqlite3.Error as e:
                    print(f"SQLite error on attempt {attempt + 1}: {e}")
                    print("Traceback:", traceback.format_exc())
                    if use_transaction:
                        self.conn.rollback()
                    if attempt == retries - 1:
                        if data and enable_emergency_dumps:
                            try:
                                table_name = extract_table_name(sql)
                            except Exception as e:
                                table_name = None
                            if isinstance(data, list) and all(isinstance(d, tuple) for d in data):
                                for d in data:
                                    self.dump_to_csv(table_name, d)  # Dump data to CSV on final failure
                            else:
                                self.dump_to_csv(table_name, data)  # Dump data to CSV on final failure
                            print(f"Data dumped to CSV file: {self._db_name}")
                            if not sustain_emergency_dumps:
                                raise RuntimeError(f"Failed to execute SQL query: {self._db_name} - {e}")
                        else:
                            raise RuntimeError(f"Failed to execute SQL query: {self._db_name} - {e}")
                    time.sleep(1)  # Wait before retrying
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {self._db_name} - {e}") from e
        finally:
            self.close_connection()


    def dump_to_csv(self, table_name, data):
        """ Dump data to a CSV file as a fallback
        :param table_name:
        """
        destination_folder = Manipulator.create_folders(directories="dumps", parent_dir=os.path.dirname(self.db_file))[0]
        filepath = os.path.join(destination_folder, f"db_{self._db_name}_t_{table_name}_id_{get_numeric_hash()}{random.randint(0,9)}.csv")
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

    def create_table(self, table_name: str, columns: list, table_constraints: str = ''):
        """
        Create a table using a list of column definitions and optional table constraints.

        :param table_name: Name of the table to create.
        :param columns: List of column definitions in the format (name, data_type, constraints).
                        Example: [('id', 'INTEGER', 'PRIMARY KEY'), ('name', 'TEXT', 'NOT NULL')]
        :param table_constraints: Optional string of table-level constraints.
                                  Example: 'FOREIGN KEY (column_name) REFERENCES other_table (column_name)'
        """
        try:
            # Construct the column definitions string
            columns_str = ', '.join([f"{col[0]} {col[1]} {col[2]}" for col in columns])

            # Include table constraints if provided
            if table_constraints:
                columns_str = f"{columns_str}, {table_constraints}"

            # Construct the CREATE TABLE SQL statement
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"

            # Execute the SQL statement
            self.safe_execute(query, use_transaction=True)
            print("Table created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create table in {self._db_name}: {e}")

    def insert(self, table, data, use_transaction=False, update_on_conflict=True):
        """
        Insert data into a table. If a conflict on the primary key occurs, optionally update the existing row.

        :param table: A string specifying the table to insert data into.
        :param data: A dictionary where the keys are column names and the values are data to be inserted.
                     Example data: {'name': 'Alice', 'age': 25}
        :param use_transaction: If True, a transaction will be used to ensure data integrity.
        :param update_on_conflict: If True, update the existing row on primary key conflict. If False, raise an error.
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join('?' for _ in data)

        unique_constraints = []
        try:
            unique_constraints = self.get_unique_constraints(table)
            primary_key_columns = self.get_primary_key_columns(table)

            if primary_key_columns:
                unique_constraints.append(primary_key_columns)
        except Exception as e:
            logging.error(f"Error fetching unique constraints for {table}: {e}")

        if update_on_conflict and len(unique_constraints) > 0:
            update_placeholders = ', '.join(f"{col} = EXCLUDED.{col}" for col in data.keys())

            # Create ON CONFLICT clause for all constraints
            on_conflict_clauses = [
                f"ON CONFLICT({', '.join(constraint)}) DO UPDATE SET {update_placeholders}"
                for constraint in unique_constraints
            ]
            on_conflict_query = ' '.join(on_conflict_clauses)

            query = f"""
                        INSERT INTO {table} ({columns}) VALUES ({placeholders})
                        {on_conflict_query};
                    """
            query_data = tuple(data.values())
        else:
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
            query_data = tuple(data.values())

        try:
            self.safe_execute(query, query_data, use_transaction=use_transaction)
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise RuntimeError(f"Conflict detected when inserting data into {table}: {e}")
            else:
                raise

    def update(self, table, data, condition, use_transaction=False):
        """
        Update data in a table based on a condition.

        :param table: A string specifying the table to update.
        :param data: A dictionary where keys are column names to be updated, and values are the new data values.
        :param condition: A string specifying the SQL condition for updating records.
        Example: 'id = 1'
        """
        updates = ', '.join(f"{k} = ?" for k in data.keys())
        query = f"UPDATE {table} SET {updates} WHERE {condition}"
        self.safe_execute(query, tuple(data.values()), use_transaction=use_transaction)

    def delete(self, table, condition, use_transaction=False):
        """
        Delete data from a table based on a condition.

        :param table: A string specifying the table to delete from.
        :param condition: A string specifying the SQL condition for deleting records.
        :param use_transaction: If True, a transaction will be used to ensure data integrity.

        Example: 'id = 1'
        """
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
                try:
                    self.flush_batch()  # Flush existing batch if table name changes
                except Exception as e:
                    print(f"Failed to insert batch data into {self._db_name} - {self.batch_table}: {e}")
            self.batch_table = table
            self.batch_data.append(data)
            if len(self.batch_data) >= self.batch_size:
                try:
                    self.flush_batch()
                except Exception as e:
                    print(f"Failed to insert batch data into {self._db_name} - {table}: {e}")

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

        try:
            columns = ', '.join(self.batch_data[0].keys())
            placeholders = ', '.join('?' for _ in self.batch_data[0])
        except IndexError:
            logging.warning("No data found in the batch.")
            return
        except TypeError:
            logging.error("Invalid data format in the batch.") # TODO: Dump the data in the batch so we do not lose it?
            return

        unique_constraints = []
        try:
            unique_constraints = self.get_unique_constraints(self.batch_table)
            primary_key_columns = self.get_primary_key_columns(self.batch_table)

            if primary_key_columns:
                unique_constraints.append(primary_key_columns)
        except Exception as e:
            logging.error(f"Error fetching unique constraints for {self.batch_table}: {e}")

        try:
            if self.batch_update_on_conflict and len(unique_constraints) > 0:

                update_placeholders = ', '.join(f"{col} = EXCLUDED.{col}" for col in self.batch_data[0].keys())

                # Create ON CONFLICT clause for all constraints
                on_conflict_clauses = [
                    f"ON CONFLICT({', '.join(constraint)}) DO UPDATE SET {update_placeholders}"
                    for constraint in unique_constraints
                ]
                on_conflict_query = ' '.join(on_conflict_clauses)

                query = f"""
                            INSERT INTO {self.batch_table} ({columns}) VALUES ({placeholders})
                            {on_conflict_query};
                            """
                data = [tuple(d.values()) for d in self.batch_data]
            else:
                query = f"INSERT INTO {self.batch_table} ({columns}) VALUES ({placeholders});"
                data = [tuple(d.values()) for d in self.batch_data]

            self.safe_execute(query, data, use_transaction=True, enable_emergency_dumps=True,
                              sustain_emergency_dumps=True)
            self.batch_data = []  # Clear the batch after successful insertion
            print(f"Batch data inserted into {self._db_name} - {self.batch_table}.")
        except sqlite3.Error as e:
            self.conn.rollback()
            if not self.batch_update_on_conflict and "UNIQUE constraint failed" in str(e):
                raise RuntimeError(f"Conflict detected when inserting batch data into {self.batch_table}: {e}")
            else:
                raise RuntimeError(f"Error inserting batch data: {self._db_name} - {e}")

    def get_unique_constraints(self, table_name):
        """
        Retrieves the columns involved in unique constraints for the specified table.

        :param table_name: The name of the table.
        """
        cursor = self.conn.execute(f"PRAGMA index_list('{table_name}')")
        unique_constraints = []
        for row in cursor.fetchall():
            if row[2]:  # row[2] is True if the index is unique
                index_name = row[1]
                index_info = self.conn.execute(f"PRAGMA index_info('{index_name}')").fetchall()
                unique_columns = [info[2] for info in index_info]  # info[2] is the column name
                unique_constraints.append(unique_columns)
        return unique_constraints

    def get_primary_key_columns(self, table_name):
        """
        Retrieves the primary key columns for the specified table.
        """
        cursor = self.conn.execute(f"PRAGMA table_info('{table_name}')")
        primary_key_columns = [row[1] for row in cursor.fetchall() if
                               row[5]]  # row[5] is True if the column is part of the primary key
        return primary_key_columns if primary_key_columns else None

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
        except Exception as e:
            print(f"Error accessing SQLite database: {self._db_name} - {e}")
            return []

    def get_column_names(self, table_name, exclude_autoincrement: bool = True):
        """
            Retrieves the column names for the specified table, excluding autoincrement columns.

            :param table_name: Name of the table
            :param exclude_autoincrement: Whether to exclude autoincrement primary key columns
            :return: List of column names excluding autoincrement columns
            """
        import re

        cursor = self.conn.cursor()

        # Get table creation SQL
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        table_sql = cursor.fetchone()[0]

        # Find autoincrement columns using regex
        autoincrement_columns = re.findall(r'(\w+)\s+INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT', table_sql, re.IGNORECASE)

        # Get all column names from PRAGMA table_info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()

        # Exclude autoincrement columns
        if exclude_autoincrement:
            column_names = [col[1] for col in columns_info if col[1] not in autoincrement_columns]
        else:
            column_names = [col[1] for col in columns_info]

        return column_names

    def gather_dump_data(self, table_name: Optional[str] = None, dumps_folder: str = "dumps", delete_dumps: bool = False, update_on_conflict: bool = True):
        """
        Retrieve data from CSV files in the "dumps" folder and insert it into the SQLite database.
        """

        def parse_dump_name(filepath):
            """
            Extracts the table name from a file path based on the new naming convention.

            :param filepath: The file path string
            :return: Extracted table name if the pattern matches, else None
            """
            pattern = r't_([^_]+)_id'  # Regex to find text between 't_' and '_id'
            filename = os.path.basename(filepath)
            match = re.search(pattern, filename)

            if match:
                return match.group(1)
            else:
                return None

        try:
            if not table_name:
                default_table_name = self.get_table_names()[0]
                if not default_table_name:
                    raise RuntimeError(
                        f"Table name not specified and cannot be extracted from the database. Table name: {default_table_name}")
            else:
                default_table_name = table_name
        except Exception as e:
            logging.error(f"Error when accessing database table: {self._db_name} - {e}")
            default_table_name = table_name

        try:
            # Check if the dumps folder exists
            if not Manipulator.is_valid_directory_path(dumps_folder):
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
                        # Extract the table name from the CSV file name
                        table_name = parse_dump_name(csv_file)
                        table_name = table_name if table_name and table_name != 'None' else default_table_name

                        # Construct placeholders for the SQL query
                        column_names = self.get_column_names(table_name)

                        # Read data from the CSV file
                        with open(csv_file, 'r', newline='') as file:
                            reader = csv.reader(file)
                            # next(reader, None)  # Skip the header if present
                            for row in reader:

                                # Construct a dictionary of column names and data values
                                data = {col: cell for col, cell in zip(column_names, row)}

                                # Insert the data into the database and update on conflict
                                self.insert(table_name, data, use_transaction=True, update_on_conflict=update_on_conflict)

                        # Delete the CSV file after inserting its data into the database
                        if delete_dumps:
                            Manipulator.delete_file(csv_file)
                            print(f"Data from {csv_file} inserted into the database and file removed.")
                        else:
                            print(f"Data from {csv_file} inserted into the database.")
                    except Exception as e:
                        print(f"Error processing CSV file {csv_file}: {e}")
        except Exception as e:
            print(f"Error accessing dumps folder: {self._db_name} - {e}")

    def __del__(self):
        # Cleanup code here
        try:
            self.flush_batch()
        except Exception as e:
            print(f"Failed to insert batch data into {self._db_name} - {self.batch_table}: {e}")


def merge_databases(db1_path: str, db2_path: str, output_db_path: str):
    # Initialize DatabaseManipulator instances for both databases
    db1 = DatabaseManipulator(db1_path)
    db2 = DatabaseManipulator(db2_path)
    output_db = DatabaseManipulator(output_db_path)

    try:
        # Open connections
        db1.create_connection()
        db2.create_connection()
        output_db.create_connection()

        # Get all table names from the first database
        table_names = db1.get_table_names()

        for table in table_names:
            print(f"Merging table: {table}")
            db1_columns = db1.get_column_names(table, exclude_autoincrement=False)
            db2_columns = db2.get_column_names(table, exclude_autoincrement=False)

            # Ensure that both tables have the same columns
            if set(db1_columns) != set(db2_columns):
                raise RuntimeError(f"Table columns mismatch in {table}")

            # Fetch the table schema from the first database
            columns_info = db1.fetch_all(f"PRAGMA table_info({table})")
            primary_keys = [col[1] for col in columns_info if col[5]]
            columns_definitions = ', '.join(
                [f"{col[1]} {col[2]} PRIMARY KEY" if col[5] and len(primary_keys) == 1 else f"{col[1]} {col[2]}" for col
                 in columns_info]
            )

            if len(primary_keys) > 1:
                primary_key_str = f", PRIMARY KEY ({', '.join(primary_keys)})"
            else:
                primary_key_str = ""

            create_table_query = f"CREATE TABLE IF NOT EXISTS {table} ({columns_definitions}{primary_key_str});"
            output_db.safe_execute(create_table_query)

            # Fetch all data from the first database table
            db1_data = db1.fetch_all(f"SELECT * FROM {table}")
            for row in db1_data:
                data = dict(zip(db1_columns, row))
                placeholders = ', '.join('?' for _ in data)
                columns_str = ', '.join(data.keys())
                query = f"INSERT OR IGNORE INTO {table} ({columns_str}) VALUES ({placeholders})"
                output_db.safe_execute(query, tuple(data.values()))

            # Fetch all data from the second database table and insert or ignore conflicts
            db2_data = db2.fetch_all(f"SELECT * FROM {table}")
            for row in db2_data:
                data = dict(zip(db2_columns, row))
                placeholders = ', '.join('?' for _ in data)
                columns_str = ', '.join(data.keys())
                query = f"INSERT OR IGNORE INTO {table} ({columns_str}) VALUES ({placeholders})"
                output_db.safe_execute(query, tuple(data.values()))

        print("Databases merged successfully into the third database.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error merging databases: {e}")

    finally:
        # Close connections
        db1.close_connection()
        db2.close_connection()
        output_db.close_connection()

    return output_db_path


def extract_table_name(query):
    """
    Extract the table name from an SQLite query.

    :param query: SQL query string
    :return: Table name if found, else None
    """
    # Define regular expressions for different SQL commands
    patterns = [
        r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # SELECT ... FROM table_name
        r"JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # ... JOIN table_name ...
        r"INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # INSERT INTO table_name ...
        r"UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # UPDATE table_name SET ...
        r"TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)

    return None