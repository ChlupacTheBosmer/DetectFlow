import os
import unittest
import tempfile
import shutil
import csv
from unittest.mock import patch, mock_open
from detectflow.manipulators.database_manipulator import DatabaseManipulator

class TestDatabaseManipulator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_file = os.path.join(self.test_dir, 'test.db')
        self.manipulator = DatabaseManipulator(self.db_file)

    def tearDown(self):
        self.manipulator.close_connection()
        shutil.rmtree(self.test_dir)

    def test_create_connection(self):
        self.manipulator.create_connection()
        self.assertIsNotNone(self.manipulator.conn)

    def test_close_connection(self):
        self.manipulator.create_connection()
        self.manipulator.close_connection()
        self.assertIsNone(self.manipulator.conn)

    def test_execute_query(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.execute_query("INSERT INTO test (name) VALUES (?)", ("John",))
        result = self.manipulator.fetch_one("SELECT * FROM test")
        print(result)
        self.assertEqual(result[1], "John")

    def test_safe_execute(self):
        self.manipulator.create_connection()
        self.manipulator.safe_execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.safe_execute("INSERT INTO test (name) VALUES (?)", ("John",))
        result = self.manipulator.fetch_one("SELECT * FROM test")
        self.assertEqual(result[1], "John")

    @patch('detectflow.manipulators.database_manipulator.open', new_callable=mock_open, read_data='data')
    def test_dump_to_csv(self, mock_file):
        self.manipulator.dump_to_csv(("test_data",))
        mock_file.assert_called_once()

    def test_fetch_all(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.execute_query("INSERT INTO test (name) VALUES (?)", ("John",))
        self.manipulator.execute_query("INSERT INTO test (name) VALUES (?)", ("Jane",))
        result = self.manipulator.fetch_all("SELECT * FROM test")
        self.assertEqual(len(result), 2)

    def test_fetch_one(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.execute_query("INSERT INTO test (name) VALUES (?)", ("John",))
        result = self.manipulator.fetch_one("SELECT * FROM test")
        self.assertEqual(result[1], "John")

    def test_create_table(self):
        columns = [("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
                   ("name", "TEXT", "NOT NULL")]
        self.manipulator.create_table("test", columns)
        table_names = self.manipulator.get_table_names()
        self.assertIn("test", table_names)

    def test_insert(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.insert("test", {"name": "John"})
        result = self.manipulator.fetch_one("SELECT * FROM test")
        print(result)
        self.assertEqual(result[1], "John")

    def test_update(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.insert("test", {"name": "John"})
        self.manipulator.update("test", {"name": "Jane"}, "id = 1")
        result = self.manipulator.fetch_one("SELECT * FROM test")
        self.assertEqual(result[1], "Jane")

    def test_delete(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.insert("test", {"name": "John"})
        self.manipulator.delete("test", "id = 1")
        result = self.manipulator.fetch_all("SELECT * FROM test")
        self.assertEqual(len(result), 0)

    def test_add_to_batch(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.add_to_batch("test", {"name": "John"})
        self.manipulator.add_to_batch("test", {"name": "Jane"})
        self.manipulator.flush_batch()
        result = self.manipulator.fetch_all("SELECT * FROM test")
        self.assertEqual(len(result), 2)

    def test_flush_batch(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.add_to_batch("test", {"name": "John"})
        self.manipulator.add_to_batch("test", {"name": "Jane"})
        self.manipulator.flush_batch()
        result = self.manipulator.fetch_all("SELECT * FROM test")
        self.assertEqual(len(result), 2)

    def test_get_table_names(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test1 (id INTEGER PRIMARY KEY, name TEXT)")
        self.manipulator.execute_query("CREATE TABLE test2 (id INTEGER PRIMARY KEY, name TEXT)")
        table_names = self.manipulator.get_table_names()
        self.assertIn("test1", table_names)
        self.assertIn("test2", table_names)

    def test_get_column_names(self):
        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        column_names = self.manipulator.get_column_names("test")
        self.assertEqual(column_names, ["name", "age"])

    @patch('detectflow.manipulators.database_manipulator.Manipulator.list_files')
    @patch('detectflow.manipulators.database_manipulator.Manipulator.is_valid_directory_path')
    def test_gather_dump_data(self, mock_is_valid_directory_path, mock_list_files):
        mock_is_valid_directory_path.return_value = True
        mock_list_files.return_value = [os.path.join(self.test_dir, f"emergency_dump_{self.manipulator._db_name}_123.csv")]

        with open(mock_list_files.return_value[0], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["John"])
            writer.writerow(["Jane"])

        self.manipulator.create_connection()
        self.manipulator.execute_query("CREATE TABLE test (name TEXT)")
        self.manipulator.gather_dump_data("test")
        result = self.manipulator.fetch_all("SELECT * FROM test")
        self.assertEqual(len(result), 2)

if __name__ == '__main__':
    unittest.main()

