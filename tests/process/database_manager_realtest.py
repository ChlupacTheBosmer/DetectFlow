import multiprocessing
import time

from detectflow.process.database_manager import DatabaseManager, start_db_manager
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.manipulators.s3_manipulator import S3Manipulator
import os
import unittest
import tempfile
import shutil
from multiprocessing import Manager
import psutil


def list_running_processes(name):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'username', 'status']):
        try:
            # Get process details as a dictionary
            process_info = proc.info
            if name in process_info['name']:
                print(
                    f"PID: {process_info['pid']}, Name: {process_info['name']}, User: {process_info['username']}, Status: {process_info['status']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Skip processes that no longer exist or cannot be accessed


class TestDatabaseManager():
    def __init__(self):
        from detectflow.config import DETECTFLOW_DIR

        self.test_dir = tempfile.mkdtemp()

        self.path = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "process", "db")
        self.rec_id_base = 'GR2_L1_TolUmb'
        self.timestamp = '20220524_07_44'
        db_paths = {}
        for i in range(2, 3):
            db_name = f'{self.rec_id_base}{i}'
            db_path = os.path.join(self.path, f'{db_name}.db')
            db_paths[db_name] = db_path

        self.s3_manipulator = S3Manipulator()

        manager_dict = start_db_manager(db_paths=db_paths, batch_size=100, backup_interval=500, dataloader=None,
                                        database_structure=None)

        self.db_manager = manager_dict['manager']
        self.db_manager_process = manager_dict['process']
        self.control_queue = manager_dict['control_queue']
        self.queue = manager_dict['data_queue']

    def test_start_db_manager(self):
        assert self.db_manager_process.is_alive()

        # Try looking for the process via name - may not work due to the test environment
        list_running_processes("DatabaseManager")

        print("Test start db manager passed")

    def test_add_database(self):
        from detectflow.process.database_manager import add_database_to_db_manager

        db_file = os.path.join(self.path, f'{self.rec_id_base}4.db')
        add_database_to_db_manager(self.control_queue, f'{self.rec_id_base}4', db_file)

        print("Test add database passed")

    def test_adding_data(self, n=10, db_index=None):
        from detectflow.utils.sampler import Sampler
        from detectflow.utils.extract_data import extract_data_from_result
        import random

        for i in range(n):
            res = Sampler.create_sample_detection_result()
            boxes = Sampler.create_sample_bboxes(as_detection_boxes=True)
            number_suffix = random.randint(2, 3) if db_index is None else db_index
            res.source_path = os.path.join(self.path, f'{self.rec_id_base}{number_suffix}_{self.timestamp}.mp4')
            res.reference_boxes = boxes

            data_entry = extract_data_from_result(res)
            #print(data_entry)

            self.queue.put(data_entry)

        print("Test adding data passed")

    def test_adding_video_data(self, db_index=None):
        from detectflow.utils.extract_data import extract_data_from_video
        import random

        for i in range(1):
            number_suffix = random.randint(2, 3) if db_index is None else db_index
            video_path = os.path.join(self.path, f'{self.rec_id_base}{number_suffix}_{self.timestamp}.mp4')
            data_entry = extract_data_from_video(video_path=video_path, frame_skip=20)

            self.queue.put(data_entry)

        print("Test adding video data passed")

    def test_flush_all_db_manager(self):
        from detectflow.process.database_manager import flush_all_db_manager

        flush_all_db_manager(self.control_queue)

        print("Test flush all db manager passed")

    def test_flush_one_db_manager(self):
        from detectflow.process.database_manager import flush_one_db_manager

        # Add some data
        self.test_adding_data(20, 2)

        # FLush one specific database
        flush_one_db_manager(self.control_queue, 'GR2_L1_TolUmb2')

        print("Test flush one db manager passed")

    def test_backup_db_to_s3(self):
        from detectflow.process.database_manager import backup_file_db_manager

        backup_file_db_manager(self.control_queue, 'GR2_L1_TolUmb2')

        time.sleep(15)

        files = self.s3_manipulator.list_files_s3('gr2-l1', 'GR2_L1_TolUmb02', return_full_path=False)

        print([f for f in files if f.endswith('.db')])

        self.s3_manipulator.delete_file_s3('gr2-l1', 'GR2_L1_TolUmb02/GR2_L1_TolUmb2.db')

        print("Test backup db to s3 passed")

    def test_stop_db_manager(self):
        from detectflow.process.database_manager import stop_db_manager

        stop_db_manager(self.control_queue, self.db_manager_process)

        print("Test stop db manager passed")

    def test_fetch_db(self):
        from detectflow.process.database_manager import fetch_file_db_manager

        db_file = os.path.join(self.path, f'{self.rec_id_base}2.db')
        fetch_file_db_manager(self.control_queue, 's3://gr2-l1/GR2_L1_TolUmb02/GR2_L1_TolUmb2.db', db_file)

        print("Test fetch db passed")

    # def tear_down(self):
    #     self.db_manager.clean_up()
    #     self.db_manager_process.terminate()
    #     self.db_manager_process.join()
    #     shutil.rmtree(self.test_dir)
    #     print("Tear down completed")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    t = TestDatabaseManager()
    t.test_start_db_manager()
    time.sleep(10)
    t.test_add_database()
    # t.test_adding_data()
    #t.test_adding_video_data()
    # t.test_flush_all_db_manager()
    # t.test_flush_one_db_manager()
    # t.test_backup_db_to_s3()
    #t.test_stop_db_manager()
    t.test_fetch_db()
    #t.tear_down()



