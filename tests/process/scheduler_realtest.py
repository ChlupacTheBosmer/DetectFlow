import os
import json
import time
import unittest
from pathlib import Path
from detectflow.process.scheduler import Scheduler
from detectflow.config import ROOT

class TestScheduler(unittest.TestCase):

    def setUp(self):
        # Setup the environment variables and paths for testing
        cd = os.path.dirname(ROOT)
        self.remote_work_dir = "/storage/brno2/home/USER/scheduler_test"
        self.local_work_dir = r'D:\Dílna\Kutění\Python\DetectFlow\tests\process\test'
        self.database_path = r'D:\Dílna\Kutění\Python\DetectFlow\tests\process\test\jobs.db'
        self.bucket_name = "cz1-m1"
        self.remote_host = "perian.metacentrum.cz"
        self.username = "USER"
        self.ssh_password = 'PASSWORD'

        # Create output directory if it does not exist
        Path(self.local_work_dir).mkdir(parents=True, exist_ok=True)

        # Initialize the Scheduler instance
        self.scheduler = Scheduler(local_work_dir=self.local_work_dir,
                                   remote_work_dir=self.remote_work_dir,
                                   database_path=self.database_path)

    # def tearDown(self):
    #     # Cleanup the test environment
    #     if os.path.exists(self.output_path):
    #         for root, dirs, files in os.walk(self.output_path, topdown=False):
    #             for name in files:
    #                 os.remove(os.path.join(root, name))
    #             for name in dirs:
    #                 os.rmdir(os.path.join(root, name))
    #         os.rmdir(self.output_path)

    def test_submit_jobs(self):
        resources = {
            "walltime": 1,  # in hours
            "mem": 4,  # in GB
            "cpus": 1,
            "scratch": 1  # in GB
        }

        # Call the submit_jobs method with real parameters
        self.scheduler.submit_jobs(bucket_name=self.bucket_name,
                                   directory_filter="CZ1_M1_AraHir01/",
                                   job_config_path=r'D:\Dílna\Kutění\Python\DetectFlow\tests\process\test\test_task_dir.json',
                                   python_script_path='/storage/brno2/home/USER/scheduler_test/test.py',
                                   use_gpu=False,
                                   resources=resources,
                                   username=self.username,
                                   remote_host=self.remote_host,
                                   ssh_password=self.ssh_password,
                                   ignore_ssh_auth=False)

        # Validate the database entries
        query = "SELECT * FROM jobs"
        jobs = self.scheduler.database_manipulator.fetch_all(query)
        self.assertGreater(len(jobs), 0, "No jobs were logged in the database.")

    def test_monitor_jobs(self):
        user_email = "EMAIL"

        # Start monitoring jobs
        self.scheduler.monitor_jobs(user_email=user_email,
                                    username=self.username,
                                    remote_host=self.remote_host,
                                    ssh_password=self.ssh_password,
                                    ignore_ssh_auth=False)


if __name__ == "__main__":
    unittest.main()
