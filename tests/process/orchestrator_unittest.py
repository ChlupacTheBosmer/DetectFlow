import unittest
from detectflow.process.orchestrator import Task, Orchestrator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.callbacks.orchestrator_process_video import diagnose_video_callback
from queue import Queue
import os

class TestTask(unittest.TestCase):

    def setUp(self):
        self.directory = "/test/directory"
        self.video_files = ["/test/directory/video1.mp4", "/test/directory/video2.avi"]
        self.status = {
            "/test/directory/video1.mp4": 0,
            "/test/directory/video2.avi": -1
        }
        self.task = Task(self.directory, self.video_files, self.status)

    def test_get_status(self):
        self.assertEqual(self.task.get_status("/test/directory/video1.mp4"), 0)
        self.assertEqual(self.task.get_status("/test/directory/video2.avi"), -1)
        self.assertEqual(self.task.get_status("/test/directory/video3.mkv"), 0)  # Default status for non-existent file

    def test_files_property(self):
        self.assertEqual(self.task.files, self.video_files)

    def test_statuses_property(self):
        self.assertEqual(self.task.statuses, [0, -1])

    def test_data_property(self):
        expected_data = {
            'directory': self.directory,
            'video_files': self.video_files,
            'status': self.status
        }
        self.assertEqual(self.task.data, expected_data)

    def test_repr(self):
        expected_repr = f"Task(directory={self.directory}, video_files={self.video_files}, status={self.status})"
        self.assertEqual(repr(self.task), expected_repr)


class TestOrchestrator(unittest.TestCase):

    CONFIG = {
        # Orchestrator keys
        "input_data": ["s3://gr2-l1/GR2_L1_TolUmb02/GR2_L1_TolUmb02_20220523_08_25.mp4",
                       "s3://gr2-l1/GR2_L1_TolUmb02/GR2_L1_TolUmb02_20220523_08_55.mp4"],
        "checkpoint_dir": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "task_name": "test_task",
        "batch_size": 2,
        "max_workers": 2,
        "force_restart": True,
        "scratch_path": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "user_name": "USER",
        "process_task_callback": diagnose_video_callback
    }

    CONFIG_dir = {
        # Orchestrator keys
        "input_data": ["s3://gr2-l1/GR2_L1_TolUmb02/",
                       "s3://gr2-l1/GR2_L1_TolUmb02/"],
        "checkpoint_dir": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "task_name": "test_task_dir",
        "batch_size": 1,
        "max_workers": 2,
        "force_restart": True,
        "scratch_path": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "user_name": "USER",
        "process_task_callback": diagnose_video_callback
    }

    CALLBACK_CONFIG = {
        # Callback keys
        }

    def setUp(self):
        self.config_path = None
        self.config_format = "json"
        self.config_defaults = None

        self.dataloader = Dataloader()

        self.orchestrator = Orchestrator(
            config_path=self.config_path,
            config_format=self.config_format,
            config_defaults=self.config_defaults,
            dataloader=self.dataloader,
            parallelism="process",
            **self.CONFIG_dir,
            **self.CALLBACK_CONFIG
        )

    def test_input_data(self):
        from detectflow.manipulators.s3_manipulator import S3Manipulator
        input_data = self.CONFIG_dir['input_data']

        man = S3Manipulator()

        print(man.is_s3_directory(input_data[0]))
        print(man.is_s3_file(input_data[0]))

        bucket_name, prefix = man.parse_s3_path(input_data[0])
        print(bucket_name, prefix)

        print(man.list_directories_s3(bucket_name, prefix, full_path=True))

        print(man.list_files_s3(bucket_name, prefix, regex=r"^(?!.*^\.).*(?<=\.mp4|\.avi|\.mkv)$",
                       return_full_path=True))

    def test_initialize(self):
        self.assertIsInstance(self.orchestrator.dataloader, Dataloader)
        self.assertEqual(self.orchestrator.checkpoint_dir, self.CONFIG['checkpoint_dir'])
        self.assertEqual(self.orchestrator.task_name, self.CONFIG['task_name'])
        self.assertEqual(self.orchestrator.batch_size, self.CONFIG['batch_size'])
        self.assertEqual(self.orchestrator.max_workers, self.CONFIG['max_workers'])
        self.assertEqual(self.orchestrator.force_restart, self.CONFIG['force_restart'])
        self.assertEqual(self.orchestrator.scratch_path, self.CONFIG['scratch_path'])
        self.assertEqual(self.orchestrator.user_name, self.CONFIG['user_name'])
        self.assertEqual(self.orchestrator.process_task_callback, self.CONFIG['process_task_callback'])
        self.assertIsInstance(self.orchestrator.task_queue, Queue)
        self.assertEqual(self.orchestrator.callback_config, self.CALLBACK_CONFIG)
        self.assertEqual(self.orchestrator.checkpoint_file, os.path.join(self.CONFIG['checkpoint_dir'], f"{self.CONFIG['task_name']}.json"))

    def test_save_config(self):
        self.orchestrator.config_path = os.path.join(self.CONFIG['checkpoint_dir'], "config.json")
        self.orchestrator.save_config()

    def test_load_config(self):
        self.orchestrator.config_path = os.path.join(self.CONFIG['checkpoint_dir'], "config_load.json")
        self.orchestrator.load_config()
        self.assertEqual(self.orchestrator.user_name, 'anyzd')
        self.assertEqual(self.orchestrator.task_name, f"{self.CONFIG['task_name']}_loaded_config")

    def test_run(self):
        self.orchestrator.run()
