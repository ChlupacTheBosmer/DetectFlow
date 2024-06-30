from detectflow.process.database_manager import start_db_manager
from detectflow.process.orchestrator import Orchestrator
from detectflow.config import S3_CONFIG, DETECTFLOW_DIR
from detectflow.callbacks.orchestrator_process_video import process_video_callback
from detectflow.manipulators.dataloader import Dataloader
import os

if __name__ == '__main__':

    # Define input data
    db_paths = {
        'GR2_L1_TolUmb02': r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test\GR2_L1_TolUmb02.db",
        'GR2_L1_TolUmb03': r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test\GR2_L1_TolUmb03.db"
    }

    config_path = None # r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\orchestrator.json"
    config_format = "json"
    config_defaults = None
    CONFIG = {
        # Orchestrator keys
        "input_data": ["s3://gr2-l1/GR2_L1_TolUmb02/GR2_L1_TolUmb02_20220523_08_25.mp4",
                       "s3://gr2-l1/GR2_L1_TolUmb03/GR2_L1_TolUmb03_20220524_10_10.mp4"],
        "checkpoint_dir": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "task_name": "detectflow_realtest",
        "batch_size": 2,
        "max_workers": 1,
        "force_restart": True,
        "scratch_path": r"D:\Dílna\Kutění\Python\DetectFlow\tests\process\test",
        "user_name": "USER",
        "process_task_callback": process_video_callback
    }

    # Init a database manager instance and process
    db_man_info = start_db_manager(db_paths=db_paths,
                                   batch_size=100,
                                   backup_interval=500,
                                   s3_manipulator=None,
                                   database_structure=None)
    # init dataloader
    dataloader = Dataloader(S3_CONFIG)

    # Define callback config
    CALLBACK_CONFIG = {
        # Callback keys
        "db_manager_control_queue": db_man_info.get("control_queue", None),
        "db_manager_data_queue": db_man_info.get("data_queue", None),
        "frame_batch_size": 100,
        "frame_skip": 50,
        "max_producers": 1,
        "max_consumers": 1,
        "model_config": {0: {'path': os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt'), 'conf': 0.3},
                         1: {'path': os.path.join(DETECTFLOW_DIR, 'models', 'visitors.pt'), 'conf': 0.1}},
        "device": 'cpu',
        "track_results": True,
        "tracker_type": "botsort.yaml",
        "inspect": True
    }

    # Init orchestrator
    orchestrator = Orchestrator(config_path=config_path,
                                config_format=config_format,
                                config_defaults=config_defaults,
                                dataloader=dataloader,
                                parallelism="thread",
                                **CONFIG,
                                **CALLBACK_CONFIG)

    # Run orchestrator
    orchestrator.run()