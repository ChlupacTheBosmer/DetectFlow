from detectflow.process.orchestrator import Task
from detectflow.process.database_manager import add_database_to_db_manager
from detectflow.process.frame_generator import FrameGenerator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.validators.validator import Validator
from detectflow.utils.threads import calculate_optimal_threads
from detectflow.callbacks.frame_generator_predictions import frame_generator_predict
from detectflow.utils.extract_data import extract_data_from_video
import logging
from queue import Queue
import os
from typing import Any


def diagnose_video_callback(**kwargs):
    from detectflow.video.video_data import Video

    task = kwargs.get('task', None)
    scratch = kwargs.get('scratch_path', None)
    orchestrator_control_queue = kwargs.get('orchestrator_control_queue', None)
    dataloader = None

    if not task:
        return None

    try:
        dataloader = Dataloader()
        valid, invalid = dataloader.prepare_data(task.files, scratch, False)

        if not len(valid) == len(task.files) or len(invalid) > 0:
            raise ValueError(f"Some ({len(invalid)}) videos failed validation: {invalid}")

    except Exception as e:
        logging.error(f"{task.name} - Error when loading and validating data: {e}")
        valid, invalid = [], []

    video_path = None
    try:
        for video_path in valid:
            vid = Video(video_path)
            logging.info(vid.color_variance)

            # Determine what the new status of the video is
            update_info = {"directory": task.directory,
                           "file": next((f for f in task.files if os.path.basename(f) == os.path.basename(video_path)),
                                        None), "status": -1}
            if orchestrator_control_queue:
                orchestrator_control_queue.put(('update_task', (update_info,)))
    except Exception as e:
        logging.error(f"{task.name} - Error when processing video: {video_path} - {e}")

    try:
        for video_path in valid + invalid:
            if dataloader:
                dataloader.delete_file(video_path)
            else:
                os.remove(video_path)
    except Exception as e:
        logging.error(f"{task.name} - Error when deleting files: {e}")
    finally:
        del dataloader


def process_video_callback(task: Task,
                           name: str,
                           orchestrator_control_queue: Any,
                           **kwargs):
    CONFIG_MAP = {"scratch_path": str,
                  "db_manager_control_queue": object,
                  "db_manager_data_queue": object,
                  "frame_batch_size": int,
                  "frame_skip": int,
                  "max_producers": int,
                  "max_consumers": int,
                  "model_config": dict,
                  "device": object,
                  "track_results": bool,
                  "tracker_type": str,
                  "inspect": bool}

    # Unpack Task
    directory = task.directory
    files = task.files
    print(files)

    # Fix kwargs
    if kwargs:
        try:
            Validator.fix_kwargs(CONFIG_MAP, kwargs, False)
        except Exception as e:
            logging.error(f"{name} - Error during fixing kwargs: {e}")

    # Unpack kwargs
    scratch = kwargs.get('scratch_path', None)
    db_manager_control_queue = kwargs.get('db_manager_control_queue', None)
    db_manager_data_queue = kwargs.get('db_manager_data_queue', None)
    dataloader = kwargs.get('dataloader', None)

    # Frame Generator config
    first_frame_number = 0
    frame_batch_size = kwargs.get('frame_batch_size', 50)
    frame_skip = kwargs.get('frame_skip', 15)

    # Generator thread config
    max_producers = kwargs.get('max_producers', 1)
    max_consumers = kwargs.get('max_consumers', 1)

    # Model config
    model_config = kwargs.get('model_config', None)
    device = kwargs.get('device', None)
    track_results = kwargs.get('track_results', False)
    tracker_type = kwargs.get('tracker_type', None)

    # Functionality flags
    inspect = kwargs.get('inspect', False)

    # Load videos and validate them
    try:
        dataloader = Dataloader()
        valid, invalid = dataloader.prepare_data(files, scratch, False)

        if not len(valid) == len(files) or len(invalid) > 0:
            raise ValueError(f"Some ({len(invalid)}) videos failed validation: {invalid}")

        logging.info(f"{name} - Videos loaded and validated: {valid}")
    except Exception as e:
        logging.error(f"{name} - Error when loading and validating data: {e}")
        valid, invalid = [], []

    video_path = None
    try:
        for i, video_path in enumerate(valid):

            # Get video_id
            video_id, _ = os.path.splitext(os.path.basename(video_path))

            # Get the status of the video processing
            status = task.get_status(file_name=os.path.basename(files[i]))
            first_frame_number = status if status and status != -1 else 0
            update_info = {"directory": directory, "file": files[i], "status": status}
            if status == -1:
                continue

            # Initialize its database
            add_database_to_db_manager(db_manager_control_queue, video_id, os.path.join(scratch, f"{video_id}.db"))

            # Add video details data entry to database manager queue
            if db_manager_data_queue is not None:
                video_data_entry = extract_data_from_video(video_path)
                db_manager_data_queue.put(video_data_entry)
            else:
                raise TypeError("Database task queue not defined")

            callback_config = {
                'db_manager_data_queue': db_manager_data_queue,
                'update_info': update_info,
                'orchestrator_control_queue': orchestrator_control_queue,
                'inspect': inspect,
                'model_config': model_config,
                'device': device,
                'track_results': track_results,
                'tracker_type': tracker_type,
                'scratch_path': scratch
            }

            # Generate frames and run detection
            generator = FrameGenerator(source=[video_path],
                                       output_folder=scratch,
                                       first_frame_number=first_frame_number,
                                       frame_skip=frame_skip,
                                       processing_callback=frame_generator_predict,
                                       **callback_config)

            generator.run(producers=max_producers, consumers=max_consumers, frame_batch_size=frame_batch_size)
    except Exception as e:
        logging.error(f"{name} - Error when processing video: {video_path} - {e}")

    # After processing, delete the files
    try:
        for video_path in valid + invalid:
            if dataloader:
                dataloader.delete_file(video_path)
            else:
                os.remove(video_path)
    except Exception as e:
        logging.error(f"{task.name if hasattr(task, 'name') else 'Unknown task'} - Error when deleting files: {e}")
    finally:
        del dataloader

