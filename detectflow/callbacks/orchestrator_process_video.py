from detectflow.process.orchestrator import Task
from detectflow.process.database_manager import add_database_to_db_manager
from detectflow.process.frame_generator import FrameGenerator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.validators.validator import Validator
from detectflow.models import DEFAULT_MODEL_CONFIG as model_defaults
from detectflow.callbacks.frame_generator_predictions import frame_generator_predict
from detectflow.utils.extract_data import extract_data_from_video
from detectflow.utils.name import parse_recording_name
import logging
from datetime import datetime
import os
from typing import Any, Type, Union, Optional, List


def diagnose_video_callback(**kwargs):
    from detectflow.video.video_data import Video

    task = kwargs.get('task', None)
    name = kwargs.get('name', "Diagnose Video Callback")
    scratch = kwargs.get('scratch_path', None)
    orchestrator_control_queue = kwargs.get('orchestrator_control_queue', None)
    db_manager_control_queue = kwargs.get('db_manager_control_queue', None)
    db_manager_data_queue = kwargs.get('db_manager_data_queue', None)
    dataloader = None

    # Unpack Task
    if task:
        directory = task.directory
        files = task.files

    if not task:
        return None

    # Load videos and validate them
    try:
        dataloader = Dataloader()
        valid, invalid = dataloader.prepare_videos(files, scratch, True)

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
            print("Status: ", status)
            first_frame_number = status if status and status != -1 else 0
            update_info = {"directory": directory, "file": files[i], "status": status}
            if status == -1:
                continue

            # Initialize its database
            add_database_to_db_manager(db_manager_control_queue, video_id, os.path.join(scratch, f"{video_id}.db"))

            # Add video details data entry to database manager queue
            if db_manager_data_queue is not None:  # Assumes that the video has not been processed before
                if status == 0:
                    video_data_entry = extract_data_from_video(video_path, frame_skip=100, motion_methods='SOM')
                    db_manager_data_queue.put(video_data_entry)
            else:
                raise TypeError("Database task queue not defined")
    except Exception as e:
        logging.error(f"{name} - Error when processing video: {video_path} - {e}")

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

COL_CODES = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

def process_video_callback(task: Task,
                           name: str,
                           orchestrator_control_queue: Any,
                           **kwargs):
    CONFIG_MAP = {"scratch_path": str,
                  "db_manager_control_queue": object,
                  "db_manager_data_queue": object,
                  "resource_monitor_queue": object,
                  "frame_batch_size": int,
                  "frame_skip": int,
                  "max_producers": int,
                  "max_consumers": int,
                  "model_config": (dict, list),
                  "device": object,
                  "track_results": bool,
                  "tracker_type": (str, type(None)),
                  "inspect": bool}

    logging.info(f"Processing video task: {task}: kwargs: {kwargs}")

    if name:
        worker_number = extract_worker_number(name)
        color = COL_CODES[worker_number % len(COL_CODES)]
    else:
        name = "Unknown Worker"
        color = 'black'

    # Unpack Task
    if task:
        directory = task.directory
        files = task.files
        print(files)
    else:
        logging.error(f"{name} - Task not defined")
        return

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
    resource_monitor_queue = kwargs.get('resource_monitor_queue', None)
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
    skip_empty_videos = kwargs.get('skip_empty_videos', False)
    skip_empty_frames = kwargs.get('skip_empty_frames', False)

    # Debug message
    try:
        if resource_monitor_queue:
            event_message = f"{name} start"
            resource_monitor_queue.put((event_message, color))
    except Exception as e:
        pass

    # Initial validation and debug logging
    if scratch is None:
        raise ValueError(f"{name} - Scratch path not defined")

    if not model_config or not isinstance(model_config, (list, dict)):
        logging.warning(f"{name} - Model config not defined, using default")
        model_config = model_defaults

    # Load videos and validate them
    try:
        dataloader = dataloader or Dataloader()
        valid, invalid = dataloader.prepare_videos(files, scratch, False)

        if not len(valid) == len(files) or len(invalid) > 0:
            raise ValueError(f"Some ({len(invalid)}) videos failed validation: {invalid}")

        logging.info(f"{name} - Videos loaded and validated: {valid}")
    except Exception as e:
        logging.error(f"{name} - Error when loading and validating data: {e}")
        valid, invalid = [], []

    video_path = None
    for i, video_path in enumerate(valid):

        # Retrieve the s3 path
        s3_path = next((f for f in files if os.path.basename(f) == os.path.basename(video_path)), None)

        try:
            # Get video_id
            try:
                name_info = parse_recording_name(video_path)
                video_id = name_info.get("video_id", None)
                recording_id = name_info.get("recording_id", None)
            except Exception as e:
                logging.error(f"{name} - Error when parsing video name: {e}")
                video_id, _ = os.path.splitext(os.path.basename(video_path))
                recording_id = video_id

            # Get the status of the video processing
            status = task.get_status(file_name=os.path.basename(files[i]))
            first_frame_number = status if status and status >= 0 else 0
            update_info = {"directory": directory, "file": files[i], "status": status}
            if status < 0:
                continue

            # Initialize its database
            add_database_to_db_manager(db_manager_control_queue, recording_id, os.path.join(scratch, f"{recording_id}.db"))

            # Add video details data entry to database manager queue
            video_raw_data = None
            if db_manager_data_queue is not None:  # Assumes that the video has not been processed before
                if status == 0:
                    video_data_entry, video_raw_data = extract_data_from_video(video_path,
                                                                               s3_path=s3_path,
                                                                               frame_skip=100,
                                                                               motion_methods='SOM',
                                                                               return_raw_data=True)
                    db_manager_data_queue.put(video_data_entry)
            else:
                raise TypeError("Database task queue not defined")

            # Decide whether to process the video,
            checked_flowering_minutes = False
            try:
                if video_raw_data:
                    flowering_minutes_db = get_flowering_minutes_db(recording_id=recording_id,
                                                                    folder_path=scratch,
                                                                    bucket_name='data',
                                                                    prefix='flowering-minutes',
                                                                    dataloader=dataloader)
                    if flowering_minutes_db:
                        process, flowers_frame_number = should_process_video_from_db(db_path=flowering_minutes_db,
                                                                                     video_start=video_raw_data.get(
                                                                                         'start_time'),
                                                                                     video_end=video_raw_data.get(
                                                                                         'end_time'),
                                                                                     fps=video_raw_data.get('fps'))
                        checked_flowering_minutes = True
                        if not process:
                            logging.info(f"{name} - Video {video_id} should not be processed.")
                            continue
                        else:
                            first_frame_number = max(first_frame_number, flowers_frame_number if flowers_frame_number is not None else 0)
            except Exception as e:
                logging.error(f"{name} - Error when checking flowering minutes: {e}")

            try:
                if skip_empty_videos and not checked_flowering_minutes and not should_process_video_from_boxes(video_raw_data.get('reference_boxes')):
                    logging.info(f"{name} - No consistent flowers detected. Video {video_id} should not be processed.")
                    continue
            except Exception as e:
                logging.error(f"{name} - Error when checking reference boxes: {e}")

            # Debug message
            try:
                if resource_monitor_queue:
                    event_message = f"{name} FG start"
                    resource_monitor_queue.put((event_message, color))
            except Exception as e:
                pass

            callback_config = {
                'db_manager_data_queue': db_manager_data_queue,
                'resource_monitor_queue': resource_monitor_queue,
                'worker_name': name,
                'color': color,
                'update_info': update_info,
                'orchestrator_control_queue': orchestrator_control_queue,
                'inspect': inspect,
                'model_config': model_config,
                'device': device,
                'track_results': track_results,
                'tracker_type': tracker_type,
                'scratch_path': scratch,
                'skip_empty_frames': skip_empty_frames
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


def extract_worker_number(worker_name):
    import re

    # Extract the number using regex
    try:
        match = re.search(r'#(\d+)', worker_name)
        if match:
            number = int(match.group(1))
            return number
        else:
            raise ValueError("No number found in the worker string")
    except Exception as e:
        return 0


def get_flowering_minutes_db(recording_id: str, folder_path: str, bucket_name: str, prefix: str, dataloader: Type[Dataloader]):
    """
    Fetches the database with the flowering minutes data from s3.
    Assumes that the database is named according to the naming convention and is located in the correct
    bucket and directory.

    :param recording_id: Recording ID of the video to fetch the database for (as per the naming convention).
    :param folder_path: Destination folder to save the database file.
    :param bucket_name: Name of the s3 bucket to fetch the database from.
    :param prefix: Prefix of the database file in the s3 bucket.
    :return: path to the database file.
    """

    db_path = os.path.join(folder_path, f"{recording_id}_flowering_minutes.db")

    # Check if the database already exists
    if os.path.exists(db_path):
        return db_path

    # Attempt to locate the database in the s3 bucket
    try:
        online_file = dataloader.locate_file_s3(rf"({prefix}/{recording_id}_flowering_minutes.db|{recording_id}_flowering_minutes.db)", bucket_name, selection_criteria='name')
    except Exception as e:
        logging.error(f"Error when fetching the flowering minutes database from s3: {e}")
        return None

    if online_file is None:
        logging.error(f"Flowering minutes database not found in the s3 bucket.")
        return None

    # Download the database from s3
    try:
        db_path = dataloader.download_file_s3(*dataloader.parse_s3_path(online_file), local_file_name=db_path)
    except Exception as e:
        logging.error(f"Error when downloading the flowering minutes database from s3: {e}")
        return None

    return db_path


def should_process_video_from_db(db_path: str, video_start: datetime, video_end: datetime, fps: Union[int, float]):
    from detectflow.manipulators.database_manipulator import DatabaseManipulator

    if not video_start or not video_end or not fps:
        logging.warning(f"Video start, end, or fps not defined.")
        return True, None

    # Create database manipulator
    db = DatabaseManipulator(db_path)

    if 'video_data' not in db.get_table_names():
        logging.error(f"Table 'video_data' not found in the flowering minutes database.")
        return True, None

    # Query the video_data table for the given video_id
    try:
        query = f"""
        SELECT year, month, day, hour, minute, no_of_flowers
        FROM video_data
        ORDER BY year, month, day, hour, minute
        """
        rows = db.fetch_all(query)
    except Exception as e:
        logging.error(f"Error when querying the flowering minutes database: {e}")
        return True, None

    try:
        # Convert rows to datetime and flower count
        flower_data = []
        for row in rows:
            year, month, day, hour, minute, no_of_flowers = row
            time_point = datetime(year, month, day, hour, minute)
            flower_data.append((time_point, no_of_flowers))
    except Exception as e:
        logging.error(f"Error when parsing the flowering minutes database: {e}")
        return True, None

    flowers_at_start = None
    flowers_changed_during_video = False
    first_flower_frame = None
    try:
        # Check the number of flowers at the start of the video
        for time_point, no_of_flowers in reversed(flower_data):
            if time_point <= video_start:
                flowers_at_start = no_of_flowers
                break

        if flowers_at_start is None or flowers_at_start > 0:
            # No data available before the video start
            db.close_connection()
            return True, 0

        # Check if any time point during the video duration changes the number of flowers to non-zero
        for time_point, no_of_flowers in flower_data:
            if video_start < time_point <= video_end:
                if no_of_flowers > 0:
                    flowers_changed_during_video = True
                    first_flower_frame = int((time_point - video_start).total_seconds() * fps)
                    break
    except Exception as e:
        logging.error(f"Error when checking the flowering minutes data: {e}")
        return True, None
    finally:
        db.close_connection()

    if flowers_changed_during_video:
        return True, first_flower_frame
    else:
        return False, None


def should_process_video_from_boxes(reference_boxes: Optional[List[Type["DetectionBoxes"]]]):

    from detectflow.manipulators.box_manipulator import BoxManipulator

    try:
        if not reference_boxes:
            return True

        if all(box is None for box in reference_boxes):
            logging.info("All reference boxes are None. No flowers were detected.")
            return False

        if any(box is not None for box in reference_boxes):
            reference_boxes = [box for box in reference_boxes if box is not None]

        consistent_boxes = BoxManipulator.find_consistent_boxes(reference_boxes, iou_threshold=0.6, min_frames=3)

        if not consistent_boxes or len(consistent_boxes) == 0:
            logging.info("No consistent boxes found. No flowers were detected.")
            return False
    except Exception as e:
        logging.error(f"Error when checking the reference boxes: {e}")
        return True
