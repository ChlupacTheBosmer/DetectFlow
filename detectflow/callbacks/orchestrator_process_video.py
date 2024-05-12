from detectflow.process.orchestrator import Orchestrator, Task
from detectflow.process.database_manager import DatabaseManager
from detectflow.process.frame_generator import FrameGenerator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.validators.validator import Validator
from detectflow.utils.threads import calculate_optimal_threads
from detectflow.callbacks.frame_generator_predictions import test_callback
import logging
from queue import Queue
import os

def process_video_callback(task: Task,
                           orchestrator: Orchestrator,
                           name: str,
                           **kwargs):
    CONFIG_MAP = {"scratch_path": str,
                  "db_manager": DatabaseManager,
                  "frame_batch_size": int,
                  "frame_skip": int,
                  "max_producers": int,
                  "max_workers": int,
                  "db_queue": Queue,
                  "model_config": dict,
                  "crop_imgs": bool,
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
    db_manager = kwargs.get('db_manager', None)

    # Frame Generator config
    first_frame_number = 0
    frame_batch_size = kwargs.get('frame_batch_size', 50)
    frame_skip = kwargs.get('frame_skip', 15)

    # Generator thread config
    max_producers = kwargs.get('max_producers', 1)
    max_workers = kwargs.get('max_workers', 1)

    # Generator callback config
    db_queue = kwargs.get('db_queue', None)

    # Model config
    model_config = kwargs.get('model_config', None)

    # Funcitonality flags
    crop_imgs = kwargs.get('crop_imgs', False)
    inspect = kwargs.get('inspect', False)

    # Load videos and validate them
    try:
        dataloader = Dataloader()
        valid, invalid = dataloader.prepare_data(files, scratch, False)

        if not len(valid) == len(files) or len(invalid) > 0:
            raise ValueError(f"Some ({len(invalid)}) videos failed validation: {invalid}")

    except Exception as e:
        logging.error(f"{name} - Error when loading and validating data: {e}")
    finally:
        del dataloader

    try:
        logging.info(f"{name} - Videos loaded and validated: {valid}")

        for i, video in enumerate(valid):

            # Get video_id
            video_id, _ = os.path.splitext(os.path.basename(video))

            # TODO: must match stat to file even when path is changed!
            status = task.get_status(files[i])
            first_frame_number = status if status and status != -1 else 0
            update_info = {"directory": directory, "file": files[i], "status": status}
            if status == -1:
                continue

            # Initialize its database
            db_manager.add_and_initialize_database(video_id, os.path.join(scratch, f"{video_id}.db"))

            # Generate frames and run detection
            generator = FrameGenerator(source=[video],
                                       output_folder=scratch,
                                       processing_callback=test_callback,
                                       first_frame_number=first_frame_number,
                                       frame_skip=frame_skip,
                                       db_queue=db_queue,
                                       update_info=update_info,
                                       orchestrator=orchestrator,
                                       crop_imgs=crop_imgs,
                                       inspect=inspect,
                                       model_config=model_config
                                       )

            print(type(generator))
            generator.run(nprod=max_producers,
                          ndet=calculate_optimal_threads(max_workers),
                          frame_batch_size=frame_batch_size)
    except Exception as e:
        logging.error(f"{name} - Error when processing video: {video} - {e}")