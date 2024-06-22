import logging
from detectflow.predict.predictor import Predictor
from detectflow.predict.tracker import Tracker
from detectflow.utils.inspector import Inspector
from detectflow.image.smart_crop import SmartCrop
from detectflow.video.video_inter import VideoFileInteractive


def test_callback(**kwargs): #TODO: Rename the callback function
    """
    A robust callback function template that unpacks keyword arguments (kwargs) and provides default values.

    Parameters:
    **kwargs -- A dictionary of keyword arguments where key is the argument name and value is the argument value.
               Expected keys:
               - frames_array: The array of frames (default: None if not provided)
               - frame_numbers: List of frame numbers (default: [])
               - frame_skip: Number of frames to skip between analysed frames (default: 15)
               - video_filepath: Path to the video file (default: 'Unknown')
               - video_filename: Name of the video file (default: 'Unknown')
               - visit_numbers: List of visit numbers (default: [])
               - task: The task object (default: None)
               - generator: The instance of the generator (default: None)

               Model parameters:
               - model_config: Dictionary fo models and their paths and confs (default: model_defaults - see for dict structure)

               Passed in the kwargs dict passed from orchestrator - fully optional
               - db_queue: Queue to which results should be put for logging into database (default: None)
               - orchestrator: Orchestrator object that will handle video status update (default: None)
               - update_info: Info that will be submitted back to orchestrator updating status for a video file (default: None)

               Functionality control flags
               - inspect:
               - crop_imgs:

    """

    model_defaults = {0: {'path': "/storage/brno2/home/USER/Flowers/flowers_ours_f2s/weights/best.pt",
                          'conf': 0.1},
                      1: {
                          'path': "/storage/brno2/home/USER/Dataset Composition Test/751e1107_enrich_single_cls/weights/best.pt",
                          'conf': 0.1}}

    # Unpack keyword arguments with defaults
    frames = kwargs.get('frames_array', None)
    frame_numbers = kwargs.get('frame_numbers', [])
    frame_skip = kwargs.get('frame_skip', 15)
    video_filepath = kwargs.get('video_filepath', 'Unknown')
    video_filename = kwargs.get('video_filename', 'Unknown')
    visit_numbers = kwargs.get('visit_numbers', [])
    task = kwargs.get('task', None)
    generator = kwargs.get('generator', None)

    # Model parameters
    model_config = kwargs.get('model_config', model_defaults) if kwargs.get('model_config',
                                                                            model_defaults) is not None else model_defaults

    # Optional orchestrator implementation
    db_queue = kwargs.get('db_queue', None)
    orchestrator = kwargs.get('orchestrator', None)
    update_info = kwargs.get('update_info', None)

    # Functionality flags
    crop_imgs = kwargs.get('crop_imgs', False)
    inspect = kwargs.get('inspect', False)

    # Example processing code (replace the following lines with actual processing logic)
    logging.info(f"Processing video <{video_filename}>")

    if frame_numbers:
        logging.info(f"Frame number range: {frame_numbers[0]} to {frame_numbers[-1]}")
    else:
        logging.info("No frame numbers provided.")

    ############ FLOWERS ############
    if frames is None or len(frames) == 0:
        raise ValueError("No frames passed to generator callback")
    else:
        # Define Predictor
        flower_predictor = Predictor()

        # Get the position of flowers. Return DetectionResults
        det_results = []
        for result in flower_predictor.detect(frame_numpy_array=frames[:1],
                                              model_path=model_config[0].get('path', model_defaults[0]['path']),
                                              detection_conf_threshold=model_config[0].get('conf',
                                                                                           model_defaults[0]['conf'])):
            # Append result to the list - We get a list of DetectionResults
            det_results.append(result)

        if crop_imgs and len(det_results) > 0:

            # Display all boxes
            if inspect:
                Inspector.display_frames_with_boxes([result.orig_img for result in det_results],
                                                    [result.boxes for result in det_results])

            # Check if the boxes can fit in a reasonably sized square crop TODO: Check for None before passign the detection boxes and select a frame which has detection
            for result in det_results:
                smc = SmartCrop.from_detection_results(result,
                                                       crop_size=(640, 640),
                                                       handle_overflow="expand",
                                                       max_expansion_limit=(1000, 1000),
                                                       margin=50,
                                                       exhaustive_search=True,
                                                       permutation_limit=8,
                                                       multiple_rois=True)

                smart_result = smc.smart_crop(partial_overlap=True,
                                              iou_threshold=0.3,
                                              allow_slicing=True,
                                              evenness_threshold=0.66,
                                              force_slice_empty=False,
                                              inspect=inspect)

    ############ VISITORS ############
    if frames is None or len(frames) == 0:
        raise ValueError("No frames passed to generator callback")
    else:

        # Get the video start time to manually assign it to resutls to sae processing for each result
        video = VideoFileInteractive(video_filepath, initiate_start_and_end_times=True)
        video_start_time = video.start_time if hasattr(video, 'start_time') and video.start_time else None
        video_total_frames = video.total_frames if hasattr(video, 'total_frames') and video.total_frames else None

        # Subslice for testing purposes
        frames = frames  # [:1]

        # Define Predictor
        visitor_predictor = Predictor()

        # Define Tracker
        tracker = Tracker()

        # Get the position of flowers. Return DetectionResults
        for i, result in enumerate(visitor_predictor.detect(frame_numpy_array=frames,
                                                            model_path=model_config[1].get('path',
                                                                                           model_defaults[1]['path']),
                                                            detection_conf_threshold=model_config[1].get('conf',
                                                                                                         model_defaults[
                                                                                                             1][
                                                                                                             'conf']))):

            if inspect:
                Inspector.display_frames_with_boxes([result.orig_img], [result.boxes])

            # Predictor Performs tracking on the result
            try:
                result = tracker.process_tracking(result)[0]
            except Exception as e:
                logging.error(f"Error in tracking: {e}")

            # Add attributes
            result._real_start_time = video_start_time
            result.reference_boxes = det_results[0].boxes
            result.frame_number = frame_numbers[i] if len(frame_numbers) >= i + 1 else None
            result.source_path = video_filepath

            # Add to queue
            if db_queue is not None:
                db_queue.put(result)
            else:
                raise TypeError("Database task queue not defined")

            # If orchestrator task was passed then update the progress
            if update_info is not None and len(frame_numbers) >= i + 1:

                # Determine what the new status of the video is
                update_info["status"] = -1 if abs(video_total_frames - frame_numbers[i]) <= frame_skip else \
                frame_numbers[i]

                # Call the udpate method of the orchestrator
                if orchestrator:
                    orchestrator.handle_worker_update(update_info)