import logging
import os
from detectflow.predict.predictor import Predictor
from detectflow.utils.inspector import Inspector
from detectflow.video.video_data import Video
from detectflow.utils.extract_data import extract_data_from_result
from detectflow.models import DEFAULT_MODEL_CONFIG as model_defaults
from detectflow.predict.ensembler import Ensembler


def frame_generator_predict(**kwargs):
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
               - model_config: List or Dictionary of models and their paths and confs (default: model_defaults - see for dict structure)
                               Note that the dictionary should have keys 0 and 1 for flowers and visitors models respectively.
                               Having list with two or more elements is also acceptable because when loading from json dictionary
                               is not acceptable data type. If the model_config has more than two models, Ensembler
                               combining results from multiple models will be used instead of a single Predictor.

               Passed in the kwargs dict passed from orchestrator_control_queue - fully optional
               - db_manager_data_queue: Queue to which results should be put for logging into database (default: None)
               - orchestrator_control_queue: Orchestrator queue object that will handle video status update (default: None)
               - update_info: Info that will be submitted back to orchestrator_control_queue updating status for a video file (default: None)
               - scratch_path: Path to the current scratch folder (default: None)

               Functionality control flags
               - inspect:
               - crop_imgs:

    """

    # Unpack keyword arguments with defaults
    frames = kwargs.get('frames_array', None)
    frame_numbers = kwargs.get('frame_numbers', [])
    frame_skip = kwargs.get('frame_skip', 15)
    video_filepath = kwargs.get('video_filepath', 'Unknown')
    video_filename = kwargs.get('video_filename', 'Unknown')
    visit_numbers = kwargs.get('visit_numbers', [])
    task = kwargs.get('task', None)
    name = kwargs.get('worker_name', 'Unknown Worker')
    color = kwargs.get('color', 'black')
    generator = kwargs.get('generator', None)

    # Model parameters
    model_config = kwargs.get('model_config', model_defaults) if kwargs.get('model_config',
                                                                            model_defaults) is not None else model_defaults
    track_results = kwargs.get('track_results', False)
    tracker_type = kwargs.get('tracker_type', 'botsort.yaml' if track_results else None)
    device = kwargs.get('device', None)
    logging.info(device)

    # Optional orchestrator implementation
    db_manager_data_queue = kwargs.get('db_manager_data_queue', None)
    resource_monitor_queue = kwargs.get('resource_monitor_queue', None)
    orchestrator_control_queue = kwargs.get('orchestrator_control_queue', None)
    update_info = kwargs.get('update_info', None)
    scratch_path = kwargs.get('scratch_path', None)

    # Functionality flags
    inspect = kwargs.get('inspect', False)
    skip_empty_frames = kwargs.get('skip_empty_frames', False)

    logging.info(f"Running predictions on video: <{video_filename}>")

    # Debug message
    try:
        if resource_monitor_queue:
            event_message = f"{name} got frames"
            resource_monitor_queue.put((event_message, color))
    except Exception as e:
        pass

    if frame_numbers:
        logging.info(f"Frame number range: {frame_numbers[0]} to {frame_numbers[-1]}")
    else:
        logging.info("No frame numbers provided.")

    if frames is None or len(frames) == 0:
        raise ValueError("No frames passed to generator callback")

    # Process model config to ensure it is in the correct format
    if isinstance(model_config, dict):
        model_config = [model_config[key] for key in model_config.keys()]
    elif isinstance(model_config, list):
        model_config = model_config
    else:
        raise ValueError("Model config must be a dictionary or a list")

    ############ FLOWERS ############
    # Define Predictor
    flower_predictor = Predictor()

    # Get the position of flowers. Return DetectionResults
    det_results = []
    for result in flower_predictor.detect(frame_numpy_array=frames[:1],
                                          model_path=model_config[0].get('path', model_defaults[0]['path']),
                                          detection_conf_threshold=model_config[0].get('conf',model_defaults[0]['conf']),
                                          device=device,
                                          sliced=True):
        # Append result to the list - We get a list of DetectionResults
        det_results.append(result)

        # Display all boxes
        if inspect and result is not None:
            Inspector.display_frames_with_boxes([result.orig_img], [result.boxes if hasattr(result, 'boxes') else None])

    ############ VISITORS ############
    # Get the video start time to manually assign it to results to save processing for each result
    video = Video(video_filepath)
    # video_start_time = video.start_time if hasattr(video, 'start_time') and video.start_time else None
    video_total_frames = video.total_frames if hasattr(video, 'total_frames') and video.total_frames else None

    # Pack metadata
    metadata = {
        'frame_number': frame_numbers,
        'visit_number': visit_numbers,
        'source_path': video_filepath,
        'reference_boxes': det_results[0].boxes if det_results and len(det_results) > 0 and hasattr(det_results[0], 'boxes') else None
    }

    # Define Predictor(s) and Ensembler as detector
    if len(model_config) > 2:
        visitor_predictors = [Predictor(model_path=cfg.get('path', model_defaults[1]['path']),
                                        detection_conf_threshold=cfg.get('conf', model_defaults[1]['conf']),
                                        tracker=tracker_type) for cfg in model_config[1:]]
        detector = Ensembler(predictors=visitor_predictors)
        logging.info(f"Ensembling {len(visitor_predictors)} models")

    else:
        detector = Predictor(model_path=model_config[1].get('path', model_defaults[1]['path']),
                             detection_conf_threshold=model_config[1].get('conf', model_defaults[1]['conf']),
                             tracker=tracker_type)
        logging.info(f"Using single model for visitors")

    # Define detection kwargs
    detect_kwargs = {'frame_numpy_array': frames,
                     'metadata': metadata,
                     'sliced': True,
                     'tracked': track_results,
                     'filter_tracked': False,
                     'device': device
                     }

    # Get the position of visitors. Return DetectionResults
    for i, result in enumerate(detector.detect(**detect_kwargs)):
        if result is not None:

            # Display all boxes on the frame if inspection requested
            if inspect:
                Inspector.display_frame_with_multiple_boxes(result.orig_img, [result.boxes, result.reference_boxes, result.filtered_boxes])

            # Save training data to scratch folder
            try:
                if scratch_path and result.boxes is not None:
                    result.save_dir = os.path.join(scratch_path, 'train', result.video_id)
                    result.save(sort=True, assume_folder_exists=False, save_txt=True, box_type='boxes')
            except Exception as e:
                logging.error(f"Error when saving training data: {e}")

            # Add to queue
            if result.boxes is not None or not skip_empty_frames:
                if db_manager_data_queue is not None:
                    result_data_entry = extract_data_from_result(result)
                    db_manager_data_queue.put(result_data_entry)
                else:
                    raise TypeError("Database task queue not defined")

        # If orchestrator_control_queue task was passed then update the progress
        if update_info is not None and len(frame_numbers) >= i + 1:

            # Determine what the new status of the video is
            update_info["status"] = -1 if abs(video_total_frames - frame_numbers[i]) <= frame_skip else frame_numbers[i]

            # Call the update method of the orchestrator_control_queue
            if orchestrator_control_queue:
                orchestrator_control_queue.put(('update_task', (update_info,)))

    # Debug message
    try:
        if resource_monitor_queue:
            event_message = f"{name} done frames"
            resource_monitor_queue.put((event_message, color))
    except Exception as e:
        pass

