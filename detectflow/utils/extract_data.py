from detectflow.utils.hash import get_timestamp_hash
from detectflow.validators.object_detect_validator import ObjectDetectValidator
import json
from typing import Dict, List, Optional, Type, Any
from datetime import timedelta, datetime
import logging
import numpy as np
from detectflow.predict.results import DetectionResults
from functools import lru_cache


def safe_str(value):
    return str(value) if value is not None else str(None)


def safe_int(value, default=0):
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    else:
        return int(value) if value is not None else default


def safe_float(value, default=0.0):
    return float(value) if value is not None else default


def safe_datetime(value, str_format="%Y-%m-%dT%H:%M:%S"):
    return value.strftime(str_format) if isinstance(value, datetime) else str(None)


def safe_first_element(value):
    return value[0] if value else None


def safe_last_element(value):
    return value[-1] if value else None


def convert_to_lists(value):
    from detectflow.predict.results import DetectionBoxes

    if isinstance(value, np.ndarray):
        value = value.tolist()  # Convert numpy array to list
    elif isinstance(value, tuple):
        value = list(value)  # Convert tuple to list
    elif isinstance(value, DetectionBoxes):
        value = value.tolist()
    if isinstance(value, list):
        return [convert_to_lists(item) for item in value]  # Recursively convert elements
    return value


def safe_json(value):
    return json.dumps(convert_to_lists(value) if value is not None else [])


@lru_cache(maxsize=4)
def extract_data_from_video(video_path: Optional[str] = None, video_file: Optional[Type["Video"]] = None, s3_path: str = None, return_raw_data: bool = False, **kwargs):
    """
    Extract data from a video file and pack as a dictionary

    Args:
    - video_path: str, path to video file (alternatively to video_file)
    - video_file: Video object, video file object (alternatively to video_path)
    - s3_path: str, path to the video file in a S3 storage (optional)
    - **kwargs: additional keyword arguments for VideoDiagnoser
    """
    from detectflow.video.video_data import Video
    from detectflow.video.video_diagnoser import VideoDiagnoser

    try:
        if video_file:
            video = video_file
            video_path = video.video_path
            if s3_path:
                video.s3_path = s3_path
        elif video_path:
            video = Video(video_path, s3_path=s3_path)
        else:
            raise ValueError("No video file or video object supplied")
    except Exception as e:
        raise RuntimeError(f"Error initiating Video class for video file {video_path}: {e}")

    vid_results = {}
    if video:
        attrs = ["video_id", "recording_id", "s3_bucket", "s3_directory", "extension", "start_time", "end_time", "duration", "total_frames", "fps", "focus", "blur", "contrast", "brightness", "frame_width", "frame_height"]
        for a in attrs:
            try:
                vid_results[a] = getattr(video, a)
            except Exception as e:
                logging.error(f"Error getting attribute {a} from video object: {e}")
                vid_results[a] = None

    motion_method = "TA"
    try:
        if video:
            diag_kwargs = {"video_file": video, "motion_methods": kwargs.get('motion_methods', motion_method), "frame_skip": kwargs.get('frame_skip', 5)}
        else:
            diag_kwargs = {"video_path": video_path, "motion_methods": kwargs.get('motion_methods', motion_method), "frame_skip": kwargs.get('frame_skip', 5)}
        video_diag = VideoDiagnoser(**{**kwargs, **diag_kwargs})
    except Exception as e:
        raise RuntimeError(f"Error initiating VideoDiagnoser for video file {video_path}: {e}")

    logging.info(f"VideoDiagnoser initiated for video file {video_path}")

    diag_results = {}
    if video_diag:
        vs = ["reference_boxes", "focus_accuracies", "focus_regions"]
        funcs = ["get_ref_bboxes", "get_focus_accuracies", "get_focus_regions"]
        for v, f in zip(vs, funcs):
            try:
                diag_results[v] = getattr(video_diag, f)()
            except Exception as e:
                logging.error(f"Error getting {v} from video_diag object: {e}")
                diag_results[v] = None
            finally:
                if v == 'reference_boxes':
                    video_diag._ref_bboxes = diag_results.get("reference_boxes")  # Set the protected attr manually so the ref boxes are saved for future analyses.

        logging.info(f"VideoDiagnoser results extracted for video file {video_path}")

        attrs = ["daytime"]
        for a in attrs:
            try:
                diag_results[a] = getattr(video_diag, a)
            except Exception as e:
                logging.error(f"Error getting attribute {a} from video_diag object: {e}")
                diag_results[a] = None

        logging.info(f"VideoDiagnoser attributes extracted for video file {video_path}")

        try:
            motion_data = video_diag.motion_data.get(motion_method, None)
            mean_motion = motion_data.get("mean", None) if motion_data else None
            diag_results["mean_motion"] = mean_motion
        except Exception as e:
            logging.error(f"Error getting attribute motion data from video_diag object: {e}")
            diag_results["mean_motion"] = None

        logging.info(f"VideoDiagnoser motion data extracted for video file {video_path}")

    video_data = {
        "video_id": safe_str(vid_results.get("video_id", None)),
        "recording_id": safe_str(vid_results.get("recording_id", None)),
        "s3_bucket": safe_str(vid_results.get("s3_bucket", None)),
        "s3_directory": safe_str(vid_results.get("s3_directory", None)),
        "format": safe_str(vid_results.get("extension", None)),
        "start_time": safe_datetime(vid_results.get("start_time", None)),
        "end_time": safe_datetime(vid_results.get("end_time", None)),
        "length": safe_int(vid_results.get("duration", None)),
        "total_frames": safe_int(vid_results.get("total_frames", None)),
        "fps": safe_int(vid_results.get("fps", None)),
        "focus": safe_float(vid_results.get("focus", None)),
        "blur": safe_float(vid_results.get("blur", None)),
        "contrast": safe_float(vid_results.get("contrast", None)),
        "brightness": safe_float(vid_results.get("brightness", None)),
        "daytime": diag_results.get("daytime", "NULL"),
        "focus_regions_start": safe_json(safe_first_element(diag_results.get("focus_regions"))),
        "flowers_start": safe_json(safe_first_element(diag_results.get("reference_boxes"))),
        "focus_acc_start": safe_first_element(diag_results.get("focus_accuracies")),
        "focus_regions_end": safe_json(safe_last_element(diag_results.get("focus_regions"))),
        "flowers_end": safe_json(safe_last_element(diag_results.get("reference_boxes"))),
        "focus_acc_end": safe_last_element(diag_results.get("focus_accuracies")),
        "motion": safe_float(diag_results.get("mean_motion"))
    }

    raw_data = {
        'reference_boxes': diag_results.get("reference_boxes"),
        'start_time': vid_results.get("start_time"),
        'end_time': vid_results.get("end_time"),
        'fps': vid_results.get("fps"),
        'frame_width': vid_results.get("frame_width"),
        'frame_height': vid_results.get("frame_height")
    }

    if return_raw_data:
        return video_data, raw_data
    else:
        return video_data


def extract_data_from_result(result: DetectionResults) -> Dict[str, Any]:
    """ Extract relevant data from a DetectionResult object """

    if not isinstance(result, DetectionResults):
        raise TypeError(f"Invalid type of DetectionResults object supplied to results queue: {type(result)}")

    # Required attributes
    results = {}
    required_attrs = ["frame_number", "video_time", "real_time", "recording_id", "video_id", "source_path", "reference_boxes", "boxes", "filtered_boxes", "on_flowers"]
    for a in required_attrs:
        try:
            results[a] = getattr(result, a)
        except Exception as e:
            logging.error(f"Error getting attribute {a} from DetectionResults object: {e}")
            results[a] = None

    # Extract and validate data
    results['frame_number'] = ObjectDetectValidator.validate_frame_number(safe_int(results.get('frame_number'), default=get_timestamp_hash()))

    results['video_time'] = safe_str(ObjectDetectValidator.validate_video_time(timedelta(seconds=safe_int(results.get('video_time'))))).split('.')[0]

    if results.get('real_time') is not None and isinstance(results.get('real_time'), datetime):
        real_time = results.get('real_time')
        life_time = safe_datetime(real_time, "%H:%M:%S")
        year = safe_int(real_time.year)
        month = safe_int(real_time.month)
        day = safe_int(real_time.day)
    else:
        life_time, year, month, day = None, 0, 0, 0

    results['recording_id'] = safe_str(results.get('recording_id'))
    results['video_id'] = safe_str(results.get('video_id'))
    results['video_path'] = safe_str(results.get('source_path'))

    reference_bboxes = safe_json(results.get('reference_boxes'))
    visitor_bboxes = safe_json(results.get('boxes'))
    filtered_visitor_bboxes = safe_json(results.get('filtered_boxes'))
    visit_ids_conditions = [
        hasattr(results.get('boxes'), 'id'),
        results.get('boxes').id is not None and len(results.get('boxes').id) > 0
    ]
    visit_ids = safe_json(results.get('boxes').id if all(visit_ids_conditions) else [-1.0 for _ in results.get('boxes')] if results.get('boxes') is not None else [])
    on_flower = safe_json(results.get('on_flowers'))

    flags = ""  # Placeholder for future use
    rois = ""  # Placeholder for future use

    # Return the extracted data as a dictionary
    return {
        "frame_number": results.get('frame_number'),
        "video_time": results.get('video_time'),
        "life_time": life_time,
        "year": year,
        "month": month,
        "day": day,
        "recording_id": results.get('recording_id'),
        "video_id": results.get('video_id'),
        "video_path": results.get('video_path'),
        "flower_bboxes": reference_bboxes,
        "rois": rois,
        "all_visitor_bboxes": visitor_bboxes,
        "relevant_visitor_bboxes": filtered_visitor_bboxes,
        "visit_ids": visit_ids,
        "on_flower": on_flower,
        "flags": flags
    }