from detectflow.utils.hash import get_numeric_hash
from detectflow.validators.object_detect_validator import ObjectDetectValidator
import json
from typing import Dict, List, Optional, Type, Any
from datetime import timedelta, datetime
import logging
from detectflow.predict.results import DetectionResults


def safe_str(value):
    return str(value) if value is not None else str(None)


def safe_int(value, default=0):
    return int(value) if value is not None else default


def safe_float(value, default=0.0):
    return float(value) if value is not None else default


def safe_datetime(value, str_format="%Y-%m-%dT%H:%M:%S"):
    return value.strftime(str_format) if isinstance(value, datetime) else str(None)


def safe_first_element(value):
    return value[0] if value else None


def safe_last_element(value):
    return value[-1] if value else None


def safe_json(value):
    return json.dumps(value.to_list() if value is not None else [])


def extract_data_from_video(video_path: str, s3_path: str = None):
    """ Extract data from a video file and pack as a dictionary"""
    from detectflow.video.video_data import Video
    from detectflow.video.video_diagnoser import VideoDiagnoser

    try:
        video = Video(video_path, s3_path)
    except Exception as e:
        raise RuntimeError(f"Error initiating Video class for video file {video_path}: {e}")

    vid_results = {}
    if video:
        attrs = ["video_id", "recording_id", "s3_bucket", "s3_directory", "extension", "start_time", "end_time", "duration", "total_frames", "fps", "focus", "blur", "contrast", "brightness"]
        for a in attrs:
            try:
                vid_results[a] = getattr(video, a)
            except Exception as e:
                logging.error(f"Error getting attribute {a} from video object: {e}")
                vid_results[a] = None

    motion_method = "TA"
    try:
        if video:
            kwargs = {"video_file": video, "motion_methods": motion_method, "frame_skip": 2}
        else:
            kwargs = {"video_path": video_path, "motion_methods": motion_method, "frame_skip": 2}
        video_diag = VideoDiagnoser(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error initiating VideoDiagnoser for video file {video_path}: {e}")

    diag_results = {}
    if video_diag:
        vs = ["focus_accuracies", "reference_boxes", "focus_regions"]
        funcs = ["get_focus_accuracies", "get_ref_bboxes", "get_focus_regions"]
        for v, f in zip(vs, funcs):
            try:
                diag_results[v] = getattr(video_diag, f)()
            except Exception as e:
                logging.error(f"Error getting {v} from video_diag object: {e}")
                diag_results[v] = None

        attrs = ["daytime", "thumbnail"]
        for a in attrs:
            try:
                diag_results[a] = getattr(video_diag, a)
            except Exception as e:
                logging.error(f"Error getting attribute {a} from video_diag object: {e}")
                diag_results[a] = None
        try:
            motion_data = video_diag.motion_data.get(motion_method, None)
            mean_motion = motion_data.get("mean", None) if motion_data else None
            diag_results["mean_motion"] = mean_motion
        except Exception as e:
            logging.error(f"Error getting attribute motion data from video_diag object: {e}")
            diag_results["mean_motion"] = None

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
        "thumbnail": diag_results.get("thumbnail", None),
        "focus_regions_start": safe_first_element(diag_results.get("focus_regions")),
        "flowers_start": safe_first_element(diag_results.get("reference_boxes")),
        "focus_acc_start": safe_first_element(diag_results.get("focus_accuracies")),
        "focus_regions_end": safe_last_element(diag_results.get("focus_regions")),
        "flowers_end": safe_last_element(diag_results.get("reference_boxes")),
        "focus_acc_end": safe_last_element(diag_results.get("focus_accuracies")),
        "motion": safe_float(diag_results.get("mean_motion"))
    }

    return video_data


def extract_data_from_result(result: DetectionResults) -> Dict[str, Any]:
    """ Extract relevant data from a DetectionResult object """

    if not isinstance(result, DetectionResults):
        raise TypeError(f"Invalid type of DetectionResults object supplied to results queue: {type(result)}")

    # Required attributes
    results = {}
    required_attrs = ["frame_number", "video_time", "real_time", "recording_id", "video_id", "source_path", "ref_boxes", "boxes", "fil_boxes", "on_flowers"]
    for a in required_attrs:
        try:
            results[a] = getattr(result, a)
        except Exception as e:
            logging.error(f"Error getting attribute {a} from DetectionResults object: {e}")
            results[a] = None

    # Extract and validate data
    results['frame_number'] = ObjectDetectValidator.validate_frame_number(safe_int(results.get('frame_number'), default=get_numeric_hash()))

    results['video_time'] = safe_str(ObjectDetectValidator.validate_video_time(timedelta(seconds=safe_int(results.get('video_time'))))).split('.')[0]

    if results.get('real_time') is not None and isinstance(results.get('real_time'), datetime):
        real_time = results.get('real_time')
        life_time = safe_datetime(real_time, "%H:%M:%S")
        year = safe_int(real_time.year)
        month = safe_int(real_time.month)
        day = safe_int(real_time.day)
    else:
        life_time, year, month, day = None, 0, 0, 0

    results['recording_id'], results['video_id'] = ObjectDetectValidator.validate_video_ids(safe_str(results.get('recording_id')), safe_str(results.get('video_id')))
    results['video_path'] = ObjectDetectValidator.validate_video_path(safe_str(results.get('source_path')))

    reference_bboxes = safe_json(results.get('ref_boxes'))
    visitor_bboxes = safe_json(results.get('boxes'))
    filtered_visitor_bboxes = safe_json(results.get('fil_boxes'))
    visit_ids = safe_json(results.get('boxes').id if hasattr(results.get('boxes'), 'id') else None)
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