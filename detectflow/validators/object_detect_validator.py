from detectflow.validators.validator import Validator
import numpy as np
import datetime


class ObjectDetectValidator(Validator):
    def __init__(self):

        # Run the init method of Validator parent class
        Validator.__init__(self)

    @staticmethod
    def is_valid_ndarray_list(ndarray_list) -> bool:
        """
        Check if a list is not empty and contains only np.ndarrays.
        """
        if not isinstance(ndarray_list, list) or not ndarray_list:
            return False
        return all(isinstance(item, np.ndarray) for item in ndarray_list)

    @staticmethod
    def validate_ndarray_list(func):
        """Decorator to validate the output of a function as a list of np.ndarrays."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not ObjectDetectValidator.is_valid_ndarray_list(result):
                raise ValueError("Invalid ndarray list: Must be a non-empty list of np.ndarrays")
            return result

        return wrapper

    @staticmethod
    def validate_rois_object(rois):
        """
        Validate regions of interest (ROIs).
        """
        # Single ROI validation
        if isinstance(rois, (list, tuple, np.ndarray)) and len(rois) == 4 and all(
                isinstance(x, (int, float)) for x in rois):
            return [np.array(rois)]

        # List of ROIs validation
        if isinstance(rois, (list, tuple, np.ndarray)) and all(
                isinstance(item, (list, tuple, np.ndarray)) for item in rois):
            for item in rois:
                if len(item) != 4 or not all(isinstance(x, (int, float)) for x in item):
                    return None  # Invalid ROI found
            return [np.array(item) for item in rois]

        # Invalid input
        return None

    @staticmethod
    def is_valid_rois_object(func):
        """Decorator to validate the output of a function as ROIs."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if ObjectDetectValidator.validate_rois_object(result) is None:
                raise ValueError(
                    "Invalid ROIs object: Must be a single ROI or a list/tuple of ROIs with four elements each")
            return result

        return wrapper

    @staticmethod
    def is_valid_annotation_data_array(array):
        """Check if the array consists of [duration, timestamp] pairs for annotation data."""
        if not isinstance(array, list):
            return False
        for item in array:
            if not (isinstance(item, list) and len(item) == 2):
                return False
            duration, timestamp = item
            try:
                int(duration)  # Check if duration can be converted to an int
            except (ValueError, TypeError):
                return False
            if not isinstance(timestamp, str):
                return False
        return True

    @staticmethod
    def validate_annotation_data_array(func):
        """Decorator to validate the output of a function as an array of [duration, timestamp] pairs for annotation data."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_annotation_data_array(result):
                raise ValueError("Invalid annotation data array format: Expected [duration, timestamp] pairs")
            return result

        return wrapper

    @staticmethod
    def is_valid_annotation_array(array):
        """
        Check if the array consists of valid annotation entries.
        Each entry should be a list with the format:
        [duration (int), time_of_visit (datetime), video_filepath (str), video_start_time (datetime), video_end_time (datetime)]
        """
        if not isinstance(array, list):
            return False
        for item in array:
            if not (isinstance(item, list) and len(item) == 5):
                return False
            duration, time_of_visit, video_filepath, video_start_time, video_end_time = item
            if not isinstance(duration, int):
                return False
            if not all(isinstance(x, datetime.datetime) for x in [time_of_visit, video_start_time, video_end_time]):
                return False
            if not isinstance(video_filepath, str):
                return False
        return True

    @staticmethod
    def validate_annotation_array(func):
        """Decorator to validate the output of a function as a valid annotation array."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not Validator.is_valid_annotation_array(result):
                raise ValueError("Invalid annotation array format")
            return result

        return wrapper
