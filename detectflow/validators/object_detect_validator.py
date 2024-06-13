from detectflow.validators.validator import Validator
import numpy as np
import datetime
import re

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
        elif isinstance(rois, (list, tuple, np.ndarray)) and all(
                isinstance(item, (list, tuple, np.ndarray)) for item in rois):
            for item in rois:
                if len(item) != 4 or not all(isinstance(x, (int, float)) for x in item):
                    #return None  # Invalid ROI found
                    raise ValueError("Invalid ROIs object: Each ROI must have four integers or floats")
            return [np.array(item) for item in rois]
        # Invalid input
        else:
            raise ValueError("Invalid ROIs object: Must be a single ROI or a list/tuple/numpy.arrays of ROIs with four elements each")

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

    @staticmethod
    def validate_frame_number(frame_number):

        # Frame number should not be larger than 30K ussually
        if not isinstance(frame_number, int):
            try:
                frame_number = int(frame_number)
            except Exception as e:
                print(f"Error when validating frame_number: {e}")

        if not frame_number < 30000:
            print(f"The frame number '{frame_number}' is larger than expected.")

        return frame_number

    @staticmethod
    def validate_video_time(video_time):

        # Video time will not usually be higher than 960 s
        try:
            if not isinstance(video_time, datetime.timedelta):
                raise TypeError(
                    f"Unexpected format in video_time. Expected <timedelta>, got <{type(video_time)}> instead.")

            if video_time < datetime.timedelta(seconds=0):
                raise ValueError(f"Unexpected video_time value '{video_time}.' Expected value larger than 0.")
        except Exception as e:
            print(f"Error when validating video time: {e}")

        return video_time

    @staticmethod
    def validate_video_ids(recording_id, video_id):
        # Video IDS
        try:
            # Recording ID has a format of XX(X)0_X0_XXXXXX00
            recording_id_pattern = r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}$'

            if re.match(recording_id_pattern, recording_id) is None:
                raise ValueError(f"The string '{recording_id}' does not match the expected format.")

            # Video ID has a format of XX(X)0_X0_XXXXXX00_00000000_00_00
            video_id_pattern = (r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{2}_'
                                r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])_'
                                r'([01]\d|2[0-3])_([0-5]\d)$')

            if re.match(video_id_pattern, video_id) is None:
                raise ValueError(f"The string '{video_id}' does not match the expected format.")

        except Exception as e:
            print(f"Error when validating video IDs: {e}")
            # Decide what to do to handle these errors.

        return recording_id, video_id

    @staticmethod
    def validate_video_path(video_path):
        # Path validation
        try:
            if not Validator.is_valid_file_path(video_path):
                raise ValueError(f"The video path '{video_path}' is not a valid file.")
        except Exception as e:
            print(f"Error when validating video filepath: {e}")
            # Decide what to do to handle this error.

        return video_path
