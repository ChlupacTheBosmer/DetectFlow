import math
import traceback

from ultralytics.engine.results import Boxes, Results  # Import the Boxes class from module
from sahi.prediction import PredictionResult, ObjectPrediction
import copy
import numpy as np
import os
import cv2
from typing import List, Optional, Type, Tuple
from datetime import timedelta, datetime
from detectflow.utils import BaseClass
from detectflow.utils.name import is_valid_video_id, parse_recording_name
import logging
from functools import lru_cache


def process_box(box, orig_shape, format_flag):
    format_map = {'x': 0, 'y': 1, 'w': 2, 'h': 3, 'p': 4, 'c': 5, 't': 6}
    normalize = 'n' in format_flag
    format_flag = format_flag.replace("n", "")
    iters = {'x': 0, 'y': 0}

    # Initialize new_box with the correct number of elements based on format_flag
    new_box = [0] * len(format_flag)  # Initialize for all needed values

    # Process box based on format_flag
    for i, char in enumerate(format_flag):
        if char in 'xy':
            # Multiple 'x' and 'y', handle 'xyxy' format
            new_box[format_map[char] + (iters[char] * 2)] = float(box[i])
            iters[char] = 1
        elif char in 'wh':  # Handle 'xywh' format
            x_center, y_center, width, height = box[:4]
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            new_box[:4] = [float(x_min), float(y_min), float(x_max), float(y_max)]
            break  # Exit loop after assigning xywh
        elif char in 'cp' or (char == 't' and 't' in format_map):
            # Assign class, confidence, and track ID directly
            new_box[format_map[char]] = box[i]

    # Normalize if required
    if normalize and 'xywh' in format_flag:
        new_box[0] *= orig_shape[1]  # Normalize x_min
        new_box[1] *= orig_shape[0]  # Normalize y_min
        new_box[2] *= orig_shape[1]  # Normalize x_max
        new_box[3] *= orig_shape[0]  # Normalize y_max

    # # Remove unused elements if format flag does not include 't'
    # if 't' not in format_flag:
    #     # new_box = new_box[:-1]
    #     new_box = np.hstack((new_box[:4], new_box[5:]))

    return new_box


def determine_source_type(path):
    if path is None:
        return "array"

    # Check if the source_path is a URL
    if is_valid_url(path):
        return "url"

    # File extension mapping
    video_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Get file extension
    _, file_extension = os.path.splitext(path.lower())

    # Determine the type based on the extension
    if file_extension in video_formats:
        return "video"
    elif file_extension in image_formats:
        return "image"
    else:
        return "unknown"


def get_avg_box_max_dim(detection_boxes_list: List[Type["DetectionBoxes"]]):

    boxes_list = []
    for detection_boxes in detection_boxes_list:
        if isinstance(detection_boxes, DetectionBoxes):
            boxes_list += list(detection_boxes.xywh)
    boxes = np.array(boxes_list)

    # Calculate the average of the larger dimension (width or height) for flower boxes
    if len(boxes) > 0:
        avg_box_size = sum(max(b[2], b[3]) for b in boxes) / len(boxes)
    else:
        avg_box_size = 0

    return avg_box_size


def get_avg_box_diag(detection_boxes_list: List[Type["DetectionBoxes"]]):

    from detectflow.manipulators.box_manipulator import BoxManipulator

    boxes_list = []
    for detection_boxes in detection_boxes_list:
        if isinstance(detection_boxes, DetectionBoxes):
            boxes_list += list(detection_boxes.xyxy)
    boxes = np.array(boxes_list)

    # Calculate the average of the boxes diagonal length
    if len(boxes) > 0:
        avg_diag_length = sum(math.sqrt(BoxManipulator.get_box_area(b)) for b in boxes) / len(boxes)
    else:
        avg_diag_length = 0

    return avg_diag_length


def is_valid_url(path):

    import urllib.parse

    try:
        result = urllib.parse.urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class DetectionBoxes(BaseClass):
    def __init__(self, boxes, orig_shape):

        try:
            boxes = np.array(boxes)
            if boxes.ndim == 1:
                boxes = boxes[None, :]
        except Exception as e:
            raise TypeError(f"Boxes must be a numpy array, list, or tuple: {e}")

        columns_num = boxes.shape[-1]
        if columns_num not in (4, 5, 6, 7):  # xyxy, prob, cls, track_id
            raise ValueError(f"Expected boxes of length 4, 5, 6 or 7 but got {columns_num}")

        self.is_track = columns_num == 7
        self.data = boxes
        self.orig_shape = orig_shape

    @classmethod
    def from_boxes_instance(cls, boxes: Boxes):

        if isinstance(boxes, Boxes):
            orig_shape = boxes.orig_shape
            boxes_array = np.array(boxes.data)
            format_flag = "xyxytpc" if boxes.is_track else "xyxypc"
        else:
            raise TypeError("Input must be an instance of Boxes")

        return cls.from_custom_format(boxes_array, orig_shape, format_flag)

    @classmethod
    def from_coco(cls, coco_boxes, orig_shape, format_flag: Optional[str] = None):
        """
        Will init the class instance from an array of coco formatted boxes [x_min, y_min, width, height] with any
        additional columns with more data. Default format is class, probability, track_id, but other order can be specified
        by passing a format_flag - flag must start with "xyxy" as this will be passed on after conversion.
        """
        from detectflow.manipulators.box_manipulator import BoxManipulator

        # Convert boxes
        xyxy_boxes = BoxManipulator.coco_to_xyxy(coco_boxes)

        # Default format flag
        format_flag = "xyxy" if format_flag is None else format_flag

        # Modify format flag if additional data is found and user has not defined custom flag
        if xyxy_boxes.shape[1] > 4 and format_flag is None:

            for col_index, char in zip(range(4, xyxy_boxes.shape[1]), "pct"):
                format_flag = format_flag + char

        return cls.from_custom_format(xyxy_boxes, orig_shape, format_flag)

    @classmethod
    def from_custom_format(cls, boxes, orig_shape: Tuple[int, int], format_flag: str):
        """
        Will init the class instance from an array of boxes with any format flag.
        """
        try:
            boxes = np.array(boxes)
            if boxes.ndim == 1:
                boxes = boxes[None, :]
        except Exception as e:
            raise TypeError(f"Boxes must be a numpy array, list, or tuple: {e}")

        try:
            processed_boxes = np.array([process_box(box, orig_shape, format_flag) for box in boxes])
        except Exception as e:
            raise RuntimeError(f"Error in DetectionBoxes initialization: {e}")

        return cls(processed_boxes, orig_shape)

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, 4] if self.data.shape[1] > 4 else None

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, 5] if self.data.shape[1] > 5 else None

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, 6] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        from detectflow.manipulators.box_manipulator import BoxManipulator

        return BoxManipulator.xyxy_to_xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = np.copy(self.xyxy)
        xyxy = xyxy.astype(float)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        from detectflow.manipulators.box_manipulator import BoxManipulator

        xywh = BoxManipulator.xyxy_to_xywh(self.xyxy)
        xywh = xywh.astype(float)
        xywh[..., [0, 2]] /= float(self.orig_shape[1])
        xywh[..., [1, 3]] /= float(self.orig_shape[0])
        return xywh

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    @property
    def coco(self):

        # Ensure the input is a NumPy array
        boxes_xyxy = np.array(self.xyxy)

        # Calculate width and height from xyxy format
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # x_max - x_min
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # y_max - y_min

        # Form the COCO format boxes array
        boxes_coco = np.vstack((boxes_xyxy[:, 0],  # x_min
                                boxes_xyxy[:, 1],  # y_min
                                widths,
                                heights)).T  # transpose to match the input shape (N, 4)

        return boxes_coco

    def tolist(self):
        # Convert boxes to a list of lists
        return self.data.tolist()

    def copy(self):
        #         # Creates a copy of the boxes
        #         return DetectionBoxes(self.data, self.orig_shape, "xyxycp")
        """
        Create a deep copy of the instance to ensure complete duplication of all mutable attributes.
        """
        return copy.deepcopy(self)

    def __iter__(self):
        # The __iter__ method returns the iterable object itself
        return iter(self.data)

    def __getitem__(self, key):
        # Handle integer, slice, and list/array of indices
        if isinstance(key, int) or isinstance(key, slice):
            return self.data[key]
        elif isinstance(key, (list, np.ndarray)):
            # If key is a list or NumPy array of indices
            return [self.data[i] for i in key]
        else:
            raise TypeError("Index must be int, slice, or array of indices.")

    def __contains__(self, box):
        # Check if 'box' matches any row in 'self.data' adjusted to the length of 'box'
        return any((self.data[:, :np.array(box).shape[0]] == box).all(axis=1))

    def __len__(self):
        """Return the length of the data array."""
        return len(self.data)


class DetectionResults(BaseClass):
    def __init__(self, orig_img, boxes=None, **kwargs):
        """
        Initialize the DetectionResults either by passing the same arguments as the Results class,
        by passing an instance of the Results class, or by passing a PredictionResult instance.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            names (dict): A dictionary of class names.
            boxes (DetectionBoxes, np.ndarray, List, Tuple, optional): Bounding box coordinates for each detection.
            frame_number (int): Frame number in the source video
            source_path (str): Path to source with file extension, will determine automatic detection of source type.
            source_name (str): Just a declaratory name of the source without a file extension. Used for naming output.
            visit_number (int): Number of the visit in the recording's Excel database.
            roi_number (int): ROI number when image is cropped into several parts.
            save_dir (str): Path to the directory where to save the output.

        Attributes:
            orig_img (numpy.ndarray): The original image as a numpy array.
            orig_shape (tuple): The original image shape in (height, width) format.
            frame_number (int): Frame number in the source video
            source_path (str): Path to source with file extension, will determine automatic detection of source type.
            source_name (str): Just a declaratory name of the source without a file extension. Used for naming output.
            source_type (str): "array", "video", "image", unknown" automatically determined when source_path is updated
            visit_number (int): Number of the visit in the recording's Excel database.
            roi_number (int): ROI number when image is cropped into several parts.
            reference_boxes (DetectionBoxes): Reference boxes object.
            filtered_boxes (DetectionBoxes): A property of filtered boxes based on their proximity to reference_boxes
            boxes (DetectionBoxes): A Boxes object containing the detection bounding boxes.
            names (dict): A dictionary of class names.
            save_dir (str): Path to the directory where to save the output.
            _keys (tuple): A tuple of attribute names for non-empty attributes.
        """

        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self._keys = ["boxes", "filtered_boxes"]

        if boxes is not None:
            if isinstance(boxes, DetectionBoxes):
                self.boxes = boxes
            else:
                try:
                    self.boxes = DetectionBoxes(np.array(boxes), self.orig_shape)
                except Exception as e:
                    logging.error(f"Error in creating DetectionBoxes: {str(e)}. Incorrect format of boxes.")
                    self.boxes = None
        else:
            self.boxes = None

        # Init additional attributes - can be initiated from kwargs
        self.names = None if 'names' not in kwargs else kwargs['names']
        self.frame_number = None if 'frame_number' not in kwargs else kwargs['frame_number']
        self._source_path = None if 'source_path' not in kwargs else kwargs['source_path']
        self._source_name = None if 'source_name' not in kwargs else kwargs['source_name']
        self.visit_number = None if 'visit_number' not in kwargs else kwargs['visit_number']
        self.roi_number = None if 'roi_number' not in kwargs else kwargs['roi_number']
        self.save_dir = os.getcwd() if 'save_dir' not in kwargs else kwargs['save_dir']
        self.reference_boxes = None
        self._filtered_boxes = None
        self._on_flowers = None
        self.source_type = determine_source_type(self.source_path)

        # Init attributes later populated if property method is called
        self._video_time = None  # Time of the frame in the source video (if source is video) in seconds
        self._real_time = None  # Time of the frame in real-life time or when the image was taken as a datetime object
        self._recording_id = None  # ID of the recording - e.g. CZ2_M1_AciArv01
        self._video_id = None  # ID of the video file - CZ2_M1_AciArv01_20210630_15_31

    @classmethod
    def from_results(cls, results: Results):

        if isinstance(results, Results):
            # Initialize from an instance of Results
            return cls(orig_img=results.orig_img,
                       source_path=results.path,
                       names=results.names,
                       boxes=(results.boxes.data if results.boxes.data.shape[1] < 7 else results.boxes.data[:, [0, 1, 2, 3, 5, 6, 4]]) if results.boxes is not None else None)

    @classmethod
    def from_prediction_results(cls, prediction_result):

        if isinstance(prediction_result, PredictionResult):
            boxes = [DetectionResults._convert_object_prediction_to_box_data(obj_pred) for obj_pred in
                     prediction_result.object_prediction_list]
            boxes = None if len(boxes) == 0 else np.array(boxes)

            # Initialize from a PredictionResult instance
            return cls(orig_img=np.array(prediction_result.image),
                       source_path=None,
                       names={obj_pred.category.id: obj_pred.category.name for obj_pred in
                              prediction_result.object_prediction_list},
                       boxes=boxes)

    @staticmethod
    def _convert_object_prediction_to_box_data(obj_pred: ObjectPrediction) -> List[float]:
        """
        Convert an ObjectPrediction instance to box data format.
        """
        bbox = obj_pred.bbox
        score = obj_pred.score.value
        category_id = obj_pred.category.id
        return [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, score, category_id]

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the DetectionResults object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes in self._keys and returns a new DetectionResults object with modified attributes. This
        function is internally called by methods like .new(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            DetectionResults: A new DetectionResults object with attributes modified by the applied function.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def _determine_save_folder(self, sort=False):
        if self.boxes is None or len(self.boxes.data) == 0:
            save_folder = os.path.join(self.save_dir, 'empty')
        else:
            save_folder = os.path.join(self.save_dir, 'object')
            if sort:
                conf_folder = str(np.floor(min(self.boxes.data[:, 4]) * 10) / 10)
                save_folder = os.path.join(save_folder, conf_folder)
        return save_folder

    @staticmethod
    def _get_video_file_instance(source_type, source_path):
        from detectflow.video.video_data import get_video_file_instance

        try:
            # Check if source type is 'video'
            if source_type != "video":
                raise ValueError("source is not set to 'video'")

            # Ensure that source_path is set and valid
            if not source_path or not isinstance(source_path, str):
                raise ValueError("source_path is not set or invalid.")

            # Create instance of Video
            return get_video_file_instance(source_path)

        except Exception as e:
            # Log the specific error for debugging
            raise RuntimeError(f"Error in analysing video source: {str(e)}")

    def update(self, boxes: Optional[np.ndarray] = None, reference_boxes: Optional[np.ndarray] = None):
        """Update the boxes and reference_boxes attribute of the DetectionResults object."""
        from detectflow.manipulators.box_manipulator import BoxManipulator

        if not (isinstance(boxes, np.ndarray) and isinstance(reference_boxes, np.ndarray)):
            raise ValueError("Both boxes and reference_boxes must be instances of np.ndarray.")

        if boxes is not None:
            self.boxes = DetectionBoxes(BoxManipulator.adjust_box_to_fit(boxes, self.orig_shape), self.orig_shape)
        if reference_boxes is not None:
            self.reference_boxes = DetectionBoxes(BoxManipulator.adjust_box_to_fit(reference_boxes, self.orig_shape), self.orig_shape)

    def new(self, boxes_type: str = 'boxes'):
        """
        Return a new DetectionResults object with the same image, path, and names and selected boxes.

        Args:
            boxes_type (str): Type of boxes to use ('boxes', 'reference_boxes', or 'filtered_boxes').

        """
        return DetectionResults(orig_img=self.orig_img, boxes=getattr(self, boxes_type), source_path=self.source_path, names=self.names)

    def plot(self,
             boxes: bool = True,
             reference_boxes: bool = True,
             filtered_boxes: bool = True,
             show: bool = False,
             save: bool = False):
        """
        Plot the original image with the specified boxes overlaid.

        Args:
            boxes (bool): Whether to plot the boxes.
            reference_boxes (bool): Whether to plot the reference boxes.
            filtered_boxes (bool): Whether to plot the filtered boxes.
            show (bool): Whether to display the plot.
            save (bool): Whether to save the plot.
        """
        from detectflow import Inspector

        config = {'boxes': boxes, 'reference_boxes': reference_boxes, 'filtered_boxes': filtered_boxes}

        try:
            boxes_to_plot = [getattr(self, b) for b in ['boxes', 'reference_boxes', 'filtered_boxes'] if config[b] and getattr(self, b) is not None]
            Inspector.display_frame_with_multiple_boxes(self.orig_img,
                                                        boxes_to_plot,
                                                        save_dir=self.save_dir,
                                                        filename=f"{self.source_name}_{self.frame_number}",
                                                        show=show,
                                                        save=save)
        except Exception as e:
            logging.error(f"Error in plotting DetectionResults: {e}")
            traceback.print_exc()

    def save(self, sort=False, assume_folder_exists=False, save_txt=False, box_type='boxes', extension: str = ".png"):
        if self.orig_img is None or not self.save_dir:
            raise ValueError("Image data or save directory not specified")

        save_folder = self._determine_save_folder(sort)

        if not assume_folder_exists:
            os.makedirs(save_folder, exist_ok=True)

        image_path = os.path.join(save_folder, f"{self.source_name}_{self.frame_number}{extension}")
        cv2.imwrite(image_path, cv2.cvtColor(self.orig_img, cv2.COLOR_RGB2BGR))

        if save_txt:
            self.save_txt(box_type=box_type, sort=sort, image_path=image_path)

    def save_txt(self, sort=False, assume_folder_exists=False, box_type='boxes', image_path=None, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            sort (bool): Sort the output based on confidence score.
            assume_folder_exists (bool): If True the folder checks are skipped, for optimization purposes.
            box_type (str): Type of boxes to use ('boxes', 'reference_boxes', or 'filtered_boxes').
            image_path (str | None): Path of the associated image. If provided, txt file will be saved alongside the image.
            save_conf (bool): Save confidence score or not.
        """
        boxes = getattr(self, box_type, None)
        texts = []

        # Put together the strings fo the boxes for the txt
        if boxes:
            for j, (d, cls, conf) in enumerate(zip(boxes.xywhn, boxes.cls, boxes.conf)):
                line = (cls, *d)
                texts.append(('%g ' * len(line)).rstrip() % line)

        # Determine the path for the txt file
        if not image_path:
            save_folder = self._determine_save_folder(sort)
            txt_path = os.path.join(save_folder, f"{self.source_name}_{self.frame_number}.txt")
        else:
            txt_path = os.path.splitext(image_path)[0] + '.txt'

        # Write to the txt file
        if texts:
            if not assume_folder_exists:
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'a') as f:
                f.writelines(text + '\n' for text in texts)

    @property
    def filtered_boxes(self):
        """
        Property to return filtered boxes as a DetectionBoxes instance.
        Filters self.boxes based on their proximity to self.reference_boxes.
        """
        if self._filtered_boxes is None:
            self._filtered_boxes = self.get_filtered_boxes()
        return self._filtered_boxes

    @filtered_boxes.setter
    def filtered_boxes(self, value: DetectionBoxes):
        self._filtered_boxes = value

    def get_filtered_boxes(self, radius_method=get_avg_box_max_dim):
        from detectflow.manipulators.box_manipulator import BoxManipulator

        try:
            if self.reference_boxes is None or self.boxes is None:
                raise AttributeError("Both self.reference_boxes and self.boxes must be set.")

            # Ensure that both self.boxes and self.reference_boxes have the 'xyxy' attribute
            if not hasattr(self.boxes, 'data') or not hasattr(self.reference_boxes, 'data'):
                raise AttributeError("Both self.boxes and self.reference_boxes must have the 'data' attribute.")

            # Assuming radius calculation and filtering logic remains the same
            radius = radius_method([self.boxes, self.reference_boxes])
            valid_pollinator_boxes = []
            for pb in self.boxes.data:
                for fb in self.reference_boxes.data:
                    if BoxManipulator.is_close(pb[:4], fb[:4], radius, min_or_max="min", metric="ee"):
                        valid_pollinator_boxes.append(pb)
                        break

            filtered_boxes = DetectionBoxes(np.array(valid_pollinator_boxes), self.orig_shape) if len(valid_pollinator_boxes) > 0 else None
        except Exception as e:
            logging.error(f"Error in calculating filtered boxes: {str(e)}")
            filtered_boxes = None

        return filtered_boxes

    @property
    def on_flowers(self):
        if self._on_flowers is None:
            self._on_flowers = self.get_on_flowers()
        return self._on_flowers

    def get_on_flowers(self):
        """
        Check for each box in self.boxes whether it overlaps with any box in self.reference_boxes.
        Returns a list of booleans indicating overlap with any reference_boxes.
        """
        from detectflow.manipulators.box_manipulator import BoxManipulator

        try:
            # Check if reference_boxes and boxes attributes are set and valid
            if self.reference_boxes is None or self.boxes is None:
                raise AttributeError("Both self.reference_boxes and self.boxes must be set.")

            # Ensure that both self.boxes and self.reference_boxes have the 'xyxy' attribute
            if not hasattr(self.boxes, 'xyxy') or not hasattr(self.reference_boxes, 'xyxy'):
                raise AttributeError("Both self.boxes and self.reference_boxes must have the 'xyxy' attribute.")

            overlaps = []
            for box in self.boxes.xyxy:
                overlap = any(BoxManipulator.is_overlap(box, ref_box, 0.1) for ref_box in self.reference_boxes.xyxy)
                overlaps.append(overlap)

            return overlaps

        except Exception as e:
            # You can handle specific exceptions more granularly if needed
            print(f"Error in calculating on_flowers: {str(e)}")
            return None

    @property
    def video_time(self):
        """
        Calculate the time of the frame in the source video in seconds based on frame_number and fps.
        """
        if self._video_time is None:
            self._video_time = self.get_video_time()
        return self._video_time

    @video_time.setter
    def video_time(self, value: int):
        self._video_time = value

    def get_video_time(self):

        if self.source_type != "video":
            logging.warning("Video time can only be calculated for video sources.")
            return None

        # Construct video file object instance
        try:
            video = self._get_video_file_instance(self.source_type, self.source_path)
        except RuntimeError as e:
            logging.error(f"Error in analysing video source: {str(e)}")
            return None

        # Time in video
        try:
            if self.frame_number is None:
                raise AttributeError("Frame Number is None, unable to calculate frame's time position.")

            fps = int(video.fps)

            # Calculate time in seconds
            time_in_seconds = self.frame_number / fps

        except Exception as e:
            # Log the specific error for debugging
            print(f"Error in calculating video_time: {str(e)}")
            time_in_seconds = None

        return time_in_seconds

    @property
    def real_time(self):
        if self._real_time is None:
            self._real_time = self.get_real_time()
        return self._real_time

    @real_time.setter
    def real_time(self, value: datetime):
        self._real_time = value

    def get_real_time(self):

        if self.source_type != "video":
            logging.warning("Video time can only be calculated for video sources.")
            return None

        # Construct video file object instance
        try:
            video = self._get_video_file_instance(self.source_type, self.source_path)
        except RuntimeError as e:
            logging.error(f"Error in analysing video source: {str(e)}")
            return None

        # Real Time
        try:
            real_time = video.start_time + timedelta(seconds=self.video_time if self.video_time is not None else 0)
        except Exception as e:
            try:
                # Convert the timestamp string to datetime object
                real_time = datetime.strptime(video.timestamp, "%Y%m%d_%H_%M")
            except Exception as e:
                logging.error(f"Error in calculating real_time: {str(e)}")
                real_time = None
        return real_time

    @property
    def recording_id(self):
        if self._recording_id is None:
            self._recording_id = self.get_recording_id()
        return self._recording_id

    @recording_id.setter
    def recording_id(self, value: str):
        self._recording_id = value

    def get_recording_id(self):
        if self.source_type != "video":
            logging.warning("Recording ID can only be calculated for video sources.")
            return None

        if not is_valid_video_id(self.source_name):
            logging.warning("Source name pattern not recognized. Source name treated as recording ID.")
            return self.source_name
        else:
            return parse_recording_name(self.source_path).get("recording_id", self.source_name)

    @property
    def video_id(self):
        if self._video_id is None:
            self._video_id = self.get_video_id()
        return self._video_id

    @video_id.setter
    def video_id(self, value: str):
        self._video_id = value
        self._source_name = value

    def get_video_id(self):
        if self.source_type != "video":
            logging.warning("Video ID can only be calculated for video sources.")
            return None

        if not is_valid_video_id(self.source_name):
            logging.warning("Source name pattern not recognized. Source name treated as video ID.")
            return self.source_name
        else:
            return parse_recording_name(self.source_path).get("video_id", self.source_name)

    @property
    def source_name(self):
        if self._source_name is None:
            self._source_name = os.path.splitext(os.path.basename(self.source_path))[0] if self.source_path else None
        return self._source_name

    @source_name.setter
    def source_name(self, value: str):
        self._source_name = value

    @property
    def source_path(self):
        return self._source_path

    @source_path.setter
    def source_path(self, value: str):
        """Setter for 'source_path' that updates the source_type when 'source_path' is updated."""
        self._source_path = value
        self.source_type = determine_source_type(self._source_path)

