from ultralytics.engine.results import Boxes, Results  # Import the Boxes class from module
from sahi.prediction import PredictionResult, ObjectPrediction
import copy
from typing import List, Optional
import numpy as np
from datetime import datetime, timedelta
import os
import cv2
from detectflow.video.video_inter import VideoFileInteractive
from detectflow.video.video_passive import VideoFilePassive


class DetectionBoxes(Boxes):
    def __init__(self, boxes, orig_shape, format_flag: str = 'boxes'):
        try:
            if isinstance(boxes, Boxes) and format_flag == 'boxes':
                orig_shape = boxes.orig_shape
                boxes = boxes.data.numpy()
                format_flag = "xyxypc"

            if isinstance(boxes, np.ndarray):
                boxes = boxes if boxes.ndim == 2 else [boxes]
            elif isinstance(boxes, (list, tuple)):
                boxes = boxes if isinstance(boxes[0], (list, tuple, np.ndarray)) else [boxes]
            else:
                raise TypeError("Boxes must be a numpy array, list, or tuple")

            processed_boxes = [self.process_box(box, orig_shape, format_flag) for box in boxes]
            array_boxes = np.array(processed_boxes)

            super().__init__(array_boxes, orig_shape)
        except Exception as e:
            raise RuntimeError(f"Error in DetectionBoxes initialization: {e}")

    @classmethod
    def from_boxes_instance(cls, boxes):

        if isinstance(boxes, Boxes):
            orig_shape = boxes.orig_shape
            boxes = boxes.data.numpy()
            format_flag = "xyxypc"

        return cls(boxes, orig_shape, format_flag)

    @classmethod
    def from_coco(cls, coco_boxes, orig_shape, format_flag: Optional[str] = None):
        '''
        Will init the class instance from an array of coco formated boxes [x_min, y_min, width, height] with any
        additional collumns with more data. Default format is class, probability, track_id, but other order can be specified
        by passing a format_flag - flag must start with "xyxy" as this will be passed on after conversion.
        '''
        from detectflow.manipulators.box_manipulator import BoxManipulator

        # Convert boxes
        xyxy_boxes = BoxManipulator.coco_to_xyxy(coco_boxes)

        # Default format flag
        format_flag = "xyxy" if format_flag is None else format_flag

        # Modify format flag if additional data is found and user has not defined custom flag
        if xyxy_boxes.shape[1] > 4 and format_flag is None:

            for col_index, char in zip(range(4, xyxy_boxes.shape[1]), "tpc"):
                format_flag = format_flag + char

        return cls(xyxy_boxes, orig_shape, format_flag)

    def process_box(self, box, orig_shape, format_flag):
        format_map = {'x': 0, 'y': 1, 'w': 2, 'h': 3, 't': 4, 'p': 5, 'c': 6}
        normalize = 'n' in format_flag
        format_flag = format_flag.replace("n", "")
        x_iteration = 0
        y_iteration = 0

        # Initialize new_box with the correct number of elements based on format_flag
        new_box = [0] * 7  # Initialize for all possible values

        # Process box based on format_flag
        for i, char in enumerate(format_flag):
            if char in 'xywhcpt':
                if char in 'xy':
                    # Multiple 'x' and 'y', handle 'xyxy' format
                    if char in 'x':
                        new_box[format_map[char] + (x_iteration * 2)] = float(box[i])
                        x_iteration = 1
                    else:
                        new_box[format_map[char] + (y_iteration * 2)] = float(box[i])
                        y_iteration = 1
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

        # Remove unused elements if format flag does not include 't'
        if 't' not in format_flag:
            # new_box = new_box[:-1]
            new_box = np.hstack((new_box[:4], new_box[5:]))

        return new_box

    def add_box(self, boxes, format_flag):

        # Ensure boxes are in a consistent format for processing
        if isinstance(boxes, np.ndarray):
            boxes = boxes if boxes.ndim == 2 else [boxes]
        elif isinstance(boxes, (list, tuple)):
            boxes = boxes if isinstance(boxes[0], (list, tuple, np.ndarray)) else [boxes]
        else:
            raise TypeError("Boxes must be a numpy array, list, or tuple")

        processed_boxes = [self.process_box(box, self.orig_shape, format_flag) for box in boxes]

        # Convert processed boxes to numpy array and stack with existing data
        new_boxes = np.array(processed_boxes)
        self.data = np.vstack([self.data, new_boxes])

    def to_list(self):
        # Convert boxes to a list of lists
        return self.data.tolist()

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

    def copy(self):
        #         # Creates a copy of the boxes
        #         return DetectionBoxes(self.data, self.orig_shape, "xyxycp")
        """
        Create a deep copy of the instance to ensure complete duplication of all mutable attributes.
        """
        return copy.deepcopy(self)

    def __iter__(self):
        # The __iter__ method returns the iterable object itself
        return iter(self.xyxy)

    def __getitem__(self, key):
        # Handle integer, slice, and list/array of indices
        if isinstance(key, int) or isinstance(key, slice):
            return self.xyxy[key]
        elif isinstance(key, (list, np.ndarray)):
            # If key is a list or NumPy array of indices
            return [self.xyxy[i] for i in key]
        else:
            raise TypeError("Index must be int, slice, or array of indices.")

    def __contains__(self, box):
        # Check if 'box' matches any row in 'self.xyxy'
        return any((self.xyxy == box).all(axis=1))


class DetectionResults(Results):
    def __init__(self, *args, **kwargs):
        """
        Initialize the DetectionResults either by passing the same arguments as the Results class,
        by passing an instance of the Results class, or by passing a PredictionResult instance.

        Args:

            Single argument:

                (Results, PredictionResult): Instance  will be initiated from this object instance.

            Multiple arguments:

                orig_img (numpy.ndarray): The original image as a numpy array.
                path (str): The path to the image file.
                names (dict): A dictionary of class names.
                boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
                masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
                probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
                keypoints (List[List[float]], optional): A list of detected keypoints for each object.
                frame_number (int): Frame number in the source video
                source_path (str): Path to source with file extension, will determine automatic detection of source type.
                source_name (str): Just a declaratory name of the source without a file extension. Used for naimng output.
                visit_number (int): Number of the visit in the recording's Excel database.
                roi_number (int): ROI number when image is cropped into several parts.

        Attributes:
            orig_img (numpy.ndarray): The original image as a numpy array.
            orig_shape (tuple): The original image shape in (height, width) format.
            frame_number (int): Frame number in the source video
            source_path (str): Path to source with file extension, will determine automatic detection of source type.
            source_name (str): Just a declaratory name of the source without a file extension. Used for naimng output.
            source_type (str): "array", "video", "image", unknown" automatically determined when source_path is updated
            visit_number (int): Number of the visit in the recording's Excel database.
            roi_number (int): ROI number when image is cropped into several parts.
            ref_boxes (DetectionBoxes): Reference boxes object.
            fil_boxes (DetectionBoxes): A property of filtered boxes based on their proximity to ref_boxes
            boxes (DetectionBoxes): A Boxes object containing the detection bounding boxes.
            masks (Masks, optional): A Masks object containing the detection masks.
            probs (Probs, optional): A Probs object containing probabilities of each class for classification task.
            keypoints (Keypoints, optional): A Keypoints object containing detected keypoints for each object.
            speed (dict): A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
            names (dict): A dictionary of class names.
            path (str): The path to the image file.
            _keys (tuple): A tuple of attribute names for non-empty attributes.
        """
        #         if len(args) == 1:
        #             if isinstance(args[0], Results):
        #                 # Initialize from an instance of Results
        #                 results_instance = args[0]
        #                 super().__init__(results_instance.orig_img, results_instance.path, results_instance.names,
        #                                  results_instance.boxes.data, results_instance.masks, results_instance.probs,
        #                                  results_instance.keypoints)
        #             elif isinstance(args[0], PredictionResult):
        #                 # Initialize from a PredictionResult instance
        #                 prediction_result = args[0]
        #                 self._init_from_prediction_result(prediction_result)
        #             else:
        #                 # Handle other cases or raise an error
        #                 raise TypeError("Invalid argument type for DetectionResults initialization")
        #         else:
        #             # Initialize with arguments for Results
        #             super().__init__(*args, **kwargs)


        super().__init__(*args, **kwargs)

        # Redefine boxes
        self.orig_shape = self.orig_img.shape[:2]
        self.boxes = None if self.boxes is None or len(self.boxes.data) == 0 else DetectionBoxes(
            np.array(self.boxes.data), self.orig_shape, 'xyxypc')

        # Init additional attributes - can be initiated from kwargs
        self.frame_number = None if not 'frame_number' in kwargs else kwargs['frame_number']
        self._source_path = None if not 'source_path' in kwargs else kwargs['source_path']
        self.source_name = None if not 'source_name' in kwargs else kwargs['source_name']
        self.visit_number = None if not 'visit_number' in kwargs else kwargs['visit_number']
        self.roi_number = None if not 'roi_number' in kwargs else kwargs['roi_number']
        self.ref_boxes = None
        self._fil_boxes = None
        self.source_type = determine_source_type(self.source_path)

        # Init attrributes later populated if property method is called
        self._video_time = None  # Time of the frame in the source video (if source is video) in seconds
        self._real_time = None  # Time of the frame in real life time or when the image was taken as a datetime object
        self._recording_id = None  # ID of the recording - e.g. CZ2_M1_AciArv01
        self._video_id = None  # ID of the video file - CZ2_M1_AciArv01_20210630_15_31
        self._real_start_time = None  # A datetime object that can be manually set if known to avoid getting the start time from video metadata

    @classmethod
    def from_results(cls, results):

        if isinstance(results, Results):
            # Initialize from an instance of Results
            return cls(orig_img=results.orig_img,
                       path=None,
                       names=results.names,
                       boxes=results.boxes.data if results.boxes is not None else None,
                       masks=results.masks,
                       probs=results.probs,
                       keypoints=results.keypoints)

    @classmethod
    def from_prediction_results(cls, prediction_result):

        if isinstance(prediction_result, PredictionResult):
            boxes = [DetectionResults._convert_object_prediction_to_box_data(obj_pred) for obj_pred in
                     prediction_result.object_prediction_list]
            boxes = None if len(boxes) == 0 else np.array(boxes)

            # Initialize from a PredictionResult instance
            return cls(orig_img=np.array(prediction_result.image),
                       path=None,
                       names={obj_pred.category.id: obj_pred.category.name for obj_pred in
                              prediction_result.object_prediction_list},
                       boxes=boxes,
                       masks=None,
                       probs=None,
                       keypoints=None)

    #     def _init_from_prediction_result(self, prediction_result: PredictionResult):
    #         """
    #         Initialize DetectionResults from a SAHI PredictionResult instance.
    #         """
    #         self.orig_img = np.array(prediction_result.image)
    #         self.orig_shape = self.orig_img.shape[:2]
    #         self.path = None  # Path is not available in PredictionResult
    #         self.names = {obj_pred.category.id: obj_pred.category.name for obj_pred in prediction_result.object_prediction_list}
    #         boxes_data = [self._convert_object_prediction_to_box_data(obj_pred) for obj_pred in prediction_result.object_prediction_list]
    #         self.boxes = None if len(boxes_data) == 0 else DetectionBoxes(np.array(boxes_data), self.orig_shape, 'xyxypc')
    #         # Initialize other attributes (masks, probs, keypoints) as None or appropriate default values
    #         self.masks = None
    #         self.probs = None
    #         self.keypoints = None

    @staticmethod
    def _convert_object_prediction_to_box_data(obj_pred: ObjectPrediction) -> List[float]:
        """
        Convert an ObjectPrediction instance to box data format.
        """
        bbox = obj_pred.bbox
        score = obj_pred.score.value
        category_id = obj_pred.category.id
        return [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, score, category_id]

    def set_ref_boxes(self, boxes, format_flag='xyxypc'):
        """
        Sets the ref_boxes attribute with the given boxes in DetectionBoxes format.

        :param boxes: The boxes data to set as ref_boxes.
        :param format_flag: The format flag for the DetectionBoxes initialization.
        """
        self.ref_boxes = DetectionBoxes(boxes, self.orig_shape, format_flag)

    def edge_within_radius(self, p_edge_min, p_edge_max, f_edge_min, f_edge_max, radius):
        """
        Checks if any part of a pollinator bbox edge is within a specified radius of a flower bbox edge.
        """
        # Extend the flower bbox edges by the radius
        f_min_extended = f_edge_min - radius
        f_max_extended = f_edge_max + radius

        # Check if any part of the pollinator edge is within the extended flower bbox edge
        return (f_min_extended <= p_edge_min <= f_max_extended) or \
            (f_min_extended <= p_edge_max <= f_max_extended) or \
            (p_edge_min <= f_min_extended <= p_edge_max) or \
            (p_edge_min <= f_max_extended <= p_edge_max)

    def is_bbox_close_to_flower(self, pollinator_bbox, flower_bbox, radius):
        """
        Checks if any part of the pollinator bbox is within a specified radius of the flower bbox.
        """
        # Unpack pollinator and flower bbox coordinates
        p_x_min, p_y_min, p_x_max, p_y_max = pollinator_bbox
        f_x_min, f_y_min, f_x_max, f_y_max = flower_bbox

        # Check if any edge of the pollinator bbox is within radius of the flower bbox
        return (self.edge_within_radius(p_x_min, p_x_max, f_x_min, f_x_max, radius) or
                self.edge_within_radius(p_y_min, p_y_max, f_y_min, f_y_max, radius))

    def calculate_dynamic_radius(self, flower_boxes, pollinator_boxes):
        # Calculate the average of the larger dimension (width or height) for flower boxes
        avg_flower_size = sum(max(fb[2], fb[3]) for fb in flower_boxes) / len(flower_boxes) if len(
            flower_boxes) > 0 else 0

        # Calculate the average of the larger dimension (width or height) for pollinator boxes
        avg_pollinator_size = sum(max(pb[2], pb[3]) for pb in pollinator_boxes) / len(pollinator_boxes) if len(
            pollinator_boxes) > 0 else 0

        # Calculate dynamic radius as average of these average sizes
        dynamic_radius = (avg_flower_size + avg_pollinator_size) / 2

        return dynamic_radius

    def _determine_save_folder(self, sort=False):
        if self.boxes is None or len(self.boxes.data) == 0:
            save_folder = os.path.join(self.save_dir, 'empty')
        else:
            save_folder = os.path.join(self.save_dir, 'object')
            if sort:
                conf_folder = str(np.floor(min(self.boxes.data[:, 4]) * 10) / 10)
                save_folder = os.path.join(save_folder, conf_folder)
        return save_folder

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
            box_type (str): Type of boxes to use ('boxes', 'ref_boxes', or 'fil_boxes').
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

    def save_crop(self, save_dir, box_type='boxes',
                  assume_folder_exists=False, square=False,
                  extension: str = ".png"):  # TODO: Fix, will not work as did not the save_txt
        """
        Save cropped predictions to `save_dir/crops/cls/self.source_name_self.frame_number_d.cls.*`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            box_type (str): Type of boxes to use ('boxes', 'ref_boxes', or 'fil_boxes').
            assume_folder_exists (bool): If False, check if the save directory exists and create if not.
            square (bool): If True, save crops as square.
            extension (str): .png or .jpg or other extension type
        """
        boxes = getattr(self, box_type, None)
        if not boxes:
            raise ValueError(f"Box type '{box_type}' not found in DetectionResults")

        crops_dir = os.path.join(save_dir, 'crops')
        if not assume_folder_exists:
            os.makedirs(crops_dir, exist_ok=True)

        for d in boxes:
            class_dir = os.path.join(crops_dir, self.names[int(d.cls)])
            os.makedirs(class_dir, exist_ok=True)
            save_one_box(d.xyxy, self.orig_img.copy(),
                         file=os.path.join(class_dir, f'{self.source_name}_{self.frame_number}_{d.cls}{extension}'),
                         square=square, BGR=True)

    @property
    def fil_boxes(self):
        """
        Property to return filtered boxes as a DetectionBoxes instance.
        Filters self.boxes based on their proximity to self.ref_boxes.
        """
        if self.ref_boxes is None or self.boxes is None:
            return None

        # Assuming radius calculation and filtering logic remains the same
        radius = self.calculate_dynamic_radius(self.ref_boxes.data, self.boxes.data)
        valid_pollinator_boxes = []
        for pb in self.boxes.data:
            for fb in self.ref_boxes.data:
                if self.is_bbox_close_to_flower(pb[:4], fb[:4], radius):
                    valid_pollinator_boxes.append(pb)
                    break

        self._fil_boxes = DetectionBoxes(np.array(valid_pollinator_boxes), self.orig_shape, 'xyxypc') if len(
            valid_pollinator_boxes) > 0 else None
        # print(self._fil_boxes)
        return self._fil_boxes

    @property
    def on_flowers(self):
        """
        Check for each box in self.boxes whether it overlaps with any box in self.ref_boxes.
        Returns a list of booleans indicating overlap with any ref_boxes.
        """
        from detectflow.manipulators.box_manipulator import BoxManipulator

        try:
            # Check if ref_boxes and boxes attributes are set and valid
            if self.ref_boxes is None or self.boxes is None:
                return None

            # Ensure that both self.boxes and self.ref_boxes have the 'xyxy' attribute
            if not hasattr(self.boxes, 'xyxy') or not hasattr(self.ref_boxes, 'xyxy'):
                raise AttributeError("Both self.boxes and self.ref_boxes must have the 'xyxy' attribute.")

            overlaps = []
            for box in self.boxes.xyxy:
                overlap = any(BoxManipulator.is_overlap(box, ref_box, 0.1) for ref_box in self.ref_boxes.xyxy)
                overlaps.append(overlap)

            return overlaps

        except Exception as e:
            # You can handle specific exceptions more granularly if needed
            print(f"Error in calculating on_flowers: {str(e)}")
            return None

    def _get_video_file_instance(self, inter: bool = False):

        try:
            # Check if source type is 'video'
            if self.source_type != "video":
                raise ValueError("source is not set to 'video'")

            # Ensure that source_path is set and valid
            if not self.source_path or not isinstance(self.source_path, str):
                raise ValueError("source_path is not set or invalid.")

            # Create instance of VideoFilePassive and retrieve fps
            return VideoFileInteractive(self.source_path,
                                        initiate_start_and_end_times=True) if inter else VideoFilePassive(
                self.source_path)

        except Exception as e:
            # Log the specific error for debugging
            raise RuntimeError(f"Error in analysing video source: {str(e)}")

    def _get_video_details_passive(self):
        '''
        Extract details from the video - does all at once so the passive video file is not
        created fir each property separately.
        '''

        # Construct video file object instance
        try:
            video = self._get_video_file_instance()
        except RuntimeError as e:
            print(e)
            return

        # Time in video
        if not hasattr(self, '_video_time') or self._video_time is None:
            try:
                # Check if fps attribute is available and valid
                if not hasattr(video, 'fps') or not video.fps:
                    raise AttributeError("Unable to retrieve 'fps' from the video file.")

                if self.frame_number is None:
                    raise AttributeError("Frame Number is None, unable to calculate frame's time position.")

                fps = int(video.fps)

                # Calculate time in seconds
                time_in_seconds = self.frame_number / fps

                # Set the attribute only if not set manually for example
                self._video_time = time_in_seconds if self._video_time is None else self._video_time

            except Exception as e:
                # Log the specific error for debugging
                print(f"Error in calculating video_time: {str(e)}")
                return

        if (not hasattr(self, '_recording_id') or self._recording_id is None) or \
                (not hasattr(self, '_video_id') or self._video_id is None):
            # Recording and Video IDs
            try:
                if not hasattr(video, 'recording_identifier') or not video.recording_identifier:
                    raise AttributeError("Unable to retrieve 'recording ID' from the video file.")

                recording_id = video.recording_identifier

                # Set the attribute only if not set manually for example
                self._recording_id = recording_id if self._recording_id is None else self._recording_id

                if not hasattr(video, 'timestamp') or not video.timestamp:
                    raise AttributeError("Unable to retrieve 'video ID' from the video file.")

                timestamp = video.timestamp

                # Set the attribute only if not set manually for example.
                self.video_id = f"{recording_id}_{timestamp}" if self._video_id is None else self._video_id  # Note that public attribute is assigned, setter property method will run.

            except AttributeError as e:
                # Log the specific error for debugging
                print(f"Error in extracting IDs: {str(e)}")
                return

    def _get_video_details_inter(self):
        '''
        Extract details from the video - does all at once so the active video file is not
        created fir each property separately.
        '''
        if not hasattr(self, '_real_start_time') or self._real_start_time is None:
            # Construct video file object instance
            try:
                video = self._get_video_file_instance(True)
            except RuntimeError as e:
                print(e)
                return

            # Real Time
            try:
                # Check if the start time attr is available
                if hasattr(video, 'start_time') and video.start_time:

                    start_time = video.start_time

                    pos_in_seconds = self.video_time if self.video_time is not None else 0

                    # Create a time delta object
                    time_difference = timedelta(seconds=pos_in_seconds)

                    real_time = start_time + time_difference

                else:

                    # Check if the timestamp attr is available
                    if not hasattr(video, 'timestamp') or not video.timestamp:
                        raise AttributeError("Unable to retrieve real time from the video file.")

                    timestamp = video.timestamp

                    # Convert the timestamp string to datetime object
                    real_time = datetime.strptime(timestamp, "%Y%m%d_%H_%M")

            except Exception as e:
                # Log the specific error for debugging
                print(f"Error in extracting real time: {str(e)}")
                return
        else:

            # Position of frame in video
            pos_in_seconds = self.video_time if self.video_time is not None else 0

            # Create a time delta object
            time_difference = timedelta(seconds=pos_in_seconds)

            # Adjust start time by delta
            real_time = self._real_start_time + time_difference

        # Set the attribute only if not set manually for example
        self._real_time = real_time if self._real_time is None else self._real_time

    @property
    def video_time(self):
        """
        Calculate the time of the video in seconds based on frame_number and fps.
        """

        if not hasattr(self, '_video_time') or self._video_time is None:
            self._get_video_details_passive()

        return self._video_time

    @property
    def real_time(self):

        if not hasattr(self, '_real_time') or self._real_time is None:

            if self.source_type == "video":
                self._get_video_details_inter()
            elif self.source_type == "image":
                raise NotImplementedError("This functionality is not yet implemented")
            else:
                pass

        return self._real_time

    @property
    def recording_id(self):

        if not hasattr(self, '_recording_id') or self._recording_id is None:
            self._get_video_details_passive()

        return self._recording_id

    @property
    def video_id(self):

        if not hasattr(self, '_video_id') or self._video_id is None:
            self._get_video_details_passive()

        return self._video_id

    @video_id.setter
    def video_id(self, new_value: str):
        self._video_id = new_value
        self.source_name = new_value

    @property
    def source_path(self):
        return self._source_path

    @source_path.setter
    def source_path(self, new_value: str):
        """Setter for 'source_path' that updates the source_type when 'source_path' is updated."""

        self._source_path = new_value
        self.source_type = determine_source_type(self.source_path)


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

def is_valid_url(path):

    import urllib.parse

    try:
        result = urllib.parse.urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False