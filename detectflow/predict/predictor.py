from typing import List, Union, Optional, Generator
import os
import logging
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import predict
from ultralytics.models.yolo import YOLO
from detectflow.predict.results import DetectionResults
from detectflow.manipulators.frame_manipulator import FrameManipulator
from detectflow.validators.validator import Validator
from ultralytics.engine.results import Results
from sahi.prediction import PredictionResult


class Predictor:
    YOLO_CONFIG_MAP = {
        'conf': float,
        'iou': float,
        'imgsz': (int, tuple),
        'half': bool,
        'device': (str, type(None)),
        'max_det': int,
        'vid_stride': bool,
        'stream_buffer': bool,
        'visualize': bool,
        'augment': bool,
        'agnostic_nms': bool,
        'classes': (list, type(None)),
        'retina_masks': bool,
        'embed': (list, type(None))
    }

    SAHI_CONFIG_MAP = {
        "model_type": str,
        "model_path": (str, type(None)),
        "model_config_path": (str, type(None)),
        "model_confidence_threshold": float,
        "model_device": (str, type(None)),
        "model_category_mapping": (dict, type(None)),
        "model_category_remapping": (dict, type(None)),
        "no_standard_prediction": bool,
        "no_sliced_prediction": bool,
        "image_size": (int, type(None)),
        "slice_height": int,
        "slice_width": int,
        "overlap_height_ratio": float,
        "overlap_width_ratio": float,
        "postprocess_type": str,
        "postprocess_match_metric": str,
        "postprocess_match_threshold": float,
        "postprocess_class_agnostic": bool,
        "novisual": bool,
        "view_video": bool,
        "frame_skip_interval": int,
        "export_pickle": bool,
        "export_crop": bool,
        "dataset_json_path": (bool, type(None)),
        "project": str,
        "name": str,
        "visual_bbox_thickness": (int, type(None)),
        "visual_text_size": (float, type(None)),
        "visual_text_thickness": (int, type(None)),
        "visual_hide_labels": bool,
        "visual_hide_conf": bool,
        "visual_export_format": str,
        "verbose": int,
        "return_dict": bool,
        "force_postprocess_type": bool
    }

    def __init__(self,
                 model_path: str = os.path.join('resources', 'yolo', 'best.pt'),
                 detection_conf_threshold: float = 0.5):

        # Assign attributes
        self.model_path = model_path
        self.detection_conf = detection_conf_threshold

    def detect(self,
               frame_numpy_array: Union[np.ndarray, List[np.ndarray]],
               metadata: dict = {},
               model_path: Optional[str] = None,
               detection_conf_threshold: Optional[float] = None,
               image_size: Optional[tuple] = None,
               sliced=True,
               save: bool = False,
               save_txt: bool = False,
               yolo_config: Optional[dict] = {},
               sahi_config: Optional[dict] = {},
               **kwargs):
        ''' Function that runs prediction using YOLO framework and YOLOv8 model.

            Args:
            -   Passed values override class attributes which are used as defaults. Any number of
                kwargs can be passed which should be arguments for configurating the YOLO model.predict()
                function. These args are validated to only include viable arguments. Alternatively a dictionary
                of arguments for yolo predict function can be passed as a named argument, same goes for sahi config
                arguments.

                Valid kwargs:
                    'source': str,
                    'conf': float,
                    'iou': float,
                    'imgsz': (int, tuple),
                    'half': bool,
                    'device': (type(None), str),
                    'max_det': int,
                    'vid_stride': bool,
                    'stream_buffer': bool,
                    'visualize': bool,
                    'augment': bool,
                    'agnostic_nms': bool,
                    'classes': (type(None), list),
                    'retina_masks': bool,
                    'embed': (type(None), list)
        '''

        # Get arguments, use class attributes as defaults
        model_path = model_path if model_path is not None else self.model_path
        conf = detection_conf_threshold if detection_conf_threshold is not None else self.detection_conf
        frame_height, frame_width = FrameManipulator.get_frame_dimensions(frame_numpy_array)
        image_size = image_size if image_size is not None else (frame_width, frame_height)

        # Prepare input for detection
        if isinstance(frame_numpy_array, np.ndarray):
            if frame_numpy_array.ndim == 3:
                # Convert the 3D frame into a 4D array with only one frame
                frame_numpy_array = np.expand_dims(frame_numpy_array, axis=0)

            # Flatten the 4D array into a list of 3D arrays
            list_of_frames = np.split(frame_numpy_array, frame_numpy_array.shape[0], axis=0)
            list_of_frames = [np.squeeze(frame) for frame in list_of_frames]  # Remove singleton dimension

        elif isinstance(frame_numpy_array, list) and all(isinstance(frame, np.ndarray) for frame in frame_numpy_array):
            list_of_frames = frame_numpy_array
        else:
            raise ValueError(f"Frames passed in an invalid format. Type: {type(frame_numpy_array)}")

        # Validate and fix kwargs dictionary and sort it into yolo and sahi arg dicts
        yolo_kwargs = {}
        sahi_kwargs = {}
        if kwargs:
            (yolo_kwargs, sahi_kwargs) = Validator.sort_and_validate_dict(kwargs, self.YOLO_CONFIG_MAP,
                                                                          self.SAHI_CONFIG_MAP)

        # Merged kwargs with config
        yolo_merged_config = {**yolo_kwargs, **yolo_config}
        sahi_merged_config = {**sahi_kwargs, **sahi_config}

        # Validate Metadata and fix missing values
        frame_numbers = metadata.get('frame_numbers', [i for i in range(1, len(frame_numpy_array))])
        if 'frame_numbers' not in metadata:
            logging.debug(f"(Predictor): Key 'frame_numbers' not found in metadata dictionary.")

        visit_numbers = metadata.get('visit_numbers', [0 for _ in frame_numpy_array])
        if 'visit_numbers' not in metadata:
            logging.debug(f"(Predictor): Key 'visit_numbers' not found in metadata dictionary.")

        roi_number = metadata.get('roi_number', 0)
        if 'roi_number' not in metadata:
            logging.debug(f"(Predictor): Key 'roi_number' not found in metadata dictionary.")

        # Check for available CUDA devices and assign value
        device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        logging.debug(f"(Predictor): Detected <{torch.cuda.device_count()}> CUDA devices.")

        # Run inference on the source
        if sliced:

            # Define SAHI settings
            model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                       model_path=model_path,
                                                       confidence_threshold=conf,
                                                       device=device,
                                                       load_at_init=True,
                                                       save_txt=save_txt,
                                                       save=save,
                                                       **yolo_merged_config)
            # **yolo_kwargs)
            logging.debug(f"(Predictor): YOLO SAHI model initiated")

            # Run prediction using SAHI
            results = predict(detection_model=model,
                              source=list_of_frames,
                              slice_height=640,
                              slice_width=640,
                              overlap_height_ratio=0.2,
                              overlap_width_ratio=0.2,
                              # TODO: Move to kwargs dictionary passed to the function when confirmed that it works
                              verbose=0,
                              **sahi_merged_config
                              );
        else:
            # Load a pretrained YOLOv8n model
            model = YOLO(model_path)
            logging.debug(f"(Predictor): YOLO model initiated")

            results = model(list_of_frames,
                            stream=True,
                            save_txt=save_txt,
                            save=save,
                            device=device,
                            imgsz=image_size,
                            conf=conf,
                            **yolo_merged_config)  # generator of Results objects

        # Process the generator
        for i, r in enumerate(results):
            frame_number = None if len(frame_numbers) < i + 1 else frame_numbers[i]
            visit_number = 0 if len(visit_numbers) < i + 1 else visit_numbers[i]

            # Process results
            detection_result = self._process_detection_results(r)
            detection_result.frame_number = frame_number if frame_number is not None else 0
            detection_result.visit_number = visit_number if visit_number is not None else 0
            detection_result.roi_number = roi_number if roi_number is not None else 0
            logging.info(f"(Predictor): YOLO result generated by the detect method")

            yield detection_result

    def _process_detection_results(self, result: Generator) -> DetectionResults:
        detection_result = None
        # Result exists
        if result is not None:
            try:

                if isinstance(result, Results):
                    detection_result = DetectionResults.from_results(result)
                elif isinstance(result, PredictionResult):
                    detection_result = DetectionResults.from_prediction_results(result)
                else:
                    detection_result = None

            except Exception as e:
                logging.error(f"(Predictor): Exception during result processing: {e}")
                raise

        return detection_result