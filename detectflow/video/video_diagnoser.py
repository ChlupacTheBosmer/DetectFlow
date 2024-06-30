from typing import Optional, List, Tuple, Union
import cv2
import numpy as np
import os
from datetime import timedelta
import logging
import traceback
from detectflow.validators.validator import Validator
from detectflow.utils.input import validate_flags
from detectflow.validators.object_detect_validator import ObjectDetectValidator
from detectflow.utils.sampler import Sampler
from detectflow.predict.predictor import Predictor
from detectflow.utils.inspector import Inspector
from detectflow.video.motion_detector import MotionDetector
from detectflow.validators.video_validator import VideoValidator
from detectflow import BoxManipulator
from detectflow.predict.results import DetectionBoxes
from detectflow.utils.pdf_creator import PDFCreator, DiagPDFCreator
from detectflow.video.video_data import Video
from detectflow.config import DETECTFLOW_DIR
from PIL import Image as PILImage


class VideoDiagnoser:
    # Define class level constants
    METHOD_MAP = {
        0: 'SOM',
        1: 'OF',
        2: 'BS',
        3: 'FM',
        4: 'TA'}

    def __init__(self,
                 video_path: Optional[str] = None,
                 video_file: Optional[Video] = None,
                 flowers_model_path: str = os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt'),
                 flowers_model_conf: float = 0.3,
                 motion_methods: Optional[Union[str, int, List, Tuple]] = "SOM",
                 frame_skip: int = 1,
                 brightness_threshold: int = 50,
                 saturation_threshold: int = 10,
                 verbose: bool = True,
                 output_path: Optional[str] = None,
                 **kwargs):

        if video_path is None and video_file is None:
            raise ValueError("Either 'video_path' or 'video_file' must be provided.")

        try:
            # Set attributes
            self.video_path = video_path if video_path else video_file.video_path
            self.output_path = output_path
            self.flowers_model_path = flowers_model_path
            self.flowers_model_conf = flowers_model_conf
            self.motion_methods = motion_methods
            self.frame_skip = frame_skip
            self.brightness_threshold = brightness_threshold
            self.saturation_threshold = saturation_threshold
            self.verbose = verbose

            # Validate attributes, will raise error if something is wrong
            self._validate_attributes()

            # Define result data attributes
            self._rois = None
            self._motion_data = None
            self.frame_width = None
            self.frame_height = None
            self._daytime = None
            self._frames = None
            self._ref_bboxes = None
            self._focus_regions = None
            self._focus_accuracies = None
            self._thumbnails = None
            self._report_data = None

        except Exception as e:
            raise RuntimeError("ERROR: (Video Diagnoser) A critical error occurred during attribute initiation.") from e

        # Initiate video file
        try:
            self.video_file = video_file if video_file else Video(self.video_path)
        except Exception as e:
            raise RuntimeError(
                f"ERROR: (Video Diagnoser) Failed to initiate video file: {os.path.basename(self.video_path)}.") from e

        # Extract basic info and assign attributes
        try:
            # File information
            self.file_extension = self.video_file.extension
            self.filename = self.video_file.video_name
            self.video_origin = "MS" if self.video_file.extension == ".avi" else "VT"

            # Video information
            self.fps = self.video_file.fps
            self.total_frames = self.video_file.total_frames
            self.duration = self.video_file.duration  # timedelta
            self.frame_height = self.video_file.frame_height
            self.frame_width = self.video_file.frame_width
            self.validated_methods = video_file.readers if video_file else None

            # Recording information
            self.recording_identifier = self.video_file.recording_id
            self.video_identifier = self.video_file.video_id

            # Time Information
            self.start_time = self.video_file.start_time  # datetime
            self.end_time = self.video_file.end_time  # datetime
        except Exception as e:
            raise (
                f"ERROR: (Video Diagnoser) Failed to extract information from video file: {os.path.basename(self.video_path)}.") from e

        # Validate video file
        if not video_file and not self._validate_video():
            raise RuntimeError("ERROR: (Video Diagnoser) Video could not be validated with any method.")

    def _validate_attributes(self):
        """
         Will validate the potentially wrong attributes. Will raise an error if something is wrong.
         Motion methods will be fixes if invalid.
        """
        if not Validator.is_valid_file_path(self.video_path):
            raise ValueError(f"Invalid video file path: {self.video_path}")
        if self.output_path and not Validator.is_valid_directory_path(self.output_path):
            try:
                os.makedirs(self.output_path, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Invalid output path: {self.output_path}. Error: {e}")
        if self.motion_methods:
            self.motion_methods = validate_flags(self.motion_methods, self.METHOD_MAP, True)
        if not self.flowers_model_path or not Validator.is_valid_file_path(self.flowers_model_path):
            raise ValueError(f"Invalid model weights file path: {self.flowers_model_path}")

    def _validate_video(self):
        try:
            result = VideoValidator.validate_video_readers(self.video_path)
        except Exception as e:
            logging.warning(f"INFO: (Video Diagnoser) Error during video validation: {e}.")
            result = {}

        self.validated_methods = [method for method, status in result.items() if status]

        if any(result.values()):
            logging.info("INFO: (Video Diagnoser) Video validated with at least one method.")
            return True
        elif all(result.values()):
            logging.info("INFO: (Video Diagnoser) Video validated with all methods.")
            return True
        else:
            logging.error("ERROR: (Video Diagnoser) Video could not be validated with any method.")
            return False

    def _check_frames(self):
        # Check if the self.example_frames attribute exists and is not empty and if all elements in self.example_frames are NumPy arrays
        if not hasattr(self, '_frames') or self._frames is None or not ObjectDetectValidator.is_valid_ndarray_list(self._frames):
            self._get_frames()

    def _choose_frame_reader(self):
        # Check if validated_methods attribute exists
        if hasattr(self, 'validated_methods'):
            validated_methods = self.validated_methods

            # Prioritize frame readers based on the validation results
            if "decord" in validated_methods:
                return "decord"
            elif "opencv" in validated_methods:
                return "opencv"
            elif "imageio" in validated_methods:
                return "imageio"
            else:
                # Fallback if none of the validated methods is available
                logging.info("No validated method available. Falling back to default method.")
        else:
            # If validated_methods does not exist, use the default reader
            logging.info("Validated methods not set. Using default frame reader.")
            return "decord"

    @property
    def frames(self):
        if self._frames is None:
            self._frames = self._get_frames()
        return self._frames

    def _get_frames(self):
        # Read 12 evenly distributed frames
        try:
            # Get frames
            frames = Sampler.sample_frames(self.video_path,
                                           num_frames=12,
                                           output_format='list',
                                           distribution='even',
                                           reader=self._choose_frame_reader())
        except Exception as e:
            logging.error(
                f"ERROR: (Video Diagnoser) Failed to extract frames from video file: {os.path.basename(self.video_path)}. Error: {e}")
            traceback.print_exc()
            frames = None
        return frames

    @property
    def daytime(self):
        if self._daytime is None:
            self._daytime = self.get_daytime()
        return self._daytime

    def get_daytime(self, brightness_threshold: Optional[int] = None, saturation_threshold: Optional[int] = None):

        # Override default attribute or get default
        brightness_threshold = brightness_threshold if brightness_threshold is not None else self.brightness_threshold
        saturation_threshold = saturation_threshold if saturation_threshold is not None else self.saturation_threshold

        # Check if the self.example_frames attribute exists
        self._check_frames()

        try:
            self._daytime = self.video_file.get_picture_quality(24).get_daytime(brightness_threshold, saturation_threshold)
        except Exception as e:
            logging.error(
                f"ERROR: (Video Diagnoser) Failed to analyze whether video is day or night: {os.path.basename(self.video_path)}. Error: {e}")
            self._daytime = None
        return self._daytime

    @property
    def ref_bboxes(self):
        '''
        Will return the DetectionBoxes objects detected with the specified model in the first and last sampled frame.
        '''
        # Run the diagnostics again only if not run before
        if not self._ref_bboxes:
            self._ref_bboxes = self.get_ref_bboxes()
        return self._ref_bboxes[0] if self._ref_bboxes and len(self._ref_bboxes) > 0 else None

    def get_ref_bboxes(self):
        # Check if the frames list is valid for processing
        if ObjectDetectValidator.is_valid_ndarray_list(self.frames):  # Assuming frames is a class attribute
            try:
                ref_bboxes = []

                # Init Predictor
                predictor = Predictor()

                # Detect flowers in the sliced frames
                for result in predictor.detect(frame_numpy_array=self.frames,
                                               model_path=self.flowers_model_path,
                                               detection_conf_threshold=self.flowers_model_conf,
                                               device='cpu'
                                               ):

                    # Store detected boxes for further analysis
                    ref_bboxes.append(result.boxes)
            except Exception as e:
                logging.error(
                    f"ERROR: (Video Diagnoser) Failed constructing reference boxes for {os.path.basename(self.video_path)}. Error: {e}")
                traceback.print_exc()
                ref_bboxes = None
        else:
            logging.warning("WARNING: (Video Diagnoser) Frames are not valid for processing.")
            ref_bboxes = None
        return ref_bboxes

    def analyze_ref_bboxes(self):
        '''
        Wrapper function for analysing boxes of the first and the last frame
        '''
        labels = ("Start", "End")
        results = {}
        ref_bboxes = [self._ref_bboxes[0], self._ref_bboxes[-1]] if len(self._ref_bboxes) > 0 else None

        try:
            if ref_bboxes is not None:
                for boxes, label in zip(ref_bboxes, labels):
                    if isinstance(boxes, DetectionBoxes):
                        results[label] = BoxManipulator.get_boxes_summary(boxes)
            else:
                logging.error(f"Reference boxes analysis could not be performed: Reference boxes are not available.")
                results = None
        except Exception as e:
            logging.error(f"Reference boxes analysis could not be performed: Error: {e}")
            traceback.print_exc()
            results = None
        return results

    @property
    def focus_regions(self):
        if not self._focus_regions:
            self._focus_regions = self.get_focus_regions()
        return self._focus_regions[0] if self._focus_regions is not None and len(self._focus_regions) > 0 else None

    def get_focus_regions(self):
        focus_regions = []
        try:
            for frame_number in Sampler.get_frame_numbers(self.total_frames, 12, 'even', 'list'):
                focus_regions.append(self.video_file.get_picture_quality(frame_number).focus_regions)
        except Exception as e:
            logging.error(f"Focus regions could not be extracted: Error: {e}")
            traceback.print_exc()
            focus_regions = None
        return focus_regions

    @property
    def focus_accuracy(self):
        if not self._focus_accuracies:
            self._focus_accuracies = self.get_focus_accuracies()
        return self._focus_accuracies[0] if self._focus_accuracies is not None and len(self._focus_accuracies) > 0 else None

    def get_focus_accuracies(self):

        if not self._ref_bboxes:
            _ = self.ref_bboxes

        if not self._focus_regions:
            _ = self.focus_regions

        focus_accuracies = []
        focus_regions = []
        if self._ref_bboxes:
            try:
                for frame_regions, boxes in zip(self._focus_regions, self._ref_bboxes):
                    focus_accuracy = BoxManipulator.get_coverage(boxes, frame_regions)
                    focus_accuracies.append(focus_accuracy)
            except Exception as e:
                logging.error(f"Focus accuracy could not be calculated: Error: {e}")
                traceback.print_exc()
                focus_accuracies = None
        return focus_accuracies

    @property
    def thumbnail(self):
        if not self._thumbnails:
            self._thumbnails = self.get_thumbnails()
        return self._thumbnails[0] if self._thumbnails is not None and len(self._thumbnails) > 0 else None

    def get_thumbnails(self):
        if not self._ref_bboxes:
            _ = self.ref_bboxes

        thumbnails = []
        try:
            for frame_number, boxes in zip(Sampler.get_frame_numbers(self.total_frames, 12, 'even', 'list'), self._ref_bboxes):
                _, focus_area = self.video_file.get_picture_quality(frame_number).get_focus()
                thumbnail = self.video_file.get_picture_quality(frame_number).get_focus_inspection(focus_area)
                if boxes is not None and isinstance(boxes, DetectionBoxes):
                    for (x1, y1, x2, y2) in boxes.xyxy:
                        cv2.rectangle(thumbnail, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                thumbnails.append(thumbnail)
        except Exception as e:
            logging.error(f"Thumbnails could not be extracted: Error: {e}")
            traceback.print_exc()
            thumbnails = None
        return thumbnails

    @property
    def motion_data(self):

        if not self._motion_data:
            self._motion_data = self.analyze_motion_data()
        return self._motion_data

    def analyze_motion_data(self,
                            motion_methods: Optional[Union[str, int, List, Tuple]] = None,
                            rois: Optional[Union[List, Tuple, np.ndarray, DetectionBoxes]] = None):

        motion_data = None
        if not self._motion_data or motion_methods or rois:

            # Use passed argument or the class attribute
            motion_methods = validate_flags(motion_methods, self.METHOD_MAP,True) if motion_methods is not None else self.motion_methods
            rois = rois if rois is not None else self.ref_bboxes

            # Only continue if any methods were specified
            if motion_methods:

                # Check if the self.example_frames attribute exists
                self._check_frames()

                # Check if validated_methods attribute exists
                if hasattr(self, 'validated_methods'):
                    validated_methods = self.validated_methods
                else:
                    validated_methods = None

                # Define frame reader method
                frame_reader_method = "decord" if "decord" in validated_methods else "imageio"

                try:
                    # Initialize the motion detector
                    motion_detector = MotionDetector(video_path=self.video_path,
                                                     methods=motion_methods,
                                                     frame_reader_method=frame_reader_method,
                                                     fps=self.fps,
                                                     smooth=True,
                                                     smooth_time=3,
                                                     frame_skip=self.frame_skip,
                                                     high_movement=True,
                                                     high_movement_thresh=None,
                                                     high_movement_time=2,
                                                     rois=rois,
                                                     rois_select="all",
                                                     visualize=False)

                    # Run analysis and retrieve data
                    motion_data = motion_detector.analyze()
                except Exception as e:
                    logging.error(f"Motion analysis could not be performed: Error: {e}")
                    traceback.print_exc()

        return motion_data

    @property
    def report_data(self):
        if not self._report_data:
            self._report_data = self._get_report_data()
        return self._report_data

    def _get_report_data(self,
                         flowers_model_path: str = os.path.join(
                             '/storage/brno2/home/USER/Flowers/flowers_ours_f2s/weights', 'best.pt'),
                         flowers_model_conf: float = 0.3,
                         motion_methods: Optional[Union[str, List, Tuple]] = None,
                         brightness_threshold: Optional[int] = 50,
                         saturation_threshold: Optional[int] = 10):
        # Pack data
        output_data = {}
        try:
            _ = self.ref_bboxes
            output_data["basic_data"] = {}
            output_data["basic_data"]["duration"] = self.duration
            output_data["basic_data"]["start_time"] = self.start_time
            output_data["basic_data"]["end_time"] = self.end_time
            output_data["basic_data"]["video_id"] = self.video_identifier
            output_data["basic_data"]["recording_id"] = self.recording_identifier
            output_data["basic_data"]["total_frames"] = self.total_frames
            output_data["basic_data"]["frame_rate"] = int(self.fps)
            output_data["basic_data"]["format"] = self.file_extension
            output_data["basic_data"]["video_origin"] = self.video_origin
            output_data["basic_data"]["validated_methods"] = self.validated_methods
            output_data["roi_bboxes"] = [self._ref_bboxes[0], self._ref_bboxes[-1]] if len(self._ref_bboxes) > 0 else [self.ref_bboxes, self.ref_bboxes]
            output_data["roi_data"] = self.analyze_ref_bboxes()
            output_data["basic_data"]["frame_width"] = self.frame_width
            output_data["basic_data"]["frame_height"] = self.frame_height
            output_data["motion_data"] = self._motion_data if self._motion_data else self.analyze_motion_data(motion_methods=motion_methods)
            output_data["daytime"] = self._daytime if self._daytime is not None else self.get_daytime(
                brightness_threshold, saturation_threshold)
            output_data["frames"] = self.frames if self.frames is not None else self._get_frames()
        except Exception as e:
            logging.error(f"ERROR: (Video Diagnoser) Error packing output data. Error: {e}")
            traceback.print_exc()

        return output_data

    def pdf_report(self, output_path: Optional[str] = None):

        # Get and validate output path
        output_path = output_path if output_path is not None and Validator.is_valid_directory_path(
            output_path) else self.output_path

        if output_path:
            try:
                # Repack Output Data
                report_data = self._get_report_data()

                if report_data:
                    self._create_pdf(report_data, output_path)
                else:
                    raise ValueError(f"No output data available")
            except Exception as e:
                logging.error(f"No report generated: Error: {e}")
                traceback.print_exc()

    def report(self,
               flowers_model_path: Optional[str] = None,
               flowers_model_conf: Optional[float] = None,
               motion_methods: Optional[Union[str, List, Tuple]] = None,
               brightness_threshold: Optional[int] = 50,
               saturation_threshold: Optional[int] = 10,
               verbose: Optional[bool] = None):

        # Combine arguments with initialized configuration, these override the config defined in init
        flowers_model_path = flowers_model_path if flowers_model_path is not None else self.flowers_model_path
        flowers_model_conf = flowers_model_conf if flowers_model_conf is not None else self.flowers_model_conf
        motion_methods = motion_methods if motion_methods is not None else self.motion_methods
        brightness_threshold = brightness_threshold if brightness_threshold is not None else self.brightness_threshold
        saturation_threshold = saturation_threshold if saturation_threshold is not None else self.saturation_threshold
        verbose = verbose if verbose is not None else self.verbose

        # Repack Output Data
        report_data = self._get_report_data(flowers_model_path=flowers_model_path,
                                            flowers_model_conf=flowers_model_conf, motion_methods=motion_methods,
                                            brightness_threshold=brightness_threshold,
                                            saturation_threshold=saturation_threshold)

        # Print output to the console
        if verbose:
            self._print_report()

        return report_data

    def _print_report(self):

        # Print basic information
        self._print_basic_data()

        # Print flower data and frame thumbnails
        self._print_flower_data()

        # Print motion data and plots
        self._print_motion_data()

    def _print_basic_data(self):
        try:
            # Ensure necessary attributes are present and of the correct type
            if not hasattr(self, 'duration') or not isinstance(self.duration, timedelta):
                raise AttributeError("Missing or invalid attribute 'duration'")
            if not hasattr(self, 'start_time') or not hasattr(self, 'end_time'):
                raise AttributeError("Missing or invalid attributes 'start_time' or 'end_time'")

            # Format duration
            duration = self.duration.total_seconds()
            duration_minutes = duration // 60
            duration_seconds = duration % 60
            duration_string = f"{int(duration_minutes):02d}:{int(duration_seconds):02d}"

            # Format the time as "HH:mm:ss"
            start_time_string = self.start_time.strftime("%H:%M:%S")
            end_time_string = self.end_time.strftime("%H:%M:%S")

            # Format the date as "DD.MM.YYYY"
            start_date_string = self.start_time.strftime("%d.%m.%Y")
            end_date_string = self.end_time.strftime("%d.%m.%Y")

            print("DIAGNOSTIC REPORT --------------------------------")
            print("Video ID:", self.video_identifier)
            print("Recording ID:", self.recording_identifier)
            print("Duration:", duration_string)
            print("Start Date:", start_date_string)
            print("Start Time:", start_time_string)
            print("End Date:", end_date_string)
            print("End Time:", end_time_string)
            print("Total Frames:", self.total_frames)
            print("Frame Rate:", self.fps)
            print("Format:", self.file_extension)
            print("Video Origin:", self.video_origin)
            print("Decord Validation:", "decord" in self.validated_methods)
            print("OpenCV Validation:", "opencv" in self.validated_methods)
            print("ImageIO Validation:", "imageio" in self.validated_methods)
            print("Frame Width:", self.frame_width)
            print("Frame Height:", self.frame_height)
            print("Daytime:", self.daytime)

        except (AttributeError, KeyError) as e:
            logging.error(f"Error: {e.__class__.__name__}: {e}")
            traceback.print_exc()
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()

    def _print_flower_data(self):

        # Flower data and frames
        print("Flower Data and Frames")

        # Check if the self.example_frames attribute exists
        self._check_frames()

        if self.ref_bboxes and self.frames:
            # plot the frames with bboxes
            if len(self._ref_bboxes) > 0:
                try:
                    Inspector.display_frames_with_boxes(self.frames, [self._ref_bboxes[i] if i < len(self._ref_bboxes) else self._ref_bboxes[-1] for i in range(len(self.frames))])
                except Exception as e:
                    logging.error(f"ERROR: (Video Diagnoser) An error occurred while plotting frames. Error: {e}")

            try:
                for key, values in self.analyze_ref_bboxes().items():
                    if not isinstance(values, dict):
                        raise ValueError(f"Expected a dictionary for the value of '{key}', but got {type(values)}")

                    print(f"{key} Data:")
                    for sub_key, sub_value in values.items():
                        if not isinstance(sub_key, str):
                            raise ValueError(f"Expected a string for the sub-key in '{key}', but got {type(sub_key)}")

                        name = ' '.join([word.capitalize() for word in sub_key.split('_')])
                        print(f"  {name}: {sub_value}")
                    print()
            except Exception as e:
                logging.error(f"ERROR: (Video Diagnoser) An unexpected error occurred: {e}")
            print()

    def _print_motion_data(self):

        # Check attributes, it is safe to assume they will exist at this point, runs after data packing
        if self._motion_data:
            # Motion data and plots
            print("Motion Detection Summary")
            try:
                motion_data = self._motion_data
                if not isinstance(motion_data, dict):
                    raise TypeError("motion_data is not a dictionary")

                if motion_data is not None:
                    for key, value in motion_data.items():
                        if not isinstance(value, dict):
                            raise TypeError(f"The value associated with key '{key}' in motion_data is not a dictionary")

                        # Check if the 'plot' key exists in the nested dictionary
                        if 'plot' in value:
                            try:
                                plot = value['plot']
                                if hasattr(plot, 'number') and callable(getattr(plot, 'show', None)):
                                    # Convert the plot to an image stream
                                    img_data = PDFCreator.create_plot_image(plot)

                                    # Convert BytesIO to PIL Image
                                    if img_data is not None:
                                        pil_img = PILImage.open(img_data)
                                        Inspector.display_images(pil_img)
                                else:
                                    raise TypeError(f"The 'plot' in key '{key}' is not a valid plot object")
                            except Exception as e:
                                logging.error(
                                    f"ERROR: (Video Diagnoser) Error processing plot for key '{key}'. Error: {e}")
            except TypeError as te:
                logging.error(f"ERROR: (Video Diagnoser) TypeError: {te}")
            except KeyError:
                logging.error("ERROR: (Video Diagnoser) 'motion_data' key not found in output_data")
            except Exception as e:
                logging.error(f"ERROR: (Video Diagnoser) Unexpected error occurred. Error: {e}")

    def _create_pdf(self, output_data, output_path):

        # Init PDF Creator
        try:
            pdf_creator = DiagPDFCreator(f"{self.video_identifier}_Diag_Report.pdf", output_path, output_data)
        except ImportError:
            logging.error("ERROR: (Video Diagnoser) PDF diagnostic report cannot be generated as the required packages are unavailable. Install DetectFlow with pip install detectflow[pdf].")
            return

        # Build the document
        success = pdf_creator.create_pdf()
        if success:
            logging.info(f"INFO: (Video Diagnoser) Build PDF successfully.")
        else:
            logging.error(f"ERROR: (Video Diagnoser) Error building PDF.")

