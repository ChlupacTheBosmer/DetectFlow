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
from detectflow.manipulators.box_analyser import BoxAnalyser
from detectflow.predict.results import DetectionBoxes
from detectflow.utils.pdf_creator import PDFCreator, DiagPDFCreator
from detectflow.video.video_inter import VideoFileInteractive
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
                 video_path: str,
                 validate: bool = True,
                 flowers_model_path: str = os.path.join('/storage/brno2/home/USER/Flowers/flowers_ours_f2s/weights',
                                                        'best.pt'),
                 flowers_model_conf: float = 0.3,
                 motion_methods: Optional[Union[str, int, List, Tuple]] = "SOM",
                 frame_skip: int = 1,
                 color_variance_threshold: int = 10,
                 verbose: bool = True,
                 output_path: Optional[str] = None
                 ):

        try:
            # Set attributes
            self.video_path = video_path
            self.output_path = output_path
            self.validate = validate
            self.flowers_model_path = flowers_model_path
            self.flowers_model_conf = flowers_model_conf
            self.motion_methods = motion_methods
            self.frame_skip = frame_skip
            self.color_variance_threshold = color_variance_threshold
            self.verbose = verbose

            # Validate attributes, will raise error if something is wrong
            self._validate_attributes()

            # Define result data attributes
            self._rois = None
            self._motion_data = None
            self.frame_width = None
            self.frame_height = None
            self._daytime = None
            self.example_frames = None
            self._ref_bboxes = None
            self.validated_methods = None
            self._output_data = None

        except Exception as e:
            raise RuntimeError("ERROR: (Video Diagnoser) A critical error occurred during attribute initiation.") from e

        try:
            self._init_video_file()
            self._extract_info()
        except Exception as e:
            raise RuntimeError(
                "ERROR: (Video Diagnoser) A critical error occurred during video file data extraction.") from e

        # Validate video file
        if self.validate:
            if not self.validate_video():
                raise RuntimeError("ERROR: (Video Diagnoser) Video could not be valdiated with any method.")

    def _validate_attributes(self):
        '''
         Will validate the potentially wrong attributes. Will raise an error if something is wrong.
         Motion methods will be fixes if invalid.
        '''
        if not Validator.is_valid_file_path(self.video_path):
            raise ValueError(f"Invalid video file path: {self.video_path}")
        if self.output_path and not Validator.is_valid_directory_path(self.output_path):
            raise ValueError(f"Invalid output path: {self.output_path}")
        if self.motion_methods:
            self.motion_methods = validate_flags(self.motion_methods, self.METHOD_MAP, True)
        if not self.flowers_model_path or not Validator.is_valid_file_path(self.flowers_model_path):
            raise ValueError(f"Invalid model weights file path: {self.flowers_model_path}")

    def _init_video_file(self):
        # Initiate video file
        try:
            self.video_file = VideoFileInteractive(self.video_path, None, (0, 0))
        except Exception as e:
            raise RuntimeError(
                f"ERROR: (Video Diagnoser) Failed to initiate video file: {os.path.basename(self.video_path)}.") from e

    def _extract_info(self):
        # Extract basic info and assign attributes
        try:
            # File information
            self.file_extension = self.video_file.file_extension
            self.filename = self.video_file.filename
            self.video_origin = self.video_file.video_origin

            # Video information
            self.fps = self.video_file.fps
            self.total_frames = self.video_file.total_frames
            self.duration = timedelta(seconds=0) if self.fps == 0 else timedelta(
                seconds=self.total_frames / self.fps)  # timedelta

            # Recording information
            self.recording_identifier = self.video_file.recording_identifier
            self.video_identifier = os.path.splitext(os.path.basename(self.video_path))[0]

            # Time Information
            self.start_time = self.video_file.start_time  # datetime
            self.end_time = self.video_file.end_time  # datetime
        except Exception as e:
            raise (
                f"ERROR: (Video Diagnoser) Failed to extract information from video file: {os.path.basename(self.video_path)}.") from e

    def _check_attribute(self, attribute_name):

        # Check if the attribute exists and is not None
        return hasattr(self, attribute_name) and getattr(self, attribute_name) is not None

    def _check_frames(self):
        # Check if the self.example_frames attribute exists and is not empty
        if not hasattr(self, 'example_frames') or not self.example_frames:
            self._get_frames()
        # Further check if all elements in self.example_frames are NumPy arrays
        elif not ObjectDetectValidator.is_valid_ndarray_list(self.example_frames):
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

    def _get_frames(self):
        # Read 12 evenly distributed frames
        try:

            # Get frames
            self.example_frames = Sampler.sample_frames(self.video_path,
                                                        num_frames=12,
                                                        output_format='list',
                                                        distribution='even',
                                                        reader=self._choose_frame_reader())

            # Get frame dimensions
            if ObjectDetectValidator.is_valid_ndarray_list(self.example_frames):
                self.frame_height, self.frame_width, _ = self.example_frames[0].shape

        except Exception as e:
            logging.error(
                f"ERROR: (Video Diagnoser) Failed to extract frames from video file: {os.path.basename(self.video_path)}. Error: {e}")
            traceback.print_exc()

    def validate_video(self):
        # Run validator on video
        validator = VideoValidator(self.video_path, self.video_origin)
        result = validator.validate_video_readers()

        # List to hold the keys where the value is True
        self.validated_methods = []

        # Iterate over the dictionary and check each value
        for method, status in result.items():
            if status:
                self.validated_methods.append(method)

        if any(result.values()):
            logging.info("INFO: (Video Diagnoser) Video valdiated with at least one method.")
            return True
        elif all(result.values()):
            logging.info("INFO: (Video Diagnoser) Video valdiated with all methods.")
            return True
        else:
            logging.error("ERROR: (Video Diagnoser) Video could not be valdiated with any method.")
            return False

    def _is_nighttime(self, frame, threshold=10):

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the variance of the grayscale image
        color_variance = np.var(gray_frame)

        # If the color variance is low, it's likely a nighttime infrared video
        return color_variance < threshold

    def analyze_daytime(self, color_variance_threshold: Optional[int] = None):

        # Overide default attribute or get default
        color_variance_threshold = color_variance_threshold if color_variance_threshold is not None else self.color_variance_threshold

        # Check if the self.example_frames attribute exists
        self._check_frames()

        try:
            if ObjectDetectValidator.is_valid_ndarray_list(self.example_frames):
                if self._is_nighttime(self.example_frames[0], color_variance_threshold):
                    self._daytime = "Night"
                else:
                    self._daytime = "Day"
                return self._daytime
        except Exception as e:
            logging.error(
                f"ERROR: (Video Diagnoser) Failed to analyze whether video is day or night: {os.path.basename(self.video_path)}. Error: {e}")
            return None  # Or handle the error as needed

    @property
    def daytime(self):

        if self._check_attribute("_daytime"):

            # Return the attribute
            return self._daytime

        else:

            # Run the analysis method
            return self.analyze_daytime()

    @property
    def ref_bboxes(self):
        '''
        Will return the DetectionBoxes objects detected with the specified model in the first and last sampled frame.
        '''
        # Run the diagnostics again only if not run before
        if self._check_attribute("_ref_bboxes"):

            return self._ref_bboxes

        else:

            # Check if the self.example_frames attribute exists
            self._check_frames()

            # Check if the frames list is valid for processing
            if ObjectDetectValidator.is_valid_ndarray_list(self.example_frames):  # Assuming frames is a class attribute

                try:
                    # Select only the first and last frame for analysis, if multiple frames are present
                    frames_slice = [self.example_frames[0], self.example_frames[-1]] if len(
                        self.example_frames) > 1 else [self.example_frames[0]]

                    # Init list and dict
                    self._ref_bboxes = []

                    # Init Predictor
                    predictor = Predictor()

                    # Detect flowers in the sliced frames
                    for result in predictor.detect(frame_numpy_array=frames_slice,
                                                   model_path=self.flowers_model_path,
                                                   detection_conf_threshold=self.flowers_model_conf):
                        # Get boxes object from the result
                        flower_boxes = result.boxes

                        # Store detected boxes for further analysis
                        self._ref_bboxes.append(flower_boxes)

                    return self._ref_bboxes

                except Exception as e:
                    logging.error(
                        f"ERROR: (Video Diagnoser) Failed constructing reference boxes for {os.path.basename(self.video_path)}. Error: {e}")
                    traceback.print_exc()
                    return None

    @property
    def rois(self):

        # Run the diagnostics again only if not run before
        if not self._check_attribute("_rois"):
            # Get the boxes if not already available
            if not self._check_attribute("_ref_bboxes"):
                ref_bboxes = self.ref_bboxes
            else:
                ref_bboxes = self._ref_bboxes

            # Define ROIs from the first frame's detected flower boxes
            self._rois = ref_bboxes[0].xyxy if len(ref_bboxes) > 0 and isinstance(ref_bboxes[0],
                                                                                  DetectionBoxes) else None

        return self._rois

    def analyze_ref_bboxes(self):
        '''
        Wrapper function for analysing boxes of the first and the last frame
        '''
        labels = ("Start", "End")
        results = {}

        if not self._check_attribute("_ref_bboxes"):
            ref_bboxes = self.ref_bboxes
        else:
            ref_bboxes = self._ref_bboxes

        try:
            if ref_bboxes is not None:
                for boxes, label in zip(ref_bboxes, labels):
                    if isinstance(boxes, DetectionBoxes):
                        results[label] = BoxAnalyser.analyze_boxes(boxes)
                return results
            else:
                logging.error(f"Reference boxes analysis could not be performed: Reference boxes are not available.")
                return None
        except Exception as e:
            logging.error(f"Reference boxes analysis could not be performed: Error: {e}")
            traceback.print_exc()
            return None

    def analyze_motion_data(self,
                            motion_methods: Optional[Union[str, int, List, Tuple]] = None,
                            rois: Optional[Union[List, Tuple, np.ndarray]] = None,
                            flowers_model_path: Optional[str] = None,
                            flowers_model_conf: Optional[float] = None):

        if not self._check_attribute("_motion_data") or motion_methods or rois:

            # Use passed argument or the class attribute
            motion_methods = validate_flags(motion_methods, self.METHOD_MAP,True) if motion_methods is not None else self.motion_methods
            rois = rois if rois is not None else self.rois
            flowers_model_path = flowers_model_path if flowers_model_path is not None else self.flowers_model_path
            flowers_model_conf = flowers_model_conf if flowers_model_conf is not None else self.flowers_model_conf

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
                                                     rois_model_path=flowers_model_path,
                                                     rois_model_conf=flowers_model_conf,
                                                     visualize=False)

                    # Run analysis and retrieve data
                    self._motion_data = motion_detector.analyze()
                except Exception as e:
                    logging.error(f"Motion analysis could not be performed: Error: {e}")
                    traceback.print_exc()
                    return None

        return self._motion_data

    @property
    def motion_data(self):

        if not self._check_attribute("_motion_data"):
            return self.analyze_motion_data()
        else:
            return self._motion_data

    def pdf_report(self, output_path: Optional[str] = None):

        # Get and validate output path
        output_path = output_path if output_path is not None and Validator.is_valid_directory_path(
            self.output_path) else self.output_path

        if output_path:
            try:
                # Repack Output Data
                output_data = self._get_output_data()

                if output_data:
                    self._create_pdf(output_data, output_path)
                else:
                    raise ValueError(f"No output data available")
            except Exception as e:
                logging.error(f"No report generated: Error: {e}")
                traceback.print_exc()

    def report(self,
               flowers_model_path: Optional[str] = None,
               flowers_model_conf: Optional[float] = None,
               motion_methods: Optional[Union[str, List, Tuple]] = None,
               color_variance_threshold: Optional[int] = 10,
               verbose: Optional[bool] = None):

        # Combine arguments with initialized configuration, these override the config defined in init
        flowers_model_path = flowers_model_path if flowers_model_path is not None else self.flowers_model_path
        flowers_model_conf = flowers_model_conf if flowers_model_conf is not None else self.flowers_model_conf
        motion_methods = motion_methods if motion_methods is not None else self.motion_methods
        color_variance_threshold = color_variance_threshold if color_variance_threshold is not None else self.color_variance_threshold
        verbose = verbose if verbose is not None else self.verbose

        # Repack Output Data
        output_data = self._get_output_data(flowers_model_path=flowers_model_path,
                                            flowers_model_conf=flowers_model_conf,
                                            motion_methods=motion_methods,
                                            color_variance_threshold=color_variance_threshold)

        # Print output to the console
        if verbose:
            self._print_report()

        return output_data

    def _get_output_data(self,
                         flowers_model_path: str = os.path.join(
                             '/storage/brno2/home/USER/Flowers/flowers_ours_f2s/weights', 'best.pt'),
                         flowers_model_conf: float = 0.3,
                         motion_methods: Optional[Union[str, List, Tuple]] = None,
                         color_variance_threshold: int = 10):
        # Pack data
        output_data = {}
        try:
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
            output_data["roi_bboxes"] = self._ref_bboxes if self._check_attribute("_ref_bboxes") else self.ref_bboxes
            output_data["roi_data"] = self.analyze_ref_bboxes()
            output_data["basic_data"]["frame_width"] = self.frame_width
            output_data["basic_data"]["frame_height"] = self.frame_height
            output_data["motion_data"] = self._motion_data if self._check_attribute(
                "_motion_data") else self.analyze_motion_data(motion_methods=motion_methods,
                                                              flowers_model_path=flowers_model_path,
                                                              flowers_model_conf=flowers_model_conf)
            output_data["daytime"] = self._daytime if self._check_attribute("_daytime") else self.analyze_daytime(
                color_variance_threshold)
            output_data["frames"] = self.example_frames if self._check_attribute(
                "example_frames") else self._get_frames()
        except Exception as e:
            logging.error(f"ERROR: (Video Diagnoser) Error packing output data. Error: {e}")
            traceback.print_exc()

        # Assign to attribute, for caching
        self._output_data = output_data

        return output_data

    @property
    def report_data(self):

        if not self._check_attribute("_output_data"):
            return self._get_output_data()
        else:
            return self._output_data

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

        # Check that _ref_bboxes exists
        if not self._check_attribute("_ref_bboxes"):
            _ = self.ref_bboxes

        if self._check_attribute("_ref_bboxes") and self._check_attribute("example_frames"):
            try:
                frames = self.example_frames
                if not isinstance(frames, list):
                    raise TypeError("frames is not a list")

                # Create a list from DetectionBoxes objects
                flower_boxes = [boxes.xyxy.tolist() for boxes in self._ref_bboxes if isinstance(boxes, DetectionBoxes)]

                # Ensure flower_boxes is a list and has the expected structure
                if not isinstance(flower_boxes, list) or not all(isinstance(box, list) and box for box in flower_boxes):
                    raise ValueError("flower_boxes is not properly structured or empty")

                # plot the frames with bboxes
                if len(flower_boxes) > 0:
                    try:
                        detection_boxes_list = [self._ref_bboxes[0] for _ in range(len(frames) - 1)] + self._ref_bboxes[
                                                                                                       -1:]
                        Inspector.display_frames_with_boxes(frames, detection_boxes_list)
                    except AttributeError as ae:
                        logging.error(
                            f"ERROR: (Video Diagnoser) AttributeError: Missing or incorrect attributes in flower_boxes objects. {ae}")
                    except Exception as e:
                        logging.error(f"ERROR: (Video Diagnoser) An error occurred while plotting frames. Error: {e}")
            except KeyError:
                logging.error("ERROR: (Video Diagnoser) 'frames' key not found in output_data")
            except TypeError as te:
                logging.error(f"ERROR: (Video Diagnoser) TypeError: {te}")
            except ValueError as ve:
                logging.error(f"ERROR: (Video Diagnoser) ValueError: {ve}")
            except Exception as e:
                logging.error(f"ERROR: (Video Diagnoser) Unexpected error occurred. Error: {e}")

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
            except ValueError as ve:
                logging.error(f"ERROR: (Video Diagnoser) Value Error: {ve}")
            except Exception as e:
                logging.error(f"ERROR: (Video Diagnoser) An unexpected error occurred: {e}")
            print()

    def _print_motion_data(self):

        # Check attributes, it is safe to assume they will exist at this point, runs after data packing
        if self._check_attribute("_motion_data"):
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