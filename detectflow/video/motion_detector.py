
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Tuple, Optional
import os
import random
import traceback
from detectflow.video.frame_reader import FrameReader
from detectflow.validators.input_validator import InputValidator
from detectflow.validators.object_detect_validator import ObjectDetectValidator
from detectflow.predict.predictor import Predictor
from detectflow.predict.results import DetectionBoxes
import logging
import imageio


class MotionDetector():
    """
    A class for detecting motion in video files using various algorithms.

    Attributes:
        video_path (str): Path to the video file for motion analysis.
        methods (list): List of methods to use for motion detection.
        frame_reader_method (str): Method to read video frames ('imageio' or 'decord').
        fps (int): Frames per second of the video.
        smooth (bool): Whether to apply smoothing to the motion data.
        smooth_time (int): Time in seconds over which to smooth the motion data.
        frame_skip (int): Number of frames to skip between analyses to speed up processing.
        high_movement (bool): Whether to detect high movement periods.
        high_movement_thresh (float, int): Threshold for detecting high movement.
        high_movement_time (int): Minimum duration of high movement to report.
        rois (list): Regions of interest within the video frames.
        rois_select (str): Method for selecting regions of interest ('random', 'all', 'cluster').
        rois_model_path (str): Path to the model for automatic ROI detection.
        rois_model_conf (float): Confidence threshold for the ROI model.
        visualize (bool): Whether to visualize the motion data.

    Methods:
        analyze(): Analyzes the video for motion using selected methods.
        load_frame_reader(): Loads the frame reader based on the specified method.
        detect_rois(): Detects regions of interest in the video automatically using a model.
        analyze_motion_data(motion_data, method_name): Analyzes the motion data to calculate statistics and detect high movement periods.
        calculate_smoothed_movements(movements): Applies smoothing to the motion data.
        calculate_threshold(movements): Calculates the threshold for detecting high movement.
        identify_high_movement_periods(movement_data, threshold, min_duration): Identifies periods of high movement.
        plot_motion_data(motion_data, method_name, mean_movement, smoothed_movements, high_movement_periods): Plots the motion data.
        get_video_frame_rate(): Retrieves the frame rate of the video.
        perform_motion_detection(): Performs motion detection using the specified methods.
    """

    # Define class level constants
    METHOD_MAP = {
        0: 'SOM',
        1: 'OF',
        2: 'BS',
        3: 'FM',
        4: 'TA'}

    ROIS_SELECT_MAP = {
        'random': 0,
        'all': 1,
        'cluster': 2}

    def __init__(self,
                 video_path: str,
                 methods: Union[Tuple, List, str, int] = 'TA',
                 frame_reader_method: str = "imageio",
                 fps: Optional[Union[float, int]] = None,
                 smooth: bool = True,
                 smooth_time: int = 3,
                 frame_skip: int = 1,
                 high_movement: bool = True,
                 high_movement_thresh: Optional[Union[int, float]] = None,
                 high_movement_time: int = 2,
                 rois: Optional[List] = None,
                 rois_select: str = "random",
                 rois_model_path: Optional[str] = None,
                 rois_model_conf: float = 0.3,
                 visualize: bool = True
                 ):
        """
        Initializes the MotionDetector with the specified parameters for video analysis.

        Args:
            video_path (str): Path to the video file.
            methods (Union[Tuple, List, str, int]): Detection methods to use, can be a list, tuple, single string, or index.
            frame_reader_method (str): Method to read frames ('imageio' or 'decord').
            fps (Optional[Union[float, int]]): Frames per second of the video; calculated if not provided.
            smooth (bool): Whether to apply smoothing to the motion data.
            smooth_time (int): Time in seconds for averaging the motion data.
            frame_skip (int): Number of frames to skip between detections.
            high_movement (bool): Flag to enable high movement detection.
            high_movement_thresh (Optional[Union[int, float]]): Threshold value for high movement detection.
            high_movement_time (int): Minimum duration in seconds to consider as high movement.
            rois (Optional[List]): List of regions of interest (ROIs).
            rois_select (str): Method for selecting ROIs ('random', 'all', 'cluster').
            rois_model_path (Optional[str]): Path to a model for detecting ROIs automatically.
            rois_model_conf (float): Confidence threshold for the ROI model.
            visualize (bool): Enable visualization of results.
        """

        # Assign attributes
        self.video_path = video_path
        self.fps = int(fps) if fps is not None else int(self.get_video_frame_rate())
        self.frame_reader_method = frame_reader_method
        self.smooth = smooth
        self.smooth_time = smooth_time
        self.frame_skip = frame_skip
        self.high_movement = high_movement
        self.high_movement_thresh = high_movement_thresh
        self.high_movement_time = high_movement_time
        self.rois_select = rois_select
        self.rois_model_path = rois_model_path
        self.rois_model_conf = rois_model_conf
        self.visualize = visualize

        # Validate and assign methods
        try:
            self.methods = InputValidator.validate_flags(methods, self.METHOD_MAP, True)
        except (AssertionError, TypeError):
            methods = ["TA"]

        #  Validate rois
        try:
            self.rois = ObjectDetectValidator.validate_rois_object(rois)
        except Exception as e:
            logging.error(f"ERROR: (Motion Detector): ROIs were passed in an incorect format - {rois} - {e}")
            tb_str = traceback.format_exc()
            print(tb_str)
            self.rois = None

    def analyze(self):
        """
        Analyzes the video for motion using the selected methods, detects high movement periods, and generates
        visual plots if enabled.

        Returns:
            dict: A dictionary containing the analysis results for each motion detection method used.
            Each key corresponds to a method and holds another dictionary with keys for raw data, mean,
            smoothed data, high movement frames, high movement periods, and plot (if visualization is enabled).
        """

        # Laod frame reader, if it fails, return None
        if not self.load_frame_reader():
            return None

        # Automatically detect rois using a YOLOv8 model
        print(self.rois)
        if self.rois is None:
            self.detect_rois()

        # Perform motion detection
        try:
            self.motion_data = self.perform_motion_detection()
        except Exception as e:
            logging.error(f"ERROR: (Motion Detector): Failed to calculate motion data - {e}")
            tb_str = traceback.format_exc()
            print(tb_str)
        finally:
            self.frame_reader.close()

        # Analyze the motion data and prepare output of the class
        if self.motion_data:

            # Prepare output data dictionary
            output_data = {key: {'raw_data': None,
                                 'mean': None,
                                 'smooth_data': None,
                                 'high_movement_periods_f': None,
                                 'high_movement_periods_t': None,
                                 'plot': None} for key in self.motion_data.keys()}

            for method in self.methods:

                # Calculate analytics and prepare plot
                try:
                    data = self.motion_data.get(method, [])
                except KeyError as key:
                    logging.error(
                        f"ERROR: (Motion Detector) KeyError: The key '{key}' was not found in the motion_data dictionary.")
                    data = []

                mean_movement, smoothed_movements, high_movement_frames, high_movement_periods = self.analyze_motion_data(
                    data, method)
                plot = self.plot_motion_data(data, method, mean_movement, smoothed_movements, high_movement_periods,
                                             self.frame_skip)

                # Populate output_data
                try:
                    output_data[method]['raw_data'] = data
                    output_data[method]['mean'] = mean_movement
                    output_data[method]['smooth_data'] = smoothed_movements
                    output_data[method]['high_movement_periods_f'] = high_movement_frames
                    output_data[method]['high_movement_periods_t'] = high_movement_periods
                    output_data[method]['plot'] = plot
                except KeyError as key:
                    logging.error(
                        f"ERROR: (Motion Detector) KeyError: The key '{key}' was not found in the output_data dictionary.")

            return output_data

    def load_frame_reader(self):
        """
        Initializes the frame reader based on the specified method. Updates the frame reader method to 'imageio'
        if an invalid method is provided.

        Returns:
            bool: True if the frame reader is successfully initialized, False otherwise.
        """

        # Load FrameReader
        try:
            if not (self.frame_reader_method == "imageio" or self.frame_reader_method == "decord"):
                logging.warning("WARN: (Motion Detector): Invalid frame reader method")
                self.frame_reader_method = "imageio"

            # Init Frame Reader
            self.frame_reader = FrameReader(self.video_path, self.frame_reader_method)
            return True

        except Exception as e:
            logging.error(f"ERROR: (Motion Detector): Failed to initiate frame reader - {e}")
            tb_str = traceback.format_exc()
            print(tb_str)
            return False

    def detect_rois(self):
        """
        Automatically detects regions of interest (ROIs) in the video using a specified model if `rois_model_path`
        is provided. This method is invoked if ROIs are not manually specified.

        Modifies:
            self.rois (list): Updates the regions of interest based on model predictions.
        """

        # Automatically detect rois
        if self.rois_model_path is not None:

            # Get frame size
            try:
                tmp_frame = self.frame_reader.get_frame(5)
                frame_height, frame_width, _ = tmp_frame.shape
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Failed to load frame and get its shape - {e}")
                tb_str = traceback.format_exc()
                print(tb_str)
                self.rois = None
                return

            # Get the position of flowers
            try:
                # Initiate predictor
                predictor = Predictor()

                # Detect flowers in the frame
                flower_boxes = []
                for result in predictor.detect(frame_numpy_array=tmp_frame,
                                               model_path=self.rois_model_path,
                                               detection_conf_threshold=self.rois_model_conf):
                    # Get boxes object from the result
                    flower_boxes = result.boxes

                if len(flower_boxes) > 0:
                    # Check if the first item is an instance of DetectionBoxes
                    if isinstance(flower_boxes[0], DetectionBoxes):
                        flower_rois = []
                        for box in flower_boxes[0].xyxy:
                            flower_rois.append(box)
                        self.rois = ObjectDetectValidator.validate_rois_object(flower_rois)
                    else:
                        self.rois = None
                else:
                    self.rois = None
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Failed to detect ROIs - {e}")
                tb_str = traceback.format_exc()
                print(tb_str)
                self.rois = None
            finally:
                # Inform user
                logging.info(f"INFO: (Motion Detector): Automatically detected ROIs - {self.rois}")

    def analyze_motion_data(self, motion_data, method_name):
        """
        Analyzes motion data to calculate mean movement, apply smoothing, and identify periods of high movement.

        Args:
            motion_data (list): List of motion values for each frame.
            method_name (str): Name of the motion detection method used.

        Returns:
            tuple: A tuple containing the mean movement, smoothed movement data, frames identified as high movement, and the periods of high movement.
        """

        # Example analysis function - to be adapted for each method

        if len(motion_data) > 0:
            # Calculate mean
            try:
                mean_movement = 0 if len(motion_data) == 0 else sum(motion_data) / len(motion_data)
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Failed to calculate motion data mean - {e}")
                tb_str = traceback.format_exc()
                print(tb_str)
                mean_movement = 0

            # Calculate smoothed motion data
            smoothed_movements = []
            if self.smooth:
                smoothed_movements = self.calculate_smoothed_movements(motion_data)

            # Calculate periods of high movement
            high_movement_periods = []
            if self.high_movement:
                movement_data = smoothed_movements if len(smoothed_movements) > 0 else motion_data
                threshold = self.high_movement_thresh if self.high_movement_thresh is not None else self.calculate_threshold(
                    smoothed_movements)
                min_duration = self.high_movement_time
                high_movement_frames, high_movement_periods = self.identify_high_movement_periods(movement_data,
                                                                                                  threshold,
                                                                                                  min_duration)
            return mean_movement, smoothed_movements, high_movement_frames, high_movement_periods
        else:
            return 0, [], [], [], []

    def calculate_smoothed_movements(self, movements):
        """
        Applies a smoothing function over the motion data using a moving average defined by the `smooth_time`.

        Args:
            movements (list): List of motion metrics to smooth.

        Returns:
            list: Smoothed motion metrics.
        """

        if len(movements) > 0:
            try:
                frame_rate = self.fps
                window_size = frame_rate * self.smooth_time
                smoothed_movements = np.convolve(movements, np.ones(window_size) / window_size, mode='valid')
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Failed to calculate smoothed motion data - {e}")
                print(traceback.format_exc())
                smoothed_movements = []
        return smoothed_movements

    def calculate_threshold(self, movements):
        """
        Calculates a threshold for detecting high movement based on the mean and standard deviation of
        the smoothed movements.

        Args:
            movements (list): Smoothed motion metrics.

        Returns:
            float: Calculated threshold for high movement detection.
        """

        # Implement threshold calculation
        try:
            threshold = np.mean(movements) + np.std(movements)
        except Exception as e:
            logging.error(
                f"ERROR: (Motion Detector): Failed to calculate high movement periods threshold. Defaulting to 0.4 - {e}")
            print(traceback.format_exc())
            threshold = 0.4
        return threshold

    def identify_high_movement_periods(self, movement_data, threshold, min_duration):
        """
        Identifies periods of high movement based on a given threshold and minimum duration.

        Args:
            movement_data (list): List of motion metrics for each frame.
            threshold (float): Threshold to classify high movement.
            min_duration (int): Minimum duration (in seconds) for a period to be considered significant high movement.

        Returns:
            tuple: A tuple containing two lists: one for the frames of high movement periods and another for the
            corresponding start and end times.
        """

        frame_rate = self.fps
        high_movement_frames = []
        high_movement_periods = []
        start_frame = None

        if len(movement_data) > 0:
            try:
                for i, amount in enumerate(movement_data):
                    # Check if the movement amount is above the threshold
                    if amount > threshold:
                        if start_frame is None:
                            start_frame = i * self.frame_skip  # Mark the start of a high movement period
                    else:
                        if start_frame is not None:
                            # Check if the duration meets the minimum required duration
                            end_frame = i * self.frame_skip
                            duration = (end_frame - start_frame) / frame_rate
                            if duration >= min_duration:
                                start_time = start_frame / frame_rate
                                end_time = end_frame / frame_rate
                                high_movement_frames.append((start_frame, end_frame))
                                high_movement_periods.append((start_time, end_time))
                            start_frame = None  # Reset the start frame for the next period

                # Check for any ongoing high movement period at the end of the data
                if start_frame is not None:
                    end_frame = len(movement_data) - 1
                    duration = (end_frame - start_frame) / frame_rate
                    if duration >= min_duration:
                        start_time = start_frame / frame_rate
                        end_time = end_frame / frame_rate
                        high_movement_frames.append((start_frame, end_frame))
                        high_movement_periods.append((start_time, end_time))
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Failed to calculate high movement periods - {e}")
                print(traceback.format_exc())

        return high_movement_frames, high_movement_periods

    def plot_motion_data(self, motion_data, method_name, mean_movement, smoothed_movements, high_movement_periods, n):
        """
        Generates a plot of the motion data along with the mean, smoothed data, and periods of high movement if
        visualization is enabled.

        Args:
            motion_data (list): Raw motion data for each frame.
            method_name (str): Name of the motion detection method used.
            mean_movement (float): Mean of the motion data.
            smoothed_movements (list): Smoothed motion data.
            high_movement_periods (list): List of tuples indicating the start and end times of high movement periods.
            n (int): Sampling rate, the number of frames between each analyzed frame.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object with the plotted data, or None if an error
            occurs during plotting.
        """
        # Ensure interactive mode is off
        plt.ioff()  # This turns off interactive plotting

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            # Get frame rate
            frame_rate = self.fps

            # Generate the frame indices for x-axis
            frame_indices = range(0, n * len(motion_data), n)

            # Plot the raw movement
            ax.plot(frame_indices, motion_data, label='Movement Amount', color='mediumaquamarine', linewidth=1)

            # Plot the smooth movement
            if self.smooth:
                window_size = frame_rate * self.smooth_time
                smoothed_frame_indices = range(n * (window_size // 2), n * (len(smoothed_movements) + window_size // 2),
                                               n)
                ax.plot(smoothed_frame_indices, smoothed_movements,
                        color='orange', linewidth=2, label=f'Smoothed Movement ({self.smooth_time}s average)')

            # Highlight periods of high movement on the x-axis
            if self.high_movement:
                for start_time, end_time in high_movement_periods:
                    ax.axvspan(start_time * frame_rate, end_time * frame_rate, color='gold', alpha=0.3)

            # Plot the mean movement line
            ax.axhline(y=mean_movement, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_movement:.4f}')

            # Labels and titles
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Motion Amount')
            ax.set_title(f'{method_name} - {os.path.basename(self.video_path)}')
            ax.legend()

            if self.visualize:
                plt.tight_layout()
                plt.show()
            else:
                plt.close(fig)  # Close the figure to prevent it from displaying in the notebook

        except Exception as e:
            logging.error(f"ERROR: (Motion Detector): Failed to prepare data plot - {e}")
            print(traceback.format_exc())
            fig = None

        return fig

    def get_video_frame_rate(self):
        """
        Retrieves the frame rate of the video either using cv2 or imageio depending on which method is successful.

        Returns:
            float: The frame rate of the video, or None if both methods fail.
        """

        video_path = self.video_path

        # First, try using cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps

        # If cv2 fails, use imageio
        try:
            with imageio.get_reader(video_path) as video:
                fps = video.get_meta_data()['fps']
                return fps
        except Exception as e:
            logging.error(f"Failed to get frame rate using both cv2 and imageio: {e}")
            print(traceback.format_exc())

        return None

    # Perform motion detection based on the selected methods
    def perform_motion_detection(self):
        """
        Executes motion detection across the video using the specified methods. It handles the selection and
        processing of each frame and region of interest (ROI), applying different motion detection algorithms.

        Returns:
            dict: A dictionary where each key is a motion detection method (e.g., 'SOM', 'OF') and the value is a
            list of motion metrics calculated for each frame.
        """

        motion_data = {}
        frame_number = 0

        # Define selected roi
        if self.rois is not None and isinstance(self.rois, list):
            # TODO: roi selection should be implemented
            selected_roi = random.choice(self.rois).astype(int)

            # Convert roi to x,y,w,h
            selected_roi = self.convert_coordinates(selected_roi)
        else:
            tmp_frame = self.frame_reader.get_frame(5)
            frame_height, frame_width, _ = tmp_frame.shape
            selected_roi = (0, 0, int(frame_width), int(frame_height))

        # Initialize data structures for each motion detection method
        if 'SOM' in self.methods:
            motion_data['SOM'] = []
        if 'OF' in self.methods:
            motion_data['OF'] = []
        if 'BS' in self.methods:
            motion_data['BS'] = []
        if 'FM' in self.methods:
            motion_data['FM'] = []
        if 'TA' in self.methods:
            motion_data['TA'] = []

        previous_frame = None
        while True:
            try:
                frame = self.frame_reader.get_frame(frame_number)
                # print("Read frame")
                if frame is None:
                    break

                if 'SOM' in self.methods:
                    som_metric = self.detect_motion_som(selected_roi, previous_frame, frame)
                    motion_data['SOM'].append(som_metric)

                if 'OF' in self.methods:
                    of_metric = self.detect_motion_of(selected_roi, previous_frame, frame,
                                                      frame_number - self.frame_skip, frame_number)
                    motion_data['OF'].append(of_metric)

                if 'BS' in self.methods:
                    bs_metric = self.detect_motion_bs(selected_roi, frame, frame_number)
                    motion_data['BS'].append(bs_metric)

                if 'FM' in self.methods:
                    fm_metric = self.detect_motion_fm(selected_roi, frame, frame_number)
                    motion_data['FM'].append(fm_metric)

                if 'TA' in self.methods:
                    ta_metric = self.detect_motion_ta(selected_roi, frame, frame_number)
                    motion_data['TA'].append(ta_metric)

                frame_number += self.frame_skip
                previous_frame = frame
            except Exception as e:
                logging.error(f"ERROR: (Motion Detector): Error during motion data calculation - {e}")
                print(traceback.format_exc())

        return motion_data

    def detect_motion_som(self, roi, frame1, frame2):
        """
        Detects motion using the Sum of Absolute Differences method between two consecutive frames
        within the specified region of interest (ROI).

        Args:
            roi (tuple): The region of interest in the format (x, y, width, height).
            frame1 (ndarray): The first frame to compare.
            frame2 (ndarray): The second frame to compare.

        Returns:
            float: Normalized motion metric based on the sum of absolute differences in the ROI.
        """

        # Check if both frames are provided
        if frame1 is None or frame2 is None:
            return 0

        # Crop the frames to the ROI
        roi_frame1 = frame1[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        roi_frame2 = frame2[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # Convert to grayscale and compute the absolute difference between frames
        gray1 = cv2.cvtColor(roi_frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi_frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)

        # Apply Gaussian blur, threshold, and morphological operations
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours and filter out large ones
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        small_movement_amount = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) < 500)

        # Normalize the movement amount by the area of the ROI
        roi_area = roi[2] * roi[3]
        normalized_movement_amount = small_movement_amount / roi_area

        return normalized_movement_amount

    def detect_motion_of(self, roi, old_frame, new_frame, old_frame_number, new_frame_number):
        """
        Detects motion using Optical Flow method (Lucas-Kanade) between two frames. Points of interest are
        tracked from the old frame to the new frame within the specified ROI.

        Args:
            roi (tuple): The region of interest in the format (x, y, width, height).
            old_frame (ndarray): The previous frame for motion detection.
            new_frame (ndarray): The current frame for motion detection.
            old_frame_number (int): The frame number of the old frame.
            new_frame_number (int): The frame number of the new frame.

        Returns:
            float: Normalized motion metric calculated from the optical flow vectors.
        """

        # Check if both frames are provided
        if old_frame is None or new_frame is None:
            return 0

        # Define feature_params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Define lk_params for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Create mask for ROI
        roi_mask = np.zeros_like(old_gray)
        roi_mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1

        # Detect features in the old frame
        if old_frame_number == 0:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)
        else:
            p0 = self.previous_points_of

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points and calculate motion
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        motion_metric = np.mean(np.linalg.norm(good_new - good_old, axis=1))

        # Normalize by ROI area
        normalized_motion_metric = motion_metric / (roi[2] * roi[3])

        # Store the points for the next frame
        self.previous_points_of = good_new.reshape(-1, 1, 2)

        return normalized_motion_metric

    def detect_motion_bs(self, roi, frame, frame_number):
        """
        Detects motion using a background subtraction algorithm (MOG2) on the specified frame within the ROI.

        Args:
            roi (tuple): The region of interest in the format (x, y, width, height).
            frame (ndarray): The current frame on which to perform motion detection.
            frame_number (int): The frame number of the current frame.

        Returns:
            float: Normalized motion metric based on the area of the detected foreground objects.
        """
        # Initialize the background subtractor if it's the first frame
        if frame_number == 0:
            self.backSub = cv2.createBackgroundSubtractorMOG2()

        # Apply background subtraction
        fgMask = self.backSub.apply(frame)

        # Consider only the ROI
        fgMask_roi = fgMask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        contours, _ = cv2.findContours(fgMask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate motion metric as the sum of contour areas
        motion_metric = sum(cv2.contourArea(contour) for contour in contours)

        # Normalize by ROI area
        normalized_motion_metric = motion_metric / (roi[2] * roi[3])
        return normalized_motion_metric

    def detect_motion_fm(self, roi, frame, frame_number):
        """
        Detects motion using feature matching (SIFT and FLANN) between the current frame and a previously
        stored frame within the ROI.

        Args:
            roi (tuple): The region of interest in the format (x, y, width, height).
            frame (ndarray): The current frame for motion detection.
            frame_number (int): The frame number of the current frame.

        Returns:
            float: Normalized motion metric based on the number of good feature matches.
        """

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass an empty dictionary

        # Initialize FLANN based matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # Initialize keypoints and descriptors for the first frame
        if frame_number == 0:
            kp1, des1 = sift.detectAndCompute(frame_gray, None)
            self.previous_kp_fm = kp1
            self.previous_des_fm = des1
            return 0  # No motion metric for the first frame

        kp2, des2 = sift.detectAndCompute(frame_gray, None)

        # Matching descriptor using KNN algorithm
        matches = flann.knnMatch(self.previous_des_fm, des2, k=2)

        # Store all the good matches as per Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Calculate motion metric as the number of good matches
        motion_metric = len(good)

        # Normalize by ROI area
        normalized_motion_metric = motion_metric / (roi[2] * roi[3])

        # Update keypoints and descriptors for the next frame
        self.previous_kp_fm = kp2
        self.previous_des_fm = des2

        return normalized_motion_metric

    def detect_motion_ta(self, roi, frame, frame_number, avg_over_frames=30):
        """
        Detects motion by comparing the current frame to an averaged historical frame within the specified ROI,
        using a threshold on the absolute frame difference.

        Args:
            roi (tuple): The region of interest in the format (x, y, width, height).
            frame (ndarray): The current frame for motion detection.
            frame_number (int): The frame number of the current frame.
            avg_over_frames (int): Number of frames to average for the baseline comparison.

        Returns:
            float: Normalized motion metric based on the area of differences detected.
        """

        roi_area = roi[2] * roi[3]

        # Initialize an accumulator for the frames if it's the first frame
        if frame_number == 0:
            self.frames_accumulator_ta = np.zeros((roi[3], roi[2]), dtype=np.float32)
            self.frame_counter_ta = 0

        # Crop to ROI and convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # Accumulate frames for the initial average calculation
        if self.frame_counter_ta < avg_over_frames:
            self.frames_accumulator_ta += gray_frame.astype(np.float32)
            self.frame_counter_ta += 1

            # Return 0 as motion metric until the initial average frame is established
            if self.frame_counter_ta < avg_over_frames:
                return 0

            # Calculate the initial average frame after enough frames are accumulated
            if self.frame_counter_ta == avg_over_frames:
                self.avg_frame_ta = self.frames_accumulator_ta / avg_over_frames

        # Calculate the absolute difference between the current frame and the average frame
        frame_diff = cv2.absdiff(gray_frame.astype(np.float32), self.avg_frame_ta)

        # Threshold the difference to get binary motion mask
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate the sum of the motion mask to get an idea of how much motion there is
        motion_metric = np.sum(motion_mask) / 255

        # Normalize the motion metric by the area of the ROI
        normalized_motion_metric = motion_metric / roi_area

        # Update the average frame with the current frame
        self.avg_frame_ta = ((self.avg_frame_ta * self.frame_counter_ta) + gray_frame.astype(np.float32)) / (
                    self.frame_counter_ta + 1)
        self.frame_counter_ta += 1

        return normalized_motion_metric

    def convert_coordinates(self, coords):
        """
        Converts a list or array of coordinates from (x1, y1, x2, y2) format to (x, y, width, height).

        Args:
            coords (list or ndarray): Coordinates to be converted.

        Returns:
            ndarray: Converted coordinates in the format (x, y, width, height).
        """
        # Convert the list to a numpy array if it isn't already
        coords = np.array(coords)

        # Calculate width and height
        w = coords[2] - coords[0]
        h = coords[3] - coords[1]

        # Construct the new coordinates array (x, y, w, h)
        new_coords = np.array([coords[0], coords[1], w, h])

        return new_coords

