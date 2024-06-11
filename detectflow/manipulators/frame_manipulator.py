import numpy as np
import cv2
import random
from typing import List, Tuple, Union, Optional, Dict
from numbers import Real
import logging
import os


class FrameManipulator:
    def __init__(self):
        pass

    @staticmethod
    def get_frame_dimensions(frames):
        frame_height, frame_width = None, None
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                # Single 3D frame (height, width, channels)
                frame_height, frame_width, _ = frames.shape
            elif frames.ndim == 4:
                # 4D array (batch of frames)
                _, frame_height, frame_width, _ = frames.shape
            elif frames.ndim == 2:
                # Single 2D frame (height, width for grayscale)
                frame_height, frame_width = frames.shape
                frame_width, frame_height = 1, frame_height  # Adjust for compatibility if needed
        elif isinstance(frames, list) and frames:
            # Handle a list of frames; assume all frames have the same dimensions
            first_frame = frames[0]
            if isinstance(first_frame, np.ndarray):
                if first_frame.ndim == 3:
                    frame_height, frame_width, _ = first_frame.shape
                elif first_frame.ndim == 2:
                    frame_height, frame_width = first_frame.shape
                    frame_width, frame_height = 1, frame_height  # Adjust for compatibility if needed
            else:
                raise ValueError(f"Frames in the list must be numpy arrays. Check the data format. Type: {type(first_frame)}")
        else:
            raise ValueError(f"Unsupported data format or empty list provided. Type: {type(frames)}")
        return frame_height, frame_width

    @staticmethod
    def upscale_frame_if_needed(frame, target_size):
        frame_height, frame_width, _ = frame.shape
        if frame_height < target_size[1] or frame_width < target_size[0]:
            scaling_factor = max(target_size[0] / frame_width, target_size[1] / frame_height)
            new_width = int(round(frame_width * scaling_factor))
            new_height = int(round(frame_height * scaling_factor))
            upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return upscaled_frame
        return frame

    @staticmethod
    def validate_and_prepare_rois(
            rois: Union[Tuple[Real, Real, Real, Real], List[Union[Tuple[Real, Real, Real, Real], List[Real]]]]):
        if isinstance(rois, (tuple, list)) and len(rois) == 4 and all(isinstance(coord, Real) for coord in rois):
            rois = [rois]  # Convert single ROI to a list of ROIs
        if not all(isinstance(roi, (tuple, list)) and len(roi) == 4 for roi in rois):
            raise ValueError("ROIs must be a tuple or a list of tuples/lists with four real number elements each")
        return rois

    @staticmethod
    def calculate_adjusted_roi(roi: Tuple[int, int, int, int], frames_shape: Tuple[int, int],
                               crop_size: Tuple[int, int], offset_range: int):

        # Convert ROI coordinates to integers
        x_min, y_min, x_max, y_max = map(int, roi)

        # Calculate target size for cropping or upscaling
        target_width, target_height = crop_size if crop_size else (x_max - x_min, y_max - y_min)

        # Calculate random offset within specified range
        x_offset = random.randint(-offset_range, offset_range)
        y_offset = random.randint(-offset_range, offset_range)

        # Adjust ROI coordinates with offset and ensure it's within frame boundaries
        x_min = max(min(x_min + x_offset, frames_shape[1] - target_width), 0)
        y_min = max(min(y_min + y_offset, frames_shape[0] - target_height), 0)

        return x_min, y_min, target_width, target_height

    @staticmethod
    def crop_frame(frame: np.ndarray,
                   rois: Union[Tuple[Real, Real, Real, Real], List[Union[Tuple[Real, Real, Real, Real], List[Real]]]],
                   crop_size: Optional[Tuple[int, int]] = None,
                   offset_range: int = 100,
                   metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[Dict]]:

        # Ensure frame is 3D
        if frame.ndim != 3:
            raise ValueError("frame must be a 3D numpy array")

        # Validate and prepare rois in correct format
        rois = FrameManipulator.validate_and_prepare_rois(rois)

        # Initialize lsit of crops
        cropped_frames_info = []

        # Use the first ROI for cropping a single frame
        for roi in rois:

            # Calculate roi
            x_min, y_min, target_width, target_height = FrameManipulator.calculate_adjusted_roi(roi, frame.shape[:-1],
                                                                                                crop_size, offset_range)

            # Crop the frame
            cropped_frame = frame[y_min:y_min + target_height, x_min:x_min + target_width, :]

            # Prepare metadata info if it was provided
            if metadata:
                meta_info = {key: value for key, value in metadata.items()}
            else:
                meta_info = {}
            meta_info['roi_coords'] = (x_min, y_min, x_min + target_width, y_min + target_height)

            cropped_frames_info.append((cropped_frame, meta_info))

        return cropped_frames_info

    @staticmethod
    def crop_frames(frames_array: np.ndarray,
                    rois: Union[Tuple[Real, Real, Real, Real], List[Union[Tuple[Real, Real, Real, Real], List[Real]]]],
                    crop_size: Optional[Tuple[int, int]] = None,
                    offset_range: int = 100,
                    metadata: Optional[Dict] = None) -> List[Tuple[np.ndarray, Optional[Dict]]]:

        # Check frames array foramt and convert single frame into a multiple frame array format
        if not isinstance(frames_array, np.ndarray) or frames_array.ndim != 4:
            if frames_array.ndim == 3:

                # Convert the 3D frame into a 4D array with only one frame
                frames_array = np.expand_dims(frames_array, axis=0)

            else:
                raise ValueError("frames_array must be a 4D numpy array")

        # Validate and prepare rois in correct format
        rois = FrameManipulator.validate_and_prepare_rois(rois)

        # Initialize lsit of crops
        cropped_frames_info = []

        for roi in rois:

            # Calculate roi
            x_min, y_min, target_width, target_height = FrameManipulator.calculate_adjusted_roi(roi, frames_array.shape[
                                                                                                     1:-1], crop_size,
                                                                                                offset_range)

            # Bulk crop the frames with single operation
            cropped_frames = frames_array[:, y_min:y_min + target_height, x_min:x_min + target_width, :]

            # Prepare metadata info if it was provided
            if metadata:
                meta_info = {key: value for key, value in metadata.items()}
            else:
                meta_info = {}
            meta_info['roi_coords'] = (x_min, y_min, x_min + target_width, y_min + target_height)

            cropped_frames_info.append((cropped_frames, meta_info))

        return cropped_frames_info  # [(frames: np.ndarray, meta_info: dict)]

    #     @staticmethod
    #     def crop_frames(frames_array: np.ndarray,
    #                     rois: Union[Tuple[Real, Real, Real, Real], List[Union[Tuple[Real, Real, Real, Real], List[Real]]]],
    #                     crop_size: Optional[Tuple[int, int]] = None,
    #                     offset_range: int = 100,
    #                     metadata: Optional[Dict] = None) -> List[Tuple[np.ndarray, Optional[Dict]]]:

    #         if not isinstance(frames_array, np.ndarray) or frames_array.ndim != 4:

    #             if frames_array.ndim == 3:
    #                 # Convert the 3D frame into a 4D array with only one frame
    #                 frames_array = np.expand_dims(frames_array, axis=0)
    #             else:
    #                 raise ValueError("frames_array must be a 4D numpy array")

    #         if isinstance(rois, (tuple, list)) and len(rois) == 4 and all(isinstance(coord, Real) for coord in rois):
    #             rois = [rois]

    #         if not all(isinstance(roi, (tuple, list)) and len(roi) == 4 for roi in rois):
    #             raise ValueError("ROIs must be a tuple or a list of tuples/lists with four real number elements each")

    #         cropped_frames_info = []

    #         for roi in rois:
    #             # Convert ROI coordinates to integers
    #             x_min, y_min, x_max, y_max = map(int, roi)

    #             # Calculate target size for cropping or upscaling
    #             target_width = crop_size[0] if crop_size else (x_max - x_min)
    #             target_height = crop_size[1] if crop_size else (y_max - y_min)

    #             # Calculate random offset within specified range
    #             x_offset = random.randint(-offset_range, offset_range)
    #             y_offset = random.randint(-offset_range, offset_range)

    #             # Adjust ROI coordinates with offset and ensure it's within frame boundaries
    #             x_min = max(min(x_min + x_offset, frames_array.shape[2] - target_width), 0)
    #             y_min = max(min(y_min + y_offset, frames_array.shape[1] - target_height), 0)

    #             # Check and perform upscaling if needed - TODO: Reflect this in adjustment, include this in metadata report
    #             #frames_array = np.array([FrameManipulator.upscale_frame_if_needed(frame, (target_width, target_height)) for frame in frames_array])

    #             # Crop the frames
    #             cropped_frames = frames_array[:, y_min:y_min + target_height, x_min:x_min + target_width, :]

    #             # Prepare metadata if provided
    #             if metadata:
    #                 meta_info = {key: value for key, value in metadata.items()}
    #             else:
    #                 meta_info = {}
    #             meta_info['roi_coords'] = (x_min, y_min, x_min + target_width, y_min + target_height)

    #             cropped_frames_info.append((cropped_frames, meta_info))

    #         return cropped_frames_info

    @staticmethod
    def resize_frames(frames, target_size, preference='balance'):
        """
        Resize frames in a 4D array or a list to a specified target size, either upscaling or downscaling as needed.

        Args:
            frames (np.ndarray or List[np.ndarray]): A 4D array, 3D array with single frame, or a list of frames (numpy arrays).
            target_size (Tuple[int, int]): Target size (width, height) to resize frames to.
            preference (str): Preference for resizing. Options: 'speed', 'quality', 'balance'.
                              'speed' uses cv2.INTER_NEAREST,
                              'quality' uses cv2.INTER_CUBIC,
                              'balance' uses cv2.INTER_LINEAR.

        Returns:
            np.ndarray or List[np.ndarray]: Resized frames in the same format as input.

        Raises:
            TypeError: If the input is not a 4D numpy array or a list of numpy arrays.
            ValueError: If the frames in the input list have inconsistent dimensions.
            ValueError: If an invalid preference option is provided.
        """
        # Determine the interpolation method based on preference
        if preference == 'speed':
            interpolation = cv2.INTER_NEAREST
        elif preference == 'quality':
            interpolation = cv2.INTER_CUBIC
        elif preference == 'balance':
            interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError("Invalid preference. Options are 'speed', 'quality', 'balance'.")

        # Resize logic
        if isinstance(frames, np.ndarray):

            if frames.ndim == 3:
                # Convert the 3D frame into a 4D array with only one frame
                frames = np.expand_dims(frames, axis=0)

            if frames.ndim == 4:
                return np.array([cv2.resize(frame, target_size, interpolation=interpolation) for frame in frames])


        elif isinstance(frames, list) and all(isinstance(frame, np.ndarray) for frame in frames):
            resized_frames = []
            for frame in frames:
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("All frames in the list must be 3D arrays with 3 channels.")
                resized_frame = cv2.resize(frame, target_size, interpolation=interpolation)
                resized_frames.append(resized_frame)
            return resized_frames


        else:
            raise TypeError("Input must be a 4D numpy array or a list of 3D numpy arrays.")

    @staticmethod
    def calculate_target_adjust_image_size(current_dims, min_dims):
        """
        Adjusts the image dimensions to meet minimum width and height requirements
        while maintaining the aspect ratio.

        Parameters:
        current_dims (tuple): The current dimensions of the image (width, height).
        min_dims (tuple): The minimum required dimensions (width, height).

        Returns:
        tuple: New dimensions (width, height) that meet the minimum requirements
               and maintain the aspect ratio.
        """
        current_width, current_height = current_dims
        min_width, min_height = min_dims

        # Calculate the scaling factors needed for width and height
        width_scale = min_width / current_width if current_width < min_width else 1
        height_scale = min_height / current_height if current_height < min_height else 1

        # Choose the larger scaling factor to ensure both dimensions meet or exceed the requirements
        scale_factor = max(width_scale, height_scale)

        # Calculate new dimensions based on the scaling factor
        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        return (new_width, new_height)

    @staticmethod
    def calculate_largest_roi(image_size: Tuple[int, int], target_aspect_ratio: Real):
        """
        Calculate the largest ROI that fits within an image size and meets the target aspect ratio.

        Parameters:
        image_size (tuple): The size of the image as (width, height).
        target_aspect_ratio (Real): The target aspect ratio.

        Returns:
        tuple: The width and height of the largest possible ROI that fits within the image size and matches the aspect ratio of the target size.
        """
        img_width, img_height = image_size

        # Calculate possible ROI dimensions that fit within the image size while maintaining the target aspect ratio
        possible_width = img_height * target_aspect_ratio
        possible_height = img_width / target_aspect_ratio

        if possible_width <= img_width:
            # Width is the limiting dimension
            roi_width = int(possible_width)
            roi_height = img_height
        else:
            # Height is the limiting dimension
            roi_width = img_width
            roi_height = int(possible_height)

        return (roi_width, roi_height)

    @staticmethod
    def save_frame(frame, filename, directory, extension='png'):
        """
        Saves a frame (np.ndarray) to a specified file location with a given filename and extension.

        Args:
        frame (np.ndarray): The image frame to save.
        filename (str): The complete path along with the filename where the frame should be saved.
        extension (str): The image file extension (default is 'jpg').

        Returns:
        bool: True if the frame is saved successfully, False otherwise.
        """
        # Validate the frame input
        if not isinstance(frame, np.ndarray):
            logging.info("Error: The frame must be a NumPy ndarray.")
            return False

        # Validate the image extension
        valid_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
        if extension.lower() not in valid_extensions:
            logging.info(
                f"Error: Unsupported file extension '{extension}'. Supported extensions are: {valid_extensions}")
            return False

        # Ensure the filename ends with the correct extension
        if not filename.lower().endswith(f'.{extension.lower()}'):
            filename += f'.{extension}'

        # Validate and prepare the directory path
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                logging.info(f"Error: Unable to create directory '{directory}'. {str(e)}")
                return False

        # Attempt to save the frame
        try:
            destination = os.path.join(directory, filename)
            success = cv2.imwrite(destination, frame)
            if not success:
                logging.info("Error: Failed to save the frame. Please check the frame data and file path.")
                return False
        except Exception as e:
            logging.info(f"Error: An exception occurred while saving the frame: {str(e)}")
            return False

        logging.info(f"Frame saved successfully at {destination}")
        return True