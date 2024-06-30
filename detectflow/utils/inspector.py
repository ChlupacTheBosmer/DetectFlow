import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import cycle
import numpy as np
from PIL import Image as PILImage
import io
from detectflow.predict.results import DetectionBoxes
from typing import List, Tuple, Union, Optional
import os


class Inspector:
    def __init__(self):
        pass

    @staticmethod
    def display_frames_with_boxes(frames: Union[List[np.ndarray], Tuple[np.ndarray, ...], np.ndarray],
                                  detection_boxes_list: Optional[Union[DetectionBoxes, List, Tuple, np.ndarray]] = None,
                                  figsize: Tuple[int, int] = (12, 8)):
        """
        Displays multiple frames each with their corresponding bounding boxes.

        Args:
        - frames (Union[List[np.ndarray], Tuple[np.ndarray, ...], np.ndarray]): A list or 4D numpy array of frames.
        - detection_boxes_list (Optional[Union[DetectionBoxes, List, Tuple, np.ndarray]]): A list of DetectionBoxes objects corresponding to each frame.
        - figsize (Tuple[int, int]): Size of the figure for each frame plot.
        """
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                frames = list([frames])
            elif frames.ndim == 4:
                frames = list(frames)
            else:
                raise ValueError("Invalid shape of the frames array. Expected 4D or 3D array.")
        elif not isinstance(frames, (tuple, list)) or not all(isinstance(frame, np.ndarray) for frame in frames):
            raise ValueError("Invalid type of the frames. Expected list, tuple or numpy array.")

        # If is single DetectionBoxes object, convert to list
        if isinstance(detection_boxes_list, DetectionBoxes):
            detection_boxes_list = [detection_boxes_list.xyxy]
        # If is a collection of objects
        elif isinstance(detection_boxes_list, (np.ndarray, tuple, list)):
            # if is a collection of DetectionBoxes objects convert to list
            if all(isinstance(detection_boxes, (DetectionBoxes, type(None))) for detection_boxes in detection_boxes_list):
                detection_boxes_list = [detection_boxes.xyxy if detection_boxes else None for detection_boxes in detection_boxes_list]
            # If it is a collection of numpy arrays or lists or tuples
            elif all(isinstance(detection_boxes, (np.ndarray, tuple, list)) for detection_boxes in detection_boxes_list):
                if all(isinstance(coords, (int, float)) for coords in detection_boxes_list[0]):
                    detection_boxes_list = [detection_boxes_list]
                elif all(isinstance(detection_boxes, (np.ndarray, tuple, list)) for detection_boxes in detection_boxes_list[0]):
                    pass
                else:
                    raise ValueError("Invalid type of the detection_boxes_list.")
            else:
                raise ValueError("Invalid type of the detection_boxes_list.")
        elif detection_boxes_list is None:
            detection_boxes_list = []
        else:
            raise ValueError("Invalid type of the detection_boxes_list.")

        for i, frame in enumerate(frames):
            detection_boxes = None if len(detection_boxes_list) < i + 1 else detection_boxes_list[i]
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.imshow(frame)

            if detection_boxes is not None:
                for bbox in detection_boxes:
                    x_min, y_min, x_max, y_max = bbox[:4]
                    width, height = x_max - x_min, y_max - y_min
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)

            plt.show()

    @staticmethod
    def display_images(images, figsize=(12, 8)):
        """
        Displays images, which can be a single image or a list of images.
        The images can be in the form of a PIL Image, a BytesIO stream, or a numpy array.

        Args:
        - images (Union[BytesIO, PILImage, np.ndarray, List[Union[BytesIO, PILImage, np.ndarray]]]): An image or list of images in various formats.
        - figsize (tuple): Size of the figure for each image plot.
        """
        # Ensure input is iterable (list); if not, make it a list
        if not isinstance(images, (list, tuple)) and not (isinstance(images, np.ndarray) and images.ndim == 4):
            images = [images]

        for image in images:
            fig, ax = plt.subplots(1, figsize=figsize)

            # Check if the image is a BytesIO stream
            if isinstance(image, io.BytesIO):
                image.seek(0)  # Ensure the stream is at the beginning
                img = PILImage.open(image)
                img = np.array(img)  # Convert PIL image to numpy array for plotting
            elif isinstance(image, PILImage.Image):
                img = np.array(image)  # Convert PIL image to numpy array
            elif isinstance(image, np.ndarray):
                img = image  # Use directly for plotting
            else:
                raise ValueError("Unsupported image format")

            # Display the image
            ax.imshow(img)
            plt.axis('off')  # Hide the axis
            plt.show()

    @staticmethod
    def display_frame_with_multiple_boxes(frame: np.ndarray,
                                          detection_boxes_list: Union[List['DetectionBoxes'], List[np.ndarray], Tuple[np.ndarray, ...]] = None,
                                          figsize: Tuple[int, int] = (12, 8),
                                          show: bool = True,
                                          save: bool = False,
                                          filename: Optional[str] = None,
                                          save_dir: Optional[str] = None):
        """
        Displays a single frame with several sets of bounding boxes, each with a different color.

        Args:
        - frame (np.ndarray): A single frame to display.
        - detection_boxes_list (Union[List['DetectionBoxes'], List[np.ndarray], Tuple[np.ndarray, ...]]): A list of DetectionBoxes objects or lists/tuples of bounding boxes.
        - figsize (Tuple[int, int]): Size of the figure for the plot.
        - save_path (Optional[str]): Path to save the plot. If None, the plot is saved to self.save_dir.
        """
        if frame.ndim != 3:
            raise ValueError("Invalid shape of the frame array. Expected 3D array.")

        if detection_boxes_list is None:
            detection_boxes_list = []

        colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(frame)

        for detection_boxes, color in zip(detection_boxes_list, colors):
            if isinstance(detection_boxes, DetectionBoxes):
                detection_boxes = detection_boxes.xyxy

            if detection_boxes is not None:
                for bbox in detection_boxes:
                    x_min, y_min, x_max, y_max = bbox[:4]
                    conf = bbox[4] if len(bbox) > 4 else None
                    width, height = x_max - x_min, y_max - y_min
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

                    if conf is not None:
                        # Draw a rectangle for the confidence score
                        conf_rect = patches.Rectangle((x_min, y_min - 15), width=50, height=15, linewidth=1, edgecolor=color,
                                                      facecolor=color, alpha=0.5)
                        ax.add_patch(conf_rect)
                        ax.text(x_min, y_min - 5, f'{conf:.2f}', color='white', fontsize=8, verticalalignment='center')

        if show:
            plt.show()

        if save:
            save_dir = save_dir or os.getcwd()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if filename is None:
                filename = "frame_with_boxes"
            save_path = os.path.join(save_dir, f"{filename}.png")
            fig.savefig(save_path)

        plt.close(fig)

