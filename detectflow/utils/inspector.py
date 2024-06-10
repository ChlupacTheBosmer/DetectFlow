import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image as PILImage
import io
from detectflow.predict.results import DetectionBoxes

class Inspector:
    def __init__(self):
        pass

    @staticmethod
    def display_frames_with_boxes(frames, detection_boxes_list=[], figsize=(12, 8)):
        """
        Displays multiple frames each with their corresponding bounding boxes.

        Args:
        - frames (List[np.ndarray] or np.ndarray): A list or 4D numpy array of frames.
        - detection_boxes_list (List[DetectionBoxes]): A list of DetectionBoxes objects corresponding to each frame.
        - figsize (tuple): Size of the figure for each frame plot.
        """
        if isinstance(frames, np.ndarray):
            frames = list(frames)

        for i, frame in enumerate(frames):
            detection_boxes = None if len(detection_boxes_list) < i + 1 else detection_boxes_list[i]
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.imshow(frame)

            if detection_boxes is not None and isinstance(detection_boxes, DetectionBoxes):
                for bbox in detection_boxes.xyxy:
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
        if not isinstance(images, list) or not (isinstance(images, np.ndarray) and images.ndim == 4):
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