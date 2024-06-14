import numpy as np
import random
from detectflow.predict.results import DetectionResults, DetectionBoxes
from ultralytics.engine.results import Results
from detectflow.video.video_data import Video


class Sampler:
    def __init__(self):
        pass

    @staticmethod
    def get_frame_numbers(total_frames: int, num_of_frames: int = 12, distribution='random', output_type="tuple"):
        '''
        Will return a collection (tuple or list) of frame numbers based o nthe selected distribution (random or even).

        Args:
        - total_frames (int): Total number of frames in a video
        - num_of_frames (int): Number of sampled frames
        - distribution (str): 'random' or 'even' determines whether the frame numbers will be evenly or randomly distributed across the video.
        - output_type (str): 'tuple' or 'list' determines what type will the returned collection be.

         Returns:
        - Union[Tuple[int], List[int]]: Collection of frame numbers.
        '''
        if total_frames < num_of_frames:
            raise ValueError(f"Total number of frames must be at least {num_of_frames}.")

        # Select random frames
        if distribution == 'random':
            selected_frames = random.sample(range(total_frames), min(num_of_frames, total_frames))
        else:
            # Calculate the interval between the frames to be selected
            interval = (total_frames - (num_of_frames - 2)) // (num_of_frames - 1)

            # Generate a list of frame numbers, starting with the 5th frame
            selected_frames = [5 + i * interval for i in range(num_of_frames)]

            # Ensure the last frame is the 5th from the end
            selected_frames[-1] = total_frames - 5

        if output_type == 'tuple':
            return tuple(selected_frames)
        else:
            return selected_frames

    @staticmethod
    def sample_frames(video_path, num_frames=1, output_format='array', distribution='random', reader='decord'):
        """
        Reads a specified number of random frames from a video file.

        Args:
        - video_path (str): Path to the video file.
        - num_frames (int): Number of random frames to read.
        - output_format (str): The format of the output, 'list' for list of frames, 'array' for 4D numpy array.
        - distribution (str): 'random' or 'even' determines whether the frame numbers will be evenly or randomly distributed across the video.
        - reader (str): The prefered reader method ('opencv', 'imageio', 'decord'). Imageio is always used for .avi videos.

        Returns:
        - Union[List[np.ndarray], np.ndarray]: List of frames or 4D numpy array of frames.
        """
        #         # Open the video file
        #         cap = cv2.VideoCapture(video_path)
        #         if not cap.isOpened():
        #             raise IOError("Could not open video file")

        # Fix input
        reader = reader if reader in ('opencv', 'imageio', 'decord') else 'decord'

        try:
            # Init video file
            video = Video(video_path, reader_method=reader)

        except Exception as e:
            raise RuntimeError("Error when initializing video file.") from e

        try:
            # Get frame numbers
            frame_numbers = Sampler.get_frame_numbers(video.total_frames, num_frames, distribution, 'list')

            #  Read frames
            frames = [np.array(result.get('frame', None)) for result in video.read_video_frame(frame_numbers, False)]
        except Exception as e:
            raise RuntimeError("Error when reading frames.") from e

        # Convert to 4D numpy array if requested
        if output_format == 'array':
            if frames:
                frames = np.stack(frames)
            else:
                frames = np.array([], dtype=object)

        return frames

    @staticmethod
    def get_random_sample(collection, sample_size, include_keys=None, exclude_keys=None):
        """
        Returns a random sample from a collection.

        :param collection: A collection to sample from (list, tuple, numpy array, or dictionary).
        :param sample_size: The number of items to randomly select.
        :param include_keys: Optional list of keys to include if collection is a dictionary.
        :param exclude_keys: Optional list of keys to exclude if collection is a dictionary.
        :return: A list containing a random sample of items.
        :raises ValueError: If the sample size is not appropriate or the input is not valid.
        """
        if isinstance(collection, dict):
            keys = list(collection.keys())
            if include_keys is not None:
                keys = [key for key in keys if key in include_keys]
            if exclude_keys is not None:
                keys = [key for key in keys if key not in exclude_keys]
            items = [collection[key] for key in keys]
        elif isinstance(collection, (list, tuple, np.ndarray)):
            items = list(collection)
        else:
            raise TypeError("The collection must be a list, tuple, numpy array, or dictionary.")

        if not items:
            raise ValueError("The collection is empty or no items match the criteria.")

        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("The 'sample_size' must be a positive integer.")

        if sample_size > len(items):
            raise ValueError("Sample size cannot be greater than the number of items in the collection.")

        return random.sample(items, sample_size)

    @staticmethod
    def create_sample_image(grid_size=8, square_size=128):
        """
        Creates a checkered multicolored pattern as a NumPy array.

        Args:
        - grid_size (int): The number of squares along one dimension of the grid.
        - square_size (int): The size of each square in pixels.

        Returns:
        - A NumPy array representing the image.
        """
        # Calculate overall image size
        img_size = grid_size * square_size
        # Initialize an empty image
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Generate colors for squares
        colors = np.random.randint(0, 256, size=(grid_size, grid_size, 3), dtype=np.uint8)

        for i in range(grid_size):
            for j in range(grid_size):
                # Determine the square's coordinates
                x_min, y_min = i * square_size, j * square_size
                x_max, y_max = x_min + square_size, y_min + square_size
                # Assign color to the square
                image[x_min:x_max, y_min:y_max] = colors[i, j] if (i + j) % 2 == 0 else colors[i, j - 1]

        return image

    @staticmethod
    def create_sample_bboxes(grid_size=8, square_size=128, num_boxes=5, as_detection_boxes=False):

        # Generate random bounding boxes
        boxes = []
        for _ in range(num_boxes):
            # Randomly choose a grid cell to start the box
            start_x = random.randint(0, grid_size - 1)
            start_y = random.randint(0, grid_size - 1)
            # Randomly determine the size of the box (at least 1 cell, at most extending to the grid edge)
            end_x = random.randint(start_x, grid_size - 1)
            end_y = random.randint(start_y, grid_size - 1)
            # Convert grid cell coordinates to pixel coordinates
            box = (start_x * square_size, start_y * square_size, (end_x + 1) * square_size, (end_y + 1) * square_size)
            boxes.append(box)

        if as_detection_boxes:
            image_size = grid_size * square_size
            return DetectionBoxes(boxes, (image_size, image_size), "xyxy")
        else:
            return boxes

    @staticmethod
    def create_sample_image_with_bboxes(grid_size=8, square_size=128, num_boxes=5, as_detection_boxes=False):
        """
        Creates a checkered multicolored pattern as a NumPy array and returns a set of bounding boxes.

        Args:
        - grid_size (int): The number of squares along one dimension of the grid.
        - square_size (int): The size of each square in pixels.
        - num_boxes (int): The number of bounding boxes to generate.

        Returns:
        - image: A NumPy array representing the image.
        - boxes: A list of tuples representing the bounding boxes in (x1, y1, x2, y2) format or DetectionBoxes object.
        """

        image = Sampler.create_sample_image(grid_size, square_size)
        boxes = Sampler.create_sample_bboxes(grid_size, square_size, num_boxes, as_detection_boxes)

        return image, boxes

    @staticmethod
    def create_sample_detection_result(grid_size=8, square_size=128, num_boxes=5):

        # Generate image and boxes
        image, boxes = Sampler.create_sample_image_with_bboxes(grid_size, square_size, num_boxes, True)

        # Init the object instance
        result = DetectionResults(image, "/sample_image.png", {0: "0"})

        # Assign test attributes
        result.boxes = boxes
        result.ref_boxes = Sampler.create_sample_bboxes(grid_size=8, square_size=24, num_boxes=2,
                                                        as_detection_boxes=True)
        result.frame_number = random.randint(1, 30000)
        result.source_path = "/storage/brno2/home/USER/videos/first_batch/CZ1_M2_MyoPar03/CZ1_M2_MyoPar03_20210519_12_07.mp4"
        result.source_name = "CZ1_M2_MyoPar03_20210519_12_07"
        result.visit_number = "420"
        result.roi_number = None

        return result

    @staticmethod
    def create_sample_results(grid_size=8, square_size=128, num_boxes=5):

        # Generate image and boxes
        image, boxes = Sampler.create_sample_image_with_bboxes(grid_size, square_size, num_boxes, False)

        # Init the object instance
        result = Results(image, "/sample_image.png", {0: "0"})

        return result
