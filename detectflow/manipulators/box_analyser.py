import cv2
import numpy as np
from sklearn.cluster import KMeans
from detectflow.predict.results import DetectionBoxes
from detectflow.predict.tracker import Tracker
from typing import Optional

class BoxAnalyser:
    def __init__(self):
        pass

    @staticmethod
    def box_area(box):
        """ Calculate the area of a bounding box. """
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def box_dimensions(box):
        """Return the width and height of a bounding box."""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width, height

    @staticmethod
    def sort_boxes(boxes, sort_by='area', ascending=True):
        """
        Sort an array of bounding boxes based on specified criteria and order.

        Parameters:
        boxes (np.ndarray): Array of bounding boxes in xyxy format.
        sort_by (str): Criteria to sort by ('area', 'width', 'height', 'largest_dim').
        ascending (bool): If True, sort in ascending order, else in descending order.

        Returns:
        np.ndarray: Sorted array of bounding boxes.
        """
        if sort_by == 'area':
            # Calculate area for each box and sort by this key
            keys = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        elif sort_by == 'width':
            # Calculate width for each box and sort by this key
            keys = boxes[:, 2] - boxes[:, 0]
        elif sort_by == 'height':
            # Calculate height for each box and sort by this key
            keys = boxes[:, 3] - boxes[:, 1]
        elif sort_by == 'largest_dim':
            # Calculate the largest dimension for each box and sort by this key
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keys = np.maximum(widths, heights)
        else:
            raise ValueError(f"Unsupported sort criterion: {sort_by}")

        # Determine the sort order
        if ascending:
            sorted_indices = np.argsort(keys)
        else:
            sorted_indices = np.argsort(keys)[::-1]

        # Return the sorted array of boxes
        return boxes[sorted_indices]

    @staticmethod
    def analyze_boxes(detection_boxes: DetectionBoxes):
        areas = []
        for box in detection_boxes.xyxy:
            # Calculate the area of each bounding box
            areas.append(BoxAnalyser.box_area(box))

        # Calculate variance of the areas
        area_variance = np.var(areas)

        # Identify large discrepancies in sizes
        mean_area = np.mean(areas)
        discrepancies = [area for area in areas if area > mean_area * 2 or area < mean_area / 2]

        # Metrics that might be of interest
        num_boxes = len(areas)
        max_area = max(areas) if areas else None
        min_area = min(areas) if areas else None
        area_range = max_area - min_area if areas else None

        results = {
            "number_of_boxes": num_boxes,
            "max_area": max_area,
            "min_area": min_area,
            "area_range": area_range,
            "area_variance": area_variance,
            "discrepancies_count": len(discrepancies)
        }

        return results

    @staticmethod
    def analyze_bbox_distribution(bboxes, img_size, grid_size=(320, 320)):
        """
        Analyze the distribution of bounding boxes across an image using a grid.

        Parameters:
        - bboxes (np.ndarray): Array of bounding boxes in xyxy format.
        - img_size (tuple): Size of the image as (width, height).
        - grid_size (tuple): Dimensions of the grid (number of columns, number of rows).

        Returns:
        - float: Distribution index as the ratio of non-empty cells to total cells.
        """
        grid_width, grid_height = img_size[0] / grid_size[0], img_size[1] / grid_size[1]
        occupancy_grid = np.zeros(grid_size, dtype=bool)

        # Mark grid cells that have any overlap with any bbox
        for bbox in bboxes:
            # Calculate the range of grid cells the bbox might overlap
            start_col = int(bbox[0] / grid_width)
            end_col = int(np.ceil(bbox[2] / grid_width)) - 1
            start_row = int(bbox[1] / grid_height)
            end_row = int(np.ceil(bbox[3] / grid_height)) - 1

            occupancy_grid[start_row:end_row + 1, start_col:end_col + 1] = True

        # Calculate the distribution index
        non_empty_cells = np.sum(occupancy_grid)
        total_cells = np.prod(grid_size)
        distribution_index = non_empty_cells / total_cells

        return distribution_index

    @staticmethod
    def calculate_iou(box1, box2):
        """ Calculate Intersection over Union (IoU) of two bounding boxes. """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    @staticmethod
    def is_close(box1, box2, threshold=20):
        """
        Check if two bounding boxes are close based on distance of centers.

        Args:
        - box1 (tuple, list, np.ndarray): Bounding box in xyxy format.
        - box2 (tuple, list, np.ndarray): Bounding box in xyxy format.
        - threshold (float): Maximum distance between centers for boxes to be considered close.
        Returns:
        - bool: True if the boxes are close, False otherwise.
        """
        center1 = BoxAnalyser.box_center(box1)
        center2 = BoxAnalyser.box_center(box2)
        distance = np.linalg.norm(np.array(center1) - np.array(center2))
        return distance < threshold

    @staticmethod
    def is_contained(box1, box2):
        """
        Check if box1 is contained within box2.

        Args:
        - box1 (tuple, list, np.ndarray): Bounding box in xyxy format.
        - box2 (tuple, list, np.ndarray): Bounding box in xyxy format.
        Returns:
        - bool: True if the boxes are close, False otherwise.
        """
        return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

    @staticmethod
    def find_consistent_boxes(detection_results: list, iou_threshold=0.5, min_frames=3):
        """ Find boxes that appear consistently across frames. """
        consistent_results = []
        # For each result get boxes object
        for i, current_result in enumerate(detection_results):
            consistent_boxes_current = []

            # Continue only if the boxes object is not None
            current_result_boxes = current_result.boxes
            if current_result_boxes is not None:
                orig_shape = current_result_boxes.orig_shape

                # For each box in the current boxes object
                for current_box in current_result_boxes.xyxy:
                    count = 0

                    # Loop over all other results
                    for other_result in detection_results:

                        # Continue only if the boxes of the other result is not None
                        other_result_boxes = other_result.boxes
                        if other_result_boxes is not None:

                            # Loop over all bboxes in the other boxes object
                            for other_box in other_result_boxes.xyxy:
                                if BoxAnalyser.calculate_iou(current_box, other_box) > iou_threshold:
                                    count += 1
                                    break
                    if count >= min_frames:
                        consistent_boxes_current.append(tuple(current_box))

                # Create an updated boxes object and assign it to the current results boxes attribute
                det = None if len(consistent_boxes_current) == 0 else DetectionBoxes(consistent_boxes_current,
                                                                                     orig_shape, "xyxy")
                current_result.boxes = det
                consistent_results.append(current_result)
        return consistent_results
        # return list(set(consistent_boxes))  # Remove duplicates

    @staticmethod
    def box_center(box):
        """ Calculate the center of a bounding box. """
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return (x_center, y_center)

    @staticmethod
    def find_moving_boxes(detection_results: list, tracker: Optional[Tracker] = None, movement_threshold=10):
        """
        Find boxes from the first frame that move in subsequent frames using tracking.
        :param detection_results: List of DetectionResults objects for each frame.
        :param tracker: Tracker object to track boxes across frames.
        :param movement_threshold: Threshold for determining if a box is moving.
        :return: DetectionResults object with moving boxes from the first frame in DetectionResults.boxes attribute.
        """
        result = None
        if detection_results and len(detection_results) > 0:
            # Initialize Tracker
            tracker = Tracker() if tracker is None else tracker
            # Run tracking on the frames
            tracked_results = tracker.process_tracking(
                detection_results)  # Returns list of DetectionResults updated for each passed Result

            # Get boxes from the first frame
            first_frame_boxes = None
            for tracked_result in tracked_results:
                if tracked_result is not None and tracked_result.boxes is not None:  # Check for None values
                    first_frame_boxes = tracked_result.boxes.data
                    break
            if first_frame_boxes is None:
                return None
            moving_boxes_from_first_frame = []

            # Create a dictionary to map track IDs to first frame boxes
            track_id_to_first_frame_box = {box[-1]: box for box in first_frame_boxes}

            # Analyze movement of boxes
            for tracked_result in tracked_results[1:]:  # Start from second frame
                if tracked_result is not None and tracked_result.boxes is not None:  # Check for None values
                    for box in tracked_result.boxes.data:
                        track_id = box[-1]  # Track ID
                        if track_id in track_id_to_first_frame_box:
                            # Compare with the corresponding box in the first frame
                            first_frame_box = track_id_to_first_frame_box[track_id]
                            distance = np.linalg.norm(np.array(BoxAnalyser.box_center(box)) - np.array(
                                BoxAnalyser.box_center(first_frame_box)))
                            if distance > movement_threshold:
                                moving_boxes_from_first_frame.append(first_frame_box)
                                del track_id_to_first_frame_box[track_id]  # Remove track ID to prevent duplicates

            # Remove duplicates using a custom function
            unique_moving_boxes = BoxAnalyser.remove_duplicate_boxes(moving_boxes_from_first_frame)
            mov = None if len(unique_moving_boxes) == 0 else DetectionBoxes(unique_moving_boxes,
                                                                            detection_results[0].boxes.orig_shape,
                                                                            "xyxy")
            detection_results[0].boxes = mov
            result = None if mov is None else detection_results[0]
        return result

    @staticmethod
    def remove_duplicate_boxes(boxes):
        """
        Custom function to remove duplicate boxes.
        :param boxes: List of bounding boxes.
        :return: List of unique bounding boxes.
        """
        unique_boxes = []
        seen = set()
        for box in boxes:
            box_tuple = tuple(box[:-1])  # Exclude track ID for comparison
            if box_tuple not in seen:
                seen.add(box_tuple)
                unique_boxes.append(box)
        return unique_boxes

    @staticmethod
    def find_outlier_boxes(frames_bboxes, size_threshold=1.5, central_tendency='mean'):
        """
        Find outlier boxes in terms of size.
        :param frames_bboxes: List of bounding boxes for each frame.
        :param size_threshold: Threshold for determining outliers.
        :param central_tendency: Method for calculating central tendency ('mean' or 'median').
        """
        # Calculate the metrics for all bboxes
        all_boxes = [box for frame in frames_bboxes for box in frame.xyxy]
        areas = [BoxAnalyser.box_area(box) for box in all_boxes]
        if central_tendency == 'median':
            central_area = np.median(areas)
        else:
            central_area = np.mean(areas)
        std_area = np.std(areas)

        # For each frame detection boxes get the outliers
        outliers = []
        for frame_bboxes in frames_bboxes:
            orig_shape = frame_bboxes.orig_shape
            outlier_boxes = [box for box in frame_bboxes.xyxy if
                             abs(BoxAnalyser.box_area(box) - central_area) > size_threshold * std_area]
            out = None if len(outlier_boxes) == 0 else DetectionBoxes(outlier_boxes, orig_shape, "xyxy")
            outliers.append(out)

        return outliers

    @staticmethod
    def extract_bbox_area(frame, bbox):
        """ Extracts the area inside a bounding box from a frame. """
        x_min, y_min, x_max, y_max = bbox
        return frame[int(y_min):int(y_max), int(x_min):int(x_max)]

    @staticmethod
    def dominant_color(image, k=3, color_ranges=None):
        """
        Finds the dominant color in an image, excluding specified colors.
        :param image: Image from which to extract the dominant color.
        :param k: Number of clusters for KMeans.
        :param color_ranges: Dict with color ranges to exclude.
        :return: The dominant color in BGR format, or None if excluded.
        """
        if color_ranges is None:
            color_ranges = {
                'green': [(87, 162), (80, 255)],  # Hue range for green, Saturation range
                'brown': [(10, 20), (50, 255)],
                'black': [(0, 180), (0, 50)],  # Saturation less than 50 for black
                'grey': [(0, 180), (0, 50)]  # Low saturation for grey
            }

        # Reshape the image to be a list of pixels and apply KMeans
        pixels = image.reshape((image.shape[0] * image.shape[1], 3))
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Check each dominant color against the exclusion ranges
        for color in dominant_colors:
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
            h, s, v = hsv_color[0][0]
            excluded = False
            for h_range, s_range in color_ranges.values():
                if h_range[0] <= h <= h_range[1] and s_range[0] <= s <= s_range[1]:
                    excluded = True
                    break
            if not excluded:
                return color

        return None

    @staticmethod
    def filter_bbox_by_color(frame_bboxes, frames, k=3):
        """ Filters bounding boxes based on color analysis. """
        color_filtered_boxes = []
        for frame, bboxes in zip(frames, frame_bboxes):
            filtered_boxes = []
            orig_shape = bboxes.orig_shape
            for bbox in bboxes.xyxy:
                bbox_area = BoxAnalyser.extract_bbox_area(frame, bbox)
                dom_color = BoxAnalyser.dominant_color(bbox_area, k=k)
                if dom_color is not None:
                    filtered_boxes.append(bbox)
                    print(dom_color)
            fil = None if len(filtered_boxes) == 0 else DetectionBoxes(filtered_boxes, orig_shape, "xyxy")
            color_filtered_boxes.append(fil)
        return color_filtered_boxes