import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import logging
from itertools import permutations
from typing import List, Tuple, Union, Optional

from detectflow.predict.tracker import Tracker
from detectflow.predict.results import DetectionResults, DetectionBoxes
from typing import Union, Callable, Optional


def try_convert_to_numpy(collection):
    if collection is None:
        raise ValueError("Invalid argument: None")

    if isinstance(collection, np.ndarray):
        return collection

    try:
        array = np.array(collection)
    except Exception as e:
        raise ValueError(f"Failed to convert value to numpy array: {e}")

    return array


def boxes_edge_edge_distance(box1: Union[np.ndarray, List, Tuple], box2: Union[np.ndarray, List, Tuple], min_or_max: str = "max"):
    """
    Calculate the minimum or maximum distance between the edges of two bounding boxes (box1 and box2).

    Args:
    - box1 (np.ndarray, List, Tuple): Array of the first bounding box in format [x_min, y_min, x_max, y_max].
    - box2 (np.ndarray, List, Tuple): Array of the second bounding box in format [x_min, y_min, x_max, y_max].
    - min_or_max (str): Whether to calculate the minimum or maximum distance. Options: "min" or "max".

    Returns:
    - float: Minimum or maximum distance between the edges of box1 and box2.
    """

    for box in [box1, box2]:
        try_convert_to_numpy(box)

    if min_or_max not in ["min", "max"]:
        raise ValueError(f"min_or_max must be 'min' or 'max', got {min_or_max}")

    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    # Calculate the minimum distance
    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    if min_or_max == "max" and dx == dy == 0:
        # Adjust the distance for the "max" case when the boxes overlap
        dx = max(abs(x1_max - x2_min), abs(x2_max - x1_min))
        dy = max(abs(y1_max - y2_min), abs(y2_max - y1_min))

    return np.sqrt(dx ** 2 + dy ** 2)


def boxes_center_edge_distance(box1: Union[np.ndarray, List, Tuple], box2: Union[np.ndarray, List, Tuple], min_or_max: str = "max"):
    """
    Calculate the distance from the center of a bounding box (bbox1) to the furthest point of another bounding box (bbox2).
    Can be used as metric for analyse_clusters method of BoxManipulator.

    Args:
    - bbox1 (np.ndarray, List, Tuple): Array of the first bounding box (xyxy).
    - bbox2 (np.ndarray, List, Tuple): Array of the second bounding box (xyxy).
    - min_or_max (str): Whether to calculate the minimum or maximum distance. Options: "min" or "max".

    Returns:
    - float: Distance from the center of bbox1 to the furthest point of bbox2.
    """

    for box in [box1, box2]:
        try_convert_to_numpy(box)

    # Calculate the centers and dimensions
    x1_center, y1_center = BoxManipulator.get_box_center(box1)

    # Calculate distances from bbox1 center to bbox2 corners
    distances = [
        distance.euclidean((x1_center, y1_center), (box2[0], box2[1])),
        distance.euclidean((x1_center, y1_center), (box2[2], box2[1])),
        distance.euclidean((x1_center, y1_center), (box2[0], box2[3])),
        distance.euclidean((x1_center, y1_center), (box2[2], box2[3]))
    ]

    # Return the maximum distance
    return max(distances) if min_or_max == "max" else min(distances)


def boxes_center_center_distance(box1: Union[np.ndarray, List, Tuple], box2: Union[np.ndarray, List, Tuple]):
    """
    Calculate the distance from the center of a bounding box (bbox1) to the center of another bounding box (bbox2).
    Can be used as metric for analyse_clusters method of BoxManipulator.

    Args:
    - bbox1 (np.ndarray, List, Tuple): Array of the first bounding box.
    - bbox2 (np.ndarray, List, Tuple): Array of the second bounding box.

    Returns:
    - float: Distance from the center of bbox1 to the center of bbox2.
    """

    for box in [box1, box2]:
        try_convert_to_numpy(box)

    # Calculate the centers
    x1_center, y1_center = BoxManipulator.get_box_center(box1)
    x2_center, y2_center = BoxManipulator.get_box_center(box2)

    # Calculate the Euclidean distance between the centers
    return distance.euclidean((x1_center, y1_center), (x2_center, y2_center))


class BoxManipulator:
    def __init__(self):
        pass

    @staticmethod
    def get_box_area(box: Union[np.ndarray, List, Tuple]):
        """ Calculate the area of a bounding box. """
        if box is None:
            raise ValueError("Invalid argument: None")
        else:
            return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def get_box_dimensions(box: Union[np.ndarray, List, Tuple]):
        """Return the width and height of a bounding box."""
        if box is None:
            raise ValueError("Invalid argument: None")
        else:
            width = box[2] - box[0]
            height = box[3] - box[1]
            return width, height

    @staticmethod
    def get_box_center(box: Union[np.ndarray, List, Tuple]):
        """ Calculate the center of a bounding box. """
        if box is None:
            raise ValueError("Invalid argument: None")
        else:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            return x_center, y_center

    @staticmethod
    def get_distance_between_boxes(box1: Union[np.ndarray, List, Tuple], box2: Union[np.ndarray, List, Tuple], min_or_max="min", metric="cc"):
        """
        Calculate the distance between two bounding boxes based on the specified metric.
        :param box1: First bounding box in format [x_min, y_min, x_max, y_max] from which the distance of the second box is calculated.
        :param box2: Second bounding box in format [x_min, y_min, x_max, y_max] to which the distance is calculated.
        :param min_or_max: str - Whether to calculate the minimum or maximum distance. Options: "min" or "max".
        :param metric: str - The metric to use for calculating the distance. Options: "cc" (center-center), "ce" (center-edge), "ee" (edge-edge).
        :return:
        """

        if box1 is None or box2 is None:
            raise ValueError("Invalid argument: None")

        try:
            box1 = np.array(box1)
            box2 = np.array(box2)
        except Exception as e:
            raise ValueError(f"Failed to convert boxes to numpy arrays: {e}")

        if metric == "cc":
            return boxes_center_center_distance(box1, box2)
        elif metric == "ce":
            return boxes_center_edge_distance(box1, box2, min_or_max)
        elif metric == "ee":
            return boxes_edge_edge_distance(box1, box2, min_or_max)

    @staticmethod
    def get_sorted_boxes(boxes: Union[DetectionBoxes, np.ndarray], sort_by: str = 'area', ascending: bool = True):
        """
        Sort an array of bounding boxes based on specified criteria and order.

        Parameters:
        boxes (Union[DetectionBoxes, np.ndarray]): Array of bounding boxes in xyxy format.
        sort_by (str): Criteria to sort by ('area', 'width', 'height', 'largest_dim').
        ascending (bool): If True, sort in ascending order, else in descending order.

        Returns:
        np.ndarray: Sorted array of bounding boxes.
        """

        if boxes is None:
            raise ValueError("Invalid argument: None")

        # Uses boxes.data to preserve additional info if present
        if isinstance(boxes, DetectionBoxes):
            return_detection_boxes = True
            orig_shape = boxes.orig_shape
            boxes = boxes.data
        else:
            return_detection_boxes = False
            orig_shape = None

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
        return boxes[sorted_indices] if not return_detection_boxes else DetectionBoxes(boxes[sorted_indices], orig_shape)

    @staticmethod
    def get_roi(detection_boxes: Union[DetectionBoxes, np.ndarray],
                img_dims: Tuple[int, int],
                crop_size=(640, 640),
                handle_overflow='expand',
                max_expansion_limit=(1000, 1000),
                margin=0):
        """
        Construct an ROI around the given cluster of boxes, adhering to the specified requirements.

        img_dims - (width, height)
        """

        if detection_boxes is None or img_dims is None:
            return None

        # Calculate the cluster bbox
        cluster_x_min, cluster_y_min, cluster_x_max, cluster_y_max = BoxManipulator.get_combined_box(detection_boxes)

        # Calculate the minimum required dimensions to include the cluster with margin
        min_required_width = (cluster_x_max - cluster_x_min) + 2 * margin
        min_required_height = (cluster_y_max - cluster_y_min) + 2 * margin

        # Determine the bounds for expansion based on the overflow policy
        if handle_overflow == "expand":
            max_required_dimension = max(min_required_width, min_required_height)
            max_width, max_height = min(max_expansion_limit[0], max(max_required_dimension, crop_size[0])), min(
                max_expansion_limit[1], max(max_required_dimension, crop_size[0]))
        else:  # handle_overflow == "strict"
            max_width, max_height = crop_size

        # Determine initial dimensions based on crop_size, ensuring we don't start smaller
        initial_width = max(max_width, min_required_width)
        initial_height = max(max_height, min_required_height)

        # Adjust to maintain aspect ratio
        aspect_ratio = max_width / max_height
        if initial_width / initial_height > aspect_ratio:
            # Width is too wide for the aspect ratio, increase height
            initial_height = initial_width / aspect_ratio
        else:
            # Height is too tall for the aspect ratio, increase width
            initial_width = initial_height * aspect_ratio

        # Center the ROI around the cluster's centroid, as close as possible
        centroid_x = (cluster_x_min + cluster_x_max) / 2
        centroid_y = (cluster_y_min + cluster_y_max) / 2

        roi_x_min = max(0, centroid_x - initial_width / 2)
        roi_y_min = max(0, centroid_y - initial_height / 2)

        # Adjustments to ensure ROI fits within image boundaries
        roi_x_min = min(roi_x_min, img_dims[0] - initial_width)
        roi_y_min = min(roi_y_min, img_dims[1] - initial_height)
        roi_x_max = roi_x_min + initial_width
        roi_y_max = roi_y_min + initial_height

        # Final check to ensure ROI fits within the image
        if roi_x_max > img_dims[0] or roi_y_max > img_dims[1]:
            return None  # Unable to construct a valid ROI that fits within the image boundaries

        # Ensure the ROI does not exceed the maximum allowed dimension
        if initial_width > max_width or initial_height > max_height:
            return None  # Unable to construct a valid ROI without exceeding crop size

        return roi_x_min, roi_y_min, roi_x_max, roi_y_max

    @staticmethod
    def get_optimal_roi(detection_boxes: DetectionBoxes,
                        img_dims: Tuple[int, int],
                        crop_size: Tuple[int, int] = (640, 640),
                        handle_overflow: str = "expand",
                        max_expansion_limit: Tuple[int, int] = (1000, 1000),
                        margin: int = 100,
                        exhaustive_search: bool = False,
                        permutation_limit: int = 9,
                        multiple_rois: bool = False,
                        ignore_empty: bool = True,
                        partial_overlap: bool = False,
                        overlap_threshold: float = 0.5):
        '''
        Note that img_dims format is (width, height)
        '''

        def _calculate_eps(crop_size, handle_overflow):
            if handle_overflow == "expand":
                return min(crop_size) // 2
            else:
                return (min(crop_size) // 2) - (min(crop_size) // 20)

        def _adjust_dims_to_aspect_ratio(size1, size2):
            """Adjust size2 to match the aspect ratio of size1."""
            aspect_ratio1 = size1[0] / size1[1]

            # Calculate aspect ratio for size2 and compare
            aspect_ratio2 = size2[0] / size2[1]

            # If the aspect ratios are the same (considering a small tolerance), no adjustment is needed
            if abs(aspect_ratio1 - aspect_ratio2) < 1e-6:
                return size2

            # Adjust size2 to match the aspect ratio of size1
            # Determine whether to adjust width or height based on which adjustment would minimize the size change
            adjusted_width2 = int(size2[1] * aspect_ratio1)  # Adjust width based on size1's aspect ratio
            adjusted_height2 = int(size2[0] / aspect_ratio1)  # Adjust height based on size1's aspect ratio

            # Choose adjustment that downsizes size2 (if necessary) to ensure it adheres to aspect ratio1 without exceeding original dimensions
            if adjusted_width2 <= size2[0]:
                # Adjusting width downsizes or keeps size2 the same, so we choose this adjustment
                return adjusted_width2, size2[1]
            else:
                # Adjusting height is necessary; this will not exceed the original height of size2
                return size2[0], adjusted_height2

        empty = False

        if detection_boxes is None:

            if not ignore_empty:
                # Create a center dummy box to create a central ROI for empty detection boxes
                dummy_boxes = np.array([[max(0, img_dims[0] // 2 - 10), max(0, img_dims[1] // 2 - 10),
                                         min(img_dims[0], img_dims[0] // 2 + 10),
                                         min(img_dims[1], img_dims[1] // 2 + 10)]])
                detection_boxes = DetectionBoxes(dummy_boxes, img_dims[::-1])
                empty = True
            else:
                return None

        if not isinstance(detection_boxes, DetectionBoxes):
            raise TypeError(f"Expected a DetectionBoxes instance, got {type(detection_boxes)} instead")
        elif not hasattr(detection_boxes, 'xyxy'):
            raise AttributeError(f"DetectionBoxes instance does not have the expected 'xyxy' attribute.")

        # Adjust max expansion aspect ratio to match the crop size aspect ratio
        max_expansion_limit = _adjust_dims_to_aspect_ratio(crop_size, max_expansion_limit)

        # Automatically limit crop size and max expansion size depending on the size of the image
        crop_size = BoxManipulator.adjust_dims_to_fit(crop_size, img_dims)
        max_expansion_limit = BoxManipulator.adjust_dims_to_fit(max_expansion_limit, img_dims)

        # Initiate remaining_boxes as all boxes and list of rois
        remaining_boxes = detection_boxes
        done_boxes = []
        rois = []

        while remaining_boxes is not None and len(remaining_boxes) > 0:

            logging.debug(f"Analysing boxes: {remaining_boxes.xyxy}")
            logging.debug(f"Done boxes: {done_boxes}")

            # Step 1: Analyze clusters
            dynamic_eps = _calculate_eps(crop_size, handle_overflow)
            min_samples = max(1, len(remaining_boxes) // 20)  # Example dynamic calculation
            cluster_detection_boxes, cluster_dict = BoxManipulator.get_clusters(remaining_boxes, eps=dynamic_eps,
                                                                                min_samples=min_samples)

            logging.debug(f"Cluster dict: {cluster_dict}")

            # Step 2: Sort clusters by importance (number of boxes)
            sorted_clusters = sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)

            # Iterate over clusters to find the best ROI
            best_roi, boxes_included = BoxManipulator._expand_cluster(sorted_clusters, img_dims, crop_size,
                                                                      handle_overflow, max_expansion_limit, margin,
                                                                      exhaustive_search=exhaustive_search,
                                                                      permutation_limit=permutation_limit)

            logging.debug(f"Found a ROI: {best_roi} with boxes {boxes_included.xyxy}")

            # Append the resulting list
            rois.append((best_roi, boxes_included if not empty else None))

            # Append the done boxes
            done_boxes = done_boxes + [box for box in np.array(boxes_included.data)]

            # If partial overlap is allowed boxes that overlap partially with the roi (with some threshold) will be treated as included in the roi
            if partial_overlap:
                for box in np.array([bbox for bbox in remaining_boxes if bbox not in np.array(done_boxes)]):
                    # Check if more than threshold fraction of the bbox is within the roi
                    if box is not None:
                        if BoxManipulator.is_overlap(box, best_roi, overlap_threshold=overlap_threshold):
                            logging.debug(f"A partial overlap detected with a box: {box}")
                            done_boxes = done_boxes + [box]

            # Get boxes remaining after finding the best crop
            if multiple_rois and len(done_boxes) < len(detection_boxes):
                remaining = np.array([box for box in detection_boxes if box not in np.array(done_boxes)])
                remaining_boxes = DetectionBoxes(remaining, img_dims[::-1]) if len(remaining) > 0 else None
            else:
                break

        # Return the list of rois and the boxes included in them List[(roi, boxes)]
        return rois

    @staticmethod
    def _expand_cluster(sorted_clusters,
                        img_dims,
                        crop_size,
                        handle_overflow,
                        max_expansion_limit,
                        margin,
                        exhaustive_search: bool = False,
                        permutation_limit: int = 7):

        # Define initial values
        best_roi = None
        best_included_boxes = None
        max_boxes_included = 0
        all_boxes = np.array([box for cluster in sorted_clusters for box in cluster[1].data])
        total_boxes = len(all_boxes)

        # For each cluster try searching for the best possible ROI by expanding the cluster with more boxes
        for cluster_label, cluster_boxes in sorted_clusters:

            # Get cluster characteristics
            # cluster_boxes = cluster_boxes.xyxy
            # current_cluster_bbox = BoxManipulator.calculate_cluster_bbox(cluster_boxes)
            included_boxes = cluster_boxes.copy()

            # During first iteration assign the first cluster values as the best by default
            max_boxes_included = len(included_boxes) if len(included_boxes) > max_boxes_included else max_boxes_included
            best_roi = BoxManipulator.get_roi(included_boxes, img_dims, crop_size, handle_overflow, max_expansion_limit,
                                              margin) if best_roi is None else best_roi
            best_included_boxes = included_boxes if best_included_boxes is None else best_included_boxes

            logging.debug(f"Current cluster: #{cluster_label} - {cluster_boxes.xyxy}")
            logging.debug(f"Current cluster box: {BoxManipulator.get_combined_box(cluster_boxes)}")

            # Get boxes remaining out of cluster and filter them by distance for efficient search
            remaining_boxes = np.array([box for box in all_boxes if box not in cluster_boxes])
            feasible_boxes = BoxManipulator._get_boxes_filtered_by_distance(included_boxes, remaining_boxes,
                                                                            handle_overflow, crop_size,
                                                                            max_expansion_limit)

            if len(remaining_boxes) == 0:

                logging.debug(f"All boxes in one cluster.")

                # When there are no reamining boxes, simply construct the roi from the current cluster bbox
                best_roi = BoxManipulator.get_roi(included_boxes, img_dims, crop_size, handle_overflow,
                                                  max_expansion_limit, margin)
                # TODO: This assumes that ROI can be created around every cluster defined by the clsuter analysis as best_roi being None is not handled

                break

            #                 # TODO: How to decide ties when there are more than one clusters with the same number of boxes and any cant be added to any of the clusters?
            else:
                if exhaustive_search:
                    # Perform exhaustive search through permutations of the remaining boxes

                    # Sort boxes by their distance to the cluster
                    nearest_boxes, distances = BoxManipulator._get_boxes_sorted_by_distance(included_boxes,
                                                                                            feasible_boxes,
                                                                                            permutation_limit)

                    for n in range(1, len(nearest_boxes) + 1):
                        for permutation in permutations(nearest_boxes, n):

                            temp_included_boxes = np.append(included_boxes, permutation, axis=0)
                            # temp_cluster_bbox = BoxManipulator.calculate_cluster_bbox(temp_included_boxes)

                            # TODO: Better manage large duplication of code
                            temp_roi = BoxManipulator.get_roi(temp_included_boxes, img_dims, crop_size, handle_overflow,
                                                              max_expansion_limit, margin)

                            if temp_roi is not None and len(temp_included_boxes) > max_boxes_included:
                                best_roi = temp_roi
                                best_included_boxes = temp_included_boxes
                                max_boxes_included = len(temp_included_boxes)

                                logging.debug(f"Better ROI found. Number of boxes: {len(temp_included_boxes)}")

                            if max_boxes_included == total_boxes:
                                logging.debug(f"All boxes fitted.")
                                return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1])
                else:
                    # Proximity-based search

                    # Sort boxes by their distance to the cluster
                    nearest_boxes, distances = BoxManipulator._get_boxes_sorted_by_distance(included_boxes,
                                                                                            feasible_boxes)

                    for i, box in enumerate(nearest_boxes):

                        logging.debug(f"Testing box: {box} Distance: {distances[i]}")

                        temp_included_boxes = np.append(included_boxes, [box], axis=0)
                        # temp_cluster_bbox = BoxManipulator.calculate_cluster_bbox(temp_included_boxes)

                        temp_roi = BoxManipulator.get_roi(temp_included_boxes, img_dims, crop_size, handle_overflow,
                                                          max_expansion_limit, margin)

                        if temp_roi is not None and len(temp_included_boxes) > max_boxes_included:
                            best_roi = temp_roi
                            best_included_boxes = temp_included_boxes
                            max_boxes_included = len(temp_included_boxes)

                            logging.debug(f"Better ROI found. Number of boxes: {len(temp_included_boxes)}")

                        if max_boxes_included == total_boxes:
                            logging.debug(f"All boxes fitted.")
                            return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1])

        return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1])

    @staticmethod
    def _get_boxes_sorted_by_distance(cluster_boxes: Union[DetectionBoxes, np.ndarray],
                                      feasible_boxes: Union[DetectionBoxes, np.ndarray],
                                      number_limit: int = 0):
        """
        Sort the feasible boxes by their distance to the cluster limited in number by the number_limit.

        :param cluster_boxes: Union[DetectionBoxes, np.ndarray] - The boxes that are already included in the cluster
        :param feasible_boxes:  Union[DetectionBoxes, np.ndarray] - The boxes that are not included in the cluster
        :param number_limit: int - The maximum number of boxes to return

        :return: nearest_boxes: np.ndarray - The boxes sorted by their distance to the cluster
        :return: distances: np.ndarray - The distances of the nearest boxes to the cluster
        """

        # Calculate distances from each feasible box to the cluster
        cluster_bbox = BoxManipulator.get_combined_box(cluster_boxes)
        distances = np.array([boxes_center_center_distance(cluster_bbox, box) for box in feasible_boxes])
        nearest_boxes_indices = np.argsort(distances)

        # Limit the closest boxes to the number_limit if there are more than number_limit
        if len(nearest_boxes_indices) > number_limit > 0:
            nearest_boxes_indices = nearest_boxes_indices[:number_limit]

        # Define new set of the nearest boxes
        nearest_boxes = feasible_boxes[nearest_boxes_indices]

        return nearest_boxes, distances[nearest_boxes_indices]

    @staticmethod
    def _get_boxes_filtered_by_distance(cluster_boxes: Union[DetectionBoxes, np.ndarray],
                                        remaining_boxes: Union[DetectionBoxes, np.ndarray],
                                        handle_overflow: str,
                                        crop_size: Tuple[int, int],
                                        max_expansion_limit: Tuple[int, int]):
        """
        Filter out boxes that are obviously outside the feasible range for inclusion in the ROI,
        based on the furthest edges in x and y dimensions.
        """
        # Diag info
        logging.debug(
            f"Remaining boxes BEFORE filtering: {remaining_boxes.xyxy if isinstance(remaining_boxes, DetectionBoxes) else remaining_boxes}")

        feasible_boxes = []

        # Determine the bounds for expansion based on the overflow policy
        if handle_overflow == "expand":
            max_width, max_height = max_expansion_limit
        else:  # handle_overflow == "ignore"
            max_width, max_height = crop_size

        cluster_x_min, cluster_y_min, cluster_x_max, cluster_y_max = BoxManipulator.get_combined_box(cluster_boxes)

        # Calculate the maximum allowable expansion distances from the cluster bbox
        max_expand_x = max_width - (cluster_x_max - cluster_x_min)
        max_expand_y = max_height - (cluster_y_max - cluster_y_min)

        for box in remaining_boxes:
            # Check if the box can fit within the expanded limits
            if (box[2] <= cluster_x_max + max_expand_x and box[0] >= cluster_x_min - max_expand_x) and \
                    (box[3] <= cluster_y_max + max_expand_y and box[1] >= cluster_y_min - max_expand_y):
                feasible_boxes.append(box)

        # Diag info
        logging.debug(f"Remaining boxes AFTER filtering: {feasible_boxes}")

        return np.array(feasible_boxes)

    @staticmethod
    def get_combined_box(boxes: Union[DetectionBoxes, np.ndarray]):
        """
        Combine bounding boxes into a single bounding box that encompasses all.
        The boxes must be in the format (x_min, y_min, x_max, y_max).
        Note that any additional info such as cls, prob etc. will be lost.

        Parameters:
        boxes (Union[DetectionBoxes, np.ndarray]): The bounding boxes to combine.

        Returns:
        np.ndarray: A bounding box that encompasses both input boxes.
        """

        if isinstance(boxes, DetectionBoxes):
            boxes = boxes.xyxy
        else:
            boxes = try_convert_to_numpy(boxes)

        x_min, y_min = np.min(boxes, axis=0)[:2]
        x_max, y_max = np.max(boxes, axis=0)[2:4]
        return np.array([x_min, y_min, x_max, y_max])

    # @staticmethod
    # def calculate_distance_to_cluster(cluster_boxes: Union[DetectionBoxes, np.ndarray],
    #                                   box: Optional[Union[List, Tuple, np.ndarray]]):
    #
    #     # Calculate the cluster
    #     cluster_bbox = BoxManipulator.calculate_cluster_bbox(cluster_boxes)
    #
    #     return boxes_center_center_distance(cluster_bbox, box)
    #
    #     # # Calculate the centroid of the cluster bounding box
    #     # cluster_centroid = [(cluster_bbox[0] + cluster_bbox[2]) / 2, (cluster_bbox[1] + cluster_bbox[3]) / 2]
    #     # # Calculate the centroid of the box
    #     # box_centroid = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    #     #
    #     # # Calculate Euclidean distance between the centroids
    #     # distance = np.sqrt((cluster_centroid[0] - box_centroid[0]) ** 2 + (cluster_centroid[1] - box_centroid[1]) ** 2)
    #     #
    #     # return distance

    @staticmethod
    def get_clusters(detection_boxes: Union[DetectionBoxes, np.ndarray],
                     eps: int = 30,
                     min_samples: int = 1,
                     metric: Callable = boxes_center_edge_distance):
        """
        Analyze clusters using DBSCAN with a custom metric.

        Args:
        - detection_boxes (DetectionBoxes): An instance of DetectionBoxes containing bounding boxes.
        - eps (int): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        - metric (function): A custom metric function to calculate the distance between two bounding boxes.

        Returns:
        - Tuple[DetectionBoxes or np.ndarray, dict]: A DetectionBoxes instance or a np.ndarray with boxes around each cluster,
          and a dictionary with cluster labels as keys and DetectionBoxes instances or np.ndarrays of boxes belonging to each cluster.
        """

        if isinstance(detection_boxes, DetectionBoxes):
            return_detection_boxes = True
            try:
                bboxes = detection_boxes.data
            except AttributeError:
                raise AttributeError("DetectionBoxes instance does not have the expected 'xyxy' attribute.")
        elif isinstance(detection_boxes, np.ndarray):
            return_detection_boxes = False
            bboxes = detection_boxes
        else:
            raise TypeError(f"Expected a DetectionBoxes instance, got {type(detection_boxes)} instead")

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(bboxes)
        labels = clustering.labels_

        cluster_dict = {}

        # If no clusters could be found even though there are boxes (happens when boxes are too large to meet
        # the eps requirement as individual boxes) then create a dictionary with clusters being the boxes.
        if np.all(np.array(labels) == -1):
            for i, box in enumerate(bboxes):
                cluster_dict[i] = DetectionBoxes(box, detection_boxes.orig_shape) if return_detection_boxes else box[None, :]

        for label in np.unique(labels):
            if label == -1:
                continue  # Ignore noise
            cluster_bboxes = bboxes[labels == label]
            cluster_dict[label] = DetectionBoxes(cluster_bboxes, detection_boxes.orig_shape) if return_detection_boxes else cluster_bboxes

        # Create bounding boxes for each cluster
        cluster_boxes = []
        for cluster_label, cluster_detection_boxes in cluster_dict.items():
            cluster_bboxes = cluster_detection_boxes.xyxy if isinstance(cluster_detection_boxes, DetectionBoxes) else cluster_detection_boxes
            x_min, y_min = np.min(cluster_bboxes[:, :2], axis=0)
            x_max, y_max = np.max(cluster_bboxes[:, 2:4], axis=0)
            cluster_boxes.append([x_min, y_min, x_max, y_max])
        if not len(cluster_boxes) > 0:
            cluster_detection_boxes = None
        elif return_detection_boxes:
            cluster_detection_boxes = DetectionBoxes(np.array(cluster_boxes), detection_boxes.orig_shape)
        else:
            cluster_detection_boxes = np.array(cluster_boxes)

        return cluster_detection_boxes, cluster_dict

    @staticmethod
    def get_union_of_boxes(boxes: np.ndarray, max_width, max_height):
        """
        Create a mask representing the union of multiple bounding boxes.

        Parameters:
            boxes (np.ndarray): Array of bounding boxes (xyxy).
            max_width (int): The width of the mask.
            max_height (int): The height of the mask.

        Returns:
            np.ndarray: Binary mask representing the union of the bounding boxes.
        """
        if len(boxes) == 0:
            return np.zeros((max_height, max_width), dtype=np.uint8)

        # Create an empty mask
        mask = np.zeros((max_height, max_width), dtype=np.uint8)

        # Draw the boxes on the mask
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            mask[y1:y2, x1:x2] = 1

        return mask

    @staticmethod
    def get_coverage(interest_bboxes: Union[Tuple, List, np.ndarray, DetectionBoxes], reference_bboxes: Union[Tuple, List, np.ndarray, DetectionBoxes]):
        """
        Quantify the exact proportion of the area of the bounding boxes of interest that is covered by reference boxes.

        Parameters:
            interest_bboxes (np.ndarray): Array of bounding boxes (xyxy) for areas of interest.
            reference_bboxes (np.ndarray): Array of bounding boxes (xyxy) for areas of reference boxes.

        Returns:
            float: Proportion of area of interest that is covered by thew reference boxes.
        """

        # Ensure the input boxes are numpy arrays
        results = {}
        for boxes_name in ['interest_bboxes', 'reference_bboxes']:
            boxes = locals()[boxes_name]
            results[boxes_name] = boxes
            if isinstance(boxes, DetectionBoxes):
                results[boxes_name] = boxes.xyxy
            elif isinstance(boxes, (list, tuple)):
                results[boxes_name] = np.array(boxes)
            elif isinstance(boxes, np.ndarray):
                pass
            else:
                raise ValueError(f"Invalid input type for {boxes_name}")

        interest_bboxes = results['interest_bboxes']
        reference_bboxes = results['reference_bboxes']

        # Determine the size of the image that can contain all boxes
        max_x = int(max(np.max(interest_bboxes[:, [0, 2]]), np.max(reference_bboxes[:, [0, 2]])))
        max_y = int(max(np.max(interest_bboxes[:, [1, 3]]), np.max(reference_bboxes[:, [1, 3]])))

        # Create masks for the union of interest areas and reference areas
        interest_mask = BoxManipulator.get_union_of_boxes(interest_bboxes, max_x, max_y)
        reference_mask = BoxManipulator.get_union_of_boxes(reference_bboxes, max_x, max_y)

        # Calculate the total area of interest
        total_interest_area = np.sum(interest_mask)

        # Calculate the intersection of interest areas and reference areas
        intersection_mask = cv2.bitwise_and(interest_mask, reference_mask)
        total_covered_area = np.sum(intersection_mask)

        if total_interest_area == 0:
            return 0.0

        return total_covered_area / total_interest_area

    @staticmethod
    def get_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.

        Args:
        - box1 (np.ndarray): First bounding box in xyxy format.
        - box2 (np.ndarray): Second bounding box in xyxy format.
        """

        x_min1, y_min1, x_max1, y_max1 = box1[:4]
        x_min2, y_min2, x_max2, y_max2 = box2[:4]

        # Calculate the intersection rectangle
        inter_x_min = max(x_min1, x_min2)
        inter_y_min = max(y_min1, y_min2)
        inter_x_max = min(x_max1, x_max2)
        inter_y_max = min(y_max1, y_max2)

        # Calculate area of intersection rectangle
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate the area of both bounding boxes
        box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

        # Calculate the area of the union of both bounding boxes
        union_area = box1_area + box2_area - inter_area

        # Calculate Intersection over Union (IoU)
        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def get_overlap(box1, box2):
        """
        Calculate the overlap of two bounding boxes. The overlap is defined as the ratio of the intersection area to the
         area of the first box.

        Args:
        - box1 (np.ndarray): First bounding box in xyxy format.
        - box2 (np.ndarray): Second bounding box in xyxy format.
        """
        x1, y1, x2, y2 = box1[:4]
        x3, y3, x4, y4 = box2[:4]

        # Calculate the intersection rectangle
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate the area of the first bounding box
        box1_area = (x2 - x1) * (y2 - y1)
        return inter_area / box1_area if box1_area != 0 else 0

    @staticmethod
    def get_boxes_distribution_index(bboxes, img_size, grid_size=(3, 3)):
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
            start_col = max(int(bbox[0] / grid_width), 0)
            end_col = min(int(np.ceil(bbox[2] / grid_width)), grid_size[0]) - 1
            start_row = max(int(bbox[1] / grid_height), 0)
            end_row = min(int(np.ceil(bbox[3] / grid_height)), grid_size[1]) - 1

            occupancy_grid[start_row:end_row + 1, start_col:end_col + 1] = True

        # Calculate the distribution index
        non_empty_cells = np.sum(occupancy_grid)
        total_cells = np.prod(grid_size)
        distribution_index = non_empty_cells / total_cells

        return distribution_index

    @staticmethod
    def get_boxes_summary(detection_boxes: DetectionBoxes):
        areas = []
        for box in detection_boxes.xyxy:
            # Calculate the area of each bounding box
            areas.append(BoxManipulator.get_box_area(box))

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
    def get_boxes_filtered_by_color(frame_bboxes: List[DetectionBoxes], frames: List[np.ndarray], k=3):
        """ Filters bounding boxes based on color analysis. """
        from detectflow.image.color import get_image_dominant_color

        color_filtered_boxes = []
        for frame, bboxes in zip(frames, frame_bboxes):
            if bboxes is None:
                color_filtered_boxes.append(None)
                continue
            filtered_boxes = []
            orig_shape = bboxes.orig_shape
            for bbox in bboxes.data:
                bbox_area = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                dom_color = get_image_dominant_color(bbox_area, k=k)
                if dom_color is not None:
                    filtered_boxes.append(bbox)
                    print(dom_color)
            fil = None if len(filtered_boxes) == 0 else DetectionBoxes(np.array(filtered_boxes), orig_shape)
            color_filtered_boxes.append(fil)
        return color_filtered_boxes

    @staticmethod
    def is_similar(box1, box2, iou_threshold: Union[float, int] = 0):
        """
        Check if two boxes overlap with an option to specify an overlap threshold.
        If no threshold is passed, any overlap is detected.
        A threshold of 1.0 checks whether they are identical.
        Accepts boxes in the format "xyxy".
        """

        # Calculate Intersection over Union (IoU)
        iou = BoxManipulator.get_iou(box1, box2)

        # Check if IoU exceeds the overlap_threshold
        return iou > iou_threshold

    @staticmethod
    def is_overlap(box1, box2, overlap_threshold: Union[float, int] = 0):
        """
        Check if a box overlaps with another with an option to specify an overlap threshold.
        If no threshold is passed, any overlap is detected.
        A threshold of 1.0 checks whether the entire first box overlaps with the second.
        Accepts boxes in the format "xyxy".
        """

        # Calculate overlap, how much of the first box is overlapping with the second
        iou = BoxManipulator.get_overlap(box1, box2)

        # Check if IoU exceeds the overlap_threshold
        return iou > overlap_threshold

    @staticmethod
    def is_close(box1, box2, threshold=20, min_or_max="min", metric='cc'):
        """
        Check if two bounding boxes are close based on distance of centers.

        Args:
        - box1 (tuple, list, np.ndarray): Bounding box in xyxy format.
        - box2 (tuple, list, np.ndarray): Bounding box in xyxy format.
        - threshold (float): Maximum distance between centers for boxes to be considered close.
        - min_or_max (str): Whether to use the maximum or minimum dimension for distance calculation.
        - metric (str): Metric to use for distance calculation ('cc' for center-to-center, 'ce' for center-to-edge, 'ee' for edge-to-edge).
        Returns:
        - bool: True if the boxes are close, False otherwise.
        """

        if metric == 'cc':
            distance = boxes_center_center_distance(box1, box2)
        elif metric == 'ce':
            distance = boxes_center_edge_distance(box1, box2, min_or_max=min_or_max)
        else:
            distance = boxes_edge_edge_distance(box1, box2, min_or_max=min_or_max)
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
    def adjust_dims_to_fit(roi_size: Union[List, Tuple, np.ndarray], max_size: Union[List, Tuple, np.ndarray]):
        """Adjust box_size to fit within max_size while maintaining aspect ratio."""
        aspect_ratio = roi_size[0] / roi_size[1]
        new_width, new_height = roi_size

        if new_width > max_size[0]:
            new_width = max_size[0]
            new_height = int(new_width / aspect_ratio)
        if new_height > max_size[1]:
            new_height = max_size[1]
            new_width = int(new_height * aspect_ratio)

        # Ensure the adjusted dimensions do not exceed the image dimensions
        new_width = min(new_width, max_size[0])
        new_height = min(new_height, max_size[1])

        return new_width, new_height

    @staticmethod
    def adjust_box_to_fit(boxes: np.ndarray, shape: Tuple[int,int]):
        """
        Takes a numpy array of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

        :param boxes: (np.ndarray) Array with the bounding boxes to clip.
        :param shape: (Tuple[int,int]) Tuple containing the shape of the image.
        :return (numpy.ndarray): Clipped boxes
        """
        boxes = try_convert_to_numpy(boxes)

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    @staticmethod
    def adjust_results_to_roi(detection_results: DetectionResults, roi: tuple) -> DetectionResults:
        """
        Adjusts bounding box coordinates in DetectionResults according to the specified ROI.
        Modifies the DetectionResults objects in place.

        :param detection_results: DetectionResults object.
        :param roi: Region of Interest coordinates in the format (x_min, y_min, x_max, y_max).
        :return DetectionResults: Instance with updated bounding boxes.
        """
        if not isinstance(roi, tuple) or len(roi) != 4:
            raise ValueError("ROI must be a tuple of four elements (x_min, y_min, x_max, y_max).")

        if not isinstance(detection_results, DetectionResults):
            raise TypeError(f"Expected a DetectionResults instance, got {type(detection_results)} instead")

        if detection_results.boxes is None:
            return detection_results

        try:
            new_boxes = BoxManipulator.adjust_boxes_to_roi(detection_results.boxes, roi)
            detection_results.boxes = new_boxes  # Update boxes in place
        except Exception as e:
            print(f"Error adjusting boxes: {str(e)}")
            # Optionally handle the error or leave the result as is
            detection_results.boxes = None

        return detection_results

    @staticmethod
    def adjust_boxes_to_roi(boxes: DetectionBoxes, roi: tuple) -> Union[DetectionBoxes, None]:
        """
        Adjusts the coordinates of bounding boxes based on the ROI.

        :param boxes: DetectionBoxes object containing the bounding boxes.
        :param roi: Region of Interest coordinates (x_min, y_min, x_max, y_max).
        :return: DetectionBoxes object with adjusted boxes.
        """

        if boxes is not None:
            x_min_roi, y_min_roi, x_max_roi, y_max_roi = roi
            adjusted_boxes = []

            for box in boxes.data:
                x_min, y_min, x_max, y_max = box[:4]
                # Adjust coordinates relative to the ROI
                new_x_min = max(x_min - x_min_roi, 0)
                new_y_min = max(y_min - y_min_roi, 0)
                new_x_max = min(x_max - x_min_roi, x_max_roi - x_min_roi)
                new_y_max = min(y_max - y_min_roi, y_max_roi - y_min_roi)
                # Ensure the box is within the ROI
                if new_x_min < new_x_max and new_y_min < new_y_max:
                    adjusted_box = [new_x_min, new_y_min, new_x_max, new_y_max] + list(box[4:])  # Additional data is preserved
                    adjusted_boxes.append(adjusted_box)

            return DetectionBoxes(np.array(adjusted_boxes), (x_max_roi - x_min_roi, y_max_roi - y_min_roi))
        else:
            return None

    @staticmethod
    def adjust_results_for_resize(detection_results: DetectionResults, new_shape: tuple) -> DetectionResults:
        """
        Adjusts bounding box coordinates in DetectionResults according to the new image size.
        Modifies the DetectionResults objects in place.

        :param detection_results: DetectionResults object.
        :param new_shape: New image size in the format (width, height).
        :return: DetectionResults with updated bounding boxes.
        """
        if not isinstance(new_shape, tuple) or len(new_shape) != 2:
            raise ValueError("New shape must be a tuple of two elements (width, height).")

        if not isinstance(detection_results, DetectionResults):
            raise TypeError(f"Expected a DetectionResults instance, got {type(detection_results)} instead")

        if detection_results.boxes is None:
            return detection_results

        try:
            new_boxes = BoxManipulator.adjust_boxes_for_resize(detection_results.boxes, detection_results.orig_shape,
                                                               new_shape)
            detection_results.boxes = new_boxes  # Update boxes in place
        except Exception as e:
            print(f"Error adjusting boxes for resize: {str(e)}")
            # Optionally handle the error or leave the result as is
            detection_results.boxes = None

        return detection_results

    @staticmethod
    def adjust_boxes_for_resize(boxes: DetectionBoxes, orig_shape: tuple, new_shape: tuple) -> Union[DetectionBoxes, None]:
        """
        Adjusts the coordinates of bounding boxes based on the new image size.

        :param boxes: DetectionBoxes object containing the bounding boxes.
        :param orig_shape: Original shape of the image (height, width).
        :param new_shape: New shape of the image (width, height).
        :return: DetectionBoxes object with adjusted boxes.
        """

        if boxes is not None:

            orig_height, orig_width = orig_shape
            new_width, new_height = new_shape
            width_ratio = new_width / orig_width
            height_ratio = new_height / orig_height

            adjusted_boxes = []
            for box in boxes.data:
                x_min, y_min, x_max, y_max = box[:4]
                # Scale coordinates relative to the new size
                new_x_min = x_min * width_ratio
                new_y_min = y_min * height_ratio
                new_x_max = x_max * width_ratio
                new_y_max = y_max * height_ratio
                adjusted_box = [new_x_min, new_y_min, new_x_max, new_y_max] + list(box[4:])  # Additional data is preserved
                adjusted_boxes.append(adjusted_box)

            return DetectionBoxes(np.array(adjusted_boxes), new_shape)
        else:
            return None

    @staticmethod
    def remove_contained_boxes(boxes: Union[DetectionBoxes, np.ndarray, List, Tuple]):
        non_overlapping_boxes = []
        orig_shape = None

        if isinstance(boxes, DetectionBoxes):
            orig_shape = boxes.orig_shape
            boxes = boxes.data
            return_detection_boxes = True
        else:
            return_detection_boxes = False

        for i in range(len(boxes)):
            box1 = boxes[i]
            is_contained = False
            for j in range(len(boxes)):
                if i != j:
                    box2 = boxes[j]
                    if BoxManipulator.is_contained(box1, box2):
                        is_contained = True
                        break
            if not is_contained:
                non_overlapping_boxes.append(box1)

        return DetectionBoxes(np.array(non_overlapping_boxes), orig_shape) if return_detection_boxes else np.array(non_overlapping_boxes)

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
            box_tuple = tuple(box[:min(len(box)-1, 6)])  # Exclude track ID for comparison
            if box_tuple not in seen:
                seen.add(box_tuple)
                unique_boxes.append(box)
        return unique_boxes

    @staticmethod
    def find_consistent_boxes(detection_results: List[Union[DetectionResults, DetectionBoxes]], iou_threshold=0.5, min_frames=3):
        """
        Identifies and returns bounding boxes that consistently appear across multiple frames based on Intersection over
        Union (IoU) criteria. This method iterates through each detection result, comparing bounding boxes across all
        results to find those that appear in a minimum number of frames specified by `min_frames`. A bounding box is
        considered consistent if it has an IoU greater than `iou_threshold` with boxes in other frames.

        Args:
            detection_results (List[Union[DetectionResults, DetectionBoxes]]): A list of DetectionResults or
                                                                               DetectionBoxes instances containing the
                                                                               detection results for each frame.
            iou_threshold (float): The IoU threshold for considering two boxes as the same object.
                                   Boxes with IoU above this threshold are considered matches.
            min_frames (int): The minimum number of frames in which a box must appear to be considered consistent.

        Returns:
            List[Union[DetectionResults, DetectionBoxes]]: A list of DetectionResults or DetectionBoxes instances,
                                                           each containing only the bounding boxes that were found to be
                                                           consistent across the specified number of frames.
        Note:
            - The method modifies the input `detection_results` by filtering out inconsistent boxes.
            - Boxes are compared using their IoU, and a box is considered consistent if it appears with an IoU above the
              threshold in at least `min_frames` different frames.
            - The original shape of the boxes is preserved in the output.
        """
        consistent_results = []
        # For each result get boxes object
        for i, current_result in enumerate(detection_results):
            consistent_boxes_current = []

            # Continue only if the boxes object is not None
            current_result_boxes = current_result.boxes if isinstance(current_result, DetectionResults) else (current_result if isinstance(current_result, DetectionBoxes) else None)
            if current_result_boxes is None:
                continue

            # Get the original shape of the boxes
            orig_shape = current_result_boxes.orig_shape

            # For each box in the current boxes object
            for current_box in current_result_boxes.data:
                count = 0

                # Loop over all other results
                for other_result in detection_results:

                    # Continue only if the boxes of the other result is not None
                    other_result_boxes = other_result.boxes if isinstance(other_result, DetectionResults) else (other_result if isinstance(other_result, DetectionBoxes) else None)
                    if other_result_boxes is None:
                        continue

                    # Loop over all bboxes in the other boxes object
                    for other_box in other_result_boxes.xyxy:
                        if BoxManipulator.get_iou(current_box, other_box) > iou_threshold:
                            count += 1
                            break
                if count >= min_frames:
                    consistent_boxes_current.append(tuple(current_box))

            # Create an updated boxes object and assign it to the current results boxes attribute
            det = None if len(consistent_boxes_current) == 0 else DetectionBoxes(consistent_boxes_current, orig_shape)
            if isinstance(current_result, DetectionResults):
                current_result.boxes = det
            elif isinstance(current_result, DetectionBoxes):
                current_result = det
            consistent_results.append(current_result)
        return consistent_results
        # return list(set(consistent_boxes))  # Remove duplicates

    @staticmethod
    def find_moving_boxes(detection_results: List[DetectionResults], tracker: Optional[Tracker] = None, movement_threshold=10):
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
            tracked_results = []
            for result in detection_results:
                tracked_result = tracker.process_tracking(result, filter=True)
                tracked_results.append(tracked_result)

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
                            distance = np.linalg.norm(np.array(BoxManipulator.get_box_center(box)) - np.array(
                                BoxManipulator.get_box_center(first_frame_box)))
                            if distance > movement_threshold:
                                moving_boxes_from_first_frame.append(first_frame_box)
                                del track_id_to_first_frame_box[track_id]  # Remove track ID to prevent duplicates

            # Remove duplicates using a custom function
            unique_moving_boxes = BoxManipulator.remove_duplicate_boxes(moving_boxes_from_first_frame)
            mov = None if len(unique_moving_boxes) == 0 else DetectionBoxes(unique_moving_boxes,
                                                                            detection_results[0].boxes.orig_shape)
            detection_results[0].boxes = mov
            result = None if mov is None else detection_results[0]
        return result

    @staticmethod
    def find_outlier_boxes(frames_bboxes: List[DetectionBoxes], size_threshold=1.5, central_tendency='mean'):
        """
        Find outlier boxes in terms of size.
        :param frames_bboxes: List of bounding boxes for each frame.
        :param size_threshold: Threshold for determining outliers.
        :param central_tendency: Method for calculating central tendency ('mean' or 'median').
        """
        # Calculate the metrics for all bboxes
        all_boxes = [box for frame in frames_bboxes for box in frame.xyxy]
        areas = [BoxManipulator.get_box_area(box) for box in all_boxes]
        if central_tendency == 'median':
            central_area = np.median(areas)
        else:
            central_area = np.mean(areas)
        std_area = np.std(areas)

        # For each frame detection boxes get the outliers
        outliers = []
        for frame_bboxes in frames_bboxes:
            if frame_bboxes is None:
                outliers.append(None)
                continue
            orig_shape = frame_bboxes.orig_shape
            outlier_boxes = [box for box in frame_bboxes.data if
                             abs(BoxManipulator.get_box_area(box) - central_area) > size_threshold * std_area]
            out = None if len(outlier_boxes) == 0 else DetectionBoxes(outlier_boxes, orig_shape)
            outliers.append(out)

        return outliers

    @staticmethod
    def coco_to_xyxy(boxes_coco):
        """
        Convert a single bounding box or an array of bounding boxes from COCO format to xyxy format.
        Handles both individual boxes (1D arrays) and arrays of boxes (2D arrays).

        Parameters:
        boxes_coco (np.ndarray): An array where each row (or the array itself if a single box)
                                 is a bounding box in the format (x_min, y_min, width, height, [cls, prob, id]).

        Returns:
        np.ndarray: An array where each row (or the array itself if a single box)
                    is a bounding box in the format (x_min, y_min, x_max, y_max, [cls, prob, id]).
        """

        if not isinstance(boxes_coco, np.ndarray):
            try:
                boxes_coco = np.array(boxes_coco)
            except Exception as e:
                raise ValueError(f"Invalid input type for boxes: {str(e)}")

        if boxes_coco.ndim == 1:
            # It's a single box, reshape it to a 2D array with one row
            boxes_coco = boxes_coco.reshape(1, -1)

        # Extract bounding box dimensions
        x_min = boxes_coco[:, 0]
        y_min = boxes_coco[:, 1]
        width = boxes_coco[:, 2]
        height = boxes_coco[:, 3]

        # Calculate x_max and y_max from COCO format
        x_max = x_min + width
        y_max = y_min + height

        # Prepare the new xyxy format boxes array
        # Concatenate the new x_max, y_max with original x_min, y_min and any additional columns
        boxes_xyxy = np.hstack((boxes_coco[:, :2], x_max[:, None], y_max[:, None], boxes_coco[:, 4:]))

        if boxes_xyxy.shape[0] == 1:
            # If there was originally only one box, return it as a 1D array
            return boxes_xyxy[0]
        else:
            return boxes_xyxy

    @staticmethod
    def xyxy_to_coco(boxes_xyxy):
        """
        Convert a single bounding box or an array of bounding boxes from xyxy format to COCO format.
        Handles both individual boxes (1D arrays) and arrays of boxes (2D arrays), preserving additional data.

        Parameters:
        boxes_xyxy (np.ndarray): An array where each row (or the array itself if a single box)
                                 is a bounding box with potential additional data,
                                 in the format (x_min, y_min, x_max, y_max, [cls, prob, id]).

        Returns:
        np.ndarray: An array where each row (or the array itself if a single box)
                    is a bounding box in the COCO format (x_min, y_min, width, height, [cls, prob, id]).
        """

        if not isinstance(boxes_xyxy, np.ndarray):
            try:
                boxes_xyxy = np.array(boxes_xyxy)
            except Exception as e:
                raise ValueError(f"Invalid input type for boxes: {str(e)}")

        if boxes_xyxy.ndim == 1:
            # It's a single box, reshape it to a 2D array with one row
            boxes_xyxy = boxes_xyxy.reshape(1, -1)

        # Calculate width and height from xyxy format
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # x_max - x_min
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # y_max - y_min

        # Form the COCO format boxes array, preserving any additional columns
        if boxes_xyxy.shape[1] > 4:
            additional_data = boxes_xyxy[:, 4:]  # Extract additional data if present
            boxes_coco = np.hstack((boxes_xyxy[:, :2], widths[:, None], heights[:, None], additional_data))
        else:
            boxes_coco = np.hstack((boxes_xyxy[:, :2], widths[:, None], heights[:, None]))

        if boxes_coco.shape[0] == 1:
            # If there was originally only one box, return it as a 1D array
            return boxes_coco[0]
        else:
            return boxes_coco

    @staticmethod
    def xyxy_to_xywh(boxes_xyxy):
        """
        Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, boxes_xywh, width, height) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            boxes_xyxy (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.

        Returns:
            boxes_xywh (np.ndarray): The bounding box coordinates in (x, boxes_xywh, width, height) format.
        """
        if not boxes_xyxy.shape[-1] == 4:
            raise ValueError(f"Expected input shape with last dimension 4, got {boxes_xyxy.shape}")

        boxes_xywh = np.empty_like(boxes_xyxy)
        boxes_xywh[..., 0] = (boxes_xyxy[..., 0] + boxes_xyxy[..., 2]) / 2  # x center
        boxes_xywh[..., 1] = (boxes_xyxy[..., 1] + boxes_xyxy[..., 3]) / 2  # boxes_xywh center
        boxes_xywh[..., 2] = boxes_xyxy[..., 2] - boxes_xyxy[..., 0]  # width
        boxes_xywh[..., 3] = boxes_xyxy[..., 3] - boxes_xyxy[..., 1]  # height

        return boxes_xywh

    @staticmethod
    def xywh_to_xyxy(boxes_xywh):
        """
        Convert bounding box coordinates from (x, boxes_xyxy, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            boxes_xywh (np.ndarray): The input bounding box coordinates in (x, boxes_xyxy, width, height) format.

        Returns:
            boxes_xyxy (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        if not boxes_xywh.shape[-1] == 4:
            raise ValueError(f"Expected input shape with last dimension 4, got {boxes_xywh.shape}")

        boxes_xyxy = np.empty_like(boxes_xywh)
        dw = boxes_xywh[..., 2] / 2  # half-width
        dh = boxes_xywh[..., 3] / 2  # half-height
        boxes_xyxy[..., 0] = boxes_xywh[..., 0] - dw  # top left x
        boxes_xyxy[..., 1] = boxes_xywh[..., 1] - dh  # top left boxes_xyxy
        boxes_xyxy[..., 2] = boxes_xywh[..., 0] + dw  # bottom right x
        boxes_xyxy[..., 3] = boxes_xywh[..., 1] + dh  # bottom right boxes_xyxy

        return boxes_xyxy
