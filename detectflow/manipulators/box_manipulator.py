import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import logging
from itertools import permutations
from scipy.spatial.distance import euclidean
from typing import List, Tuple
from detectflow.predict.results import DetectionBoxes


class BoxManipulator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_optimal_roi(detection_boxes: DetectionBoxes,
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
                              iou_threshold: float = 0.5):
        '''
        Note that img_dims format is (width, height)
        '''
        empty = False

        if detection_boxes is None:

            if not ignore_empty:
                # Create a cewnter dummy box to create a central ROI for empty detection boxes
                dummy_boxes = np.array([[max(0, img_dims[0] // 2 - 10), max(0, img_dims[1] // 2 - 10),
                                         min(img_dims[0], img_dims[0] // 2 + 10),
                                         min(img_dims[1], img_dims[1] // 2 + 10)]])
                detection_boxes = DetectionBoxes(dummy_boxes, img_dims[::-1], 'xyxy')
                empty = True
            else:
                return None

        if not isinstance(detection_boxes, DetectionBoxes):
            raise TypeError(f"Expected a DetectionBoxes instance, got {type(detection_boxes)} instead")

        try:
            bboxes = detection_boxes.xyxy
        except AttributeError:
            raise AttributeError(f"DetectionBoxes instance does not have the expected 'xyxy' attribute.")

        # Adjust max expansion aspect ratio to match the crop size aspect ratio
        max_expansion_limit = BoxManipulator.match_aspect_ratio(crop_size, max_expansion_limit)

        # Automatically limit crop size and max expansion size depending on the size of the image
        crop_size = BoxManipulator.adjust_size_to_img(crop_size, img_dims)
        max_expansion_limit = BoxManipulator.adjust_size_to_img(max_expansion_limit, img_dims)

        # Initiate remaining_boxes as all boxesand list of rois
        remaining_boxes = detection_boxes
        done_boxes = []
        rois = []

        while len(remaining_boxes) > 0:

            logging.debug(f"Analysing boxes: {remaining_boxes.xyxy}")
            logging.debug(f"Done boxes: {done_boxes}")

            # Step 1: Analyze clusters
            dynamic_eps = BoxManipulator.dynamic_eps_calculation(crop_size, handle_overflow)
            min_samples = max(1, len(remaining_boxes) // 20)  # Example dynamic calculation
            cluster_detection_boxes, cluster_dict = BoxManipulator.analyze_clusters(remaining_boxes, eps=dynamic_eps,
                                                                                    min_samples=min_samples)

            logging.debug(f"Cluster dict: {cluster_dict}")

            # Step 2: Sort clusters by importance (number of boxes)
            sorted_clusters = sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)

            # Iterate over clusters to find the best ROI
            best_roi, boxes_included = BoxManipulator.expand_cluster(remaining_boxes, sorted_clusters, img_dims,
                                                                     crop_size, handle_overflow, max_expansion_limit,
                                                                     margin, exhaustive_search=exhaustive_search,
                                                                     permutation_limit=permutation_limit)

            logging.debug(f"Found a ROI: {best_roi} with boxes {boxes_included.xyxy}")

            # Append the resulting list
            rois.append((best_roi, boxes_included if not empty else None))

            # Append the done boxes
            done_boxes = done_boxes + [box for box in np.array(boxes_included.xyxy)]

            # If partial overlap is allowed boxes that overlap partially with the roi (with some threshold) will be treated as included in the roi
            if partial_overlap:
                for box in np.array([bbox for bbox in remaining_boxes if bbox not in np.array(done_boxes)]):
                    # Check if more than threshold fraction of the bbox is within the roi
                    if box is not None:
                        if BoxManipulator.is_overlap(box, best_roi, overlap_threshold=iou_threshold):
                            logging.debug(f"A partial overlap detected with a box: {box}")
                            done_boxes = done_boxes + [box]

            # Get boxes remaining after finding the best crop
            if multiple_rois and len(done_boxes) < len(detection_boxes):
                remaining_boxes = DetectionBoxes(
                    np.array([box for box in detection_boxes if box not in np.array(done_boxes)]), img_dims[::-1],
                    'xyxy')
            else:
                break

        # Return the list of rois and the boxes included in them List[(roi, boxes)]
        return rois

    @staticmethod
    def adjust_size_to_img(size, img_size):
        """Adjust size to fit within img_size while maintaining aspect ratio."""
        aspect_ratio = size[0] / size[1]
        new_width, new_height = size

        if new_width > img_size[0]:
            new_width = img_size[0]
            new_height = int(new_width / aspect_ratio)
        if new_height > img_size[1]:
            new_height = img_size[1]
            new_width = int(new_height * aspect_ratio)

        # Ensure the adjusted dimensions do not exceed the image dimensions
        new_width = min(new_width, img_size[0])
        new_height = min(new_height, img_size[1])

        return new_width, new_height

    @staticmethod
    def match_aspect_ratio(size1, size2):
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
            return (adjusted_width2, size2[1])
        else:
            # Adjusting height is necessary; this will not exceed the original height of size2
            return (size2[0], adjusted_height2)

    @staticmethod
    def dynamic_eps_calculation(crop_size, handle_overflow):
        if handle_overflow == "expand":
            return min(crop_size) // 2
        else:
            return (min(crop_size) // 2) - (min(crop_size) // 20)

    @staticmethod
    def expand_cluster(detection_boxes,
                       sorted_clusters,
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
        all_boxes = np.array([box for cluster in sorted_clusters for box in cluster[1].xyxy])
        total_boxes = len(all_boxes)

        # For each cluster try searching for the best possible ROI by expanding the cluster with more boxes
        for cluster_label, cluster_boxes in sorted_clusters:

            # Get cluster characteristics
            # cluster_boxes = cluster_boxes.xyxy
            # current_cluster_bbox = BoxManipulator.calculate_cluster_bbox(cluster_boxes)
            included_boxes = cluster_boxes.copy()

            # During first iteration assign the first cluster values as the best by default
            max_boxes_included = len(included_boxes) if len(included_boxes) > max_boxes_included else max_boxes_included
            best_roi = BoxManipulator.construct_roi(included_boxes, img_dims, crop_size, handle_overflow,
                                                    max_expansion_limit, margin) if best_roi is None else best_roi
            best_included_boxes = included_boxes if best_included_boxes is None else best_included_boxes

            logging.debug(f"Current cluster: #{cluster_label} - {cluster_boxes.xyxy}")
            logging.debug(f"Current cluster box: {BoxManipulator.calculate_cluster_bbox(cluster_boxes)}")

            # Get boxes remaining out of cluster and filter them by distance for efficient search
            remaining_boxes = np.array([box for box in all_boxes if box not in cluster_boxes])
            feasible_boxes = BoxManipulator.filter_boxes_by_distance(included_boxes, remaining_boxes, handle_overflow,
                                                                     crop_size, max_expansion_limit)

            if len(remaining_boxes) == 0:

                logging.debug(f"All boxes in one cluster.")

                # When there are no reamining boxes, simply construct the roi from the current cluster bbox
                best_roi = BoxManipulator.construct_roi(included_boxes, img_dims, crop_size, handle_overflow,
                                                        max_expansion_limit, margin)
                # TODO: This assumes that ROI can be created around every cluster defined by the clsuter analysis as best_roi being None is not handled

                break

            #                 # TODO: How to decide ties when there are more than one clusters with the same number of boxes and any cant be added to any of the clusters?
            else:
                if exhaustive_search:
                    # Perform exhaustive search through permutations of the remaining boxes

                    # Sort boxes by their distance to the cluster TODO: Should be moved to the BoxAnalyser class
                    nearest_boxes, distances = BoxManipulator.sort_boxes_by_distance(included_boxes, feasible_boxes,
                                                                                     permutation_limit)

                    for n in range(1, len(nearest_boxes) + 1):
                        for permutation in permutations(nearest_boxes, n):

                            temp_included_boxes = np.append(included_boxes, permutation, axis=0)
                            # temp_cluster_bbox = BoxManipulator.calculate_cluster_bbox(temp_included_boxes)

                            temp_roi = BoxManipulator.construct_roi(temp_included_boxes, img_dims, crop_size,
                                                                    handle_overflow, max_expansion_limit, margin)

                            if temp_roi is not None and len(temp_included_boxes) > max_boxes_included:
                                best_roi = temp_roi
                                best_included_boxes = temp_included_boxes
                                max_boxes_included = len(temp_included_boxes)

                                logging.debug(f"Better ROI found. Number of boxes: {len(temp_included_boxes)}")

                            if max_boxes_included == total_boxes:
                                logging.debug(f"All boxes fitted.")
                                return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1], 'xyxy')
                else:
                    # Proximity-based search

                    # Sort boxes by their distance to the cluster TODO: Should be moved to the BoxAnalyser class
                    nearest_boxes, distances = BoxManipulator.sort_boxes_by_distance(included_boxes, feasible_boxes)

                    for i, box in enumerate(nearest_boxes):

                        logging.debug(f"Testing box: {box} Distance: {distances[i]}")

                        temp_included_boxes = np.append(included_boxes, [box], axis=0)
                        # temp_cluster_bbox = BoxManipulator.calculate_cluster_bbox(temp_included_boxes)

                        temp_roi = BoxManipulator.construct_roi(temp_included_boxes, img_dims, crop_size,
                                                                handle_overflow, max_expansion_limit, margin)

                        if temp_roi is not None and len(temp_included_boxes) > max_boxes_included:
                            best_roi = temp_roi
                            best_included_boxes = temp_included_boxes
                            max_boxes_included = len(temp_included_boxes)

                            logging.debug(f"Better ROI found. Number of boxes: {len(temp_included_boxes)}")

                        if max_boxes_included == total_boxes:
                            logging.debug(f"All boxes fitted.")
                            return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1], 'xyxy')

        return best_roi, DetectionBoxes(np.array(best_included_boxes), img_dims[::-1], 'xyxy')

    @staticmethod
    def sort_boxes_by_distance(cluster_boxes: Union[DetectionBoxes, np.ndarray],
                               feasible_boxes: Union[DetectionBoxes, np.ndarray], number_limit: int = 0):

        # Calculate distances from each feasible box to the cluster
        distances = np.array(
            [BoxManipulator.calculate_distance_to_cluster(cluster_boxes, box) for box in feasible_boxes])
        nearest_boxes_indices = np.argsort(distances)

        # Limit to the number_limit closest boxes if there are more than number_limit
        if number_limit > 0 and len(nearest_boxes_indices) > number_limit:
            nearest_boxes_indices = nearest_boxes_indices[:number_limit]

        # Define new set of the nearest boxes
        nearest_boxes = feasible_boxes[nearest_boxes_indices]

        return nearest_boxes, distances[nearest_boxes_indices]

    @staticmethod
    def filter_boxes_by_distance(cluster_boxes: Union[DetectionBoxes, np.ndarray],
                                 remaining_boxes: Union[DetectionBoxes, np.ndarray],
                                 handle_overflow, crop_size, max_expansion_limit):
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

        cluster_x_min, cluster_y_min, cluster_x_max, cluster_y_max = BoxManipulator.calculate_cluster_bbox(
            cluster_boxes)

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
    def calculate_cluster_bbox(cluster_boxes: Union[DetectionBoxes, np.ndarray]):

        if isinstance(cluster_boxes, DetectionBoxes):
            cluster_boxes = cluster_boxes.xyxy

        x_min, y_min = np.min(cluster_boxes, axis=0)[:2]
        x_max, y_max = np.max(cluster_boxes, axis=0)[2:]
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def calculate_distance_to_cluster(cluster_boxes: Union[DetectionBoxes, np.ndarray], box):

        # Calculate the cluster
        cluster_bbox = BoxManipulator.calculate_cluster_bbox(cluster_boxes)

        # Calculate the centroid of the cluster bounding box
        cluster_centroid = [(cluster_bbox[0] + cluster_bbox[2]) / 2, (cluster_bbox[1] + cluster_bbox[3]) / 2]
        # Calculate the centroid of the box
        box_centroid = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        # Calculate Euclidean distance between the centroids
        distance = np.sqrt((cluster_centroid[0] - box_centroid[0]) ** 2 + (cluster_centroid[1] - box_centroid[1]) ** 2)
        return distance

    @staticmethod
    def construct_roi(detection_boxes: Union[DetectionBoxes, np.ndarray], img_dims, crop_size=(640, 640),
                      handle_overflow='expand',
                      max_expansion_limit=(1000, 1000), margin=0):
        """
        Construct an ROI around the given cluster of boxes, adhering to the specified requirements.

        img_dims - (width, height)
        """

        # Calculate the cluster bbox
        cluster_x_min, cluster_y_min, cluster_x_max, cluster_y_max = BoxManipulator.calculate_cluster_bbox(
            detection_boxes)

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

        #         logging.info(f"Centroid: {centroid_x}, {centroid_y}")
        #         logging.info(f"Initial width: {initial_width}, Initial heigth: {initial_height}")

        roi_x_min = max(0, centroid_x - initial_width / 2)
        roi_y_min = max(0, centroid_y - initial_height / 2)

        #         logging.info(f"ROI MIN: {roi_x_min}, {roi_y_min}")
        #         logging.info(f"IMG DIMS: width: {img_dims[0]}, height: {img_dims[1]}")

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

        #         logging.info(f"ROI: {roi_x_min}, {roi_y_min}, {roi_x_max}, {roi_y_max}")

        return roi_x_min, roi_y_min, roi_x_max, roi_y_max

    #################################

    #     @staticmethod
    #     def can_fit_in_region(detection_boxes, crop_size=(640, 640), handle_overflow='expand',
    #                           max_expansion_limit=(1000, 1000), margin=0):
    #         """
    #         Checks if all bounding boxes from a DetectionBoxes instance can fit within a specified region,
    #         with an option to expand the region up to a maximum limit, including a margin around the boxes.

    #         Parameters:
    #         - detection_boxes (DetectionBoxes): An instance of DetectionBoxes containing bounding boxes.
    #         - crop_size (tuple): A tuple (width, height) specifying the size of the region. Default is (640, 640).
    #         - handle_overflow (str): Strategy for handling cases where bounding boxes exceed the region.
    #                                   Options: 'expand' to expand the region, 'ignore' to ignore overflow.
    #         - max_expansion_limit (tuple): Maximum limit (width, height) to which the region can be expanded.
    #         - margin (int): Margin in pixels to be added around each bounding box.

    #         Returns:
    #         - bool: True if all boxes can fit within the specified region or expanded region, False otherwise.
    #         - tuple: The bounding box of the encompassing region, adjusted based on the handle_overflow strategy,
    #                  expansion limit, and margin.
    #         """
    #         if not isinstance(detection_boxes, DetectionBoxes):
    #             raise TypeError(f"Expected a DetectionBoxes instance, got {type(detection_boxes)} instead")

    #         try:
    #             bboxes = detection_boxes.xyxy
    #         except AttributeError:
    #             raise AttributeError(f"DetectionBoxes instance does not have the expected 'xyxy' attribute.")

    #         x_min, y_min = np.min(bboxes[:, :2], axis=0) - margin
    #         x_max, y_max = np.max(bboxes[:, 2:4], axis=0) + margin

    #         aspect_ratio = crop_size[0] / crop_size[1]
    #         region_width, region_height = crop_size

    #         while (x_max - x_min > region_width or y_max - y_min > region_height) and \
    #               (region_width < max_expansion_limit[0] and region_height < max_expansion_limit[1]):
    #             region_width = min(region_width + aspect_ratio, max_expansion_limit[0])
    #             region_height = min(region_height + 1, max_expansion_limit[1])

    #         if x_max - x_min > region_width or y_max - y_min > region_height:
    #             if handle_overflow == 'ignore':
    #                 return False, None
    #             else:
    #                 logging.error("Unable to fit all bounding boxes within the maximum expansion limit.")
    #                 return False, None

    #         x_min = max(0, min(x_min, bboxes[:, 0].max() - region_width))
    #         y_min = max(0, min(y_min, bboxes[:, 1].max() - region_height))
    #         x_max = x_min + region_width
    #         y_max = y_min + region_height

    #         return True, (x_min, y_min, x_max, y_max)

    @staticmethod
    def analyze_clusters(detection_boxes, eps=30, min_samples=1):
        """
        Analyze clusters using DBSCAN with a custom metric.

        Args:
        - detection_boxes (DetectionBoxes): An instance of DetectionBoxes containing bounding boxes.
        - eps (int): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
        - Tuple[DetectionBoxes, dict]: A DetectionBoxes instance with boxes around each cluster,
          and a dictionary with cluster labels as keys and DetectionBoxes instances of boxes belonging to each cluster.
        """
        if not isinstance(detection_boxes, DetectionBoxes):
            raise TypeError(f"Expected a DetectionBoxes instance, got {type(detection_boxes)} instead")

        try:
            bboxes = detection_boxes.xyxy
        except AttributeError:
            raise AttributeError("DetectionBoxes instance does not have the expected 'xyxy' attribute.")

        def custom_metric(bbox1, bbox2):
            return BoxManipulator.bbox_cluster_metric(bbox1, bbox2)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=custom_metric).fit(bboxes)
        labels = clustering.labels_

        cluster_dict = {}

        # If no clusters could be found even though thereare boxes, which happens when boxes are to large to fit
        # the eps requirement even on their own create a dictionary with clusters being the boxes.
        if np.all(np.array(labels) == -1):
            for i, box in enumerate(bboxes):
                cluster_dict[i] = DetectionBoxes(box, detection_boxes.orig_shape, 'xyxy')

        for label in np.unique(labels):
            if label == -1:
                continue  # Ignore noise
            cluster_bboxes = bboxes[labels == label]
            cluster_dict[label] = DetectionBoxes(cluster_bboxes, detection_boxes.orig_shape, 'xyxy')

        # Create bounding boxes for each cluster
        cluster_boxes = []
        for cluster_label, cluster_detection_boxes in cluster_dict.items():
            cluster_bboxes = cluster_detection_boxes.xyxy
            x_min, y_min = np.min(cluster_bboxes[:, :2], axis=0)
            x_max, y_max = np.max(cluster_bboxes[:, 2:4], axis=0)
            cluster_boxes.append([x_min, y_min, x_max, y_max])
        cluster_detection_boxes = DetectionBoxes(np.array(cluster_boxes), detection_boxes.orig_shape, 'xyxy') if len(
            cluster_boxes) > 0 else None

        return cluster_detection_boxes, cluster_dict

    @staticmethod
    def bbox_cluster_metric(bbox1, bbox2):
        """
        Calculate the distance from the center of a bounding box (bbox1) to the furthest point of another bounding box (bbox2).
        Both bounding boxes are obtained from DetectionBoxes instances.

        Args:
        - detection_boxes1 (DetectionBoxes): An instance of DetectionBoxes for the first bounding box.
        - detection_boxes2 (DetectionBoxes): An instance of DetectionBoxes for the second bounding box.

        Returns:
        - float: Distance from the center of bbox1 to the furthest point of bbox2.
        """

        if not (isinstance(bbox1, np.ndarray) and isinstance(bbox2, np.ndarray)):
            raise TypeError(f"Expected np.ndarray instances, got {type(bbox1)} and {type(bbox2)}")

        # Calculate the centers and dimensions
        x1_center, y1_center = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
        x2_center, y2_center = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
        width2, height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        # Calculate the corners of the second bbox
        x2_min = x2_center - width2 / 2
        x2_max = x2_center + width2 / 2
        y2_min = y2_center - height2 / 2
        y2_max = y2_center + height2 / 2

        # Calculate distances from bbox1 center to bbox2 corners
        distances = [
            distance.euclidean((x1_center, y1_center), (x2_min, y2_min)),
            distance.euclidean((x1_center, y1_center), (x2_max, y2_min)),
            distance.euclidean((x1_center, y1_center), (x2_min, y2_max)),
            distance.euclidean((x1_center, y1_center), (x2_max, y2_max))
        ]

        # Return the maximum distance
        return max(distances)

    @staticmethod
    def is_overlap(box1, box2, overlap_threshold=0):
        """
        Check if two boxes overlap with an option to specify an overlap threshold.
        If no threshold is passed, any overlap is detected.
        A threshold of 1.0 checks whether they are identical.
        Accepts boxes in the format "xyxy".
        """
        # TODO: BoxAnalyser already has a function for calculating IoU and it should be used here as well

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
        IoU = inter_area / union_area if union_area > 0 else 0

        # Check if IoU exceeds the overlap_threshold
        return IoU > overlap_threshold

    @staticmethod
    def adjust_boxes_to_roi(detection_results: List[DetectionResults], roi: tuple) -> List[DetectionResults]:
        """
        Adjusts bounding box coordinates in DetectionResults according to the specified ROI.
        Modifies the DetectionResults objects in place.

        :param detection_results: List of DetectionResults objects.
        :param roi: Region of Interest coordinates in the format (x_min, y_min, x_max, y_max).
        :return: List of DetectionResults with updated bounding boxes.
        """
        if not isinstance(roi, tuple) or len(roi) != 4:
            raise ValueError("ROI must be a tuple of four elements (x_min, y_min, x_max, y_max).")

        for result in detection_results:
            if result is None or result.boxes is None:
                continue

            try:
                new_boxes = BoxManipulator._adjust_boxes(result.boxes, roi, result.orig_shape)
                result.boxes = new_boxes  # Update boxes in place
            except Exception as e:
                print(f"Error adjusting boxes: {str(e)}")
                # Optionally handle the error or leave the result as is
                result.boxes = None

        return detection_results

    @staticmethod
    def _adjust_boxes(boxes: DetectionBoxes, roi: tuple, orig_shape) -> DetectionBoxes:
        """
        Adjusts the coordinates of bounding boxes based on the ROI.

        :param boxes: DetectionBoxes object containing the bounding boxes.
        :param roi: Region of Interest coordinates (x_min, y_min, x_max, y_max).
        :param orig_shape: Original shape of the image.
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
                    adjusted_box = [new_x_min, new_y_min, new_x_max, new_y_max] + list(box[4:])
                    adjusted_boxes.append(adjusted_box)

            return DetectionBoxes(np.array(adjusted_boxes), (x_max_roi - x_min_roi, y_max_roi - y_min_roi), 'xyxypc')
        else:
            return None

    @staticmethod
    def adjust_boxes_for_resize(detection_results: List[DetectionResults], new_shape: tuple) -> List[DetectionResults]:
        """
        Adjusts bounding box coordinates in DetectionResults according to the new image size.
        Modifies the DetectionResults objects in place.

        :param detection_results: List of DetectionResults objects.
        :param new_shape: New image size in the format (width, height).
        :return: List of DetectionResults with updated bounding boxes.
        """
        if not isinstance(new_shape, tuple) or len(new_shape) != 2:
            raise ValueError("New shape must be a tuple of two elements (width, height).")

        for result in detection_results:
            if result is None or result.boxes is None:
                continue

            try:
                new_boxes = BoxManipulator._adjust_boxes_for_resize(result.boxes, result.orig_shape, new_shape)
                result.boxes = new_boxes  # Update boxes in place
            except Exception as e:
                print(f"Error adjusting boxes for resize: {str(e)}")
                # Optionally handle the error or leave the result as is
                result.boxes = None

        return detection_results

    @staticmethod
    def _adjust_boxes_for_resize(boxes: DetectionBoxes, orig_shape: tuple, new_shape: tuple) -> DetectionBoxes:
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
                adjusted_box = [new_x_min, new_y_min, new_x_max, new_y_max] + list(box[4:])
                adjusted_boxes.append(adjusted_box)

            return DetectionBoxes(np.array(adjusted_boxes), new_shape, 'xyxypc')
        else:
            return None

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
            boxes_coco = np.array(boxes_coco)

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
            boxes_xyxy = np.array(boxes_xyxy)

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