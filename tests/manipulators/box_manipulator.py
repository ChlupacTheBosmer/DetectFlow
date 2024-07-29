import unittest
import numpy as np
from detectflow.manipulators.box_manipulator import boxes_center_center_distance
from detectflow.manipulators.box_manipulator import BoxManipulator
from detectflow.predict.results import DetectionBoxes


def none(*args):
    return None


class TestBoxManipulator(unittest.TestCase):
    def setUp(self):
        self.box_manipulator = BoxManipulator()
        self.boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        self.detection_boxes = DetectionBoxes(self.boxes, (100, 100))

    def test_boxes_edge_edge_distance(self):

        from detectflow.manipulators.box_manipulator import boxes_edge_edge_distance

        # Define two boxes
        box_coords_1 = [0, 0, 10, 10]
        box_coords_2 = [5, 5, 15, 15]

        for func in (np.array, list, tuple, none):

            box1 = func(box_coords_1)
            box2 = func(box_coords_2)

            config = [
                {"min_or_max": "max", "result": 21.21},
                {"min_or_max": "min", "result": 0.0}
            ]

            for c in config:
                if func is none:
                    with self.assertRaises(ValueError) as cm:
                        distance = boxes_edge_edge_distance(box1, box2, c['min_or_max'])
                        self.assertAlmostEqual(distance, c['result'], places=2)
                    self.assertEqual(str(cm.exception), "Invalid argument: None")
                else:
                    distance = boxes_edge_edge_distance(box1, box2, c['min_or_max'])
                    self.assertAlmostEqual(distance, c['result'], places=2)

    def test_boxes_center_edge_distance(self):

        from detectflow.manipulators.box_manipulator import boxes_center_edge_distance

        # Define two boxes
        box_coords_1 = [0, 0, 10, 10]
        box_coords_2 = [5, 5, 15, 15]

        for func in (np.array, list, tuple, none):

            box1 = func(box_coords_1)
            box2 = func(box_coords_2)

            config = [
                {"min_or_max": "max", "result": 14.14},
                {"min_or_max": "min", "result": 0.0}
            ]

            for c in config:
                if func is none:
                    with self.assertRaises(ValueError) as cm:
                        distance = boxes_center_edge_distance(box1, box2, c['min_or_max'])
                        self.assertAlmostEqual(distance, c['result'], places=2)
                    self.assertEqual(str(cm.exception), "Invalid argument: None")
                else:
                    distance = boxes_center_edge_distance(box1, box2, c['min_or_max'])
                    self.assertAlmostEqual(distance, c['result'], places=2)

    def test_boxes_center_center_distance(self):
        from detectflow.manipulators.box_manipulator import boxes_center_center_distance

        # Define two boxes
        box_coords_1 = [0, 0, 10, 10]
        box_coords_2 = [5, 5, 15, 15]

        for func in (np.array, list, tuple, none):

            box1 = func(box_coords_1)
            box2 = func(box_coords_2)

            config = [
                {"result": 7.07}
            ]

            for c in config:
                if func is none:
                    with self.assertRaises(ValueError) as cm:
                        distance = boxes_center_center_distance(box1, box2)
                        self.assertAlmostEqual(distance, c['result'], places=2)
                    self.assertEqual(str(cm.exception), "Invalid argument: None")
                else:
                    distance = boxes_center_center_distance(box1, box2)
                    self.assertAlmostEqual(distance, c['result'], places=2)

    def test_get_box_area(self):
        box = self.boxes[0]
        expected_area = 400

        for func in (np.array, list, tuple, none):

            box1 = func(box)

            if func is none:
                with self.assertRaises(ValueError) as cm:
                    actual_area = self.box_manipulator.get_box_area(box1)
                    self.assertEqual(expected_area, actual_area)
                self.assertEqual(str(cm.exception), "Invalid argument: None")
            else:
                actual_area = self.box_manipulator.get_box_area(box1)
                self.assertEqual(expected_area, actual_area)

    def test_get_box_dimensions(self):
        box = self.boxes[0]
        expected_dimensions = (20, 20)

        for func in (np.array, list, tuple, none):

            box1 = func(box)

            if func is none:
                with self.assertRaises(ValueError) as cm:
                    actual_dimensions = self.box_manipulator.get_box_dimensions(box1)
                    self.assertEqual(expected_dimensions, actual_dimensions)
                self.assertEqual(str(cm.exception), "Invalid argument: None")
            else:
                actual_dimensions = self.box_manipulator.get_box_dimensions(box1)
                self.assertEqual(expected_dimensions, actual_dimensions)

    def test_get_box_center(self):
        box = self.boxes[0]
        expected_center = (20, 30)

        for func in (np.array, list, tuple, none):

            box1 = func(box)

            if func is none:
                with self.assertRaises(ValueError) as cm:
                    actual_center = self.box_manipulator.get_box_center(box1)
                    self.assertEqual(expected_center, actual_center)
                self.assertEqual(str(cm.exception), "Invalid argument: None")
            else:
                actual_center = self.box_manipulator.get_box_center(box1)
                self.assertEqual(expected_center, actual_center)

    def test_get_distance_between_boxes(self):
        # Define two boxes
        box_coords_1 = [0, 0, 10, 10]
        box_coords_2 = [5, 5, 15, 15]

        for func in (np.array, list, tuple, none):

            box1 = func(box_coords_1)
            box2 = func(box_coords_2)

            config = [
                {"metric": "cc", "min_or_max": "min", "result": 7.07}, # min_or_max irrelevant
                {"metric": "ce", "min_or_max": "min", "result": 0.0},
                {"metric": "ce", "min_or_max": "max", "result": 14.14},
                {"metric": "ee", "min_or_max": "min", "result": 0.0},
                {"metric": "ee", "min_or_max": "max", "result": 21.21}
            ]

            for c in config:
                if func is none:
                    with self.assertRaises(ValueError) as cm:
                        distance = self.box_manipulator.get_distance_between_boxes(box1, box2,
                                                                                   min_or_max=c['min_or_max'],
                                                                                   metric=c["metric"])
                        self.assertAlmostEqual(distance, c['result'], places=2)
                    self.assertEqual(str(cm.exception), "Invalid argument: None")
                else:
                    distance = self.box_manipulator.get_distance_between_boxes(box1, box2, min_or_max=c['min_or_max'],
                                                                               metric=c["metric"])
                    self.assertAlmostEqual(distance, c['result'], places=2)

    def test_get_sorted_boxes(self):
        # Define a set of bounding boxes
        boxes = [
            [0, 0, 5, 10],  # area = 50, width = 5, height = 10, largest_dim = 10
            [0, 0, 20, 5],  # area = 100, width = 20, height = 5, largest_dim = 20
            [0, 0, 10, 30]  # area = 300, width = 10, height = 30, largest_dim = 30
        ]

        # Define the expected sorted boxes for each criterion and order
        expected_boxes = {
            'area': {
                True: np.array([
                    [0, 0, 5, 10],
                    [0, 0, 20, 5],
                    [0, 0, 10, 30]
                ]),
                False: np.array([
                    [0, 0, 10, 30],
                    [0, 0, 20, 5],
                    [0, 0, 5, 10]
                ])
            },
            'width': {
                True: np.array([
                    [0, 0, 5, 10],
                    [0, 0, 10, 30],
                    [0, 0, 20, 5]
                ]),
                False: np.array([
                    [0, 0, 20, 5],
                    [0, 0, 10, 30],
                    [0, 0, 5, 10]
                ])
            },
            'height': {
                True: np.array([
                    [0, 0, 20, 5],
                    [0, 0, 5, 10],
                    [0, 0, 10, 30]
                ]),
                False: np.array([
                    [0, 0, 10, 30],
                    [0, 0, 5, 10],
                    [0, 0, 20, 5]
                ])
            },
            'largest_dim': {
                True: np.array([
                    [0, 0, 5, 10],
                    [0, 0, 20, 5],
                    [0, 0, 10, 30]
                ]),
                False: np.array([
                    [0, 0, 10, 30],
                    [0, 0, 20, 5],
                    [0, 0, 5, 10]
                ])
            }
        }

        for func, kwargs in ((np.array, {}), (DetectionBoxes, {'orig_shape': (50, 50)}), (none, {})):

            boxes = func(boxes, **kwargs)

            if func is none:
                with self.assertRaises(ValueError) as cm:
                    _ = BoxManipulator.get_sorted_boxes(boxes, 'area', True)
                self.assertEqual(str(cm.exception), "Invalid argument: None")
                continue

            # Test the get_sorted_boxes method for each criterion and order
            for sort_by in ['area', 'width', 'height', 'largest_dim']:
                for ascending in [True, False]:
                    print(f"Sort by: {sort_by}, Ascending: {ascending}")
                    sorted_boxes = BoxManipulator.get_sorted_boxes(boxes, sort_by, ascending)
                    sorted_boxes = sorted_boxes.data if isinstance(sorted_boxes, DetectionBoxes) else sorted_boxes
                    np.testing.assert_array_equal(sorted_boxes, expected_boxes[sort_by][ascending])

    def test_get_roi(self):

        # Define a set of bounding boxes
        boxes = np.array([
            [100, 100, 200, 200],
            [150, 150, 250, 250]
        ])  # ROI: (90, 90, 260, 260) with margin 10 else (100, 100, 250, 250)

        boxes_expand = np.array([
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            [900, 900, 1100, 1100]
        ])  # ROI: (90, 90, 1110, 1110) with margin 10 else (100, 100, 1100, 1100)

        img_dims = (1920, 1200)
        crop_size = (640, 640)
        max_expansion_limit = (1020, 1020)
        margin = 10

        # Define expected ROI outputs
        expected_roi_expand = (0, 0, 640, 640)
        expected_roi_strict = (0, 0, 640, 640)
        expected_roi_expand_large = (90, 90, 1110, 1110)
        large_margin = 1000

        config = [
            {'boxes': boxes, "overflow": 'expand', 'margin': margin, 'result': expected_roi_expand},
            {'boxes': boxes, "overflow": 'strict', 'margin': margin, 'result': expected_roi_strict},
            {'boxes': boxes, "overflow": 'expand', 'margin': large_margin, 'result': None},
            {'boxes': boxes_expand, "overflow": 'expand', 'margin': margin, 'result': expected_roi_expand_large},
            {'boxes': boxes_expand, "overflow": 'strict', 'margin': margin, 'result': None},
            {'boxes': boxes_expand, "overflow": 'expand', 'margin': large_margin, 'result': None}
        ]

        for func, kwargs in ((np.array, {}), (DetectionBoxes, {'orig_shape': img_dims}), (none, {})):
            for c in config:
                boxes = func(c['boxes'], **kwargs)
                print(f"Testing for: \nFunction: {str(func)}, \nOverflow policy: {c['overflow']}, \nMargin: {c['margin']}, \nExpected result: {c['result']}")

                roi = BoxManipulator.get_roi(boxes, img_dims, crop_size, c['overflow'],
                                             max_expansion_limit, c['margin'])
                self.assertEqual(roi, c['result']) if c['result'] is not None and func is not none else self.assertIsNone(roi)

    def test_get_optimal_roi(self):

        boxes = np.array([
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            [900, 900, 1100, 1100]
        ])  # ROI: (90, 90, 1110, 1110) with margin 10 else (100, 100, 1100, 1100)

        img_dims = (1920, 1200)
        crop_size = (640, 640)
        max_expansion_limit = (1020, 1020)
        margin = 10

        expected_result_expand = (90.0, 90.0, 1110.0, 1110.0)
        expected_result_strict = (0.0, 0.0, 640.0, 640.0)
        expected_result_none_dummy = (640.0, 280.0, 1280.0, 920.0)

        config = [
            {'boxes': boxes, "overflow": 'expand', 'result': expected_result_expand},
            {'boxes': boxes, "overflow": 'strict', 'result': expected_result_strict}
        ]

        for func, kwargs in ((np.array, {}), (DetectionBoxes, {'orig_shape': img_dims}), (none, {})):
            for c in config:
                boxes = func(c['boxes'], **kwargs)
                print(
                    f"Testing for: \nFunction: {str(func)}, \nOverflow policy: {c['overflow']}, \nExpected result: {c['result']}")
                if func is np.array:
                    with self.assertRaises(TypeError) as cm:
                        _ = BoxManipulator.get_optimal_roi(detection_boxes=boxes, img_dims=img_dims)
                    continue

                rois = BoxManipulator.get_optimal_roi(detection_boxes=boxes, img_dims=img_dims, crop_size=crop_size,
                                                      handle_overflow=c['overflow'],
                                                      max_expansion_limit=max_expansion_limit, margin=margin,
                                                      exhaustive_search=True, permutation_limit=9, multiple_rois=False,
                                                      ignore_empty=False, partial_overlap=False, overlap_threshold=0.5)

                self.assertIsInstance(rois, list)
                self.assertIsInstance(rois[0], tuple)
                self.assertIsInstance(rois[0][0], tuple)
                self.assertEqual(rois[0][0], c['result'] if func is not none else expected_result_none_dummy)
                self.assertIsInstance(rois[0][1], DetectionBoxes) if func is not none else None
                print(rois[0][1].data if func is not none else None)

    def test_get_combined_box(self):
        boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]

        expected_combined_box = np.array([10, 20, 70, 80])

        for func, kwargs in ((np.array, {}), (list, {}), (DetectionBoxes, {'orig_shape': (100,100)}), (none, {})):

            boxes = func(boxes, **kwargs)
            print(
                f"Testing for: Function: {str(func)}")
            if func is none:
                with self.assertRaises(ValueError) as cm:
                    actual_combined_box = self.box_manipulator.get_combined_box(boxes)
                    np.testing.assert_array_equal(expected_combined_box, actual_combined_box)
                self.assertEqual(str(cm.exception), "Invalid argument: None")
            else:
                actual_combined_box = self.box_manipulator.get_combined_box(boxes)
                np.testing.assert_array_equal(expected_combined_box, actual_combined_box)

    def test_get_clusters(self):

        from detectflow.manipulators.box_manipulator import boxes_edge_edge_distance

        # Define a set of bounding boxes
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [400, 400, 500, 500],
            [405, 405, 505, 505],
            [800, 800, 900, 900]
        ])

        detection_boxes = DetectionBoxes(boxes, (1000, 1000))

        # Expected clusters
        expected_cluster_boxes = np.array([
            [100, 100, 205, 205],
            [400, 400, 505, 505],
            [800, 800, 900, 900]
        ])

        expected_cluster_dict = {
            0: np.array([[100, 100, 200, 200], [105, 105, 205, 205]]),
            1: np.array([[400, 400, 500, 500], [405, 405, 505, 505]]),
            2: np.array([[800, 800, 900, 900]])
        }

        config = [
            {'boxes': detection_boxes, 'eps': 150, 'min_samples': 1, 'metric': boxes_edge_edge_distance,
             'expected_clusters': expected_cluster_boxes, 'expected_dict': expected_cluster_dict}
        ]

        for func, kwargs in ((np.array, {}), (DetectionBoxes, {'orig_shape': (1000, 1000)}), (none, {})):
            for c in config:
                print(f"Testing for: \nFunction: {str(func)}, \nEps: {c['eps']}, \nMin samples: {c['min_samples']}, \nMetric: {c['metric']}")
                boxes = func(boxes, **kwargs)
                if func is none:
                    with self.assertRaises(TypeError) as cm:
                        clusters, cluster_dict = BoxManipulator.get_clusters(boxes, eps=c['eps'],
                                                                             min_samples=c['min_samples'],
                                                                             metric=c['metric'])
                    self.assertEqual(str(cm.exception),
                                     f"Expected a DetectionBoxes instance, got {type(func(c['boxes']))} instead")
                else:
                    clusters, cluster_dict = BoxManipulator.get_clusters(boxes, eps=c['eps'],
                                                                         min_samples=c['min_samples'],
                                                                         metric=c['metric'])

                    if func is DetectionBoxes:
                        self.assertTrue(isinstance(clusters, DetectionBoxes))
                        clusters_data = clusters.data
                    else:
                        clusters_data = clusters

                    np.testing.assert_array_equal(clusters_data, c['expected_clusters'])

                    for key, value in cluster_dict.items():
                        expected_value = c['expected_dict'][key]
                        actual_value = value.data if isinstance(value, DetectionBoxes) else value
                        np.testing.assert_array_equal(actual_value, expected_value)

    def test_get_coverage(self):

        # Define a set of bounding boxes
        interest_bboxes = np.array([
            [100, 100, 200, 200],
            [150, 150, 250, 250]
        ])
        reference_bboxes = np.array([
            [120, 120, 180, 180],
            [160, 160, 220, 220]
        ])

        # Define expected coverage output
        expected_coverage = 0.3885  # 6800 (intersection) / 17500 (total interest area)

        # Define additional test cases
        interest_bboxes_full_coverage = np.array([
            [100, 100, 200, 200]
        ])
        reference_bboxes_full_coverage = np.array([
            [100, 100, 200, 200]
        ])
        expected_full_coverage = 1.0

        interest_bboxes_no_coverage = np.array([
            [100, 100, 200, 200]
        ])
        reference_bboxes_no_coverage = np.array([
            [300, 300, 400, 400]
        ])
        expected_no_coverage = 0.0

        config = [
            {'interest_bboxes': interest_bboxes, 'reference_bboxes': reference_bboxes,
             'expected_coverage': expected_coverage},
            {'interest_bboxes': interest_bboxes_full_coverage, 'reference_bboxes': reference_bboxes_full_coverage,
             'expected_coverage': expected_full_coverage},
            {'interest_bboxes': interest_bboxes_no_coverage, 'reference_bboxes': reference_bboxes_no_coverage,
             'expected_coverage': expected_no_coverage},
        ]

        for c in config:
            coverage = BoxManipulator.get_coverage(c['interest_bboxes'], c['reference_bboxes'])
            self.assertAlmostEqual(coverage, c['expected_coverage'], places=2)

        # Test with DetectionBoxes instance
        detection_boxes_interest = DetectionBoxes(interest_bboxes, (1000, 1000))
        detection_boxes_reference = DetectionBoxes(reference_bboxes, (1000, 1000))

        coverage = BoxManipulator.get_coverage(detection_boxes_interest, detection_boxes_reference)
        self.assertAlmostEqual(coverage, expected_coverage, places=2)

        # Test invalid input types
        with self.assertRaises(ValueError) as cm:
            _ = BoxManipulator.get_coverage("invalid", reference_bboxes)
        self.assertEqual(str(cm.exception), "Invalid input type for interest_bboxes")

        with self.assertRaises(ValueError) as cm:
            _ = BoxManipulator.get_coverage(interest_bboxes, "invalid")
        self.assertEqual(str(cm.exception), "Invalid input type for reference_bboxes")

    def test_get_iou(self):

        # Define bounding boxes for testing
        box1 = [50, 50, 150, 150]
        box2 = [100, 100, 200, 200]
        box3 = [150, 150, 250, 250]
        box4 = [300, 300, 400, 400]

        # Calculate expected IoU values
        expected_iou_1_2 = 0.1429  # Overlapping area: 2500, Union area: 17500
        expected_iou_1_3 = 0.0  # No overlap
        expected_iou_1_4 = 0.0  # No overlap
        expected_iou_2_3 = 0.1429  # Overlapping area: 2500, Union area: 17500
        expected_iou_3_4 = 0.0  # No overlap

        config = [
            {'box1': box1, 'box2': box2, 'expected_iou': expected_iou_1_2},
            {'box1': box1, 'box2': box3, 'expected_iou': expected_iou_1_3},
            {'box1': box1, 'box2': box4, 'expected_iou': expected_iou_1_4},
            {'box1': box2, 'box2': box3, 'expected_iou': expected_iou_2_3},
            {'box1': box3, 'box2': box4, 'expected_iou': expected_iou_3_4},
        ]

        for c in config:
            iou = BoxManipulator.get_iou(c['box1'], c['box2'])
            self.assertAlmostEqual(iou, c['expected_iou'], places=4)

        # Test identical boxes
        identical_box = [100, 100, 200, 200]
        expected_iou_identical = 1.0

        iou_identical = BoxManipulator.get_iou(identical_box, identical_box)
        self.assertEqual(iou_identical, expected_iou_identical)

        # Test edge cases with no area
        zero_area_box = [100, 100, 100, 100]
        expected_iou_zero_area = 0.0

        iou_zero_area = BoxManipulator.get_iou(box1, zero_area_box)
        self.assertEqual(iou_zero_area, expected_iou_zero_area)

        iou_zero_area_reverse = BoxManipulator.get_iou(zero_area_box, box1)
        self.assertEqual(iou_zero_area_reverse, expected_iou_zero_area)

    def test_get_boxes_distribution_index(self):

        # Define a set of bounding boxes
        bboxes = np.array([
            [50, 50, 150, 150],
            [300, 300, 450, 450],
            [600, 50, 750, 200],
            [900, 900, 1100, 1100]
        ])
        img_size = (1200, 1200)
        grid_size = (3, 3)

        # Calculate expected distribution index
        expected_distribution_index = 5 / 9  # 5 out of 9 grid cells are occupied

        # Test the method with the given bounding boxes
        distribution_index = BoxManipulator.get_boxes_distribution_index(bboxes, img_size, grid_size)
        self.assertAlmostEqual(distribution_index, expected_distribution_index, places=4)

        # Test with bounding boxes filling the entire grid
        bboxes_full = np.array([
            [0, 0, 1200, 1200]
        ])
        expected_distribution_index_full = 1.0  # Entire grid is occupied

        distribution_index_full = BoxManipulator.get_boxes_distribution_index(bboxes_full, img_size, grid_size)
        self.assertEqual(distribution_index_full, expected_distribution_index_full)

        # Test with no bounding boxes
        bboxes_empty = np.array([])
        expected_distribution_index_empty = 0.0  # No grid cells are occupied

        distribution_index_empty = BoxManipulator.get_boxes_distribution_index(bboxes_empty, img_size, grid_size)
        self.assertEqual(distribution_index_empty, expected_distribution_index_empty)

    def test_is_contained(self):
        from detectflow.manipulators.box_manipulator import BoxManipulator

        # Define bounding boxes for testing
        box1 = [50, 50, 150, 150]
        box2 = [0, 0, 200, 200]
        box3 = [100, 100, 200, 200]
        box4 = [75, 75, 125, 125]
        box5 = [50, 50, 150, 150]  # identical to box1
        box6 = [0, 0, 100, 100]

        config = [
            {'box1': box1, 'box2': box2, 'expected_result': True},  # box1 is contained within box2
            {'box1': box1, 'box2': box3, 'expected_result': False},  # box1 is not contained within box3
            {'box1': box4, 'box2': box1, 'expected_result': True},  # box4 is contained within box1
            {'box1': box1, 'box2': box5, 'expected_result': True},  # box1 is identical to box5
            {'box1': box1, 'box2': box6, 'expected_result': False}  # box1 is not contained within box6
        ]

        for c in config:
            result = BoxManipulator.is_contained(c['box1'], c['box2'])
            self.assertEqual(result, c['expected_result'])

        # Test with numpy arrays
        box1_np = np.array(box1)
        box2_np = np.array(box2)

        result_np = BoxManipulator.is_contained(box1_np, box2_np)
        self.assertTrue(result_np)

        # Test with lists
        box1_list = list(box1)
        box2_list = list(box2)

        result_list = BoxManipulator.is_contained(box1_list, box2_list)
        self.assertTrue(result_list)

        # Test with tuples
        box1_tuple = tuple(box1)
        box2_tuple = tuple(box2)

        result_tuple = BoxManipulator.is_contained(box1_tuple, box2_tuple)
        self.assertTrue(result_tuple)

    def test_adjust_box_to_fit(self):
        from detectflow.manipulators.box_manipulator import BoxManipulator

        # Define a set of bounding boxes
        boxes = np.array([
            [50, 50, 150, 150],
            [300, 300, 450, 450],
            [600, 50, 750, 200],
            [900, 900, 1100, 1100]
        ])
        shape = (800, 800)

        # Expected clipped boxes
        expected_clipped_boxes = np.array([
            [50, 50, 150, 150],
            [300, 300, 450, 450],
            [600, 50, 750, 200],
            [800, 800, 800, 800]  # Box adjusted to fit within the shape
        ])

        # Test the method with the given bounding boxes
        clipped_boxes = BoxManipulator.adjust_box_to_fit(boxes, shape)
        np.testing.assert_array_equal(clipped_boxes, expected_clipped_boxes)

        # Test with boxes completely outside the shape
        boxes_outside = np.array([
            [-100, -100, -50, -50],
            [900, 900, 1000, 1000]
        ])
        expected_clipped_boxes_outside = np.array([
            [0, 0, 0, 0],
            [800, 800, 800, 800]
        ])

        clipped_boxes_outside = BoxManipulator.adjust_box_to_fit(boxes_outside, shape)
        np.testing.assert_array_equal(clipped_boxes_outside, expected_clipped_boxes_outside)

        # Test with some boxes partially outside the shape
        boxes_partial = np.array([
            [100, 100, 900, 900],
            [-50, -50, 100, 100]
        ])
        expected_clipped_boxes_partial = np.array([
            [100, 100, 800, 800],
            [0, 0, 100, 100]
        ])

        clipped_boxes_partial = BoxManipulator.adjust_box_to_fit(boxes_partial, shape)
        np.testing.assert_array_equal(clipped_boxes_partial, expected_clipped_boxes_partial)

        # Test with numpy array of different shape
        boxes_diff_shape = np.array([
            [50, 50, 150, 150],
            [300, 300, 450, 450]
        ])
        shape_diff = (400, 400)
        expected_clipped_boxes_diff_shape = np.array([
            [50, 50, 150, 150],
            [300, 300, 400, 400]  # Box adjusted to fit within the new shape
        ])

        clipped_boxes_diff_shape = BoxManipulator.adjust_box_to_fit(boxes_diff_shape, shape_diff)
        np.testing.assert_array_equal(clipped_boxes_diff_shape, expected_clipped_boxes_diff_shape)

    def test_remove_contained_boxes(self):
        # Define a set of bounding boxes where box1 is contained within box2
        box1 = np.array([1, 1, 2, 2])
        box2 = np.array([0, 0, 3, 3])
        boxes = np.array([box1, box2])

        # Call the remove_contained_boxes method
        result = self.box_manipulator.remove_contained_boxes(boxes)

        # Check if the resulting boxes only contain box2
        expected_result = np.array([box2])
        np.testing.assert_array_equal(result, expected_result)

    def test_remove_duplicate_boxes(self):
        from detectflow.manipulators.box_manipulator import BoxManipulator

        # Define a set of bounding boxes, including duplicates
        boxes = [
            [50, 50, 150, 150, 1],
            [300, 300, 450, 450, 2],
            [50, 50, 150, 150, 3],  # Duplicate of the first box (different ID)
            [600, 50, 750, 200, 4],
            [50, 50, 150, 150, 1],  # Exact duplicate of the first box (same ID)
            [900, 900, 1100, 1100, 5]
        ]

        # Expected unique boxes (duplicates removed)
        expected_unique_boxes = [
            [50, 50, 150, 150, 1],
            [300, 300, 450, 450, 2],
            [600, 50, 750, 200, 4],
            [900, 900, 1100, 1100, 5]
        ]

        # Test the method with the given bounding boxes
        unique_boxes = BoxManipulator.remove_duplicate_boxes(boxes)
        self.assertEqual(unique_boxes, expected_unique_boxes)

        # Test with no duplicates
        boxes_no_duplicates = [
            [10, 10, 20, 20, 1],
            [30, 30, 40, 40, 2],
            [50, 50, 60, 60, 3]
        ]
        expected_unique_no_duplicates = boxes_no_duplicates.copy()

        unique_boxes_no_duplicates = BoxManipulator.remove_duplicate_boxes(boxes_no_duplicates)
        self.assertEqual(unique_boxes_no_duplicates, expected_unique_no_duplicates)

        # Test with all duplicates
        boxes_all_duplicates = [
            [10, 10, 20, 20, 1],
            [10, 10, 20, 20, 2],
            [10, 10, 20, 20, 3]
        ]
        expected_unique_all_duplicates = [[10, 10, 20, 20, 1]]

        unique_boxes_all_duplicates = BoxManipulator.remove_duplicate_boxes(boxes_all_duplicates)
        self.assertEqual(unique_boxes_all_duplicates, expected_unique_all_duplicates)

if __name__ == '__main__':
    unittest.main()

