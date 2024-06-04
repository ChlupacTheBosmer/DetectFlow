import unittest
import numpy as np
from detectflow.manipulators.box_manipulator import BoxManipulator, boxes_centers_distance

class TestBoxManipulator(unittest.TestCase):
    def setUp(self):
        self.box_manipulator = BoxManipulator()

    def test_analyze_clusters(self):
        # Define a set of bounding boxes as input
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
            [51, 51, 61, 61]
        ])

        # Define the expected output
        expected_cluster_boxes = np.array([
            [0, 0, 11, 11],
            [50, 50, 61, 61]
        ])

        # Call the method to test
        cluster_boxes, cluster_dict = self.box_manipulator.analyze_clusters(boxes, eps=15, min_samples=1, metric=boxes_centers_distance)

        # Check if the resulting cluster boxes are as expected
        np.testing.assert_array_equal(cluster_boxes, expected_cluster_boxes)

        # Check if the resulting cluster dictionary is as expected
        self.assertEqual(len(cluster_dict), 2)
        for cluster in cluster_dict.values():
            self.assertEqual(len(cluster), 2)

    def test_combine_boxes(self):
        box1 = np.array([1, 1, 3, 3, 0.5, 0.9, 1])
        box2 = np.array([2, 2, 4, 4, 0.6, 0.8, 2])

        combined_box = BoxManipulator.combine_boxes(box1, box2)

        # Check if the combined box has the correct dimensions
        self.assertEqual(combined_box[0], 1)  # x_min
        self.assertEqual(combined_box[1], 1)  # y_min
        self.assertEqual(combined_box[2], 4)  # x_max
        self.assertEqual(combined_box[3], 4)  # y_max

        # Check if the additional data from the first box is preserved
        self.assertEqual(combined_box[4], 0.5)  # cls
        self.assertEqual(combined_box[5], 0.9)  # prob
        self.assertEqual(combined_box[6], 1)    # id

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

if __name__ == '__main__':
    unittest.main()