import unittest
import numpy as np
from detectflow import BoxAnalyser

class TestQuantifyFocusCoverage(unittest.TestCase):
    def test_quantify_focus_coverage(self):
        interest_bboxes = np.array([
            [50, 50, 150, 150],
            [200, 200, 300, 300]
        ])

        focus_bboxes = np.array([
            [100, 100, 250, 250],
            [250, 250, 350, 350],
            [100, 0, 200, 200]
        ])

        expected_coverage = 0.5
        actual_coverage = BoxAnalyser.calculate_coverage(interest_bboxes, focus_bboxes)
        print(actual_coverage)

        self.assertAlmostEqual(actual_coverage, expected_coverage, places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)