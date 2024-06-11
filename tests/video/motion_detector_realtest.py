import unittest
import numpy as np
import os
from detectflow.video.motion_detector import MotionDetector
from detectflow.predict.results import DetectionBoxes

class TestMotionDetectorIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up real paths for video and model
        cls.video_path = './resources/GR2_L1_TolUmb2_20220524_07_44.mp4'
        cls.rois = [  # Example ROIs
            [100, 200, 400, 500],  # Format: [x1, y1, x2, y2]
            [200, 300, 500, 600],
            [300, 400, 600, 700]
        ]
        cls.methods = ['SOM', 'OF', 'BS', 'FM', 'TA']

    def test_motion_detection(self):
        # Create MotionDetector instance
        detector = MotionDetector(
            video_path=self.video_path,
            methods=self.methods,
            frame_reader_method="imageio",
            fps=None,  # Will be calculated from video
            smooth=True,
            smooth_time=3,
            frame_skip=1,
            high_movement=True,
            high_movement_thresh=None,
            high_movement_time=2,
            rois=self.rois,
            rois_select="random",
            visualize=True  # Set to True to visualize the results
        )

        # Perform motion detection
        motion_data = detector.analyze()

        # Check the structure of the motion_data
        self.assertIsInstance(motion_data, dict)
        for method in self.methods:
            self.assertIn(method, motion_data)
            self.assertIsInstance(motion_data[method], dict)
            self.assertIn('raw_data', motion_data[method])
            self.assertIn('mean', motion_data[method])
            self.assertIn('smooth_data', motion_data[method])
            self.assertIn('high_movement_periods_f', motion_data[method])
            self.assertIn('high_movement_periods_t', motion_data[method])
            self.assertIn('plot', motion_data[method])

            # Print results for manual verification
            print(f"\nMethod: {method}")
            print(f"Raw Data: {motion_data[method]['raw_data'][:10]}")  # Print first 10 data points for brevity
            print(f"Mean Movement: {motion_data[method]['mean']}")
            print(f"Smoothed Data: {motion_data[method]['smooth_data'][:10]}")  # Print first 10 data points
            print(f"High Movement Periods (Frames): {motion_data[method]['high_movement_periods_f']}")
            print(f"High Movement Periods (Time): {motion_data[method]['high_movement_periods_t']}")
            print(f"Plot: {motion_data[method]['plot']}")

if __name__ == '__main__':
    unittest.main()