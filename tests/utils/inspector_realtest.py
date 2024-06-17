import unittest
import numpy as np
from detectflow.utils.sampler import Sampler
from detectflow.utils.inspector import Inspector
from detectflow.predict.results import DetectionBoxes

class TestInspector(unittest.TestCase):
    def setUp(self):
        # Initialize your Inspector object
        self.inspector = Inspector()

        # Initialize your frames and DetectionBoxes objects
        # Replace with your actual frames and DetectionBoxes objects
        self.frame, self.detection_boxes = Sampler.create_sample_image_with_bboxes(as_detection_boxes=True)

    def test_display_frames_with_boxes(self):
        # Test the display_frames_with_boxes method
        # This is a visual method, so we can't assert an expected result.
        # We just call the method to make sure no exceptions are raised.
        try:
            print(type(self.frame))
            self.inspector.display_frames_with_boxes([self.frame], [self.detection_boxes])
            self.inspector.display_frames_with_boxes(self.frame, [self.detection_boxes.xyxy])
            self.inspector.display_frames_with_boxes([self.frame], self.detection_boxes.xyxy)
        except Exception as e:
            self.fail(f"display_frames_with_boxes raised Exception unexpectedly: {str(e)}")

if __name__ == '__main__':
    unittest.main()