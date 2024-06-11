import unittest
import os
from datetime import timedelta, datetime
import numpy as np

# Import the VideoDiagnoser class
from detectflow import DetectionBoxes
from detectflow.video.video_diagnoser import VideoDiagnoser

class TestVideoDiagnoserIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up real paths for video and model
        cls.video_path = './resources/GR2_L1_TolUmb2_20220524_07_44.mp4'
        cls.flowers_model_path = '../../detectflow/models/flowers.pt'
        cls.output_path = './test_output'

        # Create VideoDiagnoser instance
        cls.diagnoser = VideoDiagnoser(
            video_path=cls.video_path,
            flowers_model_path=cls.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=cls.output_path
        )

    def test_attributes(self):
        self.assertIsInstance(self.diagnoser.video_path, str)
        self.assertIsInstance(self.diagnoser.flowers_model_path, str)
        self.assertIsInstance(self.diagnoser.flowers_model_conf, float)
        self.assertIsInstance(self.diagnoser.motion_methods, list)
        self.assertIsInstance(self.diagnoser.frame_skip, int)
        self.assertIsInstance(self.diagnoser.color_variance_threshold, int)
        self.assertIsInstance(self.diagnoser.verbose, bool)
        self.assertIsInstance(self.diagnoser.output_path, str)

    def test_video_info_attributes(self):
        self.assertIsInstance(self.diagnoser.file_extension, str)
        self.assertIsInstance(self.diagnoser.filename, str)
        self.assertIsInstance(self.diagnoser.video_origin, str)
        self.assertIsInstance(self.diagnoser.fps, int)
        self.assertIsInstance(self.diagnoser.total_frames, int)
        self.assertIsInstance(self.diagnoser.duration, timedelta)
        self.assertIsInstance(self.diagnoser.frame_height, int)
        self.assertIsInstance(self.diagnoser.frame_width, int)
        self.assertIsInstance(self.diagnoser.recording_identifier, str)
        self.assertIsInstance(self.diagnoser.video_identifier, str)
        self.assertIsInstance(self.diagnoser.start_time, datetime)
        self.assertIsInstance(self.diagnoser.end_time, datetime)

    def test_frames_property(self):
        frames = self.diagnoser.frames
        self.assertIsInstance(frames, list)
        self.assertTrue(all(isinstance(frame, np.ndarray) for frame in frames))
        self.assertTrue(all(len(frame.shape) == 3 for frame in frames))  # Check 3 dimensions (height, width, channels)

    def test_daytime_property(self):
        daytime = self.diagnoser.daytime
        print(daytime)
        self.assertIn(daytime, [True, False])

    # def test_ref_bboxes_property(self):
    #     ref_bboxes = self.diagnoser.ref_bboxes
    #     self.assertIsInstance(ref_bboxes, list)
    #     self.assertTrue(all(isinstance(box, DetectionBoxes) for box in ref_bboxes))

    # def test_motion_data_property(self):
    #     motion_data = self.diagnoser.motion_data
    #     self.assertIsInstance(motion_data, dict)

    # def test_report_data_property(self):
    #     report_data = self.diagnoser.report_data
    #     self.assertIsInstance(report_data, dict)
    #
    # def test_pdf_report(self):
    #     self.diagnoser.pdf_report(output_path=self.output_path)
    #     expected_pdf_path = os.path.join(self.output_path, f"{self.diagnoser.video_identifier}_Diag_Report.pdf")
    #     self.assertTrue(os.path.exists(expected_pdf_path))
    #
    # def test_report_method(self):
    #     report_data = self.diagnoser.report()
    #     self.assertIsInstance(report_data, dict)

    # @classmethod
    # def tearDownClass(cls):
    #     # Clean up generated files if necessary
    #     pdf_report_path = os.path.join(cls.output_path, f"{cls.diagnoser.video_identifier}_Diag_Report.pdf")
    #     if os.path.exists(pdf_report_path):
    #         os.remove(pdf_report_path)

if __name__ == '__main__':
    unittest.main()