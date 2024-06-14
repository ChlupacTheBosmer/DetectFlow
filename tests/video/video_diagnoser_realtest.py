import unittest
import os
from datetime import timedelta, datetime
import numpy as np

# Import the VideoDiagnoser class
from detectflow import DetectionBoxes, Inspector
from detectflow.video.video_diagnoser import VideoDiagnoser


class TestVideoDiagnoserIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up real paths for video and model
        video_path = './resources/GR2_L1_TolUmb2_20220524_07_44.mp4'
        flowers_model_path = '../../detectflow/models/flowers.pt'
        output_path = './test_output'

        # Create VideoDiagnoser instance
        cls.diagnoser = VideoDiagnoser(
            video_path=video_path,
            flowers_model_path=flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="TA",
            frame_skip=500,
            color_variance_threshold=10,
            verbose=True,
            output_path=output_path
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
        self.assertTrue(len(frames) > 0)
        self.assertTrue(all(len(frame.shape) == 3 for frame in frames))  # Check 3 dimensions (height, width, channels)

        # Inspect
        print("Number of frames: ", len(frames))
        print("Type of frame: ", type(frames[0]) if len(frames) > 0 else None)
        print("Frame dimensions: ", frames[0].shape if len(frames) > 0 else None)
        Inspector.display_images(frames[0] if len(frames) > 0 else None)

    def test_daytime_property(self):
        daytime = self.diagnoser.daytime
        self.assertIn(daytime, [True, False])
        print(daytime)

    def test_ref_bboxes_property(self):
        ref_bboxes = self.diagnoser.ref_bboxes
        self.assertIsInstance(ref_bboxes, DetectionBoxes)
        print(ref_bboxes.xyxy)

    def test_motion_data_property(self):
        motion_data = self.diagnoser.motion_data
        self.assertIsInstance(motion_data, dict)

        # Check the main keys
        method_keys = ['SOM', 'OF', 'BS', 'FM', 'TA']
        self.assertTrue(any(k in motion_data for k in method_keys))
        for key in motion_data.keys():

            # Check the sub-keys
            sub_dict = motion_data[key]
            self.assertIn("raw_data", sub_dict)
            self.assertIn("mean", sub_dict)
            self.assertIn("smooth_data", sub_dict)
            self.assertIn("high_movement_periods_f", sub_dict)
            self.assertIn("high_movement_periods_t", sub_dict)
            self.assertIn("plot", sub_dict)

            print(sub_dict["mean"])

    def test_focus_regions_property(self):
        focus_regions = self.diagnoser.focus_regions
        self.assertIsInstance(focus_regions, DetectionBoxes)
        print(focus_regions.xyxy)
        Inspector.display_frames_with_boxes([self.diagnoser.frames[0]], [focus_regions])

    def test_focus_accuracy_property(self):
        focus_accuracy = self.diagnoser.focus_accuracy
        self.assertIsInstance(focus_accuracy, float)
        print("Focus accuracy: ", focus_accuracy)

    def test_thumbnail_property(self):
        thumbnail = self.diagnoser.thumbnail
        self.assertIsInstance(thumbnail, np.ndarray)
        self.assertEqual(len(thumbnail.shape), 3)
        Inspector.display_images(thumbnail)

    def test_report_data_property(self):
        report_data = self.diagnoser.report_data
        self.assertIsInstance(report_data, dict)

        self.assertIn("basic_data", report_data)
        self.assertIn("duration", report_data["basic_data"])
        self.assertIn("start_time", report_data["basic_data"])
        self.assertIn("end_time", report_data["basic_data"])
        self.assertIn("video_id", report_data["basic_data"])
        self.assertIn("recording_id", report_data["basic_data"])
        self.assertIn("total_frames", report_data["basic_data"])
        self.assertIn("frame_rate", report_data["basic_data"])
        self.assertIn("format", report_data["basic_data"])
        self.assertIn("video_origin", report_data["basic_data"])
        self.assertIn("validated_methods", report_data["basic_data"])
        self.assertIn("frame_width", report_data["basic_data"])
        self.assertIn("frame_height", report_data["basic_data"])

        self.assertIn("roi_bboxes", report_data)
        self.assertIn("roi_data", report_data)

        self.assertIn("motion_data", report_data)
        self.assertIn("daytime", report_data)
        self.assertIn("frames", report_data)


if __name__ == '__main__':
    unittest.main()