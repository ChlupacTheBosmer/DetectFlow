import unittest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
import os

# Import the VideoDiagnoser class
from detectflow import VideoDiagnoser

class TestVideoDiagnoser(unittest.TestCase):

    def setUp(self):
        self.video_path = './resources/GR2_L1_TolUmb2_20220524_07_44.mp4'
        self.flowers_model_path = '../../detectflow/models/flowers.pt'
        self.output_path = './test_output'

    def test_initialization(self):
        diagnoser = VideoDiagnoser(
            video_path=self.video_path,
            flowers_model_path=self.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=self.output_path
        )
        self.assertEqual(diagnoser.video_path, self.video_path)
        self.assertEqual(diagnoser.flowers_model_path, self.flowers_model_path)
        self.assertEqual(diagnoser.flowers_model_conf, 0.3)
        self.assertEqual(diagnoser.motion_methods, ["SOM"])
        self.assertEqual(diagnoser.frame_skip, 1)
        self.assertEqual(diagnoser.color_variance_threshold, 10)
        self.assertEqual(diagnoser.verbose, True)
        self.assertEqual(diagnoser.output_path, self.output_path)

    @patch('detectflow.Video')
    @patch('detectflow.Sampler.sample_frames')
    def test_get_frames(self, mock_sample_frames, MockVideo):
        # Create a single mock object to use in both the sample_frames return and the expected result
        mock_frame = Mock()
        mock_sample_frames.return_value = [mock_frame]

        MockVideo.return_value.fps = 30
        diagnoser = VideoDiagnoser(
            video_path=self.video_path,
            flowers_model_path=self.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=self.output_path
        )
        frames = diagnoser._get_frames()
        mock_sample_frames.assert_called_once_with(
            self.video_path,
            num_frames=12,
            output_format='list',
            distribution='even',
            reader='decord'
        )
        self.assertEqual(frames, [mock_frame])

    @patch('detectflow.PictureQualityAnalyzer.get_daytime', return_value="day")
    def test_get_daytime(self, mock_get_daytime):
        diagnoser = VideoDiagnoser(
            video_path=self.video_path,
            flowers_model_path=self.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=self.output_path
        )
        daytime = diagnoser.get_daytime()
        self.assertEqual(daytime, "day")

    @patch('detectflow.Predictor.detect')
    @patch('detectflow.ObjectDetectValidator.is_valid_ndarray_list', return_value=True)
    @patch('detectflow.Sampler.sample_frames', return_value=[Mock()])
    def test_get_ref_bboxes(self, mock_sample_frames, mock_is_valid_ndarray_list, mock_detect):
        # Create a single mock frame
        mock_frame = Mock()
        mock_sample_frames.return_value = [mock_frame]

        # Configure the mock predictor and detect method
        mock_boxes = [Mock()]
        mock_detection_result = Mock()
        mock_detection_result.boxes = mock_boxes
        mock_detect.return_value = [mock_detection_result]

        diagnoser = VideoDiagnoser(
            video_path=self.video_path,
            flowers_model_path=self.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=self.output_path
        )

        ref_bboxes = diagnoser.get_ref_bboxes()
        mock_detect.assert_called_once()

        # Instead of comparing mocks directly, check if the structure is as expected
        print(ref_bboxes)
        self.assertIsInstance(ref_bboxes, list)
        self.assertEqual(len(ref_bboxes), 1)
        self.assertIsInstance(ref_bboxes[0], list)
        self.assertEqual(len(ref_bboxes[0]), 1)
        self.assertIsInstance(ref_bboxes[0][0], Mock)

    def test_get_frames_real_video(self):
        import cv2
        import numpy as np
        diagnoser = VideoDiagnoser(
            video_path=self.video_path,
            flowers_model_path=self.flowers_model_path,
            flowers_model_conf=0.3,
            motion_methods="SOM",
            frame_skip=1,
            color_variance_threshold=10,
            verbose=True,
            output_path=self.output_path
        )
        frames = diagnoser._get_frames()

        # Check that frames are a list and contain numpy arrays
        self.assertIsInstance(frames, list)
        self.assertTrue(all(isinstance(frame, np.ndarray) for frame in frames))

        # Check that each frame is a valid image
        for frame in frames:
            self.assertEqual(len(frame.shape), 3)  # Should have 3 dimensions (height, width, channels)
            self.assertTrue(frame.shape[0] > 0)    # Height should be greater than 0
            self.assertTrue(frame.shape[1] > 0)    # Width should be greater than 0
            self.assertTrue(frame.shape[2] in [1, 3, 4])  # Channels should be 1 (grayscale), 3 (RGB), or 4 (RGBA)

if __name__ == '__main__':
    unittest.main()
