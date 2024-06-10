import unittest
import os
import logging
import cv2
import decord
import numpy as np
from unittest.mock import patch, MagicMock

from detectflow.video.frame_reader import FrameReader

class TestFrameReader(unittest.TestCase):
    def setUp(self):
        self.video_path = './resources/GR2_L1_TolUmb2_20220524_07_44.mp4'
        self.frame_indices = [0, 10, 20]
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Mock frame as a black image

    @patch('detectflow.validators.video_validator.VideoValidator.validate_video')
    def test_get_reader_default(self, mock_validate_video):
        mock_validate_video.return_value = {'decord': False, 'imageio': False, 'opencv': True}
        reader = FrameReader(self.video_path)
        self.assertEqual(reader.get_reader(), reader.read_frames_opencv2)

    @patch('detectflow.validators.video_validator.VideoValidator.validate_video')
    def test_get_reader_with_validation(self, mock_validate_video):
        mock_validate_video.return_value = {'decord': True, 'imageio': False, 'opencv': False}
        reader = FrameReader(self.video_path)
        self.assertEqual(reader.get_reader(), reader.read_frames_decord)

    @patch('detectflow.validators.video_validator.VideoValidator.validate_video')
    def test_get_reader_with_method(self, mock_validate_video):
        reader = FrameReader(self.video_path, reader_method='imageio')
        self.assertEqual(reader.get_reader(), reader.read_frames_imageio)

    @patch('detectflow.validators.video_validator.VideoValidator.validate_video')
    def test_get_reader_invalid_method(self, mock_validate_video):
        reader = FrameReader(self.video_path, reader_method='invalid_method')
        self.assertEqual(reader.get_reader(), reader.read_frames_opencv2)

    @patch('cv2.VideoCapture')
    def test_read_frames_opencv2(self, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.read.side_effect = [(True, self.mock_frame)] * len(self.frame_indices)
        mock_videocapture.return_value = mock_cap

        reader = FrameReader(self.video_path)
        frames = list(reader.read_frames_opencv2(self.frame_indices))

        self.assertEqual(len(frames), len(self.frame_indices))
        for frame, idx in zip(frames, self.frame_indices):
            self.assertEqual(frame['frame_number'], idx)
            np.testing.assert_array_equal(frame['frame'], self.mock_frame)

    @patch('imageio.get_reader')
    def test_read_frames_imageio(self, mock_get_reader):
        mock_video = MagicMock()
        mock_video.get_data.side_effect = [self.mock_frame] * len(self.frame_indices)
        mock_get_reader.return_value = mock_video

        reader = FrameReader(self.video_path)
        frames = list(reader.read_frames_imageio(self.frame_indices))

        self.assertEqual(len(frames), len(self.frame_indices))
        for frame, idx in zip(frames, self.frame_indices):
            self.assertEqual(frame['frame_number'], idx)
            np.testing.assert_array_equal(frame['frame'], self.mock_frame)

    @patch('imageio.get_reader')
    @patch('cv2.VideoCapture')
    def test_read_frames_imageio_fallback(self, mock_videocapture, mock_get_reader):
        mock_video = MagicMock()
        mock_video.get_data.side_effect = Exception('Read error')
        mock_get_reader.return_value = mock_video

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [(True, self.mock_frame)] * len(self.frame_indices)
        mock_videocapture.return_value = mock_cap

        reader = FrameReader(self.video_path)
        frames = list(reader.read_frames_imageio(self.frame_indices))

        self.assertEqual(len(frames), len(self.frame_indices))
        for frame, idx in zip(frames, self.frame_indices):
            self.assertEqual(frame['frame_number'], idx)
            np.testing.assert_array_equal(frame['frame'], self.mock_frame)

# TODO: Add test fro decord. Last time I had trouble mocking the frame for assertion at the end. It seemed to wokr though.

if __name__ == '__main__':
    unittest.main()

