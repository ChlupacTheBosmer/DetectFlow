import unittest
from unittest.mock import patch, Mock, MagicMock
import os
from datetime import datetime, timedelta

# Import the Video class
from detectflow.video.video_data import Video


class TestVideo(unittest.TestCase):

    def setUp(self):
        self.video_path = "./resources/GR2_L1_TolUmb2_20220524_07_44.mp4"
        self.s3_bucket = "test_bucket"
        self.s3_directory = "test_directory"
        self.s3_path = f"s3://{self.s3_bucket}/{self.s3_directory}/GR2_L1_TolUmb2_20220524_07_44.mp4"

    def test_initialization(self):
        video = Video(video_path=self.video_path, s3_bucket=self.s3_bucket, s3_directory=self.s3_directory)
        self.assertEqual(video.video_path, self.video_path)
        self.assertEqual(video.video_name, os.path.basename(self.video_path))
        self.assertEqual(video.s3_bucket, self.s3_bucket)
        self.assertEqual(video.s3_directory, self.s3_directory)
        self.assertEqual(video.s3_path, self.s3_path)
        self.assertEqual(video.reader_method, "decord")

    def test_initialization_with_missing_parameters(self):
        video = Video(video_path=self.video_path)
        self.assertEqual(video.video_path, self.video_path)
        self.assertEqual(video.video_name, os.path.basename(self.video_path))
        self.assertIsNone(video.s3_bucket)
        self.assertIsNone(video.s3_directory)
        self.assertIsNone(video.s3_path)

    def test_parse_recording_name(self):
        video = Video(video_path=self.video_path)
        parsed_data = video.parse_recording_name(self.video_path)
        self.assertEqual(parsed_data['recording_id'], 'GR2_L1_TolUmb2')
        self.assertEqual(parsed_data['timestamp'], '20220524_07_44')
        self.assertEqual(parsed_data['locality'], 'GR2')
        self.assertEqual(parsed_data['transect'], 'L1')
        self.assertEqual(parsed_data['plant_id'], 'TolUmb2')
        self.assertEqual(parsed_data['date'], '20220524')
        self.assertEqual(parsed_data['hour'], '07')
        self.assertEqual(parsed_data['minute'], '44')
        self.assertEqual(parsed_data['extension'], '.mp4')

    @patch('detectflow.validators.video_validator.VideoValidator')
    def test_get_readers(self, MockVideoValidator):
        mock_validator = MockVideoValidator.return_value
        mock_validator.validate_video_readers.return_value = {"opencv": True, "imageio": False, "decord": False}

        video = Video(video_path=self.video_path)
        self.assertEqual(video.readers, ["opencv"])
        MockVideoValidator.assert_called_once_with(self.video_path)

    @patch('hachoir.parser.createParser')
    @patch('hachoir.metadata.extractMetadata')
    def test_get_fps(self, mock_createParser, mock_extractMetadata):
        # Mocking metadata extraction
        mock_metadata = Mock()
        mock_metadata.get.return_value = 30.0
        mock_extractMetadata.return_value = mock_metadata

        video = Video(video_path=self.video_path)
        self.assertEqual(video.fps, 30)

    @patch('cv2.VideoCapture')
    def test_get_total_frames(self, mock_VideoCapture):
        mock_capture = mock_VideoCapture.return_value
        mock_capture.get.side_effect = [1000, 25]  # Total frames and fps

        video = Video(video_path=self.video_path)
        self.assertEqual(video.total_frames, 1000)
        mock_capture.release.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('hachoir.parser.createParser')
    @patch('hachoir.metadata.extractMetadata')
    def test_get_duration(self, mock_createParser, mock_extractMetadata, mock_VideoCapture):
        # Mocking VideoCapture for cv2
        mock_capture = mock_VideoCapture.return_value
        mock_capture.get.side_effect = [1000, 25]  # Total frames and fps

        # Mocking metadata extraction
        mock_metadata = Mock()
        mock_metadata.get.return_value = "0:0:40"  # Duration 40 seconds
        mock_extractMetadata.return_value = mock_metadata

        video = Video(video_path=self.video_path)
        self.assertEqual(video.duration, timedelta(seconds=40))
        mock_capture.release.assert_called_once()

    @patch('hachoir.parser.createParser')
    @patch('hachoir.metadata.extractMetadata')
    def test_get_start_time(self, mock_createParser, mock_extractMetadata):
        # Mocking metadata extraction
        mock_metadata = Mock()
        mock_metadata.get.return_value = "2022-01-01 12:30:00"
        mock_extractMetadata.return_value = mock_metadata

        video = Video(video_path=self.video_path)
        self.assertEqual(video.start_time, datetime(2022, 1, 1, 12, 30, 0))

    @patch('hachoir.parser.createParser')
    @patch('hachoir.metadata.extractMetadata')
    def test_get_end_time(self, mock_createParser, mock_extractMetadata):
        # Mocking metadata extraction
        mock_metadata = Mock()
        mock_metadata.get.side_effect = ["2022-01-01 12:30:00", "0:0:40"]  # Start time and duration
        mock_extractMetadata.return_value = mock_metadata

        video = Video(video_path=self.video_path)
        self.assertEqual(video.end_time, datetime(2022, 1, 1, 12, 30, 40))

    @patch('cv2.VideoCapture')
    def test_get_frame_shape(self, mock_VideoCapture):
        mock_capture = mock_VideoCapture.return_value
        mock_capture.get.side_effect = [1920, 1080]

        video = Video(video_path=self.video_path)
        self.assertEqual(video.frame_width, 1920)
        self.assertEqual(video.frame_height, 1080)
        mock_capture.release.assert_called_once()

    @patch('detectflow.video.picture_quality.PictureQualityAnalyzer')
    @patch.object(Video, 'read_video_frame', return_value=[{'frame': Mock()}])
    def test_picture_quality_properties(self, mock_read_video_frame, MockPictureQualityAnalyzer):
        # Mock PictureQualityAnalyzer
        mock_analyzer = MockPictureQualityAnalyzer.return_value
        mock_analyzer.blur = 0.5
        mock_analyzer.focus = 0.7
        mock_analyzer.focus_regions = [(0, 0, 10, 10)]
        mock_analyzer.contrast = 0.8
        mock_analyzer.brightness = 0.6
        mock_analyzer.color_variance = 0.4

        video = Video(video_path=self.video_path)

        self.assertEqual(video.blur, 0.5)
        self.assertEqual(video.focus, 0.7)
        self.assertEqual(video.focus_regions, [(0, 0, 10, 10)])
        self.assertEqual(video.contrast, 0.8)
        self.assertEqual(video.brightness, 0.6)
        self.assertEqual(video.color_variance, 0.4)

    def test_eq(self):
        video1 = Video(video_path=self.video_path)
        video2 = Video(video_path=self.video_path)
        self.assertEqual(video1, video2)


if __name__ == '__main__':
    unittest.main()
