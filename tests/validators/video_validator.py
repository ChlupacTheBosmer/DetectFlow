import unittest
import os
import time
from detectflow.validators.video_validator import VideoValidator
from detectflow.config import DETECTFLOW_DIR


class TestVideoValidator(unittest.TestCase):

    def setUp(self):
        # Set up paths to test video files
        self.valid_video_mp4 = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "video", "resources", "GR2_L1_TolUmb2_20220524_07_44.mp4")
        self.valid_video_avi = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "video", "resources", "CZ2_T1_AciArv04_20210623_11_05.avi")
        self.invalid_video = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "video", "resources", "invalid_video.mp4")  # This should be a path to a corrupt or invalid video file

        # Ensure the test files exist
        assert os.path.exists(self.valid_video_mp4), f"Test video file does not exist: {self.valid_video_mp4}"
        assert os.path.exists(self.valid_video_avi), f"Test video file does not exist: {self.valid_video_avi}"
        assert os.path.exists(self.invalid_video), f"Test video file does not exist: {self.invalid_video}"

    def test_validate_valid_video_mp4(self):
        result = VideoValidator.validate_video_readers(self.valid_video_mp4)
        print(result)
        self.assertTrue(result['opencv'] or result['imageio'] or result['decord'],
                        "None of the readers validated the MP4 video.")

    def test_validate_valid_video_avi(self):
        result = VideoValidator.validate_video_readers(self.valid_video_avi)
        print(result)
        self.assertTrue(result['opencv'] or result['imageio'] or result['decord'],
                        "None of the readers validated the AVI video.")

    def test_validate_invalid_video(self):
        result = VideoValidator.validate_video_readers(self.invalid_video)
        print(result)
        self.assertFalse(result['opencv'] or result['imageio'] or result['decord'],
                         "Invalid video should not be validated by any reader.")


class TestVideoValidatorCache(unittest.TestCase):

    def setUp(self):
        self.valid_video_mp4 = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "video", "resources",
                                            "GR2_L1_TolUmb2_20220524_07_44.mp4")
        self.valid_video_avi = os.path.join(os.path.dirname(DETECTFLOW_DIR), "tests", "video", "resources",
                                            "CZ2_T1_AciArv04_20210623_11_05.avi")

        # Ensure the test files exist
        assert os.path.exists(self.valid_video_mp4), f"Test video file does not exist: {self.valid_video_mp4}"
        assert os.path.exists(self.valid_video_avi), f"Test video file does not exist: {self.valid_video_avi}"

    def test_cache_single_instance(self):

        # Measure the time for the first call
        start_time = time.time()
        result1 = VideoValidator.validate_video_readers(self.valid_video_mp4)
        first_call_duration = time.time() - start_time

        # Measure the time for the second call
        start_time = time.time()
        result2 = VideoValidator.validate_video_readers(self.valid_video_mp4)
        second_call_duration = time.time() - start_time

        self.assertTrue(result1['opencv'] or result1['imageio'] or result1['decord'],
                        "None of the readers validated the video on first call.")
        self.assertEqual(result1, result2, "Cached result should be the same as the first call result.")
        self.assertLess(second_call_duration, first_call_duration,
                        "Second call should be faster due to caching.")
        print(f"First call duration: {first_call_duration:.4f} s")
        print(f"Second call duration: {second_call_duration:.4f} s")

    def test_cache_multiple_instances(self):
        validator = VideoValidator

        # Call validate_video_readers on the first instance
        start_time = time.time()
        result1 = VideoValidator.validate_video_readers(self.valid_video_mp4)
        first_call_duration = time.time() - start_time

        # Call validate_video_readers on the second instance
        start_time = time.time()
        result2 = validator.validate_video_readers(self.valid_video_mp4)
        second_call_duration = time.time() - start_time

        self.assertEqual(result1, result2,
                         "Validation results should be the same for different instances with the same video.")

        # Check the cache status indirectly by ensuring no errors occur on repeated validation
        self.assertTrue(result1['opencv'] or result1['imageio'] or result1['decord'],
                        "None of the readers validated the video on first instance.")
        self.assertTrue(result2['opencv'] or result2['imageio'] or result2['decord'],
                        "None of the readers validated the video on second instance.")
        print(f"First call duration: {first_call_duration:.4f} s")
        print(f"Second call duration: {second_call_duration:.4f} s")


if __name__ == "__main__":
    unittest.main()
