from typing import Optional
import os
import imageio
import cv2
import decord
from functools import lru_cache

class VideoValidator:
    def __init__(self):
        pass

    @staticmethod
    @lru_cache(maxsize=32)
    def validate_video_readers(video_path, min_duration_sec=5):
        #print(f"INFO: (Video Validator) Starting validation for video: {os.path.basename(self.video_path)}")

        results = {
            'opencv': False,
            'imageio': False,
            'decord': False
        }

        # Attempt to validate with each reader based on the origin
        try:
            results['opencv'] = VideoValidator._validate_with_cv2(video_path, min_duration_sec)
            results['imageio'] = VideoValidator._validate_with_imageio(video_path, min_duration_sec)
            results['decord'] = VideoValidator._validate_with_decord(video_path, min_duration_sec)
        except Exception as e:
            print(f"ERROR: (Video Validator) Error validating video: {e}")

        print(f"INFO: (Video Validator) Validated video: {os.path.basename(video_path)}")

        # Reporting all methods which successfully validated the video
        return results

    @staticmethod
    @lru_cache(maxsize=32)
    def validate_video(video_path):
        """
        Validate video using cv2 and, if that fails, try with imageio.
        """
        if VideoValidator._validate_with_cv2_for_download(video_path):
            return True
        elif VideoValidator._validate_with_imageio_for_download(video_path):
            return True
        else:
            return False

    @staticmethod
    def _validate_with_cv2_for_download(video_path):
        """
        Validate the downloaded video using cv2.
        """
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            return ret
        except Exception as e:
            print(f"ERROR: (Video Validator) Error validating with cv2: {e}")
            return False
        finally:
            cap.release()  # Ensure the capture is always released, even if an error occurs

    @staticmethod
    def _validate_with_imageio_for_download(video_path):
        """
        Validate the downloaded video using imageio.
        """
        try:
            with imageio.get_reader(video_path) as video:
                _ = video.get_data(0)
                return True
        except Exception as e:
            print(f"ERROR: (Video Validator) Error validating with imageio: {e}")
            return False

    @staticmethod
    def _validate_with_cv2(video_path, min_duration_sec=5):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            return VideoValidator._check_video_integrity(cap, min_duration_sec, method='opencv')
        finally:
            if cap:
                cap.release()

    @staticmethod
    def _validate_with_imageio(video_path, min_duration_sec=5):
        video = None
        try:
            video = imageio.get_reader(video_path)
            return VideoValidator._check_video_integrity(video, min_duration_sec, method='imageio')
        finally:
            if video:
                del video

    @staticmethod
    def _validate_with_decord(video_path, min_duration_sec=5):
        try:
            video = decord.VideoReader(video_path)
            return VideoValidator._check_video_integrity(video, min_duration_sec, method='decord')
        except decord.DecordError:
            return False

    @staticmethod
    def _check_video_integrity(video, min_duration_sec, method):
        try:
            # Check duration and read frames logic
            if method == 'opencv':
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            elif method == 'imageio':
                meta_data = video.get_meta_data()
                fps = meta_data.get('fps', 0)
                frame_count = video.count_frames()
            elif method == 'decord':
                fps = video.get_avg_fps()
                frame_count = len(video)

            duration = frame_count / fps if fps > 0 else 0
            if duration < min_duration_sec:
                return False

            # Reading frames
            if method == 'opencv':
                success_count = sum(1 for _ in range(2) if video.read()[0])
                return success_count >= 2
            elif method == 'imageio':
                return all(video.get_data(i) is not None for i in range(2))
            elif method == 'decord':
                return all(video[i] is not None for i in range(2))

        except Exception as e:
            print(f"ERROR: (Video Validator) Error checking video integrity using {method}: {e}")
            return False