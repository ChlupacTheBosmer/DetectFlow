from typing import Optional
import os
import imageio
import cv2
import decord

class VideoValidator:
    def __init__(self, video_path, video_origin: Optional[str] = None):
        self.video_path = video_path

        # Determine video origin
        if not video_origin:
            if video_path.endswith(".mp4"):
                self.video_origin = "VT"
            elif video_path.endswith(".avi"):
                self.video_origin = "MS"
        else:
            self.video_origin = video_origin

    def validate_video_readers(self, min_duration_sec=5):
        print(f"INFO: (Video Validator) Starting validation for video: {os.path.basename(self.video_path)}")

        results = {
            'cv2': False,
            'imageio': False,
            'decord': False
        }

        # Attempt to validate with each reader based on the origin
        if self.video_origin in ['VT', 'MS']:
            results['cv2'] = self._validate_with_cv2(min_duration_sec)
            results['imageio'] = self._validate_with_imageio(min_duration_sec)
            results['decord'] = self._validate_with_decord(min_duration_sec)
        else:
            raise ValueError(f"ERROR: (Video Validator) Unsupported video_origin: {self.video_origin}")

        print(f"INFO: (Video Validator) Validated video: {os.path.basename(self.video_path)}")

        # Reporting all methods which successfully validated the video
        return results

    def validate_video(self):
        """
        Validate video using cv2 and, if that fails, try with imageio.
        """
        if self._validate_with_cv2_for_download():
            return True
        elif self._validate_with_imageio_for_download():
            return True
        else:
            return False

    def _validate_with_cv2_for_download(self):
        """
        Validate the downloaded video using cv2.
        """
        cap = cv2.VideoCapture(self.video_path)
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

    def _validate_with_imageio_for_download(self):
        """
        Validate the downloaded video using imageio.
        """
        try:
            with imageio.get_reader(self.video_path) as video:
                _ = video.get_data(0)
                return True
        except Exception as e:
            print(f"ERROR: (Video Validator) Error validating with imageio: {e}")
            return False

    def _validate_with_cv2(self, min_duration_sec):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return False
            return self._check_video_integrity(cap, min_duration_sec, method='cv2')
        finally:
            if cap:
                cap.release()

    def _validate_with_imageio(self, min_duration_sec):
        video = None
        try:
            video = imageio.get_reader(self.video_path)
            return self._check_video_integrity(video, min_duration_sec, method='imageio')
        finally:
            if video:
                del video

    def _validate_with_decord(self, min_duration_sec):
        try:
            video = decord.VideoReader(self.video_path)
            return self._check_video_integrity(video, min_duration_sec, method='decord')
        except decord.DecordError:
            return False

    def _check_video_integrity(self, video, min_duration_sec, method):
        try:
            # Check duration and read frames logic
            if method == 'cv2':
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
            if method == 'cv2':
                success_count = sum(1 for _ in range(2) if video.read()[0])
                return success_count >= 2
            elif method == 'imageio':
                return all(video.get_data(i) is not None for i in range(2))
            elif method == 'decord':
                return all(video[i] is not None for i in range(2))

        except Exception as e:
            print(f"ERROR: (Video Validator) Error checking video integrity using {method}: {e}")
            return False