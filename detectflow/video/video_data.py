from typing import Optional, Type
import os
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import datetime
from datetime import datetime, timedelta
import logging
import cv2

from detectflow.validators.s3_validator import S3Validator
from detectflow.validators.video_validator import VideoValidator
from detectflow.video.frame_reader import FrameReader
from detectflow.video.picture_quality import PictureQualityAnalyzer
from functools import lru_cache


class Video(FrameReader):
    """
    A class for managing all video data.
    """

    def __init__(self,
                 video_path: str,
                 s3_bucket: Optional[str] = None,
                 s3_directory: Optional[str] = None,
                 s3_path: Optional[str] = None,
                 reader_method: Optional[str] = None):
        """
        Initializes the Video class.
        """
        # Parse the recording name
        parsed_data = self.parse_recording_name(video_path)

        # Set the attributes
        self.recording_id = parsed_data.get("recording_id", None)
        self.timestamp = parsed_data.get("timestamp", None)
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.video_id = self.video_name
        self.s3_bucket = s3_bucket if s3_bucket else (S3Validator._parse_s3_path(s3_path)[0] if s3_path else None)
        self.s3_directory = s3_directory if s3_directory else (S3Validator._parse_s3_path(s3_path)[1] if s3_path else None)
        self.s3_path = s3_path if s3_path else f"s3://{s3_bucket}/{s3_directory}/{self.video_name}" if s3_bucket and s3_directory else None
        self.extension = parsed_data.get("extension", None)

        self._duration = None
        self._total_frames = None
        self._fps = None
        self._readers = None
        self._start_time = None
        self._end_time = None

        self._frame_width = None
        self._frame_height = None

        self._focus = None
        self._focus_regions = None
        self._focus_heatmap = None
        self._blur = None
        self._brightness = None
        self._contrast = None
        self._color_variance = None

        # Call the __init__ method of FrameReader
        FrameReader.__init__(self, video_path, reader_method=None)
        self.reader_method = next((reader for reader in self.READERS if reader in self.readers), "opencv") if not reader_method else reader_method # Has to be updated after the init of the FrameReader

    @property
    def readers(self):
        if not self._readers:
            self._readers = self.get_readers()
        return self._readers

    def get_readers(self):
        # Run validator on video
        try:
            result = VideoValidator(self.video_path).validate_video_readers()
        except Exception as e:
            logging.warning(f'Unable to validate video: {self.video_name}. Using default reader. Exception: {e}')
            return ["opencv"]

        # return list of validated reader methods
        return [method for method, status in result.items() if status]

    def parse_recording_name(self, video_path: str):

        filename = os.path.basename(video_path)

        # Prepare name elements
        try:
            locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")
            file_extension = os.path.splitext(filename)[-1]
        except Exception as e:
            logging.warning(f'Unable to parse video recording name: {self.video_path}. Exception: {e}')
            return {"recording_id": filename,
                    "timestamp": None,
                    "locality": None,
                    "transect": None,
                    "plant_id": None,
                    "date": None,
                    "hour": None,
                    "minute": None,
                    "extension": os.path.splitext(filename)[-1]}

        # Define compound info
        recording_identifier = "_".join([locality, transect, plant_id])
        timestamp = "_".join([date, hour, minutes])

        return {"recording_id": recording_identifier,
                "timestamp": timestamp,
                "locality": locality,
                "transect": transect,
                "plant_id": plant_id,
                "date": date,
                "hour": hour,
                "minute": minutes,
                "extension": file_extension}

    @property
    def fps(self):
        if self._fps is None:
            self._fps = self.get_fps()

        return self._fps

    def get_fps(self):

        try:
            try:
                # Try to get the fps using cv2
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            except Exception as e:
                # Get the fps using meatadata
                parser = createParser(self.video_path)
                metadata = extractMetadata(parser)
                fps = float(metadata.get("frame_rate"))
        except Exception as e:
            logging.warning(f'Unable to read video fps: {self.video_path}. Exception: {e}')
            fps = 25

        return int(fps)

    @property
    def total_frames(self):
        if self._total_frames is None:
            self._total_frames = self.get_total_frames()

        return self._total_frames

    def get_total_frames(self):

        try:
            try:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception as e:
                # Will return timedelta object and get total number of seconds and calculate total frame number
                total_frames = int(self.duration.total_seconds() * self.fps)
        except Exception as e:
            logging.warning(f'Unable to read total frames: {self.video_path}. Exception: {e}')
            total_frames = None

        return total_frames

    @property
    def duration(self):
        if not self._duration:
            self._duration = self.get_duration()

        return self._duration

    def get_duration(self):

        try:
            try:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                duration = timedelta(seconds=total_frames / fps)
            except Exception as e:
                # Get the duration from metadata
                parser = createParser(self.video_path)
                metadata = extractMetadata(parser)
                duration = str(metadata.get("duration"))
                hours, minutes, seconds = map(float, duration.split(':'))
                duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except Exception as e:
            logging.warning(f'Unable to read video duration from video: {self.video_path}. Exception: {e}')
            duration = None

        return duration

    @property
    def start_time(self):
        if not self._start_time:
            self._start_time = self.get_start_time()

        return self._start_time

    def get_start_time(self):
        try:
            # Get the creation date from metadata
            parser = createParser(self.video_path)
            metadata = extractMetadata(parser)
            modify_date = str(metadata.get("creation_date"))  # 2022-05-24 08:29:09

            # Convert the date into a datetime object
            start_time = datetime.strptime(modify_date, '%Y-%m-%d %H:%M:%S')
            logging.debug("Obtained video start time from metadata.")
        except Exception as e:
            logging.warning(f'Unable to read video start time from video metadata: {self.video_path}. Trying alternatives. Exception: {e}')

            # TODO: Implement alternative methods to get start time - via Google Vision API
            start_time = None

        # Use hour and minute info from the recording name (no issue with timezone) and add precisely extracted seconds
        return datetime.strptime(self.timestamp, '%Y%m%d_%H_%M') + timedelta(seconds=start_time.second)

    @property
    def end_time(self):
        if not self._end_time:
            self._end_time = self.get_end_time()

        return self._end_time

    def get_end_time(self):
        try:
            # Calculate the end time by adding the duration to the start time
            end_time = self.start_time + self.duration

            logging.debug("Obtained video end time from metadata.")
        except Exception as e:
            logging.warning(f'Unable to calculate video end time from start time: {self.video_path}. Trying alternatives. Exception: {e}')

            # TODO: Implement alternative methods to get end time - via Google Vision API
            end_time = None

        return end_time

    @property
    def frame_width(self):
        if not self._frame_width:
            self._frame_width, self._frame_height = self.get_frame_shape()

        return self._frame_width

    @property
    def frame_height(self):
        if not self._frame_height:
            self._frame_width, self._frame_height = self.get_frame_shape()

        return self._frame_height

    def get_frame_shape(self):

        try:
            try:
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception as e:
                # TODO: Implement alternative methods to get frame dimensions using a reader.
                frame_width, frame_height = None, None
        except Exception as e:
            logging.warning(f'Unable to read video frame dimensions: {self.video_path}. Exception: {e}')
            frame_width, frame_height = None, None

        return frame_width, frame_height

    @lru_cache(maxsize=64)
    def get_picture_quality(self, frame_number: int = 0):
        frame_number = max(0, frame_number if frame_number < self.total_frames else self.total_frames - 1)
        frame = self.read_video_frame(frame_indices=frame_number, stream=False)[0].get("frame")
        return PictureQualityAnalyzer(frame)

    @property
    def blur(self):
        if not self._blur:
            self._blur = self.get_picture_quality(24).blur
        return self._blur

    @property
    def focus(self):
        if not self._focus:
            self._focus = self.get_picture_quality(24).focus
        return self._focus

    @property
    def focus_regions(self):
        if not self._focus_regions:
            self._focus_regions = self.get_picture_quality(24).focus_regions
        return self._focus_regions

    @property
    def focus_heatmap(self):
        if self._focus_heatmap is None:
            self._focus_heatmap = self.get_picture_quality(24).focus_heatmap
        return self._focus_heatmap

    @property
    def contrast(self):
        if not self._contrast:
            self._contrast = self.get_picture_quality(24).contrast
        return self._contrast

    @property
    def brightness(self):
        if not self._brightness:
            self._brightness = self.get_picture_quality(24).brightness
        return self._brightness

    @property
    def color_variance(self):
        if not self._color_variance:
            self._color_variance = self.get_picture_quality(24).color_variance
        return self._color_variance

    def __str__(self):
        return f"Video: {self.video_name} - Video ID: {self.video_id} - Recording ID: {self.recording_id} - S3 Path: {self.s3_path}"

    def __repr__(self):
        return f"Video({self.video_path}, s3_bucket={self.s3_bucket}, s3_directory={self.s3_directory}, s3_path={self.s3_path}, reader_method={self.reader_method})"

    def __eq__(self, other):
        if isinstance(other, Video):
            return self.video_name == other.video_name
        return False

