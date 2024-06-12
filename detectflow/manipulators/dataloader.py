import os
import logging
from typing import List, Tuple, Union, Optional
from detectflow.config import S3_CONFIG
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.video_validator import VideoValidator


class Dataloader(S3Manipulator, Manipulator):
    def __init__(
            self,
            cfg_file: str = S3_CONFIG):

        # Call the __init__ method of the S3Manipulator
        S3Manipulator.__init__(self, cfg_file)

        # Call the __init__ method of Manipulator
        Manipulator.__init__(self)

    def validate_video(self, video_path: str) -> bool:
        """
        Validate that a downloaded video is not corrupted.

        :param video_path: Path to the downloaded video.
        :return: True if the video is valid, False otherwise.
        """
        try:
            # Validate filepath
            if self.is_valid_file_path(video_path):

                # Validate download
                if not VideoValidator(video_path).validate_video():
                    logging.error(f"Download validation failed for video: {video_path}")
                    return False

                # Additional validations can be performed here if necessary

                return True
            else:
                raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
        except Exception as e:
            logging.error(f"Error during video validation for {video_path}: {e}")
            return False

    def prepare_data(self, video_paths: List[str], target_directory: str, remove_invalid: bool = True) -> tuple[
        list[str], list[str]]:
        """
        Prepare video data by downloading or moving files to the target directory and validating them.

        :param video_paths: List of S3 or local file paths to videos.
        :param target_directory: Directory where videos should be stored and validated.
        :return: List of paths to valid video files in the target directory.
        """
        # Prepare the destination folder
        os.makedirs(target_directory, exist_ok=True)

        valid_videos = []
        invalid_videos = []
        for video_path in video_paths:
            try:
                if self.is_s3_file(video_path):
                    # Handle S3 file
                    bucket_name, key = self._parse_s3_path(video_path)
                    local_path = os.path.join(target_directory, os.path.basename(key))
                    if self.is_valid_file_path(local_path) or self.download_file_s3(bucket_name, key, local_path):
                        if self.validate_video(local_path):
                            valid_videos.append(local_path)
                        else:
                            logging.error(f"Validation failed for downloaded video: {local_path}")
                            invalid_videos.append(local_path)
                            if remove_invalid:
                                Manipulator.delete_file(local_path)
                elif self.is_valid_file_path(video_path):
                    dest_path = os.path.join(target_directory, os.path.basename(video_path))
                    if self.is_valid_file_path(dest_path) or Manipulator.move_file(video_path, target_directory,
                                                                                        overwrite=True, copy=True):
                        if self.validate_video(dest_path):
                            valid_videos.append(dest_path)
                        else:
                            logging.error(f"Validation failed for transfered video: {dest_path}")
                            invalid_videos.append(dest_path)
                            if remove_invalid:
                                Manipulator.delete_file(dest_path)
                else:
                    logging.error(f"Invalid video path provided: {video_path}")
            except Exception as e:
                logging.error(f"Error processing video {video_path}: {e}")
        return valid_videos, invalid_videos
