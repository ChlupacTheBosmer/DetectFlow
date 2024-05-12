import os
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.s3_validator import S3Validator
from detectflow.validators.video_validator import VideoValidator
from detectflow.validators.validator import Validator
import logging
from typing import List, Tuple, Union, Optional


class Dataloader(S3Manipulator, Manipulator):
    def __init__(
            self,
            cfg_file: str = "/storage/brno2/home/USER/.s3.cfg",
            validator=None):

        # Init the validator if none is passed
        if validator is None:
            self.validator = S3Validator()
        else:
            self.validator = validator

        # Call the __init__ method of the S3Manipulator
        S3Manipulator.__init__(self, cfg_file, self.validator)

        # Call the __init__ method of Manipulator
        Manipulator.__init__(self, self.validator)

    def validate_video(self, video_path: str) -> bool:
        """
        Validate that a downloaded video is not corrupted.

        :param video_path: Path to the downloaded video.
        :param video_origin: The origin of the video, used for extended validation if needed.
        :return: True if the video is valid, False otherwise.
        """
        try:
            # Validate filepath
            if self.validator.is_valid_file_path(video_path):
                validator = VideoValidator(video_path)

                # Validate download
                if not validator.validate_video():
                    logging.error(f"Download validation failed for video: {video_path}")
                    return False

                # Additional validations can be performed here if necessary

                return True
            else:
                raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
        except Exception as e:
            logging.error(f"Error during video validation for {video_path}: {e}")
            return False

    def prepare_data(self, video_paths: List[str], target_directory: str, remove_invalid: bool = True) -> List[str]:
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
                if self.validator.is_s3_file(video_path):
                    # Handle S3 file
                    bucket_name, key = self.validator._parse_s3_path(video_path)
                    local_path = os.path.join(target_directory, os.path.basename(key))
                    if Validator.is_valid_file_path(local_path) or self.download_file_s3(bucket_name, key, local_path):
                        if self.validate_video(local_path):
                            valid_videos.append(local_path)
                        else:
                            logging.error(f"Validation failed for downloaded video: {local_path}")
                            invalid_videos.append(local_path)
                            if remove_invalid:
                                Manipulator.delete_file(local_path)
                elif Validator.is_valid_file_path(video_path):
                    dest_path = os.path.join(target_directory, os.path.basename(video_path))
                    if Validator.is_valid_file_path(dest_path) or Manipulator.move_file(video_path, target_directory,
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
