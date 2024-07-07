import os
import logging
from typing import List, Tuple, Union, Optional
from detectflow.config import S3_CONFIG
from detectflow.utils import DOWNLOADS_DIR
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.video_validator import VideoValidator
from datetime import datetime


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
                if not VideoValidator.validate_video(video_path):
                    logging.error(f"Download validation failed for video: {video_path}")
                    return False

                # Additional validations can be performed here if necessary

                return True
            else:
                raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
        except Exception as e:
            logging.error(f"Error during video validation for {video_path}: {e}")
            return False

    def prepare_videos(self, video_paths: List[str], target_directory: str, remove_invalid: bool = True) -> tuple[
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
                    bucket_name, key = self.parse_s3_path(video_path)
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

    def backup_file_s3(self, bucket_name, directory_name, local_file_path, validate_upload=True, validate_callback=None, fallback_bucket_name='data'):
        try:
            if not self.is_s3_bucket(bucket_name):
                logging.warning(f"Bucket {bucket_name} does not exist. Attempting backup into the '{fallback_bucket_name}' bucket.")
                bucket_name = fallback_bucket_name
                self.create_bucket_s3(bucket_name)
        except Exception as e:
            logging.error(f"Error during bucket resolution. Attempted to create bucket: {bucket_name}: {e}")
            bucket_name = fallback_bucket_name
            if self.is_s3_bucket(bucket_name):
                logging.warning(f"Bucket {bucket_name} exists. Proceeding with backup into the '{fallback_bucket_name}' bucket.")

        try:
            if not self.is_s3_directory(f"{bucket_name}/{directory_name}"):
                logging.warning(f"Directory {directory_name} does not exist. File backed up into the '{fallback_bucket_name}' bucket.")
                bucket_name = fallback_bucket_name
                self.create_directory_s3(bucket_name, directory_name)
        except Exception as e:
            logging.error(
                f"Error during directory resolution. Attempted to create directory: {bucket_name}/{directory_name}: {e}")
            bucket_name = fallback_bucket_name
            if self.is_s3_directory(f"{bucket_name}/{directory_name}"):
                logging.warning(
                    f"Directory {bucket_name}/{directory_name} exists. Proceeding with backup into the '{fallback_bucket_name}' bucket.")

        s3_file_path = f"{directory_name}{os.path.basename(local_file_path)}"  # Path in the bucket where the file will be uploaded
        try:
            # Upload a file to the specified directory
            self.upload_file_s3(bucket_name, local_file_path, s3_file_path)
            logging.info(f"Uploaded {local_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
        except Exception as e:
            logging.error(f"Failed to upload {local_file_path} to S3: {e}")

        if validate_upload:
            self.validate_backup_s3(bucket_name, s3_file_path, local_file_path, validate_callback)

    def validate_backup_s3(self, bucket_name: str, s3_file_path: str, orig_file_path: str, validate_callback=None):
        from detectflow.utils.file import compare_file_sizes

        tmp_folder = os.path.join(DOWNLOADS_DIR, os.path.dirname(orig_file_path))
        tmp_file_path = os.path.join(tmp_folder, os.path.basename(orig_file_path))
        os.makedirs(tmp_folder, exist_ok=True)

        try:
            # Validate the upload
            if self.is_s3_file(f"s3://{bucket_name}/{s3_file_path}"):
                self.download_file_s3(bucket_name, s3_file_path, tmp_file_path)
                if compare_file_sizes(orig_file_path, tmp_file_path, 0.05):  # Tolerance of 5%
                    if validate_callback:
                        if not validate_callback(filepath=tmp_file_path, s3_path=s3_file_path,
                                                 orig_filepath=orig_file_path):
                            raise RuntimeError(
                                f"Failed to validate the upload of {orig_file_path} to S3 bucket {bucket_name}/{s3_file_path} using custom callback.")
                    logging.info(
                        f"Successfully validated the upload of {orig_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
                else:
                    raise RuntimeError(
                        f"Partially validated the upload of {orig_file_path} to S3 bucket {bucket_name}/{s3_file_path}. Size discrepancy found.")
            else:
                raise RuntimeError(
                    f"Failed to validate the upload of {orig_file_path} to S3 bucket {bucket_name}/{s3_file_path}.")
        except Exception as e:
            raise RuntimeError(f"Validation error: {e}") from e

    def locate_file_s3(self, pattern: str, bucket_name: str, selection_criteria: str = 'name'):

        # Check online for the file
        try:
            online_files = self.find_files_s3(pattern, bucket_name)
            if len(online_files) > 1:
                online_files = self.sort_files_s3(online_files, sort_by=selection_criteria, ascending=True)
            online_file = online_files[0] if len(online_files) > 0 else None
        except Exception as e:
            logging.error(f"Error while searching file {pattern} in S3: {e}")
            online_file = None

        return online_file

    def locate_file_local(self, pattern: str, folder_path: str, selection_criteria: str = 'name'):

        # Check in scratch for the file
        try:
            local_files = self.find_files(folder_path, pattern)
            if len(local_files) > 1:
                local_files = Manipulator.sort_files(local_files, sort_by=selection_criteria, ascending=True)
            local_file = local_files[0] if len(local_files) > 0 else None
        except Exception as e:
            logging.error(f"Error while searching file {pattern} from local directory: {e}")
            local_file = None

        return local_file

    def get_version_metadata_s3(self, s3_path: str):
        import pytz

        try:
            if s3_path:
                online_date = self.get_metadata_s3(s3_path, 'LastModified')
                if online_date.tzinfo is not None and online_date.utcoffset() is not None:
                    local_timezone = pytz.timezone("Europe/Prague")
                    online_date = online_date.astimezone(local_timezone).replace(tzinfo=None)
                metadata_dict = {
                               'file': s3_path,
                               'date': online_date,
                               'size': int(self.get_metadata_s3(s3_path, 'ContentLength'))
                }
            else:
                metadata_dict = None
        except Exception as e:
            logging.error(f"Error while getting metadata for file {os.path.basename(s3_path)} from S3: {e}")
            metadata_dict = {'file': s3_path, 'date': None, 'size': None}

        return metadata_dict

    def get_version_metadata_local(self, file_path: str):
        try:
            if file_path:
                metadata_dict = {
                              'file': file_path,
                              'date': datetime.fromtimestamp(os.path.getmtime(file_path)),
                              'size': int(os.path.getsize(file_path))
                }
            else:
                metadata_dict = None
        except Exception as e:
            logging.error(f"Error while getting metadata for file {os.path.basename(file_path)} from local directory: {e}")
            metadata_dict = {'file': file_path, 'date': None, 'size': None}

        return metadata_dict
