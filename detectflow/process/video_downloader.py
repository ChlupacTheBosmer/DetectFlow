import os
import random
import json
import logging
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.manipulator import Manipulator
from detectflow.utils import DOWNLOADS_DIR, CHECKPOINTS_DIR


class VideoDownloader:
    """
    VideoDownloader is designed to facilitate downloading videos from an S3 storage,
    with support for ordered and random downloading modes, post-download processing via callbacks,
    and flexible directory handling.

    Attributes:
        manipulator (S3Manipulator): An instance of S3Manipulator for handling S3 operations.
        checkpoint_file (str): Path to the file used for storing download progress, allowing for resume capability.
        whitelist_buckets (list): Optional list of buckets to exclusively include in the download process.
        blacklist_buckets (list): Optional list of buckets to exclude from the download process.
        whitelist_directories (list): Optional list of directories to exclusively include in the download process.
        blacklist_directories (list): Optional list of directories to exclude from the download process.
        parent_directory (str): The root directory where downloaded videos are stored.
        maintain_structure (bool): If True, maintains the bucket and directory structure in the parent directory.
        delete_after_process (bool): If True, deletes videos after post-download processing is complete.
        post_download_callback (callable): An optional callback function that is executed on each downloaded video file.

    The callback function should accept a single argument, the path to the downloaded video, and perform any desired operations.

    Methods:
        _read_checkpoint: Reads the checkpoint file to resume downloads from the last processed point.
        _update_checkpoint: Updates the checkpoint file with the current download progress.
        _remove_checkpoint: Deletes the checkpoint file, indicating that the download process has completed successfully.
        _filter_buckets: Filters the list of buckets based on the whitelist and blacklist.
        _filter_directories: Filters the list of directories based on the whitelist and blacklist.
        _get_download_path: Generates the download path for a video, based on the maintain_structure setting.
        _should_process_directory: Determines whether a given directory should be processed, based on checkpoint data.
        _download_videos: Downloads videos from a list, calls the post-download callback, and optionally deletes the videos.
        download_videos_ordered: Downloads videos in an ordered manner, processing directories as specified.
        download_videos_random: Downloads a random set of videos from each directory, up to a specified batch size.

    Example:
        To use VideoDownloader with a custom callback function that logs the path of each downloaded video:

        >>> def log_video_path(video_path):
        ...     print(f"Video downloaded to: {video_path}")
        ...
        >>> downloader = VideoDownloader(
        ...     manipulator=S3Manipulator(),
        ...     checkpoint_file="path/to/checkpoint.json",
        ...     parent_directory="/path/to/download/directory",
        ...     maintain_structure=True,
        ...     delete_after_process=False,
        ...     post_download_callback=log_video_path
        ... )
        >>> downloader.download_videos_ordered(batch_size=5)
    """

    def __init__(self,
                 manipulator: S3Manipulator,
                 checkpoint_file=os.path.join(CHECKPOINTS_DIR, 'video_downloader_checkpoint.json'),
                 whitelist_buckets=None,
                 blacklist_buckets=None,
                 whitelist_directories=None,
                 blacklist_directories=None,
                 parent_directory=DOWNLOADS_DIR,
                 maintain_structure=True,
                 delete_after_process=False,
                 processing_callback=None):

        self.manipulator = manipulator
        self.checkpoint_file = checkpoint_file
        self.whitelist_buckets = whitelist_buckets if whitelist_buckets is not None else []
        self.blacklist_buckets = blacklist_buckets if blacklist_buckets is not None else []
        self.whitelist_directories = whitelist_directories if whitelist_directories is not None else []
        self.blacklist_directories = blacklist_directories if blacklist_directories is not None else []
        self.parent_directory = parent_directory
        self.maintain_structure = maintain_structure
        self.delete_after_process = delete_after_process
        self.processing_callback = processing_callback

    def _read_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"last_bucket": "", "last_directory": "", "downloaded_videos": []}  # Default structure

    def _update_checkpoint(self, last_bucket, last_directory, downloaded_videos=[]):
        checkpoint_data = {
            "last_bucket": last_bucket,
            "last_directory": last_directory,
            "downloaded_videos": downloaded_videos
        }
        with open(self.checkpoint_file, 'w') as file:
            json.dump(checkpoint_data, file, indent=4)

    def _remove_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logging.info("Process completed successfully. Checkpoint file removed.")

    def _filter_buckets(self, buckets):
        if self.whitelist_buckets:
            buckets = [bucket for bucket in buckets if bucket in self.whitelist_buckets]
        if self.blacklist_buckets:
            buckets = [bucket for bucket in buckets if bucket not in self.blacklist_buckets]
        return buckets

    def _filter_directories(self, directories):
        if self.whitelist_directories:
            directories = [directory for directory in directories if directory in self.whitelist_directories]
        if self.blacklist_directories:
            directories = [directory for directory in directories if directory not in self.blacklist_directories]
        return directories

    def _get_download_path(self, bucket, directory):
        if self.maintain_structure:
            # Create a structured path: parent_directory/bucket/directory
            download_path = os.path.join(self.parent_directory, bucket, directory)
            Manipulator.create_folders(download_path)  # Assuming this creates the path if it doesn't exist
        else:
            # Use the parent directory as the download path
            download_path = self.parent_directory
        return download_path

    def _should_process_directory(self, checkpoint, current_bucket, current_directory):
        # If there's no checkpoint, we're starting fresh, so process everything
        if not checkpoint:
            return True

        last_bucket = checkpoint.get("last_bucket", "")
        last_directory = checkpoint.get("last_directory", "")

        # If we're in the same bucket and at or past the last processed directory, start/resume processing
        if current_bucket == last_bucket and current_directory >= last_directory:
            return True
        # If we're past the last processed bucket, start/resume processing
        elif current_bucket > last_bucket:
            return True
        # Otherwise, skip this directory
        else:
            return False

    def _download_videos(self, bucket, directory, videos_to_download):
        downloaded_videos = []
        for video_path in videos_to_download:
            local_file_name = os.path.basename(video_path)
            download_path = self._get_download_path(bucket, directory)
            destination_path = os.path.join(download_path, local_file_name)

            logging.info(f"Downloading video: {video_path} to {destination_path}")

            # Download logic
            self.manipulator.download_file_s3(bucket_name=bucket, file_name=self.manipulator._parse_s3_path(video_path)[1],
                                              local_file_name=destination_path)

            callback_result = None
            if self.processing_callback:
                try:
                    callback_result = self.processing_callback(destination_path)
                except Exception as e:
                    logging.error(f"Error during post-download callback for {destination_path}: {str(e)}")

            if self.delete_after_process:
                logging.info(f"Deleting video after processing: {destination_path}")
                Manipulator.delete_file(destination_path)

            yield destination_path, callback_result

    def download_videos_ordered(self):
        checkpoint = self._read_checkpoint()
        buckets = self._filter_buckets(self.manipulator.list_buckets_s3())
        logging.info(f"Processing buckets: {buckets}")
        for bucket in buckets:
            directories = self._filter_directories(self.manipulator.list_directories_s3(bucket_name=bucket))
            for directory in directories:
                if self._should_process_directory(checkpoint, bucket, directory):
                    logging.info(f"Processing directory: {directory} in bucket: {bucket}")
                    video_files = self.manipulator.list_files_s3(bucket_name=bucket, folder_name=directory,
                                                                 regex=r'\.(mp4|avi)$', return_full_path=True)
                    # Filter out already downloaded videos if resuming
                    videos_to_download = [v for v in video_files if
                                          os.path.basename(v) not in checkpoint.get("downloaded_videos", [])]

                    for destination_path, callback_result in self._download_videos(bucket, directory,
                                                                                   videos_to_download):
                        logging.info(f"Processed {destination_path} with callback result: {callback_result}")
                        # Assuming the callback doesn't change the video file name, so we use the basename for the checkpoint.
                        checkpoint["downloaded_videos"].append(os.path.basename(destination_path))
                        self._update_checkpoint(bucket, directory, checkpoint["downloaded_videos"])

                        yield (destination_path, callback_result)

        self._remove_checkpoint()

    def download_videos_random(self, sample_size: int = 5):
        checkpoint = self._read_checkpoint()
        buckets = self._filter_buckets(self.manipulator.list_buckets_s3())
        logging.info(f"Processing buckets: {buckets}")
        for bucket in buckets:
            directories = self._filter_directories(self.manipulator.list_directories_s3(bucket_name=bucket))
            for directory in directories:
                if self._should_process_directory(checkpoint, bucket, directory):
                    logging.info(f"Processing directory: {directory} in bucket: {bucket}")
                    video_files = self.manipulator.list_files_s3(bucket_name=bucket, folder_name=directory,
                                                                 regex=r'\.(mp4|avi)$', return_full_path=True)
                    # print(video_files)
                    available_videos = [v for v in video_files if
                                        os.path.basename(v) not in checkpoint.get("downloaded_videos", [])]
                    if len(available_videos) < sample_size:
                        logging.warning(f"Not enough new videos in {directory}, found only {len(available_videos)}")
                        continue
                    selected_videos = random.sample(available_videos, sample_size)
                    for destination_path, callback_result in self._download_videos(bucket, directory, selected_videos):
                        logging.info(
                            f"Processed {destination_path} with callback result: {None if callback_result is None else 'OK'}")
                        # Update the checkpoint similarly as in ordered download
                        checkpoint["downloaded_videos"].append(os.path.basename(destination_path))
                        self._update_checkpoint(bucket, directory, checkpoint["downloaded_videos"])

                        yield (destination_path, callback_result)
        self._remove_checkpoint()