from detectflow.config import TESTS_DIR
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.process.video_downloader import VideoDownloader
from detectflow.callbacks.video_downloader import diagnose_video
from detectflow.manipulators.input_manipulator import InputManipulator
import os
from functools import partial
import logging


if __name__ == "__main__":

    # Set your parameters
    output_path = os.path.join(TESTS_DIR, "temp")
    regex = r"^(?!\.)[^.]*\.(mp4|avi)$" # not starting with dot
    batch_size = 1

    # Callback parameters
    frame_skip = 100
    motion_method = 'SOM'

    # Partially define the callback function
    callback = partial(diagnose_video, output_path=output_path, frame_skip=frame_skip, motion_method=motion_method)

    # Execution logic
    os.makedirs(output_path, exist_ok=True)

    # Get the s3 manipulator
    s3_manipulator = S3Manipulator()

    excel_files = s3_manipulator.list_files_s3(bucket_name='excels', folder_name='', regex=r".*\.xlsx$", return_full_path=False)

    # Specify the buckets and directories to whitelist
    buckets, directories = [], []
    for f in excel_files:
        # Get the bucket and validate
        bucket = InputManipulator.get_bucket_name_from_id(os.path.splitext(os.path.basename(f))[0])
        if not s3_manipulator.is_s3_bucket(bucket):
            logging.info(f"Bucket {bucket} does not exist.")
        else:
            buckets.append(bucket)

        # Get the directory and validate
        directory = f"{InputManipulator.zero_pad_id(os.path.splitext(os.path.basename(f))[0])}/"
        if not s3_manipulator.is_s3_directory(f"{bucket}/{directory}"):
            logging.info(f"Directory {directory} does not exist in bucket {bucket}.")
        else:
            directories.append(directory)

    # Initialize the downloader
    downloader = VideoDownloader(manipulator=s3_manipulator,
                                 checkpoint_file=os.path.join(output_path, "checkpoint.json"),
                                 whitelist_buckets=buckets,
                                 whitelist_directories=directories,
                                 delete_after_process=True,
                                 processing_callback=callback)

    for _, results in downloader.download_videos_ordered(regex=regex, batch_size=batch_size, parallelism=False):
        if results:
            pass  # Do something with the results