import boto3
import botocore
import configparser
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union
import time
from detectflow.validators.s3_validator import S3Validator
from detectflow.utils.s3.cfg import parse_s3_config


class S3Manipulator(S3Validator):
    def __init__(self, cfg_file: str = "detectflow/config/.s3.cfg"): # TODO: Add find file automatic location of the config file

        # Run the init method of S3Validator parent class
        S3Validator.__init__(self, cfg_file)

        self.endpoint_url, self.aws_access_key_id, self.aws_secret_access_key = parse_s3_config(cfg_file)
        region_name = 'eu-west-2'

        # Initialize the S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=region_name
        )

    def list_files_s3(self, bucket_name: str, folder_name: str, regex: str = None, return_full_path: bool = True) -> \
    List[str]:
        """
        Lists files in a specified bucket and folder on S3, optionally filtered by a regex pattern.

        :param bucket_name: Name of the S3 bucket.
        :param folder_name: Name of the folder within the bucket.
        :param regex: Optional regex pattern to filter file names.
        :param return_full_path: Return full S3 paths if True, else just file names.
        :return: Sorted list of file keys or paths in the specified folder, optionally filtered by regex.
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
            if 'Contents' in response:
                pattern = re.compile(regex) if regex else None
                files = [item['Key'] for item in response['Contents'] if
                         not pattern or pattern.search(os.path.basename(item['Key']))]

                # Sorting files alphabetically
                files.sort()

                if return_full_path:
                    return [f"s3://{bucket_name}/{file}" for file in files]
                else:
                    return [os.path.basename(file) for file in files]
            else:
                logging.info(f"No contents found in {bucket_name}/{folder_name}")
                return []
        except re.error as e:
            logging.error(f"Regex error: {e}")
            return []
        except botocore.exceptions.NoCredentialsError:
            logging.error("No credentials provided for AWS S3 access.")
            return []
        except botocore.exceptions.ClientError as error:
            error_code = error.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logging.error(f"Bucket does not exist: {bucket_name}")
            else:
                logging.error(f"Client error occurred: {error}")
            return []
        except Exception as e:
            logging.error(f"An error occurred while listing files in {bucket_name}/{folder_name}: {e}")
            return []

    def download_file_s3(self, bucket_name, file_name, local_file_name, max_attempts=3, delay=2):
        """
        Download a file with retry logic.

        :param bucket_name: Name of the S3 bucket.
        :param file_name: Name of the file in the S3 bucket.
        :param local_file_name: Path where the file will be saved locally.
        :param max_attempts: Maximum number of retry attempts.
        :param delay: Delay between retries in seconds.
        :return: Path to the downloaded file, or None if download fails.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                self.s3_client.download_file(bucket_name, file_name, local_file_name)
                return local_file_name  # Return the local file path upon successful download
            except botocore.exceptions.ClientError as error:
                logging.error(f"Failed to download {file_name}: {error}")
                attempt += 1
                time.sleep(delay)
                if attempt == max_attempts:
                    raise
        return None  # Return None if all attempts fail

    def download_files_s3_batch(self, bucket_name, file_pairs, max_workers=5, max_attempts=3):
        """
        Download multiple files in parallel with enhanced error handling.

        :param bucket_name: S3 bucket name.
        :param file_pairs: List of tuples, where each tuple contains (source_key, destination_path).
        :param max_workers: Maximum number of parallel download threads.
        :param max_attempts: Maximum number of attempts for each download.
        :return: List of paths to successfully downloaded files.
        """
        successful_downloads = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.download_file_s3, bucket_name, source, target, max_attempts): (source, target) for
                source, target in file_pairs}

            for future in as_completed(future_to_file):
                source, target = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful_downloads.append(target)
                    else:
                        logging.error(f"Failed to download {source} after {max_attempts} attempts.")
                except Exception as e:
                    logging.error(f"Error downloading {source} to {target}: {e}")

        return successful_downloads

    def list_buckets_s3(self, regex: str = None) -> List[str]:
        """
        List all S3 buckets, optionally filtered by a regex pattern.

        :param regex: Optional regex pattern to filter bucket names.
        :return: Sorted list of bucket names, optionally filtered by regex.
        """
        try:
            response = self.s3_client.list_buckets()
            pattern = re.compile(regex) if regex else None
            buckets = [bucket['Name'] for bucket in response.get('Buckets', []) if
                       not pattern or pattern.search(bucket['Name'])]

            # Sorting bucket names alphabetically
            buckets.sort()
            return buckets
        except re.error as e:
            logging.error(f"Regex error: {e}")
            return []
        except botocore.exceptions.NoCredentialsError:
            logging.error("No credentials provided for AWS S3 access.")
            return []
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while listing buckets: {error}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error occurred while listing buckets: {e}")
            return []

    def list_directories_s3(self, bucket_name: str, prefix: str = '', regex: str = None, full_path: bool = False) -> \
    List[str]:
        """
        Lists all the directories within a specified bucket and prefix on an S3 service,
        optionally filtered by a regex pattern and can return full paths including the bucket name.

        Parameters:
            bucket_name (str): The name of the S3 bucket.
            prefix (str, optional): The prefix to filter directories. Defaults to an empty string.
            regex (str, optional): Optional regex pattern to filter directory names.
            full_path (bool, optional): If set to True, returns full S3 paths including the bucket name.

        Returns:
            List[str]: A sorted list of directory paths within the specified bucket and prefix,
                       optionally filtered by regex and can include full S3 path.

        Example:
            >>> directories = list_directories_s3('my-bucket', 'my/prefix/', regex='^pattern.*', full_path=True)
            >>> print(directories)
            ['s3://my-bucket/my/prefix/pattern-dir1/', 's3://my-bucket/my/prefix/pattern-dir2/', ...]
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
            pattern = re.compile(regex) if regex else None

            directories = []
            if 'CommonPrefixes' in response:
                for cp in response['CommonPrefixes']:
                    directory_path = cp['Prefix']
                    if full_path:
                        directory_path = f"s3://{bucket_name}/{directory_path}"  # Format full S3 path
                    if not pattern or pattern.search(directory_path):
                        directories.append(directory_path)

            # Sorting directories alphabetically
            directories.sort()
            return directories
        except re.error as e:
            logging.error(f"Regex error: {e}")
            return []
        except botocore.exceptions.Boto3Error as error:
            logging.error(f"Error accessing S3: {error}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error while listing directories in S3: {e}")
            return []

    def download_directory_s3(self, bucket_name: str, folder_name: str, local_dir: str,
                              use_batch_download: bool = False, max_workers: int = 5):
        """
        Downloads an entire directory from an S3 bucket to a local directory.

        Parameters:
            bucket_name (str): The name of the S3 bucket.
            folder_name (str): The folder within the S3 bucket.
            local_dir (str): Local directory to save files.
            use_batch_download (bool): Use batch download if True, else download files one by one.
            max_workers (int): Number of parallel threads for batch download.

        Returns:
            List[str]: List of paths to successfully downloaded files.

        Raises:
            Exception: If there is an issue with downloading the files.
        """
        successful_downloads = []
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name, Delimiter='/')

            if 'Contents' in response:
                file_pairs = []
                for item in response['Contents']:
                    file_name = item['Key']
                    if not file_name.endswith('/'):  # Ignore directory placeholders
                        local_file_path = os.path.join(local_dir, os.path.relpath(file_name, folder_name))
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        file_pairs.append((file_name, local_file_path))

                if use_batch_download:
                    # Downloading files in batch
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(self.download_file_s3, bucket_name, source, target): (source, target)
                                   for source, target in file_pairs}
                        for future in as_completed(futures):
                            source, target = futures[future]
                            try:
                                future.result()
                                successful_downloads.append(target)
                            except Exception as e:
                                logging.error(f"Error downloading {source} to {target}: {e}")
                else:
                    # Downloading files one by one
                    for source, target in file_pairs:
                        try:
                            download_result = self.download_file_s3(bucket_name, source, target)
                            if download_result:  # Check if the download was successful
                                successful_downloads.append(target)
                        except Exception as e:
                            logging.error(f"Error downloading {source} to {target}: {e}")
                            raise  # Re-raise the exception to notify the caller

            return successful_downloads
        except Exception as e:
            logging.error(f"Error occurred while downloading directory from S3: {e}")
            raise  # Re-raise the exception to notify the caller

    def find_files_s3(self, file_name: str, bucket_name: str = None) -> List[str]:
        """
        Search for a file with a specific name in S3 storage.

        :param file_name: The name of the file to search for, including extension.
        :param bucket_name: Optional specific bucket to search in. If None, searches all buckets.
        :return: List of S3 paths to files with the specified name.
        """
        found_files = []
        try:
            if bucket_name:
                buckets = [bucket_name]
            else:
                buckets = [bucket['Name'] for bucket in self.s3_client.list_buckets()['Buckets']]

            for bucket in buckets:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket):
                    if "Contents" in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith(file_name):
                                found_files.append(f"s3://{bucket}/{obj['Key']}")

            return found_files
        except boto3.exceptions.Boto3Error as error:
            logging.error(f"Error accessing S3: {error}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error while searching for files in S3: {e}")
            return []


    def check_file_exists_s3(self, bucket_name, file_name):
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=file_name)
            return True
        except botocore.exceptions.ClientError:
            return False

    def create_directory_s3(self, bucket_name: str, directory_path: str):
        """
        Create a directory (placeholder object) in an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :param directory_path: Path of the directory to create.
        """
        if not directory_path.endswith('/'):
            directory_path += '/'
        try:
            if not self.is_s3_directory(f"s3://{bucket_name}/{directory_path}"):
                # Creating a placeholder object for the directory
                self.s3_client.put_object(Bucket=bucket_name, Key=directory_path)
                logging.info(f"Directory '{directory_path}' created in bucket '{bucket_name}'.")
            else:
                logging.info(f"Directory '{directory_path}' already exists in bucket '{bucket_name}'.")
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while creating directory in S3: {error}")
            raise


    def create_bucket_s3(self, bucket_name: str):
        """
        Create a bucket (with directory as a placeholder object) in an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        """
        # Check if the bucket exists
        if not self.is_s3_bucket(bucket_name):
            logging.info(f"Bucket '{bucket_name}' does not exist, creating it.")
            self.create_directory_s3(bucket_name, '')  # Create the bucket as a directory placeholder
        else:
            logging.info(f"Bucket '{bucket_name}' already exists.")


    def upload_file_s3(self, bucket_name: str, file_path: str, s3_path: str, max_attempts=3, delay=2):
        """
        Upload a file to a specified path in an S3 bucket with retry mechanism.

        :param bucket_name: Name of the S3 bucket.
        :param file_path: Path to the file to upload.
        :param s3_path: Path in the S3 bucket to upload the file to.
        :param max_attempts: Maximum number of retry attempts.
        :param delay: Delay between retries in seconds.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                self.s3_client.upload_file(file_path, bucket_name, s3_path)
                logging.info(f"File '{file_path}' uploaded to '{s3_path}' in bucket '{bucket_name}'.")
                return  # Upload successful
            except botocore.exceptions.ClientError as error:
                logging.error(f"Attempt {attempt + 1} failed to upload file '{file_path}' to S3: {error}")
                attempt += 1
                if attempt < max_attempts:
                    time.sleep(delay)
                else:
                    raise  # Re-raise the exception after final attempt


    def upload_directory_s3(self, local_directory: str, bucket_name: str, s3_path: str, max_attempts=3, delay=2):
        """
        Uploads a local directory (including its subdirectories) to an S3 bucket using the comprehensive upload_file_s3 method.

        :param local_directory: Local directory to upload.
        :param bucket_name: Name of the S3 bucket.
        :param s3_path: S3 path where the directory will be uploaded.
        :param max_attempts: Maximum number of retry attempts for each file.
        :param delay: Delay between retries in seconds for each file.
        """
        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_file_path = os.path.join(s3_path, relative_path).replace('\\', '/')
                try:
                    self.upload_file_s3(bucket_name, local_path, s3_file_path, max_attempts, delay)
                except Exception as error:
                    logging.error(f"Error occurred while uploading '{local_path}' to '{s3_file_path}': {error}")
                    raise  # Optional: Decide whether to stop the entire process or continue with other files


    def delete_directory_s3(self, bucket_name: str, s3_directory_path: str):
        """
        Deletes a directory and its contents from an S3 bucket using delete_file_s3 method.

        :param bucket_name: Name of the S3 bucket.
        :param s3_directory_path: S3 directory path to delete.
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_directory_path):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        self.delete_file_s3(bucket_name, obj['Key'])
                    logging.info(f"Objects under '{s3_directory_path}' deleted from bucket '{bucket_name}'.")
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while deleting directory from S3: {error}")
            raise


    def delete_file_s3(self, bucket_name: str, file_path: str):
        """
        Delete a file from an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :param file_path: Path of the file in the S3 bucket to delete.
        """
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=file_path)
            logging.info(f"File '{file_path}' deleted from bucket '{bucket_name}'.")
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while deleting file from S3: {error}")
            raise


    def copy_file_s3(self, source_bucket_name: str, source_file_path: str, dest_bucket_name: str, dest_file_path: str):
        """
        Copy a file within S3 from one location to another.

        :param source_bucket_name: Name of the source S3 bucket.
        :param source_file_path: Path of the file in the source S3 bucket.
        :param dest_bucket_name: Name of the destination S3 bucket.
        :param dest_file_path: Path in the destination S3 bucket where the file will be copied to.
        """
        copy_source = {'Bucket': source_bucket_name, 'Key': source_file_path}
        try:
            self.s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket_name, Key=dest_file_path)
            logging.info(
                f"File '{source_file_path}' from bucket '{source_bucket_name}' copied to '{dest_file_path}' in bucket '{dest_bucket_name}'.")
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while copying file in S3: {error}")
            raise


    def generate_presigned_url_s3(self, bucket_name: str, object_key: str, expiration=3600):
        """
        Generate a presigned URL for an S3 object.

        :param bucket_name: Name of the S3 bucket.
        :param object_key: Key of the object in S3 for which to generate the URL.
        :param expiration: Time in seconds for the presigned URL to remain valid (default 1 hour).
        :return: Presigned URL as a string, or None in case of failure.
        """
        try:
            url = self.s3_client.generate_presigned_url('get_object',
                                                        Params={'Bucket': bucket_name, 'Key': object_key},
                                                        ExpiresIn=expiration)
            return url
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error generating presigned URL for '{object_key}': {error}")
            return None


    def sync_directory_s3(self, local_directory: str, bucket_name: str, s3_path: str, download_missing=False,
                          sync_deletion=False):
        """
        Synchronize a local directory with an S3 path using file modification times.

        :param local_directory: Local directory to synchronize.
        :param bucket_name: Name of the S3 bucket.
        :param s3_path: Path in the S3 bucket to synchronize with.
        :param download_missing: If True, download files missing in the local directory from S3.
        :param sync_deletion: If True, delete files in S3 that are not present in the local directory.
        """
        try:
            local_files = set()
            for root, dirs, files in os.walk(local_directory):
                for filename in files:
                    local_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(local_path, local_directory)
                    s3_file_path = os.path.join(s3_path, relative_path).replace('\\', '/')
                    local_files.add(s3_file_path)

                    if not self.check_file_exists_s3(bucket_name, s3_file_path) or self._is_file_modified(local_path,
                                                                                                          bucket_name,
                                                                                                          s3_file_path):
                        self.upload_file_s3(bucket_name, local_path, s3_file_path)
                        logging.info(f"Synchronized '{local_path}' to '{s3_file_path}' in bucket '{bucket_name}'.")

            if download_missing or sync_deletion:
                s3_files = self.list_files_s3(bucket_name, s3_path)
                for s3_file in s3_files:
                    local_file = os.path.join(local_directory, os.path.relpath(s3_file, s3_path))
                    if download_missing and s3_file not in local_files:
                        self.download_file_s3(bucket_name, s3_file, local_file)
                    elif sync_deletion and s3_file not in local_files:
                        self.delete_file_s3(bucket_name, s3_file)

        except Exception as e:
            logging.error(f"Error occurred while syncing directory to S3: {e}")
            raise

    def _is_file_modified(self, local_path: str, bucket_name: str, s3_file_path: str): # DONE: Fix the tzutc reference
        """
        Check if a local file is modified compared to its version in S3.

        :param local_path: Local file path.
        :param bucket_name: Name of the S3 bucket.
        :param s3_file_path: Path of the file in S3.
        :return: True if local file is modified, False otherwise.
        """
        from dateutil.tz import tzutc

        try:
            local_modified_time = os.path.getmtime(local_path)
            response = self.s3_client.head_object(Bucket=bucket_name, Key=s3_file_path)
            s3_modified_time = response['LastModified'].replace(tzinfo=tzutc()).timestamp()

            return local_modified_time > s3_modified_time
        except botocore.exceptions.ClientError as error:
            logging.error(f"Error occurred while checking file modification: {error}")
            return True  # Assume modified in case of error