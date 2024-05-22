import logging
import os
import re
import shutil
from typing import List, Optional, Tuple, Union
from detectflow.validators.validator import Validator


class Manipulator(Validator):
    def __init__(self):
        Validator.__init__(self)

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def list_files(path: str, regex: str = None, extensions: Tuple[str, ...] = ('.mp4', '.avi'),
                   return_full_path: bool = True) -> List[str]:
        """
        List files in a specified local directory with given file extensions, optionally filtered by a regex pattern.

        :param path: Path to the local directory.
        :param regex: Optional regex pattern to filter file names.
        :param extensions: A tuple of file extensions to include.
        :param return_full_path: Return full paths if True, else just file names.
        :return: Ordered list of file paths or names with specified extensions and optional regex pattern.
        """
        try:
            pattern = re.compile(regex) if regex else None
            files = [f for f in sorted(os.listdir(path))
                     if os.path.isfile(os.path.join(path, f))
                     and f.lower().endswith(extensions)
                     and (not pattern or pattern.match(f))]

            if return_full_path:
                return [os.path.join(path, f) for f in files]
            else:
                return files
        except FileNotFoundError:
            logging.error(f"Directory not found: {path}")
            return []
        except re.error as e:
            logging.error(f"Regex error: {e}")
            return []
        except OSError as error:
            logging.error(f"Error accessing directory {path}: {error}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error while listing files in {path}: {e}")
            return []

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def list_folders(directory: str, regex: str = None, return_full_path: bool = True) -> Tuple[str, ...]:
        """
        List all subdirectories in a given directory, optionally filtering by a regex pattern, and
        either return full paths or just folder names.

        Parameters:
            directory (str): Path to the directory.
            regex (str, optional): Optional regex pattern to filter folder names.
                                   Regular expressions can be used to match folder names.
                                   Example:
                                     - '^pattern' matches any folder name starting with 'pattern'
                                     - 'pattern$' matches any folder name ending with 'pattern'
                                     - '.*pattern.*' matches any folder name containing 'pattern'
            return_full_path (bool): Return full paths if True, else just folder names.

        Returns:
            Tuple[str, ...]: Tuple of sorted subdirectory paths or names, depending on return_full_path.

        Raises:
            FileNotFoundError: If the specified directory is not found.
            PermissionError: If there is a permission error accessing the directory.
            OSError: For other OS-related errors.
            re.error: If there is an error in the provided regex pattern.
        """
        try:
            entries = os.listdir(directory)
            if regex:
                pattern = re.compile(regex)
                folders = sorted(entry for entry in entries if
                                 os.path.isdir(os.path.join(directory, entry)) and pattern.match(entry))
            else:
                folders = sorted(entry for entry in entries if os.path.isdir(os.path.join(directory, entry)))

            if return_full_path:
                return tuple(os.path.join(directory, folder) for folder in folders)
            else:
                return tuple(folders)
        except FileNotFoundError:
            logging.error(f"Directory not found: {directory}")
            return ()
        except PermissionError:
            logging.error(f"Permission denied when accessing directory: {directory}")
            return ()
        except OSError as error:
            logging.error(f"Error accessing directory {directory}: {error}")
            return ()
        except re.error as e:
            logging.error(f"Regex error: {e}")
            return ()
        except Exception as e:
            logging.error(f"Unexpected error while listing folders in {directory}: {e}")
            return ()

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def create_folders(directories: Union[str, List[str]], parent_dir: str = '') -> List[str]:
        """
        Create directories based on a list of folder names or paths. If a directory already exists, it is skipped.

        :param directories: A single folder name, a partial path, or a list/tuple of names/paths.
        :param parent_dir: The parent directory under which the folders will be created.
        :return: List of paths to successfully created directories.
        """
        successful_creations = []
        if isinstance(directories, str):
            directories = [directories]

        for directory in directories:
            full_path = os.path.join(parent_dir, directory)
            try:
                os.makedirs(full_path, exist_ok=True)
                successful_creations.append(full_path)
            except OSError as error:
                logging.error(f"Error creating directory {full_path}: {error}")

        return successful_creations

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def move_file(source_path: str, dest_path: str, filename: Optional[str] = None, overwrite: bool = False,
                  copy: bool = False):
        """
        Move or copy a file from the source path to the destination path.

        :param source_path: Path to the source file.
        :param dest_path: Path to the destination directory.
        :param filename: Optional new filename for the video in the destination directory.
        :param overwrite: Whether to overwrite the file if it exists in the destination.
        :param copy: Set to True to copy the file instead of moving.
        :return: Path of the moved or copied file in the destination directory or None if the operation fails.
        """
        try:
            # Check if the source file exists
            if not os.path.isfile(source_path):
                logging.error(f"Source file does not exist: {source_path}")
                return None

            # Determine the destination file path
            dest_file_path = os.path.join(dest_path, filename if filename else os.path.basename(source_path))

            # Check for overwrite condition
            if os.path.exists(dest_file_path) and not overwrite:
                logging.error(f"Destination file already exists and overwrite is False: {dest_file_path}")
                return None

            # Create the destination directory if it doesn't exist
            os.makedirs(dest_path, exist_ok=True)

            # Move or copy the file based on the copy flag
            if copy:
                shutil.copy2(source_path, dest_file_path)  # Use copy2 to preserve metadata
            else:
                shutil.move(source_path, dest_file_path)

            return dest_file_path
        except PermissionError:
            logging.error(f"Permission denied when moving/copying file from {source_path} to {dest_path}")
        except OSError as error:
            logging.error(f"Error moving/copying file from {source_path} to {dest_path}: {error}")
        except Exception as e:
            logging.error(f"Unexpected error while moving/copying file: {e}")

        return None

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def delete_file(file_path):
        """Delete a local file."""
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                raise FileNotFoundError()
        except OSError as error:
            logging.error(f"Error deleting file {file_path}: {error}")
        except FileNotFoundError:
            logging.error(f"File does not exist {file_path}")

    # @profile_function_call(logging.getLogger(__name__))
    @staticmethod
    def find_files(root_path: str, file_name: str, sort_by: str = 'modification') -> List[str]:
        """
        Search for files with a specific name within the filesystem starting from a root directory and sort them.

        :param root_path: The root directory to start the search from.
        :param file_name: The name of the file to search for, including extension.
        :param sort_by: Criterion to sort the found files ('modification', 'creation', 'size', 'name').
        :return: List of paths to files with the specified name, sorted based on the criterion.
        """
        matching_files = []
        try:
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if file == file_name:
                        full_path = os.path.join(root, file)
                        matching_files.append(full_path)

            # Sorting logic based on the chosen criterion
            if sort_by == 'modification':
                matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            elif sort_by == 'creation':
                matching_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
            elif sort_by == 'size':
                matching_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            elif sort_by == 'name':
                matching_files.sort(key=lambda x: os.path.basename(x))
            else:
                logging.warning(f"Unknown sorting criterion: {sort_by}. Files will be returned unsorted.")

            return matching_files
        except FileNotFoundError:
            logging.error(f"Root directory not found: {root_path}")
            return []
        except OSError as error:
            logging.error(f"Error accessing directory {root_path}: {error}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error while searching for files in {root_path}: {e}")
            return []
