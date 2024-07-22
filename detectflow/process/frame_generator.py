from typing import List, Tuple, Union, Optional, Dict
import logging
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
import pandas as pd
from detectflow.manipulators.video_manipulator import VideoManipulator
from detectflow.video.video_data import Video
from detectflow.validators.validator import Validator
import sqlite3
import traceback
#TODO: Add imports for video classes and functions dealing with the excel fiel and dataframe


def read_dataframe_from_db(
        folder_path: str,
        db_name: str,
        table_name: str = "metadata"
):
    """
    Reads a DataFrame from a SQLite database table.

    Args:
    folder_path (str): The directory where the SQLite database is located.
    db_name (str): The name of the SQLite database file.
    table_name (str): The name of the table to read the data from. Defaults to 'metadata'.

    Returns:
    pd.DataFrame: The DataFrame read from the specified database table.

    This function creates a connection to a SQLite database located at the specified folder path and
    reads data from the specified table into a pandas DataFrame. The function closes the database connection
    after reading the data.
    """

    # Create a connection object
    conn = sqlite3.connect(os.path.join(folder_path, db_name))

    # Using the connection, the SQL table is read into a dataframe
    dataframe = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    # Close the connection
    conn.close()

    return dataframe


class FrameGeneratorTask:
    def __init__(self, frames_array, metadata):
        self.frames_array = frames_array
        self.metadata = metadata

    def get_meta(self, key):
        return self.metadata.get(key)


class FrameGenerator:
    # Define the type for a single ROI, which is a list of four integers, and the dict or list of ROIs
    ROI = List[int]
    ROIsListFormat = List[List[Union[ROI, str]]]
    ROIsDictFormat = Dict[str, List[ROI]]
    ROIsArgumentType = Optional[Union[ROIsListFormat, ROIsDictFormat]]  # rois argument type - list or dict

    def __init__(self,
                 source: Optional[Union[str, List]],
                 excel_path: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 first_frame_number: Optional[Union[int, Dict]] = None,  # TODO: Could be a dict
                 frames_per_visit: int = 20,
                 frame_skip: int = 15,
                 queue_size: int = 2,
                 rois: ROIsArgumentType = None,
                 crop_size: int = 640,
                 offset_range: int = 100,
                 name_prefix: str = "",
                 processing_callback=None,
                 **kwargs):
        """
        Initialize the FrameGenerator with video file paths and optional Excel path for annotations.
        """
        try:
            self.dataframe = self._get_dataframe(excel_path) if excel_path is not None else None
            self.annotation_data_array = VideoManipulator.get_annotation_data_array() if self.dataframe is not None else None
            self.video_paths = self._get_source(source) if source is not None else None
            self.output_folder = output_folder or os.getcwd()
            self.first_frame_number = first_frame_number if first_frame_number is not None else {
                os.path.basename(video): 0 for video in self.video_paths}
            self.frames_per_visit = frames_per_visit
            self.frame_skip = frame_skip
            self.queue_size = queue_size
            self.crop_size = crop_size
            self.offset_range = offset_range
            self.name_prefix = name_prefix
            self.output_folder = output_folder
            self.processing_callback = processing_callback
            self.config = kwargs
            self.frame_batch_size = None
        except FileNotFoundError as e:
            logging.error(f"Initialization failed due to a missing file: {e}")
            # Consider halting initialization or setting defaults
            raise
        except ValueError as e:
            logging.error(f"Initialization failed due to a value error: {e}")
            # Consider halting initialization or setting defaults
            raise
        except Exception as e:
            logging.error(f"Initialization failed due to an unexpected error: {e}")
            # Consider halting initialization or setting defaults
            raise

        # Validate and convert the rois argument
        try:
            if rois is not None:
                self.rois = self._validate_and_convert_rois(rois)
        except (ValueError, TypeError) as e:
            logging.error(f"Error in ROIs argument: {e}")
            self.rois = None

    @Validator.validate_paths
    def _get_source(self, source):
        videos = None
        # If source is a directory path
        if isinstance(source, str):
            if os.path.isdir(source):

                # Assign value as video folder path
                video_folder_path = source

                # Load videos from folder
                videos = VideoManipulator.load_videos(video_folder_path)

            else:
                raise FileNotFoundError(f"The directory '{source}' does not exist.")
        else:
            # if videos are directly a list of videos we will use that directly
            if isinstance(source, list):
                videos = source
            else:
                raise TypeError("Expected a path to video folder or a list of video paths, got a different type.")

        # If the annotation_data_array attribute exists, therefore if excel was supplied and successfully loaded
        if hasattr(self, "annotation_data_array") and getattr(self, "annotation_data_array") is not None:
            # Get only relevant videos
            videos = VideoManipulator.get_relevant_video_paths(videos, self.annotation_data_array)

        return videos

    @Validator.validate_dataframe
    def _get_dataframe(self, excel_path):
        dataframe = None
        if excel_path is not None:
            # Load excel from file and exctract data
            try:
                dataframe = read_dataframe_from_db(os.path.dirname(excel_path),
                                                   f"{os.path.basename(excel_path)[:-5]}.db")
            except Exception as e:
                logging.error(f"Exception when reading .db: {e}")
                # excel = anno_data.Annotation_watcher_file(excel_path, True, True, True, True)
                # dataframe = excel.dataframe
                dataframe = None

        return dataframe

    def _validate_and_convert_rois(self, rois):
        converted_rois = {}

        if isinstance(rois, list):
            for item in rois:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("Each item in the list should be a list containing ROIs and a video filepath.")

                rois_list, video_filepath = item

                if not all(isinstance(roi, list) and len(roi) == 4 for roi in rois_list):
                    raise ValueError("Each ROI should be a list of four coordinates.")

                if not isinstance(video_filepath, str):
                    raise ValueError("The video filepath should be a string.")

                video_basename = os.path.basename(video_filepath)
                converted_rois[video_basename] = rois_list

        elif isinstance(rois, dict):
            for video_basename, rois_list in rois.items():
                if not isinstance(video_basename, str):
                    raise ValueError("The key in the dictionary should be a string representing the video basename.")

                if not all(isinstance(roi, list) and len(roi) == 4 for roi in rois_list):
                    raise ValueError("Each ROI should be a list of four coordinates.")

            converted_rois = rois

        else:
            raise TypeError("ROIs should be either a list of [ROIs, video_filepath] or a dictionary.")

        return converted_rois

    def _get_frame_dict(self):

        # Get video data
        video_data, video_files = VideoManipulator.get_video_data(self.video_paths, True)

        # Check video data for end times that are earlier due to wrong timezone in metadata (changes list in place)
        VideoManipulator.adjust_timestamps(video_data)

        if hasattr(self, "annotation_data_array") and getattr(self, "annotation_data_array") is not None:
            # Construct the valid annotation data array which contains visit data coupled with the path to the
            # video containing the visit
            # valid annotation entries in array have the following format:
            # [duration, time_of_visit, video_filepath, video_start_time, video_end_time]
            valid_annotations_array = VideoManipulator.construct_valid_annotation_array(self.annotation_data_array,
                                                                                        video_data)

            # If following Watchers file append also visitor id
            visitor_id = None # TODO: Fix this, just made sure it runs
            for annotation, vis_id in zip(valid_annotations_array, visitor_id):
                annotation += vis_id

            # Generate which frames should be exported for each video
            frame_dict = self._generate_frames_for_visits(valid_annotations_array, video_files)
        else:
            # Generate which frames should be exported for entire videos
            frame_dict = self._generate_frames_for_entire_videos(video_files)

        # Assign result to the attribute
        self.frame_dict = frame_dict

        return frame_dict

    def _generate_frames_for_visits(self, valid_annotations_array, video_files):
        # Retrieve frames per visit and frame skip settings from instance
        frames_per_visit = self.frames_per_visit
        frame_skip = self.frame_skip

        # Validate the input arrays
        if not valid_annotations_array:
            raise ValueError("The valid_annotations_array is empty")
        if not video_files:
            raise ValueError("The video_files list is empty")

        # Initialize dictionaries to store results
        frames_per_video_dict = {}
        visitor_category_dict = {}

        # Process each annotation in the valid annotations array
        processed_indices = set()
        for iter_num, annotation in enumerate(valid_annotations_array):
            if iter_num in processed_indices:
                continue  # Skip already processed annotations
            # Skip processing if the annotation is empty
            if not annotation:
                continue

            # Extract the video filepath from the annotation
            _, _, video_filepath, *_ = annotation

            # Find all annotations that match the current video file
            matching_annotations = [
                (iter_num, duration, time_of_visit, fp, video_start_time, *_)
                for duration, time_of_visit, fp, video_start_time, *_ in valid_annotations_array
                if fp == video_filepath
            ]

            # Store visitor category, if available
            if len(annotation) > 5:
                visitor_category_dict[iter_num] = annotation[5]

            # Lists to store frame numbers and visit numbers
            list_of_visit_frames = []
            list_of_visit_number = []

            # Process each matching annotation
            for visit_number, duration, time_of_visit, video_filepath, video_start_time, *_ in matching_annotations:
                # Attempt to retrieve fps and total frame count for the video
                try:
                    fps, total_frames = next((video_file.fps, video_file.total_frames) for video_file in video_files if
                                             video_file.filename == os.path.basename(video_filepath))
                except StopIteration:
                    logging.error(f"No video file found for visit at {video_filepath}.")
                    continue

                # Calculate the start time in seconds from the video start
                try:
                    time_from_start = int(
                        (pd.to_datetime(time_of_visit, format='%Y%m%d_%H_%M_%S') - video_start_time).total_seconds())
                except ValueError as e:
                    logging.error(f"Invalid date format in time_of_visit: {e}")
                    continue

                # Calculate the range of frames for the visit
                first_frame = time_from_start * fps
                adjusted_visit_duration = (min(
                    time_from_start * fps + int(duration) * fps,
                    total_frames
                ) - time_from_start * fps) // fps
                last_frame = first_frame + (adjusted_visit_duration * fps)

                # Generate frame numbers for the visit
                list_of_visit_frames += list(range(first_frame, last_frame, frame_skip if frames_per_visit < 1 else (
                        (adjusted_visit_duration * fps) // frames_per_visit)))
                list_of_visit_number += [visit_number for _ in list_of_visit_frames]

            # Keep track of the annotations already processed as matched_annotations
            processed_indices.update(set([visit_number for visit_number, _, _, _, _, _ in matching_annotations]))

            # Store the frame and visit number information for each video
            frames_per_video_dict[os.path.basename(video_filepath)] = (tuple(list_of_visit_frames),
                                                                       tuple(list_of_visit_number),
                                                                       visitor_category_dict)

        return frames_per_video_dict

    def _generate_frames_for_entire_videos(self, video_files):
        # Assign variables
        frame_skip = self.frame_skip

        # Check if the list of video files is empty and raise an error if so
        if not video_files:
            raise ValueError("The video_files list is empty")

        # Initialize a dictionary to store frame data for each video
        frames_per_video_dict = {}

        # Initialize a dictionary for visitor categories, defaulting to None
        visitor_category_dict = {0: None}

        # Iterate over each video file in the provided list
        for video_file in video_files:
            try:
                # Attempt to retrieve the frames per second (fps) and total frame count from the video file
                fps, total_frames = video_file.fps, video_file.total_frames
            except AttributeError as e:
                # Print an error message and skip this video file if fps or total_frames are missing
                logging.error(f"Missing fps or total_frames in video file: {video_file.filename}. Error: {e}")
                continue

            # Extract the base name of the video file for use in the dictionary
            video_basename = os.path.basename(video_file.filename)

            # Get the first frame number for the video
            first_frame = self.first_frame_number.get(video_basename, 0) if isinstance(self.first_frame_number,
                                                                                       dict) else self.first_frame_number

            # Generate a list of frame numbers, skipping frames according to the frame_skip value
            list_of_frames = list(range(first_frame, total_frames, frame_skip))

            # Create a corresponding list of visit numbers, all set to 0 (indicating no specific visit)
            list_of_visit_numbers = [0 for _ in list_of_frames]

            # Add the frame and visit number data to the dictionary for the current video file
            frames_per_video_dict[video_basename] = (tuple(list_of_frames),
                                                     tuple(list_of_visit_numbers),
                                                     visitor_category_dict)

        # Return the dictionary containing frame data for each video
        return frames_per_video_dict

    # Create destination folders : TODO: To predictor callback method
    #     output_folders = []
    #     output_folders.append(os.path.join(self.output_folder, "visitor"), exist_ok = True)
    #     output_folders.append(os.path.join(self.output_folder, "empty"), exist_ok = True)
    #     output_folders.append(self._create_conf_folders(threshold, os.path.join(self.output_folder, "visitor")))
    #     output_folders = Manipulator.create_folders(output_folders)

    def _create_conf_folders(self, threshold, parent_folder):
        try:
            # Validate the threshold
            if not 0.1 <= threshold <= 0.9:
                raise ValueError("Invalid threshold. Please choose a value between 0.1 and 0.9.")

            # Generate folder paths based on the threshold
            folder_list = [f"{parent_folder}/{round(x / 10, 1)}" for x in range(int(threshold * 10), 10)]
            folder_list.append(f"{parent_folder}/rest")  # Include 'rest' folder

            return folder_list

        except Exception as e:
            logging.error(f"An error occurred in create_conf_folders: {e}")
            return [f"{parent_folder}/rest"]

    def run(self, producers: int = 1, consumers: int = 1, frame_batch_size: int = 100):

        # Assign attributes and variables
        self.frame_batch_size = frame_batch_size
        frame_dict = self.frame_dict if hasattr(self, "frame_dict") and self.frame_dict else self._get_frame_dict()  # If not created get dict
        video_files = tuple(Video(filepath) for filepath in self.video_paths)

        try:
            queue = Queue(maxsize=self.queue_size)  # create shared queue
            with ThreadPoolExecutor(max_workers=producers + consumers) as executor:  # create workers
                # Chunk the passed video files to chunks with the size of nprod
                chunks = iter(video_files)
                for chunk in iter(lambda: list(itertools.islice(chunks, producers)), []):
                    logging.debug("Checking videos in this chunk.")
                    producer_futures = []

                    # A producer task created for each video_file if the frame data for this file is found in the dict
                    for video_file in chunk:
                        logging.debug("Does this video have visits?")
                        if frame_dict.get(video_file.video_name):
                            logging.debug("Yes - creating producer per video.")
                            future = executor.submit(self._producer_task, video_file,
                                                     frame_dict[video_file.video_name][0],
                                                     frame_dict[video_file.video_name][1], queue)
                            producer_futures.append(future)
                            logging.debug(f"Producer futures: {producer_futures}")

                    # For each chunk ndet number of consumer tasks are created
                    consumer_futures = [executor.submit(self.consumer_task, n, queue, **self.config) for n in range(consumers)]
                    logging.debug(f"Consumer futures: {consumer_futures}")

                    # Wait until all producers are finished
                    for future in as_completed(producer_futures):
                        logging.debug(f"<{future}> completed.")
                        pass

                    # Stopper sentient values are put to the queue for each detector
                    for _ in range(consumers):
                        queue.put(FrameGeneratorTask(None, None))
        except Exception as e:
            logging.error(f"Error in FrameGenerator run method: {e}")
            # Consider re-raising or handling the exception

    def _chunk_tuples(self, frame_numbers_tuple1, frame_numbers_tuple2, chunk_size):
        # Generate chunks of frame numbers and visit indices
        min_length = min(len(frame_numbers_tuple1), len(frame_numbers_tuple2))
        for i in range(0, min_length, chunk_size):
            end = min(i + chunk_size, min_length)
            yield (frame_numbers_tuple1[i:end], frame_numbers_tuple2[i:end], end - i)

    def _producer_task(self, video_object_file, frame_indices, visit_indices, queue):
        filename = None
        try:
            filepath = video_object_file.video_path
            filename = video_object_file.video_name

            print(f"(P) - Producer <{filename}> successfully created")

            # Iterate through chunks of frames
            frame_batch_chunks = self._chunk_tuples(frame_indices, visit_indices, self.frame_batch_size)
            for frame_numbers_chunk, visit_numbers_chunk, actual_chunk_size in frame_batch_chunks:
                logging.debug(f"(P) - Producer <{filename}> created a chunk of size: {actual_chunk_size}")

                # Pre-allocate 4D array with frame dimensions and the current chunk size
                frame_height, frame_width = video_object_file.frame_height, video_object_file.frame_width
                frames_array = np.zeros((actual_chunk_size, frame_height, frame_width, 3), dtype=np.uint8)
                print(f"(P) - Producer <{filename}> created a chunk of size: {actual_chunk_size}")

                # Read frames with a reader that automatically decides based on video origin (defaults to decord)
                for idx, frame_dict in enumerate(video_object_file.read_video_frame(frame_indices=frame_numbers_chunk, stream=True)):
                    frames_array[idx] = frame_dict['frame']
                    logging.debug(f"(P) - Adding a frame into array <{idx}>")
                logging.debug("(P) - Array created")

                meta_data = {
                    'frame_numbers': frame_numbers_chunk,
                    'visit_numbers': visit_numbers_chunk,
                    'video_path': filepath,
                    'video_name': filename
                }
                logging.debug("(P) - Metadata packed")
                queue.put(FrameGeneratorTask(frames_array, meta_data))
                logging.debug("(P) - Package added to the queue")
                logging.debug(
                    f"(P) - Producer <{filename}> added batch <{frame_numbers_chunk[0]} - {frame_numbers_chunk[-1:][0]}> to queue.")
        except Exception as e:
            logging.error(f"Error in producer task for {filename}: {e}")
            # Decide whether to retry, log and continue, or stop

    def consumer_task(self, name, queue, **kwargs):
        logging.debug(f"(C) - Consumer {name} created.")
        while True:
            try:
                # Get task from the queue
                task = queue.get()

                # Terminate thread if item is none sentinel value
                if task.frames_array is None and task.metadata is None:
                    print(f"(C) - Consumer {name} finished.")
                    break

                # Unpack data from the task
                frames_array = task.frames_array
                frame_numbers = task.get_meta('frame_numbers')
                video_filepath = task.get_meta('video_path')
                video_filename = task.get_meta('video_name')
                visit_numbers = task.get_meta('visit_numbers')

                print(f"(C) - Consumer {name} got element <{frame_numbers[0]} - {frame_numbers[-1:][0]}>")

                # Call the processing callback if it's set
                if self.processing_callback:
                    try:
                        self.processing_callback(
                            frames_array=frames_array,
                            frame_numbers=frame_numbers,
                            frame_skip=self.frame_skip,
                            video_filepath=video_filepath,
                            video_filename=video_filename,
                            visit_numbers=visit_numbers,
                            task=task,
                            **kwargs
                        )
                    except Exception as callback_exc:
                        logging.error(f"Error during processing callback in consumer task {name}: {callback_exc}")
                        traceback.print_exc()
                        # Consider whether to continue or break the loop based on the nature of the error

            except Exception as e:
                logging.error(f"Error in consumer task {name}: {e}")
                # Decide whether to break the loop or continue processing based on the nature of the error


def get_frame_batch_size(frame_height, frame_width, max_memory_usage_bytes: int=300000000):
    frame_size_bytes = frame_height * frame_width * 3
    batch_size = int(max_memory_usage_bytes // frame_size_bytes)
    return batch_size

