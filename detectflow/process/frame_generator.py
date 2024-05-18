from typing import List, Tuple, Union, Optional, Dict
import logging
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
import pandas as pd
from detectflow.manipulators.video_manipulator import VideoManipulator
from detectflow.video.video_passive import VideoFilePassive
from detectflow.validators.validator import Validator
#TODO: Add imports for video classes and functions dealing with the excel fiel and dataframe

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
            self.crop_size = crop_size
            self.offset_range = offset_range
            self.name_prefix = name_prefix
            self.output_folder = output_folder
            self.processing_callback = processing_callback
            self.config = kwargs
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
                excel = anno_data.Annotation_watcher_file(excel_path, True, True, True, True)
                dataframe = excel.dataframe

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

    def run(self, nprod: int = 1, ndet: int = 1, frame_batch_size: int = 100):

        # Assign attributes and variables
        self.frame_batch_size = frame_batch_size
        frame_dict = self.frame_dict if hasattr(self,
                                                "frame_dict") and self.frame_dict else self._get_frame_dict()  # If not created get dict
        video_files = tuple(VideoFilePassive(filepath) for filepath in self.video_paths)

        try:
            queue = Queue()  # create shared queue
            with ThreadPoolExecutor(max_workers=nprod + ndet) as executor:  # create workers
                chunks = iter(video_files)
                for chunk in iter(lambda: list(itertools.islice(chunks, nprod)),
                                  []):  # Chunk the passed video files to chunks with the size of nprod
                    logging.debug("Checking videos in this chunk.")
                    producer_futures = []
                    # A producer task is created for each video_fiel if the frame data for this file are found in the dict
                    for video_file in chunk:
                        logging.debug("Does this video have visits?")
                        if frame_dict.get(video_file.filename):
                            logging.debug("Yes - creating producer per video.")
                            future = executor.submit(self._producer_task, video_file,
                                                     frame_dict[video_file.filename][0],
                                                     frame_dict[video_file.filename][1], queue)
                            producer_futures.append(future)
                            logging.debug(f"Producer futures: {producer_futures}")

                    # For each chunk ndet number of detector tasks are created
                    detector_futures = [executor.submit(self.detector_task, n, queue, **self.config) for n in
                                        range(ndet)]
                    logging.debug(f"Detector futures: {detector_futures}")

                    # Wait until all producers are finished
                    for future in as_completed(producer_futures):
                        logging.debug(f"<{future}> completed.")
                        pass

                    # Stopper sentient values are put to the queue for each detector
                    for _ in range(ndet):
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
        try:
            filepath = video_object_file.filepath
            filename = video_object_file.filename

            print(f"(P) - Producer <{filename}> successfully created")

            # Iterate through chunks of frames
            frame_batch_chunks = self._chunk_tuples(frame_indices, visit_indices, self.frame_batch_size)
            for frame_numbers_chunk, visit_numbers_chunk, actual_chunk_size in frame_batch_chunks:
                logging.debug(f"(P) - Producer <{filename}> created a chunk of size: {actual_chunk_size}")

                # Pre-allocate 4D array with frame dimensions and the current chunk size
                frame_height, frame_width = video_object_file.get_frame_shape()
                frames_array = np.zeros((actual_chunk_size, frame_width, frame_height, 3), dtype=np.uint8)
                print(f"(P) - Producer <{filename}> created a chunk of size: {actual_chunk_size}")

                # Read frames with a reader that automatically decides based on video origin (defaults to decord)
                frame_generator = video_object_file.read_video_frame(frame_numbers_chunk, True, 'decord')
                for idx, frame_list in enumerate(frame_generator):
                    frames_array[idx] = frame_list[3]
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

    def detector_task(self, name, queue, **kwargs):
        logging.debug(f"(D) - Detector {name} created.")
        while True:
            try:
                # Get task from the queue
                task = queue.get()

                # Terminate thread if item is none sentinel value
                if task.frames_array is None and task.metadata is None:
                    print(f"(D) - Detector {name} finished.")
                    break

                # Unpack data from the task
                frames_array = task.frames_array
                frame_numbers = task.get_meta('frame_numbers')
                video_filepath = task.get_meta('video_path')
                video_filename = task.get_meta('video_name')
                visit_numbers = task.get_meta('visit_numbers')

                print(f"(D) - Detector {name} got element <{frame_numbers[0]} - {frame_numbers[-1:][0]}>")

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
                            generator=self,
                            **kwargs
                        )
                    except Exception as callback_exc:
                        logging.error(f"Error during processing callback in detector task {name}: {callback_exc}")
                        # Consider whether to continue or break the loop based on the nature of the error

            except Exception as e:
                logging.error(f"Error in detector task {name}: {e}")
                # Decide whether to break the loop or continue processing based on the nature of the error

# def process_frames_callback(**kwargs):

#     KWARGS_MAP = {
#         'flower_model_path': str,
#         'flower_detection_conf_threshold': float,
#         'visitor_model_path': str,
#         'visitor_detection_conf_threshold': float,
#         'task': FrameGeneratorTask,
#         ''

#     }

#     # Process input and assign variables
#     if kwargs:
#         predictor = Predictor() if not 'predictor' in kwargs else kwargs['predictor']
#         flower_model_path = None if not 'flower_model_path' not in kwargs else kwargs['flower_model_path']
#         flower_conf = 0.3 if not 'flower_detection_conf_threshold' not in kwargs else kwargs['flower_detection_conf_threshold']
#         visitor_model_path = None if not 'visitor_model_path' not in kwargs else kwargs['visitor_model_path']
#         visitor_conf = 0.3 if not 'visitor_detection_conf_threshold' not in kwargs else kwargs['visitor_detection_conf_threshold']
#         if 'task' in kwargs:
#             metadata = kwargs['task'].metadata
#             frames_array = kwargs['task'].frames_array
#             frame_height, frame_width, _

#     # For the first 12 frames (or random sample) in the batch detect the flowers
#     for result in predictor.detect(self,
#                                    frame_numpy_array = frames_array[0],
#                                    metadata = metadata,
#                                    model_path = flower_model_path,
#                                    detection_conf_threshold = flower_conf):
#         flower_boxes =

#     # Get consistent flowers or Remove outliers

#     # Run the easy check for crop

#     # (?) Analyze clusters

#     # Crop frames

#     # Run Detections on frame batch

#     # Track, update results

#     # Save results

#     # (PostProcess data)

#     # Send results to database queue


#     # Run detection on frame batch
#     detection_threshold = self.detection_conf_threshold # May be different from the sorting threshold to include less certain results but discriminate them in sorting
#     detection_metadata = detect_visitors_in_frame_array_sahi(frames_array, metadata, self.model_path, detection_threshold);

#     # Process results
#     for idx, (frame_number, roi_number, visit_number, detection, _, boxes, confs, classes) in enumerate(detection_metadata):

#         print(idx, frame_number)

#         # Define name and path variables
#         frame_prefix = f'{self.name_prefix}_' if self.name_prefix != "" else ''
#         frame_name = f"{frame_prefix}{video_filename}_{roi_number}_{frame_number}_{visit_number}_{frame_width}_{frame_height}.jpg"
#         output_path = os.path.join(self.output_folder, "visitor") if detection > 0 else os.path.join(self.output_folder, "empty")

#         if any(conf >= self.sort_conf_threshold for conf in confs):
#             # Find the minimum value of confs
#             min_conf = min(confs)
#             action_taken = False

#             # Iterate through the folder list in reverse order
#             for limit in reversed(self.conf_limits):
#                 if min_conf > limit:
#                     # Save frame into the folder corresponding to the current conf limit
#                     print(f"The minimum value in {confs} is larger than {limit}")
#                     frame_path = os.path.join(output_path, str(limit), frame_name)
#                     label_path = os.path.join(output_path, str(limit), f"{frame_name[:-4]}")
#                     action_taken = True
#                     break

#             # If no action has been taken, then 'pass'
#             if not action_taken:
#                 frame_path = os.path.join(output_path, "rest", frame_name)
#                 label_path = os.path.join(output_path, "rest", f"{frame_name[:-4]}")
#         else:
#             detection = 0
#             frame_path = os.path.join(self.output_folder, "empty", frame_name)

#         # Save image and annotation file
#         try:
#             image_rgb = cv2.cvtColor(frames_array[idx], cv2.COLOR_BGR2RGB)
#             cv2.imwrite(frame_path, image_rgb)
#             print(f"(D) SAVED: - {frame_path}")
#             if detection > 0:
#                 save_label_file(label_path, boxes, classes, confs)
#         except Exception as e:
#             print("Error", e)
#             tb_str = traceback.format_exc()
#             print(tb_str)

