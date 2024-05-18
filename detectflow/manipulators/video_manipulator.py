import os
from datetime import datetime, timedelta
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.validator import Validator
from detectflow.validators.object_detect_validator import ObjectDetectValidator
from detectflow.video.video_inter import VideoFileInteractive
import pandas as pd

class VideoManipulator:

    @staticmethod
    def load_videos(video_folder_path):
        # If the path is valid folder path
        video_filepaths = None
        if Validator.is_valid_directory_path(video_folder_path):
            try:  # The list_files function validated files, validation so no validation on return
                video_filepaths = Manipulator.list_files(video_folder_path, extensions=('.mp4', '.avi'),
                                                         return_full_path=True)
            except:
                raise
        else:
            raise ValueError("Invalid video folder path")

        return video_filepaths

    @staticmethod
    @Validator.validate_paths
    def get_relevant_video_paths(video_filepaths, annotation_data_array):

        new_video_filepaths = set()
        if video_filepaths and annotation_data_array:
            for visit_data in annotation_data_array:
                for filepath in video_filepaths:
                    try:
                        if visit_data[1][:-9] in os.path.basename(filepath):
                            time_difference = datetime.strptime(visit_data[1][-8:-3], '%H_%M') - datetime.strptime(
                                filepath[-9:-4], '%H_%M')
                            if timedelta() <= time_difference <= timedelta(minutes=15):
                                new_video_filepaths.add(filepath)
                                break
                    except Exception as e:
                        raise ValueError(f"Error processing time data in {visit_data}: {e}")

            if not new_video_filepaths:
                raise ValueError("No relevant video paths found matching the annotations")

        return list(new_video_filepaths)

    @staticmethod
    def get_video_data(video_filepaths, return_video_file_objects: bool = False):
        if not video_filepaths:
            raise ValueError("The video_filepaths list is empty")

        video_data = []
        video_files = []
        for filepath in video_filepaths:
            if filepath.endswith(('.mp4', '.avi')):
                try:
                    video = VideoFileInteractive(filepath, None, (
                    0, 0))  # TODO: Consider changing the class to bypass the tkinter manual extraction
                    video_files.append(video)
                    video_data_entry = [video.filepath, video.start_time, video.end_time]
                    video_data.append(video_data_entry)
                except Exception as e:
                    raise Exception(f"Failed to process video '{filepath}': {e}")

        if return_video_file_objects:
            return video_data, video_files
        else:
            return video_data

    @staticmethod
    @ObjectDetectValidator.validate_annotation_data_array
    def get_annotation_data_array(dataframe):
        annotation_data_array = None
        if dataframe is not None and not dataframe.empty:
            annotation_data_array = dataframe.loc[:, ['duration', 'ts']].values.tolist()
            visitor_id = dataframe.loc[:, ['vis_id']].values.tolist()
        return annotation_data_array

    @staticmethod
    @ObjectDetectValidator.validate_annotation_array
    def construct_valid_annotation_array(annotation_data_array, video_data):
        if not annotation_data_array:
            raise ValueError("The annotation_data_array is empty")

        if not video_data:
            raise ValueError("The video_data list is empty")

        valid_annotations_array = []
        for index, annotation_data in enumerate(annotation_data_array):
            try:
                annotation_time = pd.to_datetime(annotation_data[1], format='%Y%m%d_%H_%M_%S')
            except ValueError as e:
                raise ValueError(f"Invalid date format in annotation_data at index {index}: {e}")

            try:
                relevant_video_data = next(
                    (video_data_entry for video_data_entry in video_data if
                     video_data_entry[1] <= annotation_time <= video_data_entry[2]),
                    None)
            except Exception as e:
                raise Exception(f"Error finding relevant video data for annotation at index {index}: {e}")

            if relevant_video_data:
                valid_annotation_data_entry = annotation_data + relevant_video_data
                valid_annotations_array.append(valid_annotation_data_entry)

        if not valid_annotations_array:
            raise ValueError("No valid annotations were found matching the video data")

        return valid_annotations_array

    @staticmethod
    def adjust_timestamps(list_of_lists):
        if not isinstance(list_of_lists, list):
            raise TypeError("Expected a list of lists.")

        for item in list_of_lists:
            # Validate the structure of each item
            if not (isinstance(item, list) and len(item) >= 3):
                raise ValueError("Each item in the list must be a list with at least three elements.")

            try:
                first_timestamp, second_timestamp = item[1], item[2]

                # Validate that the timestamps are datetime objects
                if not (isinstance(first_timestamp, datetime) and isinstance(second_timestamp, datetime)):
                    raise TypeError("Timestamps must be datetime objects.")

                # Check if the second timestamp is earlier than the first
                if second_timestamp < first_timestamp:
                    # Add one hour to the second timestamp
                    corrected_timestamp = second_timestamp + timedelta(hours=1)
                    # Update the timestamp in the list
                    item[2] = corrected_timestamp

            except Exception as e:
                # Handle any unexpected errors during processing
                print(f"Error adjusting timestamps for item {item}: {e}")
                continue  # Optionally continue processing the rest of the list or return/raise an error
