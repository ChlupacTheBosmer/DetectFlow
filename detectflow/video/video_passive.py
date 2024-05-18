# This file contains the video data classes
#
# Default python modules
import datetime
import os
# Other modules
import cv2
import imageio
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
from decord import VideoReader
from decord import cpu, gpu
from typing import Union, List, Generator, Tuple
import logging

class VideoFilePassive:
    __slots__ = ('file_extension', 'filename', 'filepath', 'fps', 'recording_details', 'recording_identifier', 'rois', 'timestamp', 'total_frames', 'video_origin')

    def __init__(self, filepath):

        # Check requirements
        self.check_requirements(filepath)

        # Define variables
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        # Determine video origin
        if filepath.endswith(".mp4"):
            self.video_origin = "VT"
        elif filepath.endswith(".avi"):
            self.video_origin = "MS"

        # Define variables
        self.filename = os.path.basename(self.filepath)
        (self.recording_identifier, self.timestamp, self.recording_details,
         time_details, self.file_extension) = self.get_data_from_recording_name()
        self.rois = []

        # # Chose a different method based on whether the video is a Vivotek .mp4 or a Milesight .avi
        # if self.video_origin == "VT":
        #     self.cap = cv2.VideoCapture(self.filepath)

        # Get basic video properties
        self.fps = self.get_video_fps()
        self.total_frames = self.get_video_total_frames()

    def check_requirements(self, filepath):

        # Check requirements
        if not filepath.endswith(".mp4") or filepath.endswith(".avi"):
            logging.error("Invalid file type. Provide path to a valid video file.")

    def get_data_from_recording_name(self):

        filename = os.path.basename(self.filepath)

        # Prepare name elements
        locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")
        file_extension = filename[-4:]

        # Define compound info
        recording_identifier = "_".join([locality, transect, plant_id])
        timestamp = "_".join([date, hour, minutes])

        return recording_identifier, timestamp, [locality, transect, plant_id], [date, hour,
                                                                                 minutes], file_extension

    def get_video_fps(self):

        try:
            if self.video_origin == "MS":
                # Get the creation date from metadata
                parser = createParser(self.filepath)
                metadata = extractMetadata(parser)
                fps = float(metadata.get("frame_rate"))
            else:
                cap = cv2.VideoCapture(self.filepath)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
        except Exception as e:
            logging.warning(f'Unable to read video fps: {self.filepath}. Exception: {e}')
            fps = 25
        return int(fps)

    def get_video_total_frames(self):

        if self.video_origin == "MS":

            # Will return timedelta object
            duration = self.get_video_duration()

            # Get total number of seconds
            duration = duration.total_seconds()

            # Get fps
            if self.fps is not None:
                fps = self.fps
            else:
                fps = self.get_video_fps()

            # Calculate total_frames
            total_frames = int(duration * fps)

        else:
            cap = cv2.VideoCapture(self.filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        return total_frames

    def get_video_duration(self):

        try:
            # Get the creation date from metadata
            parser = createParser(self.filepath)
            metadata = extractMetadata(parser)
            duration = str(metadata.get("duration"))
        except Exception as e:
            logging.warning(f'Unable to read video duration from video: {self.filepath}. Exception: {e}')
            duration = input(f"Unable to read video duration from video: {self.filepath}\n"
                             "Enter duration time (hh:mm:ss): ")

        # Split duration string into hours, minutes, and seconds
        duration_parts = duration.split(':')
        hours, minutes, seconds = map(float, duration_parts)

        # Calculate the timedelta for the duration
        duration = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

        return duration

    def read_video_frame(self, frame_indices: Union[List[int], Tuple[int, ...], int] = 0, stream: bool = True, prioritize: str = 'opencv') -> Union[List, Generator]:

        # Process the argument
        frame_indices = [frame_indices] if not isinstance(frame_indices, (list, tuple)) else frame_indices

        # Based on the video origin, choose the appropriate reader. The "prioritize" arg sets default reader
        prioritized_reader = self.read_frames_opencv2 if prioritize == 'opencv' else self.read_frames_decord
        chosen_reader = self.read_frames_imageio if self.video_origin == "MS" else prioritized_reader

        if stream:
            return (frame for frame in chosen_reader(frame_indices))  # This returns a generator
        else:
            return list(chosen_reader(frame_indices))  # This returns a list

    # def frame_reader_wrapper(self, frame_reader_func, frame_indices: Union[List[int], int] = 0, stream: bool = True):
    #     if stream:
    #         yield from frame_reader_func(frame_indices)
    #     else:
    #         frames = list(frame_reader_func(frame_indices))
    #         return frames

    def read_frames_opencv2(self, frame_indices):

        # Initiate the VideoCapture
        cap = cv2.VideoCapture(self.filepath)

        try:
            for frame_number in frame_indices:

                # Set cap position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()

                if success:
                    yield (self.recording_identifier, self.timestamp, frame_number, frame, self.rois)
        except Exception as e:
            pass
        finally:
            if cap:
                cap.release()

    def read_frames_imageio(self, frame_indices):

        # Open the video file using imageio
        video = imageio.get_reader(self.filepath)

        try:
            for frame_number in frame_indices:
                # Read the frame
                frame = video.get_data(frame_number)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                yield (self.recording_identifier, self.timestamp, frame_number, frame, self.rois)
        except IndexError:
            yield from self.read_frames_opencv2(frame_indices)
        finally:
            if video:
                del video

    def read_frames_decord(self, frame_indices):

        vr = VideoReader(self.filepath, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

        # load the VideoReader
        try:
            for frame_number in frame_indices:
                frame = vr[frame_number]  # read an image from the capture
                yield (self.recording_identifier, self.timestamp, frame_number, frame.asnumpy(), self.rois)
        except Exception as e:
            pass
        finally:
            if vr:
                del vr

    def get_frame_shape(self):

        # Will retrieve the dimensions of the video
        if self.video_origin == "MS":
            video_reader = imageio.get_reader(self.filepath)

            try:
                metadata = video_reader.get_meta_data()
                frame_width = metadata["size"][0]
                frame_height = metadata["size"][1]
            except:
                # Read the first frame to get its shape properties
                first_frame = video_reader.get_data(0)
                frame_height, frame_width, _ = first_frame.shape
        else:
            try:
                cap = cv2.VideoCapture(self.filepath)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except:
                raise

        return frame_width, frame_height


