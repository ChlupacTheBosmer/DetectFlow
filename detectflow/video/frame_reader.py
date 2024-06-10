import cv2
import imageio
import numpy as np
from decord import VideoReader, cpu
from typing import List, Tuple, Union, Generator, Optional
from detectflow.validators.video_validator import VideoValidator
import logging
import os

class SimpleFrameReader:
    def __init__(self, video_path, method='imageio'):
        self.video_path = video_path
        self.method = method
        self.reader = None

        if method == 'imageio':
            self.reader = imageio.get_reader(video_path)
        elif method == 'decord':
            self.reader = VideoReader(video_path, ctx=cpu(0))

    def get_frame(self, frame_number):
        try:
            if self.method == 'imageio':
                frame = self.reader.get_data(frame_number)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.method == 'decord':
                frame = self.reader[frame_number].asnumpy()
        except:
            frame = None
        return frame

    def close(self):
        if self.method == 'imageio':
            del self.reader
        elif self.method == 'decord':
            del self.reader


class FrameReader:

    def __init__(self, video_path: str, reader_method: Optional[str] = None):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.reader_method = reader_method

        # Maps method strings to readers and also sets order of preference
        self.READERS = {"decord": self.read_frames_decord,
                        "opencv": self.read_frames_opencv2,
                        "imageio": self.read_frames_imageio}

    def get_reader(self):

        if self.reader_method is None:
            # Run validator on video
            try:
                result = VideoValidator(self.video_path).validate_video()
                return next((self.READERS.get(reader, self.read_frames_opencv2) for reader in self.READERS if result.get(reader, False)), None)
            except Exception as e:
                logging.warning(f'Unable to validate video: {self.video_name}. Using default reader. Exception: {e}')
                return self.READERS.get('opencv')
        else:
            return self.READERS.get(self.reader_method, self.read_frames_opencv2)


    def read_video_frame(self, frame_indices: Union[List[int], Tuple[int, ...], int] = 0, stream: bool = True) -> Union[List, Generator]:

        # Process the argument to be a collection of frame indices
        frame_indices = [frame_indices] if not isinstance(frame_indices, (list, tuple)) else frame_indices

        # Choose the appropriate reader and return either as a generator or a list
        return (frame for frame in self.get_reader()(frame_indices)) if stream else list(self.get_reader()(frame_indices))

    def read_frames_opencv2(self, frame_indices):

        # Initiate the VideoCapture
        cap = cv2.VideoCapture(self.video_path)
        frame_number = None
        try:
            for frame_number in frame_indices:

                # Set cap position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()

                if success:
                    yield {"frame_number": frame_number, "frame": frame}
        except Exception as e:
            logging.warning(f"Error reading frame number: {frame_number} with OpenCV: {e}")
        finally:
            if cap:
                cap.release()

    def read_frames_imageio(self, frame_indices, enable_fallback: bool = True):

        # Open the video file using imageio
        video = imageio.get_reader(self.video_path)
        frame_number = None
        try:
            for frame_number in frame_indices:
                # Read the frame
                frame = video.get_data(frame_number)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                yield {"frame_number": frame_number, "frame": frame}
        except Exception as e:
            # If reading fails continue using the fallback opencv reader
            if enable_fallback:
                yield from self.read_frames_opencv2(
                    frame_indices[frame_indices.index(frame_number):] if frame_number in frame_indices else [])
            else:
                logging.warning(f"Error reading frame number: {frame_number} with imageio: {e}")
        finally:
            if video:
                del video

    def read_frames_decord(self, frame_indices, enable_fallback: bool = True):

        # Open the video file using decord
        vr = VideoReader(self.video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
        frame_number = None
        try:
            for frame_number in frame_indices:
                frame = vr[frame_number]  # read an image from the capture
                frame = np.ndarray(frame)
                yield {"frame_number": frame_number, "frame": frame}
        except Exception as e:
            # If reading fails continue using the fallback opencv reader
            if enable_fallback:
                yield from self.read_frames_opencv2(
                    frame_indices[frame_indices.index(frame_number):] if frame_number in frame_indices else [])
            else:
                logging.warning(f"Error reading frame number: {frame_number} with decord: {e}")
        finally:
            if vr:
                del vr

