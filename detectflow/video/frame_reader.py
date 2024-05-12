import cv2
import imageio
from decord import VideoReader, cpu

class FrameReader:
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