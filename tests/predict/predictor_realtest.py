from detectflow.predict.predictor import Predictor
from detectflow.predict.tracker import Tracker
from detectflow.video.video_data import Video
from detectflow.utils.sampler import Sampler
from detectflow.utils.inspector import Inspector
from detectflow.config import DETECTFLOW_DIR
import unittest
import os

class test_predictor(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor(tracker="botsort.yaml")

        # Put together the path to the test video file
        self.video_path = os.path.join(os.path.dirname(DETECTFLOW_DIR), 'tests', 'video', 'resources',
                                  'GR2_L1_TolUmb2_20220524_07_44.mp4')
        self.flower_model_path = os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt')
        self.visitor_model_path = os.path.join(DETECTFLOW_DIR, 'models', 'visitors.pt')

        # Init video file
        self.video = Video(self.video_path)

        # Get dummy frame numbers
        self.frame_nums = list(range(0, 100, 10))

        # Get frames from video
        self.frames = [frame_dict['frame'] for frame_dict in self.video.read_video_frame(self.frame_nums, False)]

    def test_detect(self):

        # Get dummy reference boxes
        reference_boxes = Sampler.create_sample_bboxes(as_detection_boxes=True)

        metadata = {
            'frame_number': self.frame_nums,
            'visit_number': 1,
            'source_path': self.video_path,
            'reference_boxes': [reference_boxes for _ in self.frames]
        }

        # Call the predict method with real parameters
        for result in self.predictor.detect(frame_numpy_array=self.frames,
                                            model_path=self.flower_model_path,
                                            detection_conf_threshold=0.3,
                                            metadata=metadata,
                                            tracked=True,
                                            filter_tracked=False,
                                            device='cpu'):

            # print len of boxes after tracking
            print("Boxes AFTER tracking", len(result.boxes))
            print(result.boxes.data)

            print('')
            print('Result attributes:')
            print(result.frame_number)
            print(result.boxes)
            print(result.visit_number)
            print(result.source_path)
            print(result.source_name)
            print(result.filtered_boxes)

            Inspector.display_frames_with_boxes([result.orig_img], [result.boxes if hasattr(result, 'boxes') else None])

    def test_caching_model(self):
        import time

        # It will call the method three times with same model path
        durations = []
        for i in range(3):
            start_time = time.time()
            # Call the predict method with real parameters
            for result in self.predictor.detect(frame_numpy_array=self.frames[:1],
                                                model_path=self.flower_model_path,
                                                detection_conf_threshold=0.3,
                                                tracked=True,
                                                filter_tracked=False,
                                                device='cpu'):

                print('')
                print('Result attributes:')
                print(result.frame_number)

                Inspector.display_frames_with_boxes([result.orig_img], [result.boxes if hasattr(result, 'boxes') else None])
            durations.append(time.time() - start_time)

        for i, duration in enumerate(durations):
            print(f"Duration #{i+1}: {duration:.4f} s")
