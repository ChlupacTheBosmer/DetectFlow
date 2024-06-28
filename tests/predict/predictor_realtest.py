from detectflow.predict.predictor import Predictor
from detectflow.predict.tracker import Tracker
from detectflow.video.video_data import Video
from detectflow.utils.sampler import Sampler
from detectflow.utils.inspector import Inspector
from detectflow.config import ROOT
import unittest
import os

class test_predictor(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor(tracker="botsort.yaml")

    def test_detect(self):

        # Put together the path to the test video file
        video_path = os.path.join(os.path.dirname(ROOT), 'tests', 'video', 'resources', 'GR2_L1_TolUmb2_20220524_07_44.mp4')

        # Init video file
        vid = Video(video_path)

        # Get dummy frame numbers
        frame_nums = list(range(0, 100, 10))

        # Get frames from video
        frames = [frame_dict['frame'] for frame_dict in vid.read_video_frame(frame_nums, False)]

        # Get dummy reference boxes
        reference_boxes = Sampler.create_sample_bboxes(as_detection_boxes=True)

        metadata = {
            'frame_number': frame_nums,
            'visit_number': 1,
            'source_path': video_path,
            'reference_boxes': [reference_boxes for _ in frames]
        }

        # Call the predict method with real parameters
        for result in self.predictor.detect(frame_numpy_array=frames,
                                            model_path=os.path.join(ROOT, 'models', 'flowers.pt'),
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