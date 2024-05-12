from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from typing import Optional, List
from detectflow.predict.results import DetectionResults, DetectionBoxes


class Tracker:
    # A mapping of tracker types to corresponding tracker classes
    TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

    def __init__(self, tracker_type: str = "botsort.yaml", trackers_number: int = 1):
        self.tracker_type = tracker_type
        self.trackers_number = trackers_number
        self.trackers = self._load_trackers()

    def _load_trackers(self):
        try:
            tracker = check_yaml(self.tracker_type)
            cfg = IterableSimpleNamespace(**yaml_load(self.tracker_type))

            if cfg.tracker_type not in ['bytetrack', 'botsort']:
                raise ValueError(f"Unsupported tracker type: '{cfg.tracker_type}'")

            trackers = []
            for _ in range(self.trackers_number):
                tracker = self.TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                trackers.append(tracker)

            return trackers
        except Exception as e:
            raise Exception(f"Error in loading trackers: {e}")

    def process_tracking(self,
                         detection_results: Optional[List],
                         batch_size: int = 1,
                         persist: bool = True):

        """
        Process a batch of frames with its detection results and update with object tracking.

        Args:
            detection_results: (list, optional): List of DetectionResults objects with detections. Also contains frames in attr
            batch_size: (int): Currently this is number of repeated runs of the tracker. NotImplemented.
            persist (bool, optional): Whether to persist the tracker if it already exists. Defaults to False.
        """

        if isinstance(detection_results, DetectionResults):
            detection_results = [detection_results]

        if not self.trackers:
            raise Exception("Trackers not initialized")
        if len(detection_results) == 0:
            return []

        updated_detections = []
        try:
            for i in range(
                    batch_size):  # TODO: If you want to support more than one batch you need to ensure iteration over different sources
                for detection_result in detection_results:
                    if (not persist):
                        self.trackers[i].reset()
                    else:
                        if detection_result is None or detection_result.boxes is None:  # Check for None values
                            print("No detection")
                            updated_detections.append(detection_result)
                            continue
                        detection = detection_result.boxes.cpu().numpy()
                        # print(detection)
                        if len(detection) == 0:
                            print("No detection")
                            updated_detections.append(detection_result)
                            continue

                    # print(detection_result.orig_img)
                    tracks = self.trackers[i].update(detection, detection_result.orig_img)

                    if len(tracks) == 0:
                        print("No tracks")
                        updated_detections.append(detection_result)
                        continue

                    idx = tracks[:, -1].astype(int)
                    detection = detection.data[idx]
                    # print(tracks[:,:-1])
                    detection_result.boxes = DetectionBoxes(tracks[:, :-1], detection_result.boxes.orig_shape,
                                                            "xyxytpc")
                    updated_detections.append(detection_result)

            return updated_detections
        except Exception as e:
            raise Exception(f"Error in processing tracking: {e}")