# from concurrent.futures import ThreadPoolExecutor, as_completed
# import logging
# import numpy as np
# from detectflow.predict.results import DetectionResults
# from detectflow.predict.predictor import Predictor
# import os
#
#
# class Ensembler(Predictor): #TODO: Check and finish while implementing support for multiple models such as motion enriched etc.
#
#     def __init__(self,
#                  model_path: str = os.path.join('resources', 'yolo', 'best.pt'),
#                  detection_conf_threshold: float = 0.5):
#
#         # Assign attributes
#         self.model_path = model_path
#         self.detection_conf = detection_conf_threshold
#
#         # Run Predictor constructor
#         super().__init__(self.model_path, self.detection_conf)
#
#     def ensemble_detect(self, frame_numpy_array, metadata, models, model_kwargs=None):
#         """
#         Perform detection using an ensemble of models and aggregate results using weighted scoring.
#
#         Args:
#             frame_numpy_array (np.ndarray): Array of frames for detection.
#             metadata (dict): Metadata associated with the frames.
#             models (list): List of model paths for the ensemble.
#             model_kwargs (list): Additional arguments for the detect method for each model. List of kwargs dicts.
#
#         Returns:
#             Generator of aggregated DetectionResults.
#         """
#         model_kwargs = [{} for model in models] if model_kwargs is None else model_kwargs
#         results_by_model = {}
#         with ThreadPoolExecutor(max_workers=len(models)) as executor:
#             # Start a detection task for each model
#             future_to_model = {executor.submit(self.detect, frame_numpy_array, metadata, model_path=model, **kw): model
#                                for model, kw in zip(models, model_kwargs)}
#
#             # Collect results as they complete
#             for future in as_completed(future_to_model):
#                 model = future_to_model[future]
#                 try:
#                     results_by_model[model] = future.result()
#                 except Exception as exc:
#                     logging.error(f'Model {model} generated an exception: {exc}')
#
#         # Call the weighted scoring function (to be implemented)
#         return self._weighted_scoring_ensemble(results_by_model, frame_numpy_array, metadata)
#
#     def _weighted_scoring_ensemble(self, results_by_model, frame_numpy_array, metadata):
#         """
#         Aggregate detection results using weighted scoring.
#
#         Args:
#             results_by_model (dict): Dictionary of detection results keyed by model.
#             frame_numpy_array (np.ndarray): Array of frames for detection.
#             metadata (dict): Metadata associated with the frames.
#
#         Returns:
#             Generator of aggregated DetectionResults.
#         """
#         all_detections = self._gather_detections(results_by_model)
#         scored_detections = self._calculate_scores(all_detections)
#         merged_detections = self._merge_and_filter_detections(scored_detections)
#
#         for frame in frame_numpy_array:
#             # Generate DetectionResults for each frame
#             # Assuming DetectionResults can be initialized with a list of boxes
#             yield DetectionResults(boxes=merged_detections, orig_img=frame, **metadata)
#
#     def _gather_detections(self, results_by_model):
#         """
#         Gather all detections from each model.
#
#         Args:
#             results_by_model (dict): Dictionary of detection results keyed by model.
#
#         Returns:
#             List: All detections with model, box coordinates, and confidence score.
#         """
#         all_detections = []
#         for model, results in results_by_model.items():
#             for result in results:
#                 for box in result.boxes.data:
#                     all_detections.append((model, box[:4], box[4]))
#         return all_detections
#
#     def _calculate_scores(self, all_detections, iou_threshold=0.5):
#         """
#         Calculate scores for each detection.
#
#         Args:
#             all_detections (list): List of all detections.
#             iou_threshold (float): Threshold for IoU to consider boxes as overlapping.
#
#         Returns:
#             List: Detections with calculated scores.
#         """
#         scored_detections = []
#         for i, (model_i, box_i, conf_i) in enumerate(all_detections):
#             overlap_count = 1
#             total_conf = conf_i
#
#             for j, (model_j, box_j, conf_j) in enumerate(all_detections):
#                 if i != j and self._iou(box_i,
#                                         box_j) > iou_threshold:  # TODO: Implement or pass the BoxAnalyser IoU function
#                     overlap_count += 1
#                     total_conf += conf_j
#
#             score = total_conf / overlap_count
#             scored_detections.append((box_i, score))
#
#         return scored_detections
#
#     def _merge_and_filter_detections(self, scored_detections, score_threshold=0.5):
#         """
#         Merge overlapping detections and filter them based on score threshold.
#
#         Args:
#             scored_detections (list): List of detections with scores.
#             score_threshold (float): Threshold score to keep a detection.
#
#         Returns:
#             List: Filtered and merged detections.
#         """
#         # Sort detections by score in descending order
#         scored_detections.sort(key=lambda x: x[1], reverse=True)
#
#         merged_detections = []
#         while scored_detections:
#             # Take the detection with the highest score
#             current = scored_detections.pop(0)
#             box_current, score_current = current
#
#             # Only consider detections with a score above the threshold
#             if score_current < score_threshold:
#                 continue
#
#             # Check for overlap with remaining detections
#             non_overlapping = []
#             for other in scored_detections:
#                 box_other, _ = other
#                 if self._iou(box_current, box_other) < 0.5:
#                     non_overlapping.append(other)
#
#             # Update the list of detections to consider
#             scored_detections = non_overlapping
#
#             # Add the current detection to the merged list
#             merged_detections.append(current)
#
#         return merged_detections

from typing import List, Optional, Generator, Union
import numpy as np
import concurrent.futures
from detectflow.predict.predictor import Predictor
from detectflow.predict.results import DetectionResults, DetectionBoxes
from detectflow.predict.tracker import Tracker
import logging


class Ensembler:
    def __init__(self, predictors: List[Predictor], tracker: Optional[str] = None):
        """
        Ensembler class to run predictions using multiple models and combine the results.
        Args:
            predictors (List[Predictor]): List of Predictor instances to use for ensemble detection.
            tracker (Optional[str]): Tracker type to use for tracking merged results.
        """
        self.predictors = predictors
        self.tracker = Tracker(tracker_type=tracker) if tracker else None

#TODO: Sort kwrgs and sahi and yolo config before passing into predictors
    def detect(self,
               frame_numpy_array: Union[np.ndarray, List[np.ndarray]],
               metadata: dict = None,
               image_size: Optional[tuple] = None,
               sliced: bool = True,
               tracked: bool = False,
               filter_tracked: bool = False,
               save: bool = False,
               save_txt: bool = False,
               device: Optional[str] = None,
               yolo_config: Optional[dict] = {},
               sahi_config: Optional[dict] = {},
               **kwargs) -> Generator[DetectionResults, None, None]:
        """
        Run detection on the given frames using all predictors and combine the results.
        Args:
            frame_numpy_array (Union[np.ndarray, List[np.ndarray]]): The input frames as a numpy array (4D array for multiple frames, 3D array for a single frame, or list of 3D arrays).
            metadata (dict, optional): Metadata to pass to each predictor.
            image_size (tuple, optional): Size of the image for prediction.
            sliced (bool): Whether to slice the image for prediction.
            tracked (bool): Whether to apply tracking to the results.
            filter_tracked (bool): Whether to filter the tracked boxes.
            save (bool): Whether to save the results.
            save_txt (bool): Whether to save the results in txt format.
            device (str, optional): The device to run the model on.
            yolo_config (dict, optional): YOLO specific configuration.
            sahi_config (dict, optional): SAHI specific configuration.
            **kwargs: Additional arguments for the prediction.
        Yields:
            DetectionResults: Combined detection results from all models for each frame.
        """
        # Prepare input for detection
        if isinstance(frame_numpy_array, np.ndarray):
            if frame_numpy_array.ndim == 3:
                # Convert the 3D frame into a 4D array with only one frame
                frame_numpy_array = np.expand_dims(frame_numpy_array, axis=0)

            elif frame_numpy_array.ndim == 1:
                # Handle the case where a single image might be passed in improperly as a 1D array
                raise ValueError("Frames passed as a 1D array. Expected 3D or 4D array for proper processing.")

            # Flatten the 4D array into a list of 3D arrays
            list_of_frames = np.split(frame_numpy_array, frame_numpy_array.shape[0], axis=0)
            list_of_frames = [np.squeeze(frame) for frame in list_of_frames]  # Remove singleton dimension

        elif isinstance(frame_numpy_array, list) and all(isinstance(frame, np.ndarray) for frame in frame_numpy_array):
            list_of_frames = frame_numpy_array
        else:
            raise ValueError(f"Frames passed in an invalid format. Type: {type(frame_numpy_array)}")

        # Iterate over frames and run detection on each frame
        for i, frame in enumerate(list_of_frames):
            # Ensure frame is at least a 3D array
            if frame.ndim != 3:
                raise ValueError(f"Invalid frame dimension {frame.ndim}. Expected 3D array for each frame.")

            combined_boxes = []
            detection_result = None

            # Prepare metadata for the current frame
            frame_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)) and len(value) > i:
                        frame_metadata[key] = value[i]
                    else:
                        frame_metadata[key] = value

            # Run detection with each predictor concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_predictor = {
                    executor.submit(
                        self._run_predictor,
                        predictor,
                        frame,
                        frame_metadata,
                        image_size=image_size,
                        sliced=sliced,
                        save=save,
                        save_txt=save_txt,
                        device=device,
                        yolo_config=yolo_config,
                        sahi_config=sahi_config,
                        **kwargs
                    ): predictor for predictor in self.predictors
                }
                for future in concurrent.futures.as_completed(future_to_predictor):
                    result = future.result()
                    if result is not None:
                        detection_result = result
                        if result.boxes is not None:
                            combined_boxes.append(result.boxes)

            # Merge detection boxes
            try:
                merged_boxes = self._merge_boxes(combined_boxes, frame.shape[:2])
            except Exception as e:
                logging.error(f"Error merging boxes from ensembler models: {e}")
                merged_boxes = None

            # The detection result taken from the last predictor for metadata is updated with merged boxes
            detection_result.boxes = merged_boxes

            # Apply tracking if tracker is available
            if tracked and self.tracker:
                try:
                    detection_result = self.tracker.process_tracking(detection_result, filter=filter_tracked)
                except Exception as e:
                    logging.error(f"Error in tracking: {e}")

            yield detection_result

    def _run_predictor(self, predictor: Predictor, frame: np.ndarray, metadata: dict,
                       image_size: Optional[tuple] = None, sliced: bool = True,
                       save: bool = False, save_txt: bool = False, device: Optional[str] = None,
                       yolo_config: Optional[dict] = {}, sahi_config: Optional[dict] = {}, **kwargs) -> DetectionResults:
        """
        Run the predictor on a single frame and return the result.
        Args:
            predictor (Predictor): The predictor instance to run.
            frame (np.ndarray): The frame to run detection on.
            metadata (dict): Metadata to pass to the predictor.
            image_size (tuple, optional): Size of the image for prediction.
            sliced (bool): Whether to slice the image for prediction.
            save (bool): Whether to save the results.
            save_txt (bool): Whether to save the results in txt format.
            device (str, optional): The device to run the model on.
            yolo_config (dict, optional): YOLO specific configuration.
            sahi_config (dict, optional): SAHI specific configuration.
            **kwargs: Additional arguments for the prediction.
        Returns:
            DetectionResults: The detection results from the predictor.
        """
        try:
            results_generator = predictor.detect(
                frame,
                metadata=metadata,
                image_size=image_size,
                sliced=sliced,
                save=save,
                save_txt=save_txt,
                device=device,
                yolo_config=yolo_config,
                sahi_config=sahi_config,
                **kwargs
            )
            for result in results_generator:
                return result
        except Exception as e:
            logging.error(f"Error running predictor: {e}")

        return None

    def _merge_boxes(self, boxes_list: List[DetectionBoxes], orig_shape: tuple) -> DetectionBoxes:
        """
        Merge bounding boxes from different models based on IoU.
        Args:
            boxes_list (List[DetectionBoxes]): List of DetectionBoxes from different models.
            orig_shape (tuple): Original shape of the image (height, width).
        Returns:
            DetectionBoxes: Merged DetectionBoxes instance.
        """
        if not boxes_list:
            return None

        # Flatten list of DetectionBoxes into a single array of boxes
        all_boxes = np.vstack([boxes.data for boxes in boxes_list if boxes is not None])

        # If no boxes exist, return None
        if all_boxes.size == 0:
            return None

        # Sort by confidence in descending order
        all_boxes = all_boxes[all_boxes[:, 4].argsort()[::-1]]

        # Perform Non-Maximum Suppression (NMS) to merge overlapping boxes
        merged_boxes = []
        while len(all_boxes) > 0:
            # Take the box with the highest confidence
            current_box = all_boxes[0]
            merged_boxes.append(current_box)

            # Calculate IoU with the remaining boxes
            ious = self._calculate_iou(current_box, all_boxes)

            # Suppress boxes with IoU greater than a threshold (e.g., 0.5)
            all_boxes = all_boxes[ious <= 0.5]

        return DetectionBoxes(np.array(merged_boxes), orig_shape)

    @staticmethod
    def _calculate_iou(box, boxes):
        """
        Calculate Intersection over Union (IoU) between a box and a list of boxes.
        Args:
            box (np.ndarray): A single bounding box.
            boxes (np.ndarray): An array of bounding boxes to compare against.
        Returns:
            np.ndarray: Array of IoU values.
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate areas of each box
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Calculate union area
        union = box_area + boxes_area - intersection

        # Compute IoU
        iou = intersection / union
        return iou

