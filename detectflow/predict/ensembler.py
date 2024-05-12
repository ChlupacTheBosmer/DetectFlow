from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
from detectflow.predict.results import DetectionResults
from detectflow.predict.predictor import Predictor
import os


class Ensembler(Predictor): #TODO: Check and finish while implementing support for multiple models such as motion enriched etc.

    def __init__(self,
                 model_path: str = os.path.join('resources', 'yolo', 'best.pt'),
                 detection_conf_threshold: float = 0.5):

        # Assign attributes
        self.model_path = model_path
        self.detection_conf = detection_conf_threshold

        # Run Predictor constructor
        super().__init__(self.model_path, self.detection_conf)

    def ensemble_detect(self, frame_numpy_array, metadata, models, model_kwargs=None):
        """
        Perform detection using an ensemble of models and aggregate results using weighted scoring.

        Args:
            frame_numpy_array (np.ndarray): Array of frames for detection.
            metadata (dict): Metadata associated with the frames.
            models (list): List of model paths for the ensemble.
            model_kwargs (list): Additional arguments for the detect method for each model. List of kwargs dicts.

        Returns:
            Generator of aggregated DetectionResults.
        """
        model_kwargs = [{} for model in models] if model_kwargs is None else model_kwargs
        results_by_model = {}
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Start a detection task for each model
            future_to_model = {executor.submit(self.detect, frame_numpy_array, metadata, model_path=model, **kw): model
                               for model, kw in zip(models, model_kwargs)}

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results_by_model[model] = future.result()
                except Exception as exc:
                    logging.error(f'Model {model} generated an exception: {exc}')

        # Call the weighted scoring function (to be implemented)
        return self._weighted_scoring_ensemble(results_by_model, frame_numpy_array, metadata)

    def _weighted_scoring_ensemble(self, results_by_model, frame_numpy_array, metadata):
        """
        Aggregate detection results using weighted scoring.

        Args:
            results_by_model (dict): Dictionary of detection results keyed by model.
            frame_numpy_array (np.ndarray): Array of frames for detection.
            metadata (dict): Metadata associated with the frames.

        Returns:
            Generator of aggregated DetectionResults.
        """
        all_detections = self._gather_detections(results_by_model)
        scored_detections = self._calculate_scores(all_detections)
        merged_detections = self._merge_and_filter_detections(scored_detections)

        for frame in frame_numpy_array:
            # Generate DetectionResults for each frame
            # Assuming DetectionResults can be initialized with a list of boxes
            yield DetectionResults(boxes=merged_detections, orig_img=frame, **metadata)

    def _gather_detections(self, results_by_model):
        """
        Gather all detections from each model.

        Args:
            results_by_model (dict): Dictionary of detection results keyed by model.

        Returns:
            List: All detections with model, box coordinates, and confidence score.
        """
        all_detections = []
        for model, results in results_by_model.items():
            for result in results:
                for box in result.boxes.data:
                    all_detections.append((model, box[:4], box[4]))
        return all_detections

    def _calculate_scores(self, all_detections, iou_threshold=0.5):
        """
        Calculate scores for each detection.

        Args:
            all_detections (list): List of all detections.
            iou_threshold (float): Threshold for IoU to consider boxes as overlapping.

        Returns:
            List: Detections with calculated scores.
        """
        scored_detections = []
        for i, (model_i, box_i, conf_i) in enumerate(all_detections):
            overlap_count = 1
            total_conf = conf_i

            for j, (model_j, box_j, conf_j) in enumerate(all_detections):
                if i != j and self._iou(box_i,
                                        box_j) > iou_threshold:  # TODO: Implement or pass the BoxAnalyser IoU function
                    overlap_count += 1
                    total_conf += conf_j

            score = total_conf / overlap_count
            scored_detections.append((box_i, score))

        return scored_detections

    def _merge_and_filter_detections(self, scored_detections, score_threshold=0.5):
        """
        Merge overlapping detections and filter them based on score threshold.

        Args:
            scored_detections (list): List of detections with scores.
            score_threshold (float): Threshold score to keep a detection.

        Returns:
            List: Filtered and merged detections.
        """
        # Sort detections by score in descending order
        scored_detections.sort(key=lambda x: x[1], reverse=True)

        merged_detections = []
        while scored_detections:
            # Take the detection with the highest score
            current = scored_detections.pop(0)
            box_current, score_current = current

            # Only consider detections with a score above the threshold
            if score_current < score_threshold:
                continue

            # Check for overlap with remaining detections
            non_overlapping = []
            for other in scored_detections:
                box_other, _ = other
                if self._iou(box_current, box_other) < 0.5:
                    non_overlapping.append(other)

            # Update the list of detections to consider
            scored_detections = non_overlapping

            # Add the current detection to the merged list
            merged_detections.append(current)

        return merged_detections