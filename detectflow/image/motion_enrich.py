import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from detectflow.validators.validator import Validator
from detectflow.validators.video_validator import VideoValidator
from detectflow.manipulators.frame_manipulator import FrameManipulator
import logging


@dataclass
class MotionEnrichResult:
    def __init__(self,
                 rgb_frame: np.ndarray,
                 grey_frame: Optional[np.ndarray] = None,
                 frame_number: Optional[int] = None,
                 roi: Optional[Union[Tuple, List]] = None
                 ):
        '''
        Args:
        - frame - Motion enriched frame
        - greyscale_frame - Greyscale version of motion enriched frame with colorful mask
        - video_path - Path to original video file
        - frame_number
        - roi
        '''
        self.rgb_frame = rgb_frame
        self.grey_frame = grey_frame
        self.frame_number = frame_number
        self.roi = roi


class MotionEnrich:
    def __init__(self,
                 video_path: Optional[str] = None,
                 metadata: Optional[Union[Tuple, List]] = None,
                 buffer_size: int = 60,
                 preload_frames: int = 200,
                 backSub_history: int = 100,
                 cluster_threshold: int = 100,
                 alpha: float = 0.4,
                 frame_skip: int = 1,
                 method: str = 'imageio',
                 save_video: bool = False,
                 save_dir: Optional[str] = None,
                 video_type: str = "color"):

        '''
        Metadata format: [(frame_no, (x1, y1, x2, y2))]
        '''
        # Video data
        self.video_path = video_path
        self.metadata = metadata
        self.method = method
        self.save_video = save_video
        self.save_dir = save_dir
        self.video_type = video_type

        # Config
        self.buffer_size = buffer_size
        self.preload_frames = preload_frames
        self.backSub_history = backSub_history
        self.cluster_threshold = cluster_threshold
        self.alpha = alpha
        self.frame_skip = frame_skip

        # Hidden config
        self.kernel_small = np.ones((2, 2), np.uint8)
        self.kernel_large = np.ones((5, 5), np.uint8)
        self.kernel_expanded = np.ones((13, 13), np.uint8)

        # Func attributed
        self._backSub = cv2.createBackgroundSubtractorMOG2(history=self.backSub_history)
        self._frame_buffer = []
        self._prev_gray = None
        self._buffer_clean = None

    # @profile_memory(logging.getLogger(__name__))
    def run(self,
            video_path: Optional[str] = None,
            metadata: Optional[Union[Tuple, List]] = None,
            save_video: bool = False,
            save_dir: Optional[str] = None,
            video_type: str = "color"):

        # Assign attributes
        self.video_path = video_path if video_path is not None else self.video_path
        self.metadata = metadata if metadata is not None else self.metadata
        self.save_video = save_video
        self.save_dir = save_dir if save_dir is not None else self.save_dir
        self.video_type = video_type if video_type is not None else self.video_type
        video_writer = None

        # Check metadata format
        if not (self.metadata and len(self.metadata) > 0 and all([isinstance(item[0], int) for item in self.metadata])):
            raise ValueError(f"Invalid metadata format, frame numbers not found: {self.metadata[0]}")

        # Create video file
        if not self.video_path:
            raise ValueError(f"No video_path supplied.")

        if not Validator.is_valid_file_path(self.video_path):
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        if not VideoValidator(self.video_path).validate_video():
            raise RuntimeError(f"Video file could not be validated")

        video = VideoFilePassive(self.video_path)

        # Find clusters of frames
        clusters = self._find_clusters(self.metadata, self.cluster_threshold)

        # Process each cluster
        for cluster in clusters:
            frame_numbers = [tup[0] for tup in cluster]  # Extract frame numbers from each tuple in the cluster
            # export_motion_highlighted_frames(valid[0], frame_numbers, [empty_folder, visitor_folder, bw_empty_folder, bw_visitor_folder], cluster, [frames_db, bw_frames_db])

            current_cluster_start = max(min(frame_numbers) - (self.preload_frames * self.frame_skip),
                                        0)  # Start 200 frames before the first target frame or from frame 0

            # Will get frame numbers to be retrieved and analysed
            frame_numbers_set = set(frame_numbers)
            frames = sorted(frame_numbers + [frame_number for frame_number in
                                             range(current_cluster_start, max(frame_numbers) + 1, self.frame_skip) if
                                             frame_number not in frame_numbers_set])

            # For each frame in the generator will perform the analysis
            counter = 0
            for result_list in video.read_video_frame(frames, True, "decord"):
                frame = result_list[3] if len(result_list) >= 4 else None
                frame_number = result_list[2] if len(result_list) >= 3 else None
                if frame is None or frame_number is None:
                    break

                # Process frame
                fgMask, gray = self.process(frame)

                # Enrich frame
                if frame_number in frame_numbers:
                    motion_enriched_frame, bw_motion_enriched_frame = self.enrich(frame, gray, fgMask)

                    # Get index position of the metadata for the current frame number in cluster metadata subset
                    idx_position = counter
                    counter += 1

                    # Crop frame
                    try:
                        crop = self.crop_frame(motion_enriched_frame, cluster[idx_position])
                        bw_crop = self.crop_frame(bw_motion_enriched_frame, cluster[idx_position])
                    except ValueError as ve:
                        logging.error(f"Value error when cropping the frames: {ve}")
                        crop, bw_crop = motion_enriched_frame, bw_motion_enriched_frame
                    except RuntimeError as re:
                        logging.error(f"Runtime error when cropping the frames: {re}")
                        crop, bw_crop = motion_enriched_frame, bw_motion_enriched_frame
                    except Exception as e:
                        raise RuntimeError(f"Unexpected error when cropping frames. Terminating.") from e

                    # Update OF history
                    self._prev_gray = gray

                    # Crate video
                    if self.save_video:
                        if video_writer is None:
                            height, width, _ = crop.shape
                            video_writer = self._create_videowriter((width, height), 30)
                        else:
                            img = crop if self.video_type == "color" or self.video_type == "colour" else bw_crop
                            video_writer.write(img)
                            if frame_number == frames[-1]:
                                video_writer.release()

                    yield MotionEnrichResult(crop, bw_crop, frame_number, self.metadata[idx_position][1])

            # Clear buffer
            self._frame_buffer.clear()

    def _create_videowriter(self, frame_size, fps):

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Define the codec (platform dependent)
        video_out_path = os.path.join(self.save_dir,
                                      f"{os.path.splitext(os.path.basename(self.video_path))[0]}_enriched_{self.video_type}.mp4")
        return cv2.VideoWriter(video_out_path, fourcc, fps, frame_size)

    # @profile_memory(logging.getLogger(__name__))
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_gray = gray if self._prev_gray is None else self._prev_gray
        fgMask = self._backSub.apply(frame)

        self._frame_buffer.append(fgMask)
        if len(self._frame_buffer) > self.buffer_size or (self._buffer_clean and self._buffer_clean > 0):
            self._frame_buffer.pop(0)
        self._buffer_clean = self._buffer_clean - 1 if self._buffer_clean is not None else 3

        return fgMask, gray

    # @profile_memory(logging.getLogger(__name__))
    def enrich(self, frame, gray, fgMask):

        # Process the current frame with long interval difference
        first_mask = self._frame_buffer[0]
        _, thresholded_fgMask = cv2.threshold(fgMask - first_mask, 250, 255, cv2.THRESH_BINARY)

        # Compute the optical flow
        flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize magnitude to 0-255 scale for use as an alpha mask
        try:
            normalized_magnitude = np.uint8(255 * (magnitude / (magnitude.max() if magnitude.max() != 0 else 1)))
        except Exception as e:
            normalized_magnitude = magnitude

        # Dilate the foreground mask to create an expanded area
        expanded_fgMask = cv2.dilate(fgMask, self.kernel_expanded, iterations=1)
        fgMask = self.refine_mask(fgMask)

        # Combine the optical flow magnitude with the expanded foreground mask
        enhanced_mask = cv2.bitwise_and(normalized_magnitude, expanded_fgMask.astype(np.uint8))
        combined_mask = cv2.bitwise_or(enhanced_mask, fgMask.astype(np.uint8))

        # Create a color mask (turquoise)
        color_mask = cv2.merge([combined_mask, combined_mask * 0, combined_mask])

        # Blend the color mask with the frame and create a greyscale version of the frame
        motion_enriched_frame = cv2.addWeighted(frame, 1, color_mask, self.alpha, 0)
        bw_motion_enriched_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bw_motion_enriched_frame = cv2.addWeighted(bw_motion_enriched_frame, 1, color_mask, self.alpha, 0)

        return motion_enriched_frame, bw_motion_enriched_frame

    def refine_mask(self, mask):
        mask = cv2.erode(mask, self.kernel_small, iterations=2)
        mask = cv2.dilate(mask, self.kernel_large, iterations=1)
        return mask

    def crop_frame(self, frame, metadata):

        if len(metadata) == 2:
            _, roi = metadata
            x1, y1, x2, y2 = roi
        else:
            raise ValueError("Invalid metadata format passed. No ROI information.")

        # Crop frames
        crop_list = FrameManipulator.crop_frame(frame, rois=(x1, y1, x2, y2))
        crop = crop_list[0][0] if crop_list and len(crop_list) > 0 and len(crop_list[0]) > 0 else None

        if not isinstance(crop, np.ndarray):
            raise RuntimeError(f"No frame crops were extracted. Type: {type(frame)}")

        return crop

    def _find_clusters(self, metadata, threshold=100):
        """
        Finds clusters of tuples where frame numbers are separated by no more than `threshold` frames.
        The frame number is assumed to be at index 0 in each tuple.
        """
        if not metadata:
            return []

        # Sort tuples by the frame number
        sorted_metadata = sorted(metadata, key=lambda x: x[0])
        clusters = [[sorted_metadata[0]]]

        for current_tuple in sorted_metadata[1:]:
            last_frame_in_cluster = clusters[-1][-1][0]
            if current_tuple[0] - last_frame_in_cluster <= threshold:
                clusters[-1].append(current_tuple)
            else:
                clusters.append([current_tuple])

        return clusters