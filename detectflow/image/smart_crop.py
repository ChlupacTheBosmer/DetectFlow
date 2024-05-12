from typing import List, Optional, Tuple, Union
from detectflow.predict.results import DetectionResults, DetectionBoxes
import numpy as np
import logging
from detectflow.manipulators.box_manipulator import BoxManipulator
from detectflow.utils.inspector import Inspector
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation
from detectflow.manipulators.frame_manipulator import FrameManipulator
from detectflow.manipulators.box_analyser import BoxAnalyser

class CropResult:
    def __init__(self,
                 crops: List,
                 annotations: List):
        self.crops = crops
        self.annotations = annotations


class SmartCrop:
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 annotations: Optional[Union[DetectionBoxes, np.ndarray]] = None,
                 crop_size: Tuple = (640, 640),
                 handle_overflow: str = "expand",
                 max_expansion_limit: Tuple = (1000, 1000),
                 margin: int = 100,
                 exhaustive_search: bool = False,
                 permutation_limit: int = 7,
                 multiple_rois: bool = False):
        '''
        Initialize the instance of the SmartCrop when passing an image and annotations.
        '''

        # Assign the basic attributes
        self.image = image
        self.image_size = image.shape[:2][::-1]  # Reverse to make sure the order is (width, height)
        self.image_aspect_ratio = self.image_size[0] / self.image_size[1]
        self.annotations = annotations
        self.crop_size = crop_size
        self.crop_aspect_ratio = self.crop_size[0] / self.crop_size[1]
        self.handle_overflow = handle_overflow
        self.max_expansion_limit = max_expansion_limit
        self.margin = margin
        self.exhaustive_search = exhaustive_search
        self.permutation_limit = permutation_limit
        self.multiple_rois = multiple_rois

    @classmethod
    def from_detection_results(cls,
                               detection_results: Optional[DetectionResults],
                               crop_size: Tuple = (640, 640),
                               handle_overflow: str = "expand",
                               max_expansion_limit: Tuple = (1000, 1000),
                               margin: int = 100,
                               exhaustive_search: bool = False,
                               permutation_limit: int = 7,
                               multiple_rois: bool = False):
        '''
        Initialize the instance of the SmartCrop when passing DetectionResults instance.
        '''

        return cls(image=detection_results.orig_img,
                   annotations=detection_results.boxes,
                   crop_size=crop_size,
                   handle_overflow=handle_overflow,
                   max_expansion_limit=max_expansion_limit,
                   margin=margin,
                   exhaustive_search=exhaustive_search,
                   permutation_limit=permutation_limit,
                   multiple_rois=multiple_rois)

    def crop(self,
             inspect: bool = False,
             auto_resize: bool = False,
             ignore_empty: bool = False,
             partial_overlap: bool = False,
             iou_threshold: float = 0.5,
             crop_size: Optional[Tuple[int, int]] = None,
             handle_overflow: Optional[str] = None,
             max_expansion_limit: Optional[Tuple] = None,
             margin: Optional[int] = None,
             exhaustive_search: Optional[bool] = None,
             permutation_limit: Optional[int] = None,
             multiple_rois: Optional[bool] = None):
        '''
        Search for the best solution to crop image based on the requirements
        '''

        logging.info("Running crop()")

        # Crop size and multiple_rois can be specified manually overriding the one set while init-ing the class instance
        crop_size = self.crop_size if crop_size is None else crop_size
        handle_overflow = self.handle_overflow if handle_overflow is None else handle_overflow
        max_expansion_limit = self.max_expansion_limit if max_expansion_limit is None else max_expansion_limit
        margin = self.margin if margin is None else margin
        exhaustive_search = self.exhaustive_search if exhaustive_search is None else exhaustive_search
        permutation_limit = self.permutation_limit if permutation_limit is None else permutation_limit
        multiple_rois = self.multiple_rois if multiple_rois is None else multiple_rois

        # Calculate the roi(s) for cropping
        rois = BoxManipulator.calculate_optimal_roi(self.annotations,
                                                    self.image_size,
                                                    crop_size,
                                                    handle_overflow,
                                                    max_expansion_limit,
                                                    margin,
                                                    exhaustive_search,
                                                    permutation_limit,
                                                    multiple_rois,
                                                    ignore_empty,
                                                    partial_overlap,
                                                    iou_threshold)

        crops = []
        annotations = []
        adjusted_boxes = None

        if rois is not None:
            for roi, included_boxes in rois:
                if roi is not None:

                    # Crop frames
                    (crop_list, meta) = FrameManipulator.crop_frames(self.image, roi, offset_range=0)[0]

                    crop = crop_list[0]

                    # Adjust bboxes for the crop
                    if included_boxes is not None:
                        # logging.info(f"Original boxes: {self.annotations}")
                        adjusted_boxes = BoxManipulator._adjust_boxes(boxes=self.annotations,
                                                                      roi=roi,
                                                                      orig_shape=self.image_size)

                    # Display crops with adjusted bboxes
                    #                     if inspect:
                    #                         Inspector.display_frames_with_boxes([crop], [adjusted_boxes])

                    if crop.shape[::-1][1:] != crop_size and auto_resize:

                        # Rescale crop to requested dimensions and adjust the bboxes
                        orig_crop_size = crop.shape[::-1][1:]
                        crop = FrameManipulator.resize_frames(crop, crop_size)[0]
                        if adjusted_boxes is not None:
                            adjusted_boxes = BoxManipulator._adjust_boxes_for_resize(boxes=adjusted_boxes,
                                                                                     orig_shape=orig_crop_size,
                                                                                     new_shape=crop_size)

                    # Print result
                    # logging.info(f"Adjusted boxes: {adjusted_boxes}")

                    # Display crops with adjusted bboxes
                    if inspect:
                        Inspector.display_frames_with_boxes([crop], [adjusted_boxes])

                    # Append result lists
                    crops.append(crop)
                    annotations.append(adjusted_boxes)

        return CropResult(crops, annotations)

    def tile(self,
             overlap_height_ratio: Optional[float] = 0.2,
             overlap_width_ratio: Optional[float] = 0.2,
             min_area_ratio: float = 0.1,
             inspect: bool = False):
        '''
        Slice image into tiles
        '''

        logging.info("Running tile()")

        if self.image_size[0] < self.crop_size[0] or self.image_size[1] < self.crop_size[1]:

            # Will calculate the target size to which to resize the image to make sure at least one tile fits while maintaining aspect ratio
            target_size = FrameManipulator.calculate_target_adjust_image_size(self.image_size, self.crop_size)
            print(target_size)

            # Upscale frame and adjust annotations
            image = FrameManipulator.resize_frames([self.image], target_size, preference='balance')[0]
            print(image.shape[::-1][1:])
            orig_annotations = BoxManipulator._adjust_boxes_for_resize(self.annotations, self.image_size[::-1],
                                                                       image.shape[::-1][1:])
        else:
            image = self.image
            orig_annotations = self.annotations

        annotations = []

        # Convert each bounding box to a CocoAnnotation object
        if orig_annotations is not None:
            for bbox, cls in zip(orig_annotations.xyxy, orig_annotations.cls):
                #                 x_min, y_min, x_max, y_max = bbox
                #                 width, height = x_max - x_min, y_max - y_min
                #                 coco_bbox = [x_min, y_min, width, height]

                coco_bbox = BoxManipulator.xyxy_to_coco(bbox)

                annotation = CocoAnnotation.from_coco_bbox(
                    bbox=coco_bbox,
                    category_id=cls,
                    category_name=cls,
                    iscrowd=0
                )
                annotation.image_id = 1  # Setting the image ID
                annotations.append(annotation)

            # Example of what one annotation looks like
            logging.debug(annotations[0])
        else:
            annotations = None

        # Slice image
        results = slice_image(
            image=image,
            coco_annotation_list=annotations,
            slice_height=self.crop_size[1],
            slice_width=self.crop_size[0],
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            auto_slice_resolution=True,
            min_area_ratio=min_area_ratio,
            verbose=False,
        )

        # Process results
        crops = []
        sliced_annotations = []
        for sliced_image in results.sliced_image_list:

            # Append image
            crops.append(sliced_image.image)

            # Extract annotations from coco annotations
            annotations = [coco_annotation.bbox for coco_annotation in sliced_image.coco_image.annotations]
            logging.info(annotations)

            if len(annotations) > 0:
                # Create detection boxes for slice
                xyxy_annotations = BoxManipulator.coco_to_xyxy(annotations)
                detection_boxes = DetectionBoxes(np.array(xyxy_annotations), sliced_image.image.shape[:2], 'xyxy')
            else:
                detection_boxes = None

            # Display sliced frame if applicable
            if inspect:
                Inspector.display_frames_with_boxes([sliced_image.image], detection_boxes_list=[detection_boxes])

        return CropResult(crops, sliced_annotations)

    def rescale(self,
                ignore_aspect_ratio: bool = False,
                inspect: bool = False):
        '''
        Rescale the image to met the requirements
        '''

        logging.info("Running rescale()")

        if self.image_aspect_ratio == self.crop_aspect_ratio or ignore_aspect_ratio:

            # Get default values from instance attributes
            image = self.image
            orig_annotations = self.annotations

        else:

            # Subcrop to get the same aspect ratio as crop_size
            result = self.subcrop()
            image = result.crops[0]
            orig_annotations = result.annotations[0]

        # Upscale frame and adjust annotations
        crop = FrameManipulator.resize_frames([image], self.crop_size, preference='balance')[0]
        annotations = BoxManipulator._adjust_boxes_for_resize(orig_annotations, image.shape[::-1][1:], self.crop_size)

        if inspect:
            Inspector.display_frames_with_boxes([crop], detection_boxes_list=[annotations])

        return CropResult([crop], [annotations])

    def subcrop(self,
                inspect: bool = False):
        '''
        Subcrop the image to change the aspect ratio while losing the least amount of information.
        '''

        logging.info("Running subcrop()")

        # Calculate subscrop dimensions
        roi_size = FrameManipulator.calculate_largest_roi(self.image_size, self.crop_aspect_ratio)

        # Crop the image with the crop_size set to the custom dimensions
        return self.crop(inspect=inspect,
                         auto_resize=True,
                         ignore_empty=False,
                         partial_overlap=True,
                         iou_threshold=0.3,
                         crop_size=roi_size,
                         handle_overflow="strict",
                         margin=0,
                         exhaustive_search=True,
                         permutation_limit=5,
                         multiple_rois=False)

    def smart_crop(self,
                   partial_overlap: bool = False,
                   iou_threshold: float = 0.5,
                   allow_slicing: bool = True,
                   eveness_threshold: float = 0.66,
                   force_slice_empty: bool = True,
                   inspect: bool = False):
        '''
        Analyse the image and annotations and chose the best option of cropping among crop, tiles, rescale methods
        '''

        logging.info("Running smart_crop()")

        # Prepare conditions

        # Image conditions
        same_aspect_ratio = self.image_aspect_ratio == self.crop_aspect_ratio  # is img and crop AR the same?
        all_img_dims_in_expansion_limit = all(img_dim <= limit for img_dim, limit in zip(self.image_size,
                                                                                         self.max_expansion_limit))  # are all img dims smaller than their corresponding expansion limit of crop?
        any_img_dim_large = any(img_dim > 2 * crop_dim for img_dim, crop_dim in zip(self.image_size,
                                                                                    self.crop_size))  # is any img dim larger than twice the corresponding crop dim?
        all_img_dims_over_crop_size = all(img_dim > 1.3 * crop_dim for img_dim, crop_dim in zip(self.image_size,
                                                                                                self.crop_size))  # are all img dims 1.3 times larger than corresponding crop dims?

        # Boxes conditions
        empty = True if self.annotations is None else False

        grid_dim = max(min(self.image_size) // 2, min(self.crop_size) // 2)  # grid dims based on img or crop dims
        box_dist_idx = 0 if empty else BoxAnalyser.analyze_bbox_distribution(self.annotations.xyxy, self.image_size,
                                                                             grid_size=(grid_dim,
                                                                                        grid_dim))  # analyse distribution of bboxes
        even_dist = box_dist_idx >= eveness_threshold  # are boxes distributed in more than set proportion of grid cells?

        box_size_limit = self.max_expansion_limit if self.handle_overflow == "expand" else self.crop_size
        any_large_box = False if empty else any(box_dim >= crop_dim for box_dim, crop_dim in
                                                [(BoxAnalyser.box_dimensions(box), box_size_limit) for box in
                                                 self.annotations.xyxy])  # is any box higher or wider than the crop size?

        # If empty and force slice empties is True then slice
        if empty and force_slice_empty:
            logging.info("Slicing empty image")

            return self.tile(overlap_height_ratio=0.2, overlap_width_ratio=0.2, min_area_ratio=0.1, inspect=inspect)

        # If the image has the same aspect ratio as the desired crop and the expansion limit is not exceeded
        if same_aspect_ratio and all_img_dims_in_expansion_limit:

            logging.info("Rescaling image")

            # Then rescale the image
            return self.rescale(inspect=inspect)

        # If any bbox is larger than the crop size and cannot be fitted in a crop
        elif any_large_box:

            logging.info("Subcroping image")

            # Then subcrop the image and resize
            # TODO: note that this will result in subcroping if any box even just one of many is larger than crop
            # but it does take into account whether the expansion policy is set to strict or expand.
            return self.subcrop(inspect=inspect)

        # If any img dim is twice as large as the crop size and both img dims are larger than crop_size
        elif allow_slicing and any_img_dim_large and all_img_dims_over_crop_size and even_dist:

            logging.info("Slicing image")

            # Then image is considered large with even distribution of boxes and may be suitable for slicing
            return self.tile(overlap_height_ratio=0.2, overlap_width_ratio=0.2, min_area_ratio=0.1, inspect=inspect)

        # Else
        else:

            logging.info("Cropping image")

            # Then perform standard crop
            return self.crop(inspect=inspect,
                             auto_resize=True,
                             ignore_empty=False,
                             partial_overlap=partial_overlap,
                             iou_threshold=iou_threshold)