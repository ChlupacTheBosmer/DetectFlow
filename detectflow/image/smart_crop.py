from typing import List, Optional, Tuple, Union
from detectflow.predict.results import DetectionResults, DetectionBoxes
import numpy as np
import logging
from detectflow.utils.inspector import Inspector
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation
from detectflow.manipulators.frame_manipulator import FrameManipulator
from detectflow.manipulators.box_manipulator import BoxManipulator
import os
import glob
from PIL import Image
import traceback
from detectflow.handlers.checkpoint_handler import CheckpointHandler
from detectflow.utils.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from detectflow.utils.file import yolo_label_load

class CropResult:
    def __init__(self,
                 crops: List,
                 annotations: List):
        self.crops = crops
        self.annotations = annotations


class SmartCrop:
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 annotations: Optional[DetectionBoxes] = None):
        '''
        Initialize the instance of the SmartCrop when passing an image and annotations.
        '''

        # Assign the basic attributes
        self.image = image
        self.image_size = image.shape[:2][::-1]  # Reverse to make sure the order is (width, height)
        self.image_aspect_ratio = self.image_size[0] / self.image_size[1]
        self.annotations = annotations
        self.crop_aspect_ratio = None

    @classmethod
    def from_detection_results(cls, detection_results: Optional[DetectionResults]):
        '''
        Initialize the instance of the SmartCrop when passing DetectionResults instance.
        '''

        return cls(image=detection_results.orig_img,
                   annotations=detection_results.boxes)

    def crop(self,
             inspect: bool = False,
             auto_resize: bool = False,
             ignore_empty: bool = False,
             partial_overlap: bool = False,
             iou_threshold: float = 0.5,
             crop_size: Tuple[int, int] = (640, 640),
             handle_overflow: str = "expand",
             max_expansion_limit: Tuple = (1000, 1000),
             margin: int = 50,
             exhaustive_search: bool = False,
             permutation_limit: int = 7,
             multiple_rois: bool = False):
        '''
        Search for the best solution to crop image based on the requirements
        '''

        logging.info("Running crop()")

        self.crop_aspect_ratio = crop_size[0] / crop_size[1]

        # Calculate the roi(s) for cropping
        rois = BoxManipulator.get_optimal_roi(self.annotations,
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
                        adjusted_boxes = BoxManipulator.adjust_boxes_to_roi(boxes=self.annotations, roi=roi)

                    # Display crops with adjusted bboxes
                    #                     if inspect:
                    #                         Inspector.display_frames_with_boxes([crop], [adjusted_boxes])

                    if crop.shape[::-1][1:] != crop_size and auto_resize:

                        # Rescale crop to requested dimensions and adjust the bboxes
                        orig_crop_size = crop.shape[::-1][1:]
                        crop = FrameManipulator.resize_frames(crop, crop_size)[0]
                        if adjusted_boxes is not None:
                            adjusted_boxes = BoxManipulator.adjust_boxes_for_resize(boxes=adjusted_boxes,
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
             crop_size: Tuple[int, int] = (640, 640),
             overlap_height_ratio: Optional[float] = 0.2,
             overlap_width_ratio: Optional[float] = 0.2,
             min_area_ratio: float = 0.1,
             inspect: bool = False):
        '''
        Slice image into tiles
        '''

        logging.info("Running tile()")

        if self.image_size[0] < crop_size[0] or self.image_size[1] < crop_size[1]:

            # Will calculate the target size to which to resize the image to make sure at least one tile fits while maintaining aspect ratio
            target_size = FrameManipulator.calculate_target_adjust_image_size(self.image_size, crop_size)
            print(target_size)

            # Upscale frame and adjust annotations
            image = FrameManipulator.resize_frames([self.image], target_size, preference='balance')[0]
            print(image.shape[::-1][1:])
            orig_annotations = BoxManipulator.adjust_boxes_for_resize(self.annotations, self.image_size[::-1],
                                                                      image.shape[::-1][1:])
        else:
            image = self.image
            orig_annotations = self.annotations

        annotations = []

        # Convert each bounding box to a CocoAnnotation object
        if orig_annotations is not None:
            orig_classes = orig_annotations.cls if orig_annotations.cls is not None else [0 for _ in orig_annotations]
            orig_confs = orig_annotations.conf if orig_annotations.conf is not None else [1.0 for _ in orig_annotations]
            for bbox, cls, conf in zip(orig_annotations.xyxy, orig_classes, orig_confs):
                #                 x_min, y_min, x_max, y_max = bbox
                #                 width, height = x_max - x_min, y_max - y_min
                #                 coco_bbox = [x_min, y_min, width, height]

                coco_bbox = BoxManipulator.xyxy_to_coco(bbox)

                annotation = CocoAnnotation.from_coco_bbox(
                    bbox=coco_bbox,
                    category_id=cls,
                    category_name=conf,
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
            slice_height=crop_size[1],
            slice_width=crop_size[0],
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
            confs = [float(coco_annotation.category_name) for coco_annotation in sliced_image.coco_image.annotations]
            classes = [int(coco_annotation.category_id) for coco_annotation in sliced_image.coco_image.annotations]

            # Stack the lists as columns into a 2D array
            data = np.column_stack((annotations, confs, classes))

            logging.debug(annotations)

            if len(data) > 0:
                # Create detection boxes for slice
                xyxy_data = BoxManipulator.coco_to_xyxy(data)
                detection_boxes = DetectionBoxes(np.array(xyxy_data), sliced_image.image.shape[:2])
            else:
                detection_boxes = None

            # Append annotations
            sliced_annotations.append(detection_boxes)

            # Display sliced frame if applicable
            if inspect:
                Inspector.display_frames_with_boxes([sliced_image.image], detection_boxes_list=[detection_boxes])

        return CropResult(crops, sliced_annotations)

    def rescale(self,
                crop_size: Tuple[int, int] = (640, 640),
                ignore_aspect_ratio: bool = False,
                inspect: bool = False):
        '''
        Rescale the image to meet the requirements
        '''

        logging.info("Running rescale()")
        self.crop_aspect_ratio = crop_size[0] / crop_size[1]

        if self.image_aspect_ratio == self.crop_aspect_ratio or ignore_aspect_ratio:

            # Get default values from instance attributes
            image = self.image
            orig_annotations = self.annotations

        else:

            # Subcrop to get the same aspect ratio as crop_size
            result = self.subcrop(self.crop_aspect_ratio)
            image = result.crops[0]
            orig_annotations = result.annotations[0]

        # Upscale frame and adjust annotations
        crop = FrameManipulator.resize_frames([image], crop_size, preference='balance')[0]
        annotations = BoxManipulator.adjust_boxes_for_resize(orig_annotations, image.shape[::-1][1:], crop_size)

        if inspect:
            Inspector.display_frames_with_boxes([crop], detection_boxes_list=[annotations])

        return CropResult([crop], [annotations])

    def adjust(self, crop_size: Tuple[int, int], inspect: bool = False):
        """
        Adjust the image to fit the crop size while maintaining the aspect ratio.

        Args:
            crop_size (Tuple[int, int]): The target crop size.
            inspect (bool): If True, display the adjusted image.

        Returns:
            Tuple[np.ndarray, Optional[DetectionBoxes], Tuple[int, int]]: The adjusted image, annotations, and new image size.
        """
        logging.info("Running adjust()")
        try:
            factor = max(self.image_size[0] / crop_size[0], self.image_size[1] / crop_size[1])
            if factor > 1:
                target_size = (int(self.image_size[0] * factor), int(self.image_size[1] * factor))
                resized_image = FrameManipulator.resize_frames(self.image, target_size)[0]
                if resized_image is not None:
                    new_image_size = resized_image.shape[:2][::-1]
                    if self.annotations is not None:
                        adjusted_annotations = BoxManipulator.adjust_boxes_for_resize(self.annotations,
                                                                                      orig_shape=self.image_size,
                                                                                      new_shape=target_size)
                    else:
                        adjusted_annotations = None
                else:
                    raise RuntimeError("Failed to resize image")

                if inspect:
                    Inspector.display_frames_with_boxes([resized_image], [adjusted_annotations])

                return resized_image, adjusted_annotations, new_image_size
        except Exception as e:
            logging.error(f"Failed to adjust image: {e}")
            return None, None, None


    def subcrop(self,
                aspect_ratio: Union[int, float] = 1,
                inspect: bool = False):
        '''
        Subcrop the image to change the aspect ratio while losing the least amount of information.
        '''

        logging.info("Running subcrop()")

        # Calculate subscrop dimensions
        roi_size = FrameManipulator.calculate_largest_roi(self.image_size, aspect_ratio)

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
                   crop_size: Tuple[int, int] = (640, 640),
                   handle_overflow: str = "expand",
                   max_expansion_limit: Tuple = (1000, 1000),
                   margin: int = 50,
                   partial_overlap: bool = False,
                   iou_threshold: float = 0.5,
                   exhaustive_search: bool = False,
                   permutation_limit: int = 7,
                   multiple_rois: bool = False,
                   allow_slicing: bool = True,
                   evenness_threshold: float = 0.66,
                   crop_size_multiplier: float = 2.0,
                   overlap_height_ratio: float = 0.2,
                   overlap_width_ratio: float = 0.2,
                   min_area_ratio: float = 0.1,
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
                                                                                         max_expansion_limit))  # are all img dims smaller than their corresponding expansion limit of crop?
        any_img_dim_large = any(img_dim > crop_size_multiplier * crop_dim for img_dim, crop_dim in zip(self.image_size,
                                                                                    crop_size))  # is any img dim larger than twice the corresponding crop dim?
        all_img_dims_over_crop_size = all(img_dim > 1.3 * crop_dim for img_dim, crop_dim in zip(self.image_size,
                                                                                                crop_size))  # are all img dims 1.3 times larger than corresponding crop dims?

        # Boxes conditions
        empty = True if not isinstance(self.annotations, DetectionBoxes) else False

        grid_dim = (3, 2) if self.image_aspect_ratio > 1 else (2, 3)  # grid dims based on img or crop dims
        box_dist_idx = 0 if empty else BoxManipulator.get_boxes_distribution_index(self.annotations.xyxy,
                                                                                   self.image_size, grid_size=grid_dim)  # analyse distribution of bboxes
        even_dist = box_dist_idx >= evenness_threshold  # are boxes distributed in more than set proportion of grid cells?

        box_size_limit = max_expansion_limit if handle_overflow == "expand" else crop_size
        any_large_box = False if empty else any(box_dim >= crop_dim for box_dim, crop_dim in
                                                [(BoxManipulator.get_box_dimensions(box), box_size_limit) for box in
                                                 self.annotations.xyxy])  # is any box higher or wider than the crop size?

        # If empty and force slice empties is True then slice
        if empty and force_slice_empty:
            logging.info("Slicing empty image")

            return self.tile(crop_size=crop_size,
                             overlap_height_ratio=overlap_height_ratio,
                             overlap_width_ratio=overlap_width_ratio,
                             min_area_ratio=min_area_ratio,
                             inspect=inspect)

        # If the image has the same aspect ratio as the desired crop and the expansion limit is not exceeded
        if same_aspect_ratio and all_img_dims_in_expansion_limit:

            logging.info("Rescaling image")

            # Then rescale the image
            return self.rescale(crop_size=crop_size, inspect=inspect)

        # If any bbox is larger than the crop size and cannot be fitted in a crop
        elif any_large_box:

            logging.info("Subcroping image")

            # Then subcrop the image and resize
            # TODO: note that this will result in subcroping if any box even just one of many is larger than crop
            # but it does take into account whether the expansion policy is set to strict or expand.
            return self.subcrop(aspect_ratio=self.crop_aspect_ratio, inspect=inspect)

        # If any img dim is twice as large as the crop size and both img dims are larger than crop_size
        elif allow_slicing and any_img_dim_large and all_img_dims_over_crop_size and even_dist:

            logging.info("Slicing image")

            # Then image is considered large with even distribution of boxes and may be suitable for slicing
            return self.tile(crop_size=crop_size,
                             overlap_height_ratio=0.2,
                             overlap_width_ratio=0.2,
                             min_area_ratio=0.1,
                             inspect=inspect)

        # Else
        else:

            logging.info("Cropping image")

            # Then perform standard crop
            result = self.crop(inspect=inspect,
                               auto_resize=True,
                               ignore_empty=False,
                               partial_overlap=partial_overlap,
                               iou_threshold=iou_threshold,
                               margin=margin,
                               exhaustive_search=exhaustive_search,
                               permutation_limit=permutation_limit,
                               multiple_rois=multiple_rois)

            if result.crops is None or len(result.crops) == 0 and allow_slicing:
                logging.warning("Failed to construct valid crop of the image. Slicing instead.")

                new_image, new_annotations, new_image_size = self.adjust(crop_size=crop_size, inspect=inspect)
                if new_image is not None:
                    self.image = new_image
                    self.annotations = new_annotations
                    self.image_size = new_image_size

                    return self.tile(crop_size=crop_size,
                                     overlap_height_ratio=overlap_height_ratio,
                                     overlap_width_ratio=overlap_width_ratio,
                                     min_area_ratio=min_area_ratio,
                                     inspect=inspect)
            else:
                return result


class FrameCropper(CheckpointHandler):

    CONFIG_DEF = {'handle_overflow': "expand",
                   'max_expansion_limit': (900, 900),
                   'margin': 25,
                   'exhaustive_search': True,
                   'permutation_limit': 8,
                   'multiple_rois': True,
                   'partial_overlap': False,
                   'iou_threshold': 0.8,
                   'allow_slicing': True,
                   'evenness_threshold': 0.66,
                   'force_slice_empty': True,
                   'inspect': False}

    def __init__(self, checkpoint_file='checkpoint.json', **kwargs):
        super().__init__(checkpoint_file)

        self.config = {}
        for key, value in kwargs.items():
            if key not in ['crop_size']:
                self.config[key] = value

        for key, value in self.CONFIG_DEF.items():
            if key not in self.config:
                self.config[key] = value

    def process_image(self, image_file: str, output_folder: str, crop_size: Tuple[int, int], extension_in: str = '.png', extension_out: str = '.jpg'):
        """
        Process a single image file, crop it based on the bounding boxes from the corresponding .txt file,
        and save the results to the output folder.
        """
        logging.info(f"Processing: {image_file}")

        # Image
        try:
            image = np.array(Image.open(image_file))
            image_shape = image.shape[:2]
        except Exception as e:
            logging.error(f"Failed to load image: {image_file}")
            return

        # Label
        txt_file = None
        try:
            txt_file = image_file.replace(extension_in, '.txt')
            if os.path.exists(txt_file):
                boxes = yolo_label_load(txt_file)
                detection_boxes = DetectionBoxes.from_custom_format(boxes[:, 1:], tuple(image_shape), "nxywh") if boxes is not None else None
            else:
                detection_boxes = None
        except Exception as e:
            logging.error(f"Failed to load label: {txt_file}")
            return

        #SmartCrop
        try:
            smart_crop = SmartCrop(image=image,
                                   annotations=detection_boxes)
            crop_result = smart_crop.smart_crop(
                                   crop_size=crop_size,
                                   **self.config)

            if not crop_result:
                logging.warning(f"No crop result created for {image_file}")
            else:
                logging.info(f"Created {len(crop_result.crops)} crops for {image_file}")

                for idx, crop in enumerate(crop_result.crops):

                    # Take boxes and add dummy cls and probs
                    boxes = crop_result.annotations[idx]
                    if isinstance(boxes, DetectionBoxes):
                        updated_boxes = np.hstack((boxes.data, np.zeros((boxes.data.shape[0], 2))))
                        boxes.data = updated_boxes
                    result = DetectionResults(orig_img=crop, boxes=boxes)

                    # Assign attribute to results for correct naming convention
                    result.save_dir = output_folder
                    try:
                        parts = os.path.basename(image_file).split(".")[0].split("_")
                        result.source_name = f"{'_'.join(parts[:-1])}_{idx}"
                        result.frame_number = parts[-1]
                    except Exception as e:
                        logging.error(f"Failed to assign source name: {image_file}")
                        result.source_name = f"{os.path.basename(image_file).split('.')[0]}_{idx}"
                        result.frame_number = idx

                    # Save the result
                    result.save(save_txt=True, extension=extension_out)

                # Update checkpoint after successfully processing the image
                if len(crop_result.crops) > 0:
                    self.update_checkpoint(**{image_file: 1})
        except Exception as e:
            logging.error(f"ERROR: {e} {image_file}, {detection_boxes}")
            traceback.print_exc()

    def run(self, source: Union[str, Dataset], output_folder: str, crop_size: Tuple[int, int] = (640, 640),
            max_workers: int = 4, extension_in: str = '.png', extension_out: str = '.jpg'):
        """
        Process images from a source, which can be either a folder path or a Dataset object, crop the images based
        on the bounding boxes from corresponding .txt files, and save the results to a new folder.
        """
        os.makedirs(output_folder, exist_ok=True)

        # Determine if source is a folder path or Dataset object
        if isinstance(source, str):
            image_files = glob.glob(os.path.join(source, '**', f'*{extension_in}'), recursive=True)
        elif isinstance(source, Dataset):
            image_files = [file_info['full_path'] for file_info in source.values()]
        else:
            raise ValueError("Source must be a folder path (str) or Dataset object")

        # Initialize or load the checkpoint data
        self.initialize_checkpoint_data(image_files)

        def process_file(image_file):
            self.process_image(image_file, output_folder, crop_size, extension_in, extension_out)

        def process_files(files):
            for image_file in files:
                if self.get_checkpoint_data(image_file, 0) == 1:
                    continue  # Skip already processed files

                # Process the file without threading
                process_file(image_file)

        if max_workers == 1:
            process_files(image_files)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, image_file) for image_file in image_files if self.get_checkpoint_data(image_file, 0) == 0]

                for future in as_completed(futures):
                    try:
                        future.result()  # Get the result to raise any exceptions
                    except Exception as e:
                        logging.info(f"Exception occurred during processing: {e}")

                if all(value == 1 for value in self.checkpoint_data.values()):
                    logging.info("All images have been processed. Removing checkpoint.")
                    self.remove_checkpoint()

    def initialize_checkpoint_data(self, image_files):
        """
        Initialize the checkpoint data with the image files.
        """
        for image_file in image_files:
            if image_file not in self.checkpoint_data:
                self.update_checkpoint(**{image_file: 0})




