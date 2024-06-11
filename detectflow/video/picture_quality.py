import cv2
import numpy as np
from detectflow import DetectionBoxes, BoxManipulator
from detectflow.manipulators.box_manipulator import boxes_centers_distance


class PictureQualityAnalyzer:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self._blur = None
        self._focus = None
        self._focus_regions = None
        self._focus_heatmap = None
        self._contrast = None
        self._brightness = None
        self._color_variance = None

    @property
    def blur(self):
        if self._blur is None:
            self._blur = self.get_blur()
        return self._blur

    def get_blur(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    @property
    def focus(self):
        if self._focus is None:
            self._focus, _ = self.get_focus()
        return self._focus

    def get_focus(self, threshold: float = 0.5, sobel_kernel_size: int = 3):

        # Make sure sobel_kernel_size is odd
        sobel_kernel_size = sobel_kernel_size if sobel_kernel_size % 2 != 0 else sobel_kernel_size + 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Sobel operators for edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        focus_measure = np.mean(sobel_mag)

        # Normalize the Sobel magnitude
        sobel_mag = sobel_mag / sobel_mag.max()

        # Detect the area with the highest focus
        _, most_focused_area = cv2.threshold(sobel_mag, threshold, 1.0, cv2.THRESH_BINARY)

        return focus_measure, most_focused_area

    @property
    def focus_regions(self):
        if self._focus_regions is None:
            _, focus_area = self.get_focus()
            self._focus_regions = self.get_focus_regions(focus_area)
        return self._focus_regions

    def get_focus_regions(self, focus_area: np.ndarray, contour_threshold: int = 10, merge_threshold: int = 20):
        # Convert focus_area to 3 channels and resize to match frame dimensions if needed
        focus_area_resized = cv2.resize(focus_area, (self.frame.shape[1], self.frame.shape[0]))
        # focus_area_colored = np.stack((focus_area_resized,) * 3, axis=-1) * 255

        # Dilate the focus area to merge nearby contours
        kernel = np.ones((10, 10), np.uint8)  # Increase kernel size for more merging
        focus_area_dilated = cv2.dilate(focus_area_resized.astype(np.uint8), kernel, iterations=3)

        # Find contours in the dilated focus area
        contours, _ = cv2.findContours(focus_area_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > contour_threshold:  # Filter small contours
                # Calculate the bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Append box in xyxy format
                bboxes.append(BoxManipulator.coco_to_xyxy(np.array([x, y, w, h])))

        # Merge overlapping or closely neighboring bounding boxes
        if not len(bboxes) > 0:
            final_bboxes = None
        else:
            merged_bboxes, _ = BoxManipulator.analyze_clusters(detection_boxes=np.array(bboxes),
                                                               eps=merge_threshold,
                                                               metric=boxes_centers_distance)

            # Create DetectionBoxes object
            merged_bboxes = DetectionBoxes(merged_bboxes, (self.frame.shape[0], self.frame.shape[1]), "xyxy")

            # Remove smaller boxes completely inside larger boxes
            final_bboxes = BoxManipulator.remove_contained_boxes(merged_bboxes)

        return final_bboxes

    @property
    def focus_heatmap(self):
        if self._focus_heatmap is None:
            _, focus_area = self.get_focus()
            self._focus_heatmap = self.get_focus_heatmap(focus_area)
        return self._focus_heatmap

    def get_focus_heatmap(self, focus_area: np.ndarray, blur_amount: int = 15):

        # Make sure blur is odd
        blur_amount = blur_amount if blur_amount % 2 != 0 else blur_amount + 1

        # Blur the focus area to create a gradient effect
        heatmap = cv2.GaussianBlur(focus_area, (blur_amount, blur_amount), 0)

        # Normalize the heatmap to range [0, 255]
        heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Apply a color map to the heatmap
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_HOT)

        return heatmap_colored

    def get_focus_inspection(self, focus_area: np.ndarray, show_heatmap: bool = True, blur_amount: int = 15,
                             contour_threshold: int = 10, merge_threshold: int = 20):

        # Make sure blur is odd
        blur_amount = blur_amount if blur_amount % 2 != 0 else blur_amount + 1

        # Convert focus_area to 3 channels and resize to match frame dimensions if needed
        focus_area_resized = cv2.resize(focus_area, (self.frame.shape[1], self.frame.shape[0]))
        # focus_area_colored = np.stack((focus_area_resized,) * 3, axis=-1) * 255

        # Get focus regions
        highlighted_frame = self.frame.copy()
        bboxes = self.focus_regions(highlighted_frame, focus_area, contour_threshold, merge_threshold)

        if bboxes is not None:
            for (x1, y1, x2, y2) in bboxes.xyxy:
                cv2.rectangle(highlighted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Calculate the centroid of the bounding box
                cx = x1 + ((x2 - x1) // 2)
                cy = y1 + ((y2 - y1) // 2)
                cv2.circle(highlighted_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        if show_heatmap:
            heatmap = self.focus_heatmap(focus_area_resized, blur_amount)
            highlighted_frame = cv2.addWeighted(highlighted_frame, 0.6, heatmap, 0.4, 0)

        return highlighted_frame

    @property
    def contrast(self):
        if self._contrast is None:
            self._contrast = self.get_contrast()
        return self._contrast

    def get_contrast(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return gray.std()

    @property
    def brightness(self):
        if self._brightness is None:
            self._brightness = self.get_brightness()
        return self._brightness

    def get_brightness(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2].mean()

    @property
    def color_variance(self):
        if self._color_variance is None:
            self._color_variance = self.get_color_variance()
        return self._color_variance

    def get_color_variance(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return np.var(gray)

    def get_daytime(self, color_variance_threshold: int = 10):
        # If the color variance is higher than threshold, it's likely a daytime video
        return bool(self.color_variance > color_variance_threshold)
