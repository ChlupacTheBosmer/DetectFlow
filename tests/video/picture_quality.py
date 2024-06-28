import unittest
import numpy as np
import cv2
from detectflow.video.picture_quality import PictureQualityAnalyzer

class TestPictureQualityAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a synthetic frame for testing
        cls.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(cls.frame, (20, 20), (80, 80), (255, 255, 255), -1)  # Add a white square

    def setUp(self):
        self.analyzer = PictureQualityAnalyzer(self.frame)

    def test_blur(self):
        blur = self.analyzer.blur
        self.assertIsInstance(blur, float)
        self.assertGreaterEqual(blur, 0)

    def test_focus(self):
        focus, focus_area = self.analyzer.get_focus()
        self.assertIsInstance(focus, float)
        self.assertGreaterEqual(focus, 0)
        self.assertIsInstance(focus_area, np.ndarray)
        self.assertEqual(focus_area.shape, self.frame.shape[:2])

    def test_focus_regions(self):
        focus_regions = self.analyzer.focus_regions
        self.assertIsNotNone(focus_regions)

    def test_focus_heatmap(self):
        focus_heatmap = self.analyzer.focus_heatmap
        self.assertIsInstance(focus_heatmap, np.ndarray)
        self.assertEqual(focus_heatmap.shape[2], 3)  # Check if it's a 3-channel image

    def test_contrast(self):
        contrast = self.analyzer.contrast
        self.assertIsInstance(contrast, float)
        self.assertGreaterEqual(contrast, 0)

    def test_brightness(self):
        brightness = self.analyzer.brightness
        self.assertIsInstance(brightness, float)
        self.assertGreaterEqual(brightness, 0)

    def test_color_variance(self):
        color_variance = self.analyzer.color_variance
        self.assertIsInstance(color_variance, float)
        self.assertGreaterEqual(color_variance, 0)

    def test_daytime(self):
        is_daytime = self.analyzer.get_daytime(brightness_threshold=10)
        self.assertIsInstance(is_daytime, bool)

    def test_get_blur(self):
        blur = self.analyzer.get_blur()
        self.assertIsInstance(blur, float)
        self.assertGreaterEqual(blur, 0)

    def test_get_focus(self):
        focus, focus_area = self.analyzer.get_focus(threshold=0.5, sobel_kernel_size=3)
        self.assertIsInstance(focus, float)
        self.assertGreaterEqual(focus, 0)
        self.assertIsInstance(focus_area, np.ndarray)
        self.assertEqual(focus_area.shape, self.frame.shape[:2])

    def test_get_focus_regions(self):
        _, focus_area = self.analyzer.get_focus()
        focus_regions = self.analyzer.get_focus_regions(focus_area, contour_threshold=10, merge_threshold=20)
        self.assertIsNotNone(focus_regions)

    def test_get_focus_heatmap(self):
        _, focus_area = self.analyzer.get_focus()
        heatmap = self.analyzer.get_focus_heatmap(focus_area, blur_amount=15)
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape[2], 3)  # Check if it's a 3-channel image

    def test_get_contrast(self):
        contrast = self.analyzer.get_contrast()
        self.assertIsInstance(contrast, float)
        self.assertGreaterEqual(contrast, 0)

    def test_get_brightness(self):
        brightness = self.analyzer.get_brightness()
        self.assertIsInstance(brightness, float)
        self.assertGreaterEqual(brightness, 0)

    def test_get_color_variance(self):
        color_variance = self.analyzer.get_color_variance()
        self.assertIsInstance(color_variance, float)
        self.assertGreaterEqual(color_variance, 0)

    def test_get_daytime(self):
        print("Brightness: ", self.analyzer.brightness)
        print("Saturation: ", self.analyzer.saturation)
        is_daytime = self.analyzer.get_daytime(brightness_threshold=10, saturation_threshold = 10)
        self.assertIsInstance(is_daytime, bool)

    def test_real_test(self):
        from detectflow.video.video_data import Video

        vid = Video(video_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\CZ2_T1_AciArv02_20210619_23_21.mp4")

        analyzer = vid.get_picture_quality(25)

        print("Blur: ", analyzer.blur)
        print("Focus: ", analyzer.focus)
        print("Contrast: ", analyzer.contrast)
        print("Brightness: ", analyzer.brightness)
        print("Saturation: ", analyzer.saturation)
        print("Color variance: ", analyzer.color_variance)

        vid = Video(
            video_path=r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb3_20220524_07_44.mp4")

        analyzer = vid.get_picture_quality(25)

        print("Blur: ", analyzer.blur)
        print("Focus: ", analyzer.focus)
        print("Contrast: ", analyzer.contrast)
        print("Brightness: ", analyzer.brightness)
        print("Saturation: ", analyzer.saturation)
        print("Color variance: ", analyzer.color_variance)

if __name__ == '__main__':
    unittest.main()