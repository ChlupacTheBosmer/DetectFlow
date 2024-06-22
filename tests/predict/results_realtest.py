import unittest
from detectflow.utils.sampler import Sampler
import os
import unittest
import numpy as np
from detectflow.predict.results import DetectionBoxes, DetectionResults


class TestDetectionBoxes(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array([(0, 0, 1, 1, 0.5, 6, 10),
                               (0, 0, 2, 2, 0.5, 6, 10),
                               [0, 0, 3, 3, 0.5, 6, 10],
                               [0, 0, 4, 4, 0.5, 6, 10]])
        self.orig_shape = (10, 10)
        self.detection_boxes = DetectionBoxes(self.boxes, self.orig_shape)

    def test_init(self):
        self.assertEqual(self.detection_boxes.data.shape, self.boxes.shape)
        self.assertEqual(self.detection_boxes.orig_shape, self.orig_shape)
        self.assertTrue(self.detection_boxes.is_track)

    def test_from_boxes_instance(self):
        from ultralytics.engine.results import Boxes

        boxes = Boxes(self.boxes[:, [0, 1, 2, 3, 6, 4, 5]], self.orig_shape)

        # Create a new instance of DetectionBoxes
        new_detection_boxes = DetectionBoxes.from_boxes_instance(boxes)

        # Check that the new instance has the same data and orig_shape as the original
        print(new_detection_boxes.data)
        print(self.detection_boxes.data)
        self.assertTrue(np.array_equal(new_detection_boxes.data, self.detection_boxes.data))
        self.assertEqual(new_detection_boxes.orig_shape, self.detection_boxes.orig_shape)

    def test_from_custom_format(self):
        # Define a custom format
        format_flag = "xyxytcp"

        # Create a new instance of DetectionBoxes using the custom format
        new_detection_boxes = DetectionBoxes.from_custom_format(self.boxes[:, [0, 1, 2, 3, 6, 5, 4]], self.orig_shape, format_flag)

        # Check that the new instance has the same data and orig_shape as the original
        self.assertTrue(np.array_equal(new_detection_boxes.data, self.detection_boxes.data))
        self.assertEqual(new_detection_boxes.orig_shape, self.detection_boxes.orig_shape)

    def test_properties(self):
        self.assertEqual(self.detection_boxes.xyxy.shape, self.boxes[:, :4].shape)
        self.assertEqual(self.detection_boxes.conf.shape, self.boxes[:, 4].shape)
        self.assertEqual(self.detection_boxes.cls.shape, self.boxes[:, 5].shape)
        self.assertEqual(self.detection_boxes.id.shape, self.boxes[:, 6].shape)

    def test_attributes(self):
        # Test the xyxy attribute
        self.assertTrue(np.array_equal(self.detection_boxes.xyxy, self.boxes[:, :4]))

        # Test the conf attribute
        self.assertTrue(np.array_equal(self.detection_boxes.conf, self.boxes[:, 4]))

        # Test the cls attribute
        self.assertTrue(np.array_equal(self.detection_boxes.cls, self.boxes[:, 5]))

        # Test the id attribute
        self.assertTrue(np.array_equal(self.detection_boxes.id, self.boxes[:, 6]))

        # Test the xywh attribute
        xywh = np.copy(self.boxes[:, :4])
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        xywh[:, 0] += xywh[:, 2] / 2
        xywh[:, 1] += xywh[:, 3] / 2
        self.assertTrue(np.array_equal(self.detection_boxes.xywh, xywh))

        # Test the xyxyn attribute
        xyxyn = np.copy(self.boxes[:, :4])
        xyxyn[:, [0, 2]] /= self.orig_shape[1]
        xyxyn[:, [1, 3]] /= self.orig_shape[0]
        self.assertTrue(np.array_equal(self.detection_boxes.xyxyn, xyxyn))

        # Test the xywhn attribute
        xywhn = np.copy(xywh)
        xywhn[:, [0, 2]] /= self.orig_shape[1]
        xywhn[:, [1, 3]] /= self.orig_shape[0]
        self.assertTrue(np.array_equal(self.detection_boxes.xywhn, xywhn))

        # Test the shape attribute
        self.assertEqual(self.detection_boxes.shape, self.boxes.shape)

        # Test the coco attribute
        coco = np.copy(self.boxes[:, :4])
        coco[:, 2] -= coco[:, 0]
        coco[:, 3] -= coco[:, 1]
        self.assertTrue(np.array_equal(self.detection_boxes.coco, coco))

    def test_methods(self):
        self.assertEqual(len(self.detection_boxes), len(self.boxes))
        self.assertTrue(np.array_equal(self.detection_boxes[0], self.boxes[0]))
        self.assertTrue(self.boxes[0] in self.detection_boxes)
        self.assertTrue(np.array_equal(self.detection_boxes.tolist(), self.boxes.tolist()))

        copied_detection_boxes = self.detection_boxes.copy()
        self.assertTrue(np.array_equal(copied_detection_boxes.data, self.detection_boxes.data))

        coco_boxes = self.detection_boxes.coco
        self.assertEqual(coco_boxes.shape, self.boxes[:, :4].shape)


class TestDetectionResults(unittest.TestCase):
    def setUp(self):

        self.image, self.boxes = Sampler.create_sample_image_with_bboxes(grid_size=512, square_size=2, num_boxes=4, as_detection_boxes=False)

        additional_data = np.array([(0.5, 6, 10),
                               (0.5, 6, 10),
                               [0.5, 6, 10],
                               [0.5, 6, 10]])

        self.boxes = np.column_stack((np.array(self.boxes), additional_data))

        kwargs = {
            "names": {"visitor": 0},
            "frame_number": 666,
            "source_path": r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb2_20220524_07_44.mp4",
            "visit_number": 0,
            "roi_number": 1
        }

        self.detection_results = DetectionResults(self.image, self.boxes, **kwargs)
        path = os.path.dirname(os.path.realpath(__file__))
        self.detection_results.save_dir = os.path.join(path, "test_results")

    def test_init(self):
        self.assertIsInstance(self.detection_results, DetectionResults)

    def test_from_results(self):
        from ultralytics.engine.results import Results

        kwargs = {
            "names": {"visitor": 0},
            "path": r"\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb2_20220524_07_44.png"
        }

        print(kwargs)

        res = Results(self.image, boxes=np.array(self.boxes)[:,[0,1,2,3,6,4,5]], **kwargs) # boxes must be passed in the format expected by Results (xyxytpc)

        new_detection_results = DetectionResults.from_results(res)

        self.assertIsInstance(new_detection_results, DetectionResults)
        self.assertTrue(np.array_equal(new_detection_results.orig_img, self.detection_results.orig_img))
        self.assertEqual(new_detection_results.orig_shape, self.detection_results.orig_shape)
        self.assertTrue(np.array_equal(new_detection_results.boxes.data, self.detection_results.boxes.data)) # Checks that the init correctly translated the boxes to xyxypct format
        print(new_detection_results.boxes.data)
        print(self.detection_results.boxes.data)

    def test_from_prediction_result(self):

        from sahi.prediction import PredictionResult, ObjectPrediction

        ops = [ObjectPrediction(bbox=box[:4], category_id=int(box[5]), score=box[4]) for box in self.boxes]

        pr = PredictionResult(ops, self.image)

        self.assertIsInstance(pr, PredictionResult)

        new_detection_results = DetectionResults.from_prediction_results(pr)

        self.assertIsInstance(new_detection_results, DetectionResults)
        self.assertTrue(np.array_equal(new_detection_results.orig_img, self.detection_results.orig_img))
        self.assertEqual(new_detection_results.orig_shape, self.detection_results.orig_shape)
        self.assertTrue(np.array_equal(new_detection_results.boxes.data, self.detection_results.boxes.data[:,:6]))
        print(new_detection_results.boxes.data)
        print(self.detection_results.boxes.data)

    def test_attributes(self):

        from detectflow.predict.results import determine_source_type

        # Test the orig_img attribute
        self.assertTrue(np.array_equal(self.detection_results.orig_img, self.image))

        # Test the orig_shape attribute
        self.assertEqual(self.detection_results.orig_shape, self.image.shape[:2])
        print("orig_shape", self.detection_results.orig_shape)

        # Test the boxes attribute
        self.assertTrue(np.array_equal(self.detection_results.boxes.data, self.boxes))
        print("boxes", self.detection_results.boxes.data)

        # Test the names attribute
        self.assertEqual(self.detection_results.names, {"visitor": 0})
        print("names", self.detection_results.names)

        # Test the frame_number attribute
        self.assertEqual(self.detection_results.frame_number, 666)
        print("frame_number", self.detection_results.frame_number)

        # Test the source_path attribute
        self.assertEqual(self.detection_results.source_path, r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb2_20220524_07_44.mp4")
        print("source_path", self.detection_results.source_path)

        # Test the visit_number attribute
        self.assertEqual(self.detection_results.visit_number, 0)
        print("visit_number", self.detection_results.visit_number)

        # Test the roi_number attribute
        self.assertEqual(self.detection_results.roi_number, 1)
        print("roi_number", self.detection_results.roi_number)

        # Test the save_dir attribute
        self.assertEqual(self.detection_results.save_dir, os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_results"))
        print("save_dir", self.detection_results.save_dir)

        # Test the source_name attribute
        self.assertEqual(self.detection_results.source_name, os.path.splitext(os.path.basename(self.detection_results.source_path))[0])
        print("source_name", self.detection_results.source_name)

        # Test the source_type attribute
        self.assertEqual(self.detection_results.source_type, determine_source_type(self.detection_results.source_path))
        print("source_type", self.detection_results.source_type)

    def test_video_time(self):
        # Test the video_time property
        print("video_time", self.detection_results.video_time)
        self.assertEqual(type(self.detection_results.video_time), float)
        self.assertAlmostEqual(self.detection_results.video_time, self.detection_results.frame_number / 25, delta=0.5)

    def test_real_time(self):
        from datetime import datetime

        # Test the real_time property
        # Replace with your expected value
        print(self.detection_results.real_time)
        self.assertEqual(type(self.detection_results.real_time), datetime)

    def test_recording_id(self):
        # Test the recording_id property
        # Replace with your expected value
        expected_recording_id = 'GR2_L1_TolUmb2'
        self.assertEqual(self.detection_results.recording_id, expected_recording_id)

    def test_video_id(self):
        # Test the video_id property
        # Replace with your expected value
        expected_video_id = 'GR2_L1_TolUmb2_20220524_07_44'
        self.assertEqual(self.detection_results.video_id, expected_video_id)

    def test_source_path_setter(self):
        # Test the source_path property
        # Replace with your expected value
        expected_source_path = r"\resources\CZ2_M1_AraHir2_20220801_10_56.jpeg"
        self.detection_results.source_path = expected_source_path
        self.assertEqual(self.detection_results.source_path, expected_source_path)
        self.assertEqual(self.detection_results.source_name, 'CZ2_M1_AraHir2_20220801_10_56')
        self.assertEqual(self.detection_results.source_type, 'image')

        self.assertEqual(self.detection_results.video_id, None)

    def test_fil_boxes(self):

        from detectflow.predict.results import get_avg_box_max_dim, get_avg_box_diag
        from detectflow.utils.inspector import Inspector

        # Test the filtered_boxes property
        # Replace with your expected value
        reference_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3, as_detection_boxes=True)
        self.detection_results.reference_boxes = reference_boxes

        print(len(self.detection_results.boxes), len(self.detection_results.filtered_boxes))
        print("Radius: ",
              get_avg_box_max_dim([self.detection_results.boxes, self.detection_results.reference_boxes]))
        print("Diag Radius: ",
              get_avg_box_diag([self.detection_results.boxes, self.detection_results.reference_boxes]))
        print("Boxes: ", self.detection_results.boxes.xywh)
        print("Reference Boxes: ", self.detection_results.reference_boxes.xywh)
        print("Filtered Boxes: ", self.detection_results.filtered_boxes.xywh)

        self.assertEqual(type(self.detection_results.filtered_boxes), DetectionBoxes)

        Inspector.display_frame_with_multiple_boxes(frame=self.detection_results.orig_img,
                                                    detection_boxes_list=[self.detection_results.boxes, self.detection_results.reference_boxes, self.detection_results.filtered_boxes])

    def test_on_flowers(self):
        # Test the on_flowers property
        reference_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3,
                                                       as_detection_boxes=True)
        self.detection_results.reference_boxes = reference_boxes

        self.assertTrue(all(type(b) is bool for b in self.detection_results.on_flowers))
        print("On Flowers: ", self.detection_results.on_flowers)

    def test_save(self):
        self.detection_results.save(save_txt=True)

    def test_save_txt(self):
        self.detection_results.save_txt()

    def test_get_item(self):
        reference_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3,
                                                       as_detection_boxes=True)
        self.detection_results.reference_boxes = reference_boxes

        self.assertIsInstance(self.detection_results[0], DetectionResults)
        print("New selected instance: ", self.detection_results[0])

        self.assertIsInstance(self.detection_results[:2], DetectionResults)
        print("New sliced instance: ", self.detection_results[:2])

    def test_len(self):

        print(len(self.detection_results))

    def test_apply(self):

        new_detection_results = self.detection_results._apply('copy')

        self.assertIsInstance(new_detection_results, DetectionResults)
        self.assertEqual(new_detection_results.orig_shape, self.detection_results.orig_shape)
        print(self.detection_results.boxes.data)
        print(new_detection_results.boxes.data)

    def test_update(self):

        new_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3,
                                                       as_detection_boxes=True)

        new_reference_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3,
                                                 as_detection_boxes=True)

        self.detection_results.update(boxes=new_boxes.data, reference_boxes=new_reference_boxes.data)

        self.assertTrue(np.array_equal(self.detection_results.boxes.data, new_boxes.data))
        self.assertTrue(np.array_equal(self.detection_results.reference_boxes.data, new_reference_boxes.data))

    def test_new(self):

        new_detection_results = self.detection_results.new()

        self.assertIsInstance(new_detection_results, DetectionResults)
        self.assertEqual(new_detection_results.orig_shape, self.detection_results.orig_shape)
        print(new_detection_results.boxes.data)
        print(self.detection_results.boxes.data)

    def test_plot(self):

        reference_boxes = Sampler.create_sample_bboxes(grid_size=128, square_size=8, num_boxes=3,
                                                       as_detection_boxes=True)
        self.detection_results.reference_boxes = reference_boxes

        self.detection_results.plot(show=True, save=True)

if __name__ == '__main__':
    unittest.main()

