import unittest
import os
from datetime import datetime, timedelta
from detectflow.utils.excel import AnnotationFile, Annotation, AnnotationList, AnnotationFileDiagnoser
from detectflow.config import TESTS_DIR


class TestAnnotationFile(unittest.TestCase):

    def setUp(self):
        # Set up the path to the Excel files for testing
        files = [
            os.path.join(TESTS_DIR, 'utils', 'resources', 'GR2_L2_LavSto2_filtered_valid.xlsx'),
            os.path.join(TESTS_DIR, 'utils', 'resources', 'GR2_L2_LavSto2_filtered.xlsx'),
            os.path.join(TESTS_DIR, 'utils', 'resources', 'GR2_L1_TolUmb1_unfiltered.xlsx'),
        ]
        self.excel_file_path = files[0]

    def test_annotation_file_loading(self):
        # Instantiate the AnnotationFile with the provided path
        annotation_file = AnnotationFile(filepath=self.excel_file_path)

        # Check if the dataframe is loaded properly
        self.assertIsNotNone(annotation_file.dataframe, "Dataframe should not be None after loading the file.")

        # Perform additional checks on the dataframe, such as checking column names, data types, etc.
        expected_columns = ['year', 'month', 'day', 'hour_a', 'min_a', 'sec_a', 'visit', 'duration',
                            'hour_d', 'min_d', 'sec_d', 'vis_id', 'vis_ord', 'no_flo', 'f_pol',
                            'f_nec', 'f_fp', 'con_m', 'no_con_m', 'con_f', 'no_con_f', 'ts']

        self.assertTrue(all(column in annotation_file.dataframe.columns for column in expected_columns),
                        "Dataframe does not contain the expected columns.")

        # Check if 'ts' column is constructed correctly
        for ts_value in annotation_file.dataframe['ts']:
            self.assertRegex(ts_value, r'\d{4}\d{2}\d{2}_\d{2}_\d{2}_\d{2}', "Timestamp format is incorrect.")


class TestAnnotationFileDiagnoser(unittest.TestCase):
    def setUp(self):
        # Set up paths to test files
        self.test_dir = TESTS_DIR
        self.test_excel_file = os.path.join(self.test_dir, 'utils', 'resources', 'GR2_L2_LavSto2_filtered_valid.xlsx')

        # Make sure the test directory exists
        if not os.path.isdir(self.test_dir):
            os.makedirs(self.test_dir)

        # Add a check to ensure the test Excel file exists
        if not os.path.isfile(self.test_excel_file):
            raise FileNotFoundError(f"Test Excel file not found: {self.test_excel_file}")

    def test_check_excel_file_success(self):
        diagnoser = AnnotationFileDiagnoser(str(self.test_excel_file))
        report = diagnoser.report

        # Check that the excel_status is 'success'
        self.assertEqual(report['excel_status'], 'success')
        print(report['excel_status'])

        # Check the dataframe diagnosis
        self.assertIn('dataframe_diagnosis', report)
        print(report['dataframe_diagnosis'])
        self.assertIn('shape', report['dataframe_diagnosis'])
        self.assertIn('head', report['dataframe_diagnosis'])
        self.assertIn('describe', report['dataframe_diagnosis'])
        self.assertIn('non_null_counts', report['dataframe_diagnosis'])
        self.assertIn('unique_counts', report['dataframe_diagnosis'])
        self.assertIn('nan_counts', report['dataframe_diagnosis'])

    def test_check_excel_file_error(self):
        # Provide an incorrect file path to trigger an error
        files = [
            os.path.join(self.test_dir, 'utils', 'resources', 'GR2_L2_LavSto2_invalid_structure.xlsx')
        ]
        for file in files:
            print("Testing file:", file)
            diagnoser = AnnotationFileDiagnoser(str(file))
            report = diagnoser.report

            # Check that the excel_status is 'error'
            self.assertEqual(report['excel_status'], 'error')
            print(report['excel_status'])

            # Check the error message
            self.assertIn('error_message', report)
            self.assertIsInstance(report['error_message'], str)
            print(report['error_message'])

    def test_check_video_alignment(self):
        from detectflow.manipulators.manipulator import Manipulator
        # Provide an incorrect file path to trigger an error
        files = [
            os.path.join(self.test_dir, 'utils', 'resources', 'GR2_L2_LavSto2_filtered.xlsx')
        ]

        video_files = Manipulator.list_files(r"D:\Dílna\Kutění\Python\ICCS\icvt\videos", regex=r"GR2_L2_LavSto", extensions=('.mp4', '.avi'), return_full_path=True)

        for file in files:
            print("Testing file:", file)
            diagnoser = AnnotationFileDiagnoser(str(file), video_files)
            report = diagnoser.report

            # Check that the excel_status is 'error'
            self.assertEqual(report['video_alignment_status'], 'success')
            print(report['video_alignment_status'])

            # # Check the error message
            # self.assertIn('error_message', report)
            # self.assertIsInstance(report['error_message'], str)
            # print(report['error_message'])


class TestAnnotation(unittest.TestCase):

    def setUp(self):
        self.start_time = datetime(2022, 5, 24, 7, 44, 6)
        self.end_time = datetime(2022, 5, 24, 7, 49, 6)
        self.duration = timedelta(minutes=5)
        self.video_path = os.path.join(TESTS_DIR, 'video', 'resources', 'GR2_L1_TolUmb3_20220524_07_44.mp4')
        self.video_fps = 25


    def test_annotation_initialization(self):
        annotation = Annotation(
            video_path=self.video_path,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration,
            video_fps=self.video_fps
        )
        self.assertEqual(annotation.start_frame, 0)
        print(annotation.start_frame)
        self.assertEqual(annotation.end_frame, 7500)
        print(annotation.end_frame)
        self.assertEqual(annotation.video_path, self.video_path)
        print(annotation.video_path)
        self.assertEqual(annotation.start_time, self.start_time)
        print(annotation.start_time)
        self.assertEqual(annotation.end_time, self.end_time)
        print(annotation.end_time)
        self.assertEqual(annotation.start_video_time, timedelta(seconds=0))
        print(annotation.start_video_time)
        self.assertEqual(annotation.end_video_time, timedelta(seconds=300))
        print(annotation.end_video_time)
        self.assertEqual(annotation.duration, self.duration)
        print(annotation.duration)
        self.assertEqual(annotation.video_fps, self.video_fps)
        print(annotation.video_fps)

    def test_annotation_from_timestamp(self):
        annotation = Annotation.from_timestamp("20220524_07_44_06", 300, video_path=self.video_path, video_fps=self.video_fps)
        self.assertEqual(annotation.start_time, self.start_time)
        self.assertEqual(annotation.end_time, self.end_time)
        self.assertEqual(annotation.duration, self.duration)
        self.assertEqual(annotation.video_path, self.video_path)
        self.assertEqual(annotation.video_fps, self.video_fps)

    def test_annotation_duration_calculation(self):
        annotation = Annotation(start_time=self.start_time, end_time=self.end_time, video_fps=self.video_fps)
        self.assertEqual(annotation.duration, self.duration)

    def test_annotation_start_frame_calculation(self):
        annotation = Annotation(start_video_time=timedelta(seconds=0), video_fps=self.video_fps)
        self.assertEqual(annotation.start_frame, 0)

    def test_annotation_end_frame_calculation(self):
        annotation = Annotation(end_video_time=timedelta(seconds=300), video_fps=self.video_fps)
        self.assertEqual(annotation.end_frame, 9000)

class TestAnnotationList(unittest.TestCase):

    def setUp(self):
        self.annotation1 = Annotation(
            start_frame=0,
            end_frame=9000,
            video_path="path/to/your/video1.mp4",
            start_time=datetime(2023, 6, 1, 14, 30, 0),
            end_time=datetime(2023, 6, 1, 14, 35, 0),
            start_video_time=timedelta(seconds=0),
            end_video_time=timedelta(seconds=300),
            duration=timedelta(minutes=5),
            video_fps=30
        )
        self.annotation2 = Annotation(
            start_frame=9001,
            end_frame=18000,
            video_path="path/to/your/video2.mp4",
            start_time=datetime(2023, 6, 1, 14, 35, 0),
            end_time=datetime(2023, 6, 1, 14, 40, 0),
            start_video_time=timedelta(seconds=301),
            end_video_time=timedelta(seconds=600),
            duration=timedelta(minutes=5),
            video_fps=30
        )
        self.annotations = AnnotationList(self.annotation1, self.annotation2)

    def test_annotation_list_initialization(self):
        self.assertEqual(len(self.annotations), 2)
        self.assertIn(self.annotation1, self.annotations)
        self.assertIn(self.annotation2, self.annotations)

    def test_annotation_list_recording_id(self):
        self.annotation1.recording_id = "rec1"
        self.annotation2.recording_id = "rec1"
        self.assertEqual(self.annotations.recording_id, "rec1")

    def test_annotation_list_video_paths(self):
        self.assertEqual(len(self.annotations.video_paths), 2)
        self.assertIn("path/to/your/video1.mp4", self.annotations.video_paths)
        self.assertIn("path/to/your/video2.mp4", self.annotations.video_paths)

    def test_annotation_list_visits(self):
        self.assertEqual(self.annotations.visits, 2)

    def test_annotation_list_detections(self):
        # Assume detection_results are properly mocked or real data is provided
        self.assertEqual(self.annotations.detections, 0)  # Adjust according to real detection results

    def test_annotation_list_visitors(self):
        # Assume detection_results with visitors are properly mocked or real data is provided
        self.assertEqual(self.annotations.visitors, 0)  # Adjust according to real detection results

    def test_annotation_list_detection_results(self):
        # Assume detection_results are properly mocked or real data is provided
        self.assertEqual(len(self.annotations.detection_results), 0)  # Adjust according to real detection results

if __name__ == "__main__":
    unittest.main()
