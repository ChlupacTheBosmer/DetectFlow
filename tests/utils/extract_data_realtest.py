import unittest
import os
import time
import threading
import multiprocessing
from detectflow.utils.extract_data import extract_data_from_video


class TestExtractDataFromVideo(unittest.TestCase):

    def setUp(self):
        # Set up path to test video file
        self.video_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb2_20220524_07_44.mp4"

        # Ensure the test file exists
        assert os.path.exists(self.video_path), f"Test video file does not exist: {self.video_path}"

    def test_extract_data_timing(self):

        # Call extract_data_from_video on the first instance
        start_time = time.time()
        data = extract_data_from_video(video_path=self.video_path, frame_skip=500)
        first_call_duration = time.time() - start_time

        # Call extract_data_from_video on the second instance
        start_time = time.time()
        data2 = extract_data_from_video(video_path=self.video_path, frame_skip=500)
        second_call_duration = time.time() - start_time

        # Ensure data is extracted correctly
        self.assertIsNotNone(data, "Data should not be None on first call.")
        self.assertIsNotNone(data2, "Data should not be None on second call.")

        # Ensure the extracted data is the same
        self.assertEqual(data, data2, "Data extracted on first and second call should be the same.")

        # Ensure the second call is faster due to caching
        self.assertLess(second_call_duration, first_call_duration,
                        "Second call should be faster due to caching.")

        print("First call duration: ", first_call_duration)
        print("Second call duration: ", second_call_duration)

    def test_extract_data_with_threads(self):
        # To store the results and execution time from threads
        results = [None, None]
        durations = [None, None]

        def thread_function(index):
            start_time = time.time()
            results[index] = extract_data_from_video(video_path=self.video_path, frame_skip=500)
            durations[index] = time.time() - start_time

        # Thread 1
        thread1 = threading.Thread(target=thread_function, args=(0,))
        thread1.start()
        thread1.join()

        # Thread 2
        thread2 = threading.Thread(target=thread_function, args=(1,))
        thread2.start()
        thread2.join()

        # Ensure data is extracted correctly
        self.assertIsNotNone(results[0], "Data should not be None for thread 1.")
        self.assertIsNotNone(results[1], "Data should not be None for thread 2.")

        # Ensure the extracted data is the same
        self.assertEqual(results[0], results[1], "Data extracted by both threads should be the same.")

        # Ensure at least one thread call is significantly faster, indicating cache usage
        self.assertTrue(
            durations[0] < durations[1] * 0.8 or durations[1] < durations[0] * 0.8,
            "One of the calls should be significantly faster, indicating cache usage."
        )

        print("Thread 1 duration: ", durations[0])
        print("Thread 2 duration: ", durations[1])

def process_function(index, results, durs, video_path):
    start_time = time.time()
    results[index] = extract_data_from_video(video_path=video_path, frame_skip=500)
    durs[index] = time.time() - start_time

def extract_data_with_processes():
    video_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\video\resources\GR2_L1_TolUmb2_20220524_07_44.mp4"

    # To store the results and execution time from processes
    manager = multiprocessing.Manager()
    results = manager.list([None, None])
    durs = manager.list([None, None])

    # Process 1
    process1 = multiprocessing.Process(target=process_function, args=(0, results, durs, video_path))
    process1.start()
    process1.join()

    # Process 2
    process2 = multiprocessing.Process(target=process_function, args=(1, results, durs, video_path))
    process2.start()
    process2.join()


    print("Process 1 duration: ", durs[0])
    print("Process 2 duration: ", durs[1])


if __name__ == "__main__":
    #unittest.main()
    extract_data_with_processes()
