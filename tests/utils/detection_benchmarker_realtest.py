import os
import unittest
from detectflow.utils.benchmark import DetectionBenchmarker
from detectflow.config import TESTS_DIR

class TestBenchmarker(unittest.TestCase):

    def setUp(self):
        self.db_path = os.path.join(TESTS_DIR, 'resources', 'benchmark', 'CZ1_M1_AraHir01.db')
        self.excel_path = os.path.join(TESTS_DIR, 'resources', 'benchmark', 'CZ1_M1_AraHir01.xlsx')
        self.checkpoint_file = os.path.join(TESTS_DIR, 'resources', 'benchmark', 'checkpoint.json')

        self.benchmarker = DetectionBenchmarker(db_path=self.db_path, excel_path=self.excel_path, checkpoint_file=self.checkpoint_file)

    def test_get_ground_truth(self):
        #self.benchmarker.get_ground_truth()
        pass

    def test_benchmark(self):
        results = self.benchmarker.benchmark(output_dir=os.path.join(TESTS_DIR, 'resources', 'benchmark', 'output'))
        print("Comparison Results:")
        for metric, value in results.items():
            print(f"{metric}: {value}")

        print(f"Detected {len(results['Detected Visits'])} out of {len(results['Total Visits'])} visits")


if __name__ == '__main__':
    unittest.main()

