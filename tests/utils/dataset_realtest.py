import os
import unittest
from detectflow.utils.dataset import Dataset, get_file_data, DatasetDiagnoser, DatasetSlicer
from detectflow.config import ROOT_DIR

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and some files for testing
        self.test_dir = os.path.join(ROOT_DIR, 'tests', 'resources', 'dataset')
        self.image_file = os.path.join('object', 'CZ1_M1_VerArv01_20210517_11_25_14880.png')
        self.label_file = os.path.join('object', 'CZ1_M1_VerArv01_20210517_11_25_14880.txt')

    def test_get_file_data_with_label(self):
        from detectflow.utils.file import is_yolo_label

        root = os.getcwd()
        folder = os.path.join(self.test_dir)
        file = os.path.basename(self.image_file)

        print(is_yolo_label(self.label_file))

        result = get_file_data(root, folder, file)
        print(result)

    def test_get_file_data_without_label(self):
        root = os.getcwd()
        folder = os.path.join(self.test_dir, 'empty')
        file = 'CZ1_M2_VerArv02_20210518_18_11_10847.png'

        with open(os.path.join(folder, file), 'w') as f:
            f.write('test image content without label')

        result = get_file_data(root, folder, file)
        print(result)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_folder = os.path.join(ROOT_DIR, 'tests', 'resources', 'dataset')
        self.dataset = Dataset.from_folder(self.dataset_folder)

    def test_basic_properties(self):

        # Print basic attributes
        print('Attributes:')
        print('Size: ', self.dataset.size)
        print('Train partition size: ', self.dataset.train_size)
        print('Val partition size: ', self.dataset.val_size)
        print('Test partition size: ', self.dataset.test_size)
        print('Number of frames with detection: ', self.dataset.object)
        print('Number of frames without detection: ', self.dataset.empty)
        print('List of dataset partitions: ', self.dataset.dataset_names)
        print('List of parent folders: ', self.dataset.parent_folders)

    def test_subset_dataset(self):

        # Test getting a random subset of data
        subset = self.dataset.get_random_subset(10, True, by='parent_folder')

        # Print basic attributes
        print("\n")
        print('Subset Attributes:')
        print('Size: ', subset.size)
        print('Train partition size: ', subset.train_size)
        print('Val partition size: ', subset.val_size)
        print('Test partition size: ', subset.test_size)
        print('Number of frames with detection: ', subset.object)
        print('Number of frames without detection: ', subset.empty)
        print('List of dataset partitions: ', subset.dataset_names)
        print('List of parent folders: ', subset.parent_folders)

    def test_merge_datasets(self):

        # Test merging two subsets
        subset = self.dataset.get_random_subset(10, True, by='parent_folder')
        subset_2 = self.dataset.get_random_subset(10, False)
        merge = Dataset.from_datasets([subset, subset_2])

        # Check merging results
        print('Merged dataset size: ', merge.size)
        print('Merged dataset train partition size: ', merge.train_size)
        print('Merged dataset val partition size: ', merge.val_size)

    def test_dataset_with_detection_boxes(self):

        print("Subset with detection boxes constructed:")
        for value in self.dataset.with_detection_boxes.values():
            print(type(value))

    def test_inspect_one_key_dictionary(self):

        print("Inspect one key dictionary:")
        print(self.dataset['CZ1_M2_VerArv02_20210518_18_11_10847.png'])

    def test_visualize_image(self):

        print("Visualize one image:")
        self.dataset.visualize(key='NOR1_S4_MelPra01_20220626_17_23_18956.png')
        self.dataset.visualize(key='CZ2_T1_AnaArv02_20210618_14_16_804.png')

    def test_reorganize_files(self):

        print("Test reorganization of files:")
        self.dataset.reorganize_files(base_folder=os.path.join(ROOT_DIR, 'tests', 'resources', 'dataset'), by='empty_object', keep_empty_separated=True)

        print("Dataset after reorganization:")
        for i, value in enumerate(self.dataset.values()):

            print(value)

            if i > 10:
                break

    def test_split_dataset(self):

        self.dataset.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        print('Train partition size: ', self.dataset.train_size)
        print('Val partition size: ', self.dataset.val_size)
        print('Test partition size: ', self.dataset.test_size)

    def test_sanity_check(self):
        self.dataset.sanity_check()

    def test_generate_yolo_files(self):
        self.dataset.generate_yolo_files(destination=os.path.join(ROOT_DIR, 'tests', 'resources', 'yolo_files'),
                                         classes={0: 'object', 1: 'empty'},
                                         absolute_paths=False,
                                         generate_yaml=True)


class TestDiagnoser(unittest.TestCase):

    def setUp(self):
        self.dataset_folder = os.path.join(ROOT_DIR, 'tests', 'resources', 'dataset')
        self.dataset = Dataset.from_folder(self.dataset_folder)
        self.diagnoser = DatasetDiagnoser(self.dataset)

    def test_plot_image_size(self):
        try:
            self.diagnoser.plot_image_size_distribution()
        except Exception as e:
            raise RuntimeError("Error when plotting image size distribution") from e

    def test_plot_bbox_count(self):
        try:
            self.diagnoser.plot_bbox_count()
        except Exception as e:
            raise RuntimeError("Error when plotting bbox count") from e

    def test_plot_bbox_heatmap(self):
        try:
            self.diagnoser.plot_bbox_heatmap(image_size=(640, 640))
        except Exception as e:
            raise RuntimeError("Error when plotting bbox heatmap") from e

    def test_plot_class_distribution(self):
        try:
            self.diagnoser.plot_class_distribution()
        except Exception as e:
            raise RuntimeError("Error when plotting class distribution") from e

    def test_plot_bbox_size_distribution(self):
        try:
            self.diagnoser.plot_bbox_size_distribution()
        except Exception as e:
            raise RuntimeError("Error when plotting bbox size distribution") from e

    def test_plot_aspect_ratio_distribution(self):
        try:
            self.diagnoser.plot_aspect_ratio_distribution()
        except Exception as e:
            raise RuntimeError("Error when plotting aspect ratio distribution") from e

    def test_plot_image_brightness_distribution(self):
        try:
            self.diagnoser.plot_image_brightness_distribution()
        except Exception as e:
            raise RuntimeError("Error when plotting image brightness distribution") from e

    def test_collect_data(self):
        try:
            self.diagnoser.detect_class_imbalance(threshold=0.1)
        except Exception as e:
            raise RuntimeError("Error when plotting class imbalance") from e


class TestDatasetSlicer(unittest.TestCase):

    def setUp(self):
        self.dataset_folder = os.path.join(ROOT_DIR, 'tests', 'resources', 'to_slice')

    def test_from_dataset(self):
        self.dataset = Dataset.from_folder(self.dataset_folder)
        self.slicer = DatasetSlicer(file_dict=self.dataset)

        slices = self.slicer.slice()

        print(f"Original dataset size: {self.dataset.size}")
        print(f"Number of slices: {len(slices)}")
        for i, s in enumerate(slices):
            print(f"Slice {i+1} size: {s.size}")

    def test_from_folder(self):
        self.slicer = DatasetSlicer(source=self.dataset_folder)

        slices = self.slicer.slice()

        print(f"Number of slices: {len(slices)}")
        for i, s in enumerate(slices):
            print(f"Slice {i+1} size: {s.size}")

    def test_save_slices(self):
        self.slicer = DatasetSlicer(source=self.dataset_folder)

        slices = self.slicer.save_slices(os.path.join(ROOT_DIR, 'tests', 'resources', 'slices'))

        print(f"Number of slices: {len(slices)}")
        for i, s in enumerate(slices):
            print(f"Slice {i + 1} size: {s.size}")

if __name__ == '__main__':
    unittest.main()