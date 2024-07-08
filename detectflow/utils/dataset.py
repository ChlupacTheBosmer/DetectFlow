from detectflow.utils.file import is_yolo_label
from detectflow.manipulators.database_manipulator import DatabaseManipulator
import random
import shutil
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from multiprocessing import Pool, cpu_count


def get_file_data(root: str, folder: str, file: str) -> Dict[str, Union[str, None]]:
    """
    Get information about a single file.
    """

    # Image file
    full_path = os.path.join(folder, file)
    relative_path = os.path.relpath(full_path, root)

    # Parent folder
    parent_folder = os.path.basename(folder)

    # Label file
    label_file = os.path.splitext(full_path)[0] + '.txt'
    label_path = os.path.relpath(label_file, root) if os.path.exists(label_file) and is_yolo_label(label_file) else None

    return {'full_path': full_path, 'relative_path': relative_path, 'label_full_path': label_file, 'label_relative_path': label_path, 'parent_folder': parent_folder}


def get_relative_path(full_path, folder_name):
    # Split the full path into parts
    path_parts = full_path.split(os.sep)

    # Find the index of the known folder in the path
    try:
        start_index = path_parts.index(folder_name)
    except ValueError:
        # Known folder not found in the path
        return None

    # Reconstruct the path from the known folder
    relative_path_parts = path_parts[start_index + 1:]

    # Join the parts back into a path
    relative_path = os.sep.join(relative_path_parts)

    return relative_path


class Dataset(dict):
    def __init__(self):
        super().__init__()

        self.dataset_name = None

    @classmethod
    def from_folder(cls, folder_path: str) -> 'Dataset':
        """
        Initialize the Dataset from a folder structure.
        """
        dataset = cls()

        # Root folder
        dataset.dataset_name = os.path.dirname(folder_path)

        for root, _, files in os.walk(folder_path):

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    file_info = get_file_data(folder_path, root, file)

                    dataset[file] = {
                        'name': file,
                        'dataset': 'train',
                        'full_path': file_info.get('full_path', None),
                        'relative_path': file_info.get('relative_path', None),
                        'detection': True if file_info.get('label_full_path', None) else False,
                        'label_full_path': file_info.get('label_full_path', None),
                        'label_relative_path': file_info.get('label_relative_path', None),
                        'parent_folder': file_info.get('parent_folder', None)
                    }
        return dataset

    @classmethod
    def from_database(cls, db_path: str) -> 'Dataset':
        """
        Initialize the Dataset from a SQLite database.
        """
        dataset = cls()
        manipulator = DatabaseManipulator(db_path)
        try:
            if 'metadata' in manipulator.get_table_names():
                data = manipulator.fetch_all("SELECT name, dataset, full_path, label_path, parent_folder FROM metadata")
            else:
                raise AttributeError(f"Table 'metadata' not found in database {db_path}.")
        except Exception as e:
            logging.error(f"Error loading database {db_path}: {e}")
            return dataset
        finally:
            manipulator.close_connection()

        dataset.dataset_name = os.path.basename(db_path).split('.')[0]
        for row in data:
            dataset[row[0]] = {
                'name': row[0],
                'dataset': row[1],
                'full_path': row[2],
                'relative_path': get_relative_path(row[2], row[1]),
                'detection': True if row[3] not in [None, "", "NULL"] else False,
                'label_full_path': row[3],
                'label_relative_path': get_relative_path(row[3], row[1]),
                'parent_folder': row[4]
            }
        return dataset

    @classmethod
    def from_yolo_txt(cls, train_file: str, val_file: Optional[str] = None,
                      test_file: Optional[str] = None, path_to_root: str = None) -> 'Dataset':
        """
        Initialize the Dataset from YOLO .txt files containing paths to images.
        """
        dataset = cls()

        def add_to_dataset(txt_file: str, set_type: str):
            with open(txt_file, 'r') as file:
                for line in file:
                    img_path = line.strip()
                    file_name = os.path.basename(img_path)
                    label_file = img_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                    dataset[file_name] = {
                        'name': file_name,
                        'dataset': set_type,
                        'full_path': img_path if not path_to_root else os.path.join(path_to_root, img_path),
                        'relative_path': img_path if path_to_root else os.path.relpath(img_path, path_to_root),
                        'detection': True if os.path.exists(label_file) and is_yolo_label(label_file) else False,
                        'label_full_path': label_file if not path_to_root else os.path.join(path_to_root, label_file) if os.path.exists(label_file) and is_yolo_label(label_file) else None,
                        'label_relative_path': label_file if path_to_root else os.path.relpath(label_file, path_to_root) if os.path.exists(label_file) and is_yolo_label(label_file) else None,
                        'parent_folder': os.path.basename(os.path.dirname(img_path))
                    }

        add_to_dataset(train_file, 'train')
        if val_file:
            add_to_dataset(val_file, 'val')
        if test_file:
            add_to_dataset(test_file, 'test')

        return dataset

    @property
    def size(self):
        return len(self)

    @property
    def train_size(self):
        return sum(1 for v in self.values() if v['dataset'] == 'train')

    @property
    def val_size(self):
        return sum(1 for v in self.values() if v['dataset'] == 'val')

    @property
    def test_size(self):
        return sum(1 for v in self.values() if v['dataset'] == 'test')

    def get_subset(self, subset_type: str) -> 'Dataset':
        if subset_type not in ['train', 'val', 'test']:
            raise ValueError("subset_type must be 'train', 'val', or 'test'")
        subset = Dataset()
        for key, value in self.items():
            if value['dataset'] == subset_type:
                subset[key] = value
        return subset

    @property
    def object(self):
        return sum(1 for v in self.values() if v['detection'])

    @property
    def empty(self):
        return sum(1 for v in self.values() if not v['detection'])

    @property
    def dataset_names(self):
        return set(v['dataset'] for v in self.values())

    @property
    def parent_folders(self):
        return set(v['parent_folder'] for v in self.values())

    def class_distribution(self, class_names: dict) -> dict:
        distribution = {class_name: 0 for class_name in class_names.values()}
        for file_info in self.values():
            try:
                if file_info['detection'] and file_info['label_full_path']:
                    with open(file_info['label_full_path'], 'r') as file:
                        for line in file:
                            class_id = int(line.strip().split()[0])
                            class_name = class_names[class_id]
                            distribution[class_name] += 1
            except Exception as e:
                logging.error(f"Error reading label file {file_info['label_full_path']}: {e}")
        return distribution

    @property
    def with_detection_boxes(self):
        """
        Returns the dataset with added 'detection_boxes' key for each entry.
        """
        self.construct_detection_boxes()
        return self

    def construct_detection_boxes(self):
        """
        Reads annotations from YOLO files and constructs DetectionBoxes objects.
        """
        from detectflow.utils.file import open_image, yolo_label_load
        from detectflow.predict.results import DetectionBoxes

        for file_info in self.values():
            if file_info['detection'] and file_info['label_full_path']:
                try:
                    image_path = file_info['full_path']
                    image_shape = open_image(image_path).shape[:2]
                    boxes = yolo_label_load(file_info['label_full_path'])
                    file_info['detection_boxes'] = DetectionBoxes.from_custom_format(boxes[:, 1:], tuple(image_shape),
                                                                        "nxywh") if boxes is not None else None
                except Exception as e:
                    logging.error(f"Error reading label file {file_info['label_full_path']}: {e}")
                    file_info['detection_boxes'] = None
            else:
                file_info['detection_boxes'] = None

    def get_random_subset(self, size: int, balanced: bool = False, by: str = 'dataset') -> 'Dataset':
        """
        Generate a random subset of the dataset.

        by: str valid keys of the file dictionary
        """
        if by in self.values():
            raise KeyError(f"Key '{by}' not found in dataset values.")

        if balanced:
            subsets = {k: [] for k in set(v[by] for v in self.values())}
            for k, v in self.items():
                subsets[v[by]].append(k)
            subset = Dataset()
            while len(subset) < size:
                for set_type, items in subsets.items():
                    if items and len(subset) < size:
                        chosen = random.choice(items)
                        subset[chosen] = self[chosen]
                        items.remove(chosen)
            return subset
        else:
            keys = random.sample(list(self.keys()), size)
            subset = Dataset()
            for key in keys:
                subset[key] = self[key]
            return subset

    def reorganize_files(self,
                         base_folder: str,
                         by: str,
                         keep_empty_separated: bool = True,
                         regex_pattern: Optional[str] = None):
        def move_file(file_info, dest_folder):
            os.makedirs(dest_folder, exist_ok=True)
            new_full_path = os.path.join(dest_folder, file_info['name'])
            shutil.move(file_info['full_path'], new_full_path)
            file_info['full_path'] = new_full_path
            file_info['relative_path'] = os.path.relpath(new_full_path, base_folder)

            if file_info['label_full_path']:
                new_label_full_path = os.path.join(dest_folder, os.path.basename(file_info['label_full_path']))
                shutil.move(file_info['label_full_path'], new_label_full_path)
                file_info['label_full_path'] = new_label_full_path
                file_info['label_relative_path'] = os.path.relpath(new_label_full_path, base_folder)

        def move_files_within_folder(folder, file_info):
            if keep_empty_separated:
                if file_info['detection']:
                    dest_folder = os.path.join(folder, 'object')
                else:
                    dest_folder = os.path.join(folder, 'empty')
            else:
                dest_folder = folder
            move_file(file_info, dest_folder)

        if by == 'empty_object':
            empty_folder = os.path.join(base_folder, 'empty')
            object_folder = os.path.join(base_folder, 'object')
            for file_info in self.values():
                try:
                    if file_info['detection']:
                        move_file(file_info, object_folder)
                    else:
                        move_file(file_info, empty_folder)
                except Exception as e:
                    logging.error(f"Error moving file {file_info['name']}: {e}")
        elif by == 'dataset_type':
            for file_info in self.values():
                dest_folder = os.path.join(base_folder, file_info['dataset'])
                try:
                    move_files_within_folder(dest_folder, file_info)
                except Exception as e:
                    logging.error(f"Error moving file {file_info['name']}: {e}")
        elif by == 'regex' and regex_pattern is not None:
            match_folder = os.path.join(base_folder, 'match')
            other_folder = os.path.join(base_folder, 'other')
            for key, file_info in self.items():
                try:
                    if re.match(regex_pattern, key):
                        move_files_within_folder(match_folder, file_info)
                    else:
                        move_files_within_folder(other_folder, file_info)
                except Exception as e:
                    logging.error(f"Error moving file {file_info['name']}: {e}")

    def split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float = 0.0):
        """
        Split the dataset into train, val, and test partitions based on given ratios.

        Args:
            train_ratio (float): Ratio of the dataset to be assigned to the training set.
            val_ratio (float): Ratio of the dataset to be assigned to the validation set.
            test_ratio (float): Ratio of the dataset to be assigned to the test set.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.0")

        file_list = list(self.keys())
        random.shuffle(file_list)

        num_files = len(file_list)
        train_end = int(train_ratio * num_files)
        val_end = train_end + int(val_ratio * num_files)

        for idx, file in enumerate(file_list):
            if idx < train_end:
                self[file]['dataset'] = 'train'
            elif idx < val_end:
                self[file]['dataset'] = 'val'
            else:
                self[file]['dataset'] = 'test'

    def generate_yolo_files(self, destination: str, classes: dict, absolute_paths: bool = False,
                            generate_yaml: bool = False):
        """
        Generate YOLO dataset files.

        Args:
            destination (str): The directory where the .txt and .yaml files will be saved.
            classes (dict): Dictionary of class names to be included in the .yaml file. {0: 'class1', 1: 'class2', ...}
            absolute_paths (bool): Whether to use absolute paths in the .txt files.
            generate_yaml (bool): Whether to generate the dataset .yaml file.
        """
        os.makedirs(destination, exist_ok=True)

        dataset_name = self.dataset_name if hasattr(self, 'dataset_name') else 'dataset'
        txt_files = {'train': os.path.join(destination, 'train.txt'),
                     'val': os.path.join(destination, 'val.txt'),
                     'test': os.path.join(destination, 'test.txt')}

        data_files = {'train': [], 'val': [], 'test': []}

        dataset_root = os.path.abspath(os.path.join(destination, os.pardir))

        for file_info in self.values():
            if file_info['dataset'] in data_files:
                if absolute_paths:
                    data_files[file_info['dataset']].append(os.path.abspath(file_info['full_path']))
                else:
                    data_files[file_info['dataset']].append(os.path.relpath(file_info['full_path'], dataset_root))

        for key, path_list in data_files.items():
            with open(txt_files[key], 'w') as f:
                for path in path_list:
                    f.write(f"{path}\n")

        if generate_yaml:
            yaml_content = {
                'path': os.path.relpath(dataset_root, destination) if not absolute_paths else dataset_root,
                'train': os.path.relpath(txt_files['train'], destination) if not absolute_paths else txt_files['train'],
                'val': os.path.relpath(txt_files['val'], destination) if not absolute_paths else txt_files['val'],
                'test': os.path.relpath(txt_files['test'], destination) if data_files['test'] else None,
                'names': classes
            }
            yaml_path = os.path.join(destination, f"{dataset_name}.yaml")
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    def visualize(self, key: str):
        """
        Visualize a sample image with bounding boxes.

        Args:
            key (str): Key of the image.
        """
        from detectflow.utils.file import open_image, yolo_label_load
        from detectflow.utils.inspector import Inspector

        image_path = self[key]['full_path']
        labels_path = self[key]['label_full_path']

        image = open_image(image_path)
        if labels_path and os.path.exists(labels_path):
            boxes = yolo_label_load(labels_path)
        else:
            boxes = None
        Inspector.display_frames_with_boxes(image, boxes)

    def sanity_check(self):
        """
        Perform sanity checks on the dataset.
        """
        from PIL import Image

        missing_labels = []
        corrupted_images = []

        for file_info in self.values():
            if file_info['detection'] and not os.path.exists(file_info['label_full_path']):
                missing_labels.append(file_info['full_path'])
            try:
                img = Image.open(file_info['full_path'])
                img.verify()
            except (IOError, SyntaxError) as e:
                corrupted_images.append(file_info['full_path'])

        print(f"Missing labels: {len(missing_labels)}")
        print(f"Corrupted images: {len(corrupted_images)}")
        return missing_labels, corrupted_images

    def balance_by_class(self, target_size: int):
        """
        Balance the dataset by class to the specified target size.

        Args:
            target_size (int): Target size for each class.
        """
        from collections import defaultdict

        class_counts = defaultdict(list)

        for file, file_info in self.items():
            if file_info['label_full_path']:
                try:
                    with open(file_info['label_full_path'], 'r') as f:
                        for line in f:
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id].append(file)
                except Exception as e:
                    logging.error(f"Error reading label file {file_info['label_full_path']}: {e}")

        balanced_files = []
        for class_id, files in class_counts.items():
            if len(files) > target_size:
                balanced_files.extend(random.sample(files, target_size))
            else:
                balanced_files.extend(files)

        balanced_dataset = Dataset()
        for file in balanced_files:
            balanced_dataset[file] = self[file]

        return balanced_dataset

    def export_annotations(self, export_format: str, export_path: str):
        """
        Export annotations in a different format.
        """
        raise NotImplementedError


# Function to process the folder
def get_dataset_database_from_files(folder_path, db_path):
    # Setup SQLite database
    manipulator = DatabaseManipulator(db_path)
    cols = [('name', 'TEXT'),
            ('dataset', 'TEXT'),
            ('full_path', 'TEXT'),
            ('label_path', 'TEXT'),
            ('parent_folder', 'TEXT')]
    manipulator.create_table('metadata', cols)

    # Walk the folder and populate database
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Get file information
                    file_info = get_file_data(folder_path, root, file)

                    # Get dataset name
                    dataset_name = os.path.basename(os.path.dirname(root))

                    # Insert record into the database
                    manipulator.add_to_batch('metadata', {'name': file, 'dataset': dataset_name,
                                                          'full_path': file_info.get('full_path', None),
                                                          'label_full_path': file_info.get('label_full_path', None),
                                                          'parent_folder': file_info.get('parent_folder', None)})
                except Exception as e:
                    logging.error(
                        f"Error processing file {file}: {e} \nNote that if the data insertion failed, emergency dumps may be available for manual reimport.")

    # manipulator cleanup
    manipulator.flush_batch()
    manipulator.close_connection()

    logging.info(f"Database populated with metadata from {folder_path}")


def get_dataset_database_from_dataset(dataset: Dataset, db_path: str):
    """
    Create a database from a Dataset instance.

    Args:
        dataset (Dataset): The dataset instance to create the database from.
        db_path (str): The path to the SQLite database.
    """
    # Setup SQLite database
    manipulator = DatabaseManipulator(db_path)
    cols = [
        ('name', 'TEXT'),
        ('dataset', 'TEXT'),
        ('full_path', 'TEXT'),
        ('label_path', 'TEXT'),
        ('parent_folder', 'TEXT')
    ]
    manipulator.create_table('metadata', cols)

    # Populate the database from the Dataset
    for file, file_info in dataset.items():
        try:
            manipulator.add_to_batch('metadata', {
                'name': file_info['name'],
                'dataset': file_info['dataset'],
                'full_path': file_info['full_path'],
                'label_path': file_info['label_full_path'],
                'parent_folder': file_info['parent_folder']
            })
        except Exception as e:
            logging.error(
                f"Error processing file {file}: {e}\nNote that if the data insertion failed, emergency dumps may be available for manual reimport.")

    # manipulator cleanup
    manipulator.flush_batch()
    manipulator.close_connection()

    logging.info(f"Database populated with metadata from the dataset")


def process_image_file(image_path: str):
    """
    Processes a single image file to extract its dimensions and associated bounding box data.
    """
    from detectflow.utils.file import yolo_label_load, open_image

    base_name, extension = os.path.splitext(image_path)
    annotation_path = f'{base_name}.txt'

    try:
        image_array = open_image(image_path)

        with Image.fromarray(image_array) as img:
            img_width, img_height = img.size

        bbox_data = []
        if os.path.exists(annotation_path):
            bboxes = yolo_label_load(annotation_path)
            bbox_count = len(bboxes)
            for bbox in bboxes:
                _, x_center, y_center, width, height = bbox
                bbox_data.append((x_center, y_center, width, height))
        else:
            bbox_count = 0
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        img_width, img_height = None, None
        bbox_count = 0
        bbox_data = []

    return (img_width, img_height), bbox_count, bbox_data


def update_heatmap_chunk(bbox_chunk, heatmap_size):
    """
    Updates a heatmap chunk with the given bounding box data.
    """
    heatmap = np.zeros(heatmap_size)
    for (x_center, y_center, width, height), (img_width, img_height) in bbox_chunk:
        x_center_pixel = int(x_center * img_width)
        y_center_pixel = int(y_center * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)

        x_min = max(0, x_center_pixel - width_pixel // 2)
        y_min = max(0, y_center_pixel - height_pixel // 2)
        x_max = min(heatmap_size[1], x_center_pixel + width_pixel // 2)
        y_max = min(heatmap_size[0], y_center_pixel + height_pixel // 2)

        heatmap[y_min:y_max, x_min:x_max] += 1
    return heatmap


from collections import defaultdict
from tqdm import tqdm

class DatasetDiagnoser:
    def __init__(self, dataset):
        self.dataset = dataset

    def collect_data(self):
        """
        Collects data from the dataset instance.
        """
        image_sizes = []
        bbox_counts = []
        bbox_positions = []
        class_counts = defaultdict(int)
        bbox_sizes = []
        image_brightness = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image_file, file_info['full_path'])
                       for file_info in self.dataset.values()]

            for future in tqdm(futures, desc="Processing images"):
                size, count, positions, classes, brightness = future.result()
                image_sizes.append(size)
                bbox_counts.append(count)
                bbox_positions.extend(positions)
                for class_id in classes:
                    class_counts[class_id] += 1
                bbox_sizes.extend([(w, h) for _, _, w, h in positions])
                image_brightness.append(brightness)

        return image_sizes, bbox_counts, bbox_positions, class_counts, bbox_sizes, image_brightness

    def process_image_file(self, image_path: str):
        """
        Processes a single image file to extract its dimensions and associated bounding box data.
        """
        from detectflow.utils.file import yolo_label_load, open_image

        base_name, extension = os.path.splitext(image_path)
        annotation_path = f'{base_name}.txt'

        try:
            image_array = open_image(image_path)

            with Image.fromarray(image_array) as img:
                img_width, img_height = img.size
                brightness = np.array(img.convert('L')).mean()

            bbox_data = []
            classes = []
            if os.path.exists(annotation_path):
                bboxes = yolo_label_load(annotation_path)
                bbox_count = len(bboxes)
                for bbox in bboxes:
                    class_id, x_center, y_center, width, height = bbox
                    bbox_data.append((x_center, y_center, width, height))
                    classes.append(class_id)
            else:
                bbox_count = 0
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            img_width, img_height = None, None
            brightness = None
            classes = []
            bbox_count = 0
            bbox_data = []

        return (img_width, img_height), bbox_count, bbox_data, classes, brightness

    def plot_image_size_distribution(self, image_sizes):
        """
        Plots the distribution of image sizes in a scatter plot.
        """
        widths, heights = zip(*image_sizes)
        plt.scatter(widths, heights)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Image Size Distribution')
        plt.show()

    def plot_bbox_count_histogram(self, bbox_counts):
        """
        Plots a histogram of the distribution of bounding box counts per image.
        """
        plt.hist(bbox_counts, bins=range(max(bbox_counts)+1))
        plt.xlabel('Number of Bounding Boxes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Bounding Box Counts per Image')
        plt.show()

    def plot_bbox_heatmap(self, bbox_positions, image_sizes, image_size=(640, 640)):
        """
        Plots a heatmap representing the coverage of bounding boxes across images.
        """
        num_cores = cpu_count()
        bbox_chunks = np.array_split(list(zip(bbox_positions, image_sizes)), num_cores)

        with Pool(num_cores) as pool:
            heatmaps = pool.starmap(update_heatmap_chunk, [(chunk, image_size, image_size) for chunk in bbox_chunks])

        combined_heatmap = np.sum(heatmaps, axis=0)

        plt.imshow(combined_heatmap, cmap='hot', interpolation='nearest')
        plt.title('Heatmap of Bounding Box Coverage')
        plt.show()

    def plot_class_distribution(self, class_counts, class_names):
        """
        Plots the distribution of classes in the dataset.
        """
        labels, counts = zip(*sorted(class_counts.items(), key=lambda x: x[0]))
        labels = [class_names[label] for label in labels]
        plt.bar(labels, counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=90)
        plt.show()

    def plot_bbox_size_distribution(self, bbox_sizes):
        """
        Plots the distribution of bounding box sizes.
        """
        widths, heights = zip(*bbox_sizes)
        plt.scatter(widths, heights)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Bounding Box Size Distribution')
        plt.show()

    def plot_aspect_ratio_distribution(self, bbox_sizes):
        """
        Plots the distribution of aspect ratios of bounding boxes.
        """
        aspect_ratios = [w / h for w, h in bbox_sizes]
        plt.hist(aspect_ratios, bins=50)
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Frequency')
        plt.title('Aspect Ratio Distribution of Bounding Boxes')
        plt.show()

    def plot_image_brightness_distribution(self, image_brightness):
        """
        Plots the distribution of average image brightness.
        """
        plt.hist(image_brightness, bins=50)
        plt.xlabel('Brightness')
        plt.ylabel('Frequency')
        plt.title('Image Brightness Distribution')
        plt.show()

    def detect_class_imbalance(self, class_counts, threshold=0.1):
        """
        Detects class imbalance in the dataset.
        """
        total = sum(class_counts.values())
        imbalance_classes = {class_id: count / total for class_id, count in class_counts.items() if count / total < threshold}
        if imbalance_classes:
            print("Classes with imbalance (less than {:.0%} of total):".format(threshold))
            for class_id, ratio in imbalance_classes.items():
                print(f"Class {class_id}: {ratio:.2%}")
        else:
            print("No class imbalance detected.")

