import os
import numpy as np
from pathlib import Path
import json
import re

def compare_file_sizes(file1, file2, tolerance=0.01):
    """
    Compare the sizes of two files with a specified tolerance.

    Parameters:
    file1 (str): Path to the first file.
    file2 (str): Path to the second file.
    tolerance (float): Tolerance level for comparison (default is 0.01).

    Returns:
    bool: True if the file sizes are within the tolerance, False otherwise.
    """
    # Get the size of both files
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    # Calculate the absolute difference and the relative tolerance
    difference = abs(size1 - size2)
    max_size = max(size1, size2)

    # Check if the difference is within the tolerance
    if difference <= max_size * tolerance:
        return True
    else:
        return False


def json_save(file='data.json', data=None, header=''):
    """
    Save JSON data to a file.

    Args:
        file (str, optional): File name. Default is 'data.json'.
        data (dict): Data to save in JSON format.
        header (str, optional): JSON header to add (will be ignored in JSON).

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in JSON format
    with open(file, 'w', errors='ignore', encoding='utf-8') as f:
        if header:
            f.write(header)  # JSON doesn't support headers, but we write it as a comment
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_load(file='data.json', append_filename=False):
    """
    Load JSON data from a file.

    Args:
        file (str, optional): File name. Default is 'data.json'.
        append_filename (bool): Add the JSON filename to the JSON dictionary. Default is False.

    Returns:
        (dict): JSON data and file name.
    """
    assert Path(file).suffix == '.json', f'Attempting to load non-JSON file {file} with json_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add JSON filename to dict and return
        data = json.loads(s) or {}  # always return a dict (json.loads() may return None for empty files)
        if append_filename:
            data['json_file'] = str(file)
        return data


def yolo_label_save(txt_file: str, boxes: np.ndarray):
    """
    Save YOLO formatted numpy array to txt file.
    Format: class x_center y_center width height (all values normalized)
    """
    with open(txt_file, 'w') as file:
        for box in boxes:
            file.write(' '.join(map(str, box)) + '\n')


# def yolo_label_load(txt_file: str) -> np.ndarray:
#     """
#     Read YOLO formatted txt file and convert to numpy array.
#     Format: class x_center y_center width height (all values normalized)
#     """
#     with open(txt_file, 'r') as file:
#         lines = file.readlines()
#     boxes = []
#     for line in lines:
#         parts = list(map(float, line.strip().split()))
#         boxes.append(parts)
#     return np.array(boxes)

def yolo_label_load(txt_file: str) -> np.ndarray:
    """
    Read YOLO-formatted txt file and convert to a NumPy array.
    The first value may be an integer class id or a string label.
    The remaining values are floats (x_center, y_center, width, height, etc.),
    all presumably normalized in [0,1].
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue  # skip blank lines

        # The first token could be numeric or a string label
        first_val = parts[0]
        # Attempt to parse the remaining tokens as floats
        try:
            coords = list(map(float, parts[1:]))
        except ValueError as e:
            raise ValueError(
                f"Could not parse numeric coordinates for line: '{line}'. Error: {e}"
            )

        # Store the row with the first value as-is plus the numeric coords
        # We must use dtype=object later, because the array has both floats and a string/float in the first column
        row = [first_val] + coords
        boxes.append(row)

    # Convert to NumPy array with `dtype=object`,
    # so that string-labeled rows and float-labeled rows both work
    return np.array(boxes, dtype=object)

def is_yolo_label(file_path):
    try:
        with open(file_path, 'r') as file:
            # Check if the file contains any digits
            return any(re.search(r'\d', line) for line in file)
    except IOError:
        return False


def open_image(image_path):
    import cv2

    try:
        image = cv2.imread(image_path)
        return image
    except Exception as e:
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return image
