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

def yolo_label_save(txt_file: str, boxes: np.ndarray):
    """
    Save YOLO formatted numpy array to txt file.
    Format: class x_center y_center width height (all values normalized)
    """
    with open(txt_file, 'w') as file:
        for box in boxes:
            file.write(' '.join(map(str, box)) + '\n')


def yolo_label_load(txt_file: str) -> np.ndarray:
    """
    Read YOLO formatted txt file and convert to numpy array.
    Format: class x_center y_center width height (all values normalized)
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        boxes.append(parts)
    return np.array(boxes)


