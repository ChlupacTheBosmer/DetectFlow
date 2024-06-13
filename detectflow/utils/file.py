import os


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

