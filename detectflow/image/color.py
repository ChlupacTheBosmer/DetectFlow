from sklearn.cluster import KMeans
import numpy as np

def get_image_dominant_color(image, k=3, color_ranges=None):
    """
    Finds the dominant color in an image, excluding specified colors.
    :param image: Image from which to extract the dominant color.
    :param k: Number of clusters for KMeans.
    :param color_ranges: Dict with color ranges to exclude.
    :return: The dominant color in BGR format, or None if excluded.
    """
    if color_ranges is None:
        color_ranges = {
            'green': [(87, 162), (80, 255)],  # Hue range for green, Saturation range
            'brown': [(10, 20), (50, 255)],
            'black': [(0, 180), (0, 50)],  # Saturation less than 50 for black
            'grey': [(0, 180), (0, 50)]  # Low saturation for grey
        }

    # Reshape the image to be a list of pixels and apply KMeans
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Check each dominant color against the exclusion ranges
    for color in dominant_colors:
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
        h, s, v = hsv_color[0][0]
        excluded = False
        for h_range, s_range in color_ranges.values():
            if h_range[0] <= h <= h_range[1] and s_range[0] <= s <= s_range[1]:
                excluded = True
                break
        if not excluded:
            return color

    return None