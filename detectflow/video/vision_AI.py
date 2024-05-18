# Extra packages
from google.cloud import vision
import cv2

# Default part of python
import re
from datetime import datetime
import os
from tkinter import filedialog
import shutil

def get_grouped_rois_from_frame(frames, unique_rois, grouping_radius_dimensions):

    grouped_centers = []

    # if isinstance(frames, np.ndarray):  # Check if frames is a single frame
    #     frames = [frames]  # Convert single frame to a list

    # for frame in frames:
    # Group centers that are close together within grouping_radius distance
    for center, overlap in zip(unique_rois, grouping_radius_dimensions):
        group_found = False
        for group in grouped_centers:
            if any(abs(center[0] - c[0]) < (overlap[0] // 2) and abs(center[1] - c[1]) < (overlap[1] // 2) for c in grouped_centers):
                grouped_centers.append(center)
                group_found = True
                break
        if not group_found:
            grouped_centers.append(center)

    return grouped_centers

def get_unique_rois_from_frame(frame, min_confidence: float = 0.2):

    script_directory = os.path.dirname(os.path.abspath(__file__))
    two_parent_folders_up = os.path.abspath(os.path.join(script_directory, '..', '..'))
    key_path = os.path.join(two_parent_folders_up, 'resources', 'key', 'ICVT.json')

    if not os.path.exists(key_path):
        install_google_api_key()

    # Set up the client using your API key or service account credentials
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)

    # Get frame shape
    image_height, image_width, _ = frame.shape

    # Encode the frame as an image in memory
    _, encoded_frame = cv2.imencode(".jpg", frame)

    # Convert the encoded image to bytes
    image_content = encoded_frame.tobytes()

    # Create an image object
    image = vision.Image(content=image_content)

    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Extract the flower bounding rectangles
    flower_centers = []
    if objects:
        for obj in objects:
            if obj.name == "Flower" and obj.score >= min_confidence:
                vertices = obj.bounding_poly.normalized_vertices
                x_min = int(vertices[0].x * image_width)
                y_min = int(vertices[0].y * image_height)
                x_max = int(vertices[2].x * image_width)
                y_max = int(vertices[2].y * image_height)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                flower_centers.append((center_x, center_y))
    else:
        pass
    return flower_centers

def get_text_with_OCR(frame):

    script_directory = os.path.dirname(os.path.abspath(__file__))
    two_parent_folders_up = os.path.abspath(os.path.join(script_directory, '..', '..'))
    key_path = os.path.join(two_parent_folders_up, 'resources', 'key', 'ICVT.json')

    if not os.path.exists(key_path):
        install_google_api_key()

    # Set up the client using your API key or service account credentials
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)

    # Encode the frame as an image in memory
    _, encoded_frame = cv2.imencode(".jpg", frame)

    # Convert the encoded image to bytes
    image_content = encoded_frame.tobytes()

    # Create an image object
    image = vision.Image(content=image_content)

    # Perform OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extract and print the detected text
    for text in texts:
        times = extract_time_from_text(text.description)

    if times:
        print(f"Valid time detected with OCR: {times[0].time()}")


    if times:
        return True, times[0].time()
    else:
        return False, None


def extract_time_from_text(text):
    time_pattern = re.compile(r'(?<!\d)(\d{1,2}:\d{2}:\d{2})(?!\d)')  # Pattern for h:mm:ss or hh:mm:ss

    matches = re.findall(time_pattern, text)
    valid_times = []

    for match in matches:
        parts = match.split(':')
        if len(parts) == 3:
            try:
                hours, minutes, seconds = map(int, parts)
                if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                    time_obj = datetime.strptime(match, "%H:%M:%S")
                    valid_times.append(time_obj)
            except ValueError:
                pass

    if not valid_times:
        return None  # No valid time found
    else:
        return valid_times

def install_google_api_key():

    # Open a file dialog to get the new file path
    new_file_path = filedialog.askopenfilename()

    # Get the root directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    two_parent_folders_up = os.path.abspath(os.path.join(script_directory, '..', '..'))

    # Create the target directory if it doesn't exist
    target_directory = os.path.join(two_parent_folders_up, "resources", "key")
    os.makedirs(target_directory, exist_ok=True)

    # Move the file to the target directory
    target_file_path = os.path.join(target_directory, os.path.basename(new_file_path))
    shutil.move(new_file_path, target_file_path)

    print(f"File moved to: {target_file_path}")
