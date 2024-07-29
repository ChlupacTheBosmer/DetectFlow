import os
import logging
import re


def parse_video_id(video_id: str):

    try:
        if not is_valid_video_id(video_id):
            raise ValueError(f'Invalid video ID: {video_id}')
        locality, transect, plant_id, date, hour, minutes = video_id.split("_")
    except Exception as e:
        return {"recording_id": None,
                "video_id": video_id,
                "timestamp": None,
                "locality": None,
                "transect": None,
                "plant_id": None,
                "date": None,
                "hour": None,
                "minute": None}

    recording_identifier = "_".join([locality, transect, plant_id])
    timestamp = "_".join([date, hour, minutes])

    return {"recording_id": recording_identifier,
            "video_id": video_id,
            "timestamp": timestamp,
            "locality": locality,
            "transect": transect,
            "plant_id": plant_id,
            "date": date,
            "hour": hour,
            "minute": minutes}


def parse_recording_name(video_path: str):

    filename = os.path.basename(video_path)

    # Prepare name elements
    try:
        locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")
        file_extension = os.path.splitext(filename)[-1]
    except Exception as e:
        logging.warning(f'Unable to parse video recording name: {video_path}. Exception: {e}')
        return {"recording_id": os.path.splitext(filename)[0],
                "video_id": os.path.splitext(filename)[0],
                "timestamp": None,
                "locality": None,
                "transect": None,
                "plant_id": None,
                "date": None,
                "hour": None,
                "minute": None,
                "extension": os.path.splitext(filename)[-1]}

    # Define compound info
    recording_identifier = "_".join([locality, transect, plant_id])
    timestamp = "_".join([date, hour, minutes])

    return {"recording_id": recording_identifier,
            "video_id": os.path.splitext(filename)[0],
            "timestamp": timestamp,
            "locality": locality,
            "transect": transect,
            "plant_id": plant_id,
            "date": date,
            "hour": hour,
            "minute": minutes,
            "extension": file_extension}


def is_valid_recording_id(recording_id):
    # Recording ID has a format of XX(X)0_X0_XXXXXX00
    recording_id_pattern = r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{1,2}$'

    return re.match(recording_id_pattern, recording_id) is not None


def is_valid_video_id(video_id):
    # Video ID has a format of XX(X)0_X0_XXXXXX00_00000000_00_00
    video_id_pattern = (r'^[A-Za-z]{2,3}\d_[A-Za-z]\d_[A-Za-z]{6}\d{1,2}_'
                        r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])_'
                        r'([01]\d|2[0-3])_([0-5]\d)$')

    return re.match(video_id_pattern, video_id) is not None

