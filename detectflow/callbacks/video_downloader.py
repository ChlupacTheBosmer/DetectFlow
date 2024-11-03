from detectflow.video.video_data import Video
from detectflow.predict.predictor import Predictor
from detectflow.utils.inspector import Inspector
from detectflow.config import DETECTFLOW_DIR
import random
import traceback
import os
import logging


def gather_flower_data(**kwargs):
    video_path = kwargs.get('video_path', None)
    s3_path = kwargs.get('s3_path', None)
    output_path = kwargs.get('output_path', None)
    inspect = kwargs.get('inspect', False)

    if not video_path:
        return None

    frame_number = None
    try:
        # Init Video instance
        video = Video(video_path)

        # Define frame numbers to read
        frame_number = random.randint(0, video.total_frames - 1)

        # Extract frame from a video
        frame = video.read_video_frame(frame_number, stream=False)[0]['frame']

        # Init Predictor instance
        p = Predictor()

        # Run detection on the frame and store result
        results = []
        for result in p.detect(frame,
                               detection_conf_threshold=0.1,
                               model_path=os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt')):

            if result is not None:
                result.source_path = video_path
                result.s3_path = s3_path
                # result.source_name = os.path.basename(video_path)
                result.frame_number = frame_number
                result.save_dir = output_path

                result.save(sort=True, save_txt=True)

                if inspect:
                    Inspector.display_frames_with_boxes(result.orig_img, result.boxes)

            results.append(result)
    except Exception as e:
        logging.error(
            f"Error extracting a frame number: {frame_number} from a video {os.path.basename(video_path)}. Returning None. {e}")
        traceback.print_exc()
        results = None
    return results


from detectflow.utils.name import parse_video_name
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.process.database_manager import VIDEOS_COLS
from detectflow.utils.extract_data import extract_data_from_video


def diagnose_video(**kwargs):
    video_path = kwargs.get('video_path', None)
    s3_path = kwargs.get('s3_path', None)
    output_path = kwargs.get('output_path', None)
    frame_skip = kwargs.get('frame_skip', 15)
    motion_method = kwargs.get('motion_method', 'SOM')

    if not video_path:
        return False

    recording_id = parse_video_name(video_path).get('recording_id', None)

    if not recording_id:
        return False

    # init database
    db_path = os.path.join(output_path, f"{recording_id}.db")
    db = DatabaseManipulator(db_path)

    if 'videos' not in db.get_table_names():
        db.create_table('videos', VIDEOS_COLS)

    try:
        # Extract data from video
        data_entry = extract_data_from_video(video_path, s3_path=s3_path, frame_skip=frame_skip, motion_methods=motion_method)

        if data_entry:
            db.insert('videos', data_entry, update_on_conflict=True)

        logging.info(f"Data extracted from video {os.path.basename(video_path)}")

        return True
    except Exception as e:
        logging.error(f"Error extracting data from video {os.path.basename(video_path)}: {e}")
        traceback.print_exc()
        return False
