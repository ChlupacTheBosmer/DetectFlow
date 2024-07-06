from detectflow.process.dataset_source_processor import DatasetSourceProcessor
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.manipulators.frame_manipulator import FrameManipulator
from detectflow.manipulators.manipulator import Manipulator
from detectflow.validators.validator import Validator
from detectflow.manipulators.input_manipulator import InputManipulator
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.image.motion_enrich import MotionEnrich, MotionEnrichResult
import os
import re
import logging


class DatasetMotionEnricher(DatasetSourceProcessor):
    def __init__(self,
                 database_path: str,
                 new_dataset_name: str,
                 working_dir: str,
                 max_workers: int = 1,
                 checkpoint_file: str = "DatasetMotionEnricher_checkpoint.json",
                 **kwargs):
        super().__init__(database_path, working_dir, max_workers, checkpoint_file, **kwargs)

        self.new_dataset_name = new_dataset_name

    def _prepare_folders(self, parent_dir, folder_names):
        folders = Manipulator.create_folders(directories=folder_names, parent_dir=parent_dir)
        if not folders and len(folders) > 0:
            raise RuntimeError(f"Failed to create destination folders: {folder_names}")
        else:
            return folders

    def _create_dataset_database(self, output_dir, dataset_name):
        db_man = DatabaseManipulator(os.path.join(output_dir, f"{dataset_name}.db"))

        columns = [
            ("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
            ("recording_id", "TEXT", ""),
            ("video_file_id", "TEXT", ""),
            ("frame_no", "INTEGER", ""),
            ("visit_no", "INTEGER", ""),
            ("crop_no", "INTEGER", ""),
            ("x1", "INTEGER", ""),
            ("y1", "INTEGER", ""),
            ("x2", "INTEGER", ""),
            ("y2", "INTEGER", ""),
            ("full_path", "TEXT", ""),
            ("label_path", "TEXT", ""),
            ("time", "TEXT", ""),
            ("parent_folder", "TEXT", "")
        ]
        db_man.create_table("metadata", columns)
        db_man.close_connection()

        return os.path.join(output_dir, f"{dataset_name}.db")

    def _cleanup_dbs(self, database_manipulators):
        # Flush batches
        for db in database_manipulators:
            db.flush_batch()
            db.close_connection()
            del db

    # @profile_memory(logging.getLogger(__name__))
    def _save_frame(self, frame, metadata, output_paths, database_manipulator):

        empty_folder = output_paths[0]
        visitor_folder = output_paths[1]

        rec_folder, video_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path, parent_folder = metadata

        # Prepare IDS
        video_id = InputManipulator.zero_pad_id(video_id)
        recording_id = '_'.join(video_id.split('_')[:3])

        # Save the result frame
        try:
            filename = f"{video_id}_{crop_no}_{frame_no}_{visit_no}_{x1},{y1}_{x2},{y2}"
            destination = visitor_folder if label_path else empty_folder
            suc = FrameManipulator.save_frame(frame, filename, destination)
        except Exception as e:
            raise RuntimeError(f"Error saving frame: {e}") from e

        # Get label file, copy it and rename it to the same name as the frame
        new_label_path = None
        if label_path:
            org_path = os.path.join(r"/storage/brno2/home/USER/datasets/Model training frames",
                                    re.sub(r'\d+', '', recording_id.split('_')[0].upper()), parent_folder, "visitor",
                                    os.path.basename(label_path.replace('\\', '/')))
            if Validator.is_valid_file_path(org_path):
                new_label_path = Manipulator.move_file(org_path, dest_path=destination, filename=f"{filename}.txt",
                                                       overwrite=True, copy=True)
            else:
                raise FileNotFoundError(f"Failed to locate label file: {org_path}")

            if not new_label_path:
                raise RuntimeError(f"Failed to copy label file: {org_path}")

        # Update database
        frame_data = {'recording_id': recording_id,
                      'video_file_id': video_id,
                      'frame_no': frame_no,
                      'visit_no': visit_no,
                      'crop_no': crop_no,
                      'x1': x1,
                      'y1': y1,
                      'x2': x2,
                      'y2': y2,
                      'full_path': os.path.join(destination, filename, '.png'),
                      'label_path': new_label_path if new_label_path else "NULL",
                      'time': '_'.join(video_id.split('_')[4:]).replace('_', ':'),
                      'parent_folder': recording_id
                      }
        database_manipulator.add_to_batch("metadata", frame_data)

    # @profile_memory(logging.getLogger(__name__))
    def process_video(self, video_id, frames_data, video_dir, dest_dirs, database_paths):
        print(f"Video ID: {video_id}, Frames: {len(frames_data)}")

        # Sort frames_data by frame number
        frames_data = sorted(frames_data, key=lambda item: item[2])

        # Prepare IDS
        org_id = video_id
        video_id = InputManipulator.zero_pad_id(video_id)
        recording_id = '_'.join(video_id.split('_')[:3])

        # Init manipulator and dataloader and database manipulator
        try:
            s3_man = S3Manipulator()
            dataloader = Dataloader()
            database_manipulators = [DatabaseManipulator(database_paths[0]), DatabaseManipulator(database_paths[1])]
        except Exception as e:
            logging.error(f"Unexpected error when creating helper class instances: {e} Video not processed: {video_id}")
            return

        # Get filepath for the video file
        try:
            bucket = InputManipulator.get_bucket_name_from_id(recording_id)
            pattern = rf"{InputManipulator.escape_string(video_id)}\.(mp4|avi)$"
            if Validator.is_valid_regex(pattern):
                files = s3_man.list_files_s3(bucket_name=bucket, folder_name=recording_id, regex=pattern,
                                             return_full_path=True)
            else:
                raise ValueError(f"Invalid reggex pattern: {pattern}.")

            if not files or not len(files) > 0:
                raise ValueError(
                    f"No matching video files found. Bucket: {bucket}, Folder: {recording_id}, Regex: {pattern}.")
        except Exception as e:
            logging.error(f"Unexpected error when searching for video file: {e} Video not processed: {video_id}")
            return

        # Prepare the video
        try:
            valid, _ = dataloader.prepare_videos(video_paths=files, target_directory=video_dir, remove_invalid=False)
            if not valid or len(valid) == 0:
                raise RuntimeError(f"Videos could not be prepared and validated: {valid}")
            else:
                video_path = valid[0]
        except Exception as e:
            logging.error(f"Unexpected error when preparing video file: {e} Video not processed: {video_id}")
            return

        # Run the frame enrichment from the video
        metadata = [(item[2], (item[5], item[6], item[7], item[8])) for item in frames_data]
        config = {"buffer_size": 30,
                  "preload_frames": 100,
                  "backSub_history": 30,
                  "cluster_threshold": 100,
                  "alpha": 0.4,
                  "frame_skip": 1,
                  "method": "decord"}
        for i, result in enumerate(MotionEnrich(video_path, metadata, **config).run()):
            error_frame = None
            try:
                if isinstance(result, MotionEnrichResult):
                    if result.frame_number:
                        error_frame = result.frame_number
                        # meta_idx = [item[2] for item in frames].index(result.frame_number)
                    else:
                        raise ValueError("Unknown MotionEnrichResult frame number. Data not saved.")

                    # Save frames and update database
                    self._save_frame(result.rgb_frame, frames_data[i], dest_dirs[0], database_manipulators[0])
                    self._save_frame(result.grey_frame, frames_data[i], dest_dirs[1], database_manipulators[1])

                else:
                    raise TypeError(f"Unexpected MotionEnrichResult type: {type(result)}")
            except Exception as e:
                logging.error(f"Unexpected error when processing MotionEnrichResult: {e} Data not saved: {error_frame}")

        # Cleanup DBs
        self._cleanup_dbs(database_manipulators)

        # Update checkpoint
        try:
            done_videos = self.get_checkpoint_data("done")
            if done_videos is not None and isinstance(done_videos, list):
                done_videos.append(org_id)
            else:
                done_videos = [org_id]
            self.update_checkpoint(done=done_videos)
        except Exception as e:
            logging.error(f"Unexpected error when saving progress: {e} Progress not saved: {org_id}")

    def run(self):

        # Prepare dirs and db_manipulators
        par_dirs = []
        dest_dirs = []
        database_paths = []
        video_dir = os.path.join(self.working_dir, "videos")
        for code in ["RGB", "GRAY"]:
            cur_dir = os.path.join(self.working_dir, code)
            par_dirs.append(cur_dir)
            dest_dirs.append(self._prepare_folders(cur_dir, ['empty', 'visitor']))
            database_paths.append(self._create_dataset_database(self.working_dir, f"{code}_{self.new_dataset_name}"))

        # Get dataset data from database
        columns = ["recording_id", "video_file_id", "frame_no", "visit_no", "crop_no", "x1", "y1", "x2", "y2",
                   "full_path", "label_path", "parent_folder"]
        dataset_data = self.get_dataset_data("metadata", columns)

        # Sort dataset data
        frames_by_video = self.sort_dataset_data(dataset_data, sort_index=1)

        # Create a new dictionary without the excluded keys from the checkpoint file
        done_videos = self.get_checkpoint_data("done") if self.get_checkpoint_data("done") is not None else []
        frames_by_video = {key: frames_by_video[key] for key in frames_by_video if
                           key not in done_videos}  # Debug: and key in ["CZ1_M1_AreSer2_20210517_13_40"]

        # Process dataset
        self.process_dataset(workers=self.max_workers, frames_by_video=frames_by_video, video_dir=video_dir,
                             dest_dirs=dest_dirs, database_paths=database_paths)
