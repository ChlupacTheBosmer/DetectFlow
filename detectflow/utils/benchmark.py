import traceback
from detectflow.utils.excel import AnnotationFile
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from tqdm import tqdm
from detectflow.handlers.checkpoint_handler import CheckpointHandler
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.process.database_manager import VISITS_COLS, VISITS_CONSTR
from detectflow.utils.input import string_to_list
import logging


class DetectionBenchmarker(CheckpointHandler):
    def __init__(self, db_path: str, excel_path: str, batch_size: int = 1000, frame_skip: int = 15, checkpoint_file: str = 'checkpoint.json'):
        super().__init__(checkpoint_file)

        if not db_path.endswith('.db'):
            raise ValueError("Database path must end with '.db'")

        if not any([excel_path.endswith(ext) for ext in ['.xlsx', '.xls']]):
            raise ValueError("Excel path must end with '.xlsx' or '.xls'")

        self.db_path = db_path
        self.excel_path = excel_path
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.annotation_dataframe = None
        self.database_manipulator = DatabaseManipulator(db_path)

    def _create_ground_truth_table(self):
        try:
            self.database_manipulator.create_table('ground_truth', columns=VISITS_COLS, table_constraints=VISITS_CONSTR)
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to create ground_truth table: {e}") from e

    def _populate_ground_truth_table(self, annotation_df: pd.DataFrame):
        # conn = self.database_manipulator.conn
        # cursor = conn.cursor()
        videos_df = self._read_videos_table()

        start_annotation_index = self.get_checkpoint_data('start_annotation_index', 0)

        update_data = []
        for idx, visit in tqdm(enumerate(annotation_df.iterrows()), total=len(annotation_df), desc="Processing annotations"):
            if idx < start_annotation_index:
                continue

            try:
                _, visit = visit
                visit_start_time = datetime.strptime(visit['ts'], '%Y%m%d_%H_%M_%S')
                duration = visit['duration']
                visit_end_time = visit_start_time + timedelta(seconds=duration)

                # Filter videos that could contain this visit
                overlapping_videos = videos_df[
                    (videos_df['start_time'] <= visit_end_time) &
                    (videos_df['end_time'] >= visit_start_time)
                ]

                if overlapping_videos.empty:
                    logging.warning(f"No overlapping videos found for visit starting at {visit_start_time}")
                    continue

                for _, video in overlapping_videos.iterrows():
                    video_id = video['video_id']
                    fps = video['fps']
                    video_start_time = video['start_time']

                    # Calculate frame numbers for the visit within this video
                    frame_start = max(int((visit_start_time - video_start_time).total_seconds() * fps), 0)
                    frame_end = min(int((visit_end_time - video_start_time).total_seconds() * fps),
                                    int(video['length'] * fps) - 1)

                    for frame_number in range(frame_start, frame_end + 1, self.frame_skip):
                        video_time = str(timedelta(seconds=frame_number / fps)).split('.')[0]  # Removing milliseconds
                        life_time = video_start_time + timedelta(seconds=frame_number / fps)
                        life_time_str = life_time.strftime('%H:%M:%S')
                        year, month, day = life_time.year, life_time.month, life_time.day

                        query = """
                            SELECT all_visitor_bboxes, relevant_visitor_bboxes, visit_ids FROM ground_truth
                            WHERE frame_number = ? AND video_id = ?
                        """
                        result = self.database_manipulator.fetch_one(query=query, params=(frame_number, video_id))

                        if result:
                            all_visitor_bboxes, relevant_visitor_bboxes, visit_ids = result
                            all_visitor_bboxes = json.loads(all_visitor_bboxes)
                            relevant_visitor_bboxes = json.loads(relevant_visitor_bboxes)
                            visit_ids = json.loads(visit_ids)
                        else:
                            all_visitor_bboxes = []
                            relevant_visitor_bboxes = []
                            visit_ids = []

                        all_visitor_bboxes.append([idx + 1, idx + 1, idx + 1, idx + 1])
                        relevant_visitor_bboxes.append([idx + 1, idx + 1, idx + 1, idx + 1])
                        visit_ids.append(idx + 1)

                        recording_id = visit.get('recording_id', '')

                        update_data.append((
                            json.dumps(all_visitor_bboxes), json.dumps(relevant_visitor_bboxes), json.dumps(visit_ids),
                            frame_number, video_id, video_time, life_time_str, year, month, day,
                            recording_id, video_id, '', json.dumps([]), json.dumps([]), json.dumps([]),
                            json.dumps([]), json.dumps([True for _ in all_visitor_bboxes]), json.dumps([])
                        ))

                        if len(update_data) >= self.batch_size:
                            query = """
                            INSERT OR REPLACE INTO ground_truth (all_visitor_bboxes, relevant_visitor_bboxes, visit_ids, 
                                                                     frame_number, video_id, video_time, life_time, year, month, day,
                                                                     recording_id, video_id, video_path, flower_bboxes, rois, 
                                                                     all_visitor_bboxes, relevant_visitor_bboxes, on_flower, flags)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """
                            self.database_manipulator.safe_execute(sql=query, data=update_data)
                            update_data = []
                            self.update_checkpoint(start_annotation_index=idx)

                # Update checkpoint to start from next annotation
                self.update_checkpoint(start_annotation_index=idx + 1)
                start_annotation_index = idx + 1

            except Exception as e:
                logging.error(f"Error processing annotation {idx}: {e}")
                traceback.print_exc()
                continue  # Skip to the next annotation

        # Execute any remaining updates
        if update_data:
            try:
                # cursor.executemany("""
                #     INSERT OR REPLACE INTO ground_truth (all_visitor_bboxes, relevant_visitor_bboxes, visit_ids,
                #                                          frame_number, video_id, video_time, life_time, year, month, day,
                #                                          recording_id, video_id, video_path, flower_bboxes, rois,
                #                                          all_visitor_bboxes, relevant_visitor_bboxes,
                #                                          on_flower, flags)
                #     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                # """, update_data)
                # conn.commit()  # Commit the transaction
                query = """
                        INSERT OR REPLACE INTO ground_truth (all_visitor_bboxes, relevant_visitor_bboxes, visit_ids, 
                                                                 frame_number, video_id, video_time, life_time, year, month, day,
                                                                 recording_id, video_id, video_path, flower_bboxes, rois, 
                                                                 all_visitor_bboxes, relevant_visitor_bboxes, on_flower, flags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                self.database_manipulator.safe_execute(sql=query, data=update_data)
            except sqlite3.Error as e:
                logging.error(f"Failed to execute batch updates: {e}")

        # conn.close()
        self.database_manipulator.close_connection()

        # If processing is completed, remove the checkpoint file
        if start_annotation_index >= len(annotation_df):
            self.remove_checkpoint()

    def _read_videos_table(self):
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT video_id, recording_id, fps, start_time, end_time, length FROM videos"
            videos_df = pd.read_sql_query(query, conn)

            # Convert start_time and end_time to datetime objects for accurate sorting
            videos_df['start_time'] = pd.to_datetime(videos_df['start_time'])
            videos_df['end_time'] = pd.to_datetime(videos_df['end_time'])

            # Sort the videos dataframe by start time
            videos_df.sort_values(by='start_time', inplace=True)
        except Exception as e:
            logging.error(f"Error reading videos table: {e}")
            videos_df = pd.DataFrame()  # Return an empty DataFrame on error
        finally:
            conn.close()

        return videos_df

    def _read_annotation_table(self, excel_path: str):

        # Read the annotation file
        try:
            annotation_file = AnnotationFile(filepath=excel_path)
            annotation_df = annotation_file.dataframe
        except Exception as e:
            raise RuntimeError(f"Error reading annotation file: {e}") from e

        return annotation_df

    def get_ground_truth(self):

        # Read the annotations table
        try:
            self.annotation_dataframe = self._read_annotation_table(self.excel_path)
        except Exception as e:
            raise RuntimeError(f"Error when performing benchmarking: {e}") from e

        # Create the ground truth table
        try:
            self._create_ground_truth_table()
        except Exception as e:
            raise RuntimeError(f"Error creating ground truth table: {e}") from e

        # Update the ground truth table
        self._populate_ground_truth_table(annotation_df=self.annotation_dataframe)

        logging.info("Ground truth table created successfully")

    def benchmark(self, min_detections: int = 1, frame_gap: int = 15, output_dir: str = "plots") -> Dict[str, float]:
        conn = sqlite3.connect(self.db_path)
        try:
            visits_df = pd.read_sql_query("SELECT * FROM visits", conn)
            ground_truth_df = pd.read_sql_query("SELECT * FROM ground_truth", conn)

            # Merge the dataframes on the relevant columns
            merged_df = pd.merge(visits_df, ground_truth_df, on=['frame_number', 'video_id'], how='outer', suffixes=('_pred', '_true'))

            # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
            merged_df['TP'] = merged_df.apply(lambda row: row['visit_ids_pred'] == row['visit_ids_true'], axis=1)
            merged_df['FP'] = merged_df.apply(lambda row: pd.isnull(row['visit_ids_true']) and not pd.isnull(row['visit_ids_pred']), axis=1)
            merged_df['FN'] = merged_df.apply(lambda row: not pd.isnull(row['visit_ids_true']) and pd.isnull(row['visit_ids_pred']), axis=1)

            TP = merged_df['TP'].sum()
            FP = merged_df['FP'].sum()
            FN = merged_df['FN'].sum()

            # Calculate Precision, Recall, and F1 Score
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            # Check for detection presence with min_detections
            gt_visits = ground_truth_df['visit_ids'].explode().unique()
            det_visits = visits_df['visit_ids'].explode().unique()

            detected_visits = [visit for visit in gt_visits if det_visits.tolist().count(visit) >= min_detections]
            detection_presence = len(detected_visits) / len(gt_visits) if len(gt_visits) > 0 else 0

            # Calculate coverage
            coverage_data = []
            for visit in gt_visits:
                gt_frames = ground_truth_df[ground_truth_df['visit_ids'].apply(lambda x: visit in json.loads(x))]
                det_frames = visits_df[visits_df['visit_ids'].apply(lambda x: visit in json.loads(x))]
                coverage = len(det_frames) / len(gt_frames) if len(gt_frames) > 0 else 0
                coverage_data.append(coverage)

            avg_coverage = sum(coverage_data) / len(coverage_data) if coverage_data else 0

            results = {
                'True Positives': TP,
                'False Positives': FP,
                'False Negatives': FN,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Total Visits': gt_visits,
                'Detected Visits': detected_visits,
                'Detection Presence': detection_presence,
                'Average Coverage': avg_coverage
            }

            self._plot_coverage(merged_df, frame_gap, output_dir)

            return results

        except Exception as e:
            logging.error(f"Error comparing visits to ground truth: {e}")
            traceback.print_exc()
            return {}

        finally:
            conn.close()

    def _plot_coverage(self, merged_df: pd.DataFrame, frame_gap: int = 15, output_dir: str = "plots",
                       sleep_interval: float = 1.0):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a new datetime column from the ground truth data
        merged_df['datetime_true'] = pd.to_datetime(
            merged_df['year_true'].astype(str) + '-' +
            merged_df['month_true'].astype(str) + '-' +
            merged_df['day_true'].astype(str) + ' ' +
            merged_df['life_time_true']
        )

        # Sort the dataframe by datetime and frame_number
        merged_df.sort_values(by=['datetime_true', 'frame_number'], inplace=True)

        # Create an artificial index to represent the correct order of frames
        merged_df['artificial_index'] = range(len(merged_df))

        # Store the mapping of artificial_index to frame_number
        index_to_frame = merged_df.set_index('artificial_index')['frame_number'].to_dict()

        # Get unique ground truth visit IDs
        gt_visits = merged_df['visit_ids_true'].explode().unique()

        for visit in gt_visits:
            # Get frames for this visit based on ground truth visit IDs
            visit_frames = merged_df[merged_df['visit_ids_true'].apply(lambda x: visit in x)]

            if visit_frames.empty:
                continue

            fig, ax = plt.subplots(figsize=(15, 2))

            def get_intervals_with_heights(frames, visit_col, index_col, frame_gap):
                intervals = []
                heights = []
                start_index = None
                end_index = None
                height = 1
                for index, row in frames.iterrows():
                    frame_index = row[index_col]
                    visits = string_to_list(row[visit_col])

                    if len(visits) > 0:
                        current_height = 1 / len(visits)  # Use the number of visits as the height
                        if start_index is None:
                            start_index = frame_index
                            end_index = frame_index
                            height = current_height
                        elif (frame_index - end_index) <= frame_gap:
                            end_index = frame_index
                            height = min(height, current_height)
                        else:
                            intervals.append((start_index, end_index - start_index + 1))
                            heights.append(height)
                            start_index = frame_index
                            end_index = frame_index
                            height = min(height, current_height)
                    else:
                        if start_index is not None:
                            intervals.append((start_index, end_index - start_index + 1))
                            heights.append(height)
                            start_index = None
                            end_index = None
                if start_index is not None:
                    intervals.append((start_index, end_index - start_index + 1))
                    heights.append(height)
                return intervals, heights

            # Ground truth intervals
            gt_intervals, gt_heights = get_intervals_with_heights(visit_frames, 'visit_ids_true', 'artificial_index',
                                                                  frame_gap)
            if gt_intervals:
                ax.broken_barh(gt_intervals, (0, 1), facecolors='red', edgecolor='black')

            # Detection intervals
            det_intervals, det_heights = get_intervals_with_heights(visit_frames, 'visit_ids_pred', 'artificial_index',
                                                                    frame_gap)
            if det_intervals:
                ax.broken_barh(det_intervals, (1, 1), facecolors='green', edgecolor='black')

            # Set x-axis labels to frame numbers
            x_ticks = [interval[0] for interval in gt_intervals] + [interval[0] + interval[1] - 1 for interval in
                                                                    gt_intervals]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([index_to_frame[idx] for idx in x_ticks])

            ax.set_xlabel('Frame numbers')
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(['GT', 'DET'])
            ax.set_title(f'Visit {visit} Ground Truth (Red) vs Detections (Green)')

            # Add legend
            red_patch = mpatches.Patch(color='red', label='Ground Truth')
            green_patch = mpatches.Patch(color='green', label='Detection')
            plt.legend(handles=[red_patch, green_patch])

            # Save plot locally
            plot_path = os.path.join(output_dir, f'visit_{visit}.png')
            plt.savefig(plot_path)
            plt.close(fig)

            logging.info(f"Saved plot for visit {visit} at {plot_path}")

            # Throttle requests by sleeping for a short interval
            time.sleep(sleep_interval)

