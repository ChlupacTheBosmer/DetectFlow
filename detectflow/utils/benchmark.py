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
from detectflow.utils.data_processor import load_dataframe, str_to_bool_list, filter_int_by_bool
import logging
import ast


def fill_reciprocal(df: pd.DataFrame, col1: str, col2: str):
    df[col1] = df.apply(lambda row: row[col2] if pd.isna(row[col1]) and not pd.isna(row[col2]) else row[col1], axis=1)
    df[col2] = df.apply(lambda row: row[col1] if pd.isna(row[col2]) and not pd.isna(row[col1]) else row[col2], axis=1)
    return df


def merge_dataframes(visits_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                     suffixes: tuple[str | None, str | None] = ('_pred', '_true')):
    try:
        for df in [visits_df, ground_truth_df]:
            cols_to_list = [
                'all_visitor_bboxes',
                'relevant_visitor_bboxes',
                'flower_bboxes',
                'visit_ids'
            ]

            for col in cols_to_list:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # Convert lists of floats to ints, string representations of lists of bools to actual lists of bools. Create a new column 'relevant_visit_ids' by filtering 'visit_ids' by 'on_flower'
            df[f'visit_ids'] = df[f'visit_ids'].apply(lambda x: [int(visit_id) for visit_id in x])
            df[f'on_flower'] = df[f'on_flower'].apply(str_to_bool_list)
            df[f'relevant_visit_ids'] = df.apply(lambda row: filter_int_by_bool(row[f'visit_ids'], row[f'on_flower']),
                                                 axis=1)
    except Exception as e:
        logging.error(
            f'Error converting string representations of lists to actual lists. Further exceptions possible: {e}')
        traceback.print_exc()

    # Merge the dataframes on the relevant columns
    try:
        merged_df = pd.merge(visits_df, ground_truth_df, on=['frame_number', 'video_id'], how='outer',
                             suffixes=suffixes)
    except Exception as e:
        raise RuntimeError(f'Error merging dataframes. Manual intervention required: {e}') from e

    try:
        # Fill missing values in the relevant columns
        cols_to_fill = [
            'visit_ids',
            'relevant_visit_ids',
            'all_visitor_bboxes',
            'relevant_visitor_bboxes',
            'flower_bboxes',
            'on_flower',
            'flags',
            'rois'
        ]

        for col in cols_to_fill:
            for suf in suffixes:
                merged_df[f'{col}{suf}'] = merged_df[f'{col}{suf}'].apply(
                    lambda x: [] if not isinstance(x, (list, tuple)) and pd.isna(x) else x)

        # Fill missing values in the relevant columns
        cols_to_recip = [
            'video_time',
            'year',
            'month',
            'day',
            'life_time',
            'recording_id',
            'video_path'
        ]

        for col in cols_to_recip:
            merged_df = fill_reciprocal(merged_df, f'{col}{suffixes[0]}', f'{col}{suffixes[1]}')
    except Exception as e:
        logging.error(f'Error filling missing values in the relevant columns. Further exceptions possible: {e}')
        traceback.print_exc()

    # Fill missing values reciprocally between the two columns
    try:
        cols_to_int = [
            'year',
            'month',
            'day'
        ]

        for col in cols_to_int:
            for suf in suffixes:
                merged_df[f'{col}{suf}'] = merged_df[f'{col}{suf}'].astype(int)
    except Exception as e:
        logging.error(f'Error converting columns to int. Further exceptions possible: {e}')
        traceback.print_exc()

    return merged_df


def calculate_metrics(merged_df: pd.DataFrame, column='visit_ids', suffixes=('_pred', '_true'), min_detections=1):
    pred_col = f'{column}{suffixes[0]}'
    true_col = f'{column}{suffixes[1]}'

    results = {}

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    try:
        merged_df['TP'] = merged_df.apply(lambda row: len(row[pred_col]) == len(row[true_col]), axis=1)
        TP = merged_df['TP'].sum()
        results['TP'] = TP
    except Exception as e:
        raise RuntimeError(f'Error calculating True Positives: {e}') from e
    try:
        merged_df['PTP'] = merged_df.apply(
            lambda row: len(row[pred_col]) > 0 and 0 < len(row[true_col]) != len(row[pred_col]), axis=1)
        PTP = merged_df['PTP'].sum()
        results['PTP'] = PTP
    except Exception as e:
        raise RuntimeError(f'Error calculating Partial True Positives: {e}') from e
    try:
        merged_df['FP'] = merged_df.apply(lambda row: len(row[true_col]) == 0 and not len(row[pred_col]) == 0, axis=1)
        FP = merged_df['FP'].sum()
        results['FP'] = FP
    except Exception as e:
        raise RuntimeError(f'Error calculating False Positives: {e}') from e
    try:
        merged_df['FN'] = merged_df.apply(lambda row: not len(row[true_col]) == 0 and len(row[pred_col]) == 0, axis=1)
        FN = merged_df['FN'].sum()
        results['FN'] = FN
    except Exception as e:
        raise RuntimeError(f'Error calculating False Negatives: {e}') from e

    # Calculate Precision, Recall, and F1 Score
    try:
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1_score
    except Exception as e:
        raise RuntimeError(f'Error calculating Precision, Recall, and F1 Score: {e}') from e

    try:
        # Get all visit numbers
        gt_visits = merged_df[true_col].explode().unique()
        gt_visits = [visit for visit in gt_visits if isinstance(visit, int)]

        detected_visits = []
        coverage = {}
        for visit in gt_visits:
            try:
                _ = int(visit)
            except ValueError:
                continue
            # Filter visit
            filtered_df = merged_df[merged_df[true_col].apply(lambda x: visit in x)]
            gt_count = filtered_df[true_col].apply(lambda x: len(x) > 0).sum()
            non_empty_count = filtered_df[pred_col].apply(lambda x: len(x) > 0).sum()
            coverage[visit] = non_empty_count / gt_count if gt_count > 0 else 0
            if non_empty_count > min_detections:
                detected_visits.append(visit)

        results['coverage_data'] = coverage
        results['avg_coverage'] = sum(coverage.values()) / len(coverage) if coverage else 0
        results['visits'] = gt_visits
        results['detected_visits'] = detected_visits
    except Exception as e:
        raise RuntimeError(f'Error calculating coverage: {e}') from e

    return results


def plot_coverage(merged_df: pd.DataFrame,
                  frame_gap: int = 15,
                  output_dir: str = "plots",
                  sleep_interval: float = 1.0,
                  column: str = 'visit_ids',
                  suffixes: tuple[str | None, str | None] = ('_pred', '_true')):

    pred_col = f'{column}{suffixes[0]}'
    true_col = f'{column}{suffixes[1]}'

    output_dir = os.path.join(output_dir, column)
    os.makedirs(output_dir, exist_ok=True)

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
    gt_visits = merged_df[true_col].explode().unique()
    gt_visits = [visit for visit in gt_visits if isinstance(visit, int)]

    for visit in gt_visits:
        # Get frames for this visit based on ground truth visit IDs
        visit_frames = merged_df[merged_df[true_col].apply(lambda x: visit in x)]

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
                visits = row[visit_col]

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
        gt_intervals, gt_heights = get_intervals_with_heights(visit_frames, true_col, 'artificial_index',
                                                              frame_gap)
        if gt_intervals:
            ax.broken_barh(gt_intervals, (0, 1), facecolors='red', edgecolor='black')

        # Detection intervals
        det_intervals, det_heights = get_intervals_with_heights(visit_frames, pred_col, 'artificial_index',
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
        plot_path = os.path.join(output_dir, f'{column}_{visit}.png')
        plt.savefig(plot_path)
        plt.close(fig)

        logging.info(f"Saved plot for visit {visit} at {plot_path}")

        # Throttle requests by sleeping for a short interval
        time.sleep(sleep_interval)


def round_to_nearest(number, frame_skip):
    return frame_skip * round(number / frame_skip)


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
            self.database_manipulator.delete_table('ground_truth')
        except sqlite3.Error as e:
            logging.warning(f"Failed to delete ground_truth table: {e}")

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
                        frame_number = round_to_nearest(frame_number, self.frame_skip)
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

                        all_visitor_bboxes.append([idx + 1, idx + 1, idx + 2, idx + 2])
                        relevant_visitor_bboxes.append([idx + 1, idx + 1, idx + 2, idx + 2])
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

        # remove the checkpoint file
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

    def benchmark(self, min_detections: int = 1, frame_gap: int = 15, output_dir: str = "plots", suffixes: tuple[str | None, str | None] = ('_pred', '_true')):
        try:
            visits_df = load_dataframe(self.db_path, 'visits')
            ground_truth_df = load_dataframe(self.db_path, 'ground_truth')
        except Exception as e:
            logging.critical(f'Error loading dataframes. Benchmarking not possible: {e}')
            traceback.print_exc()
            return

        try:
            merged_df = merge_dataframes(visits_df, ground_truth_df, suffixes=suffixes)
        except Exception as e:
            logging.critical(f'Error merging dataframes. Benchmarking not possible: {e}')
            traceback.print_exc()
            return

        try:
            all_results = calculate_metrics(merged_df, min_detections=min_detections, column='visit_ids', suffixes=suffixes)
            rel_results = calculate_metrics(merged_df, min_detections=min_detections, column='relevant_visit_ids', suffixes=suffixes)
        except Exception as e:
            logging.critical(f'Error calculating metrics. Benchmarking failed: {e}')
            traceback.print_exc()
            return None, None

        try:
            plot_coverage(merged_df, frame_gap, output_dir, column='visit_ids', suffixes=suffixes)
            plot_coverage(merged_df, frame_gap, output_dir, column='relevant_visit_ids', suffixes=suffixes)
        except Exception as e:
            logging.error(f"Error comparing visits to ground truth: {e}")
            traceback.print_exc()

        return all_results, rel_results



