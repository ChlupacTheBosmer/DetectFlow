from detectflow.utils.excel import AnnotationFile
from detectflow.config import TESTS_DIR
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from detectflow.handlers.checkpoint_handler import CheckpointHandler


def read_videos_table(sqlite_db_path: str):
    conn = sqlite3.connect(sqlite_db_path)
    query = "SELECT video_id, recording_id, fps, start_time, end_time, length FROM videos"
    videos_df = pd.read_sql_query(query, conn)
    conn.close()
    return videos_df

def create_visits_table(sqlite_db_path: str):
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            frame_number INTEGER,
            video_time TEXT,
            life_time TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            recording_id TEXT,
            video_id TEXT,
            video_path TEXT,
            flower_bboxes TEXT,
            rois TEXT,
            all_visitor_bboxes TEXT,
            relevant_visitor_bboxes TEXT,
            visit_ids TEXT,
            on_flower BOOLEAN,
            flags TEXT
        )
    """)
    conn.commit()
    conn.close()


class SQLiteUpdaterWithCheckpoints(CheckpointHandler):
    def __init__(self, sqlite_db_path, checkpoint_file='checkpoint.json'):
        super().__init__(checkpoint_file)
        self.sqlite_db_path = sqlite_db_path

    def populate_visits_table(self, videos_df: pd.DataFrame, batch_size: int = 10000, frame_skip: int = 15):
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        start_video_index = self.get_checkpoint_data('start_video_index', 0)
        start_frame_number = self.get_checkpoint_data('start_frame_number', 0)

        visits_data = []

        for video_index, row in enumerate(videos_df.iterrows()):
            if video_index < start_video_index:
                continue

            _, row = row
            video_id = row['video_id']
            recording_id = row['recording_id']
            fps = row['fps']
            start_time = datetime.strptime(row['start_time'], '%Y-%m-%dT%H:%M:%S')
            length = row['length']
            total_frames = int(length * fps)

            for frame_number in range(0, total_frames, frame_skip):
                if video_index == start_video_index and frame_number < start_frame_number:
                    continue

                video_time = str(timedelta(seconds=frame_number / fps))
                life_time = start_time + timedelta(seconds=frame_number / fps)
                life_time_str = life_time.strftime('%H:%M:%S')
                year, month, day = life_time.year, life_time.month, life_time.day

                visits_data.append((
                    frame_number, video_time, life_time_str, year, month, day,
                    recording_id, video_id, '', json.dumps([]), json.dumps([]), json.dumps([]), json.dumps([]),
                    json.dumps([]), False, json.dumps([])
                ))

                # Insert in batches
                if len(visits_data) >= batch_size:
                    cursor.executemany("""
                        INSERT INTO visits (frame_number, video_time, life_time, year, month, day, recording_id, video_id, video_path,
                                            flower_bboxes, rois, all_visitor_bboxes, relevant_visitor_bboxes, visit_ids, on_flower, flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, visits_data)
                    conn.commit()  # Commit the transaction
                    visits_data = []
                    self.update_checkpoint(start_video_index=video_index, start_frame_number=frame_number)

            # Reset start_frame_number for the next video
            start_frame_number = 0
            self.update_checkpoint(start_video_index=video_index + 1, start_frame_number=start_frame_number)

        # Insert remaining data
        if visits_data:
            cursor.executemany("""
                INSERT INTO visits (frame_number, video_time, life_time, year, month, day, recording_id, video_id, video_path,
                                    flower_bboxes, rois, all_visitor_bboxes, relevant_visitor_bboxes, visit_ids, on_flower, flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, visits_data)
            conn.commit()  # Commit the transaction

        conn.close()

    def update_visits_table(self, annotation_df: pd.DataFrame, batch_size: int = 1000, frame_skip: int = 15):
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        videos_df = self.read_videos_table()

        start_annotation_index = self.get_checkpoint_data('start_annotation_index', 0)
        start_frame_number = self.get_checkpoint_data('start_frame_number_update', 0)

        update_data = []

        for idx, visit in enumerate(annotation_df.iterrows()):
            if idx < start_annotation_index:
                continue

            _, visit = visit
            visit_start_time = datetime.strptime(visit['ts'], '%Y%m%d_%H_%M_%S')
            duration = visit['duration']
            visit_end_time = visit_start_time + timedelta(seconds=duration)

            # Filter videos that could contain this visit
            overlapping_videos = videos_df[
                (videos_df['start_time'] <= visit_end_time.strftime('%Y-%m-%dT%H:%M:%S')) &
                (videos_df['end_time'] >= visit_start_time.strftime('%Y-%m-%dT%H:%M:%S'))
            ]

            for _, video in overlapping_videos.iterrows():
                video_id = video['video_id']
                fps = video['fps']
                video_start_time = datetime.strptime(video['start_time'], '%Y-%m-%dT%H:%M:%S')

                # Calculate frame numbers for the visit within this video
                frame_start = int((visit_start_time - video_start_time).total_seconds() * fps)
                frame_end = int((visit_end_time - video_start_time).total_seconds() * fps)

                for frame_number in range(frame_start, frame_end + 1, frame_skip):
                    if idx == start_annotation_index and frame_number < start_frame_number:
                        continue

                    cursor.execute("""
                        SELECT all_visitor_bboxes, relevant_visitor_bboxes, visit_ids FROM visits
                        WHERE frame_number = ? AND video_id = ?
                    """, (frame_number, video_id))
                    result = cursor.fetchone()

                    if result:
                        all_visitor_bboxes, relevant_visitor_bboxes, visit_ids = result
                        all_visitor_bboxes = json.loads(all_visitor_bboxes)
                        relevant_visitor_bboxes = json.loads(relevant_visitor_bboxes)
                        visit_ids = json.loads(visit_ids)

                        all_visitor_bboxes.append([idx + 1, 1, 1, 1])
                        relevant_visitor_bboxes.append([idx + 1, 1, 1, 1])
                        visit_ids.append(idx + 1)

                        update_data.append((
                            json.dumps(all_visitor_bboxes), json.dumps(relevant_visitor_bboxes), json.dumps(visit_ids),
                            frame_number, video_id
                        ))

                    if len(update_data) >= batch_size:
                        cursor.executemany("""
                            UPDATE visits
                            SET all_visitor_bboxes = ?, relevant_visitor_bboxes = ?, visit_ids = ?
                            WHERE frame_number = ? AND video_id = ?
                        """, update_data)
                        conn.commit()  # Commit the transaction
                        update_data = []
                        self.update_checkpoint(start_annotation_index=idx, start_frame_number_update=frame_number)

            # Reset start_frame_number for the next annotation
            start_frame_number = 0
            self.update_checkpoint(start_annotation_index=idx + 1, start_frame_number_update=start_frame_number)

        # Execute any remaining updates
        if update_data:
            cursor.executemany("""
                UPDATE visits
                SET all_visitor_bboxes = ?, relevant_visitor_bboxes = ?, visit_ids = ?
                WHERE frame_number = ? AND video_id = ?
            """, update_data)
            conn.commit()  # Commit the transaction

        conn.close()

    def read_videos_table(self):
        conn = sqlite3.connect(self.sqlite_db_path)
        query = "SELECT video_id, recording_id, fps, start_time, end_time, length FROM videos"
        videos_df = pd.read_sql_query(query, conn)
        conn.close()
        return videos_df



sqlite_db_path = os.path.join(TESTS_DIR, 'temp', 'CZ1_M1_AraHir01.db')
checkpoint_file = 'checkpoint.json'
videos_df = read_videos_table(sqlite_db_path)

updater = SQLiteUpdaterWithCheckpoints(sqlite_db_path, checkpoint_file)
create_visits_table(sqlite_db_path)
updater.populate_visits_table(videos_df, batch_size=10000, frame_skip=30)

# Assuming annotation_df is obtained from AnnotationFile
annotation_file = AnnotationFile(filepath=os.path.join(TESTS_DIR, 'temp', 'CZ1_M1_AraHir01.xlsx'))
annotation_df = annotation_file.dataframe
updater.update_visits_table(annotation_df, batch_size=1000, frame_skip=30)
