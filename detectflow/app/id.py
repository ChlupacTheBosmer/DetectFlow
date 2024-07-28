import sys
import PyQt6
import os
pyqt = os.path.dirname(PyQt6.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt6/plugins")
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from detectflow.app.video_player import InteractiveVideoPlayer
from detectflow.app.visits_view import CustomWidget
from detectflow.app.species_view import EditableImageView
import traceback
import logging
import sys
import sqlite3
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLineEdit, QFormLayout, QCheckBox, QToolBar, QPushButton, QFileDialog, QHeaderView, QLabel
)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, QSize
from PyQt6.QtGui import QIcon
from detectflow.resources import ALT_ICONS
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QFormLayout, QCheckBox, QGroupBox, QVBoxLayout, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtWidgets import QMessageBox
from detectflow.utils.data_processor import VisitsProcessor, refine_periods, refine_visits
import os
from PyQt6.QtWidgets import QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QWidget, QListWidgetItem, QSizePolicy
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog, QDialog, QVBoxLayout, QCheckBox, QPushButton
import ast
from detectflow.app import VISIT_VIEW_HELP
from PyQt6.QtWidgets import QStackedWidget
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton
from PyQt6.QtWidgets import QStatusBar
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import QThreadPool
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject, QThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_video_id = None

        # Set window title and initial size
        self.setWindowTitle('DetectFlow ID')
        self.setGeometry(100, 100, 1200, 800)

        # Main container widget
        container = QWidget()
        self.setCentralWidget(container)

        # Main layout
        main_layout = QVBoxLayout(container)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Create top splitter
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create CustomWidget and InteractiveVideoPlayer instances
        self.custom_widget = CustomWidget(parent=self)
        self.video_player = InteractiveVideoPlayer()

        # Add widgets to the top splitter
        top_splitter.addWidget(self.video_player)
        top_splitter.addWidget(self.custom_widget)

        # Set initial sizes for the top splitter
        top_splitter.setSizes([800, 400])

        # Add the top splitter to the main splitter
        main_splitter.addWidget(top_splitter)

        # Add an empty widget for the bottom part
        bottom_widget = EditableImageView()
        main_splitter.addWidget(bottom_widget)

        # Set initial sizes for the main splitter
        main_splitter.setSizes([600, 200])

        # Add the main splitter to the main layout
        main_layout.addWidget(main_splitter)

        # Connect signals if necessary
        self.connect_signals()

    def connect_signals(self):
        self.custom_widget.update_video_id_signal.connect(self.handle_video_id_update)
        self.custom_widget.seek_video_signal.connect(self.seek_video)
        self.custom_widget.update_visits_signal.connect(self.update_visits)
        self.custom_widget.update_flowers_signal.connect(self.update_flowers)
        self.video_player.current_visits_changed.connect(self.custom_widget.update_current_visits)

    def handle_video_id_update(self, video_id):
        print(f"Received video ID: {video_id}")

        if self.current_video_id == video_id:
            return

        self.current_video_id = video_id

        s3_config = {
            'host_base': self.video_player.host_base,
            'use_https': self.video_player.use_https,
            'access_key': self.video_player.access_key,
            'secret_key': self.video_player.secret_key,
            'host_bucket': self.video_player.host_bucket
        }
        self.video_opener = VideoOpener(video_id,
                                        self.video_player.video_files,
                                        self.video_player.download_videos,
                                        self.video_player.app_data_dir,
                                        s3_config)
        self.video_opener.video_found.connect(self.open_video)
        self.video_opener.start()

    def open_video(self, file_path, index):
        self.video_player.open_file(file_path)
        self.video_player.current_video_index = index

    def seek_video(self, frame_number):
        if self.video_player.video is None:
            return
        else:
            try:
                time_in_ms = frame_number * 1000 // self.video_player.video.fps
            except Exception as e:
                logging.warning(f"Error seeking video: {e}")
                time_in_ms = frame_number * 1000 // 25
        self.video_player.set_position(time_in_ms)

    def update_visits(self, visits):
        self.bboxes_updater = BboxesUpdater(visits)
        self.bboxes_updater.bboxes_updated.connect(self.on_bboxes_updated)
        self.bboxes_updater.start()

        self.periods_updater = PeriodsUpdater(visits)
        self.periods_updater.periods_updated.connect(self.on_periods_updated)
        self.periods_updater.start()

    def on_bboxes_updated(self, bboxes):
        self.video_player.set_bounding_boxes(bboxes, persistence=15)

    def on_periods_updated(self, periods):
        self.video_player.set_highlight_periods(periods)

    def update_flowers(self, flowers):
        self.video_player.set_reference_bounding_boxes(flowers, persistence=0)


from detectflow.config import S3_CONFIG
from detectflow.utils.cfg import is_s3_config_valid, resolve_s3_config
from detectflow.manipulators.dataloader import Dataloader
from detectflow.manipulators.input_manipulator import InputManipulator
from detectflow.utils.name import is_valid_video_id, parse_video_id

def gather_video_filepaths(directory, video_extensions=None):
    if video_extensions is None:
        # Default to common video file extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

    video_filepaths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_filepaths.append(os.path.join(root, file))

    return video_filepaths

class VideoOpener(QThread):
    video_found = pyqtSignal(str, int)

    def __init__(self, video_id, video_files, download_videos, app_data_dir, s3_config: dict = None):
        super().__init__()
        self.video_id = video_id
        self.video_files = video_files
        self.download_videos = download_videos
        self.app_data_dir = app_data_dir
        self.s3_config = s3_config

    def run(self):
        # Find the video file path that contains the video_id
        for index, file_path in enumerate(self.video_files):
            if self.video_id in file_path:
                self.video_found.emit(file_path, index)
                return

        for file_path in gather_video_filepaths(self.app_data_dir):
            if self.video_id in file_path:
                self.video_files.append(file_path)
                self.video_found.emit(file_path, len(self.video_files) - 1)
                return

        # If no matching file path is found, check if download is enabled
        if self.download_videos:
            if self.s3_config is None and not is_s3_config_valid(S3_CONFIG):
                logging.warning("S3 configuration is required for downloading videos.")
                return

            # Download video
            try:
                downloaded_file_path = self.download_video(self.video_id)
            except Exception as e:
                logging.error(f"Error downloading video: {e}")
                downloaded_file_path = None

            if downloaded_file_path is not None:
                self.video_files.append(downloaded_file_path)
                self.video_found.emit(downloaded_file_path, len(self.video_files) - 1)

    def download_video(self, video_id):

        if not is_valid_video_id(video_id):
            logging.warning(f"Invalid video ID: {video_id}. Download unavailable.")
            return None

        # Extract video info
        video_info = parse_video_id(video_id)

        # Resolve S3 config
        resolve_s3_config(self.s3_config)

        # Initialize dataloader
        dataloader = Dataloader(S3_CONFIG)

        # Locate the video file on S3
        bucket_name = InputManipulator.get_bucket_name_from_id(video_id)
        directory_name = InputManipulator.zero_pad_id(video_info['recording_id'])
        recording_id = video_info['recording_id']
        date, hour, minutes = video_info['date'], video_info['hour'], video_info['minute']

        online_file = dataloader.locate_file_s3(rf'{directory_name}/({recording_id}|{directory_name})_{"_".join([date, hour, minutes])}',
                                                     bucket_name, 'name')

        if online_file is None:
            logging.warning(f"Video file for ID {video_id} not found.")
            return None

        # Define the local path for the downloaded video
        os.makedirs(os.path.join(self.app_data_dir, directory_name), exist_ok=True)
        local_path = os.path.join(self.app_data_dir, directory_name, os.path.basename(online_file))

        # Download the video file
        download_path = dataloader.download_file_s3(*dataloader.parse_s3_path(online_file), local_path)

        return download_path


class BboxesUpdater(QThread):
    bboxes_updated = pyqtSignal(dict)

    def __init__(self, visits):
        super().__init__()
        self.visits = visits

    def run(self):
        bboxes = {}
        for visitor_id, visit_dict in self.visits.items():
            for frame_number, visit_bboxes in visit_dict['frames'].items():
                if isinstance(visit_bboxes, str):
                    visit_bboxes = ast.literal_eval(visit_bboxes)
                bboxes[frame_number] = bboxes.get(frame_number, []) + [box + [visitor_id] for box in visit_bboxes]

        self.bboxes_updated.emit(bboxes)


class PeriodsUpdater(QThread):
    periods_updated = pyqtSignal(list)

    def __init__(self, visits):
        super().__init__()
        self.visits = visits

    def run(self):
        periods = []
        for visitor_id, visit_dict in self.visits.items():
            try:
                periods.append((visit_dict['period']['start_time_ms'], visit_dict['period']['end_time_ms'], visitor_id))
            except Exception:
                pass

        self.periods_updated.emit(periods)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()