import PyQt6
import os

try:
    from ctypes import windll  # Only exists on Windows.
    myappid = 'insect-communities.detectflow.id'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

pyqt = os.path.dirname(PyQt6.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt6/plugins")
from PyQt6.QtWidgets import QSplitter
from detectflow.app.video_player import InteractiveVideoPlayer
from detectflow.app.visits_view import CustomWidget
from detectflow.app.species_view import EditableImageView
import logging
import sys
import pickle
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication
)
from PyQt6.QtCore import Qt
from detectflow.resources import ALT_ICONS, IMGS
import os
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QMessageBox, QMainWindow
import ast
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, QThread, QTimer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_opener = None
        self.current_video_id = None
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(1 * 60 * 1000)  # 5 minutes

        self._app_dir = None
        self._autosave_folder = 'autosave'

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

    @property
    def app_dir(self):
        if self._app_dir is None:
            if self.video_player is not None:
                self._app_dir = self.video_player.app_data_dir
            else:
                self._app_dir = os.getcwd()
        return self._app_dir

    @app_dir.setter
    def app_dir(self, value):
        self._app_dir = value
        if self.video_player is not None:
            self.video_player.app_data_dir = value

    def connect_signals(self):
        self.custom_widget.update_video_id_signal.connect(self.handle_video_id_update)
        self.custom_widget.seek_video_signal.connect(self.handle_seek_video)
        self.custom_widget.update_visits_signal.connect(self.update_visits)
        self.custom_widget.update_flowers_signal.connect(self.update_flowers)
        self.video_player.current_visits_changed.connect(self.custom_widget.update_current_visits)

    def showEvent(self, event):
        super().showEvent(event)

        # Check for autosave files
        QTimer.singleShot(1000, self.check_for_autosave)

    def check_for_autosave(self):
        try:
            autosave_path = os.path.join(self.app_dir, self._autosave_folder, 'autosave.pkl')
            if os.path.exists(autosave_path):
                reply = QMessageBox.question(self, 'Load Autosave', "An autosave was detected. Do you want to load it?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.Yes)
                if reply == QMessageBox.StandardButton.Yes:
                    self.load_autosave(autosave_path)
        except Exception as e:
            logging.warning(f"Error checking for autosave: {e}")

    def autosave(self):
        autosave_path = os.path.join(self.app_dir, self._autosave_folder, 'autosave.pkl')
        self.save_state(autosave_path)

    def save_state(self, path):

        db_path = None
        model_data = None
        if isinstance(self.custom_widget, CustomWidget):
            try:
                db_path = self.custom_widget.db_path
                model_data = self.custom_widget.model.getRefinedDataFrameCopy() if self.custom_widget.model else None
            except Exception as e:
                logging.warning(f"Error saving DB view state: {e}")

        video_files = None
        current_video_index = None
        scene_save = None
        settings = None
        if isinstance(self.video_player, InteractiveVideoPlayer):
            try:
                video_files = self.video_player.config['video_files']
                current_video_index = self.video_player.config['current_video_index']
                scene_save = os.path.join(self.app_dir, self._autosave_folder, 'scene_autosave.bin')
                settings = self.video_player.config
            except Exception as e:
                logging.warning(f"Error saving video player state: {e}")

        data = {
            'db_path': db_path,
            'model_data': model_data,
            'video_files': video_files,
            'current_video_index': current_video_index,
            'video_scene': scene_save,
            'settings': settings
        }

        # Save the video scene separately
        try:
            self.video_player.save_scene(data['video_scene'])
        except Exception as e:
            logging.warning(f"Error saving video scene: {e}")
            data['video_scene'] = None

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.video_player.set_status_message(f"Autosave created successfully.", icon=ALT_ICONS['save'],
                                                 timeout=3000)
        except Exception as e:
            self.video_player.set_status_message(f"Error saving autosave: {e}", icon=ALT_ICONS['x-circle'], timeout=3000)

    def load_autosave(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                raise ValueError("Invalid autosave data.")

            autosave_keys = ['db_path', 'model_data', 'video_files', 'current_video_index', 'video_scene', 'settings']

            if not all([key in data for key in autosave_keys]):
                raise ValueError("Invalid autosave data.")

            self.video_player.set_status_message(f"Autosave loaded successfully.", icon=ALT_ICONS['upload'],
                                                 timeout=3000)
        except Exception as e:
            self.video_player.set_status_message(f"Error loading autosave: {e}", icon=ALT_ICONS['x-circle'],
                                                 timeout=3000)

        try:
            if isinstance(self.custom_widget, CustomWidget):
                if data['db_path'] is None or data['model_data'] is None:
                    raise ValueError("Invalid DB view state.")
                elif not os.path.exists(data['db_path']):
                    raise FileNotFoundError("DB file not found.")

                self.custom_widget.reload_state(data['db_path'], data['model_data'])
        except Exception as e:
            self.video_player.set_status_message(f"Error: {e}", icon=ALT_ICONS['x-circle'],
                                                 timeout=3000)

        try:
            print("Reloading video player...")
            if isinstance(self.video_player, InteractiveVideoPlayer):
                if data['video_files'] is None:
                    data['settings']['video_files'] = []
                    data['video_files'] = []
                    raise ValueError("No video files to reload.")

                if not all([os.path.exists(file) for file in data['video_files']]) and not data.get('settings').get('download_videos', False):
                    data['settings']['video_files'] = []
                    data['video_files'] = []
                    raise FileNotFoundError("Video file not found.")

                self.video_player.video_files = data['video_files']

                # if data['current_video_index'] is None:
                #     data['current_video_index'] = 0
                #     data['settings']['current_video_index'] = 0

                # video_file = data['video_files'][data['current_video_index']]
                # if is_valid_video_id(os.path.basename(video_file).split('.')[0]):
                #     self.handle_video_id_update(os.path.basename(video_file).split('.')[0])
                # else:
                #     self.open_video(video_file, data['current_video_index'])
        except Exception as e:
            self.video_player.set_status_message(f"Error: {e}", icon=ALT_ICONS['x-circle'], timeout=3000)

        try:
            print("Loading video scene...")
            if data['video_scene'] is not None and os.path.exists(data['video_scene']) and data['video_scene'].endswith('.bin'):
                self.video_player.load_scene(data['video_scene'])
        except Exception as e:
            self.video_player.set_status_message(f"Error loading video scene: {e}", icon=ALT_ICONS['x-circle'], timeout=3000)

        self.video_player.config = data['settings']

    def handle_video_id_update(self, video_id):
        print(f"Received video ID: {video_id}")

        media_status = self.video_player.media_player.mediaStatus()
        print(f"Current video ID: {self.current_video_id}, Media status: {media_status}")
        print(media_status == QMediaPlayer.MediaStatus.NoMedia)

        if self.current_video_id == video_id and media_status != QMediaPlayer.MediaStatus.NoMedia:
            return

        print(f"Opening video: {video_id}")

        self.disable_navigation()

        if media_status != QMediaPlayer.MediaStatus.NoMedia:
            self.video_player.media_player.pause()

        self.video_player._progress_lock = True
        self.video_player.show_progress_bar()
        self.show_message("Downloading video...", icon=ALT_ICONS['search'])

        self.current_video_id = video_id

        s3_config = {
            'host_base': self.video_player.host_base,
            'use_https': self.video_player.use_https,
            'access_key': self.video_player.access_key,
            'secret_key': self.video_player.secret_key,
            'host_bucket': self.video_player.host_bucket
        }

        if hasattr(self, 'video_opener') and self.video_opener is not None and self.video_opener.isRunning():
            self.video_opener.wait()

        print(type(self.video_player.video_files))
        print(type(self.video_player.download_videos))
        print(type(self.video_player.app_data_dir))
        print(type(s3_config))


        self.video_opener = VideoOpener(video_id,
                                        self.video_player.video_files,
                                        self.video_player.download_videos,
                                        self.video_player.app_data_dir,
                                        s3_config,
                                        progress_callback=self.show_message)
        self.video_opener.video_found.connect(self.open_video)
        self.video_opener.start()

    def open_video(self, file_path, index):
        self.enable_navigation()
        self.video_player._progress_lock = False
        if file_path is None:
            self.show_message("Video not found.", icon=ALT_ICONS['alert-circle'])
            return

        self.video_player.open_file(file_path)
        self.video_player.current_video_index = index

    def handle_seek_video(self, video_id, frame_number):
        if self.video_player.video is None:
            self.handle_video_id_update(video_id)
        else:
            self.seek_video(frame_number)

    def seek_video(self, frame_number):
        if self.video_player.video is None:
            return
        else:
            try:
                print(self.video_player.video.fps)
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

    def show_message(self, message, icon=ALT_ICONS['info']):
        self.video_player._default_message = message
        self.video_player._default_icon = icon
        self.video_player.reset_status_message()

    def disable_navigation(self):
        self.custom_widget.disable_navigation()
        self.video_player.disable_navigation()

    def enable_navigation(self):
        self.custom_widget.enable_navigation()
        self.video_player.enable_navigation()

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

    def __init__(self, video_id, video_files, download_videos, app_data_dir, s3_config: dict = None, progress_callback=None):
        print(f"VideoOpener starting...")
        super().__init__()
        self.video_id = video_id
        self.video_files = video_files
        self.download_videos = download_videos
        self.app_data_dir = app_data_dir
        self.s3_config = s3_config
        self.progress_callback = progress_callback
        print(f"VideoOpener: {video_id}, {video_files}, {download_videos}, {app_data_dir}, {s3_config}, {progress_callback}")

    def run(self):
        print(f"VideoOpener running...")
        # Find the video file path that contains the video_id
        for index, file_path in enumerate(self.video_files):
            if self.video_id in file_path:
                self.video_found.emit(file_path, index)
                print("Video not found in existing files.")
                return

        for file_path in gather_video_filepaths(self.app_data_dir):
            if self.video_id in file_path:
                self.video_files.append(file_path)
                self.video_found.emit(file_path, len(self.video_files) - 1)
                print("Video not found in app data directory.")
                return

        # If no matching file path is found, check if download is enabled
        if not self.download_videos:
            self.video_found.emit(None, 0)
            print("Download is disabled.")
            return

        if self.s3_config is None and not is_s3_config_valid(S3_CONFIG):
            logging.warning("S3 configuration is required for downloading videos.")
            self.video_found.emit(None, 0)
            return

        self.progress_callback("Downloading video...", icon=ALT_ICONS['download-cloud'])

        # Download video
        try:
            print("Downloading video...")
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
    app.setWindowIcon(QIcon(IMGS['icon']))
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()