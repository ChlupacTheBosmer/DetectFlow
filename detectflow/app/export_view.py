import sys
from PyQt6.QtWidgets import (QWidget, QToolBar, QPushButton, QTableView,
                             QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QLineEdit, QCheckBox,
                             QFileDialog, QGridLayout, QSpinBox, QDoubleSpinBox, QFrame, QHeaderView)
from PyQt6.QtCore import Qt, QSize, QModelIndex, QAbstractTableModel, QItemSelectionModel
from PyQt6.QtGui import QIcon, QColor, QStandardItemModel, QStandardItem, QBrush, QPixmap
import sys
from PyQt6.QtWidgets import QApplication, QProgressBar, QSizePolicy
import sys
from PyQt6.QtWidgets import (QWidget, QToolBar, QPushButton, QTableView,
                             QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QCheckBox,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, QSize, QModelIndex, QTimer
from PyQt6.QtGui import QIcon, QColor
from detectflow.app.visits_view import create_button
from detectflow.resources import ALT_ICONS
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.models import VISITORS_MODEL
import logging


# Custom Toolbar
class ExportViewToolbar(QToolBar):
    def __init__(self):
        super().__init__("Export Toolbar")
        self.setOrientation(Qt.Orientation.Horizontal)

        self.add_entry_button = create_button(ALT_ICONS['plus-circle'])
        self.addWidget(self.add_entry_button)

        self.remove_entry_button = create_button(ALT_ICONS['minus-circle'])
        self.addWidget(self.remove_entry_button)

        self.toggle_flag_button = create_button(ALT_ICONS['flag'])
        self.addWidget(self.toggle_flag_button)

        self.toggle_selection_mode_icons = [ALT_ICONS['menu'], ALT_ICONS['list']]
        self._toggle_selection_mode_index = 0
        self.toggle_selection_mode_button = create_button(self.toggle_selection_mode_icons[self.toggle_selection_mode_index])
        self.addWidget(self.toggle_selection_mode_button)

        self.addSeparator()

        self.save_to_db_button = create_button(ALT_ICONS['save'])
        self.addWidget(self.save_to_db_button)

        self.load_from_db_button = create_button(ALT_ICONS['database'])
        self.addWidget(self.load_from_db_button)

        self.addSeparator()

        self.export_button = create_button(ALT_ICONS['image'])
        self.addWidget(self.export_button)

    @property
    def toggle_selection_mode_index(self):
        return self._toggle_selection_mode_index

    @toggle_selection_mode_index.setter
    def toggle_selection_mode_index(self, value):
        self._toggle_selection_mode_index = value % len(self.toggle_selection_mode_icons)
        self.toggle_selection_mode_button.setIcon(
            QIcon(self.toggle_selection_mode_icons[self.toggle_selection_mode_index]))


# Custom Table Model
class ExportTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.data_list = []  # Stores the table data as list of dictionaries
        self.current_video_id = None

    def rowCount(self, parent=None):
        return len(self.data_list)

    def columnCount(self, parent=None):
        return 4  # start_frame, end_frame, video_id, flag

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            column_names = ["Start Frame", "End Frame", "Video ID", "Flag"]
            if section < len(column_names):
                return column_names[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            row = index.row()
            column = index.column()
            return self.data_list[row][column]

        if role == Qt.ItemDataRole.BackgroundRole:
            flag = self.data_list[index.row()][3]
            if flag == "TP":
                return QBrush(QColor(Qt.GlobalColor.green))
            elif flag == "FN":
                return QBrush(QColor(Qt.GlobalColor.darkGreen))
            elif flag == "FP":
                return QBrush(QColor(Qt.GlobalColor.red))
            elif flag == "TN":
                return QBrush(QColor(Qt.GlobalColor.darkRed))

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole:
            if value:  # Only set the value if it's not empty
                self.data_list[index.row()][index.column()] = value
            else:  # Retain the existing value if the new value is empty
                existing_value = self.data_list[index.row()][index.column()]
                self.data_list[index.row()][index.column()] = existing_value
            self.dataChanged.emit(index, index, (Qt.ItemDataRole.EditRole,))
            return True
        return False

    def flags(self, index):
        if index.column() == 3:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable

    def add_row(self):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        video_id = self.current_video_id or ""
        self.data_list.append(["", "", video_id, ""])
        self.endInsertRows()

    def remove_row(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        self.data_list.pop(row)
        self.endRemoveRows()

    def toggle_flag(self, selected_rows):
        for row in selected_rows:
            current_flag = self.data_list[row][3]
            new_flag = self.get_next_flag(current_flag)
            self.data_list[row][3] = new_flag
            index = self.index(row, 3)
            print(f"Row {row}: {current_flag} -> {new_flag}")
            self.dataChanged.emit(index, index)

    def get_next_flag(self, current_flag):
        flags = ["", "TP", "FP", "TN", "FN"]
        next_index = (flags.index(current_flag) + 1) % len(flags)
        return flags[next_index]

    def add_row_with_data(self, start_frame, end_frame, video_id):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.data_list.append([str(start_frame), str(end_frame), video_id, ""])  # The last entry for flag (initially empty)
        self.endInsertRows()


# Custom TableView
class ExportTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.model = ExportTableModel()
        self.setModel(self.model)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.ContiguousSelection)

        # Ensure columns are visible and fill the entire area
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def receive_data(self, data):
        print(f"Received data: {data}")
        for row in data:
            start_frame = row.get('start_frame')
            end_frame = row.get('end_frame')
            video_id = row.get('video_id')

            print(f"Adding row: {start_frame}, {end_frame}, {video_id}")

            # Insert the relevant data into the table view's model
            self.model.add_row_with_data(start_frame, end_frame, video_id)


# Custom Control Panel
class ExportControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        input_layout = QHBoxLayout()

        # Frames per Visit Group
        frames_per_visit_layout = QFormLayout()

        self.tp_input = QSpinBox()
        self.tp_input.setMinimum(0)
        self.tp_input.setValue(1)

        self.fp_input = QSpinBox()
        self.fp_input.setMinimum(0)
        self.fp_input.setValue(1)

        self.tn_input = QSpinBox()
        self.tn_input.setMinimum(0)
        self.tn_input.setValue(1)

        self.fn_input = QSpinBox()
        self.fn_input.setMinimum(0)
        self.fn_input.setValue(1)

        frames_per_visit_layout.addRow("TP:", self.tp_input)
        frames_per_visit_layout.addRow("FP:", self.fp_input)
        frames_per_visit_layout.addRow("TN:", self.tn_input)
        frames_per_visit_layout.addRow("FN:", self.fn_input)

        frames_per_visit_group = QGroupBox("Frames per Visit")
        frames_per_visit_group.setLayout(frames_per_visit_layout)

        # Detection Group
        detection_layout = QFormLayout()

        self.auto_annotate_checkbox = QCheckBox()

        self.model_path_selector = QPushButton("Select Path")
        self.model_path_selector.clicked.connect(self.select_model_path)
        self.model_path_selector.setText(VISITORS_MODEL)

        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0, 1.0)
        self.confidence_input.setSingleStep(0.01)
        self.confidence_input.setValue(0.3)

        self.annotation_limit_input = QSpinBox()
        self.annotation_limit_input.setMinimum(-1)
        self.annotation_limit_input.setValue(1)

        detection_layout.addRow("Auto-annotate:", self.auto_annotate_checkbox)
        detection_layout.addRow("Model Path:", self.model_path_selector)
        detection_layout.addRow("Confidence:", self.confidence_input)
        detection_layout.addRow("Annotation Limit:", self.annotation_limit_input)

        detection_group = QGroupBox("Detection")
        detection_group.setLayout(detection_layout)

        # Ensure that both group boxes expand to fill the available space
        frames_per_visit_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        detection_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        input_layout.addWidget(frames_per_visit_group)
        input_layout.addWidget(detection_group)

        layout.addLayout(input_layout)

        # Add Status Bar and Progress Bar
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)

        status_bar_layout = QHBoxLayout()

        # Status Bar with Icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QPixmap(ALT_ICONS['info']).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))
        self.icon_label.setContentsMargins(0, 0, 0, 0)
        status_bar_layout.addWidget(self.icon_label)

        self.status_bar = QLabel("Ready for export")
        status_bar_layout.addWidget(self.status_bar)

        status_bar_layout.addStretch()

        status_layout.addLayout(status_bar_layout)

        # Progress Bar
        self.progress_bar_w = QWidget()
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar_w.setLayout(progress_layout)
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress_bar_w.setLayout(progress_layout)
        progress_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.progress_bar_w)

        #layout.addStretch()
        layout.setStretch(layout.indexOf(input_layout), 4)
        layout.setStretch(layout.indexOf(status_layout), 1)
        layout.addLayout(status_layout)

        self.setLayout(layout)

    def select_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Path", "", "PyTorch Model Files (*.pt)")
        if file_path:
            self.model_path_selector.setText(file_path)

    def set_status_message(self, message, icon=None, timeout=0):
        self.status_bar.setText(message)
        if icon:
            self.icon_label.setPixmap(QPixmap(icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
        if timeout > 0:
            QTimer.singleShot(timeout, self.reset_status_message)

    def reset_status_message(self, message="Ready for export", icon=None):
        self.status_bar.setText(message)
        default_icon = ALT_ICONS['info']
        self.icon_label.setPixmap(
            QPixmap(icon if icon else default_icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))

    def set_progress_max(self, max_value):
        if self.progress_bar.value() < max_value:
            self.progress_bar.reset()
        self.progress_bar.setMaximum(max_value)

    def update_progress(self, value):
        print("Progress:", value)
        self.progress_bar.setValue(max(0, min(value, self.progress_bar.maximum())))

    def reset_progress(self):
        self.progress_bar.reset()
        pass

    def get_input_values(self):
        """
        Retrieve the current values from all input fields in the control panel.

        :return: A dictionary containing the input values.
        """
        return {
            'tp': self.tp_input.value(),
            'fp': self.fp_input.value(),
            'tn': self.tn_input.value(),
            'fn': self.fn_input.value(),
            'auto_annotate': self.auto_annotate_checkbox.isChecked(),
            'model_path': self.model_path_selector.text(),
            'confidence': self.confidence_input.value(),
            'annotation_limit': self.annotation_limit_input.value() if self.annotation_limit_input.value() > 0 else None
        }


# Main Widget
class ExportView(QWidget):
    def __init__(self, parent=None, app_data_dir=None):
        super().__init__()

        self.app_data_dir = app_data_dir
        self.parent = parent
        self.exporter = None

        layout = QVBoxLayout(self)

        self.table_view = ExportTableView(self)
        self.toolbar = ExportViewToolbar()

        layout.addWidget(self.toolbar)

        self.toolbar.add_entry_button.clicked.connect(self.add_entry)
        self.toolbar.remove_entry_button.clicked.connect(self.remove_entry)
        self.toolbar.toggle_flag_button.clicked.connect(self.toggle_flag)
        self.toolbar.toggle_selection_mode_button.clicked.connect(self.toggle_selection_mode)
        self.toolbar.save_to_db_button.clicked.connect(self.save_to_database)
        self.toolbar.load_from_db_button.clicked.connect(self.load_from_database)
        self.toolbar.export_button.clicked.connect(self.export_training_data)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self.table_view)
        self.control_panel = ExportControlPanel()
        splitter.addWidget(self.control_panel)

        # Adjusting sizes: table view should occupy half of the widget
        splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.5)])

        layout.addWidget(splitter)

        self.setLayout(layout)

        # Set minimum height based on contents
        self.adjustSize()

    def adjustSize(self):
        # Ensure the widget height is minimized to fit the contents
        self.control_panel.adjustSize()
        self.table_view.setMinimumHeight(self.control_panel.sizeHint().height())
        super().adjustSize()

    def add_entry(self):
        self.table_view.model.add_row()

    def remove_entry(self):
        selected_rows = self.table_view.selectionModel().selectedRows()
        if selected_rows:
            for index in sorted(selected_rows, key=lambda x: x.row(), reverse=True):
                self.table_view.model.remove_row(index.row())

    def toggle_flag(self):
        selected_rows = [index.row() for index in self.table_view.selectionModel().selectedRows()]
        if selected_rows:
            self.table_view.model.toggle_flag(selected_rows)

    def toggle_selection_mode(self):
        current_mode = self.table_view.selectionMode()
        if current_mode == QTableView.SelectionMode.ContiguousSelection:
            self.table_view.setSelectionMode(QTableView.SelectionMode.MultiSelection)
        else:
            self.table_view.setSelectionMode(QTableView.SelectionMode.ContiguousSelection)

        self.toolbar.toggle_selection_mode_index += 1

    def save_to_database(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(self.table_view, "Save Database", "", "SQLite Database Files (*.db)")

        if file_path:
            db_manipulator = DatabaseManipulator(file_path)

            # Prepare the data from the table view
            data_to_save = []
            for row in self.table_view.model.data_list:
                row_data = {
                    'start_frame': row[0],
                    'end_frame': row[1],
                    'video_id': row[2],
                    'flag': row[3]
                }
                data_to_save.append(row_data)

            # Save the data into the SQLite database using DatabaseManipulator
            db_manipulator.create_table('table_data', [
                ('start_frame', 'TEXT', ''),
                ('end_frame', 'TEXT', ''),
                ('video_id', 'TEXT', ''),
                ('flag', 'TEXT', '')
            ])

            for row_data in data_to_save:
                db_manipulator.insert('table_data', row_data)

            db_manipulator.close_connection()
            print('Data saved to SQLite database:', file_path)

    def load_from_database(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self.table_view, "Load Database", "", "SQLite Database Files (*.db)")

        if file_path:
            db_manipulator = DatabaseManipulator(file_path)

            # Load the data from the SQLite database into a DataFrame
            df = db_manipulator.load_dataframe('table_data')

            # Clear existing data in the model
            self.table_view.model.beginResetModel()
            self.table_view.model.data_list = df.values.tolist()  # Convert DataFrame to a list of lists
            self.table_view.model.endResetModel()

            db_manipulator.close_connection()
            print('Data loaded from SQLite database:', file_path)

    def export_training_data(self):
        # Gather data from the table view
        table_data = []
        for row in self.table_view.model.data_list:
            row_data = {
                'start_frame': row[0],
                'end_frame': row[1],
                'video_id': row[2],
                'flag': row[3]
            }
            if row_data['flag']:  # Ignore rows with an empty flag
                table_data.append(row_data)

        # Gather settings from the control panel
        export_settings = self.control_panel.get_input_values()

        # Start the TrainingDataExporter thread
        if self.parent and hasattr(self.parent, 'app_data_dir'):
            app_data_dir = self.parent.app_data_dir
        else:
            app_data_dir = self.app_data_dir
        if not app_data_dir:
            print("App data directory not found.")
            return

        print(f"Exporting data with settings: {export_settings}, {app_data_dir}, {table_data}")

        self.exporter = TrainingDataExporter(table_data, export_settings, self.app_data_dir, self.control_panel)
        self.exporter.export_finished.connect(self.on_export_finished)
        self.exporter.update_progress_signal.connect(self.control_panel.update_progress)
        self.exporter.reset_progress_signal.connect(self.control_panel.reset_progress)
        self.exporter.set_progress_max_signal.connect(self.control_panel.set_progress_max)
        self.exporter.start()

    def on_export_finished(self):
        print("Export completed.")


import os
import cv2  # Assuming OpenCV is used for frame extraction
import random
from PyQt6.QtCore import QThread, pyqtSignal
from detectflow.video.video_data import Video
from detectflow.predict.predictor import Predictor
from detectflow.manipulators.frame_manipulator import FrameManipulator
from detectflow.app.id import VideoOpener, gather_video_filepaths
from detectflow.manipulators.manipulator import Manipulator
from detectflow.utils.inspector import Inspector
from PyQt6.QtCore import QEventLoop


class TrainingDataExporter(QThread):
    export_finished = pyqtSignal()  # Signal emitted when export is finished
    update_progress_signal = pyqtSignal(int)
    reset_progress_signal = pyqtSignal()
    set_progress_max_signal = pyqtSignal(int)

    def __init__(self, table_data, export_settings, app_data_dir, control_panel=None, video_opener=None):
        super().__init__()
        self.frame_count = 0
        self.table_data = table_data
        self.export_settings = export_settings
        self.app_data_dir = app_data_dir
        self.train_data_dir = os.path.join(self.app_data_dir, "train_data")
        self.control_panel = control_panel
        self.video_opener = video_opener

    def run(self):
        print("Packing export data...")
        export_data = {}
        start_frame = 0
        end_frame = 0
        video_id = None
        try:
            for row in self.table_data:
                flag = row['flag']
                if not flag:  # Ignore rows with an empty flag
                    continue

                start_frame = int(row['start_frame'])
                end_frame = int(row['end_frame'])
                video_id = row['video_id']
                frames_per_visit = self.export_settings.get(flag.lower(), 0)

                # Check if we need to export frames for this flag
                indices = []
                if frames_per_visit > 0:
                    # Get frame indices to extract
                    total_frames = max(0, end_frame - start_frame)
                    if total_frames < frames_per_visit:
                        indices = list(range(start_frame, end_frame + 1))
                    else:
                        indices = random.sample(range(start_frame, end_frame + 1),
                                                  min(frames_per_visit, total_frames))

                    if flag not in export_data:
                        export_data[flag] = {}
                    if video_id not in export_data[flag]:
                        export_data[flag][video_id] = []
                    if indices:
                        export_data[flag][video_id].extend(indices)
        except Exception as e:

            self.control_panel.set_status_message(str(e), icon=ALT_ICONS['alert-circle'],
                                                  timeout=5000)

            print(f"Error during export data packing: {e}")

        # Export frames for each flag
        for flag, video_data in export_data.items():

            total_frames_to_export = sum([len(frame_indices) for frame_indices in video_data.values()])
            print(f"Total frames to export for {flag}: {total_frames_to_export}")
            self.set_progress_max_signal.emit(total_frames_to_export)
            self.frame_count = 0

            # Update status
            self.control_panel.set_status_message(f"Exporting {flag} frames...", icon=ALT_ICONS['image'])

            print(f"Exporting {flag} frames...")
            try:
                # Create flag directory if it doesn't exist
                os.makedirs(os.path.join(self.train_data_dir, flag), exist_ok=True)

                for video_id, frame_indices in video_data.items():
                    self.export_frames(video_id, flag, frame_indices)

            except Exception as e:

                self.control_panel.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)

                print(f"Error during {flag} frame export: {e}")

        # Move object frames from 'object' directories to the main folder
        self.move_object_frames()

        self.control_panel.set_status_message("Export complete!", icon=ALT_ICONS['thumbs-up'], timeout=5000)
        self.reset_progress_signal.emit()

        # Emit signal when done
        self.export_finished.emit()

    def export_frames(self, video_id, flag, frame_indices):

        # Get video path if it is locally available
        video_path = self.get_video_path(video_id)

        if not video_path or not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return

        print(f"Exporting {flag} frames for video:", video_id)

        # Open the video file
        video = Video(video_path)

        # Extract frames using the Video class
        for extraction in video.read_video_frame(frame_indices, True):

            if flag in ['TP', 'FN'] and self.export_settings['auto_annotate']:
                # print(f"Auto-annotating {flag} frames...")

                # Update status
                self.control_panel.set_status_message(f"Exporting {flag} frames for {video_id}...", icon=ALT_ICONS['box'])

                try:
                    frames = [extraction['frame']]
                    metadata = {
                        'frame_number': [extraction['frame_number']],
                        'source_path': video_path,
                        'save_dir': os.path.join(self.train_data_dir, flag)
                    }

                    self.export_annotated_frames(frames, metadata)
                except Exception as e:
                    print(f"Error auto-annotating frames: {e}")

                    self.control_panel.set_status_message(str(e),
                                                          icon=ALT_ICONS['alert-circle'], timeout=2000)

            else:
                # print(f"Saving {flag} frames...")

                # Update status
                self.control_panel.set_status_message(f"Exporting {flag} frames for {video_id}...", icon=ALT_ICONS['copy'])

                try:
                    frames = [extraction['frame']]
                    metadata = {
                        'frame_number': [extraction['frame_number']],
                        'source_path': video_path,
                        'save_dir': os.path.join(self.train_data_dir, flag)
                    }

                    self.export_empty_frames(frames, metadata)
                except Exception as e:
                    print(f"Error saving frames: {e}")

                    self.control_panel.set_status_message(str(e),
                                                          icon=ALT_ICONS['alert-circle'], timeout=5000)

    def export_annotated_frames(self, frames, metadata):

        # Initialize the predictor
        predictor = Predictor()

        # Determine maximum number of detections to make
        max_det = self.export_settings['annotation_limit'] if self.export_settings['annotation_limit'] > 0 else None

        # Perform detection on the frames
        for result in predictor.detect(frame_numpy_array=frames,
                                       metadata=metadata,
                                       model_path=self.export_settings['model_path'],
                                       detection_conf_threshold=self.export_settings['confidence'],
                                       sliced=False,
                                       device='cpu',
                                       max_det=max_det):
            try:
                result.save(save_txt=True, extension='.jpg')

            except Exception as e:
                print(f"Error saving detection result: {e}")

            print(f"Frame {result.frame_number} saved.")
            self.frame_count += 1
            self.update_progress_signal.emit(self.frame_count)

    def export_empty_frames(self, frames, metadata):

        from detectflow.utils.name import parse_video_name

        # Get video ID from the metadata
        video_id = parse_video_name(metadata.get('source_path', '')).get('video_id', None)

        for frame, frame_number in zip(frames, metadata.get('frame_number', [])):
            frame_filename = f"{video_id}_{frame_number}"
            try:
                FrameManipulator.save_frame(frame, frame_filename, metadata.get('save_dir', ''), 'jpg')
            except Exception as e:
                print(f"Error saving frame: {e}")

            self.frame_count += 1
            self.update_progress_signal.emit(self.frame_count)

    def get_video_path(self, video_id):

        for file_path in gather_video_filepaths(self.app_data_dir):
            if video_id in file_path:
                return file_path
        return None

    def move_object_frames(self):
        """
        Simple helper function to move frames from a nested 'object' directories in auto-annotated folders directly to
        the folder for frames with a specific flag.
        """

        for flag in ["TP", "FP", "TN", "FN"]:

            for subfolder in ['empty', 'object']:

                folder_path = os.path.join(self.train_data_dir, flag, subfolder)

                if not os.path.exists(folder_path):
                    continue

                # Get the parent directory of the folder
                parent_dir = os.path.dirname(folder_path)

                # Iterate through all files in the folder
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)

                    if os.path.isfile(file_path):
                        # Use the move_file method to move each file to the parent directory
                        Manipulator.move_file(file_path, parent_dir, overwrite=True)

                # After all files are moved, remove the empty folder
                try:
                    os.rmdir(folder_path)
                    logging.info(f"Removed empty folder: {folder_path}")
                except OSError as e:
                    logging.error(f"Failed to remove folder {folder_path}: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create an instance of the MainWidget
    main_widget = ExportView()

    # Set window title and initial size
    main_widget.setWindowTitle("Custom Widget Test")

    # Show the widget as a standalone window
    main_widget.show()

    # Run the application event loop
    sys.exit(app.exec())