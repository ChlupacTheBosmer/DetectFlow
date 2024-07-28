import traceback

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
from PyQt6.QtGui import QIcon, QColor, QBrush
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog, QDialog, QVBoxLayout, QCheckBox, QPushButton

from detectflow.app import VISIT_VIEW_HELP
from PyQt6.QtWidgets import QStackedWidget, QStyledItemDelegate
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton
from PyQt6.QtWidgets import QStatusBar
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject, QThread
import ast


class CustomItemDelegate(QStyledItemDelegate):
    def __init__(self, model, colored_visitor_ids):
        super().__init__()
        self.model = model
        self.colored_visitor_ids = colored_visitor_ids

    def paint(self, painter, option, index):
        visitor_id = self.model._filtered_df.loc[self.model._filtered_df.index[index.row()], 'visitor_id']
        if visitor_id in self.colored_visitor_ids:
            option.backgroundBrush = QBrush(self.get_background_color(visitor_id))
        super().paint(painter, option, index)

    def get_background_color(self, visitor_id):
        color_map = {
            1: QColor(Qt.GlobalColor.yellow),
            2: QColor(Qt.GlobalColor.green),
            3: QColor(Qt.GlobalColor.red),
            # Add more mappings as needed
        }
        return color_map.get(visitor_id, QColor(Qt.GlobalColor.white))


class CustomTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            index = self.indexAt(event.pos())
            if index.isValid():
                row = index.row()
                original_index = self.model()._filtered_df.index[row]  # Get the original index from the filtered DataFrame
                start_frame = self.model()._df.loc[original_index, 'start_frame']  # Use the original index to get the value from the original DataFrame
                if not isinstance(self.parent(), CustomWidget):
                    parent = self.parent().parent()
                else:
                    parent = self.parent()

                parent.seek_video_signal.emit(start_frame)
        super().mousePressEvent(event)


class PlotWindow(QWidget):
    def __init__(self, visits_df, video_start, video_end):
        super().__init__()
        self.setWindowTitle("Visits Plot")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Set the background color
        self.plot_widget.setBackground('white')

        self.plot_visits(visits_df, video_start, video_end)

    def plot_visits(self, visits_df, video_start, video_end):
        video_start = pd.to_datetime(video_start)
        video_end = pd.to_datetime(video_end)
        video_duration = (video_end - video_start).total_seconds()

        # Convert start_time and end_time to seconds from the video start
        visits_df['start_time'] = (pd.to_timedelta(visits_df['start_time']) - pd.to_timedelta(0, unit='s')).dt.total_seconds()
        visits_df['end_time'] = (pd.to_timedelta(visits_df['end_time']) - pd.to_timedelta(0, unit='s')).dt.total_seconds()

        # Plot the video timeline
        self.plot_widget.addItem(pg.BarGraphItem(x=[video_duration/2], height=0.25, width=video_duration, brush='grey', pen='black'))

        # Prepare intervals for plotting
        def get_intervals(df):
            intervals = []
            current_y = level_height * 1.5
            max_y = current_y

            for idx, row in df.iterrows():
                start_num = row['start_time']
                duration = row['end_time'] - row['start_time']
                overlap = False

                # Check for overlap
                for interval in intervals:
                    if not (start_num + duration < interval[0] or start_num > interval[0] + interval[1]):
                        overlap = True
                        current_y += level_height * 1
                        break

                if not overlap:
                    current_y = level_height * 1.5  # reset to first level if no overlap

                intervals.append((start_num, duration, current_y))
                max_y = max(max_y, current_y)

            return intervals, max_y

        # Plot the visits
        level_height = 0.25
        visit_intervals, max_y = get_intervals(visits_df)
        for idx, (start, duration, y) in enumerate(visit_intervals):
            self.plot_widget.addItem(pg.BarGraphItem(x=[start + duration / 2], height=0.25, width=duration, y=y, brush='green', pen='black'))

        # Set the color of the ticks and labels
        axis_color = 'black'  # Replace with your desired color

        # Bottom axis (X-axis)
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=axis_color))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=axis_color))

        # Left axis (Y-axis)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=axis_color))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=axis_color))

        # Adjust the plot
        self.plot_widget.setYRange(0, max_y + level_height)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Visits')
        self.plot_widget.getAxis('bottom').setTickSpacing(60, 10)  # Set tick spacing to 60 seconds


class ColumnSettingsDialog(QDialog):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("Column Settings")
        self.model = model

        layout = QVBoxLayout()

        self.checkboxes = {}
        cycling_columns = [col for group in model._cycle_columns for col in group]

        for column in model._df.columns:
            if column not in cycling_columns:
                checkbox = QCheckBox(column)
                checkbox.setChecked(column in model._visible_columns)
                checkbox.stateChanged.connect(self.update_column_visibility)
                self.checkboxes[column] = checkbox
                layout.addWidget(checkbox)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_and_close)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def update_column_visibility(self):
        for column, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                if column not in self.model._visible_columns:
                    self.model._hidden_columns = [col for col in self.model._hidden_columns if col != column]
            else:
                if column in self.model._visible_columns:
                    self.model._hidden_columns.append(column)
        self.model.update_visible_columns()

    def save_and_close(self):
        self.accept()


class HelpWindow(QMainWindow):
    def __init__(self, help_file):
        super().__init__()
        self.setWindowTitle("Help")
        self.setGeometry(300, 300, 600, 400)

        self.help_file = help_file

        # Main layout
        layout = QVBoxLayout()

        # Table of contents
        self.contents = QListWidget()
        self.contents.itemClicked.connect(self.display_help)
        layout.addWidget(self.contents)

        # Help text display
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        layout.addWidget(self.help_text)

        # Load help content
        self.load_help_content()

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_help_content(self):
        # Load help content from file
        if os.path.exists(self.help_file):
            with open(self.help_file, 'r') as file:
                help_data = file.read().split('## ')
                for entry in help_data:
                    if entry.strip():
                        title, content = entry.split('\n', 1)
                        item = QListWidgetItem(title.strip())
                        item.setData(Qt.ItemDataRole.UserRole, content.strip())
                        self.contents.addItem(item)

    def display_help(self, item):
        self.help_text.setHtml(item.data(Qt.ItemDataRole.UserRole))


def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes}:{seconds:02}"


class PeriodDictionaryWorker(QThread):
    result = pyqtSignal()

    def __init__(self, filtered_df, shared_data):
        super().__init__()
        self.filtered_df = filtered_df
        self.shared_data = shared_data

    def run(self):
        self.shared_data.visitor_data = {}
        for _, visit_row in self.filtered_df.iterrows():
            if visit_row['visitor_id'] not in self.shared_data.visitor_data:
                self.shared_data.visitor_data[visit_row['visitor_id']] = process_row(visit_row)
                self.shared_data.flower_data[visit_row['start_frame']] = visit_row['flower_bboxes']

        self.result.emit()


def process_row(visit_row):

    # Initialize the inner dictionary for the current visitor_id
    result = {
        'period': {
            'start_frame': visit_row['start_frame'],
            'end_frame': visit_row['end_frame'],
            'start_time': visit_row['start_time'],
            'end_time': visit_row['end_time'],
            'start_time_ms': int(visit_row['start_time'].total_seconds() * 1000),
            'end_time_ms': int(visit_row['end_time'].total_seconds() * 1000) if visit_row['end_time'] > visit_row['start_time'] else int(visit_row['end_time'].total_seconds() * 1000 + 2000),
            },
        'frames': {}
    }

    # Extract the frame numbers and visitor bboxes
    if isinstance(visit_row['visitor_bboxes'], str):
        visit_row['visitor_bboxes'] = ast.literal_eval(visit_row['visitor_bboxes'])
    if isinstance(visit_row['frame_numbers'], str):
        visit_row['frame_numbers'] = ast.literal_eval(visit_row['frame_numbers'])
    for frame_number, bbox in zip(visit_row['frame_numbers'], list(visit_row['visitor_bboxes'])):
        result['frames'][frame_number] = bbox
    return result


class PeriodDictionary:
    def __init__(self):
        self.visitor_data = {}
        self.flower_data = {}

shared_data = PeriodDictionary()


class PandasTableModel(QAbstractTableModel):
    def __init__(self, periods_df=pd.DataFrame(), videos_df=pd.DataFrame(), visits_df=pd.DataFrame(),
                 update_visits_signal=None, update_video_id_signal=None, update_flowers_signal=None):
        super().__init__()
        self.visits_dict = None
        self._df = periods_df
        self._filtered_df = periods_df.copy()
        self._videos_df = videos_df  # Add reference to video_df
        self._visits_df = visits_df

        self.colored_visitor_ids = []
        self._current_video_id = None
        self._visible_columns = self._df.columns.tolist()
        self._hidden_columns = ["video_id", "flower_bboxes", "visitor_bboxes", "visit_ids", "flags", "frame_numbers"]
        self._cycle_columns = [
            ("start_frame", "end_frame"),
            ("start_time", "end_time"),
            ("start_real_life_time", "end_real_life_time")
        ]
        self._current_cycle_index = 0
        self.update_visible_columns()

        #shared_data
        self.shared_data = shared_data

        # signals
        self.update_visits_signal = update_visits_signal
        self.update_video_id_signal = update_video_id_signal
        self.update_flowers_signal = update_flowers_signal

    def update_visible_columns(self):
        # Start by hiding the hidden columns
        visible_cols = [col for col in self._df.columns if col not in self._hidden_columns]

        # Show only the current cycle columns
        for cycle_group in self._cycle_columns:
            if cycle_group != self._cycle_columns[self._current_cycle_index]:
                visible_cols = [col for col in visible_cols if col not in cycle_group]

        self._visible_columns = visible_cols
        self.layoutChanged.emit()

    def set_colored_visitor_ids(self, packed_ids):
        self.colored_visitor_ids = [id_tuple[0] for id_tuple in packed_ids]
        self.color_map = {id_tuple[0]: id_tuple[1] for id_tuple in packed_ids}
        self.layoutChanged.emit()

    def setVideoIDFilter(self, video_id):
        self._current_video_id = video_id
        if self._current_video_id:
            self._filtered_df = self._df[self._df['video_id'] == self._current_video_id]
        else:
            self._filtered_df = self._df
        self.layoutChanged.emit()

        # Emit the video_id updated signal
        if self.update_video_id_signal:
            self.update_video_id_signal.emit(self._current_video_id)

        self.generate_initial_visits_dict()

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._visible_columns[section]
            if orientation == Qt.Orientation.Vertical:
                return str(self._filtered_df.index[section])
        return QVariant()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):

        if not index.isValid():
            return QVariant()

        col_name = self._visible_columns[index.column()]
        value = self._filtered_df[col_name].iloc[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:

            if col_name in ['start_time', 'end_time']:
                value = pd.to_timedelta(value)
                return format_timedelta(value)

            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            col_name = self._visible_columns[index.column()]
            return str(self._filtered_df[col_name].iloc[index.row()])

        if role == Qt.ItemDataRole.BackgroundRole:

            visitor_id = self._filtered_df.loc[self._filtered_df.index[index.row()], 'visitor_id']
            if visitor_id in self.colored_visitor_ids:
                return QBrush(self.color_map.get(visitor_id, QColor(Qt.GlobalColor.red)))

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole:
            col_name = self._visible_columns[index.column()]  # Get the actual column name
            print(index.column(), col_name, value)
            dtype = self._df[col_name].dtype

            # Cast value to the appropriate type
            if dtype == 'int64':
                value = int(value)
            elif dtype == 'float64':
                value = float(value)
            elif dtype == 'bool':
                value = bool(value)
            elif dtype == 'timedelta64[ns]':
                value = pd.to_timedelta(value)

            self._filtered_df.loc[self._filtered_df.index[index.row()], col_name] = value
            self._df.loc[self._filtered_df.index[index.row()], col_name] = value

            # Recalculate linked columns if necessary
            if col_name in ['start_frame', 'end_frame', 'start_time', 'end_time', 'start_real_life_time',
                            'end_real_life_time']:
                self.recalculate_linked_columns(index.row(), col_name)

            # Update the visits dictionary
            if col_name in ['start_frame', 'end_frame', 'visitor_bboxes', 'frame_numbers']:
                self.update_visits_entry(index.row())

            self.dataChanged.emit(index, index, (Qt.ItemDataRole.EditRole,))
            return True
        return False

    def generate_initial_visits_dict(self):
        print("Generating new dictionary")
        self.worker = PeriodDictionaryWorker(self._filtered_df, self.shared_data)
        self.worker.result.connect(self.handle_finished)
        self.worker.start()

    def handle_finished(self):
        self.update_visits_signal.emit(self.shared_data.visitor_data)
        self.update_flowers_signal.emit(self.shared_data.flower_data)
        self.worker = None

    def update_visits_entry(self, row):
        print("Updating visits entry")
        visit_row = self._filtered_df.iloc[row]
        self.shared_data.visitor_data[visit_row['visitor_id']] = process_row(visit_row)
        self.shared_data.flower_data[visit_row['start_frame']] = visit_row['flower_bboxes']
        self.update_visits_signal.emit(self.shared_data.visitor_data)
        self.update_flowers_signal.emit(self.shared_data.flower_data)

    def recalculate_linked_columns(self, row, changed_col):
        video_id = self._df.loc[self._filtered_df.index[row], 'video_id']
        video_info = self._videos_df[self._videos_df['video_id'] == video_id].iloc[0]
        fps = video_info['fps']
        video_start_time = pd.to_datetime(video_info['start_time'])

        if changed_col in ['start_frame', 'end_frame']:
            start_frame = self._df.loc[self._filtered_df.index[row], 'start_frame']
            end_frame = self._df.loc[self._filtered_df.index[row], 'end_frame']
            real_start_time = video_start_time + pd.to_timedelta(start_frame / fps, unit='s')
            real_end_time = video_start_time + pd.to_timedelta(end_frame / fps, unit='s')
            start_time = pd.to_timedelta(start_frame / fps, unit='s')
            end_time = pd.to_timedelta(end_frame / fps, unit='s')
            self._df.loc[self._filtered_df.index[row], 'start_time'] = start_time
            self._df.loc[self._filtered_df.index[row], 'end_time'] = end_time
            self._df.loc[self._filtered_df.index[row], 'start_real_life_time'] = real_start_time
            self._df.loc[self._filtered_df.index[row], 'end_real_life_time'] = real_end_time
            self._df.loc[self._filtered_df.index[row], 'visit_duration'] = max(0.08, (end_frame - start_frame) / fps)

        elif changed_col in ['start_time', 'end_time']:
            self._df.loc[self._filtered_df.index[row], 'start_real_life_time'] = video_start_time + pd.to_timedelta(
                self._df.loc[self._filtered_df.index[row], 'start_time'])
            self._df.loc[self._filtered_df.index[row], 'end_real_life_time'] = video_start_time + pd.to_timedelta(
                self._df.loc[self._filtered_df.index[row], 'end_time'])
            start_frame = pd.to_timedelta(
                self._df.loc[self._filtered_df.index[row], 'start_time']).total_seconds() * fps
            end_frame = pd.to_timedelta(self._df.loc[self._filtered_df.index[row], 'end_time']).total_seconds() * fps
            self._df.loc[self._filtered_df.index[row], 'start_frame'] = int(start_frame)
            self._df.loc[self._filtered_df.index[row], 'end_frame'] = int(end_frame)
            self._df.loc[self._filtered_df.index[row], 'visit_duration'] = max(0.08, (end_frame - start_frame) / fps)

        elif changed_col in ['start_real_life_time', 'end_real_life_time']:
            start_time = pd.to_datetime(
                self._df.loc[self._filtered_df.index[row], 'start_real_life_time']) - video_start_time
            end_time = pd.to_datetime(
                self._df.loc[self._filtered_df.index[row], 'end_real_life_time']) - video_start_time
            start_frame = start_time.total_seconds() * fps
            end_frame = end_time.total_seconds() * fps
            self._df.loc[self._filtered_df.index[row], 'start_frame'] = int(start_frame)
            self._df.loc[self._filtered_df.index[row], 'end_frame'] = int(end_frame)
            self._df.loc[self._filtered_df.index[row], 'start_time'] = start_time
            self._df.loc[self._filtered_df.index[row], 'end_time'] = end_time
            self._df.loc[self._filtered_df.index[row], 'visit_duration'] = max(0.08, (end_frame - start_frame) / fps)

        self.setVideoIDFilter(self._current_video_id)  # Reapply filter to update the view

    def rowCount(self, parent=QModelIndex()):
        return self._filtered_df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return len(self._visible_columns)

    def cycle_columns(self):
        self._current_cycle_index = (self._current_cycle_index + 1) % len(self._cycle_columns)
        self.update_visible_columns()

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsEnabled

        col_name = self._visible_columns[index.column()]
        if col_name == 'visit_duration':
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable

    def insertRows(self, position, rows=1, parent=QModelIndex()):
        self.beginInsertRows(parent, position, position + rows - 1)
        for i in range(rows):
            # Find unique visitor_id
            unique_visitor_id = self._df['visitor_id'].max() + 1 if not self._df['visitor_id'].empty else 1

            # Define default values for the new row
            default_values = {
                'video_id': self._current_video_id,
                'start_frame': 0,
                'end_frame': 0,
                'visit_duration': 0.0,
                'visitor_species': '',
                'visitor_id': unique_visitor_id,
                'flower_bboxes': [],
                'visitor_bboxes': [],
                'frame_numbers': [],
                'visit_ids': [],
                'start_time': pd.to_timedelta(0, unit='s'),  # Assuming these columns are present and timedelta type
                'end_time': pd.to_timedelta(0, unit='s'),
                'start_real_life_time': pd.Timestamp(0),  # Assuming these columns are present and timestamp type
                'end_real_life_time': pd.Timestamp(0),
                'flags': ""
            }

            empty_row = pd.Series(default_values, index=self._df.columns)
            self._df = pd.concat(
                [self._df.iloc[:position], pd.DataFrame([empty_row]), self._df.iloc[position:]]).reset_index(drop=True)
        self.setVideoIDFilter(self._current_video_id)  # Reapply filter
        self.endInsertRows()
        self.update_visits_entry(position)  # Update the dictionary
        return True

    def removeRows(self, position, rows=1, parent=QModelIndex()):
        self.beginRemoveRows(parent, position, position + rows - 1)
        for row in range(position, position + rows):
            visitor_id = self._filtered_df.iloc[row]['visitor_id']
            del self.shared_data.visitor_data[visitor_id]  # Remove the entry from the dictionary
        self._df.drop(self._filtered_df.index[position:position+rows], inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        self.setVideoIDFilter(self._current_video_id)  # Reapply filter
        self.endRemoveRows()
        self.update_visits_signal.emit(self.shared_data.visitor_data)
        return True

    def getDataFrame(self):
        return self._df

    def getDataFrameCopy(self):
        return self._df.copy()

    def get_background_color(self, visitor_id):
        color_map = {
            1: QColor(Qt.GlobalColor.yellow),
            2: QColor(Qt.GlobalColor.green),
            3: QColor(Qt.GlobalColor.red),
            # Add more mappings as needed
        }
        return color_map.get(visitor_id, QColor(Qt.GlobalColor.white))

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        # Create Visits Group
        create_visits_layout = QFormLayout()
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setValue(0.1)
        self.iou_threshold.setRange(0, 1)
        self.iou_threshold.setSingleStep(0.1)

        self.gap_tolerance = QSpinBox()
        self.gap_tolerance.setValue(30)
        self.gap_tolerance.setRange(0, 100)
        self.gap_tolerance.setSingleStep(1)

        self.min_box_confidence = QDoubleSpinBox()
        self.min_box_confidence.setRange(0, 1)
        self.min_box_confidence.setSingleStep(0.1)

        create_visits_layout.addRow("IOU Threshold:", self.iou_threshold)
        create_visits_layout.addRow("Gap Tolerance:", self.gap_tolerance)
        create_visits_layout.addRow("Min Box Confidence:", self.min_box_confidence)
        create_visits_group = QGroupBox("Create Visits")
        create_visits_group.setLayout(create_visits_layout)

        # Filter Visits Group
        filter_visits_layout = QFormLayout()
        self.min_duration_cb = QCheckBox()
        self.confidence_cb = QCheckBox()
        self.min_duration_le = QDoubleSpinBox()
        self.min_duration_le.setRange(0, 10000)
        self.min_duration_le.setSingleStep(1)
        self.confidence_le = QDoubleSpinBox()
        self.confidence_le.setRange(0, 1)
        self.confidence_le.setSingleStep(0.1)
        self.min_duration_le.setEnabled(False)
        self.confidence_le.setEnabled(False)

        # Connect checkboxes to custom methods
        self.min_duration_cb.stateChanged.connect(
            lambda state: self.min_duration_le.setEnabled(state == 2))
        self.confidence_cb.stateChanged.connect(
            lambda state: self.confidence_le.setEnabled(state == 2))

        filter_visits_layout.addRow(self.min_duration_cb, QLabel("By Minimum Duration:"))
        filter_visits_layout.addRow(self.min_duration_le)
        filter_visits_layout.addRow(self.confidence_cb, QLabel("By Confidence:"))
        filter_visits_layout.addRow(self.confidence_le)
        filter_visits_group = QGroupBox("Filter Visits")
        filter_visits_group.setLayout(filter_visits_layout)

        layout.addWidget(create_visits_group)
        layout.addWidget(filter_visits_group)
        self.setLayout(layout)

import logging

class CustomWidget(QWidget):
    seek_video_signal = pyqtSignal(int)
    seek_single_visit_signal = pyqtSignal(dict)
    update_visits_signal = pyqtSignal(dict)
    update_flowers_signal = pyqtSignal(dict)
    update_video_id_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()

        self.shared_data = shared_data
        self._default_message = ""
        self._default_icon = ALT_ICONS['info']

        self.buttons = {}

        # Setup layout
        layout = QVBoxLayout()

        # Setup main toolbar
        self.toolbar = QToolBar("Main Toolbar")
        layout.addWidget(self.toolbar)

        open_db_action = self.create_toolbar_button(ALT_ICONS['database'])
        open_db_action.clicked.connect(self.open_database)
        self.buttons['open_db'] = open_db_action
        self.toolbar.addWidget(open_db_action)

        save_db_action = self.create_toolbar_button(ALT_ICONS['save'])
        save_db_action.clicked.connect(self.save_database)
        self.buttons['save_db'] = save_db_action
        self.toolbar.addWidget(save_db_action)

        self.regenerate_visits_action = self.create_toolbar_button(ALT_ICONS['repeat'])
        self.regenerate_visits_action.clicked.connect(self.confirm_regenerate_visits)
        self.buttons['regenerate_visits'] = self.regenerate_visits_action
        self.toolbar.addWidget(self.regenerate_visits_action)

        self.filter_visits_action = self.create_toolbar_button(ALT_ICONS['filter'])
        self.filter_visits_action.clicked.connect(self.confirm_filter_visits)
        self.buttons['filter_visits'] = self.filter_visits_action
        self.toolbar.addWidget(self.filter_visits_action)

        help_action = self.create_toolbar_button(ALT_ICONS['help-circle'])
        help_action.clicked.connect(self.show_help)
        self.toolbar.addWidget(help_action)

        # Setup control panel
        self.control_panel = ControlPanel()
        self.control_panel.setFixedHeight(150)
        layout.addWidget(self.control_panel)

        # Setup table view toolbar
        self.table_view_toolbar = QToolBar("Table View Toolbar")
        layout.addWidget(self.table_view_toolbar)

        seek_button = self.create_toolbar_button(ALT_ICONS['video'])
        seek_button.setEnabled(False)
        seek_button.clicked.connect(self.seek_video)
        self.table_view_toolbar.addWidget(seek_button)
        self.buttons['seek'] = seek_button

        add_entry_action = self.create_toolbar_button(ALT_ICONS['plus-circle'])
        add_entry_action.clicked.connect(self.add_entry)
        self.buttons['add_entry'] = add_entry_action
        self.table_view_toolbar.addWidget(add_entry_action)

        remove_entry_action = self.create_toolbar_button(ALT_ICONS['minus-circle'])
        remove_entry_action.clicked.connect(self.confirm_remove_entry)
        self.buttons['remove_entry'] = remove_entry_action
        self.table_view_toolbar.addWidget(remove_entry_action)

        merge_entries_action = self.create_toolbar_button(ALT_ICONS['layers'])
        merge_entries_action.setEnabled(False)
        merge_entries_action.clicked.connect(self.merge_selected_entries)
        self.buttons['merge_entries'] = merge_entries_action
        self.table_view_toolbar.addWidget(merge_entries_action)

        previous_video_action = self.create_toolbar_button(ALT_ICONS['chevron-left'])
        previous_video_action.clicked.connect(self.show_previous_video)
        self.buttons['previous_video'] = previous_video_action
        self.table_view_toolbar.addWidget(previous_video_action)

        select_video_action = self.create_toolbar_button(ALT_ICONS['film'])
        select_video_action.clicked.connect(self.select_video)
        self.buttons['select_video'] = select_video_action
        self.table_view_toolbar.addWidget(select_video_action)

        next_video_action = self.create_toolbar_button(ALT_ICONS['chevron-right'])
        next_video_action.clicked.connect(self.show_next_video)
        self.buttons['next_video'] = next_video_action
        self.table_view_toolbar.addWidget(next_video_action)

        self.cycle_icons = [ALT_ICONS['hash'], ALT_ICONS['clock'], ALT_ICONS['calendar']]
        self.current_cycle_icon_index = 0
        self.cycle_columns_action = self.create_toolbar_button(self.cycle_icons[self.current_cycle_icon_index])
        self.cycle_columns_action.clicked.connect(self.cycle_columns)
        self.buttons['cycle_columns'] = self.cycle_columns_action
        self.table_view_toolbar.addWidget(self.cycle_columns_action)

        # In the __init__ method
        self.plot_icon = QIcon(ALT_ICONS['activity'])
        self.table_icon = QIcon(ALT_ICONS['table'])

        self.toggle_view_action = self.create_toolbar_button(self.plot_icon)
        self.toggle_view_action.clicked.connect(self.toggle_view)
        self.buttons['toggle_view'] = self.toggle_view_action
        self.table_view_toolbar.addWidget(self.toggle_view_action)

        settings_action = self.create_toolbar_button(ALT_ICONS['settings'])
        settings_action.clicked.connect(self.show_settings)
        self.buttons['settings'] = settings_action
        self.table_view_toolbar.addWidget(settings_action)

        # Disable all buttons except for open file
        for key, button in self.buttons.items():
            if key != 'open_db':
                button.setEnabled(False)

        # Setup stack and table view
        self.stack = QStackedWidget()
        self.table_view = CustomTableView(self)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stack.addWidget(self.table_view)

        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('white')  # Set initial background color
        self.stack.addWidget(self.plot_widget)

        layout.addWidget(self.stack)

        bottom_layout = QHBoxLayout()

        # Add a QWidget with horizontal layout to the toolbar
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        status_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        status_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Add a QLabel with an icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QPixmap(ALT_ICONS['info']).scaled(QSize(0, 0), Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))
        status_layout.addWidget(self.icon_label)
        status_layout.setStretch(status_layout.indexOf(self.icon_label), 0)

        # Setup the status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.setFixedHeight(34)
        self.status_bar.setFixedWidth(300)
        status_layout.addWidget(self.status_bar)
        status_layout.setStretch(status_layout.indexOf(self.status_bar), 1)

        # Add spacer to push the status bar to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        status_layout.addWidget(spacer)
        status_layout.setStretch(status_layout.indexOf(spacer), 4)

        # Add the status_widget to the table view toolbar
        bottom_layout.addWidget(status_widget)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

        self.model = None
        self.help_window = None  # Initialize help_window attribute
        self.current_video_id = None
        self.videos_df = None
        self.processor = None
        self._visits_df = None
        self._periods_df = None
        self._videos_df = None

    def create_toolbar_button(self, icon):
        button = QPushButton()
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(17, 17))
        button.setFixedSize(30, 30)  # Square button
        return button

    def update_button_states(self):
        selected_indexes = self.table_view.selectionModel().selectedRows()
        self.buttons['seek'].setEnabled(len(selected_indexes) == 1)
        self.buttons['merge_entries'].setEnabled(len(selected_indexes) > 1)

    def set_status_message(self, message, icon=None, timeout=0):
        self.status_bar.showMessage(message, timeout)
        if icon:
            self.icon_label.setPixmap(QPixmap(icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))
        QTimer.singleShot(timeout, self.reset_status_message)

    def reset_status_message(self):
        # Reset the status bar message to the default message
        self.status_bar.showMessage(self._default_message, 0)
        self.icon_label.setPixmap(QPixmap(self._default_icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))

    def update_video_id_status(self):
        if self.current_video_id:
            self._default_message = f"Video ID: {self.current_video_id}"
            self.set_status_message(f"Video ID: {self.current_video_id}", icon=ALT_ICONS['info'])

    def update_current_visits(self, packed_ids: list):
        if self.model:
            self.model.set_colored_visitor_ids(packed_ids)

    def init_processor(self, db_path: str):
        if not db_path:
            self.set_status_message("Please, open a database first.", icon=ALT_ICONS['alert-circle'], timeout=5000)

        # Initiate processor
        try:
            self.processor = VisitsProcessor(db_path)
        except Exception as e:
            self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

    def open_database(self):
        db_path, _ = QFileDialog.getOpenFileName(self, "Open Database", "", "SQLite Database Files (*.db)")
        if db_path:

            # Initiate processor
            self.init_processor(db_path)

            # Load periods if available
            try:
                if self.processor.has_periods:
                    periods_df = self.processor.load_periods()

                    if self.processor.videos_df is None or (isinstance(self.processor.videos_df, pd.DataFrame) and self.processor.videos_df.empty):
                        self.set_status_message("No videos found in the database", icon=ALT_ICONS['alert-circle'],
                                                timeout=5000)
                    elif self.processor.visits_df is None or (isinstance(self.processor.visits_df, pd.DataFrame) and self.processor.visits_df.empty):
                        self.set_status_message("No visits found in the database", icon=ALT_ICONS['alert-circle'],
                                                timeout=5000)
                    else:
                        self.set_tableview_model(periods_df, self.processor.videos_df, self.processor.visits_df)
                else:
                    self.set_status_message("Please, generate visits manually.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            except Exception as e:
                self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)

            self.buttons.get('regenerate_visits').setEnabled(True)

    def set_tableview_model(self, periods_df: pd.DataFrame, videos_df: pd.DataFrame, visits_df: pd.DataFrame = None):
        # Filter video_df to only those video_ids that are present in visits_df
        videos_df = videos_df[videos_df['video_id'].isin(periods_df['video_id'])]
        self._videos_df = videos_df

        self.model = PandasTableModel(periods_df,
                                      videos_df,
                                      visits_df,
                                      update_visits_signal=self.update_visits_signal,
                                      update_video_id_signal=self.update_video_id_signal,
                                      update_flowers_signal=self.update_flowers_signal)
        self.table_view.setModel(self.model)

        self.delegate = CustomItemDelegate(self.model, self.model.colored_visitor_ids)
        self.table_view.setItemDelegate(self.delegate)

        if not self._videos_df.empty:
            self.current_video_id = self._videos_df.iloc[0]['video_id']
            self.model.setVideoIDFilter(self.current_video_id)
            self.update_video_id_status()

            # Enable all buttons
            for key, button in self.buttons.items():
                button.setEnabled(True)

            selection_model = self.table_view.selectionModel()
            selection_model.selectionChanged.connect(self.update_button_states)

    def filter_visits_by_video_id(self):
        if self.model and self.current_video_id is not None:
            self.model.setVideoIDFilter(self.current_video_id)

    def save_database(self):
        if self.model:
            db_path, _ = QFileDialog.getSaveFileName(self, "Save Database", "", "SQLite Database Files (*.db)")
            if db_path:
                try:
                    self.processor = VisitsProcessor(db_path)

                    periods_df = self.model.getDataFrameCopy()

                    try:
                        periods_df = refine_periods(periods_df)
                    except Exception as e:
                        self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                        return

                    self.processor.save_periods(periods_df)
                except Exception as e:
                    self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                    return

    def seek_video(self):
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if len(selected_indexes) == 1:
            row = selected_indexes[0].row()
            original_index = self.model._filtered_df.index[row]  # Get the original index from the filtered DataFrame
            start_frame = self.model._df.loc[
                original_index, 'start_frame']  # Use the original index to get the value from the original DataFrame
            self.seek_video_signal.emit(start_frame)

            # Emit the single visit dictionary entry
            visitor_id = self.model._df.loc[original_index, 'visitor_id']
            single_visit_entry = self.shared_data.visitor_data.get(visitor_id, {})
            self.seek_single_visit_signal.emit(single_visit_entry)  # Emit the single visit dictionary entry

    def add_entry(self):
        if self.model:
            current_index = self.table_view.currentIndex()
            row = current_index.row() if current_index.isValid() else self.model.rowCount()
            self.model.insertRows(row)

    def confirm_remove_entry(self):
        current_index = self.table_view.currentIndex()
        if current_index.isValid():
            reply = QMessageBox.question(self, 'Confirm Remove Entry',
                                         "Are you sure you want to remove the selected entry?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.model.removeRows(current_index.row())

    def merge_selected_entries(self):
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if len(selected_indexes) < 2:
            return

        # Get the original indices from the filtered DataFrame
        original_indices = [self.model._filtered_df.index[idx.row()] for idx in selected_indexes]

        # Retain the first item and modify it
        primary_index = original_indices[0]
        print(original_indices)
        primary_row = self.model._df.loc[primary_index]

        # Initialize lists for merging
        flower_bboxes = list(primary_row['flower_bboxes'])
        visitor_bboxes = list(primary_row['visitor_bboxes'])
        frame_numbers = list(primary_row['frame_numbers'])
        visit_ids = list(primary_row['visit_ids'])
        flags = list(primary_row['flags'])

        start_frame = primary_row['start_frame']
        end_frame = primary_row['end_frame']

        for idx in original_indices[1:]:
            row = self.model._df.loc[idx]

            # Merge the lists
            flower_bboxes += list(row['flower_bboxes'])
            visitor_bboxes += list(row['visitor_bboxes'])
            frame_numbers += list(row['frame_numbers'])
            visit_ids += list(row['visit_ids'])
            flags += list(row['flags'])

            # Update start and end frames
            start_frame = min(start_frame, row['start_frame'])
            end_frame = max(end_frame, row['end_frame'])

        # Ensure the lengths match
        def set_list_value(index, value):
            column_index = self.model._df.columns.get_loc(index)
            current_value = self.model._df.at[primary_index, index]
            if isinstance(current_value, list) and isinstance(value, list):
                self.model._df.at[primary_index, index] = value
            else:
                self.model.setData(self.model.index(primary_index, column_index), value,
                                   role=Qt.ItemDataRole.EditRole)

        # Update the primary row with merged data using setData
        self.model._df.at[primary_index, 'start_frame'] = start_frame
        self.model._df.at[primary_index, 'end_frame'] = end_frame
        set_list_value('flower_bboxes', flower_bboxes)
        set_list_value('visitor_bboxes', visitor_bboxes)
        set_list_value('frame_numbers', frame_numbers)
        set_list_value('visit_ids', visit_ids)
        set_list_value('flags', flags)

        self.model.recalculate_linked_columns(self.model._filtered_df.index.get_loc(primary_index), 'start_frame')

        # Remove the other selected items
        for idx in original_indices[1:]:
            self.model.removeRows(self.model._filtered_df.index.get_loc(original_indices[1]))
            print("deleted: ", idx)

        print("deeted")

    def show_previous_video(self):

        if self._videos_df is not None and self.current_video_id is not None:
            current_index = self._videos_df[self._videos_df['video_id'] == self.current_video_id].index[0]
            if current_index > 0:
                self.current_video_id = self._videos_df.iloc[current_index - 1]['video_id']
                self.filter_visits_by_video_id()
                self.update_video_id_status()

    def show_next_video(self):

        if self._videos_df is not None and self.current_video_id is not None:
            current_index = self._videos_df[self._videos_df['video_id'] == self.current_video_id].index[0]
            if current_index < len(self._videos_df) - 1:
                self.current_video_id = self._videos_df.iloc[current_index + 1]['video_id']
                self.filter_visits_by_video_id()
                self.update_video_id_status()

    def select_video(self):

        if self._videos_df is not None:
            items = self._videos_df['video_id'].tolist()
            item, ok = QInputDialog.getItem(self, "Select Video ID", "Video IDs:", items, 0, False)
            if ok and item:
                self.current_video_id = item
                self.filter_visits_by_video_id()
                self.update_video_id_status()

    def cycle_columns(self):
        if self.model:
            self.model.cycle_columns()
            self.current_cycle_icon_index = (self.current_cycle_icon_index + 1) % len(self.cycle_icons)
            self.cycle_columns_action.setIcon(QIcon(self.cycle_icons[self.current_cycle_icon_index]))

    def open_plot_window(self):
        if self.model and self.processor:
            df = self.model.getDataFrameCopy()
            filtered_df = df[df['video_id'] == self.current_video_id]
            if not filtered_df.empty:
                video_info = self.processor.videos_df[self.processor.videos_df['video_id'] == self.current_video_id].iloc[0]
                video_start = video_info['start_time']
                video_end = video_info['end_time']
                self.plot_window = PlotWindow(filtered_df, video_start, video_end)
                self.plot_window.show()

    def toggle_view(self):
        if self.stack.currentIndex() == 0:  # Currently showing the table
            if self.model and self.processor:
                df = self.model.getDataFrameCopy()
                filtered_df = df[df['video_id'] == self.current_video_id]
                filtered_df = filtered_df.copy()
                if not filtered_df.empty:
                    video_info = self.processor.videos_df[self.processor.videos_df['video_id'] == self.current_video_id].iloc[0]
                    video_start = video_info['start_time']
                    video_end = video_info['end_time']
                    self.plot_visits(filtered_df, video_start, video_end)
            self.stack.setCurrentIndex(1)
            self.toggle_view_action.setIcon(self.table_icon)
        else:  # Currently showing the plot
            self.stack.setCurrentIndex(0)
            self.toggle_view_action.setIcon(self.plot_icon)

    def plot_visits(self, visits_df, video_start, video_end):
        self.plot_widget.clear()  # Clear the previous plot

        video_start = pd.to_datetime(video_start)
        video_end = pd.to_datetime(video_end)
        video_duration = (video_end - video_start).total_seconds()

        # Convert start_time and end_time to seconds from the video start
        visits_df.loc[:, f'start_time_num'] = (pd.to_timedelta(visits_df['start_time']) - pd.to_timedelta(0, unit='s')).dt.total_seconds()
        visits_df.loc[:, f'end_time_num'] = (pd.to_timedelta(visits_df['end_time']) - pd.to_timedelta(0, unit='s')).dt.total_seconds()

        # Plot the video timeline
        self.plot_widget.addItem(
            pg.BarGraphItem(x=[video_duration / 2], height=0.25, width=video_duration, brush='grey', pen='black'))

        # Prepare intervals for plotting
        def get_intervals(df, level_height):
            intervals = []
            current_y = level_height
            max_y = current_y

            for idx, row in df.iterrows():
                start_num = row['start_time_num']
                duration = row['end_time_num'] - row['start_time_num']
                overlap = False

                # Check for overlap
                for interval in intervals:
                    if not (start_num + duration < interval[0] or start_num > interval[0] + interval[1]):
                        overlap = True
                        current_y += level_height * 1  # Adjust spacing factor here
                        break

                if not overlap:
                    current_y = level_height * 1.5  # reset to first level if no overlap

                intervals.append((start_num, duration, current_y))
                max_y = max(max_y, current_y)

            return intervals, max_y

        # Plot the visits
        level_height = 0.25  # Adjust this value to change the height of each level
        visit_intervals, max_y = get_intervals(visits_df, level_height)
        for idx, (start, duration, y) in enumerate(visit_intervals):
            self.plot_widget.addItem(
                pg.BarGraphItem(x=[start + duration / 2], height=level_height, width=duration, y=y, brush='green',
                                pen='black'))

        # Adjust the plot
        self.plot_widget.setYRange(0, max_y + level_height)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Visits')
        self.plot_widget.getAxis('bottom').setTickSpacing(60, 10)  # Set tick spacing to 60 seconds

        # Set the color of the ticks and labels
        axis_color = 'black'  # Replace with your desired color
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=axis_color))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=axis_color))
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=axis_color))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=axis_color))

    def show_settings(self):
        if self.model:
            dialog = ColumnSettingsDialog(self.model)
            dialog.exec()

    def regenerate_visits(self):

        if not self.processor:
            self.set_status_message("Please open a database first.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

        # Get the values from the ControlPanel input fields
        try:
            iou_threshold = float(self.control_panel.iou_threshold.value())
        except ValueError:
            self.set_status_message("Invalid IOU Threshold value. Using 0.1", icon=ALT_ICONS['alert-circle'], timeout=5000)
            iou_threshold = 0.1

        try:
            gap_tolerance = int(self.control_panel.gap_tolerance.value())
        except ValueError:
            self.set_status_message("Invalid Gap Tolerance value. Using 30.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            gap_tolerance = 30

        try:
            min_box_confidence = float(self.control_panel.min_box_confidence.value())
        except ValueError:
            self.set_status_message("Invalid Min Box Confidence value. Using 0.0", icon=ALT_ICONS['alert-circle'], timeout=5000)
            min_box_confidence = 0.0

        # Fetch the entire DataFrame from the table view model
        try:
            self.set_status_message("Generating visits...", icon=ALT_ICONS['coffee'],
                                    timeout=0)

            # Filter the bounding boxes by confidence
            visits_df = self.processor.filter_bboxes_by_confidence(self.processor.visits_df, min_box_confidence)

            # Generate visits
            periods_df = self.processor.process_visits(visits_df,
                                               iou_threshold=iou_threshold,
                                               max_missing_frames=gap_tolerance)

            periods_df = refine_periods(periods_df)
        except Exception as e:
            self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

        self.set_status_message("Visits are ready!", icon=ALT_ICONS['gift'],
                                timeout=0)

        try:
            self.set_tableview_model(periods_df, self.processor.videos_df, visits_df)
        except Exception as e:
            self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

    def filter_visits(self):

        if not self.processor:
            self.set_status_message("Please open a database first.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

        # Get the values from the ControlPanel input fields
        try:
            min_duration_enabled = self.control_panel.min_duration_cb.isChecked()
            min_duration = self.control_panel.min_duration_le.value() if min_duration_enabled else None
        except ValueError:
            self.set_status_message("Invalid Minimum Duration value.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

        try:
            confidence_enabled = self.control_panel.confidence_cb.isChecked()
            confidence = self.control_panel.confidence_le.value() if confidence_enabled else None
        except ValueError:
            self.set_status_message("Invalid Confidence value.", icon=ALT_ICONS['alert-circle'], timeout=5000)
            return

        if self.model:
            periods_df = self.model.getDataFrameCopy()

            try:
                periods_df = refine_periods(periods_df)
            except Exception as e:
                self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                return

            if min_duration_enabled:
                self.set_status_message(f"Filtering visits...", icon=ALT_ICONS['coffee'], timeout=0)
                try:
                    periods_df = self.processor.filter_by_minimum_duration(periods_df, min_duration)
                except Exception as e:
                    self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                    return

            if confidence_enabled:
                self.set_status_message(f"Filtering visits...", icon=ALT_ICONS['coffee'], timeout=0)
                try:
                    periods_df = self.processor.filter_visits_by_confidence(periods_df, confidence)
                except Exception as e:
                    self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                    return

            try:
                self.set_tableview_model(periods_df, self.processor.videos_df, self.processor.visits_df)
            except Exception as e:
                self.set_status_message(str(e), icon=ALT_ICONS['alert-circle'], timeout=5000)
                return

    def show_help(self):
        if not self.help_window:
            self.help_window = HelpWindow(VISIT_VIEW_HELP)
        self.help_window.show()

    def confirm_regenerate_visits(self):
        reply = QMessageBox.question(self, 'Confirm Regenerate Visits',
                                     "Regenerating visits may result in loss of unsaved changes. Do you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.regenerate_visits()

    def confirm_filter_visits(self):
        reply = QMessageBox.question(self, 'Confirm Filter Visits',
                                     "Filtering may result in loss of visits if they do not meet the filter criteria. Do you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.filter_visits()

if __name__ == '__main__':

    @pyqtSlot(int)
    def seek_video(frame):
        print(f"Seeking to frame {frame}")


    @pyqtSlot(dict)
    def update_visits(visits):
        print(f"Got data about {len(visits)} visits.")


    @pyqtSlot(str)
    def update_video_id(video_id):
        print(f"Got Video ID {video_id}.")


    @pyqtSlot(str)
    def seek_video_dict(visit_dict):
        print(f"Got info about visit {visit_dict}.")

    app = QApplication(sys.argv)
    custom_widget = CustomWidget()
    custom_widget.seek_video_signal.connect(seek_video)
    custom_widget.update_visits_signal.connect(update_visits)
    custom_widget.update_video_id_signal.connect(update_video_id)
    custom_widget.seek_single_visit_signal.connect(seek_video_dict)
    custom_widget.show()
    sys.exit(app.exec())