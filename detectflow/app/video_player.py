import time
from detectflow.resources import IMGS
import sys
import PyQt6
import os
import configparser
pyqt = os.path.dirname(PyQt6.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt6/plugins")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTreeView,
                             QPushButton, QLabel, QProgressBar, QLineEdit, QTabWidget, QInputDialog,
                             QFileDialog, QTextEdit, QCheckBox, QSpinBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap, QIcon
from detectflow.config import DETECTFLOW_DIR
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout, QSizePolicy, QSpacerItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QAction
import json
from PyQt6.QtWidgets import (QFileDialog, QLineEdit, QCheckBox, QGridLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QHBoxLayout)
from PyQt6.QtWidgets import (QTreeView, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QPlainTextEdit, QTimeEdit, QSpinBox, QGraphicsView, QCheckBox, QStackedLayout, QToolBar)
from PyQt6.QtCore import Qt, QSize, QThreadPool, QEvent
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QSlider, QLabel, \
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QGraphicsPixmapItem, QFormLayout, QStatusBar, QStackedWidget
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import QUrl, Qt, QTimer, QPointF
from PyQt6.QtGui import QWheelEvent, QMouseEvent, QPen, QBrush, QFont, QTransform, QImage
from detectflow.video.video_data import Video
import hashlib
import shutil
from detectflow.app.commons import COLORS
from detectflow.resources import ICONS, IMGS
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtCore import QPoint, QDateTime
from detectflow.utils.hash import get_filepath_hash, get_timestamp_hash
from detectflow.app.screenshot import ScreenshotMixin
from PIL import ImageQt, Image
from detectflow.resources import ALT_ICONS
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QRect
from PyQt6.QtGui import QPen, QMouseEvent, QPainter, QColor
from PyQt6.QtWidgets import QGraphicsObject, QGraphicsItem, QGraphicsPixmapItem, QSlider, QListWidget, QGroupBox
from functools import partial
import os
import pickle
import json


class ColorMap:
    def __init__(self):
        self.visit_data = {}


COLOR_MAP = ColorMap()

from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QByteArray, QBuffer


def pixmap_to_bytes(pixmap):
    # Create a QByteArray to hold the image data
    byte_array = QByteArray()
    # Create a QBuffer and open it in write-only mode
    buffer = QBuffer(byte_array)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    # Save the pixmap to the buffer in PNG format
    pixmap.save(buffer, "PNG")
    # Get the raw bytes from the QByteArray
    return byte_array.data()


class VideoProgressSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.periods = []
        self.colors = COLORS
        self.color_map = COLOR_MAP
        self.color_map.visit_data = {}

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            value = self.minimum() + (self.maximum() - self.minimum()) * event.position().x() / self.width()
            self.setValue(int(value))
            event.accept()
        super().mousePressEvent(event)

    def set_periods(self, periods):
        """
        Accepts a list of tuples containing start and end values for periods to be highlighted.
        """
        self.periods = periods
        self.update()  # Trigger a repaint to show the new periods

    def clear_periods(self):
        """
        Clears all highlighted periods.
        """
        self.periods = []
        self.update()  # Trigger a repaint to remove the periods

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.periods:
            return

        painter = QPainter(self)
        overlapping_periods = self.get_overlapping_periods()

        max_overlaps = self.get_max_concurrent_periods()
        height_per_period = self.height() // (max_overlaps) if max_overlaps else self.height()

        drawn_periods = []
        level_map = {period: None for period in enumerate(self.periods)}
        for period, overlaps in overlapping_periods.items():
            start, end = period[:2]
            start_pos = self.position_from_value(start)
            end_pos = self.position_from_value(end)
            if len(overlaps) == 0:
                rect_top = 0
                rect = QRect(start_pos, rect_top, end_pos - start_pos, self.height())
                color = self.colors[0]
                painter.fillRect(rect, color)
                drawn_periods.append(period)
                level_map[period] = rect_top
                self.color_map.visit_data[period[2]] = color
            else:
                if period not in drawn_periods:
                    rect_top = 0
                    rect = QRect(start_pos, rect_top, end_pos - start_pos, height_per_period)
                    color = self.colors[0]
                    painter.fillRect(rect, color)
                    drawn_periods.append(period)
                    level_map[period] = rect_top
                    self.color_map.visit_data[period[2]] = color
                for i, overlap in enumerate(overlaps):
                    if overlap not in drawn_periods:
                        occupied_levels = [level_map[period] for period in overlapping_periods[overlap] if period in drawn_periods]
                        level = next((i for i in range(max_overlaps) if i not in occupied_levels), 0)
                        rect_top = level * height_per_period
                        start_pos = self.position_from_value(overlap[0])
                        end_pos = self.position_from_value(overlap[1])
                        rect = QRect(start_pos, rect_top, end_pos - start_pos, height_per_period)
                        color = self.colors[(i+1) % len(self.colors)]
                        painter.fillRect(rect, color)
                        drawn_periods.append(overlap)
                        level_map[overlap] = level
                        self.color_map.visit_data[overlap[2]] = color

    def get_overlapping_periods(self):
        """
        Determines overlapping periods.
        Returns a dictionary with periods as keys and a list of overlapping periods as values.
        """
        overlapping_periods = {period: [] for period in self.periods}
        for i, period1 in enumerate(self.periods):
            for j, period2 in enumerate(self.periods):
                if i != j and self.periods_overlap(period1, period2):
                    overlapping_periods[period1].append(period2)
        return overlapping_periods

    def get_max_concurrent_periods(self):
        """
        Determines the maximum number of concurrent periods at any given time.
        """
        events = []
        for start, end, *_ in self.periods:
            events.append((start, 1))  # 1 for start of period
            events.append((end, -1))  # -1 for end of period

        events.sort()
        max_concurrent = 0
        current_concurrent = 0

        for time, event in events:
            current_concurrent += event
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent

        return max_concurrent

    def periods_overlap(self, period1, period2):
        """
        Check if two periods overlap.
        """
        start1, end1 = period1[:2]
        start2, end2 = period2[:2]
        return not (end1 <= start2 or start1 >= end2)

    def position_from_value(self, value):
        """
        Converts a slider value to a pixel position.
        """
        if self.maximum() == 0 or self.width() == 0:
            return 0
        else:
            return int(value // (self.maximum() / self.width()))

class ZoomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

        event.accept()


class BoundingBoxItem:
    def __init__(self, x1, y1, x2, y2, confidence=None, **kwargs):
        self.rect_item = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
        self.rect_item.setPen(QPen(kwargs.get('color', QColor(255, 0, 0)), kwargs.get('line', 1)))  # Red color bounding box

        self.text_item = None
        self.background_rect_item = None
        if confidence is not None:
            confidence_text = f"{confidence:.2f}"
            self.text_item = QGraphicsTextItem(confidence_text)
            self.text_item.setDefaultTextColor(kwargs.get('text_color', QColor(255, 255, 255)))  # White text
            font = QFont()
            font.setPointSize(4)  # Set smaller font size
            self.text_item.setFont(font)

            # Create background rectangle for the text
            text_rect = self.text_item.boundingRect()
            self.background_rect_item = QGraphicsRectItem(0, text_rect.height() // 2, text_rect.width() // 2, text_rect.height() // 2)
            self.background_rect_item.setBrush(QBrush(kwargs.get('color', QColor(255, 0, 0))))  # Red background
            self.background_rect_item.setPen(QPen(kwargs.get('color', QColor(255, 0, 0)), kwargs.get('line', 1)))  # Remove border
            self.background_rect_item.setPos(x1, y1 - text_rect.height())
            self.text_item.setPos(x1 - text_rect.width() // 4, y1 - text_rect.height() // 1.2)

    def add_to_scene(self, scene):
        scene.addItem(self.rect_item)
        if self.text_item:
            scene.addItem(self.background_rect_item)
            scene.addItem(self.text_item)

    def remove_from_scene(self, scene):
        scene.removeItem(self.rect_item)
        if self.text_item:
            scene.removeItem(self.background_rect_item)
            scene.removeItem(self.text_item)


class VideoPlayer(QWidget):
    video_changed = pyqtSignal(str)
    current_visits_changed = pyqtSignal(list)


    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt6 Video Player")
        self.setGeometry(350, 100, 700, 500)
        self.is_fullscreen = False

        self.video = None
        self.frame_widget = None
        self.frame_layout = None
        self.current_speed_label = None
        self.frame_label = None
        self.frame_icon_label = None
        self.time_label = None
        self.playback_speed_slider = None
        self.seek_slider = None
        self.jump_forward_button = None
        self.jump_backward_button = None
        self.play_pause_button = None
        self.video_item = None
        self.graphics_scene = None
        self.graphics_view = None
        self.audio_output = None
        self.media_player = None
        self.color_map = COLOR_MAP
        self.highlight_periods = []
        self.current_highlight_periods = []

        self.video_width = 0
        self.video_height = 0
        self.fps = 25
        self.total_frames = 0

        self.bounding_boxes = {}  # Dictionary to store bounding boxes for each frame
        self.bbox_persistence = 0  # Number of frames for which bounding boxes should persist
        self.current_bboxes = []  # List to keep track of current bounding box graphics items

        self.reference_bounding_boxes = {}  # Dictionary to store reference bounding boxes for each frame
        self.reference_bbox_persistence = 0  # Number of frames for which reference bounding boxes should persist
        self.current_reference_bboxes = []  # List to keep track of current reference bounding box graphics items

        self.init_ui()

    def init_ui(self):
        self.setup_media_player()
        self.setup_controls()
        self.setup_layout()

    def set_status_message(self, message, icon=None, timeout=0):
        pass

    def disable_navigation(self):
        self.play_pause_button.setEnabled(False)
        self.jump_backward_button.setEnabled(False)
        self.jump_forward_button.setEnabled(False)
        self.seek_slider.setEnabled(False)

    def enable_navigation(self):
        self.play_pause_button.setEnabled(True)
        self.jump_backward_button.setEnabled(True)
        self.jump_forward_button.setEnabled(True)
        self.seek_slider.setEnabled(True)

    def setup_media_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        self.graphics_view = ZoomGraphicsView()
        self.graphics_scene = QGraphicsScene(self.graphics_view)
        self.graphics_scene.setParent(self)
        self.video_item = QGraphicsVideoItem()
        self.graphics_scene.addItem(self.video_item)
        self.graphics_view.setScene(self.graphics_scene)
        self.media_player.setVideoOutput(self.video_item)

    def setup_controls(self):
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        self.jump_backward_button = QPushButton()
        self.jump_backward_button.setIcon(QIcon.fromTheme("media-skip-backward"))
        self.jump_backward_button.clicked.connect(self.jump_backward)

        self.jump_forward_button = QPushButton()
        self.jump_forward_button.setIcon(QIcon.fromTheme("media-skip-forward"))
        self.jump_forward_button.clicked.connect(self.jump_forward)

        self.seek_slider = VideoProgressSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.valueChanged.connect(self.set_position)

        self.playback_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_speed_slider.setRange(0, 40)
        self.playback_speed_slider.setTickInterval(10)
        self.playback_speed_slider.setSingleStep(10)
        self.playback_speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.playback_speed_slider.setValue(10)
        self.playback_speed_slider.installEventFilter(self)
        self.playback_speed_slider.valueChanged.connect(self.set_playback_speed)

        self.time_label = QLabel('00:00 / 00:00')
        self.frame_icon_label = QLabel()
        self.frame_icon_label.setPixmap(QIcon.fromTheme("media-playback-stop").pixmap(12, 12))
        self.frame_label = QLabel('-')
        self.frame_label.setFixedWidth(30)
        self.current_speed_label = QLabel('1.0×')
        self.current_speed_label.setFixedWidth(27)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.update_frame_info)
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

    def setup_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.graphics_view)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.jump_backward_button)
        control_layout.addWidget(self.jump_forward_button)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.seek_slider)
        control_layout.setStretch(control_layout.indexOf(self.seek_slider), 4)

        self.frame_layout = QHBoxLayout()
        self.frame_layout.addWidget(self.frame_icon_label)
        self.frame_layout.addWidget(self.frame_label)
        self.frame_widget = QWidget()
        self.frame_widget.setLayout(self.frame_layout)

        control_layout.addWidget(self.frame_widget)
        control_layout.addWidget(self.playback_speed_slider)
        control_layout.setStretch(control_layout.indexOf(self.playback_speed_slider), 1)
        control_layout.addWidget(self.current_speed_label)

        layout.addLayout(control_layout)
        self.setLayout(layout)

        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-start"))
        else:
            self.media_player.play()
            self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))

    def open_file(self, file_path):
        def play_and_pause():
            self.media_player.play()
            self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
            QTimer.singleShot(100, pause_and_fit)

        def pause_and_fit():
            #self.media_player.pause()
            self.graphics_view.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.graphics_view.update()

        if file_path != '':
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.setPosition(0)
            QTimer.singleShot(100, play_and_pause)
            self.get_video_details(file_path)
            self.video_changed.emit(file_path)

    def get_video_details(self, file_path):
        self.video = Video(file_path)
        self.video_width = self.video.frame_width
        self.video_height = self.video.frame_height
        self.fps = self.video.fps
        print(self.fps)
        self.fps = self.fps if self.fps != 29 else 30
        self.mspf = int(1000 / self.fps) if self.fps > 0 else 40
        self.total_frames = self.video.total_frames

    def jump_forward(self):
        self.media_player.setPosition(self.media_player.position() + self.mspf if self.mspf and self.mspf > 0 else 40)  # Assuming 25fps, one frame is ~40ms

    def jump_backward(self):
        self.media_player.setPosition(self.media_player.position() - self.mspf if self.mspf and self.mspf > 0 else 40)  # Assuming 25fps, one frame is ~40ms

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.seek_slider.setValue(position)
        self.update_time_label()
        self.get_current_periods(position)

    def duration_changed(self, duration):
        self.seek_slider.setRange(0, duration)
        self.update_time_label()

    def set_playback_speed(self, value):
        def map_slider_value_to_speed(v):
            if v <= 20:
                return v / 10.0
            elif 20 < v <= 30:
                return v / 5.0
            else:
                return v / 4.0

        speed = map_slider_value_to_speed(value)
        #speed = value / 10.0
        self.media_player.setPlaybackRate(speed)
        self.current_speed_label.setText(f'{speed:.1f}×')

    def update_time_label(self):
        current_time = self.media_player.position() // 1000
        total_time = self.media_player.duration() // 1000
        self.time_label.setText(f'{current_time // 60:02}:{current_time % 60:02} / {total_time // 60:02}:{total_time % 60:02}')

    def update_frame_info(self):
        frame_number = self.media_player.position() // self.mspf if self.mspf and self.mspf > 0 else 40  # Assuming 25fps, one frame is ~40ms
        self.frame_label.setText(f'{frame_number}')
        self.update_bounding_boxes(frame_number)

    def set_highlight_periods(self, periods):
        self.highlight_periods = periods
        self.seek_slider.set_periods(periods)

    def clear_highlight_periods(self):
        self.seek_slider.clear_periods()

    def get_current_periods(self, time_in_milliseconds):
        new_ids = []

        for start, end, interval_id in self.highlight_periods:
            if start <= time_in_milliseconds <= end:
                new_ids.append(interval_id)

        # Filter out IDs that are not in the new list of IDs
        current_ids = [id for id in self.current_highlight_periods if id in new_ids]

        # Add any new IDs that are not already in current_ids
        for id in new_ids:
            if id not in current_ids:
                current_ids.append(id)

        if current_ids != self.current_highlight_periods:
            self.current_highlight_periods = current_ids.copy()  # Update the previous_ids
            packed_ids = [(id, self.color_map.visit_data[id]) for id in current_ids]
            self.current_visits_changed.emit(packed_ids)

    def set_bounding_boxes(self, bounding_boxes, persistence=0):
        """
        Sets the bounding boxes data and persistence.

        :param bounding_boxes: Dictionary where key is frame number and value is list of bounding boxes.
        :param persistence: Number of frames for which the bounding box should persist.
        """
        self.bounding_boxes = bounding_boxes
        self.bbox_persistence = persistence

    def set_reference_bounding_boxes(self, reference_bounding_boxes, persistence=0):
        """
        Sets the reference bounding boxes data and persistence.

        :param reference_bounding_boxes: Dictionary where key is frame number and value is list of reference bounding boxes.
        :param persistence: Number of frames for which the reference bounding box should persist.
        """
        self.reference_bounding_boxes = reference_bounding_boxes
        self.reference_bbox_persistence = persistence

    def clear_current_bounding_boxes(self):
        self.clear_bounding_boxes(self.current_bboxes)

    def clear_reference_bounding_boxes(self):
        self.clear_bounding_boxes(self.current_reference_bboxes)

    def clear_bounding_boxes(self, current_bboxes):
        for bbox in current_bboxes:
            bbox.remove_from_scene(self.graphics_scene)
        current_bboxes.clear()

    def update_bounding_boxes(self, frame_number, **kwargs):
        """
        Updates the bounding boxes displayed on the video based on the current frame number.

        :param frame_number: The current frame number in the video.
        :param kwargs: Optional arguments to customize the bounding box display.

        - Keyword Arguments:
            - color: QColor object to set the color of the bounding box.
            - text_color: QColor object to set the color of the text.
            - line: Line width of the bounding box.
        """
        # Clear current bounding boxes
        self.clear_bounding_boxes(self.current_bboxes)
        self.clear_bounding_boxes(self.current_reference_bboxes)

        # Add bounding boxes
        self.current_bboxes = self.add_bounding_boxes(self.bounding_boxes, frame_number,
                                                      self.bbox_persistence,
                                                      kwargs.get('color', QColor(255, 0, 0)),
                                                      kwargs.get('line', 1),
                                                      kwargs.get('text_color', QColor(255, 255, 255)))

        # Add reference bounding boxes
        self.current_reference_bboxes = self.add_bounding_boxes(self.reference_bounding_boxes, frame_number,
                                                                self.reference_bbox_persistence,
                                                                QColor(0, 255, 255), 1,
                                                                QColor(255, 255, 255))

    def add_bounding_boxes(self, bounding_boxes, frame_number, persistence, color, line, text_color):
        current_bboxes = []
        if persistence == 0:
            found = False
            for i in range(frame_number, -1, -1):
                if i in bounding_boxes:
                    found = True
                    for bbox in bounding_boxes[i]:

                        try:
                            if len(bbox) > 4 and isinstance(bbox[-1], int):
                                visitor_id = int(bbox[-1])
                                color = self.color_map.visit_data[visitor_id]
                        except:
                            pass

                        x1, y1, x2, y2 = bbox[:4]
                        confidence = bbox[4] if len(bbox) > 4 else None
                        mapped_x1, mapped_y1, mapped_x2, mapped_y2 = self.map_to_video_coords(x1, y1, x2, y2)
                        bbox_item = BoundingBoxItem(mapped_x1, mapped_y1, mapped_x2, mapped_y2, confidence,
                                                    color=color, text_color=text_color, line=line)
                        bbox_item.add_to_scene(self.graphics_scene)
                        current_bboxes.append(bbox_item)
                    if found:
                        break
        else:
            for i in range(frame_number, frame_number - persistence - 1, -1):
                if i in bounding_boxes:
                    for bbox in bounding_boxes[i]:

                        try:
                            if len(bbox) > 4 and isinstance(bbox[-1], int):
                                visitor_id = int(bbox[-1])
                                color = self.color_map.visit_data[visitor_id]
                        except:
                            pass

                        x1, y1, x2, y2 = bbox[:4]
                        confidence = bbox[4] if len(bbox) > 4 else None
                        mapped_x1, mapped_y1, mapped_x2, mapped_y2 = self.map_to_video_coords(x1, y1, x2, y2)
                        bbox_item = BoundingBoxItem(mapped_x1, mapped_y1, mapped_x2, mapped_y2, confidence,
                                                    color=color, text_color=text_color, line=line)
                        bbox_item.add_to_scene(self.graphics_scene)
                        current_bboxes.append(bbox_item)
        return current_bboxes

    def map_to_video_coords(self, x1, y1, x2, y2):
        """
        Maps the bounding box coordinates to the video item's coordinates.

        :param x1, y1, x2, y2: Bounding box coordinates in the video frame.
        :return: Transformed coordinates.
        """

        if self.video_width > 0 and self.video_height > 0:
            video_rect = self.video_item.boundingRect()
            width_scale = video_rect.width() / self.video_width
            height_scale = video_rect.height() / self.video_height
        else:
            video_size = self.video_item.nativeSize()
            if video_size.isValid():
                width_scale = self.video_item.boundingRect().width() / video_size.width()
                height_scale = self.video_item.boundingRect().height() / video_size.height()
            else:
                return x1, y1, x2, y2

        mapped_x1 = x1 * width_scale
        mapped_y1 = y1 * height_scale
        mapped_x2 = x2 * width_scale
        mapped_y2 = y2 * height_scale

        # Adjust for position offset of the video item
        top_left = self.video_item.boundingRect().topLeft()
        mapped_x1 += top_left.x()
        mapped_y1 += top_left.y()
        mapped_x2 += top_left.x()
        mapped_y2 += top_left.y()

        return int(mapped_x1), int(mapped_y1), int(mapped_x2), int(mapped_y2)

    def map_to_scene_coords(self, x1, y1, x2, y2):
        """
        Maps the bounding box coordinates from the graphic scene back to the video item's coordinates.

        :param x1, y1, x2, y2: Bounding box coordinates in the graphic scene.
        :return: Transformed coordinates in the video frame.
        """

        if self.video_width > 0 and self.video_height > 0:
            video_rect = self.video_item.boundingRect()
            width_scale = video_rect.width() / self.video_width
            height_scale = video_rect.height() / self.video_height
        else:
            video_size = self.video_item.nativeSize()
            if video_size.isValid():
                width_scale = self.video_item.boundingRect().width() / video_size.width()
                height_scale = self.video_item.boundingRect().height() / video_size.height()
            else:
                return x1, y1, x2, y2

        # Adjust for position offset of the video item
        top_left = self.video_item.boundingRect().topLeft()
        x1 -= top_left.x()
        y1 -= top_left.y()
        x2 -= top_left.x()
        y2 -= top_left.y()

        # Reverse the scaling
        mapped_x1 = x1 // width_scale
        mapped_y1 = y1 // height_scale
        mapped_x2 = x2 // width_scale
        mapped_y2 = y2 // height_scale

        return int(mapped_x1), int(mapped_y1), int(mapped_x2), int(mapped_y2)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            if source == self.playback_speed_slider:
                self.playback_speed_slider.setValue(10)
                self.set_playback_speed(10)
                return True
        return super().eventFilter(source, event)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.clear_reference_bounding_boxes()
            self.clear_current_bounding_boxes()

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.graphics_view.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.graphics_view.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)


class Resizer(QGraphicsObject):
    resize = pyqtSignal(QGraphicsItem.GraphicsItemChange, QPointF)

    def __init__(self, rect=QRectF(0, 0, 10, 10), parent=None):
        super().__init__(parent)
        try:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            self.rect = rect
            self.hide()
        except Exception as e:
            print(e)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            pen = QPen()
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawEllipse(self.rect)
        self.update()

    def itemChange(self, change, value):
        self.prepareGeometryChange()
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self.isSelected():
                self.resize.emit(change, self.pos())
        return super(Resizer, self).itemChange(change, value)


class ResizablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, top_left_x, top_left_y, graphic, rect=QRectF(0, 0, 100, 100), parent=None, scene=None):
        super().__init__(parent)
        self.rect = rect
        self.setPixmap(graphic)
        self.graphic = graphic
        self.mousePressPos = None
        self.mousePressRect = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setPos(top_left_x, top_left_y)

        # Resizer actions
        self.resizer = Resizer(parent=self)
        r_width = self.resizer.boundingRect().width() - 4
        self.r_offset = QPointF(r_width, r_width)
        self.resizer.setPos(self.boundingRect().bottomRight() - self.r_offset)

        slot = partial(resize_pixmap_item, self)
        self.resizer.resize.connect(slot)

    def set_tag(self, item_id):
        self.tag = item_id

    def get_tag(self):
        return self.tag

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Update the position in self.image_items if available
            if self.scene() and self.scene().parent() and hasattr(self.scene().parent(), 'image_items'):
                unique_hash = self.get_tag()
                if unique_hash in self.scene().parent().image_items:
                    self.scene().parent().image_items[unique_hash]['coords'] = (value.x(), value.y())

        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        if self.isSelected():
            self.resizer.show()
            self.resizer.setPos(self.boundingRect().bottomRight() - self.r_offset)
        else:
            self.resizer.hide()

    def hoverLeaveEvent(self, event):
        self.resizer.hide()


@pyqtSlot(QGraphicsItem.GraphicsItemChange, QPointF)
def resize_pixmap_item(item, change, value):
    pixmap = item.graphic.scaled(int(value.x()), int(value.y()),transformMode=Qt.TransformationMode.SmoothTransformation)
    item.setPixmap(pixmap)
    item.prepareGeometryChange()
    item.update()

    # Update size in self.image_items
    if item.scene() and item.scene().parent() and hasattr(item.scene().parent(), 'image_items'):
        unique_hash = item.get_tag()  # Assuming you have a method to get the item's unique tag
        if unique_hash in item.scene().parent().image_items:
            item.scene().parent().image_items[unique_hash]['size'] = (int(value.x()), int(value.y()))
            print("updated size")


class ResizableTextItem(QGraphicsTextItem):
    textChanged = pyqtSignal(str)

    def __init__(self, text, x, y, parent=None):
        super().__init__(text, parent)
        self.tag = None
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.document().contentsChanged.connect(self.on_text_changed)

    def set_tag(self, item_id):
        self.tag = item_id

    def get_tag(self):
        return self.tag

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Update the position in self.image_items if available
            if self.scene() and self.scene().parent() and hasattr(self.scene().parent(), 'text_items'):
                item_id = self.tag  # Unique ID for the text item
                if item_id in self.scene().parent().text_items:
                    self.scene().parent().text_items[item_id]['coords'] = (value.x(), value.y())
                    print("pos changed")
        return super().itemChange(change, value)

    def on_text_changed(self):
        self.textChanged.emit(self.toPlainText())


class SettingsWindow(QWidget):
    def __init__(self, video_player):
        super().__init__()
        self.video_player = video_player
        self.setWindowTitle("Settings")
        self.setGeometry(450, 150, 400, 300)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        file_settings_group = QGroupBox("File settings")
        file_settings_layout = QVBoxLayout()

        self.app_data_dir_label = QLabel(f'App data directory: {self.video_player.app_data_dir}')
        self.select_path_button = QPushButton("Select Path")
        self.select_path_button.clicked.connect(self.select_path)
        self.auto_download_checkbox = QCheckBox("Automatically download videos")
        self.auto_download_checkbox.setChecked(self.video_player.download_videos)

        file_settings_layout.addWidget(self.app_data_dir_label)
        file_settings_layout.addWidget(self.select_path_button)
        file_settings_layout.addWidget(self.auto_download_checkbox)
        file_settings_group.setLayout(file_settings_layout)

        s3_settings_group = QGroupBox("S3 details")
        s3_settings_layout = QFormLayout()

        self.host_base_input = QLineEdit(self.video_player.host_base)
        self.use_https_checkbox = QCheckBox()
        self.use_https_checkbox.setChecked(self.video_player.use_https)
        self.access_key_input = QLineEdit(self.video_player.access_key)
        self.secret_key_input = QLineEdit(self.video_player.secret_key)
        self.host_bucket_input = QLineEdit(self.video_player.host_bucket)

        self.select_file_button = QPushButton("Select .cfg File")
        self.select_file_button.clicked.connect(self.load_cfg_file)

        s3_settings_layout.addRow("Host base", self.host_base_input)
        s3_settings_layout.addRow("Use HTTPS", self.use_https_checkbox)
        s3_settings_layout.addRow("Access key", self.access_key_input)
        s3_settings_layout.addRow("Secret key", self.secret_key_input)
        s3_settings_layout.addRow("Host bucket", self.host_bucket_input)
        s3_settings_layout.addWidget(self.select_file_button)
        s3_settings_group.setLayout(s3_settings_layout)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.save_settings)

        layout.addWidget(file_settings_group)
        layout.addWidget(s3_settings_group)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def select_path(self):
        path_dialog = QFileDialog(self)
        path_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if path_dialog.exec():
            selected_path = path_dialog.selectedFiles()[0]
            self.video_player.app_data_dir = selected_path
            self.app_data_dir_label.setText(f'App data directory: {selected_path}')

    def load_cfg_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Config Files (*.cfg)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            config = configparser.ConfigParser()
            config.read(selected_file)

            if 'default' in config:
                default_config = config['default']
                self.host_base_input.setText(default_config.get('host_base', '').strip("'"))
                self.use_https_checkbox.setChecked(default_config.get('use_https', '').strip("'").lower() == 'true')
                self.access_key_input.setText(default_config.get('access_key', '').strip("'"))
                self.secret_key_input.setText(default_config.get('secret_key', '').strip("'"))
                self.host_bucket_input.setText(default_config.get('host_bucket', '').strip("'"))

    def save_settings(self):
        self.video_player.download_videos = self.auto_download_checkbox.isChecked()
        self.video_player.host_base = self.host_base_input.text()
        self.video_player.use_https = self.use_https_checkbox.isChecked()
        self.video_player.access_key = self.access_key_input.text()
        self.video_player.secret_key = self.secret_key_input.text()
        self.video_player.host_bucket = self.host_bucket_input.text()
        self.close()


class InteractiveVideoPlayer(VideoPlayer, ScreenshotMixin):
    def __init__(self):

        self.add_screenshot_button = None
        self.add_image_button = None
        self.load_button = None
        self.save_button = None
        self.toolbar = None
        self.toolbar_icons = {}
        self._current_items = {}
        self.image_items = {}
        self.text_items = {}
        self._default_message = ""
        self._default_icon = ALT_ICONS['info']
        self.current_video_index = None
        self.show_bboxes = True
        self.show_reference_bboxes = True
        self.show_highlights = True
        self._periods = []
        self._progress_lock = False

        # Ensure app_data directory exists
        self.app_data_dir = "app_data"
        os.makedirs(self.app_data_dir, exist_ok=True)

        # Settings attributes
        self.video_files = []
        self._download_videos = False
        self._host_base = ""
        self._use_https = False
        self._access_key = ""
        self._secret_key = ""
        self._host_bucket = ""
        self._config = None

        VideoPlayer.__init__(self)
        ScreenshotMixin.__init__(self)

    def setup_layout(self):
        layout = QVBoxLayout()

        # Add the toolbar
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.toolbar = QToolBar("Main Toolbar")
        self.open_video_button = self.create_toolbar_button(ALT_ICONS['video'])
        self.open_folder_button = self.create_toolbar_button(ALT_ICONS['folder-plus'])
        self.prev_video_button = self.create_toolbar_button(ALT_ICONS['chevron-left'])
        self.prev_video_button.setEnabled(False)
        self.show_playlist_button = self.create_toolbar_button(ALT_ICONS['list'])
        self.show_playlist_button.setEnabled(False)
        self.next_video_button = self.create_toolbar_button(ALT_ICONS['chevron-right'])
        self.next_video_button.setEnabled(False)

        self.save_button = self.create_toolbar_button(ALT_ICONS['save'])
        self.load_button = self.create_toolbar_button(ALT_ICONS['share'])
        self.add_image_button = self.create_toolbar_button(ALT_ICONS['image'])
        self.add_clipboard_image_button = self.create_toolbar_button(ALT_ICONS['clipboard'])
        self.add_text_button = self.create_toolbar_button(ALT_ICONS['type'])
        self.duplicate_button = self.create_toolbar_button(ALT_ICONS['plus-square'])
        self.delete_button = self.create_toolbar_button(ALT_ICONS['minus-square'])
        self.add_screenshot_button = self.create_toolbar_button(ALT_ICONS['crop'])
        self.toggle_bboxes_button = self.create_toolbar_button(IMGS['insect-box'])
        self.toggle_reference_bboxes_button = self.create_toolbar_button(ALT_ICONS['flower-box'])
        self.toggle_highlights_button = self.create_toolbar_button(ALT_ICONS['edit-3'])
        self.settings_button = self.create_toolbar_button(ALT_ICONS['settings'])


        self.toolbar.addWidget(self.open_video_button)
        self.toolbar.addWidget(self.open_folder_button)
        self.toolbar.addWidget(self.prev_video_button)
        self.toolbar.addWidget(self.show_playlist_button)
        self.toolbar.addWidget(self.next_video_button)

        self.toolbar.addSeparator()

        self.toolbar.addWidget(self.save_button)
        self.toolbar.addWidget(self.load_button)

        self.toolbar.addSeparator()

        self.toolbar.addWidget(self.add_image_button)
        self.toolbar.addWidget(self.add_clipboard_image_button)
        self.toolbar.addWidget(self.add_screenshot_button)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.add_text_button)
        self.toolbar.addWidget(self.duplicate_button)
        self.toolbar.addWidget(self.delete_button)
        self.toolbar.addSeparator()

        self.toolbar.addWidget(self.toggle_bboxes_button)
        self.toolbar.addWidget(self.toggle_reference_bboxes_button)
        self.toolbar.addWidget(self.toggle_highlights_button)

        self.toolbar.addSeparator()

        self.toolbar.addWidget(self.settings_button)
        self.toolbar.setContentsMargins(0, 0, 0, 0)

        self.top_layout.addWidget(self.toolbar)
        layout.addLayout(self.top_layout)

        # Connect buttons to functions
        self.open_video_button.clicked.connect(self.open_file_dialog)
        self.add_image_button.clicked.connect(self.add_image)
        self.add_screenshot_button.clicked.connect(self.enable_screenshot_mode)
        self.save_button.clicked.connect(self.save_scene)
        self.load_button.clicked.connect(self.load_scene)
        self.add_text_button.clicked.connect(self.add_text_item)
        self.duplicate_button.clicked.connect(self.duplicate_selected_item)
        self.delete_button.clicked.connect(self.delete_selected_item)
        self.add_clipboard_image_button.clicked.connect(self.add_image_from_clipboard)
        self.open_folder_button.clicked.connect(self.open_folder_dialog)
        self.prev_video_button.clicked.connect(self.prev_video)
        self.show_playlist_button.clicked.connect(self.show_playlist)
        self.next_video_button.clicked.connect(self.next_video)
        self.toggle_bboxes_button.clicked.connect(self.toggle_bboxes)
        self.toggle_reference_bboxes_button.clicked.connect(self.toggle_reference_bboxes)
        self.toggle_highlights_button.clicked.connect(self.toggle_highlights)
        self.settings_button.clicked.connect(self.open_settings_dialog)

        # Add a QWidget with horizontal layout to the toolbar
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        status_layout.setSpacing(0)  # Remove spacing
        status_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Add spacer to push the status bar to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        status_layout.addWidget(spacer)
        status_layout.setStretch(status_layout.indexOf(spacer), 4)

        # Add in the setup_controls method
        self.stack_widget = QStackedWidget()
        self.stack_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        status_layout.setStretch(status_layout.indexOf(self.stack_widget), 0)

        # Add a QLabel with an icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QPixmap(ALT_ICONS['info']).scaled(QSize(0, 0), Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))
        self.icon_label.setContentsMargins(10,0,10,0)
        status_layout.addWidget(self.icon_label)
        status_layout.setStretch(status_layout.indexOf(self.icon_label), 0)

        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.setFixedWidth(300)

        self.progress_bar_w = QWidget()
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar_w.setLayout(progress_layout)
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.stack_widget.addWidget(self.status_bar)
        progress_layout.addWidget(self.progress_bar)
        self.stack_widget.addWidget(self.progress_bar_w)

        status_layout.addWidget(self.stack_widget, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.toolbar.addWidget(status_widget)


        # Add the graphics view
        layout.addWidget(self.graphics_view)

        # Add the controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.jump_backward_button)
        control_layout.addWidget(self.jump_forward_button)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.seek_slider)
        control_layout.setStretch(control_layout.indexOf(self.seek_slider), 4)

        self.frame_layout = QHBoxLayout()
        self.frame_layout.addWidget(self.frame_icon_label)
        self.frame_layout.addWidget(self.frame_label)
        self.frame_widget = QWidget()
        self.frame_widget.setLayout(self.frame_layout)

        control_layout.addWidget(self.frame_widget)
        control_layout.addWidget(self.playback_speed_slider)
        control_layout.setStretch(control_layout.indexOf(self.playback_speed_slider), 1)
        control_layout.addWidget(self.current_speed_label)

        layout.addLayout(control_layout)
        self.setLayout(layout)

        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.graphics_scene.selectionChanged.connect(self.update_toolbar_buttons)

        self.clipboard = QApplication.clipboard()
        self.clipboard.dataChanged.connect(self.update_clipboard_button_state)
        self.update_clipboard_button_state()

        # Initiate the settings window
        self.settings_window = SettingsWindow(self)

    def create_toolbar_button(self, icon):
        button = QPushButton()
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(17, 17))
        button.setFixedSize(30, 30)  # Square button
        return button

    def reset_state(self):
        self.clear_highlight_periods()
        self.clear_current_bounding_boxes()
        self.clear_reference_bounding_boxes()
        self.update_bounding_boxes(0)
        self.current_video_index = None
        self.video_files = []
        self.update_select_buttons_state()

        self.media_player.setSource(QUrl())

    @property
    def download_videos(self):
        return self._download_videos

    @download_videos.setter
    def download_videos(self, value):
        self._download_videos = value

    @property
    def host_base(self):
        return self._host_base

    @host_base.setter
    def host_base(self, value):
        self._host_base = value

    @property
    def use_https(self):
        return self._use_https

    @use_https.setter
    def use_https(self, value):
        self._use_https = value

    @property
    def access_key(self):
        return self._access_key

    @access_key.setter
    def access_key(self, value):
        self._access_key = value

    @property
    def secret_key(self):
        return self._secret_key

    @secret_key.setter
    def secret_key(self, value):
        self._secret_key = value

    @property
    def host_bucket(self):
        return self._host_bucket

    @host_bucket.setter
    def host_bucket(self, value):
        self._host_bucket = value

    @property
    def config(self):
        try:
            self._config = {
                'download_videos': self.download_videos,
                'host_base': self.host_base,
                'use_https': self.use_https,
                'access_key': self.access_key,
                'secret_key': self.secret_key,
                'host_bucket': self.host_bucket,
                'app_data_dir': self.app_data_dir,
                'video_files': self.video_files,
                'current_video_index': self.current_video_index
            }
        except Exception as e:
            self._config = {
                'download_videos': False,
                'host_base': "",
                'use_https': False,
                'access_key': "",
                'secret_key': "",
                'host_bucket': "",
                'app_data_dir': "app_data",
                'video_files': [],
                'current_video_index': None
            }
        return self._config

    @config.setter
    def config(self, value):
        for key in value:
            if key in self.__dict__:
                setattr(self, key, value[key])
            setattr(self, key, value[key])

    def disable_navigation(self):
        super().disable_navigation()
        self.open_video_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.prev_video_button.setEnabled(False)
        self.show_playlist_button.setEnabled(False)
        self.next_video_button.setEnabled(False)

    def enable_navigation(self):
        super().enable_navigation()
        self.open_video_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.prev_video_button.setEnabled(True)
        self.show_playlist_button.setEnabled(True)
        self.next_video_button.setEnabled(True)

    def update_toolbar_buttons(self):
        selected_items = self.graphics_scene.selectedItems()
        self.duplicate_button.setEnabled(bool(selected_items))
        self.delete_button.setEnabled(bool(selected_items))

    def update_clipboard_button_state(self):
        mime_data = self.clipboard.mimeData()
        self.add_clipboard_image_button.setEnabled(mime_data.hasImage())

    def set_status_message(self, message, icon=None, timeout=0):
        if not self._progress_lock:
            self.stack_widget.setCurrentWidget(self.status_bar)
        self.status_bar.showMessage(message, timeout)
        if icon:
            self.icon_label.setPixmap(QPixmap(icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))

        QTimer.singleShot(timeout, self.reset_status_message)

    def reset_status_message(self, message=None, icon=None):
        # Reset the status bar message to the default message
        if not self._progress_lock:
            self.stack_widget.setCurrentWidget(self.status_bar)
        self.status_bar.showMessage(self._default_message if not message else message, 0)
        self.icon_label.setPixmap(
            QPixmap(self._default_icon if not icon else icon).scaled(QSize(17, 17), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def show_progress_bar(self):
        print("Showing progress bar")
        self._progress_lock = True
        self.stack_widget.setCurrentWidget(self.progress_bar_w)

    def current_bar(self):
        return 'status' if self.stack_widget.currentWidget() == self.status_bar else 'progress'

    def open_file(self, file_path):
        super().open_file(file_path)
        self._default_message = f"Video: {os.path.basename(file_path)}"
        self._default_icon = ALT_ICONS['info']
        self.reset_status_message()
        self.update_select_buttons_state()

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.open_file(selected_files[0])

    # Add these new methods in the VideoPlayer class
    def open_folder_dialog(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if folder_dialog.exec():
            selected_folder = folder_dialog.selectedFiles()[0]
            self.load_videos_from_folder(selected_folder)

    def load_videos_from_folder(self, folder_path):
        import os
        video_extensions = ('.mp4', '.avi', '.mkv')
        self.video_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                            f.lower().endswith(video_extensions)]
        if self.video_files:
            self.current_video_index = 0
            self.open_file(self.video_files[self.current_video_index])
            self.show_playlist_button.setEnabled(True)
            self.next_video_button.setEnabled(len(self.video_files) > 1)

    def show_playlist(self):
        self.playlist_window = QWidget()
        self.playlist_window.setWindowTitle("Playlist")
        self.playlist_layout = QVBoxLayout()
        self.playlist_list = QListWidget()
        self.playlist_list.addItems([os.path.basename(video) for video in self.video_files])
        self.playlist_list.itemDoubleClicked.connect(self.select_video_from_playlist)
        self.playlist_layout.addWidget(self.playlist_list)
        self.playlist_window.setLayout(self.playlist_layout)
        self.playlist_window.show()

    def select_video_from_playlist(self, item):
        self.current_video_index = self.playlist_list.row(item)
        self.open_file(self.video_files[self.current_video_index])

    def prev_video(self):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.open_file(self.video_files[self.current_video_index])

    def next_video(self):
        if self.current_video_index < len(self.video_files) - 1:
            self.current_video_index += 1
            self.open_file(self.video_files[self.current_video_index])

    def update_select_buttons_state(self):
        if self.current_video_index is None:
            return

        if self.current_video_index == len(self.video_files) - 1:
            self.next_video_button.setEnabled(False)
        else:
            self.next_video_button.setEnabled(True)

        if self.current_video_index == 0:
            self.prev_video_button.setEnabled(False)
        else:
            self.prev_video_button.setEnabled(True)

    def add_text_item(self, text=None):
        if text:
            ok = True
        else:
            text, ok = QInputDialog.getText(self, 'Input Text', 'Enter text:')

        if ok and text:
            text_item = ResizableTextItem(text, 50, 50)
            unique_id = get_timestamp_hash()
            text_item.set_tag(unique_id)
            self.graphics_scene.addItem(text_item)
            self.text_items[unique_id] = {
                'text': text,
                'coords': (text_item.x(), text_item.y())
            }
            self._current_items[unique_id] = text_item

            text_item.textChanged.connect(self.update_text_item)

    @pyqtSlot(str)
    def update_text_item(self, new_text):
        text_item = self.sender()
        unique_id = text_item.get_tag()
        if unique_id in self.text_items:
            self.text_items[unique_id]['text'] = new_text
            print(f"Text updated: {new_text}")

    def add_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.add_image_to_scene(file_path)

    def add_image_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasImage():
            image = clipboard.image()
            if not image.isNull():
                qimage = clipboard.image()

                # Convert QImage to PIL Image
                pil_image = ImageQt.fromqimage(qimage)
                b, r, g = pil_image.split()
                pil_image = Image.merge("RGB", (r, g, b))

                # Save the image to app_data with a unique hash
                file_path = os.path.join(self.app_data_dir,
                                         f"{get_timestamp_hash()}.png")
                pil_image.save(file_path)

                # Add image to the scene
                self.add_image_to_scene(file_path)

    def add_image_to_scene(self, file_path):
        pixmap = QPixmap(file_path)

        # # Create an instance of GraphicLayer
        item = ResizablePixmapItem(50, 50, pixmap, rect=QRectF(1, 1, int(pixmap.width()), int(pixmap.height())), scene=self.graphics_scene)
        unique_hash = get_filepath_hash(file_path)
        item.set_tag(unique_hash)

        self.graphics_scene.addItem(item)

        # Move image to app_data and rename with a unique hash
        new_file_path = os.path.join(self.app_data_dir, f"{unique_hash}.png")
        if os.path.exists(file_path):
            shutil.copy(file_path, new_file_path)

        # Store item information
        self.image_items[unique_hash] = {
            'path': new_file_path,
            'size': (pixmap.width(), pixmap.height()),
            'coords': (item.x(), item.y())
        }

        self._current_items[unique_hash] = item

        # Make the item movable and resizable
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        item.setAcceptHoverEvents(True)

    def duplicate_selected_item(self):
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            if isinstance(item, ResizablePixmapItem):
                file_path = self.image_items[item.get_tag()]['path']
                self.add_image_to_scene(file_path)
            elif isinstance(item, ResizableTextItem):
                text = self.text_items[item.get_tag()]['text']
                self.add_text_item(text)

    def delete_selected_item(self):
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            if isinstance(item, ResizablePixmapItem):
                unique_hash = item.get_tag()
                if unique_hash in self.image_items:
                    del self.image_items[unique_hash]
                    self.graphics_scene.removeItem(item)
            elif isinstance(item, ResizableTextItem):
                unique_id = item.get_tag()
                if unique_id in self.text_items:
                    del self.text_items[unique_id]
                    self.graphics_scene.removeItem(item)

    def save_scene(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Scene Configuration", "", "Binary Files (*.bin)")

        if file_path:

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            scene_data = {
                'images': {},
                'texts': {item_id: {'text': data['text'], 'coords': data['coords']} for item_id, data in
                          self.text_items.items()}
            }

            for unique_hash, item_data in self.image_items.items():
                if 'path' in item_data.keys():
                    with open(item_data['path'], 'rb') as img_file:
                        scene_data['images'][unique_hash] = {
                            'image': img_file.read(),
                            'size': item_data['size'],
                            'coords': item_data['coords']
                        }
                else:
                    pixmap = self._current_items[unique_hash].graphic
                    img_file = pixmap_to_bytes(pixmap)
                    scene_data['images'][unique_hash] = {
                        'image': img_file,
                        'size': item_data['size'],
                        'coords': item_data['coords']
                    }

            with open(file_path, 'wb') as bin_file:
                pickle.dump(scene_data, bin_file)

    def load_scene(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Scene Configuration", "", "Binary Files (*.bin)")
        if file_path:
            with open(file_path, 'rb') as bin_file:
                scene_data = pickle.load(bin_file)
                self.image_items = scene_data.get('images', {})
                self.text_items = scene_data.get('texts', {})
                self.recreate_scene()

    def recreate_scene(self):
        try:
            # Remove only the items in self._current_items, keeping the video_item
            for item in self._current_items.values():
                self.graphics_scene.removeItem(item)

            self._current_items.clear()  # Clear the dictionary

            for unique_hash, item_data in self.image_items.items():
                image_data = item_data['image']
                image_path = os.path.join(self.app_data_dir, f"{unique_hash}.png")
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)

                pixmap = QPixmap(image_path)
                print(item_data['size'])
                item = ResizablePixmapItem(0, 0, pixmap, rect=QRectF(1, 1, item_data['size'][0], item_data['size'][1]),
                                           scene=self.graphics_scene)
                resize_pixmap_item(item, change=QGraphicsItem.GraphicsItemChange.ItemPositionChange, value=QPointF(*item_data['size']))
                item.set_tag(unique_hash)
                item.setPos(*item_data['coords'])
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
                item.setAcceptHoverEvents(True)
                self.graphics_scene.addItem(item)
                self._current_items[unique_hash] = item

            # Recreate text items
            for item_id, item_data in self.text_items.items():
                text_item = ResizableTextItem(item_data['text'], item_data['coords'][0], item_data['coords'][1])
                text_item.set_tag(item_id)
                text_item.textChanged.connect(self.update_text_item)
                self.graphics_scene.addItem(text_item)
                self._current_items[item_id] = text_item
        except Exception as e:
            print(e)

    def open_settings_dialog(self):
        self.settings_window.show()

    def toggle_bboxes(self):
        self.show_bboxes = not self.show_bboxes
        icon = IMGS['insect-box'] if self.show_bboxes else ALT_ICONS['minus']
        self.toggle_bboxes_button.setIcon(QIcon(icon))
        if not self.show_bboxes:
            self.clear_current_bounding_boxes()

    def toggle_reference_bboxes(self):
        self.show_reference_bboxes = not self.show_reference_bboxes
        icon = ALT_ICONS['flower-box'] if self.show_reference_bboxes else ALT_ICONS['minus']
        self.toggle_reference_bboxes_button.setIcon(QIcon(icon))
        if not self.show_reference_bboxes:
            self.clear_reference_bounding_boxes()

    def toggle_highlights(self):
        self.show_highlights = not self.show_highlights
        icon = ALT_ICONS['edit-3'] if self.show_highlights else ALT_ICONS['edit-2']
        self.toggle_highlights_button.setIcon(QIcon(icon))
        if not self.show_highlights:
            self.clear_highlight_periods()
        else:
            self.set_highlight_periods(self._periods, override=True)

    def set_highlight_periods(self, periods, override=False):
        self._periods = periods
        if len(periods) > 800 and self.show_highlights and not (override and len(periods) < 1200):
            self.toggle_highlights()
            self.set_status_message(f"{len(periods)} highlights temporarily disabled.", icon=ALT_ICONS['alert-circle'], timeout=3000)
        if self.show_highlights:
            super().set_highlight_periods(periods)
            self.set_status_message(f"Updated visit highlight", icon=ALT_ICONS['edit-3'], timeout=3000)

    def set_bounding_boxes(self, bounding_boxes, persistence=0):
        super().set_bounding_boxes(bounding_boxes, persistence)
        self.set_status_message(f"Updated bounding boxes", icon=ALT_ICONS['maximize'], timeout=3000)

    def set_reference_bounding_boxes(self, bounding_boxes, persistence=0):
        super().set_reference_bounding_boxes(bounding_boxes, persistence)
        self.set_status_message(f"Updated reference bounding boxes", icon=ALT_ICONS['maximize'], timeout=3000)

    def clear_current_bounding_boxes(self):
        super().clear_current_bounding_boxes()
        self.set_status_message(f"Cleared bounding boxes", icon=ALT_ICONS['trash'], timeout=3000)

    def clear_reference_bounding_boxes(self):
        super().clear_reference_bounding_boxes()
        self.set_status_message(f"Cleared reference bounding boxes", icon=ALT_ICONS['trash'], timeout=3000)

    def update_bounding_boxes(self, frame_number, **kwargs):

        # Clear current bounding boxes
        self.clear_bounding_boxes(self.current_bboxes)
        self.clear_bounding_boxes(self.current_reference_bboxes)

        # Add bounding boxes
        if self.show_bboxes:
            self.current_bboxes = self.add_bounding_boxes(self.bounding_boxes, frame_number,
                                                          self.bbox_persistence,
                                                          kwargs.get('color', QColor(255, 0, 0)),
                                                          kwargs.get('line', 1),
                                                          kwargs.get('text_color', QColor(255, 255, 255)))

        # Add reference bounding boxes
        if self.show_reference_bboxes:
            self.current_reference_bboxes = self.add_bounding_boxes(self.reference_bounding_boxes, frame_number,
                                                                    self.reference_bbox_persistence,
                                                                    QColor(0, 255, 255), 1,
                                                                    QColor(255, 255, 255))

    def mousePressEvent(self, event):
        ScreenshotMixin.mousePressEvent(self, event)
        VideoPlayer.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        ScreenshotMixin.mouseReleaseEvent(self, event)
        VideoPlayer.mouseReleaseEvent(self, event)


def convert_json_to_pickle(json_file_path, pickle_file_path):
    with open(json_file_path, 'r') as json_file:
        scene_data = json.load(json_file)

    converted_data = {
        'images': {},
        'texts': scene_data.get('texts', {})
    }

    for unique_hash, item_data in scene_data.get('images', {}).items():
        with open(item_data['path'], 'rb') as img_file:
            image_data = img_file.read()

        converted_data['images'][unique_hash] = {
            'image': image_data,
            'size': item_data['size'],
            'coords': item_data['coords']
        }

    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(converted_data, pickle_file)

    print(f"Conversion complete: {json_file_path} to {pickle_file_path}")


if __name__ == "__main__":

    @pyqtSlot(str)
    def changed_video(video_path):
        print(f"Video changed: {video_path}")

    #convert_json_to_pickle(r'D:\Dílna\Kutění\Python\DetectFlow\detectflow\app\app_data\tutorial.json', r'D:\Dílna\Kutění\Python\DetectFlow\detectflow\app\app_data\tutorial.bin')
    from random import randint
    from detectflow.config import TESTS_DIR
    app = QApplication(sys.argv)
    player = InteractiveVideoPlayer()
    player.video_changed.connect(changed_video)

    video_path = os.path.join(TESTS_DIR, "video", "resources", "GR2_L1_TolUmb3_20220524_07_44.mp4")
    video = Video(video_path)
    w = video.frame_width
    h = video.frame_height
    tf = video.total_frames

    # Example bounding boxes data
    bounding_boxes = {
        0: [(0, 0, 300, 300, 0.65898)],
        250: [(328, 184, 656, 368, 0.65898)],
        500: [(0, 0, 656, 368, 0.65898)],
    }

    bounding_boxes = {}
    for i in range(0, 20000, 15):
        x1 = randint(0, w // 2)
        y1 = randint(0, h // 2)
        bounding_boxes[i] = [(x1, y1, randint(x1, w), randint(y1, h), randint(0, 100) / 100.0)]

    ref_bounding_boxes = {
        0: [(20, 30, 80, 90)],
        250: [(36, 58, 65, 69)],
        500: [(0, 0, 65, 166)],
    }

    ref_bounding_boxes = {}
    for i in range(0, 20000, 1000):
        x1 = randint(w // 4, w // 2)
        y1 = randint(h // 4, h // 2)
        ref_bounding_boxes[i] = [(x1, y1, randint(x1, w // 1.5), randint(y1, h // 1.5))]

    player.set_bounding_boxes(bounding_boxes, persistence=15)
    player.set_reference_bounding_boxes(ref_bounding_boxes, persistence=0)

    from detectflow.config import TESTS_DIR
    player.open_file(video_path)
    # Example periods to highlight
    highlight_periods = [(0, 300000), (150000, 400000), (200000, 250000), (290000, 350000), (310000, 450000), (500000, 600000)]
    player.set_highlight_periods(highlight_periods)
    player.show()
    sys.exit(app.exec())


