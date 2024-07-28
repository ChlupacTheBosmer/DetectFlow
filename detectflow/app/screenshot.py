import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPainter, QPen, QPixmap
from PyQt6.QtCore import Qt, QRect, QPoint
from io import BytesIO
import win32clipboard
from PIL import Image
from detectflow.resources import ALT_ICONS
from detectflow.video.video_data import Video


class ScreenshotOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(None)  # Make it a top-level widget
        self.setParent(parent)
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_drawing = False
        self.should_draw_rect = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def start_drawing(self, start_point):
        self.start_point = start_point
        self.is_drawing = True
        self.should_draw_rect = True
        self.update()

    def update_drawing(self, end_point):
        self.end_point = end_point
        self.update()

    def stop_drawing(self):
        self.is_drawing = False
        self.should_draw_rect = False
        self.update()

    def paintEvent(self, event):
        if self.should_draw_rect and self.is_drawing:
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point))


class ScreenshotMixin:
    def __init__(self):
        self.overlay = ScreenshotOverlay(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.hide()
        self.screenshot_enabled = False

    def enable_screenshot_mode(self):
        self.setCursor(Qt.CursorShape.CrossCursor)
        central_widget = self.centralWidget() if hasattr(self, 'centralWidget') else self
        central_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.grabMouse()
        self.overlay.setGeometry(self.geometry())
        self.overlay.show()
        self.screenshot_enabled = True
        if hasattr(self, 'set_status_message'):
            self.set_status_message(f"Select area to capture it into clipboard", icon=ALT_ICONS['crop'], timeout=0)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.screenshot_enabled:
            self.overlay.start_drawing(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self.overlay.is_drawing and self.screenshot_enabled:
            self.overlay.update_drawing(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.screenshot_enabled:
            self.overlay.update_drawing(event.position().toPoint())
            self.capture_screenshot()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            central_widget = self.centralWidget() if hasattr(self, 'centralWidget') else self
            central_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self.releaseMouse()
            self.overlay.stop_drawing()
            self.overlay.hide()
            if hasattr(self, 'set_status_message'):
                if hasattr(self, 'video') and isinstance(self.video, Video):
                    self.set_status_message(f"Video: {self.video.video_id}", icon=ALT_ICONS['info'], timeout=0)

    def capture_screenshot(self):
        rect = QRect(self.overlay.start_point + QPoint(2, 2), self.overlay.end_point - QPoint(2, 2)).normalized()
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        screenshot = pixmap.copy(rect)
        try:
            image = screenshot.toImage()
            pil_img = Image.fromqimage(image)

            output = BytesIO()
            pil_img.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]
            output.close()

            self.copy_image_to_clipboard(win32clipboard.CF_DIB, data)
        except Exception as e:
            print(f"Error capturing: {e}")

    def copy_image_to_clipboard(self, clip_type, data):
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(clip_type, data)
            win32clipboard.CloseClipboard()
        except Exception as e:
            print(f"Error processing: {e}")
            return