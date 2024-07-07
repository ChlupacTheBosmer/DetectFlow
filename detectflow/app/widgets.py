from PyQt6.QtWidgets import QApplication, QTreeView, QVBoxLayout, QWidget, QStackedLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QTime, QEvent, QCoreApplication
from PyQt6.QtGui import QPainter, QTransform
from PyQt6.QtSvgWidgets import QSvgWidget
from detectflow.resources import ICONS, IMGS
import sys
from PyQt6.QtWidgets import QApplication, QTreeView, QVBoxLayout, QWidget, QStackedLayout, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty, QTimer
from PyQt6.QtGui import QPainter, QPixmap
import sys
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QFrame, QStatusBar, QTextEdit, QLineEdit, QToolBar, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout, QSizePolicy, QProgressBar, QSpacerItem
from PyQt6.QtCore import QSize, Qt, QEvent
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class IconTextButton(QWidget):
    def __init__(self, text, icon_path, icon_size, parent=None):
        super().__init__(parent)

        # Create the button icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QIcon(icon_path).pixmap(QSize(*icon_size)))
        self.icon_label.setFixedSize(*icon_size)

        # Create the button text
        self.text_label = QLabel(text)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create the layout and add icon and text
        layout = QHBoxLayout()
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label, 1)  # Stretch factor 1 to make text fill the remaining space
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # Adjust the spacing between icon and text as needed

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def mousePressEvent(self, event):
        self.parent().mousePressEvent(event)


class Spinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self._angle = 0

        # Load custom icon
        self.pixmap = QPixmap(ICONS['loading'])  # Ensure you have a loading.png in the same directory

        self.animation = QPropertyAnimation(self, b"angle", self)
        self.animation.setStartValue(0)
        self.animation.setEndValue(360)
        self.animation.setLoopCount(-1)
        self.animation.setDuration(2000)
        self.animation.start()

    @pyqtProperty(int)
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self.update()

    def paintEvent(self, ev=None):
        painter = QPainter(self)
        painter.translate(self.width() // 2, self.height() // 2)
        painter.rotate(self._angle)
        painter.translate(-self.width() // 2, -self.height() // 2)
        painter.drawPixmap((self.width() - self.pixmap.width()) // 2, (self.height() - self.pixmap.height()) // 2, self.pixmap)


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.spinner = Spinner(self)
        layout.addWidget(self.spinner)

    def resizeEvent(self, event):
        self.setGeometry(0, 0, self.parent().width(), self.parent().height())


class TopTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        #self.setFixedHeight(40)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Highlight)
        self.initial_pos = None
        self.setStyleSheet("background-color: #282c2c; border: 0px solid black;")

        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(1, 1, 1, 1)
        title_bar_layout.setSpacing(2)

        # self.icon = QLabel(self)
        # self.icon.setPixmap(QPixmap(IMGS['icon']).scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        # self.icon.setStyleSheet(
        #     """
        #     background-color: #282c2c;
        #     border: 0px solid black;
        #     border-radius: 0px;
        #     padding: 10px;
        #     """
        # )
        # title_bar_layout.addWidget(self.icon, alignment=Qt.AlignmentFlag.AlignLeft)

        # # Add a spacer item to push other widgets to the right
        # spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # title_bar_layout.addSpacerItem(spacer)

        self.title = QLabel(f"NEco", self)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title)
        self.title.setStyleSheet(
            """
            background-color: #282c2c;
            border: 0px solid black;
            border-radius: 0px;
            """
        )
        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignLeft)

        # Min button
        self.min_button = QToolButton(self)
        min_icon = QIcon()
        min_icon.addFile(IMGS['icons8-minimize-64'])
        self.min_button.setIcon(min_icon)
        self.min_button.setIconSize(QSize(20, 20))
        self.min_button.clicked.connect(self.window().showMinimized)

        # Max button
        self.max_button = QToolButton(self)
        max_icon = QIcon()
        max_icon.addFile(IMGS['icons8-maximize-button-64'])
        self.max_button.setIcon(max_icon)
        self.max_button.setIconSize(QSize(20, 20))
        self.max_button.clicked.connect(self.window().showMaximized)

        # Close button
        self.close_button = QToolButton(self)
        close_icon = QIcon()
        close_icon.addFile(IMGS['icons8-close-64'])  # Close has only a single state.
        self.close_button.setIcon(close_icon)
        self.close_button.setIconSize(QSize(20, 20))
        self.close_button.setProperty('class', 'close_button')
        self.close_button.clicked.connect(self.window().close)

        # Normal button
        self.normal_button = QToolButton(self)
        normal_icon = QIcon()
        normal_icon.addFile(IMGS['icons8-restore-down-64'])
        self.normal_button.setIcon(normal_icon)
        self.normal_button.setIconSize(QSize(20, 20))
        self.normal_button.clicked.connect(self.window().showNormal)
        self.normal_button.setVisible(False)

        # Add buttons
        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(40, 40))
            button.setStyleSheet(
                """QToolButton { border: 0px solid white;
                                 border-radius: 0px;
                                 background-color: #282c2c;
                                 padding: 0px;
                                 margin: 0px;
                                }
                    
                    QToolButton:hover {
                        background-color: #3a3a3a;
                    }
                    QToolButton.close_button:hover {
                        background-color: #d9534f;
                    }
                """
            )
            title_bar_layout.addWidget(button)

        self.setStyleSheet(
            """
                background-color: #282c2c;
                border: 1px solid black;
            
            """
        )

    def window_state_changed(self, state):
        if state == Qt.WindowState.WindowMaximized:
            self.normal_button.setVisible(True)
            self.max_button.setVisible(False)
        else:
            self.normal_button.setVisible(False)
            self.max_button.setVisible(True)

class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.initial_pos = None
        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setSpacing(0)

        frame_layout = QHBoxLayout(self)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame = QFrame(self)
        frame.setLayout(frame_layout)

        title_widget = QWidget(self)
        title_layout = QHBoxLayout(self)
        title_widget.setLayout(title_layout)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)

        self.icon = QLabel(self)
        self.icon.setPixmap(QPixmap(IMGS['icon']).scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon.setStyleSheet(
            """
            padding: 10px;
            """
        )
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self.icon, alignment=Qt.AlignmentFlag.AlignLeft)

        self.title = QLabel(f"DetectFlow Companion", self)
        # self.title.setStyleSheet(
        #     """
        #     background-color: #282c2c;
        #        border: 0px solid black;
        #        border-radius: 0px;
        #        margin: 0px;
        #     """
        # )
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if title := parent.windowTitle():
            self.title.setText(title)
        title_layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignLeft)

        frame_layout.addWidget(title_widget, alignment=Qt.AlignmentFlag.AlignLeft)

        # Min button
        self.min_button = QToolButton(self)
        min_icon = QIcon()
        min_icon.addFile(IMGS['icons8-minimize-64'])
        self.min_button.setIcon(min_icon)
        self.min_button.setIconSize(QSize(20, 20))
        self.min_button.clicked.connect(self.window().showMinimized)

        # Max button
        self.max_button = QToolButton(self)
        max_icon = QIcon()
        max_icon.addFile(IMGS['icons8-maximize-button-64'])
        self.max_button.setIcon(max_icon)
        self.max_button.setIconSize(QSize(20, 20))
        self.max_button.clicked.connect(self.window().showMaximized)

        # Close button
        self.close_button = QToolButton(self)
        close_icon = QIcon()
        close_icon.addFile(IMGS['icons8-close-64'])  # Close has only a single state.
        self.close_button.setIcon(close_icon)
        self.close_button.setIconSize(QSize(20, 20))
        self.close_button.setProperty('class', 'close_button')
        self.close_button.clicked.connect(self.window().close)

        # Normal button
        self.normal_button = QToolButton(self)
        normal_icon = QIcon()
        normal_icon.addFile(IMGS['icons8-restore-down-64'])
        self.normal_button.setIcon(normal_icon)
        self.normal_button.setIconSize(QSize(20, 20))
        self.normal_button.clicked.connect(self.window().showNormal)
        self.normal_button.setVisible(False)

        # Add buttons
        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(40, 40))
            button.setStyleSheet(
                """QToolButton { border: 0px solid white;
                                 border-radius: 0px;
                                 background-color: #282c2c;
                                 padding: 0px;
                                 margin: 0px;
                                }

                    QToolButton:hover {
                        background-color: #3a3a3a;
                    }
                    QToolButton.close_button:hover {
                        background-color: #d9534f;
                    }
                """
            )
            frame_layout.addWidget(button)

        title_bar_layout.addWidget(frame)
        frame.setStyleSheet(
            """
            background-color: #282c2c;
               border: 0px solid black;
               border-radius: 0px;
               padding: 0px;
               margin: 0px;
            """
        )

    def window_state_changed(self, state):
        if state == Qt.WindowState.WindowMaximized:
            self.normal_button.setVisible(True)
            self.max_button.setVisible(False)
        else:
            self.normal_button.setVisible(False)
            self.max_button.setVisible(True)

class BottomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        #self.setBackgroundRole(QPalette.ColorRole.Highlight)
        self.initial_pos = None
        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setSpacing(0)

        frame_layout = QHBoxLayout(self)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame = QFrame(self)
        frame.setLayout(frame_layout)

        ssh_widget = QWidget(self)
        ssh_layout = QHBoxLayout(self)

        # # Add link icon aligned to the left
        # self.link = QLabel(f"", self)
        # self.link.setPixmap(QPixmap(ICONS['link-alt']).scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio,
        #                                                       Qt.TransformationMode.SmoothTransformation))
        # self.link.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # self.link.setStyleSheet(
        #     """
        #        background-color: #282c2c;
        #        border: 0px solid black;
        #        border-radius: 0px;
        #        margin: 0px;
        #        padding-left: 10px;
        #     """
        # )
        # ssh_layout.addWidget(self.link, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Status bar
        self.status_bar = QStatusBar()
        parent.setStatusBar(self.status_bar)
        self.status_bar.setSizeGripEnabled(False)
        self.ssh_status_label = QLabel()
        self.ssh_status_label.setPixmap(QIcon(ICONS['broken-link']).pixmap(20, 20))
        self.ssh_status_text = QLabel(f"SSH Disconnected")
        self.status_bar.addPermanentWidget(self.ssh_status_label)
        self.status_bar.addPermanentWidget(self.ssh_status_text)
        ssh_layout.addWidget(self.status_bar, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # self.title = QPushButton(f"SSH Status: Connected", self)
        # self.title.setEnabled(True)
        # self.title.setFlat(True)
        # self.title.setFixedHeight(20)
        # self.title.setProperty('class', 'highlight_info')
        # # self.title.setStyleSheet(
        # #     """
        # #        color: #fcd444;
        # #        background-color: #282c2c;
        # #        border: 0px solid black;
        # #        border-radius: 0px;
        # #        margin: 0px;
        # #     """
        # # )
        # #self.title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # # if title := parent.windowTitle():
        # #     self.title.setText(title)
        # ssh_layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        ssh_widget.setLayout(ssh_layout)
        ssh_widget.setStyleSheet(
            """
            background-color: #282c2c;
               border: 0px solid black;
               border-radius: 0px;
               margin: 0px;
            """
        )
        frame_layout.addWidget(ssh_widget, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        progress_widget = QWidget(self)
        progress_layout = QHBoxLayout(self)

        # Add link icon aligned to the right
        self.progress = QLabel(f"Submitting PBS jobs and logging into database", self)
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.progress.setStyleSheet(
            """
            QLabel{
               color: #ffd740;
               padding-right: 10px;
               }
            """
        )
        progress_layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        # Add progress bar aligned to the right.
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.progress_bar.setFixedWidth(self.width() // 4 - 125)
        self.progress_bar.setValue(75)
        self.progress_bar.setStyleSheet(
                """
                background-color: #31363b;
                """
            )
        progress_layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        # Add link icon aligned to the right
        self.message = QLabel(f"", self)
        self.message.setPixmap(QPixmap(ICONS['info-rotated-corners']).scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio,
                                                                             Qt.TransformationMode.SmoothTransformation))
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.message.setStyleSheet(
            """
               background-color: #282c2c;
               border: 0px solid black;
               border-radius: 0px;
               margin: 0px;
               padding-right: 0px;
            """
        )
        progress_layout.addWidget(self.message, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        progress_widget.setLayout(progress_layout)
        # progress_widget.setStyleSheet(
        #     """
        #     background-color: #282c2c;
        #        border: 0px solid black;
        #        border-radius: 0px;
        #        margin: 0px;
        #     """
        # )

        frame_layout.addWidget(progress_widget, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        title_bar_layout.addWidget(frame)
        frame.setStyleSheet(
            """
            background-color: #282c2c;
               border: 0px solid black;
               border-radius: 0px;
               padding: 0px;
               margin: 0px;
            """
        )

    def resizeEvent(self, event):
        # Resize the progress bar to half of the width of the widget
        new_width = self.width() // 4 - 125
        self.progress_bar.setFixedWidth(new_width)
        super().resizeEvent(event)
