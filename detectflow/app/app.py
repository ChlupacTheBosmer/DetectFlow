import sys
import PyQt6
import os
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
from PyQt6.QtWidgets import (QTreeView, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QPlainTextEdit, QTimeEdit, QSpinBox, QCheckBox, QStackedLayout, QToolBar)
from PyQt6.QtCore import Qt, QSize, QThreadPool, QEvent
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from detectflow.resources import ICONS
from functools import partial
from detectflow.handlers.ssh_handler import SSHHandler
from detectflow import __version__, __author__
from detectflow.app.threads import PopulateS3TreeTask, PopulateSSHTreeTask, SSHWorker
from detectflow.app.widgets import Spinner, LoadingOverlay, IconTextButton, TopTitleBar, BottomTitleBar, CustomTitleBar

icon_size = (20, 20)
images_dir = os.path.join(DETECTFLOW_DIR, "resources", "img")

def clear_item(item):
    if hasattr(item, "layout"):
        if callable(item.layout):
            layout = item.layout()
    else:
        layout = None

    if hasattr(item, "widget"):
        if callable(item.widget):
            widget = item.widget()
    else:
        widget = None

    if widget:
        widget.setParent(None)
    elif layout:
        for i in reversed(range(layout.count())):
            clear_item(layout.itemAt(i))


class SchedulerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # self.setWindowTitle("Scheduler App")
        # self.setWindowIcon(QIcon(os.path.join(images_dir, "icon.png")))
        self.setGeometry(100, 100, 1200, 800)

        # Store all widgets in a dictionary for easy access
        self.widgets = {}

        #self Custom title bar
        self.title_bar = CustomTitleBar(self)
        # self.title_bar.setStyleSheet(
        #     """
        #     QWidget.custom_bar {
        #         background-color: #282c2c;
        #         color: white;
        #         border: 5px solid black;
        #         border-radius: 0px;
        #         }
        #     """
        # )
        #

        self.ultra_widget = QWidget()
        self.ultra_layout = QVBoxLayout()
        self.ultra_layout.setContentsMargins(0, 0, 0, 0)
        self.ultra_layout.addWidget(self.title_bar)

        self.ultra_widget.setLayout(self.ultra_layout)
        self.setCentralWidget(self.ultra_widget)

        self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.ultra_layout.addWidget(self.central_widget)

        # Create the menu
        self.create_menu()

        # Create the content area
        self.content_area = QWidget()
        self.content_layout = QHBoxLayout(self.content_area)
        #self.main_layout.addWidget(self.content_area)
        self.main_layout.addWidget(self.content_area)

        # Create the dashboard view
        self.create_dashboard_view()

        #Add bottom TitleBar
        self.bottom_title_bar = BottomTitleBar(self)
        self.ultra_layout.addWidget(self.bottom_title_bar, alignment=Qt.AlignmentFlag.AlignBottom)

        self.ssh = None

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.window_state_changed(self.windowState())
        super().changeEvent(event)
        event.accept()

    def window_state_changed(self, state):
        self.title_bar.normal_button.setVisible(state == Qt.WindowState.WindowMaximized)
        self.title_bar.max_button.setVisible(state != Qt.WindowState.WindowMaximized)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.initial_pos = event.position().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.initial_pos is not None:
            delta = event.position().toPoint() - self.initial_pos
            self.window().move(
                self.window().x() + delta.x(),
                self.window().y() + delta.y(),
            )
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.initial_pos = None
        super().mouseReleaseEvent(event)
        event.accept()

    def create_menu(self):
        try:
            self.menu_layout = QVBoxLayout()
            self.menu_frame = QFrame()
            self.menu_frame.setLayout(self.menu_layout)
            self.menu_frame.setFixedWidth(200)
            self.main_layout.addWidget(self.menu_frame)

            # Menu Buttons
            buttons_info = [
                ("Dashboard", self.create_dashboard_view, "dashboard.svg"),
                ("Authentication", self.create_authentication_view, "login.svg"),
                ("Submit Jobs", self.create_submit_jobs_view, "arrow-up-square.svg"),
                ("Monitor Jobs", self.create_monitor_jobs_view, "arrow-down-square.svg"),
                ("Launchpad", self.create_launchpad_view, "bookmark.svg"),
                ("Tutorial", self.create_tutorial_view, "question-square.svg"),
                ("Settings", self.create_settings_view, "settings.svg")
            ]

            for text, method, icon in buttons_info:
                # Create the custom button widget
                button_widget = IconTextButton("", os.path.join(images_dir, icon), icon_size)

                # Create the button and set the custom widget as its central widget
                button = QPushButton(text)
                button.setLayout(QHBoxLayout())
                button.layout().addWidget(button_widget)
                button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                button.clicked.connect(method)

                # Add the button to the layout
                self.menu_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignTop)
        except Exception as e:
            print(f"Error in create_menu: {e}")

        #AfterInput adding buttons ad vertical spacer so the buttons stack up
        self.menu_layout.addStretch()

        # # Create a widget to hold the top two icons next to each other
        self.top_icon_widget = QWidget()
        self.top_icon_layout = QHBoxLayout()
        self.top_icon_widget.setLayout(self.top_icon_layout)
        self.menu_layout.addWidget(self.top_icon_widget)

        # Icon
        self.widgets['icon_label'] = QLabel()
        self.widgets['icon_label'].setPixmap(
            QPixmap(os.path.join(images_dir, "logo.png")).scaled(182, 72, Qt.AspectRatioMode.KeepAspectRatio))
        self.widgets['icon_label'].setContentsMargins(0, 0, 0, 0)
        self.menu_layout.addWidget(self.widgets['icon_label'],
                                       alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        #self Add label below the icon printing the name of the package, the version and the author.
        self.widgets['version_label'] = QLabel(f"DETECTFLOW\nVersion {__version__}\n{__author__}\nCharles University")
        self.widgets['version_label'].setContentsMargins(0, 0, 0, 0)
        self.widgets['version_label'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_layout.addWidget(self.widgets['version_label'], alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

    def create_dashboard_view(self):
        try:
            self.clear_view()

            # Main Dashboard layout
            dashboard_layout = QGridLayout()
            dashboard_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            self.content_layout.addLayout(dashboard_layout)
            self.widgets['dashboard_layout'] = dashboard_layout

            # Add "+" button
            add_config_btn = QPushButton()
            # Add icon with set size to the button
            add_config_btn.setIcon(QIcon(os.path.join(images_dir, 'plus.svg')))
            add_config_btn.setIconSize(QSize(50, 50))
            add_config_btn.setFixedSize(300, 300)
            add_config_btn.clicked.connect(self.add_new_config)
            dashboard_layout.addWidget(add_config_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
            self.widgets['add_config_btn'] = add_config_btn

            # Placeholder for other buttons (representing saved configs)
            for i in range(1, 4):
                config_btn = QPushButton(f"Config {i}")
                config_btn.setFixedSize(300, 300)
                dashboard_layout.addWidget(config_btn, 0, i)
                self.widgets[f'config_btn_{i}'] = config_btn
        except Exception as e:
            print(f"Error in create_dashboard_view: {e}")

    def add_new_config(self):
        try:
            # Method to handle adding a new configuration
            print("Add new configuration")
        except Exception as e:
            print(f"Error in add_new_config: {e}")

    def clear_view(self):
        # if 's3_tree_view' in self.widgets:
        #     print(self.widgets['s3_tree_view'].model().gather_selected_data())
        try:
            print("Clearing View")
            # Clear the current view from the content area
            clear_item(self.content_layout)
        except Exception as e:
            print(f"Error in clear_view: {e}")

    # Placeholder methods for other views
    def create_authentication_view(self):
        try:
            self.start_ssh_thread("USER", "PASSWORD", "skirit.metacentrum.cz")

            self.clear_view()

            # Main layout for the authentication view
            auth_layout = QGridLayout()
            self.content_layout.addLayout(auth_layout)
            self.widgets['auth_layout'] = auth_layout

            # S3 Authentication details
            s3_layout = QVBoxLayout()
            self.widgets['s3_layout'] = s3_layout
            self.widgets['s3_layout'].setSpacing(20)
            self.s3_frame = QFrame()
            self.s3_frame.setLayout(self.widgets['s3_layout'])
            self.widgets['auth_layout'].addWidget(self.s3_frame, 0, 0)

            s3_label = QLabel("S3 Storage Details")
            s3_label.setProperty('class', 'detectflow_header')
            s3_layout.addWidget(s3_label, alignment=Qt.AlignmentFlag.AlignHCenter)

            # Create a QLabel and set HTML content for justified text
            s3_hint_label = QLabel()
            s3_hint_label.setTextFormat(Qt.TextFormat.RichText)
            s3_hint_label.setWordWrap(True)
            s3_hint_label.setText("""
                        <html>
                            <head>
                                <style>
                                    body {
                                        font-family: Roboto, Roboto;
                                        font-size: 14px;
                                    }
                                    .instruction {
                                        margin-bottom: 15px;
                                    }
                                    .details {
                                        margin-left: 20px;
                                    }
                                </style>
                            </head>
                            <body>
                                <div class="instruction">
                                    <p>Please fill in the following details to connect to your S3 storage:</p>
                                    <div class="details">
                                        <p><strong>Host base</strong>: The base URL for your S3 storage.</p>
                                        <p><strong>Use HTTPS</strong>: Set to <code>True</code> for HTTPS connection, <code>False</code> for HTTP.</p>
                                        <p><strong>Access Key</strong>: Your S3 access key.</p>
                                        <p><strong>Secret Key</strong>: Your S3 secret key.</p>
                                        <p><strong>Host Bucket</strong>: The bucket hostname format (e.g., <code>{bucket}.{host_base}</code>).</p>
                                    </div>
                                </div>
                                <div class="instruction">
                                    <p>Alternatively, you can upload a <code>.s3cfg</code> file directly to the app, and the details will be filled out automatically.</p>
                                </div>
                            </body>
                            </html>
                    """)

            # Add the QTextEdit to the layout
            s3_layout.addWidget(s3_hint_label)

            self.widgets['s3_use_https'] = QCheckBox("Use HTTPS")
            s3_layout.addWidget(self.widgets['s3_use_https'])

            self.widgets['s3_host_base'] = QLineEdit()
            self.widgets['s3_host_base'].setPlaceholderText("Host Base")
            s3_layout.addWidget(self.widgets['s3_host_base'])

            self.widgets['s3_access_key'] = QLineEdit()
            self.widgets['s3_access_key'].setPlaceholderText("Access Key")
            s3_layout.addWidget(self.widgets['s3_access_key'])

            self.widgets['s3_secret_key'] = QLineEdit()
            self.widgets['s3_secret_key'].setPlaceholderText("Secret Key")
            self.widgets['s3_secret_key'].setEchoMode(QLineEdit.EchoMode.Password)
            s3_layout.addWidget(self.widgets['s3_secret_key'])

            self.widgets['s3_host_bucket'] = QLineEdit()
            self.widgets['s3_host_bucket'].setPlaceholderText("Host Bucket")
            s3_layout.addWidget(self.widgets['s3_host_bucket'])

            # File uploader for S3
            self.widgets['s3_file_uploader'] = QLabel("Drag and drop a file here")
            self.widgets['s3_file_uploader'].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.widgets['s3_file_uploader'].setStyleSheet("border: 2px dashed #aaa; padding: 20px;")
            self.widgets['s3_file_uploader'].setFixedHeight(100)
            s3_layout.addWidget(self.widgets['s3_file_uploader'])

            self.widgets['s3_file_button'] = QPushButton("Select File")
            self.widgets['s3_file_button'].clicked.connect(self.select_s3_file)
            s3_layout.addWidget(self.widgets['s3_file_button'])

            # Add a vertical spacer to push the content to the top
            s3_layout.addStretch()

            # SSH Authentication details
            ssh_layout = QVBoxLayout()
            self.widgets['ssh_layout'] = ssh_layout
            self.widgets['ssh_layout'].setSpacing(20)
            self.ssh_frame = QFrame()
            self.ssh_frame.setLayout(self.widgets['ssh_layout'])
            self.widgets['auth_layout'].addWidget(self.ssh_frame, 0, 1)

            ssh_label = QLabel("SSH Details")
            ssh_label.setProperty('class', 'detectflow_header')
            ssh_layout.addWidget(ssh_label, alignment=Qt.AlignmentFlag.AlignHCenter)

            # Create a QLabel and set HTML content for justified text
            ssh_hint_label = QLabel()
            ssh_hint_label.setTextFormat(Qt.TextFormat.RichText)
            ssh_hint_label.setWordWrap(True)
            ssh_hint_label.setText("""
                                    <html>
                                    <head>
                                        <style>
                                            body {
                                                font-family: Roboto, Roboto;
                                                font-size: 14px;
                                            }
                                            .instruction {
                                                margin-bottom: 15px;
                                            }
                                            .details {
                                                margin-left: 20px;
                                            }
                                        </style>
                                    </head>
                                    <body>
                                        <div class="instruction">
                                            <p>Please fill in the following details to connect to your SSH server:</p>
                                            <div class="details">
                                                <p><strong>Remote Host</strong>: The hostname or IP address of your SSH server.</p>
                                                <p><strong>Username</strong>: Your SSH username.</p>
                                                <p><strong>Password</strong>: Your SSH password.</p>
                                            </div>
                                        </div>
                                        <div class="instruction">
                                            <p>Alternatively, you can upload a <code>.json</code> file directly to the app, and the details will be filled out automatically.</p>
                                        </div>
                                    </body>
                                    </html>
                    """)

            # Add the QTextEdit to the layout
            ssh_layout.addWidget(ssh_hint_label)

            # Create a fixed height spacer
            ssh_layout.addItem(QSpacerItem(20, 115, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

            self.widgets['ssh_remote_host'] = QLineEdit()
            self.widgets['ssh_remote_host'].setPlaceholderText("Remote Host")
            ssh_layout.addWidget(self.widgets['ssh_remote_host'])

            self.widgets['ssh_username'] = QLineEdit()
            self.widgets['ssh_username'].setPlaceholderText("Username")
            ssh_layout.addWidget(self.widgets['ssh_username'])

            self.widgets['ssh_password'] = QLineEdit()
            self.widgets['ssh_password'].setPlaceholderText("Password")
            self.widgets['ssh_password'].setEchoMode(QLineEdit.EchoMode.Password)
            ssh_layout.addWidget(self.widgets['ssh_password'])

            self.widgets['ssh_work_dir'] = QLineEdit()
            self.widgets['ssh_work_dir'].setPlaceholderText("Work Directory")
            ssh_layout.addWidget(self.widgets['ssh_work_dir'])

            # File uploader for SSH
            self.widgets['ssh_file_uploader'] = QLabel("Drag and drop a file here")
            self.widgets['ssh_file_uploader'].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.widgets['ssh_file_uploader'].setStyleSheet("border: 2px dashed #aaa; padding: 20px;")
            self.widgets['ssh_file_uploader'].setFixedHeight(100)
            ssh_layout.addWidget(self.widgets['ssh_file_uploader'])

            self.widgets['ssh_file_button'] = QPushButton("Select File")
            self.widgets['ssh_file_button'].clicked.connect(self.select_ssh_file)
            ssh_layout.addWidget(self.widgets['ssh_file_button'])

            # Connect drag and drop for S3 and SSH file uploaders
            self.widgets['s3_file_uploader'].setAcceptDrops(True)
            self.widgets['s3_file_uploader'].dragEnterEvent = self.drag_enter_event
            self.widgets['s3_file_uploader'].dropEvent = self.drop_event_s3

            self.widgets['ssh_file_uploader'].setAcceptDrops(True)
            self.widgets['ssh_file_uploader'].dragEnterEvent = self.drag_enter_event
            self.widgets['ssh_file_uploader'].dropEvent = self.drop_event_ssh

            # Add a vertical spacer to push the content to the top
            ssh_layout.addStretch()

        except Exception as e:
            print(f"Error in create_authentication_view: {e}")

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drop_event_s3(self, event):
        try:
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.load_s3_file(file_path)
        except Exception as e:
            print(f"Error in drop_event_s3: {e}")

    def drop_event_ssh(self, event):
        try:
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.load_ssh_file(file_path)
        except Exception as e:
            print(f"Error in drop_event_ssh: {e}")

    def select_s3_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select S3 File", "", "JSON Files (*.json);;All Files (*)")
            if file_path:
                self.load_s3_file(file_path)
        except Exception as e:
            print(f"Error in select_s3_file: {e}")

    def select_ssh_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select SSH File", "",
                                                       "JSON Files (*.json);;All Files (*)")
            if file_path:
                self.load_ssh_file(file_path)
        except Exception as e:
            print(f"Error in select_ssh_file: {e}")

    def load_s3_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.widgets['s3_host_base'].setText(data.get('host_base', ''))
                self.widgets['s3_use_https'].setChecked(data.get('use_https', False))
                self.widgets['s3_access_key'].setText(data.get('access_key', ''))
                self.widgets['s3_secret_key'].setText(data.get('secret_key', ''))
                self.widgets['s3_host_bucket'].setText(data.get('host_bucket', ''))
        except Exception as e:
            print(f"Error in load_s3_file: {e}")

    def load_ssh_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.widgets['ssh_remote_host'].setText(data.get('remote_host', ''))
                self.widgets['ssh_password'].setText(data.get('password', ''))
                self.widgets['ssh_work_dir'].setText(data.get('work_dir', ''))
        except Exception as e:
            print(f"Error in load_ssh_file: {e}")

    def create_submit_jobs_view(self):
        try:
            self.clear_view()

            # Main layout for the submit jobs view
            self.widgets['submit_jobs_layout'] = QHBoxLayout()
            self.content_layout.addLayout(self.widgets['submit_jobs_layout'])

            # S3 Layout (LEFT)
            self.widgets['s3_layout'] = QVBoxLayout()
            self.widgets['s3_layout_frame'] = QFrame()
            self.widgets['s3_layout_frame'].setLayout(self.widgets['s3_layout'])
            self.widgets['submit_jobs_layout'].addWidget(self.widgets['s3_layout_frame'])

            # S3 Header
            self.widgets['s3_header'] = QLabel("S3 Storage")
            self.widgets['s3_header'].setProperty('class', 'detectflow_header')
            self.widgets['s3_layout'].addWidget(self.widgets['s3_header'], alignment=Qt.AlignmentFlag.AlignHCenter)

            # S3 Tree view
            self.widgets['s3_tree_stacked_layout'] = QStackedLayout()

            self.widgets['s3_tree_view'] = QTreeView()
            self.widgets['s3_tree_stacked_layout'].addWidget(self.widgets['s3_tree_view'])
            self.widgets['s3_tree_stacked_layout'].setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

            self.widgets['s3_tree_overlay'] = LoadingOverlay(self.widgets['s3_tree_view'])
            self.widgets['s3_tree_stacked_layout'].addWidget(self.widgets['s3_tree_overlay'])
            self.widgets['s3_tree_stacked_layout'].setCurrentWidget(self.widgets['s3_tree_overlay'])

            self.widgets['s3_layout'].addLayout(self.widgets['s3_tree_stacked_layout'])

            # Remote layout (RIGHT)
            self.widgets['remote_layout'] = QVBoxLayout()
            self.widgets['remote_layout_frame'] = QFrame()
            self.widgets['remote_layout_frame'].setLayout(self.widgets['remote_layout'])
            self.widgets['submit_jobs_layout'].addWidget(self.widgets['remote_layout_frame'])

            # Remote header
            self.widgets['ssh_header'] = QLabel("Remote Work Directory")
            self.widgets['ssh_header'].setProperty('class', 'detectflow_header')
            self.widgets['remote_layout'].addWidget(self.widgets['ssh_header'], alignment=Qt.AlignmentFlag.AlignHCenter)

            # SSH Tree View
            self.widgets['ssh_tree_stacked_layout'] = QStackedLayout()

            self.widgets['ssh_tree_view'] = QTreeView()
            self.widgets['ssh_tree_stacked_layout'].addWidget(self.widgets['ssh_tree_view'])
            self.widgets['ssh_tree_stacked_layout'].setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

            self.widgets['ssh_tree_overlay'] = LoadingOverlay(self.widgets['ssh_tree_view'])
            self.widgets['ssh_tree_stacked_layout'].addWidget(self.widgets['ssh_tree_overlay'])
            self.widgets['ssh_tree_stacked_layout'].setCurrentWidget(self.widgets['ssh_tree_overlay'])

            self.widgets['remote_layout'].addLayout(self.widgets['ssh_tree_stacked_layout'])

            # Paths labels layout
            self.widgets['paths_labels_layout'] = QHBoxLayout()
            self.widgets['remote_layout'].addLayout(self.widgets['paths_labels_layout'])

            self.widgets['folder_path_label'] = QLabel("Folder Path")
            self.widgets['paths_labels_layout'].addWidget(self.widgets['folder_path_label'])

            self.widgets['script_path_label'] = QLabel("Script Path")
            self.widgets['paths_labels_layout'].addWidget(self.widgets['script_path_label'])

            # Paths buttons layout
            self.widgets['paths_buttons_layout'] = QHBoxLayout()
            self.widgets['remote_layout'].addLayout(self.widgets['paths_buttons_layout'])

            self.widgets['select_folder_btn'] = QPushButton("Select Folder")
            self.widgets['paths_buttons_layout'].addWidget(self.widgets['select_folder_btn'])

            self.widgets['select_script_btn'] = QPushButton("Select Script")
            self.widgets['paths_buttons_layout'].addWidget(self.widgets['select_script_btn'])

            # Job Config layout
            self.widgets['job_config_layout'] = QHBoxLayout()
            self.widgets['remote_layout'].addLayout(self.widgets['job_config_layout'])

            # Config Text Edit with Open/Save Buttons
            self.widgets['config_layout'] = QVBoxLayout()
            self.widgets['job_config_layout'].addLayout(self.widgets['config_layout'])

            self.widgets['config_text_edit'] = QPlainTextEdit()
            self.widgets['config_layout'].addWidget(self.widgets['config_text_edit'])

            self.widgets['config_buttons_layout'] = QHBoxLayout()
            self.widgets['config_layout'].addLayout(self.widgets['config_buttons_layout'])

            self.widgets['open_config_btn'] = QPushButton("Open Config")
            self.widgets['config_buttons_layout'].addWidget(self.widgets['open_config_btn'])

            self.widgets['save_config_btn'] = QPushButton("Save Config")
            self.widgets['config_buttons_layout'].addWidget(self.widgets['save_config_btn'])

            # Resource Specification
            self.widgets['resources_layout'] = QVBoxLayout()
            self.widgets['job_config_layout'].addLayout(self.widgets['resources_layout'])

            # Set the stretch factors to ensure equal width
            self.widgets['job_config_layout'].setStretch(0, 1)
            self.widgets['job_config_layout'].setStretch(1, 1)

            self.widgets['walltime_label'] = QLabel("Walltime (hours)")
            self.widgets['resources_layout'].addWidget(self.widgets['walltime_label'])

            self.widgets['walltime_input'] = QTimeEdit()
            self.widgets['resources_layout'].addWidget(self.widgets['walltime_input'])

            self.widgets['cpus_label'] = QLabel("Number of CPUs")
            self.widgets['resources_layout'].addWidget(self.widgets['cpus_label'])

            self.widgets['cpus_input'] = QSpinBox()
            self.widgets['resources_layout'].addWidget(self.widgets['cpus_input'])

            self.widgets['mem_label'] = QLabel("Requested Memory (GB)")
            self.widgets['resources_layout'].addWidget(self.widgets['mem_label'])

            self.widgets['mem_input'] = QSpinBox()
            self.widgets['resources_layout'].addWidget(self.widgets['mem_input'])

            self.widgets['gpu_checkbox'] = QCheckBox("Use GPU")
            self.widgets['resources_layout'].addWidget(self.widgets['gpu_checkbox'])

            self.widgets['scratch_label'] = QLabel("Scratch Size (GB)")
            self.widgets['resources_layout'].addWidget(self.widgets['scratch_label'])

            self.widgets['scratch_input'] = QSpinBox()
            self.widgets['resources_layout'].addWidget(self.widgets['scratch_input'])

            # Submit and Stop Buttons
            self.widgets['submit_buttons_layout'] = QHBoxLayout()
            self.widgets['remote_layout'].addLayout(self.widgets['submit_buttons_layout'])

            self.widgets['submit_btn'] = QPushButton("Submit Jobs")
            self.widgets['submit_btn'].setProperty('class', 'warning')
            self.widgets['submit_buttons_layout'].addWidget(self.widgets['submit_btn'])

            self.widgets['stop_btn'] = QPushButton("Stop Jobs")
            self.widgets['stop_btn'].setProperty('class', 'danger')
            self.widgets['submit_buttons_layout'].addWidget(self.widgets['stop_btn'])

            self.populate_tree_view(None, 's3_tree_view')

            #self.populate_tree_view(None, self.ssh, '/storage/brno2/home/USER/deploy', 'ssh_tree_view')

        except Exception as e:
            print(f"Error in create_submit_jobs_view: {e}")

    TREE_VIEWS_MAP = {
        's3_tree_view': {'thread': PopulateS3TreeTask, 'layout': 's3_tree_stacked_layout'},
        'ssh_tree_view': {'thread': PopulateSSHTreeTask, 'layout': 'ssh_tree_stacked_layout'}
    }

    def populate_tree_view(self, s3_config_path=None, ssh_client = None, root_path = None, tree_view_key='s3_tree_view'):

        if tree_view_key not in 's3_tree_view':
            #ssh_client = SSHHandler('USER', 'PASSWORD', 'HOST').ssh_client
            pass

        args = (s3_config_path, partial(self.set_tree_view_model, tree_view_key=tree_view_key)) if tree_view_key == 's3_tree_view' else (ssh_client, root_path, partial(self.set_tree_view_model, tree_view_key=tree_view_key))

        task = self.TREE_VIEWS_MAP[tree_view_key]['thread'](*args)
        QThreadPool.globalInstance().start(task)

    def set_tree_view_model(self, model, tree_view_key):
        self.widgets[tree_view_key].setModel(model)

        # collapsed_slot = partial(self.handle_tree_view_item, tree_view_key=tree_view_key, event='collapsed')
        # expanded_slot = partial(self.handle_tree_view_item, tree_view_key=tree_view_key, event='expanded')

        self.widgets[tree_view_key].collapsed.connect(partial(self.handle_tree_view_item, tree_view_key=tree_view_key, event='collapsed'))
        self.widgets[tree_view_key].expanded.connect(partial(self.handle_tree_view_item, tree_view_key=tree_view_key, event='expanded'))
        self.widgets[self.TREE_VIEWS_MAP[tree_view_key]['layout']].setCurrentWidget(self.widgets[tree_view_key])

    def handle_tree_view_item(self, index, tree_view_key: str, event: str):

        EVENT_MAP = {
            'collapsed': self.widgets[tree_view_key].model().handle_item_collapsed,
            'expanded': self.widgets[tree_view_key].model().handle_item_expanded
        }

        EVENT_MAP[event](index)

    def start_ssh_thread(self, username, password, hostname):
        # self.ssh_thread = QThread()
        # self.ssh_worker = SSHWorker(username, password, hostname)
        # self.ssh_worker.moveToThread(self.ssh_thread)
        #
        # self.ssh_thread.started.connect(self.ssh_worker.run)
        # self.ssh_worker.ssh_ready.connect(self.set_ssh)
        # self.ssh_worker.error.connect(self.handle_ssh_error)
        # self.ssh_worker.ssh_ready.connect(self.ssh_thread.quit)
        # self.ssh_worker.ssh_ready.connect(self.ssh_worker.deleteLater)
        # self.ssh_thread.finished.connect(self.ssh_thread.deleteLater)
        #
        # self.ssh_thread.start()

        # Start SSH connection
        try:
            self.ssh_worker = SSHWorker(username, password, hostname)
            self.ssh_thread = QThread()
            self.ssh_worker.moveToThread(self.ssh_thread)

            self.ssh_worker.ssh_ready.connect(self.on_ssh_ready)
            self.ssh_worker.error.connect(self.on_ssh_error)
            self.ssh_thread.started.connect(self.ssh_worker.run)
            self.ssh_thread.start()
        except Exception as e:
            print(f"Error in start_ssh_thread: {e}")

    def on_ssh_ready(self, ssh_handler):
        try:
            self.ssh = ssh_handler
            self.update_ssh_status("Connected")
        except Exception as e:
            print(f"Error in on_ssh_ready: {e}")

    def on_ssh_error(self, error_message):
        try:
            self.update_ssh_status("Disconnected")
            print(f"SSH Connection Error: {error_message}")
        except Exception as e:
            print(f"Error in on_ssh_error: {e}")

    def update_ssh_status(self, status):
        if status == "Connected":
            self.bottom_title_bar.ssh_status_label.setPixmap(QIcon(ICONS['link-alt']).pixmap(20, 20))
            self.bottom_title_bar.ssh_status_text.setText("SSH Connected")
        else:
            self.bottom_title_bar.ssh_status_label.setPixmap(QIcon(ICONS['broken-link']).pixmap(20, 20))
            self.bottom_title_bar.ssh_status_text.setText("SSH Disconnected")

    def create_monitor_jobs_view(self):
        try:
            self.clear_view()
            print("Create Monitor Jobs View")
        except Exception as e:
            print(f"Error in create_monitor_jobs_view: {e}")

    def create_launchpad_view(self):
        try:
            self.clear_view()
            print("Create Launchpad View")
        except Exception as e:
            print(f"Error in create_launchpad_view: {e}")

    def create_tutorial_view(self):
        try:
            self.clear_view()
            print("Create Tutorial View")
        except Exception as e:
            print(f"Error in create_tutorial_view: {e}")

    def create_settings_view(self):
        try:
            self.clear_view()
            print("Create Settings View")
        except Exception as e:
            print(f"Error in create_settings_view: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    from qt_material import apply_stylesheet
    from detectflow.resources import CSS

    extra = {

        # Density Scale
        'density_scale': '0',

        # Button colors
        'danger': '#ef2817',
        'warning': '#f67b12',
        'success': '#0788be',

        # Font
        'font_family': 'Roboto',

    }

    # setup stylesheet
    apply_stylesheet(app, theme='dark_amber.xml', invert_secondary=False, extra=extra, css_file=CSS['custom'])

    try:
        window = SchedulerApp()
        window.show()
    except Exception as e:
        print(e)
    sys.exit(app.exec())
