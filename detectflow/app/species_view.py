import sys
import os
import pickle
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
                             QPushButton, QTableView, QHeaderView, QFileDialog,
                             QMainWindow, QApplication)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import hashlib
from detectflow.utils.hash import get_timestamp_hash
from PyQt6.QtWidgets import QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QWidget, QListWidgetItem, QSizePolicy
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QListWidget, QTextEdit, QVBoxLayout, QListWidgetItem, QToolBar, QInputDialog, QDialog, QVBoxLayout, QCheckBox, QPushButton

from detectflow.app import VISIT_VIEW_HELP
from PyQt6.QtWidgets import QStackedWidget
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QLineEdit, QLabel, QMenu
from PyQt6.QtWidgets import QStatusBar
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtGui import QImage, QBrush, QPainter
from PyQt6.QtCore import QThreadPool
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject, QThread
from detectflow.resources import ALT_ICONS


class EditableImageView(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Toolbar
        toolbar = QToolBar()

        #Add uttons
        add_btn = self.create_toolbar_button(ALT_ICONS['plus-circle'])
        add_btn.setToolTip('Add a new entry')
        toolbar.addWidget(add_btn)

        remove_btn = self.create_toolbar_button(ALT_ICONS['minus-circle'])
        remove_btn.setToolTip('Remove selected entry')
        toolbar.addWidget(remove_btn)

        save_btn = self.create_toolbar_button(ALT_ICONS['save'])
        save_btn.setToolTip('Save entries to file')
        toolbar.addWidget(save_btn)

        load_btn = self.create_toolbar_button(ALT_ICONS['folder'])
        load_btn.setToolTip('Load entries from file')
        toolbar.addWidget(load_btn)

        paste_btn = self.create_toolbar_button(ALT_ICONS['clipboard'])
        paste_btn.setToolTip('Paste image from clipboard')
        toolbar.addWidget(paste_btn)

        paste_btn.clicked.connect(self.paste_image_from_clipboard)

        # Add a QWidget with horizontal layout to the toolbar
        search_widget = QWidget()
        search_layout = QHBoxLayout(search_widget)
        search_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        search_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        search_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Add spacer to push the status bar to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        search_layout.addWidget(spacer)
        search_layout.setStretch(search_layout.indexOf(spacer), 4)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText('Search...')
        self.search_bar.textChanged.connect(self.filter_entries)
        self.search_bar.setFixedWidth(200)
        search_layout.addWidget(self.search_bar)
        search_layout.setStretch(search_layout.indexOf(self.search_bar), 1)

        toolbar.addWidget(search_widget)

        # TableView
        self.table_view = QTableView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Image', 'Species Name', 'Order', 'Video', 'Visit IDs'])
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Enable drag and drop
        self.table_view.setAcceptDrops(True)
        self.table_view.dragEnterEvent = self.dragEnterEvent
        self.table_view.dropEvent = self.dropEvent

        self.table_view.doubleClicked.connect(self.on_double_click)
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.open_context_menu)

        # Connect buttons to functions
        add_btn.clicked.connect(self.add_entry)
        remove_btn.clicked.connect(self.remove_entry)
        save_btn.clicked.connect(self.save_data)
        load_btn.clicked.connect(self.load_data)

        layout.addWidget(toolbar)
        layout.addWidget(self.table_view)
        self.setLayout(layout)

    def on_double_click(self, index):
        row = index.row()
        self.inspect_image(row)

    def create_toolbar_button(self, icon):
        button = QPushButton()
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(17, 17))
        button.setFixedSize(30, 30)  # Square button
        return button

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            image_path = url.toLocalFile()
            self.add_image_to_table(image_path)
    def filter_entries(self):
        filter_text = self.search_bar.text().lower()
        for row in range(self.model.rowCount()):
            match = False
            for column in range(self.model.columnCount()):
                item = self.model.item(row, column)
                if item and filter_text in item.text().lower():
                    match = True
                    break
            self.table_view.setRowHidden(row, not match)

    def add_entry(self):

        # Implement function to add an entry
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp)')
        if image_path:
            self.add_image_to_table(image_path)

    def remove_entry(self):
        # Implement function to remove selected entry
        selected = self.table_view.selectedIndexes()
        if selected:
            self.model.removeRow(selected[0].row())

    def save_data(self):
        data = []
        for row in range(self.model.rowCount()):
            image_item = self.model.item(row, 0)
            image_data = image_item.data(Qt.ItemDataRole.DecorationRole)

            if image_data:
                # Extract pixmap and save image path
                pixmap = image_data
                image_hash = get_timestamp_hash()
                image_path = os.path.join('app_data', f'{image_hash}.png')


                if not os.path.exists(image_path):
                    pixmap.save(image_path)

                entry = {
                    'image_path': image_path,
                    'species_name': self.model.item(row, 1).text(),
                    'order': self.model.item(row, 2).text(),
                    'video': self.model.item(row, 3).text(),
                    'visit_ids': self.model.item(row, 4).text(),
                }
                data.append(entry)

        file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Pickle Files (*.pkl)')
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Pickle Files (*.pkl)')
        if file_path:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.model.setRowCount(0)
            for entry in data:
                pixmap = QPixmap(entry['image_path']).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                item = QStandardItem()
                item.setData(pixmap, Qt.ItemDataRole.DecorationRole)
                item.setEditable(False)
                row = [
                    item,
                    QStandardItem(entry['species_name']),
                    QStandardItem(entry['order']),
                    QStandardItem(entry['video']),
                    QStandardItem(entry['visit_ids'])
                ]
                self.model.appendRow(row)

    def add_image_to_table(self, image_path):
        image_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
        new_image_path = os.path.join('app_data', f'{image_hash}.png')
        os.makedirs('app_data', exist_ok=True)
        if not os.path.exists(new_image_path):
            QPixmap(image_path).save(new_image_path)

        row_position = self.model.rowCount() - 1

        if row_position >= 0:
            # Check if the last row already has images
            image_list = self.model.item(row_position, 0).data(Qt.ItemDataRole.UserRole)
            if image_list:
                image_list.append(new_image_path)
                self.model.item(row_position, 0).setData(image_list, Qt.ItemDataRole.UserRole)
                self.update_image_mosaic(row_position)
                return

        # Create a new row for the new image
        pixmap = QPixmap(new_image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
        image_item = QStandardItem()
        image_item.setData(pixmap, Qt.ItemDataRole.DecorationRole)
        image_item.setData([new_image_path], Qt.ItemDataRole.UserRole)
        image_item.setEditable(False)

        row_position = self.model.rowCount()
        self.model.insertRow(row_position,
                             [image_item, QStandardItem(''), QStandardItem(''), QStandardItem(''), QStandardItem('')])
        self.update_image_mosaic(row_position)

    def update_image_mosaic(self, row):
        image_list = self.model.item(row, 0).data(Qt.ItemDataRole.UserRole)
        images = [QImage(image) for image in image_list]

        if images:
            mosaic_width = 100
            mosaic_height = 100
            mosaic_image = QImage(mosaic_width, mosaic_height, QImage.Format.Format_ARGB32)
            mosaic_image.fill(Qt.GlobalColor.transparent)

            x, y = 0, 0
            for img in images:
                img = img.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
                painter = QPainter(mosaic_image)
                painter.drawImage(x, y, img)
                painter.end()
                x += img.width()
                if x >= mosaic_width:
                    x = 0
                    y += img.height()

            pixmap = QPixmap.fromImage(mosaic_image)
            self.model.item(row, 0).setData(pixmap, Qt.ItemDataRole.DecorationRole)

    from PyQt6.QtGui import QClipboard

    def paste_image_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasImage():
            image = clipboard.image()
            image_hash = get_timestamp_hash()
            image_path = os.path.join('app_data', f'{image_hash}.png')
            image.save(image_path)

            self.add_image_to_table(image_path)

    from PyQt6.QtWidgets import QLabel, QDialog, QVBoxLayout

    def inspect_image(self, row):
        image_list = self.model.item(row, 0).data(Qt.ItemDataRole.UserRole)
        dialog = QDialog(self)
        layout = QVBoxLayout()

        for image_path in image_list:
            label = QLabel()
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap)
            layout.addWidget(label)

        dialog.setLayout(layout)
        dialog.exec()

    def remove_image_from_entry(self, row, image_path):
        image_list = self.model.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if image_path in image_list:
            image_list.remove(image_path)
            self.model.item(row, 0).setData(image_list, Qt.ItemDataRole.UserRole)
            self.update_image_mosaic(row)

    def open_context_menu(self, position):
        index = self.table_view.indexAt(position)
        if index.isValid():
            row = index.row()
            menu = QMenu()
            remove_action = menu.addAction("Remove Image")
            action = menu.exec(self.table_view.viewport().mapToGlobal(position))
            if action == remove_action:
                image_list = self.model.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if image_list:
                    self.remove_image_from_entry(row, image_list[-1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_widget = EditableImageView()
    main_win.setCentralWidget(main_widget)
    main_win.show()
    sys.exit(app.exec())