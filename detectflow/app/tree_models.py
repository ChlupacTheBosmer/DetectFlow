from PyQt6.QtGui import QStandardItemModel, QStandardItem, QIcon
from PyQt6.QtCore import Qt, pyqtSlot, QModelIndex
from detectflow.resources import ICONS
import os


class S3TreeModel(QStandardItemModel):
    def __init__(self, data, parent=None):
        super(S3TreeModel, self).__init__(parent)
        self.setHorizontalHeaderLabels(['S3 Buckets'])
        self.populate_model(data)

    def populate_model(self, data):
        def add_items(parent, elements):
            for key, value in elements.items():
                item = QStandardItem(key)
                item.setCheckable(True)
                if isinstance(value, dict):
                    collapsed_icon = QIcon(ICONS['folder'])
                    expanded_icon = QIcon(ICONS['open-folder'])
                    item.setIcon(collapsed_icon)
                    item.setData(collapsed_icon, Qt.ItemDataRole.UserRole + 1)
                    item.setData(expanded_icon, Qt.ItemDataRole.UserRole + 2)
                    add_items(item, value)
                elif isinstance(value, list):
                    collapsed_icon = QIcon(ICONS['folder'])
                    expanded_icon = QIcon(ICONS['open-folder'])
                    item.setIcon(collapsed_icon)
                    item.setData(collapsed_icon, Qt.ItemDataRole.UserRole + 1)
                    item.setData(expanded_icon, Qt.ItemDataRole.UserRole + 2)
                    for val in value:
                        sub_item = QStandardItem(val)
                        sub_item.setCheckable(False)
                        sub_item.setIcon(QIcon(ICONS['file']))
                        item.appendRow(sub_item)
                parent.appendRow(item)

        for bucket_name, directories in data.items():
            bucket_item = QStandardItem(bucket_name)
            bucket_item.setCheckable(True)
            collapsed_icon = QIcon(ICONS['closed-folder-plus'])
            expanded_icon = QIcon(ICONS['open-folder-plus'])
            bucket_item.setIcon(collapsed_icon)
            bucket_item.setData(collapsed_icon, Qt.ItemDataRole.UserRole + 1)
            bucket_item.setData(expanded_icon, Qt.ItemDataRole.UserRole + 2)
            self.appendRow(bucket_item)
            add_items(bucket_item, directories)

    @pyqtSlot(QModelIndex)
    def handle_item_collapsed(self, index):
        print("processing collapsed")
        item = self.itemFromIndex(index)
        collapsed_icon = item.data(Qt.ItemDataRole.UserRole + 1)
        if collapsed_icon:
            item.setIcon(collapsed_icon)

    @pyqtSlot(QModelIndex)
    def handle_item_expanded(self, index):
        item = self.itemFromIndex(index)
        expanded_icon = item.data(Qt.ItemDataRole.UserRole + 2)
        if expanded_icon:
            item.setIcon(expanded_icon)

    def gather_selected_data(self):
        def gather_items(parent_item):
            selected_buckets = {}
            for row in range(parent_item.rowCount()):
                bucket_item = parent_item.child(row)
                bucket_name = bucket_item.text()
                selected_directories = []

                for dir_row in range(bucket_item.rowCount()):
                    dir_item = bucket_item.child(dir_row)
                    if dir_item.checkState() == Qt.CheckState.Checked:
                        selected_directories.append(dir_item.text())

                if bucket_item.checkState() == Qt.CheckState.Checked:
                    if not selected_directories:
                        # If bucket is selected but no directories are selected, include all directories
                        for dir_row in range(bucket_item.rowCount()):
                            dir_item = bucket_item.child(dir_row)
                            selected_directories.append(dir_item.text())
                    selected_buckets[bucket_name] = selected_directories

            return selected_buckets

        # Get the root item and gather selected data
        root_item = self.invisibleRootItem()
        selected_data = gather_items(root_item)
        return selected_data


    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.CheckStateRole:
            item = self.itemFromIndex(index)
            item_type = item.data(Qt.ItemDataRole.UserRole + 3)
            if value == Qt.CheckState.Checked:
                self.check_all_children(item, Qt.CheckState.Checked)
            elif value == Qt.CheckState.Unchecked:
                self.uncheck_all_children(item)
                self.uncheck_parent_if_all_children_unchecked(item)
        return super().setData(index, value, role)

    def check_all_children(self, item, check_state):
        item.setCheckState(check_state)
        for row in range(item.rowCount()):
            child_item = item.child(row)
            self.check_all_children(child_item, check_state)

    def uncheck_all_children(self, item):
        item.setCheckState(Qt.CheckState.Unchecked)
        for row in range(item.rowCount()):
            child_item = item.child(row)
            self.uncheck_all_children(child_item)

    def uncheck_parent_if_all_children_unchecked(self, item):
        parent_item = item.parent()
        if parent_item is not None:
            all_unchecked = True
            for row in range(parent_item.rowCount()):
                sibling_item = parent_item.child(row)
                if sibling_item.checkState() == Qt.CheckState.Checked:
                    all_unchecked = False
                    break
            if all_unchecked:
                parent_item.setCheckState(Qt.CheckState.Unchecked)
                self.uncheck_parent_if_all_children_unchecked(parent_item)


from PyQt6.QtGui import QStandardItemModel, QStandardItem, QIcon
from PyQt6.QtCore import Qt
import os

class SSHTreeModel(QStandardItemModel):
    def __init__(self, data, parent=None):
        super(SSHTreeModel, self).__init__(parent)
        self.setHorizontalHeaderLabels(['SSH Filesystem'])
        self.icon_map = self.load_icons()
        self.populate_model(data)

    def load_icons(self):
        icons = {
            'folder_collapsed': QIcon(ICONS['folder']),
            'folder_expanded': QIcon(ICONS['open-folder']),
            'file': QIcon(ICONS['file']),
            'py': QIcon(ICONS['file']),
            'txt': QIcon(ICONS['file']),
            'json': QIcon(ICONS['file']),
            'db': QIcon(ICONS['file']),
            'png': QIcon(ICONS['file']),
            'jpeg': QIcon(ICONS['file']),
            'err': QIcon(ICONS['file']),
            'out': QIcon(ICONS['file']),
            'sh': QIcon(ICONS['file'])
        }
        return icons

    def populate_model(self, data):
        def add_items(parent, elements):
            for key, value in elements.items():
                item = QStandardItem(key)
                item.setCheckable(True)
                if isinstance(value, dict):
                    item.setIcon(self.icon_map['folder_collapsed'])
                    item.setData(self.icon_map['folder_collapsed'], Qt.ItemDataRole.UserRole + 1)
                    item.setData(self.icon_map['folder_expanded'], Qt.ItemDataRole.UserRole + 2)
                    item.setData('folder', Qt.ItemDataRole.UserRole + 3)  # Mark item as folder
                    add_items(item, value)
                elif isinstance(value, list):
                    for val in value:
                        sub_item = QStandardItem(val)
                        sub_item.setCheckable(True)
                        file_extension = val.split('.')[-1].lower()
                        sub_item.setIcon(self.icon_map.get(file_extension, self.icon_map['file']))
                        sub_item.setData('file', Qt.ItemDataRole.UserRole + 3)  # Mark item as file
                        item.appendRow(sub_item)
                parent.appendRow(item)

        for dir_name, contents in data.items():
            dir_item = QStandardItem(dir_name)
            dir_item.setCheckable(True)
            dir_item.setIcon(self.icon_map['folder_collapsed'])
            dir_item.setData(self.icon_map['folder_collapsed'], Qt.ItemDataRole.UserRole + 1)
            dir_item.setData(self.icon_map['folder_expanded'], Qt.ItemDataRole.UserRole + 2)
            dir_item.setData('folder', Qt.ItemDataRole.UserRole + 3)  # Mark item as folder
            self.appendRow(dir_item)
            add_items(dir_item, contents)

    @pyqtSlot(QModelIndex)
    def handle_item_collapsed(self, index):
        item = self.itemFromIndex(index)
        item_type = item.data(Qt.ItemDataRole.UserRole + 3)
        if item_type == 'folder':
            collapsed_icon = item.data(Qt.ItemDataRole.UserRole + 1)
            if collapsed_icon:
                item.setIcon(collapsed_icon)

    @pyqtSlot(QModelIndex)
    def handle_item_expanded(self, index):
        item = self.itemFromIndex(index)
        item_type = item.data(Qt.ItemDataRole.UserRole + 3)
        if item_type == 'folder':
            expanded_icon = item.data(Qt.ItemDataRole.UserRole + 2)
            if expanded_icon:
                item.setIcon(expanded_icon)