from PyQt6.QtCore import QRunnable, pyqtSlot
import logging
from detectflow.app.tree_models import S3TreeModel, SSHTreeModel
from detectflow.manipulators.s3_manipulator import S3Manipulator
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from detectflow.config import S3_CONFIG
import os
from functools import lru_cache
from detectflow.handlers.ssh_handler import SSHHandler
import paramiko


class PopulateS3TreeTask(QRunnable):
    def __init__(self, s3_config_path, callback):
        super(PopulateS3TreeTask, self).__init__()
        self.s3_config_path = s3_config_path if s3_config_path else S3_CONFIG
        self.data = None
        self.callback = callback

    @staticmethod
    @lru_cache(maxsize=2)
    def get_data(s3_config_path):

        try:
            print("Getting data...")
            s3_manipulator = S3Manipulator(s3_config_path)

            buckets = s3_manipulator.list_buckets_s3(regex=r'cz1-*')

            s3_data = {}
            for bucket in buckets:
                print(bucket)
                directories = s3_manipulator.list_directories_s3(bucket)
                bucket_contents = {}
                for directory in directories:
                    files = s3_manipulator.list_files_s3(bucket, directory, regex=r"^(?!\.)[^.]*\.(mp4|avi)$", return_full_path=False)
                    bucket_contents[directory] = files
                s3_data[bucket] = bucket_contents
        except Exception as e:
            logging.error(f"Error while populating S3 tree: {e}")
            s3_data = {}
        return s3_data

    @pyqtSlot()
    def run(self):
        self.data = PopulateS3TreeTask.get_data(self.s3_config_path)
        model = S3TreeModel(self.data)
        self.callback(model)


from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal

class PopulateSSHTreeTask(QRunnable):
    def __init__(self, ssh_client, root_path, callback):
        super(PopulateSSHTreeTask, self).__init__()
        self.ssh_client = ssh_client
        self.root_path = root_path
        self.callback = callback
        self.data = None

    @staticmethod
    @lru_cache(maxsize=2)
    def get_data(ssh, root_path):
        directory_structure = {}

        def recurse_path(current_path):
            contents = {}
            stdin, stdout, stderr = ssh.exec_command(f'ls -F {current_path}')
            for line in stdout:
                line = line.strip()
                if line.endswith('/'):
                    folder_name = line[:-1]
                    contents[folder_name] = recurse_path(os.path.join(current_path, folder_name))
                else:
                    contents.setdefault('files', []).append(line)
            return contents

        directory_structure[root_path] = recurse_path(root_path)
        return directory_structure

    @pyqtSlot()
    def run(self):
        self.data = self.get_data(self.ssh_client, self.root_path)
        model = SSHTreeModel(self.data)
        self.callback(model)


class SSHWorker(QObject):
    ssh_ready = pyqtSignal(object)  # Signal to indicate that SSH is ready
    error = pyqtSignal(str)  # Signal to indicate an error occurred

    def __init__(self, username, password, hostname):
        super().__init__()
        self.username = username
        self.password = password
        self.hostname = hostname

    @pyqtSlot()
    def run(self):
        try:
            ssh = SSHHandler(self.username, self.password, self.hostname)
            self.ssh_ready.emit(ssh)
        except Exception as e:
            print(e)
            #self.error.emit(str(e))
