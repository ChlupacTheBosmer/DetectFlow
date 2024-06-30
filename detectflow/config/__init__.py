import os
from detectflow.manipulators.manipulator import Manipulator

# Get the directory of this __init__.py file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTFLOW_DIR = os.path.dirname(CONFIG_DIR)
ROOT_DIR = os.path.dirname(DETECTFLOW_DIR)
TESTS_DIR = os.path.join(os.path.dirname(DETECTFLOW_DIR), 'tests')
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, 'notebooks')

# Search for .s3.cfg file in the root directory
files = Manipulator.find_files(ROOT_DIR, '.s3.cfg')
if files:
    S3_CONFIG = files[0]
else:
    S3_CONFIG = os.path.join(CONFIG_DIR, '.s3.cfg') # TODO: make it smarter by checking if the file exists and merge it with the settings file and user preferences, then implement this in the default usage in other classes
# TODO: Add find file automatic location of the config file
