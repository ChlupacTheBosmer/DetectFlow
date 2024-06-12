import os

# Get the directory of this __init__.py file
CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CONFIG_ROOT)
S3_CONFIG = os.path.join(CONFIG_ROOT, '.s3.cfg')

