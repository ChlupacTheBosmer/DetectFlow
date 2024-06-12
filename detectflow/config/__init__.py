import os

# Get the directory of this __init__.py file
CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
S3_CONFIG = os.path.join(CONFIG_ROOT, '.s3.cfg')

# Defining the public API of the package
__all__ = ['S3_CONFIG']

