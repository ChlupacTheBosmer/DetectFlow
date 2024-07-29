import os
import configparser
from detectflow.config import S3_CONFIG
from detectflow.utils import install_s3_config


def parse_s3_config(cfg_file):
    """
    Parse the .cfg file to get S3 configuration details.

    :param cfg_file: Path to the .cfg file.
    :return: A tuple of endpoint_url, aws_access_key_id, and aws_secret_access_key.
    """

    # Validate filepath
    if os.path.isfile(cfg_file):

        # Init and load config
        config = configparser.ConfigParser()
        with open(cfg_file, "r") as cfg_file_obj:  # Open the file in text mode
            config.read_file(cfg_file_obj)

        # Retrieve values
        endpoint_url = config.get('default', 'host_base')
        access_key = config.get('default', 'access_key')
        secret_key = config.get('default', 'secret_key')

        return endpoint_url, access_key, secret_key
    else:
        return None, None, None


def is_s3_config_valid(cfg_file):
    if not os.path.exists(cfg_file):
        return False

    try:
        s3_config = parse_s3_config(cfg_file)
        if any(value is None for value in s3_config):
            return False
        else:
            return True
    except Exception as e:
        return False


def resolve_s3_config(cfg_dict: dict, cfg_file: str = S3_CONFIG):
    if not is_s3_config_valid(cfg_file):
        install_s3_config(cfg_dict.get('host_base'),
                          cfg_dict.get('use_https', True),
                          cfg_dict.get('access_key'),
                          cfg_dict.get('secret_key'),
                          cfg_dict.get('host_bucket'))

