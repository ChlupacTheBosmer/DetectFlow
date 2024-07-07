import os
import configparser


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
