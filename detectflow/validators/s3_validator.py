import re
import boto3
from detectflow.utils.s3.cfg import parse_s3_config
from detectflow import S3_CONFIG

class S3Validator:
    def __init__(self, cfg_file: str = S3_CONFIG):
        """Initialize the S3 client."""
        self.endpoint_url, self.aws_access_key_id, self.aws_secret_access_key = parse_s3_config(cfg_file)
        region_name = 'eu-west-2'

        # Initialize the S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=region_name  # or your preferred region
        )

    def is_s3_bucket(self, input_data):
        """Check if the input_data is an S3 bucket."""
        try:
            self.s3_client.list_objects_v2(Bucket=input_data, MaxKeys=1)
            return True
        except self.s3_client.exceptions.NoSuchBucket:
            return False
        except Exception:
            # Handle other possible exceptions
            return False

    def is_s3_directory(self, input_data):
        """Check if the input_data is an S3 directory."""
        bucket_name, prefix = self._parse_s3_path(input_data)
        if not bucket_name:
            return False

        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/', MaxKeys=1)
            return 'CommonPrefixes' in response or 'Contents' in response
        except Exception:
            return False

    def is_s3_file(self, input_data):
        """Check if the input_data is an S3 file."""
        bucket_name, key = self._parse_s3_path(input_data)
        if not bucket_name:
            return False

        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False
        except Exception:
            return False

    @staticmethod
    def _parse_s3_path(s3_path):
        """Utility method to parse an S3 path into bucket and key/prefix."""
        match = re.match(r's3://([^/]+)/?(.*)', s3_path)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def is_valid_s3_bucket_name(self, input_data: str) -> bool:
        """
        Validate S3 bucket name as per AWS bucket naming rules.
        Reference: https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
        """
        pattern = re.compile(r'^(?!-)(?!.*--)[a-z0-9-]{3,63}(?<!-)$')
        return pattern.match(input_data) is not None