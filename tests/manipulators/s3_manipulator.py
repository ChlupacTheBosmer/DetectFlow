from detectflow.manipulators.s3_manipulator import S3Manipulator
import unittest

class TestS3Manipulator(unittest.TestCase):

    def setUp(self):
        self.s3 = S3Manipulator()

        self.bucket_name = 'test'
        self.directory = 'train/'

    def test_create_bucket(self):
        # Create a bucket
        self.s3.create_bucket_s3(self.bucket_name)

        # Check if the bucket was created
        self.assertTrue(self.s3.is_s3_bucket(self.bucket_name))

    def test_create_directory(self):
        # Create a directory
        self.s3.create_directory_s3(self.bucket_name, self.directory)

        # Check if the directory was created
        self.assertTrue(self.s3.is_s3_directory(f"s3://{self.bucket_name}/{self.directory}"))

    def test_upload_directory(self):

        directory_to_upload = r"D:\Dílna\Kutění\Python\Frames to label\CZ2_M1_AciArv1\visitor"

        # Upload a directory
        self.s3.upload_directory_s3(local_directory=directory_to_upload,
                                    bucket_name=self.bucket_name,
                                    s3_path=self.directory,
                                    max_attempts=3,
                                    delay=2)

        # Check if the directory was uploaded
        self.assertTrue(self.s3.is_s3_directory("s3://test/train/wrong/"))

    def test_delete_directory(self):
        # Delete a directory
        self.s3.delete_directory_s3(self.bucket_name, self.directory)

        # Check if the directory was deleted
        self.assertFalse(self.s3.is_s3_directory(f"s3://{self.bucket_name}/{self.directory}"))

    def test_find_files(self):

        # Find files in a directory
        files = self.s3.find_files_s3(r'GR2_L1_TolUmb02/(GR2_L1_TolUmb2|GR2_L1_TolUmb02).db', 'gr2-l1')

        print(files)

        # Check if the files were found
        self.assertTrue(files)

    def test_sort_files(self):
        # Find files in a directory
        files = self.s3.find_files_s3(r'GR2_L1_TolUmb[02|2|03|3].db', 'gr2-l1')

        # Sort files in a directory
        files = self.s3.sort_files_s3(files, sort_by='name', ascending=True)

        print(files)

        # Check if the files were sorted
        self.assertTrue(files)

    def test_get_metadata(self):
        # Get metadata of a file
        metadata = self.s3.get_metadata_s3('s3://gr2-l1/GR2_L1_TolUmb02/GR2_L1_TolUmb02_20220523_08_25.mp4',
                                           metadata_key='LastModified')

        print(type(metadata))

        # Check if the metadata was found
        self.assertTrue(metadata)