import os
import logging

import boto3
from botocore.exceptions import ClientError


class Storage:
    def __init__(self):
        self.bucket_name = os.environ["AWS_STORAGE_BUCKET_NAME"]

    def upload(self, file_name, object_name=None):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        s3_client = boto3.client("s3")
        try:
            _ = s3_client.upload_file(file_name, self.bucket_name, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def download(self, file_name, object_name):
        """Upload a file to an S3 bucket

        :param file_name: File to download to
        :param object_name: S3 object name
        :return: True if file was downloaded, else False
        """
        logging.info(
            f"Downloading from {self.bucket_name}:{object_name} to {file_name}"
        )
        # Download the file
        s3 = boto3.client("s3")
        try:
            _ = s3.download_file(self.bucket_name, object_name, file_name)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def blob_exists(self, object_name: str):
        s3_client = boto3.client("s3")
        try:
            s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError:
            return False

    def delete(self, object_name):
        """Delete an object from S3

        :param object_name: string
        """
        logging.info(f"Deleting {object_name} from {self.bucket_name} in storage")
        s3_client = boto3.client("s3")
        s3_client.delete_object(
            Bucket=self.bucket_name,
            Key=object_name,
        )
