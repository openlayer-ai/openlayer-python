"""Data upload helpers.

This module defines an interface to upload large amounts of data to
different storage backends.
"""

import os
from enum import Enum
from typing import Optional

import requests
from requests.adapters import Response
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from ... import _exceptions
from ..._client import Openlayer
from ...types.storage import PresignedURLCreateResponse


class StorageType(Enum):
    """Storage options for uploads."""

    FS = "local"
    AWS = "s3"
    GCP = "gcs"
    AZURE = "azure"


STORAGE = StorageType.AWS
REQUESTS_TIMEOUT = 60 * 60 * 3  # 3 hours
# Controls the `verify` parameter on requests in case a custom
# certificate is needed or needs to be disabled altogether
VERIFY_REQUESTS = True


class Uploader:
    """Internal class to handle http requests"""

    def __init__(self, client: Openlayer, storage: Optional[StorageType] = None):
        self.client = client
        self.storage = storage or STORAGE

    @staticmethod
    def _raise_on_respose(res: Response):
        try:
            message = res.json().get("error", res.text)
        except ValueError:
            message = res.text

        raise _exceptions.OpenlayerError(message)

    def upload(
        self,
        file_path: str,
        object_name: str,
        presigned_url_response: PresignedURLCreateResponse,
    ):
        """Generic method to upload data to the default storage medium and create the
        appropriate resource in the backend.
        """
        if self.storage == StorageType.AWS:
            return self.upload_blob_s3(
                file_path=file_path,
                object_name=object_name,
                presigned_url_response=presigned_url_response,
            )
        elif self.storage == StorageType.GCP:
            return self.upload_blob_gcs(
                file_path=file_path,
                presigned_url_response=presigned_url_response,
            )
        elif self.storage == StorageType.AZURE:
            return self.upload_blob_azure(
                file_path=file_path,
                presigned_url_response=presigned_url_response,
            )
        else:
            return self.upload_blob_local(
                file_path=file_path,
                object_name=object_name,
                presigned_url_response=presigned_url_response,
            )

    def upload_blob_s3(
        self,
        file_path: str,
        object_name: str,
        presigned_url_response: PresignedURLCreateResponse = None,
    ):
        """Generic method to upload data to S3 storage and create the appropriate
        resource in the backend.
        """

        with tqdm(
            total=os.stat(file_path).st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            colour="BLUE",
        ) as t:
            with open(file_path, "rb") as f:
                # Avoid logging here as it will break the progress bar
                fields = presigned_url_response.fields
                fields["file"] = (object_name, f, "application/x-tar")
                e = MultipartEncoder(fields=fields)
                m = MultipartEncoderMonitor(e, lambda monitor: t.update(min(t.total, monitor.bytes_read) - t.n))
                headers = {"Content-Type": m.content_type}
                res = requests.post(
                    presigned_url_response.url,
                    data=m,
                    headers=headers,
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res

    def upload_blob_gcs(self, file_path: str, presigned_url_response: PresignedURLCreateResponse):
        """Generic method to upload data to Google Cloud Storage and create the
        appropriate resource in the backend.
        """
        with open(file_path, "rb") as f:
            with tqdm(
                total=os.stat(file_path).st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                res = requests.put(
                    presigned_url_response.url,
                    data=wrapped_file,
                    headers={"Content-Type": "application/x-gzip"},
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res

    def upload_blob_azure(self, file_path: str, presigned_url_response: PresignedURLCreateResponse):
        """Generic method to upload data to Azure Blob Storage and create the
        appropriate resource in the backend.
        """
        with open(file_path, "rb") as f:
            with tqdm(
                total=os.stat(file_path).st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                res = requests.put(
                    presigned_url_response.url,
                    data=wrapped_file,
                    headers={
                        "Content-Type": "application/x-gzip",
                        "x-ms-blob-type": "BlockBlob",
                    },
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res

    def upload_blob_local(
        self,
        file_path: str,
        object_name: str,
        presigned_url_response: PresignedURLCreateResponse,
    ):
        """Generic method to transfer data to the openlayer folder and create the
        appropriate resource in the backend when using a local deployment.
        """
        with tqdm(
            total=os.stat(file_path).st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            colour="BLUE",
        ) as t:
            with open(file_path, "rb") as f:
                fields = {"file": (object_name, f, "application/x-tar")}
                e = MultipartEncoder(fields=fields)
                m = MultipartEncoderMonitor(e, lambda monitor: t.update(min(t.total, monitor.bytes_read) - t.n))
                headers = {"Content-Type": m.content_type}
                res = requests.post(
                    presigned_url_response.url,
                    data=m,
                    headers=headers,
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res
