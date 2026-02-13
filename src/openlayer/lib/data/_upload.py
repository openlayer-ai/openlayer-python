"""Data upload helpers.

This module defines an interface to upload large amounts of data to
different storage backends.
"""

import io
import os
from enum import Enum
from typing import BinaryIO, Dict, Optional, Union

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


# ----- Low-level upload functions (work with bytes or file-like objects) ---- #
def upload_bytes(
    storage: StorageType,
    url: str,
    data: Union[bytes, BinaryIO],
    object_name: str,
    content_type: str,
    fields: Optional[Dict] = None,
) -> Response:
    """Upload data to the appropriate storage backend.

    This is a convenience function that routes to the correct upload method
    based on the storage type.

    Args:
        storage: The storage backend type.
        url: The presigned URL to upload to.
        data: The data to upload (bytes or file-like object).
        object_name: The object name (used for multipart uploads).
        content_type: The MIME type of the data.
        fields: Additional fields for multipart uploads (S3 policy fields).

    Returns:
        The response from the upload request.
    """
    if storage == StorageType.AWS:
        return upload_bytes_multipart(
            url=url,
            data=data,
            object_name=object_name,
            content_type=content_type,
            fields=fields or {},
        )
    elif storage == StorageType.GCP:
        return upload_bytes_put(
            url=url,
            data=data,
            content_type=content_type,
        )
    elif storage == StorageType.AZURE:
        return upload_bytes_put(
            url=url,
            data=data,
            content_type=content_type,
            extra_headers={"x-ms-blob-type": "BlockBlob"},
        )
    else:
        # Local storage uses multipart POST (no extra fields)
        return upload_bytes_multipart(
            url=url,
            data=data,
            object_name=object_name,
            content_type=content_type,
        )


def upload_bytes_multipart(
    url: str,
    data: Union[bytes, BinaryIO],
    object_name: str,
    content_type: str,
    fields: Dict[str, str] = {},
) -> Response:
    """Upload data using multipart POST (for S3 and local storage).

    Args:
        url: The presigned URL to upload to.
        data: The data to upload (bytes or file-like object).
        object_name: The object name for the file field.
        content_type: The MIME type of the data.
        fields: Additional fields to include in the multipart form (e.g., S3 policy fields).

    Returns:
        The response from the upload request.
    """
    # Convert bytes to file-like object if needed
    if isinstance(data, bytes):
        data = io.BytesIO(data)

    # S3 requires the "file" field to be last in the multipart form.
    # Policy fields (key, policy, x-amz-credential, etc.) must come first.
    fields = fields or {}
    upload_fields = {**fields, "file": (object_name, data, content_type)}

    encoder = MultipartEncoder(fields=upload_fields)
    headers = {"Content-Type": encoder.content_type}

    response = requests.post(
        url,
        data=encoder,
        headers=headers,
        verify=VERIFY_REQUESTS,
        timeout=REQUESTS_TIMEOUT,
    )
    response.raise_for_status()
    return response


def upload_bytes_put(
    url: str,
    data: Union[bytes, BinaryIO],
    content_type: str,
    extra_headers: Dict[str, str] = {},
) -> Response:
    """Upload data using PUT request (for GCS and Azure).

    Args:
        url: The presigned URL to upload to.
        data: The data to upload (bytes or file-like object).
        content_type: The MIME type of the data.
        extra_headers: Additional headers (e.g., x-ms-blob-type for Azure).

    Returns:
        The response from the upload request.
    """
    headers = {"Content-Type": content_type, **extra_headers}

    response = requests.put(
        url,
        data=data,
        headers=headers,
        verify=VERIFY_REQUESTS,
        timeout=REQUESTS_TIMEOUT,
    )
    response.raise_for_status()
    return response


# --- High-level Uploader class (file-based uploads with progress tracking) -- #
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
                m = MultipartEncoderMonitor(
                    e, lambda monitor: t.update(min(t.total, monitor.bytes_read) - t.n)
                )
                headers = {"Content-Type": m.content_type}
                res = requests.post(
                    presigned_url_response.url,
                    data=m,
                    headers=headers,
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res

    def upload_blob_gcs(
        self, file_path: str, presigned_url_response: PresignedURLCreateResponse
    ):
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

    def upload_blob_azure(
        self, file_path: str, presigned_url_response: PresignedURLCreateResponse
    ):
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
                m = MultipartEncoderMonitor(
                    e, lambda monitor: t.update(min(t.total, monitor.bytes_read) - t.n)
                )
                headers = {"Content-Type": m.content_type}
                res = requests.post(
                    presigned_url_response.url,
                    data=m,
                    headers=headers,
                    verify=VERIFY_REQUESTS,
                    timeout=REQUESTS_TIMEOUT,
                )
        return res
