"""Module that contains the core functionality of the Openlayer Python SDK.

This module mainly defines the Api class, which is used by the OpenlayerClient
to make requests to the Openlayer API.
The StorageType enum is also defined here, which is used to specify what kind
of storage the OpenlayerClient should use for uploads.

Typical usage example:

    from . import api

    self.api = api.Api(api_key)
    endpoint = "projects"
    payload = {
        "name": name,
        "description": description,
        "taskType": task_type.value,
    }
    project_data = self.api.post_request(endpoint, body=payload)

"""
import os
import shutil
from enum import Enum

import requests
from requests.adapters import HTTPAdapter, Response, Retry
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from . import constants
from .exceptions import ExceptionMap, OpenlayerException
from .version import __version__

# Parameters for HTTP retry
HTTP_TOTAL_RETRIES = 3  # Number of total retries
HTTP_RETRY_BACKOFF_FACTOR = 2  # Wait 1, 2, 4 seconds between retries
HTTP_STATUS_FORCE_LIST = [408, 429] + list(range(500, 504)) + list(range(506, 531))
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "PUT", "POST"})

CLIENT_METADATA = {"version": __version__}


class StorageType(Enum):
    """Storage options for uploads."""

    ONPREM = "local"
    AWS = "s3"
    GCP = "gcs"
    AZURE = "azure"


STORAGE = StorageType.AWS
OPENLAYER_ENDPOINT = "https://api.openlayer.com/v1"
# Controls the `verify` parameter on requests in case a custom
# certificate is needed or needs to be disabled altogether
VERIFY_REQUESTS = True


class Api:
    """Internal class to handle http requests"""

    def __init__(self, api_key: str):
        if api_key == "" or api_key is None:
            raise OpenlayerException(
                "There is an issue instantiating the OpenlayerClient. \n"
                "An invalid API key is being provided. \n"
                "Make sure to provide a valid API key using the syntax "
                "`OpenlayerClient('YOUR_API_KEY_HERE')`. You can find your API keys "
                "in the Profile page on the Openlayer platform."
            )

        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self._headers_multipart_form_data = {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _http_request(
        method,
        url,
        headers=None,
        params=None,
        body=None,
        files=None,
        data=None,
        include_metadata=True,
    ) -> Response:
        with requests.Session() as https:
            retry_strategy = Retry(
                total=HTTP_TOTAL_RETRIES,
                backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
                status_forcelist=HTTP_STATUS_FORCE_LIST,
                allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
                raise_on_status=False,
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            https.mount("https://", adapter)

            try:
                params = params or {}
                if include_metadata:
                    params.update(CLIENT_METADATA)
                res = https.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=body,
                    files=files,
                    data=data,
                )

                return res
            except Exception as err:
                raise OpenlayerException(err) from err

    @staticmethod
    def _raise_on_respose(res: Response):
        try:
            message = res.json().get("error", res.text)
        except ValueError:
            message = res.text

        exception = ExceptionMap.get(res.status_code, OpenlayerException)
        raise exception(message, res.status_code)

    def _api_request(
        self,
        method,
        endpoint,
        headers=None,
        params=None,
        body=None,
        files=None,
        data=None,
        include_metadata=True,
    ):
        """Make any HTTP request + error handling."""

        url = f"{OPENLAYER_ENDPOINT}/{endpoint}"

        res = self._http_request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            body=body,
            files=files,
            data=data,
            include_metadata=include_metadata,
        )

        json = None
        if res.ok:
            json = res.json()
        else:
            self._raise_on_respose(res)

        return json

    def get_request(self, endpoint: str, params=None):
        """Generic GET Request Wrapper."""
        return self._api_request("GET", endpoint, headers=self._headers, params=params)

    def post_request(
        self, endpoint: str, body=None, files=None, data=None, include_metadata=True
    ):
        """Generic POST Request Wrapper."""
        return self._api_request(
            method="POST",
            endpoint=endpoint,
            headers=self._headers
            if files is None
            else self._headers_multipart_form_data,
            body=body,
            files=files,
            data=data,
            include_metadata=include_metadata,
        )

    def put_request(self, endpoint: str, body=None, files=None, data=None):
        """Generic PUT Request Wrapper."""
        return self._api_request(
            "PUT",
            endpoint,
            headers=self._headers
            if files is None
            else self._headers_multipart_form_data,
            body=body,
            files=files,
            data=data,
        )

    def upload(
        self,
        endpoint: str,
        file_path: str,
        object_name: str = None,
        body=None,
        method: str = "POST",
        storage_uri_key: str = "storageUri",
        presigned_url_endpoint: str = "storage/presigned-url",
        presigned_url_query_params: str = "",
    ):
        """Generic method to upload data to the default storage medium and create the
        appropriate resource in the backend.
        """
        if STORAGE == StorageType.AWS:
            upload = self.upload_blob_s3
        elif STORAGE == StorageType.GCP:
            upload = self.upload_blob_gcs
        elif STORAGE == StorageType.AZURE:
            upload = self.upload_blob_azure
        else:
            upload = self.transfer_blob

        return upload(
            endpoint=endpoint,
            file_path=file_path,
            object_name=object_name,
            body=body,
            method=method,
            storage_uri_key=storage_uri_key,
            presigned_url_endpoint=presigned_url_endpoint,
            presigned_url_query_params=presigned_url_query_params,
        )

    def upload_blob_s3(
        self,
        endpoint: str,
        file_path: str,
        object_name: str = None,
        body=None,
        method: str = "POST",
        storage_uri_key: str = "storageUri",
        presigned_url_endpoint: str = "storage/presigned-url",
        presigned_url_query_params: str = "",
    ):
        """Generic method to upload data to S3 storage and create the appropriate
        resource in the backend.
        """

        presigned_json = self.post_request(
            (
                f"{presigned_url_endpoint}?objectName={object_name}"
                f"&{presigned_url_query_params}"
            )
        )

        with tqdm(
            total=os.stat(file_path).st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            colour="BLUE",
        ) as t:
            with open(file_path, "rb") as f:
                # Avoid logging here as it will break the progress bar
                fields = presigned_json["fields"]
                fields["file"] = (object_name, f, "application/x-tar")
                e = MultipartEncoder(fields=fields)
                m = MultipartEncoderMonitor(
                    e, lambda monitor: t.update(min(t.total, monitor.bytes_read) - t.n)
                )
                headers = {"Content-Type": m.content_type}
                res = requests.post(
                    presigned_json["url"],
                    data=m,
                    headers=headers,
                    verify=VERIFY_REQUESTS,
                    timeout=constants.REQUESTS_TIMEOUT,
                )

        if res.ok:
            body[storage_uri_key] = presigned_json["storageUri"]
            if method == "POST":
                return self.post_request(f"{endpoint}", body=body)
            elif method == "PUT":
                return self.put_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def upload_blob_gcs(
        self,
        endpoint: str,
        file_path: str,
        object_name: str = None,
        body=None,
        method: str = "POST",
        storage_uri_key: str = "storageUri",
        presigned_url_endpoint: str = "storage/presigned-url",
        presigned_url_query_params: str = "",
    ):
        """Generic method to upload data to Google Cloud Storage and create the
        appropriate resource in the backend.
        """
        presigned_json = self.post_request(
            (
                f"{presigned_url_endpoint}?objectName={object_name}"
                f"&{presigned_url_query_params}"
            )
        )
        with open(file_path, "rb") as f:
            with tqdm(
                total=os.stat(file_path).st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                res = requests.put(
                    presigned_json["url"],
                    data=wrapped_file,
                    headers={"Content-Type": "application/x-gzip"},
                    verify=VERIFY_REQUESTS,
                    timeout=constants.REQUESTS_TIMEOUT,
                )
        if res.ok:
            body[storage_uri_key] = presigned_json["storageUri"]
            if method == "POST":
                return self.post_request(f"{endpoint}", body=body)
            elif method == "PUT":
                return self.put_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def upload_blob_azure(
        self,
        endpoint: str,
        file_path: str,
        object_name: str = None,
        body=None,
        method: str = "POST",
        storage_uri_key: str = "storageUri",
        presigned_url_endpoint: str = "storage/presigned-url",
        presigned_url_query_params: str = "",
    ):
        """Generic method to upload data to Azure Blob Storage and create the
        appropriate resource in the backend.
        """
        presigned_json = self.post_request(
            (
                f"{presigned_url_endpoint}?objectName={object_name}"
                f"&{presigned_url_query_params}"
            )
        )
        with open(file_path, "rb") as f:
            with tqdm(
                total=os.stat(file_path).st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                res = requests.put(
                    presigned_json["url"],
                    data=wrapped_file,
                    headers={
                        "Content-Type": "application/x-gzip",
                        "x-ms-blob-type": "BlockBlob",
                    },
                    verify=VERIFY_REQUESTS,
                    timeout=constants.REQUESTS_TIMEOUT,
                )
        if res.ok:
            body[storage_uri_key] = presigned_json["storageUri"]
            if method == "POST":
                return self.post_request(f"{endpoint}", body=body)
            elif method == "PUT":
                return self.put_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def transfer_blob(
        self,
        endpoint: str,
        file_path: str,
        object_name: str,
        body=None,
        method: str = "POST",
        storage_uri_key: str = "storageUri",
        presigned_url_endpoint: str = "storage/presigned-url",
        presigned_url_query_params: str = "",
    ):
        """Generic method to transfer data to the openlayer folder and create the
        appropriate resource in the backend when using a local deployment.
        """
        presigned_json = self.post_request(
            (
                f"{presigned_url_endpoint}?objectName={object_name}"
                f"&{presigned_url_query_params}"
            )
        )
        blob_path = presigned_json["storageUri"].replace("local://", "")
        dir_path = os.path.dirname(blob_path)
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as exc:
            raise OpenlayerException(f"Directory {dir_path} cannot be created") from exc
        shutil.copyfile(file_path, blob_path)
        body[storage_uri_key] = presigned_json["storageUri"]
        if method == "POST":
            return self.post_request(f"{endpoint}", body=body)
        elif method == "PUT":
            return self.put_request(f"{endpoint}", body=body)
