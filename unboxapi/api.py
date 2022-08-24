import os
import shutil
from enum import Enum

import requests
from requests.adapters import HTTPAdapter, Response, Retry
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from .exceptions import ExceptionMap, UnboxException
from .version import __version__

# Parameters for HTTP retry
HTTP_TOTAL_RETRIES = 3  # Number of total retries
HTTP_RETRY_BACKOFF_FACTOR = 2  # Wait 1, 2, 4 seconds between retries
HTTP_STATUS_FORCE_LIST = [408, 429] + list(range(500, 504)) + list(range(506, 531))
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "PUT", "POST"})

CLIENT_METADATA = {"version": __version__}


class StorageType(Enum):
    """Storage options for uploads."""

    ONPREM = 1
    AWS = 2
    GCP = 3
    AZURE = 4


STORAGE = StorageType.AWS
UNBOX_ENDPOINT = "https://api.unbox.ai/v1"


class Api:
    """Internal class to handle http requests"""

    def __init__(self, api_key: str):
        if api_key == "" or api_key is None:
            raise UnboxException(
                "There is an issue instantiating the UnboxClient. \n"
                "An invalid API key is being provided. \n"
                "Make sure to provide a valid API key using the syntax "
                "`UnboxClient('YOUR_API_KEY_HERE')`. You can find your API keys "
                "in the Profile page on the Unbox platform."
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
    ) -> Response:

        https = requests.Session()
        retry_strategy = Retry(
            total=HTTP_TOTAL_RETRIES,
            backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
            status_forcelist=HTTP_STATUS_FORCE_LIST,
            method_whitelist=HTTP_RETRY_ALLOWED_METHODS,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        https.mount("https://", adapter)

        try:
            params = params or {}
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
            raise UnboxException(err) from err

    @staticmethod
    def _raise_on_respose(res: Response):

        try:
            message = res.json().get("error", res.text)
        except ValueError:
            message = res.text

        exception = ExceptionMap.get(res.status_code, UnboxException)
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
    ):
        """Make any HTTP request + error handling."""

        url = f"{UNBOX_ENDPOINT}/{endpoint}"

        res = self._http_request(method, url, headers, params, body, files, data)

        json = None
        if res.ok:
            json = res.json()
        else:
            self._raise_on_respose(res)

        return json

    def get_request(self, endpoint: str, params=None):
        """Generic GET Request Wrapper."""
        return self._api_request("GET", endpoint, headers=self._headers, params=params)

    def post_request(self, endpoint: str, body=None, files=None, data=None):
        """Generic POST Request Wrapper."""
        return self._api_request(
            "POST",
            endpoint,
            headers=self._headers
            if files is None
            else self._headers_multipart_form_data,
            body=body,
            files=files,
            data=data,
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

    def upload(self, endpoint: str, file_path: str, object_name: str = None, body=None):
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
        )

    def upload_blob_s3(
        self, endpoint: str, file_path: str, object_name: str = None, body=None
    ):
        """Generic method to upload data to S3 storage and create the appropriate resource
        in the backend.
        """
        params = {"storageInterface": "s3", "objectName": object_name}
        presigned_json = self.get_request(f"{endpoint}/presigned-url", params=params)
        with open(file_path, "rb") as f:
            with tqdm(
                total=os.stat(file_path).st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                files = {"file": (presigned_json["id"], wrapped_file)}
                res = requests.post(
                    presigned_json["url"], data=presigned_json["fields"], files=files
                )
        if res.ok:
            body["storageUri"] = presigned_json["storageUri"]
            return self.post_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def upload_blob_gcs(
        self, endpoint: str, file_path: str, object_name: str = None, body=None
    ):
        """Generic method to upload data to Google Cloud Storage and create the
        appropriate resource in the backend.
        """
        params = {"storageInterface": "gcs", "objectName": object_name}
        presigned_json = self.get_request(f"{endpoint}/presigned-url", params=params)
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
                )
        if res.ok:
            body["storageUri"] = presigned_json["storageUri"]
            return self.post_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def upload_blob_azure(
        self, endpoint: str, file_path: str, object_name: str = None, body=None
    ):
        """Generic method to upload data to Azure Blob Storage and create the
        appropriate resource in the backend.
        """
        params = {"storageInterface": "azure", "objectName": object_name}
        presigned_json = self.get_request(f"{endpoint}/presigned-url", params=params)
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
                )
        if res.ok:
            body["storageUri"] = presigned_json["storageUri"]
            return self.post_request(f"{endpoint}", body=body)
        else:
            self._raise_on_respose(res)

    def transfer_blob(self, endpoint: str, file_path: str, object_name: str, body=None):
        """Generic method to transfer data to the unbox folder and create the appropriate
        resource in the backend when using a local deployment.
        """
        params = {"storageInterface": "local", "objectName": object_name}
        presigned_json = self.get_request(f"{endpoint}/presigned-url", params=params)
        blob_path = presigned_json["storageUri"].replace("local://", "")
        try:
            os.makedirs(blob_path, exist_ok=True)
        except OSError:
            raise UnboxException(f"Directory {blob_path} cannot be created")
        shutil.copyfile(file_path, f"{blob_path}/{object_name}")
        body["storageUri"] = presigned_json["storageUri"]
        return self.post_request(f"{endpoint}", body=body)
