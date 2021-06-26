import os
import shutil
import uuid

import requests
from requests.adapters import HTTPAdapter, Response, Retry
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from .exceptions import ExceptionMap, UnboxException

UNBOX_ENDPOINT = "https://dev.unbox.ai/api"
UNBOX_STORAGE_PATH = "~/.unbox/storage"


# Parameters for HTTP retry
HTTP_TOTAL_RETRIES = 3  # Number of total retries
HTTP_RETRY_BACKOFF_FACTOR = 2  # Wait 1, 2, 4 seconds between retries
HTTP_STATUS_FORCE_LIST = [408, 429] + list(range(500, 531))
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "POST"})


class Api:
    """ Internal class to handle http requests """

    def __init__(self, api_key: str):
        if api_key == "" or api_key is None:
            raise UnboxException("Must provide a valid API Key.")

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
            body = body or {}

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
        """Make any HTTP request + error handling. """

        url = f"{UNBOX_ENDPOINT}/{endpoint}"

        res = self._http_request(method, url, headers, params, body, files, data)

        json = None
        if res.ok:
            json = res.json()
        else:
            self._raise_on_respose(res)

        return json

    def get_request(self, endpoint: str, params=None):
        """Generic GET Request Wrapper. """
        return self._api_request("GET", endpoint, headers=self._headers, params=params)

    def post_request(self, endpoint: str, body=None, files=None, data=None):
        """Generic POST Request Wrapper. """
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

    def upload_blob(self, endpoint: str, file_path: str, body=None):
        """Generic method to upload data to S3 storage and create the appropriate resource
        in the backend.
        """
        presigned_json = self.get_request(endpoint)
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
            return self.post_request(f"{endpoint}/{presigned_json['id']}", body=body)
        else:
            self._raise_on_respose(res)

    def transfer_blob(self, endpoint: str, file_path: str, body=None):
        """Generic method to transfer data to the unbox folder and create the appropriate
        resource in the backend when using a local deployment.
        """
        id = uuid.uuid4()
        try:
            os.makedirs(f"{UNBOX_STORAGE_PATH}/{endpoint}", exist_ok=True)
        except OSError as _:
            raise UnboxException(
                f"Directory {UNBOX_STORAGE_PATH}/{endpoint} cannot be created"
            )
        shutil.copyfile(file_path, f"{UNBOX_STORAGE_PATH}/{endpoint}/{id}")
        return self.post_request(f"{endpoint}/{id}", body=body)