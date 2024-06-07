# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

import os

from ._streaming import AsyncStream as AsyncStream, Stream as Stream

from typing_extensions import override, Self

from typing import Any

from ._exceptions import APIStatusError

from ._utils import get_async_library

from . import _exceptions

import os
import asyncio
import warnings
from typing import Optional, Union, Dict, Any, Mapping, overload, cast
from typing_extensions import Literal

import httpx

from ._version import __version__
from ._qs import Querystring
from .types import shared_params
from ._utils import extract_files, maybe_transform, required_args, deepcopy_minimal, maybe_coerce_integer, maybe_coerce_float, maybe_coerce_boolean, is_given
from ._types import Omit, NotGiven, Timeout, Transport, ProxiesTypes, RequestOptions, Headers, NoneType, Query, Body, NOT_GIVEN
from ._base_client import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    ResponseT,
    SyncHttpxClientWrapper,
    AsyncHttpxClientWrapper,
    SyncAPIClient,
    AsyncAPIClient,
    make_request_options,
)
from . import resources

__all__ = ["Timeout", "Transport", "ProxiesTypes", "RequestOptions", "resources", "Openlayer", "AsyncOpenlayer", "Client", "AsyncClient"]

class Openlayer(SyncAPIClient):
    projects: resources.ProjectsResource
    commits: resources.CommitsResource
    inference_pipelines: resources.InferencePipelinesResource
    with_raw_response: OpenlayerWithRawResponse
    with_streaming_response: OpenlayerWithStreamedResponse

    # client options
    api_key: str | None
    base_url: str | None

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, base_url: str | httpx.URL | None = None, timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN, max_retries: int = DEFAULT_MAX_RETRIES, default_headers: Mapping[str, str] | None = None, default_query: Mapping[str, object] | None = None, 
    # Configure a custom httpx client.
    # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
    # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: httpx.Client | None = None, 
    # Enable or disable schema validation for data returned by the API.
    # When enabled an error APIResponseValidationError is raised
    # if the API responds with invalid data for the expected schema.
    # 
    # This parameter may be removed or changed in the future.
    # If you rely on this feature, please open a GitHub issue
    # outlining your use-case to help us decide if it should be
    # part of our public interface in the future.
    _strict_response_validation: bool = False) -> None:
        """Construct a new synchronous openlayer client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENLAYER_API_KEY`
        - `base_url` from `OPENLAYER_SERVER_URL`
        """
        if api_key is None:
          api_key = os.environ.get("OPENLAYER_API_KEY")
        self.api_key = api_key

        if base_url is None:
          base_url = os.environ.get("OPENLAYER_SERVER_URL")
        self.base_url = base_url

        if base_url is None:
          base_url = os.environ.get("OPENLAYER_BASE_URL")
        if base_url is None:
          base_url = f"https://api.openlayer.com/v1"

        super().__init__(version=__version__, base_url=base_url, max_retries=max_retries, timeout=timeout, http_client=http_client, custom_headers=default_headers, custom_query=default_query, _strict_response_validation=_strict_response_validation)

        self.projects = resources.ProjectsResource(self)
        self.commits = resources.CommitsResource(self)
        self.inference_pipelines = resources.InferencePipelinesResource(self)
        self.with_raw_response = OpenlayerWithRawResponse(self)
        self.with_streaming_response = OpenlayerWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        if api_key is None:
            return {}
        return {
            "Authorization": f"Bearer {api_key}"
        }

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
          **super().default_headers,
          "X-Stainless-Async": "false",
          **self._custom_headers,
        }

    @override
    def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
        if self.api_key and headers.get("Authorization"):
            return
        if isinstance(custom_headers.get("Authorization"), Omit):
            return

        raise TypeError("\"Could not resolve authentication method. Expected the api_key to be set. Or for the `Authorization` headers to be explicitly omitted\"")

    def copy(self, *, api_key: str | None = None, base_url: str | None = None, base_url: str | httpx.URL | None = None, timeout: float | Timeout | None | NotGiven = NOT_GIVEN, http_client: httpx.Client | None = None, max_retries: int | NotGiven = NOT_GIVEN, default_headers: Mapping[str, str] | None = None, set_default_headers: Mapping[str, str] | None = None, default_query: Mapping[str, object] | None = None, set_default_query: Mapping[str, object] | None = None, _extra_kwargs: Mapping[str, Any] = {}) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
          raise ValueError(
            'The `default_headers` and `set_default_headers` arguments are mutually exclusive'
          )

        if default_query is not None and set_default_query is not None:
          raise ValueError(
            'The `default_query` and `set_default_query` arguments are mutually exclusive'
          )

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(api_key = api_key or self.api_key, base_url = base_url or self.base_url, base_url=base_url or self.base_url, timeout=self.timeout if isinstance(timeout, NotGiven) else timeout, http_client=http_client, max_retries=max_retries if is_given(max_retries) else self.max_retries, default_headers=headers, default_query=params, **_extra_kwargs)

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(self, err_msg: str, *, body: object, response: httpx.Response,) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)

class AsyncOpenlayer(AsyncAPIClient):
    projects: resources.AsyncProjectsResource
    commits: resources.AsyncCommitsResource
    inference_pipelines: resources.AsyncInferencePipelinesResource
    with_raw_response: AsyncOpenlayerWithRawResponse
    with_streaming_response: AsyncOpenlayerWithStreamedResponse

    # client options
    api_key: str | None
    base_url: str | None

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, base_url: str | httpx.URL | None = None, timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN, max_retries: int = DEFAULT_MAX_RETRIES, default_headers: Mapping[str, str] | None = None, default_query: Mapping[str, object] | None = None, 
    # Configure a custom httpx client.
    # We provide a `DefaultAsyncHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
    # See the [httpx documentation](https://www.python-httpx.org/api/#asyncclient) for more details.
    http_client: httpx.AsyncClient | None = None, 
    # Enable or disable schema validation for data returned by the API.
    # When enabled an error APIResponseValidationError is raised
    # if the API responds with invalid data for the expected schema.
    # 
    # This parameter may be removed or changed in the future.
    # If you rely on this feature, please open a GitHub issue
    # outlining your use-case to help us decide if it should be
    # part of our public interface in the future.
    _strict_response_validation: bool = False) -> None:
        """Construct a new async openlayer client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENLAYER_API_KEY`
        - `base_url` from `OPENLAYER_SERVER_URL`
        """
        if api_key is None:
          api_key = os.environ.get("OPENLAYER_API_KEY")
        self.api_key = api_key

        if base_url is None:
          base_url = os.environ.get("OPENLAYER_SERVER_URL")
        self.base_url = base_url

        if base_url is None:
          base_url = os.environ.get("OPENLAYER_BASE_URL")
        if base_url is None:
          base_url = f"https://api.openlayer.com/v1"

        super().__init__(version=__version__, base_url=base_url, max_retries=max_retries, timeout=timeout, http_client=http_client, custom_headers=default_headers, custom_query=default_query, _strict_response_validation=_strict_response_validation)

        self.projects = resources.AsyncProjectsResource(self)
        self.commits = resources.AsyncCommitsResource(self)
        self.inference_pipelines = resources.AsyncInferencePipelinesResource(self)
        self.with_raw_response = AsyncOpenlayerWithRawResponse(self)
        self.with_streaming_response = AsyncOpenlayerWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        if api_key is None:
            return {}
        return {
            "Authorization": f"Bearer {api_key}"
        }

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
          **super().default_headers,
          "X-Stainless-Async": f'async:{get_async_library()}',
          **self._custom_headers,
        }

    @override
    def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
        if self.api_key and headers.get("Authorization"):
            return
        if isinstance(custom_headers.get("Authorization"), Omit):
            return

        raise TypeError("\"Could not resolve authentication method. Expected the api_key to be set. Or for the `Authorization` headers to be explicitly omitted\"")

    def copy(self, *, api_key: str | None = None, base_url: str | None = None, base_url: str | httpx.URL | None = None, timeout: float | Timeout | None | NotGiven = NOT_GIVEN, http_client: httpx.AsyncClient | None = None, max_retries: int | NotGiven = NOT_GIVEN, default_headers: Mapping[str, str] | None = None, set_default_headers: Mapping[str, str] | None = None, default_query: Mapping[str, object] | None = None, set_default_query: Mapping[str, object] | None = None, _extra_kwargs: Mapping[str, Any] = {}) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
          raise ValueError(
            'The `default_headers` and `set_default_headers` arguments are mutually exclusive'
          )

        if default_query is not None and set_default_query is not None:
          raise ValueError(
            'The `default_query` and `set_default_query` arguments are mutually exclusive'
          )

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(api_key = api_key or self.api_key, base_url = base_url or self.base_url, base_url=base_url or self.base_url, timeout=self.timeout if isinstance(timeout, NotGiven) else timeout, http_client=http_client, max_retries=max_retries if is_given(max_retries) else self.max_retries, default_headers=headers, default_query=params, **_extra_kwargs)

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(self, err_msg: str, *, body: object, response: httpx.Response,) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)

class OpenlayerWithRawResponse:
    def __init__(self, client: Openlayer) -> None:
        self.projects = resources.ProjectsResourceWithRawResponse(client.projects)
        self.commits = resources.CommitsResourceWithRawResponse(client.commits)
        self.inference_pipelines = resources.InferencePipelinesResourceWithRawResponse(client.inference_pipelines)

class AsyncOpenlayerWithRawResponse:
    def __init__(self, client: AsyncOpenlayer) -> None:
        self.projects = resources.AsyncProjectsResourceWithRawResponse(client.projects)
        self.commits = resources.AsyncCommitsResourceWithRawResponse(client.commits)
        self.inference_pipelines = resources.AsyncInferencePipelinesResourceWithRawResponse(client.inference_pipelines)

class OpenlayerWithStreamedResponse:
    def __init__(self, client: Openlayer) -> None:
        self.projects = resources.ProjectsResourceWithStreamingResponse(client.projects)
        self.commits = resources.CommitsResourceWithStreamingResponse(client.commits)
        self.inference_pipelines = resources.InferencePipelinesResourceWithStreamingResponse(client.inference_pipelines)

class AsyncOpenlayerWithStreamedResponse:
    def __init__(self, client: AsyncOpenlayer) -> None:
        self.projects = resources.AsyncProjectsResourceWithStreamingResponse(client.projects)
        self.commits = resources.AsyncCommitsResourceWithStreamingResponse(client.commits)
        self.inference_pipelines = resources.AsyncInferencePipelinesResourceWithStreamingResponse(client.inference_pipelines)

Client = Openlayer

AsyncClient = AsyncOpenlayer