# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping
from typing_extensions import Self, override

import httpx

from . import _exceptions
from ._qs import Querystring
from ._types import (
    Omit,
    Headers,
    Timeout,
    NotGiven,
    Transport,
    ProxiesTypes,
    RequestOptions,
    not_given,
)
from ._utils import is_given, get_async_library
from ._compat import cached_property
from ._version import __version__
from ._streaming import Stream as Stream, AsyncStream as AsyncStream
from ._exceptions import APIStatusError
from ._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
    AsyncAPIClient,
)

if TYPE_CHECKING:
    from .resources import tests, commits, storage, projects, workspaces, inference_pipelines
    from .resources.tests import TestsResource, AsyncTestsResource
    from .resources.commits.commits import CommitsResource, AsyncCommitsResource
    from .resources.storage.storage import StorageResource, AsyncStorageResource
    from .resources.projects.projects import ProjectsResource, AsyncProjectsResource
    from .resources.workspaces.workspaces import WorkspacesResource, AsyncWorkspacesResource
    from .resources.inference_pipelines.inference_pipelines import (
        InferencePipelinesResource,
        AsyncInferencePipelinesResource,
    )

__all__ = [
    "Timeout",
    "Transport",
    "ProxiesTypes",
    "RequestOptions",
    "Openlayer",
    "AsyncOpenlayer",
    "Client",
    "AsyncClient",
]


class Openlayer(SyncAPIClient):
    # client options
    api_key: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
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
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new synchronous Openlayer client instance.

        This automatically infers the `api_key` argument from the `OPENLAYER_API_KEY` environment variable if it is not provided.
        """
        if api_key is None:
            api_key = os.environ.get("OPENLAYER_API_KEY")
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("OPENLAYER_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openlayer.com/v1"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

    @cached_property
    def projects(self) -> ProjectsResource:
        from .resources.projects import ProjectsResource

        return ProjectsResource(self)

    @cached_property
    def workspaces(self) -> WorkspacesResource:
        from .resources.workspaces import WorkspacesResource

        return WorkspacesResource(self)

    @cached_property
    def commits(self) -> CommitsResource:
        from .resources.commits import CommitsResource

        return CommitsResource(self)

    @cached_property
    def inference_pipelines(self) -> InferencePipelinesResource:
        from .resources.inference_pipelines import InferencePipelinesResource

        return InferencePipelinesResource(self)

    @cached_property
    def storage(self) -> StorageResource:
        from .resources.storage import StorageResource

        return StorageResource(self)

    @cached_property
    def tests(self) -> TestsResource:
        from .resources.tests import TestsResource

        return TestsResource(self)

    @cached_property
    def with_raw_response(self) -> OpenlayerWithRawResponse:
        return OpenlayerWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> OpenlayerWithStreamedResponse:
        return OpenlayerWithStreamedResponse(self)

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
        return {"Authorization": f"Bearer {api_key}"}

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
        if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
            return

        raise TypeError(
            '"Could not resolve authentication method. Expected the api_key to be set. Or for the `Authorization` headers to be explicitly omitted"'
        )

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        http_client: httpx.Client | None = None,
        max_retries: int | NotGiven = not_given,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

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
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
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
    # client options
    api_key: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
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
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new async AsyncOpenlayer client instance.

        This automatically infers the `api_key` argument from the `OPENLAYER_API_KEY` environment variable if it is not provided.
        """
        if api_key is None:
            api_key = os.environ.get("OPENLAYER_API_KEY")
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("OPENLAYER_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openlayer.com/v1"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

    @cached_property
    def projects(self) -> AsyncProjectsResource:
        from .resources.projects import AsyncProjectsResource

        return AsyncProjectsResource(self)

    @cached_property
    def workspaces(self) -> AsyncWorkspacesResource:
        from .resources.workspaces import AsyncWorkspacesResource

        return AsyncWorkspacesResource(self)

    @cached_property
    def commits(self) -> AsyncCommitsResource:
        from .resources.commits import AsyncCommitsResource

        return AsyncCommitsResource(self)

    @cached_property
    def inference_pipelines(self) -> AsyncInferencePipelinesResource:
        from .resources.inference_pipelines import AsyncInferencePipelinesResource

        return AsyncInferencePipelinesResource(self)

    @cached_property
    def storage(self) -> AsyncStorageResource:
        from .resources.storage import AsyncStorageResource

        return AsyncStorageResource(self)

    @cached_property
    def tests(self) -> AsyncTestsResource:
        from .resources.tests import AsyncTestsResource

        return AsyncTestsResource(self)

    @cached_property
    def with_raw_response(self) -> AsyncOpenlayerWithRawResponse:
        return AsyncOpenlayerWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncOpenlayerWithStreamedResponse:
        return AsyncOpenlayerWithStreamedResponse(self)

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
        return {"Authorization": f"Bearer {api_key}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": f"async:{get_async_library()}",
            **self._custom_headers,
        }

    @override
    def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
        if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
            return

        raise TypeError(
            '"Could not resolve authentication method. Expected the api_key to be set. Or for the `Authorization` headers to be explicitly omitted"'
        )

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | NotGiven = not_given,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

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
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
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
    _client: Openlayer

    def __init__(self, client: Openlayer) -> None:
        self._client = client

    @cached_property
    def projects(self) -> projects.ProjectsResourceWithRawResponse:
        from .resources.projects import ProjectsResourceWithRawResponse

        return ProjectsResourceWithRawResponse(self._client.projects)

    @cached_property
    def workspaces(self) -> workspaces.WorkspacesResourceWithRawResponse:
        from .resources.workspaces import WorkspacesResourceWithRawResponse

        return WorkspacesResourceWithRawResponse(self._client.workspaces)

    @cached_property
    def commits(self) -> commits.CommitsResourceWithRawResponse:
        from .resources.commits import CommitsResourceWithRawResponse

        return CommitsResourceWithRawResponse(self._client.commits)

    @cached_property
    def inference_pipelines(self) -> inference_pipelines.InferencePipelinesResourceWithRawResponse:
        from .resources.inference_pipelines import InferencePipelinesResourceWithRawResponse

        return InferencePipelinesResourceWithRawResponse(self._client.inference_pipelines)

    @cached_property
    def storage(self) -> storage.StorageResourceWithRawResponse:
        from .resources.storage import StorageResourceWithRawResponse

        return StorageResourceWithRawResponse(self._client.storage)

    @cached_property
    def tests(self) -> tests.TestsResourceWithRawResponse:
        from .resources.tests import TestsResourceWithRawResponse

        return TestsResourceWithRawResponse(self._client.tests)


class AsyncOpenlayerWithRawResponse:
    _client: AsyncOpenlayer

    def __init__(self, client: AsyncOpenlayer) -> None:
        self._client = client

    @cached_property
    def projects(self) -> projects.AsyncProjectsResourceWithRawResponse:
        from .resources.projects import AsyncProjectsResourceWithRawResponse

        return AsyncProjectsResourceWithRawResponse(self._client.projects)

    @cached_property
    def workspaces(self) -> workspaces.AsyncWorkspacesResourceWithRawResponse:
        from .resources.workspaces import AsyncWorkspacesResourceWithRawResponse

        return AsyncWorkspacesResourceWithRawResponse(self._client.workspaces)

    @cached_property
    def commits(self) -> commits.AsyncCommitsResourceWithRawResponse:
        from .resources.commits import AsyncCommitsResourceWithRawResponse

        return AsyncCommitsResourceWithRawResponse(self._client.commits)

    @cached_property
    def inference_pipelines(self) -> inference_pipelines.AsyncInferencePipelinesResourceWithRawResponse:
        from .resources.inference_pipelines import AsyncInferencePipelinesResourceWithRawResponse

        return AsyncInferencePipelinesResourceWithRawResponse(self._client.inference_pipelines)

    @cached_property
    def storage(self) -> storage.AsyncStorageResourceWithRawResponse:
        from .resources.storage import AsyncStorageResourceWithRawResponse

        return AsyncStorageResourceWithRawResponse(self._client.storage)

    @cached_property
    def tests(self) -> tests.AsyncTestsResourceWithRawResponse:
        from .resources.tests import AsyncTestsResourceWithRawResponse

        return AsyncTestsResourceWithRawResponse(self._client.tests)


class OpenlayerWithStreamedResponse:
    _client: Openlayer

    def __init__(self, client: Openlayer) -> None:
        self._client = client

    @cached_property
    def projects(self) -> projects.ProjectsResourceWithStreamingResponse:
        from .resources.projects import ProjectsResourceWithStreamingResponse

        return ProjectsResourceWithStreamingResponse(self._client.projects)

    @cached_property
    def workspaces(self) -> workspaces.WorkspacesResourceWithStreamingResponse:
        from .resources.workspaces import WorkspacesResourceWithStreamingResponse

        return WorkspacesResourceWithStreamingResponse(self._client.workspaces)

    @cached_property
    def commits(self) -> commits.CommitsResourceWithStreamingResponse:
        from .resources.commits import CommitsResourceWithStreamingResponse

        return CommitsResourceWithStreamingResponse(self._client.commits)

    @cached_property
    def inference_pipelines(self) -> inference_pipelines.InferencePipelinesResourceWithStreamingResponse:
        from .resources.inference_pipelines import InferencePipelinesResourceWithStreamingResponse

        return InferencePipelinesResourceWithStreamingResponse(self._client.inference_pipelines)

    @cached_property
    def storage(self) -> storage.StorageResourceWithStreamingResponse:
        from .resources.storage import StorageResourceWithStreamingResponse

        return StorageResourceWithStreamingResponse(self._client.storage)

    @cached_property
    def tests(self) -> tests.TestsResourceWithStreamingResponse:
        from .resources.tests import TestsResourceWithStreamingResponse

        return TestsResourceWithStreamingResponse(self._client.tests)


class AsyncOpenlayerWithStreamedResponse:
    _client: AsyncOpenlayer

    def __init__(self, client: AsyncOpenlayer) -> None:
        self._client = client

    @cached_property
    def projects(self) -> projects.AsyncProjectsResourceWithStreamingResponse:
        from .resources.projects import AsyncProjectsResourceWithStreamingResponse

        return AsyncProjectsResourceWithStreamingResponse(self._client.projects)

    @cached_property
    def workspaces(self) -> workspaces.AsyncWorkspacesResourceWithStreamingResponse:
        from .resources.workspaces import AsyncWorkspacesResourceWithStreamingResponse

        return AsyncWorkspacesResourceWithStreamingResponse(self._client.workspaces)

    @cached_property
    def commits(self) -> commits.AsyncCommitsResourceWithStreamingResponse:
        from .resources.commits import AsyncCommitsResourceWithStreamingResponse

        return AsyncCommitsResourceWithStreamingResponse(self._client.commits)

    @cached_property
    def inference_pipelines(self) -> inference_pipelines.AsyncInferencePipelinesResourceWithStreamingResponse:
        from .resources.inference_pipelines import AsyncInferencePipelinesResourceWithStreamingResponse

        return AsyncInferencePipelinesResourceWithStreamingResponse(self._client.inference_pipelines)

    @cached_property
    def storage(self) -> storage.AsyncStorageResourceWithStreamingResponse:
        from .resources.storage import AsyncStorageResourceWithStreamingResponse

        return AsyncStorageResourceWithStreamingResponse(self._client.storage)

    @cached_property
    def tests(self) -> tests.AsyncTestsResourceWithStreamingResponse:
        from .resources.tests import AsyncTestsResourceWithStreamingResponse

        return AsyncTestsResourceWithStreamingResponse(self._client.tests)


Client = Openlayer

AsyncClient = AsyncOpenlayer
