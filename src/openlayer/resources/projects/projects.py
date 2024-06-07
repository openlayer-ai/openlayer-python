# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ...types import project_list_params
from .commits import (
    CommitsResource,
    AsyncCommitsResource,
    CommitsResourceWithRawResponse,
    AsyncCommitsResourceWithRawResponse,
    CommitsResourceWithStreamingResponse,
    AsyncCommitsResourceWithStreamingResponse,
)
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
    maybe_transform,
    async_maybe_transform,
)
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import (
    make_request_options,
)
from .inference_pipelines import (
    InferencePipelinesResource,
    AsyncInferencePipelinesResource,
    InferencePipelinesResourceWithRawResponse,
    AsyncInferencePipelinesResourceWithRawResponse,
    InferencePipelinesResourceWithStreamingResponse,
    AsyncInferencePipelinesResourceWithStreamingResponse,
)
from ...types.project_list_response import ProjectListResponse

__all__ = ["ProjectsResource", "AsyncProjectsResource"]


class ProjectsResource(SyncAPIResource):
    @cached_property
    def commits(self) -> CommitsResource:
        return CommitsResource(self._client)

    @cached_property
    def inference_pipelines(self) -> InferencePipelinesResource:
        return InferencePipelinesResource(self._client)

    @cached_property
    def with_raw_response(self) -> ProjectsResourceWithRawResponse:
        return ProjectsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ProjectsResourceWithStreamingResponse:
        return ProjectsResourceWithStreamingResponse(self)

    def list(
        self,
        *,
        name: str | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        task_type: Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"]
        | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ProjectListResponse:
        """
        List the projects in a user's workspace.

        Args:
          name: Filter list of items by project name.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          task_type: Filter list of items by task type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get(
            "/projects",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "name": name,
                        "page": page,
                        "per_page": per_page,
                        "task_type": task_type,
                    },
                    project_list_params.ProjectListParams,
                ),
            ),
            cast_to=ProjectListResponse,
        )


class AsyncProjectsResource(AsyncAPIResource):
    @cached_property
    def commits(self) -> AsyncCommitsResource:
        return AsyncCommitsResource(self._client)

    @cached_property
    def inference_pipelines(self) -> AsyncInferencePipelinesResource:
        return AsyncInferencePipelinesResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncProjectsResourceWithRawResponse:
        return AsyncProjectsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncProjectsResourceWithStreamingResponse:
        return AsyncProjectsResourceWithStreamingResponse(self)

    async def list(
        self,
        *,
        name: str | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        task_type: Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"]
        | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ProjectListResponse:
        """
        List the projects in a user's workspace.

        Args:
          name: Filter list of items by project name.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          task_type: Filter list of items by task type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(
            "/projects",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "name": name,
                        "page": page,
                        "per_page": per_page,
                        "task_type": task_type,
                    },
                    project_list_params.ProjectListParams,
                ),
            ),
            cast_to=ProjectListResponse,
        )


class ProjectsResourceWithRawResponse:
    def __init__(self, projects: ProjectsResource) -> None:
        self._projects = projects

        self.list = to_raw_response_wrapper(
            projects.list,
        )

    @cached_property
    def commits(self) -> CommitsResourceWithRawResponse:
        return CommitsResourceWithRawResponse(self._projects.commits)

    @cached_property
    def inference_pipelines(self) -> InferencePipelinesResourceWithRawResponse:
        return InferencePipelinesResourceWithRawResponse(self._projects.inference_pipelines)


class AsyncProjectsResourceWithRawResponse:
    def __init__(self, projects: AsyncProjectsResource) -> None:
        self._projects = projects

        self.list = async_to_raw_response_wrapper(
            projects.list,
        )

    @cached_property
    def commits(self) -> AsyncCommitsResourceWithRawResponse:
        return AsyncCommitsResourceWithRawResponse(self._projects.commits)

    @cached_property
    def inference_pipelines(self) -> AsyncInferencePipelinesResourceWithRawResponse:
        return AsyncInferencePipelinesResourceWithRawResponse(self._projects.inference_pipelines)


class ProjectsResourceWithStreamingResponse:
    def __init__(self, projects: ProjectsResource) -> None:
        self._projects = projects

        self.list = to_streamed_response_wrapper(
            projects.list,
        )

    @cached_property
    def commits(self) -> CommitsResourceWithStreamingResponse:
        return CommitsResourceWithStreamingResponse(self._projects.commits)

    @cached_property
    def inference_pipelines(self) -> InferencePipelinesResourceWithStreamingResponse:
        return InferencePipelinesResourceWithStreamingResponse(self._projects.inference_pipelines)


class AsyncProjectsResourceWithStreamingResponse:
    def __init__(self, projects: AsyncProjectsResource) -> None:
        self._projects = projects

        self.list = async_to_streamed_response_wrapper(
            projects.list,
        )

    @cached_property
    def commits(self) -> AsyncCommitsResourceWithStreamingResponse:
        return AsyncCommitsResourceWithStreamingResponse(self._projects.commits)

    @cached_property
    def inference_pipelines(self) -> AsyncInferencePipelinesResourceWithStreamingResponse:
        return AsyncInferencePipelinesResourceWithStreamingResponse(self._projects.inference_pipelines)
