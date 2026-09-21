# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from .._types import Body, Query, Headers, NotGiven, not_given
from .._utils import path_template
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.background_task_retrieve_response import BackgroundTaskRetrieveResponse

__all__ = ["BackgroundTasksResource", "AsyncBackgroundTasksResource"]


class BackgroundTasksResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> BackgroundTasksResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return BackgroundTasksResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> BackgroundTasksResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return BackgroundTasksResourceWithStreamingResponse(self)

    def retrieve(
        self,
        task_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> BackgroundTaskRetrieveResponse:
        """
        Retrieve a background task's status, progress and results.

        Endpoints that cannot answer within one request queue a task and hand back its
        id -- for example `POST /frameworks/{frameworkId}/export`. Poll this endpoint
        until `complete` is `true`, then read what the task produced from `outputs`.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not task_id:
            raise ValueError(f"Expected a non-empty value for `task_id` but received {task_id!r}")
        return self._get(
            path_template("/background-tasks/{task_id}", task_id=task_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=BackgroundTaskRetrieveResponse,
        )


class AsyncBackgroundTasksResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncBackgroundTasksResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncBackgroundTasksResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncBackgroundTasksResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncBackgroundTasksResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        task_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> BackgroundTaskRetrieveResponse:
        """
        Retrieve a background task's status, progress and results.

        Endpoints that cannot answer within one request queue a task and hand back its
        id -- for example `POST /frameworks/{frameworkId}/export`. Poll this endpoint
        until `complete` is `true`, then read what the task produced from `outputs`.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not task_id:
            raise ValueError(f"Expected a non-empty value for `task_id` but received {task_id!r}")
        return await self._get(
            path_template("/background-tasks/{task_id}", task_id=task_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=BackgroundTaskRetrieveResponse,
        )


class BackgroundTasksResourceWithRawResponse:
    def __init__(self, background_tasks: BackgroundTasksResource) -> None:
        self._background_tasks = background_tasks

        self.retrieve = to_raw_response_wrapper(
            background_tasks.retrieve,
        )


class AsyncBackgroundTasksResourceWithRawResponse:
    def __init__(self, background_tasks: AsyncBackgroundTasksResource) -> None:
        self._background_tasks = background_tasks

        self.retrieve = async_to_raw_response_wrapper(
            background_tasks.retrieve,
        )


class BackgroundTasksResourceWithStreamingResponse:
    def __init__(self, background_tasks: BackgroundTasksResource) -> None:
        self._background_tasks = background_tasks

        self.retrieve = to_streamed_response_wrapper(
            background_tasks.retrieve,
        )


class AsyncBackgroundTasksResourceWithStreamingResponse:
    def __init__(self, background_tasks: AsyncBackgroundTasksResource) -> None:
        self._background_tasks = background_tasks

        self.retrieve = async_to_streamed_response_wrapper(
            background_tasks.retrieve,
        )
