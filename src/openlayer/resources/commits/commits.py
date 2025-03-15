# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .test_results import (
    TestResultsResource,
    AsyncTestResultsResource,
    TestResultsResourceWithRawResponse,
    AsyncTestResultsResourceWithRawResponse,
    TestResultsResourceWithStreamingResponse,
    AsyncTestResultsResourceWithStreamingResponse,
)
from ..._base_client import make_request_options
from ...types.commit_retrieve_response import CommitRetrieveResponse

__all__ = ["CommitsResource", "AsyncCommitsResource"]


class CommitsResource(SyncAPIResource):
    @cached_property
    def test_results(self) -> TestResultsResource:
        return TestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> CommitsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return CommitsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> CommitsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return CommitsResourceWithStreamingResponse(self)

    def retrieve(
        self,
        project_version_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CommitRetrieveResponse:
        """
        Retrieve a project version (commit) by its id.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_version_id:
            raise ValueError(f"Expected a non-empty value for `project_version_id` but received {project_version_id!r}")
        return self._get(
            f"/versions/{project_version_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=CommitRetrieveResponse,
        )


class AsyncCommitsResource(AsyncAPIResource):
    @cached_property
    def test_results(self) -> AsyncTestResultsResource:
        return AsyncTestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncCommitsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncCommitsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncCommitsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncCommitsResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        project_version_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CommitRetrieveResponse:
        """
        Retrieve a project version (commit) by its id.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_version_id:
            raise ValueError(f"Expected a non-empty value for `project_version_id` but received {project_version_id!r}")
        return await self._get(
            f"/versions/{project_version_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=CommitRetrieveResponse,
        )


class CommitsResourceWithRawResponse:
    def __init__(self, commits: CommitsResource) -> None:
        self._commits = commits

        self.retrieve = to_raw_response_wrapper(
            commits.retrieve,
        )

    @cached_property
    def test_results(self) -> TestResultsResourceWithRawResponse:
        return TestResultsResourceWithRawResponse(self._commits.test_results)


class AsyncCommitsResourceWithRawResponse:
    def __init__(self, commits: AsyncCommitsResource) -> None:
        self._commits = commits

        self.retrieve = async_to_raw_response_wrapper(
            commits.retrieve,
        )

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithRawResponse:
        return AsyncTestResultsResourceWithRawResponse(self._commits.test_results)


class CommitsResourceWithStreamingResponse:
    def __init__(self, commits: CommitsResource) -> None:
        self._commits = commits

        self.retrieve = to_streamed_response_wrapper(
            commits.retrieve,
        )

    @cached_property
    def test_results(self) -> TestResultsResourceWithStreamingResponse:
        return TestResultsResourceWithStreamingResponse(self._commits.test_results)


class AsyncCommitsResourceWithStreamingResponse:
    def __init__(self, commits: AsyncCommitsResource) -> None:
        self._commits = commits

        self.retrieve = async_to_streamed_response_wrapper(
            commits.retrieve,
        )

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithStreamingResponse:
        return AsyncTestResultsResourceWithStreamingResponse(self._commits.test_results)
