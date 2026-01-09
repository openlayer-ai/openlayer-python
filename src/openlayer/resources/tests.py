# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional

import httpx

from ..types import test_evaluate_params, test_list_results_params
from .._types import Body, Omit, Query, Headers, NotGiven, SequenceNotStr, omit, not_given
from .._utils import maybe_transform, async_maybe_transform
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.test_evaluate_response import TestEvaluateResponse
from ..types.test_list_results_response import TestListResultsResponse

__all__ = ["TestsResource", "AsyncTestsResource"]


class TestsResource(SyncAPIResource):
    __test__ = False

    @cached_property
    def with_raw_response(self) -> TestsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return TestsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TestsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return TestsResourceWithStreamingResponse(self)

    def evaluate(
        self,
        test_id: str,
        *,
        end_timestamp: int,
        start_timestamp: int,
        inference_pipeline_id: str | Omit = omit,
        overwrite_results: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TestEvaluateResponse:
        """
        Triggers one-off evaluation of a specific monitoring test for a custom timestamp
        range. This allows evaluating tests for historical data or custom time periods
        outside the regular evaluation window schedule. It also allows overwriting the
        existing test results.

        Args:
          end_timestamp: End timestamp in seconds (Unix epoch)

          start_timestamp: Start timestamp in seconds (Unix epoch)

          inference_pipeline_id: ID of the inference pipeline to evaluate. If not provided, all inference
              pipelines the test applies to will be evaluated.

          overwrite_results: Whether to overwrite existing test results

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not test_id:
            raise ValueError(f"Expected a non-empty value for `test_id` but received {test_id!r}")
        return self._post(
            f"/tests/{test_id}/evaluate",
            body=maybe_transform(
                {
                    "end_timestamp": end_timestamp,
                    "start_timestamp": start_timestamp,
                    "inference_pipeline_id": inference_pipeline_id,
                    "overwrite_results": overwrite_results,
                },
                test_evaluate_params.TestEvaluateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestEvaluateResponse,
        )

    def list_results(
        self,
        test_id: str,
        *,
        end_timestamp: float | Omit = omit,
        include_insights: bool | Omit = omit,
        inference_pipeline_id: Optional[str] | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_version_id: Optional[str] | Omit = omit,
        start_timestamp: float | Omit = omit,
        status: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TestListResultsResponse:
        """
        List the test results for a test.

        Args:
          end_timestamp: Filter for results that use data starting before the end timestamp.

          include_insights: Include the insights linked to each test result

          inference_pipeline_id: Retrive test results for a specific inference pipeline.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_version_id: Retrive test results for a specific project version.

          start_timestamp: Filter for results that use data ending after the start timestamp.

          status: Filter by status(es).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not test_id:
            raise ValueError(f"Expected a non-empty value for `test_id` but received {test_id!r}")
        return self._get(
            f"/tests/{test_id}/results",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "end_timestamp": end_timestamp,
                        "include_insights": include_insights,
                        "inference_pipeline_id": inference_pipeline_id,
                        "page": page,
                        "per_page": per_page,
                        "project_version_id": project_version_id,
                        "start_timestamp": start_timestamp,
                        "status": status,
                    },
                    test_list_results_params.TestListResultsParams,
                ),
            ),
            cast_to=TestListResultsResponse,
        )


class AsyncTestsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncTestsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncTestsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTestsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncTestsResourceWithStreamingResponse(self)

    async def evaluate(
        self,
        test_id: str,
        *,
        end_timestamp: int,
        start_timestamp: int,
        inference_pipeline_id: str | Omit = omit,
        overwrite_results: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TestEvaluateResponse:
        """
        Triggers one-off evaluation of a specific monitoring test for a custom timestamp
        range. This allows evaluating tests for historical data or custom time periods
        outside the regular evaluation window schedule. It also allows overwriting the
        existing test results.

        Args:
          end_timestamp: End timestamp in seconds (Unix epoch)

          start_timestamp: Start timestamp in seconds (Unix epoch)

          inference_pipeline_id: ID of the inference pipeline to evaluate. If not provided, all inference
              pipelines the test applies to will be evaluated.

          overwrite_results: Whether to overwrite existing test results

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not test_id:
            raise ValueError(f"Expected a non-empty value for `test_id` but received {test_id!r}")
        return await self._post(
            f"/tests/{test_id}/evaluate",
            body=await async_maybe_transform(
                {
                    "end_timestamp": end_timestamp,
                    "start_timestamp": start_timestamp,
                    "inference_pipeline_id": inference_pipeline_id,
                    "overwrite_results": overwrite_results,
                },
                test_evaluate_params.TestEvaluateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestEvaluateResponse,
        )

    async def list_results(
        self,
        test_id: str,
        *,
        end_timestamp: float | Omit = omit,
        include_insights: bool | Omit = omit,
        inference_pipeline_id: Optional[str] | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_version_id: Optional[str] | Omit = omit,
        start_timestamp: float | Omit = omit,
        status: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> TestListResultsResponse:
        """
        List the test results for a test.

        Args:
          end_timestamp: Filter for results that use data starting before the end timestamp.

          include_insights: Include the insights linked to each test result

          inference_pipeline_id: Retrive test results for a specific inference pipeline.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_version_id: Retrive test results for a specific project version.

          start_timestamp: Filter for results that use data ending after the start timestamp.

          status: Filter by status(es).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not test_id:
            raise ValueError(f"Expected a non-empty value for `test_id` but received {test_id!r}")
        return await self._get(
            f"/tests/{test_id}/results",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "end_timestamp": end_timestamp,
                        "include_insights": include_insights,
                        "inference_pipeline_id": inference_pipeline_id,
                        "page": page,
                        "per_page": per_page,
                        "project_version_id": project_version_id,
                        "start_timestamp": start_timestamp,
                        "status": status,
                    },
                    test_list_results_params.TestListResultsParams,
                ),
            ),
            cast_to=TestListResultsResponse,
        )


class TestsResourceWithRawResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.evaluate = to_raw_response_wrapper(
            tests.evaluate,
        )
        self.list_results = to_raw_response_wrapper(
            tests.list_results,
        )


class AsyncTestsResourceWithRawResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.evaluate = async_to_raw_response_wrapper(
            tests.evaluate,
        )
        self.list_results = async_to_raw_response_wrapper(
            tests.list_results,
        )


class TestsResourceWithStreamingResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.evaluate = to_streamed_response_wrapper(
            tests.evaluate,
        )
        self.list_results = to_streamed_response_wrapper(
            tests.list_results,
        )


class AsyncTestsResourceWithStreamingResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.evaluate = async_to_streamed_response_wrapper(
            tests.evaluate,
        )
        self.list_results = async_to_streamed_response_wrapper(
            tests.list_results,
        )
