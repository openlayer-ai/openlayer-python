# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from ...types.commits import test_result_list_params
from ...types.commits.test_result_list_response import TestResultListResponse

__all__ = ["TestResultsResource", "AsyncTestResultsResource"]


class TestResultsResource(SyncAPIResource):
    __test__ = False

    @cached_property
    def with_raw_response(self) -> TestResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return TestResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TestResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return TestResultsResourceWithStreamingResponse(self)

    def list(
        self,
        project_version_id: str,
        *,
        include_archived: bool | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        status: Literal["running", "passing", "failing", "skipped", "error"] | NotGiven = NOT_GIVEN,
        type: Literal["integrity", "consistency", "performance", "fairness", "robustness"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestResultListResponse:
        """
        List the test results for a project commit (project version).

        Args:
          include_archived: Filter for archived tests.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          status: Filter list of test results by status. Available statuses are `running`,
              `passing`, `failing`, `skipped`, and `error`.

          type: Filter objects by test type. Available types are `integrity`, `consistency`,
              `performance`, `fairness`, and `robustness`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_version_id:
            raise ValueError(f"Expected a non-empty value for `project_version_id` but received {project_version_id!r}")
        return self._get(
            f"/versions/{project_version_id}/results",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "include_archived": include_archived,
                        "page": page,
                        "per_page": per_page,
                        "status": status,
                        "type": type,
                    },
                    test_result_list_params.TestResultListParams,
                ),
            ),
            cast_to=TestResultListResponse,
        )


class AsyncTestResultsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncTestResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncTestResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTestResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncTestResultsResourceWithStreamingResponse(self)

    async def list(
        self,
        project_version_id: str,
        *,
        include_archived: bool | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        status: Literal["running", "passing", "failing", "skipped", "error"] | NotGiven = NOT_GIVEN,
        type: Literal["integrity", "consistency", "performance", "fairness", "robustness"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestResultListResponse:
        """
        List the test results for a project commit (project version).

        Args:
          include_archived: Filter for archived tests.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          status: Filter list of test results by status. Available statuses are `running`,
              `passing`, `failing`, `skipped`, and `error`.

          type: Filter objects by test type. Available types are `integrity`, `consistency`,
              `performance`, `fairness`, and `robustness`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_version_id:
            raise ValueError(f"Expected a non-empty value for `project_version_id` but received {project_version_id!r}")
        return await self._get(
            f"/versions/{project_version_id}/results",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "include_archived": include_archived,
                        "page": page,
                        "per_page": per_page,
                        "status": status,
                        "type": type,
                    },
                    test_result_list_params.TestResultListParams,
                ),
            ),
            cast_to=TestResultListResponse,
        )


class TestResultsResourceWithRawResponse:
    __test__ = False

    def __init__(self, test_results: TestResultsResource) -> None:
        self._test_results = test_results

        self.list = to_raw_response_wrapper(
            test_results.list,
        )


class AsyncTestResultsResourceWithRawResponse:
    def __init__(self, test_results: AsyncTestResultsResource) -> None:
        self._test_results = test_results

        self.list = async_to_raw_response_wrapper(
            test_results.list,
        )


class TestResultsResourceWithStreamingResponse:
    __test__ = False

    def __init__(self, test_results: TestResultsResource) -> None:
        self._test_results = test_results

        self.list = to_streamed_response_wrapper(
            test_results.list,
        )


class AsyncTestResultsResourceWithStreamingResponse:
    def __init__(self, test_results: AsyncTestResultsResource) -> None:
        self._test_results = test_results

        self.list = async_to_streamed_response_wrapper(
            test_results.list,
        )
