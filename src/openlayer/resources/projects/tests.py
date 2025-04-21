# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional

import httpx

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
from ..._base_client import make_request_options
from ...types.projects import test_create_params
from ...types.projects.test_create_response import TestCreateResponse

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

    def create(
        self,
        project_id: str,
        *,
        description: Optional[object],
        name: str,
        subtype: str,
        thresholds: Iterable[test_create_params.Threshold],
        type: str,
        archived: bool | NotGiven = NOT_GIVEN,
        delay_window: Optional[float] | NotGiven = NOT_GIVEN,
        evaluation_window: Optional[float] | NotGiven = NOT_GIVEN,
        uses_ml_model: bool | NotGiven = NOT_GIVEN,
        uses_production_data: bool | NotGiven = NOT_GIVEN,
        uses_reference_dataset: bool | NotGiven = NOT_GIVEN,
        uses_training_dataset: bool | NotGiven = NOT_GIVEN,
        uses_validation_dataset: bool | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestCreateResponse:
        """
        Create a test.

        Args:
          description: The test description.

          name: The test name.

          subtype: The test subtype.

          type: The test type.

          archived: Whether the test is archived.

          delay_window: The delay window in seconds. Only applies to tests that use production data.

          evaluation_window: The evaluation window in seconds. Only applies to tests that use production
              data.

          uses_ml_model: Whether the test uses an ML model.

          uses_production_data: Whether the test uses production data (monitoring mode only).

          uses_reference_dataset: Whether the test uses a reference dataset (monitoring mode only).

          uses_training_dataset: Whether the test uses a training dataset.

          uses_validation_dataset: Whether the test uses a validation dataset.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return self._post(
            f"/projects/{project_id}/tests",
            body=maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "subtype": subtype,
                    "thresholds": thresholds,
                    "type": type,
                    "archived": archived,
                    "delay_window": delay_window,
                    "evaluation_window": evaluation_window,
                    "uses_ml_model": uses_ml_model,
                    "uses_production_data": uses_production_data,
                    "uses_reference_dataset": uses_reference_dataset,
                    "uses_training_dataset": uses_training_dataset,
                    "uses_validation_dataset": uses_validation_dataset,
                },
                test_create_params.TestCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestCreateResponse,
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

    async def create(
        self,
        project_id: str,
        *,
        description: Optional[object],
        name: str,
        subtype: str,
        thresholds: Iterable[test_create_params.Threshold],
        type: str,
        archived: bool | NotGiven = NOT_GIVEN,
        delay_window: Optional[float] | NotGiven = NOT_GIVEN,
        evaluation_window: Optional[float] | NotGiven = NOT_GIVEN,
        uses_ml_model: bool | NotGiven = NOT_GIVEN,
        uses_production_data: bool | NotGiven = NOT_GIVEN,
        uses_reference_dataset: bool | NotGiven = NOT_GIVEN,
        uses_training_dataset: bool | NotGiven = NOT_GIVEN,
        uses_validation_dataset: bool | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestCreateResponse:
        """
        Create a test.

        Args:
          description: The test description.

          name: The test name.

          subtype: The test subtype.

          type: The test type.

          archived: Whether the test is archived.

          delay_window: The delay window in seconds. Only applies to tests that use production data.

          evaluation_window: The evaluation window in seconds. Only applies to tests that use production
              data.

          uses_ml_model: Whether the test uses an ML model.

          uses_production_data: Whether the test uses production data (monitoring mode only).

          uses_reference_dataset: Whether the test uses a reference dataset (monitoring mode only).

          uses_training_dataset: Whether the test uses a training dataset.

          uses_validation_dataset: Whether the test uses a validation dataset.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return await self._post(
            f"/projects/{project_id}/tests",
            body=await async_maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "subtype": subtype,
                    "thresholds": thresholds,
                    "type": type,
                    "archived": archived,
                    "delay_window": delay_window,
                    "evaluation_window": evaluation_window,
                    "uses_ml_model": uses_ml_model,
                    "uses_production_data": uses_production_data,
                    "uses_reference_dataset": uses_reference_dataset,
                    "uses_training_dataset": uses_training_dataset,
                    "uses_validation_dataset": uses_validation_dataset,
                },
                test_create_params.TestCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestCreateResponse,
        )


class TestsResourceWithRawResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.create = to_raw_response_wrapper(
            tests.create,
        )


class AsyncTestsResourceWithRawResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.create = async_to_raw_response_wrapper(
            tests.create,
        )


class TestsResourceWithStreamingResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.create = to_streamed_response_wrapper(
            tests.create,
        )


class AsyncTestsResourceWithStreamingResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.create = async_to_streamed_response_wrapper(
            tests.create,
        )
