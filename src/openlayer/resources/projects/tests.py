# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional
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
from ...types.projects import test_list_params, test_create_params, test_update_params
from ...types.projects.test_list_response import TestListResponse
from ...types.projects.test_create_response import TestCreateResponse
from ...types.projects.test_update_response import TestUpdateResponse

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
        subtype: Literal[
            "anomalousColumnCount",
            "characterLength",
            "classImbalanceRatio",
            "expectColumnAToBeInColumnB",
            "columnAverage",
            "columnDrift",
            "columnStatistic",
            "columnValuesMatch",
            "conflictingLabelRowCount",
            "containsPii",
            "containsValidUrl",
            "correlatedFeatureCount",
            "customMetricThreshold",
            "duplicateRowCount",
            "emptyFeature",
            "emptyFeatureCount",
            "driftedFeatureCount",
            "featureMissingValues",
            "featureValueValidation",
            "greatExpectations",
            "groupByColumnStatsCheck",
            "illFormedRowCount",
            "isCode",
            "isJson",
            "llmRubricThresholdV2",
            "labelDrift",
            "metricThreshold",
            "newCategoryCount",
            "newLabelCount",
            "nullRowCount",
            "rowCount",
            "ppScoreValueValidation",
            "quasiConstantFeature",
            "quasiConstantFeatureCount",
            "sqlQuery",
            "dtypeValidation",
            "sentenceLength",
            "sizeRatio",
            "specialCharactersRatio",
            "stringValidation",
            "trainValLeakageRowCount",
        ],
        thresholds: Iterable[test_create_params.Threshold],
        type: Literal["integrity", "consistency", "performance"],
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

    def update(
        self,
        project_id: str,
        *,
        payloads: Iterable[test_update_params.Payload],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestUpdateResponse:
        """
        Update tests.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return self._put(
            f"/projects/{project_id}/tests",
            body=maybe_transform({"payloads": payloads}, test_update_params.TestUpdateParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestUpdateResponse,
        )

    def list(
        self,
        project_id: str,
        *,
        include_archived: bool | NotGiven = NOT_GIVEN,
        origin_version_id: Optional[str] | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        suggested: bool | NotGiven = NOT_GIVEN,
        type: Literal["integrity", "consistency", "performance", "fairness", "robustness"] | NotGiven = NOT_GIVEN,
        uses_production_data: Optional[bool] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestListResponse:
        """
        List tests under a project.

        Args:
          include_archived: Filter for archived tests.

          origin_version_id: Retrive tests created by a specific project version.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          suggested: Filter for suggested tests.

          type: Filter objects by test type. Available types are `integrity`, `consistency`,
              `performance`, `fairness`, and `robustness`.

          uses_production_data: Retrive tests with usesProductionData (monitoring).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return self._get(
            f"/projects/{project_id}/tests",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "include_archived": include_archived,
                        "origin_version_id": origin_version_id,
                        "page": page,
                        "per_page": per_page,
                        "suggested": suggested,
                        "type": type,
                        "uses_production_data": uses_production_data,
                    },
                    test_list_params.TestListParams,
                ),
            ),
            cast_to=TestListResponse,
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
        subtype: Literal[
            "anomalousColumnCount",
            "characterLength",
            "classImbalanceRatio",
            "expectColumnAToBeInColumnB",
            "columnAverage",
            "columnDrift",
            "columnStatistic",
            "columnValuesMatch",
            "conflictingLabelRowCount",
            "containsPii",
            "containsValidUrl",
            "correlatedFeatureCount",
            "customMetricThreshold",
            "duplicateRowCount",
            "emptyFeature",
            "emptyFeatureCount",
            "driftedFeatureCount",
            "featureMissingValues",
            "featureValueValidation",
            "greatExpectations",
            "groupByColumnStatsCheck",
            "illFormedRowCount",
            "isCode",
            "isJson",
            "llmRubricThresholdV2",
            "labelDrift",
            "metricThreshold",
            "newCategoryCount",
            "newLabelCount",
            "nullRowCount",
            "rowCount",
            "ppScoreValueValidation",
            "quasiConstantFeature",
            "quasiConstantFeatureCount",
            "sqlQuery",
            "dtypeValidation",
            "sentenceLength",
            "sizeRatio",
            "specialCharactersRatio",
            "stringValidation",
            "trainValLeakageRowCount",
        ],
        thresholds: Iterable[test_create_params.Threshold],
        type: Literal["integrity", "consistency", "performance"],
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

    async def update(
        self,
        project_id: str,
        *,
        payloads: Iterable[test_update_params.Payload],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestUpdateResponse:
        """
        Update tests.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return await self._put(
            f"/projects/{project_id}/tests",
            body=await async_maybe_transform({"payloads": payloads}, test_update_params.TestUpdateParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TestUpdateResponse,
        )

    async def list(
        self,
        project_id: str,
        *,
        include_archived: bool | NotGiven = NOT_GIVEN,
        origin_version_id: Optional[str] | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        suggested: bool | NotGiven = NOT_GIVEN,
        type: Literal["integrity", "consistency", "performance", "fairness", "robustness"] | NotGiven = NOT_GIVEN,
        uses_production_data: Optional[bool] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TestListResponse:
        """
        List tests under a project.

        Args:
          include_archived: Filter for archived tests.

          origin_version_id: Retrive tests created by a specific project version.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          suggested: Filter for suggested tests.

          type: Filter objects by test type. Available types are `integrity`, `consistency`,
              `performance`, `fairness`, and `robustness`.

          uses_production_data: Retrive tests with usesProductionData (monitoring).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return await self._get(
            f"/projects/{project_id}/tests",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "include_archived": include_archived,
                        "origin_version_id": origin_version_id,
                        "page": page,
                        "per_page": per_page,
                        "suggested": suggested,
                        "type": type,
                        "uses_production_data": uses_production_data,
                    },
                    test_list_params.TestListParams,
                ),
            ),
            cast_to=TestListResponse,
        )


class TestsResourceWithRawResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.create = to_raw_response_wrapper(
            tests.create,
        )
        self.update = to_raw_response_wrapper(
            tests.update,
        )
        self.list = to_raw_response_wrapper(
            tests.list,
        )


class AsyncTestsResourceWithRawResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.create = async_to_raw_response_wrapper(
            tests.create,
        )
        self.update = async_to_raw_response_wrapper(
            tests.update,
        )
        self.list = async_to_raw_response_wrapper(
            tests.list,
        )


class TestsResourceWithStreamingResponse:
    __test__ = False

    def __init__(self, tests: TestsResource) -> None:
        self._tests = tests

        self.create = to_streamed_response_wrapper(
            tests.create,
        )
        self.update = to_streamed_response_wrapper(
            tests.update,
        )
        self.list = to_streamed_response_wrapper(
            tests.list,
        )


class AsyncTestsResourceWithStreamingResponse:
    def __init__(self, tests: AsyncTestsResource) -> None:
        self._tests = tests

        self.create = async_to_streamed_response_wrapper(
            tests.create,
        )
        self.update = async_to_streamed_response_wrapper(
            tests.update,
        )
        self.list = async_to_streamed_response_wrapper(
            tests.list,
        )
