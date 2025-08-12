# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Optional
from typing_extensions import Literal

import httpx

from .data import (
    DataResource,
    AsyncDataResource,
    DataResourceWithRawResponse,
    AsyncDataResourceWithRawResponse,
    DataResourceWithStreamingResponse,
    AsyncDataResourceWithStreamingResponse,
)
from .rows import (
    RowsResource,
    AsyncRowsResource,
    RowsResourceWithRawResponse,
    AsyncRowsResourceWithRawResponse,
    RowsResourceWithStreamingResponse,
    AsyncRowsResourceWithStreamingResponse,
)
from ...types import inference_pipeline_update_params, inference_pipeline_retrieve_params
from ...types.inference_pipelines import export_data_params
from ..._types import NOT_GIVEN, Body, Query, Headers, NoneType, NotGiven
from ..._utils import maybe_transform, async_maybe_transform
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
from ...types.inference_pipeline_update_response import InferencePipelineUpdateResponse
from ...types.inference_pipeline_retrieve_response import InferencePipelineRetrieveResponse
from ...types.inference_pipelines.export_data_response import ExportDataResponse
from ...types.inference_pipelines.task_status_response import TaskStatusResponse

__all__ = ["InferencePipelinesResource", "AsyncInferencePipelinesResource"]


class InferencePipelinesResource(SyncAPIResource):
    @cached_property
    def data(self) -> DataResource:
        return DataResource(self._client)

    @cached_property
    def rows(self) -> RowsResource:
        return RowsResource(self._client)

    @cached_property
    def test_results(self) -> TestResultsResource:
        return TestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> InferencePipelinesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return InferencePipelinesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> InferencePipelinesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return InferencePipelinesResourceWithStreamingResponse(self)

    def retrieve(
        self,
        inference_pipeline_id: str,
        *,
        expand: List[Literal["project", "workspace"]] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineRetrieveResponse:
        """
        Retrieve inference pipeline.

        Args:
          expand: Expand specific nested objects.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return self._get(
            f"/inference-pipelines/{inference_pipeline_id}",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"expand": expand}, inference_pipeline_retrieve_params.InferencePipelineRetrieveParams
                ),
            ),
            cast_to=InferencePipelineRetrieveResponse,
        )

    def update(
        self,
        inference_pipeline_id: str,
        *,
        description: Optional[str] | NotGiven = NOT_GIVEN,
        name: str | NotGiven = NOT_GIVEN,
        reference_dataset_uri: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineUpdateResponse:
        """
        Update inference pipeline.

        Args:
          description: The inference pipeline description.

          name: The inference pipeline name.

          reference_dataset_uri: The storage uri of your reference dataset. We recommend using the Python SDK or
              the UI to handle your reference dataset updates.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return self._put(
            f"/inference-pipelines/{inference_pipeline_id}",
            body=maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "reference_dataset_uri": reference_dataset_uri,
                },
                inference_pipeline_update_params.InferencePipelineUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=InferencePipelineUpdateResponse,
        )

    def delete(
        self,
        inference_pipeline_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> None:
        """
        Delete inference pipeline.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            f"/inference-pipelines/{inference_pipeline_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    def export_data(
        self,
        inference_pipeline_id: str,
        *,
        start: int,
        end: int,
        fmt: Literal["json", "csv"],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ExportDataResponse:
        """
        Export data from inference pipeline for a specified time range.

        Args:
          start: Start timestamp (Unix timestamp in seconds) for the data export range.

          end: End timestamp (Unix timestamp in seconds) for the data export range.

          fmt: Export format. Supported formats: 'json', 'csv'.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return self._post(
            f"/inference-pipelines/{inference_pipeline_id}/export",
            body=maybe_transform(
                {
                    "start": start,
                    "end": end,
                    "fmt": fmt,
                },
                export_data_params.ExportDataParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExportDataResponse,
        )

    def get_task_status(
        self,
        task_result_url: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TaskStatusResponse:
        """
        Get the status of an export task using the task result URL.

        Args:
          task_result_url: The task result URL returned from export_data method.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not task_result_url:
            raise ValueError(
                f"Expected a non-empty value for `task_result_url` but received {task_result_url!r}"
            )
        return self._get(
            task_result_url,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TaskStatusResponse,
        )


class AsyncInferencePipelinesResource(AsyncAPIResource):
    @cached_property
    def data(self) -> AsyncDataResource:
        return AsyncDataResource(self._client)

    @cached_property
    def rows(self) -> AsyncRowsResource:
        return AsyncRowsResource(self._client)

    @cached_property
    def test_results(self) -> AsyncTestResultsResource:
        return AsyncTestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncInferencePipelinesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncInferencePipelinesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncInferencePipelinesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncInferencePipelinesResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        inference_pipeline_id: str,
        *,
        expand: List[Literal["project", "workspace"]] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineRetrieveResponse:
        """
        Retrieve inference pipeline.

        Args:
          expand: Expand specific nested objects.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return await self._get(
            f"/inference-pipelines/{inference_pipeline_id}",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"expand": expand}, inference_pipeline_retrieve_params.InferencePipelineRetrieveParams
                ),
            ),
            cast_to=InferencePipelineRetrieveResponse,
        )

    async def update(
        self,
        inference_pipeline_id: str,
        *,
        description: Optional[str] | NotGiven = NOT_GIVEN,
        name: str | NotGiven = NOT_GIVEN,
        reference_dataset_uri: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineUpdateResponse:
        """
        Update inference pipeline.

        Args:
          description: The inference pipeline description.

          name: The inference pipeline name.

          reference_dataset_uri: The storage uri of your reference dataset. We recommend using the Python SDK or
              the UI to handle your reference dataset updates.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return await self._put(
            f"/inference-pipelines/{inference_pipeline_id}",
            body=await async_maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "reference_dataset_uri": reference_dataset_uri,
                },
                inference_pipeline_update_params.InferencePipelineUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=InferencePipelineUpdateResponse,
        )

    async def delete(
        self,
        inference_pipeline_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> None:
        """
        Delete inference pipeline.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            f"/inference-pipelines/{inference_pipeline_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )

    async def export_data(
        self,
        inference_pipeline_id: str,
        *,
        start: int,
        end: int,
        fmt: Literal["json", "csv"],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ExportDataResponse:
        """
        Export data from inference pipeline for a specified time range.

        Args:
          start: Start timestamp (Unix timestamp in seconds) for the data export range.

          end: End timestamp (Unix timestamp in seconds) for the data export range.

          fmt: Export format. Supported formats: 'json', 'csv'.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not inference_pipeline_id:
            raise ValueError(
                f"Expected a non-empty value for `inference_pipeline_id` but received {inference_pipeline_id!r}"
            )
        return await self._post(
            f"/inference-pipelines/{inference_pipeline_id}/export",
            body=await async_maybe_transform(
                {
                    "start": start,
                    "end": end,
                    "fmt": fmt,
                },
                export_data_params.ExportDataParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExportDataResponse,
        )

    async def get_task_status(
        self,
        task_result_url: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TaskStatusResponse:
        """
        Get the status of an export task using the task result URL.

        Args:
          task_result_url: The task result URL returned from export_data method.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not task_result_url:
            raise ValueError(
                f"Expected a non-empty value for `task_result_url` but received {task_result_url!r}"
            )
        return await self._get(
            task_result_url,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=TaskStatusResponse,
        )


class InferencePipelinesResourceWithRawResponse:
    def __init__(self, inference_pipelines: InferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.retrieve = to_raw_response_wrapper(
            inference_pipelines.retrieve,
        )
        self.update = to_raw_response_wrapper(
            inference_pipelines.update,
        )
        self.delete = to_raw_response_wrapper(
            inference_pipelines.delete,
        )
        self.export_data = to_raw_response_wrapper(
            inference_pipelines.export_data,
        )
        self.get_task_status = to_raw_response_wrapper(
            inference_pipelines.get_task_status,
        )

    @cached_property
    def data(self) -> DataResourceWithRawResponse:
        return DataResourceWithRawResponse(self._inference_pipelines.data)

    @cached_property
    def rows(self) -> RowsResourceWithRawResponse:
        return RowsResourceWithRawResponse(self._inference_pipelines.rows)

    @cached_property
    def test_results(self) -> TestResultsResourceWithRawResponse:
        return TestResultsResourceWithRawResponse(self._inference_pipelines.test_results)


class AsyncInferencePipelinesResourceWithRawResponse:
    def __init__(self, inference_pipelines: AsyncInferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.retrieve = async_to_raw_response_wrapper(
            inference_pipelines.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            inference_pipelines.update,
        )
        self.delete = async_to_raw_response_wrapper(
            inference_pipelines.delete,
        )
        self.export_data = async_to_raw_response_wrapper(
            inference_pipelines.export_data,
        )
        self.get_task_status = async_to_raw_response_wrapper(
            inference_pipelines.get_task_status,
        )

    @cached_property
    def data(self) -> AsyncDataResourceWithRawResponse:
        return AsyncDataResourceWithRawResponse(self._inference_pipelines.data)

    @cached_property
    def rows(self) -> AsyncRowsResourceWithRawResponse:
        return AsyncRowsResourceWithRawResponse(self._inference_pipelines.rows)

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithRawResponse:
        return AsyncTestResultsResourceWithRawResponse(self._inference_pipelines.test_results)


class InferencePipelinesResourceWithStreamingResponse:
    def __init__(self, inference_pipelines: InferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.retrieve = to_streamed_response_wrapper(
            inference_pipelines.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            inference_pipelines.update,
        )
        self.delete = to_streamed_response_wrapper(
            inference_pipelines.delete,
        )
        self.export_data = to_streamed_response_wrapper(
            inference_pipelines.export_data,
        )
        self.get_task_status = to_streamed_response_wrapper(
            inference_pipelines.get_task_status,
        )

    @cached_property
    def data(self) -> DataResourceWithStreamingResponse:
        return DataResourceWithStreamingResponse(self._inference_pipelines.data)

    @cached_property
    def rows(self) -> RowsResourceWithStreamingResponse:
        return RowsResourceWithStreamingResponse(self._inference_pipelines.rows)

    @cached_property
    def test_results(self) -> TestResultsResourceWithStreamingResponse:
        return TestResultsResourceWithStreamingResponse(self._inference_pipelines.test_results)


class AsyncInferencePipelinesResourceWithStreamingResponse:
    def __init__(self, inference_pipelines: AsyncInferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.retrieve = async_to_streamed_response_wrapper(
            inference_pipelines.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            inference_pipelines.update,
        )
        self.delete = async_to_streamed_response_wrapper(
            inference_pipelines.delete,
        )
        self.export_data = async_to_streamed_response_wrapper(
            inference_pipelines.export_data,
        )
        self.get_task_status = async_to_streamed_response_wrapper(
            inference_pipelines.get_task_status,
        )

    @cached_property
    def data(self) -> AsyncDataResourceWithStreamingResponse:
        return AsyncDataResourceWithStreamingResponse(self._inference_pipelines.data)

    @cached_property
    def rows(self) -> AsyncRowsResourceWithStreamingResponse:
        return AsyncRowsResourceWithStreamingResponse(self._inference_pipelines.rows)

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithStreamingResponse:
        return AsyncTestResultsResourceWithStreamingResponse(self._inference_pipelines.test_results)
