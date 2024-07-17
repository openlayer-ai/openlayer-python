# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

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
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .test_results import (
    TestResultsResource,
    AsyncTestResultsResource,
    TestResultsResourceWithRawResponse,
    AsyncTestResultsResourceWithRawResponse,
    TestResultsResourceWithStreamingResponse,
    AsyncTestResultsResourceWithStreamingResponse,
)

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
        return InferencePipelinesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> InferencePipelinesResourceWithStreamingResponse:
        return InferencePipelinesResourceWithStreamingResponse(self)


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
        return AsyncInferencePipelinesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncInferencePipelinesResourceWithStreamingResponse:
        return AsyncInferencePipelinesResourceWithStreamingResponse(self)


class InferencePipelinesResourceWithRawResponse:
    def __init__(self, inference_pipelines: InferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

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

    @cached_property
    def data(self) -> AsyncDataResourceWithStreamingResponse:
        return AsyncDataResourceWithStreamingResponse(self._inference_pipelines.data)

    @cached_property
    def rows(self) -> AsyncRowsResourceWithStreamingResponse:
        return AsyncRowsResourceWithStreamingResponse(self._inference_pipelines.rows)

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithStreamingResponse:
        return AsyncTestResultsResourceWithStreamingResponse(self._inference_pipelines.test_results)
