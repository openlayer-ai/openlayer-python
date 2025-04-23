# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional

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
from ...types.inference_pipelines import row_update_params
from ...types.inference_pipelines.row_update_response import RowUpdateResponse

__all__ = ["RowsResource", "AsyncRowsResource"]


class RowsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> RowsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return RowsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> RowsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return RowsResourceWithStreamingResponse(self)

    def update(
        self,
        inference_pipeline_id: str,
        *,
        inference_id: str,
        row: object,
        config: Optional[row_update_params.Config] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> RowUpdateResponse:
        """
        Update an inference data point in an inference pipeline.

        Args:
          inference_id: Specify the inference id as a query param.

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
            f"/inference-pipelines/{inference_pipeline_id}/rows",
            body=maybe_transform(
                {
                    "row": row,
                    "config": config,
                },
                row_update_params.RowUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"inference_id": inference_id}, row_update_params.RowUpdateParams),
            ),
            cast_to=RowUpdateResponse,
        )


class AsyncRowsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncRowsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncRowsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncRowsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncRowsResourceWithStreamingResponse(self)

    async def update(
        self,
        inference_pipeline_id: str,
        *,
        inference_id: str,
        row: object,
        config: Optional[row_update_params.Config] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> RowUpdateResponse:
        """
        Update an inference data point in an inference pipeline.

        Args:
          inference_id: Specify the inference id as a query param.

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
            f"/inference-pipelines/{inference_pipeline_id}/rows",
            body=await async_maybe_transform(
                {
                    "row": row,
                    "config": config,
                },
                row_update_params.RowUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform({"inference_id": inference_id}, row_update_params.RowUpdateParams),
            ),
            cast_to=RowUpdateResponse,
        )


class RowsResourceWithRawResponse:
    def __init__(self, rows: RowsResource) -> None:
        self._rows = rows

        self.update = to_raw_response_wrapper(
            rows.update,
        )


class AsyncRowsResourceWithRawResponse:
    def __init__(self, rows: AsyncRowsResource) -> None:
        self._rows = rows

        self.update = async_to_raw_response_wrapper(
            rows.update,
        )


class RowsResourceWithStreamingResponse:
    def __init__(self, rows: RowsResource) -> None:
        self._rows = rows

        self.update = to_streamed_response_wrapper(
            rows.update,
        )


class AsyncRowsResourceWithStreamingResponse:
    def __init__(self, rows: AsyncRowsResource) -> None:
        self._rows = rows

        self.update = async_to_streamed_response_wrapper(
            rows.update,
        )
