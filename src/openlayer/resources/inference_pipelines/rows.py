# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, SequenceNotStr, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from ...types.inference_pipelines import row_list_params, row_update_params
from ...types.inference_pipelines.row_list_response import RowListResponse
from ...types.inference_pipelines.row_update_response import RowUpdateResponse
from ...types.inference_pipelines.row_retrieve_response import RowRetrieveResponse

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

    def retrieve(
        self,
        inference_id: str,
        *,
        inference_pipeline_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RowRetrieveResponse:
        """
        Fetch a single inference pipeline row by inference ID, including OTel steps.

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
        if not inference_id:
            raise ValueError(f"Expected a non-empty value for `inference_id` but received {inference_id!r}")
        return self._get(
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows/{inference_id}",
                inference_pipeline_id=inference_pipeline_id,
                inference_id=inference_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RowRetrieveResponse,
        )

    def update(
        self,
        inference_pipeline_id: str,
        *,
        inference_id: str,
        row: object,
        config: Optional[row_update_params.Config] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
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
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows", inference_pipeline_id=inference_pipeline_id
            ),
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

    def list(
        self,
        inference_pipeline_id: str,
        *,
        asc: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        sort_column: str | Omit = omit,
        column_filters: Optional[Iterable[row_list_params.ColumnFilter]] | Omit = omit,
        exclude_row_id_list: Optional[Iterable[int]] | Omit = omit,
        not_search_query_and: Optional[SequenceNotStr[str]] | Omit = omit,
        not_search_query_or: Optional[SequenceNotStr[str]] | Omit = omit,
        row_id_list: Optional[Iterable[int]] | Omit = omit,
        search_query_and: Optional[SequenceNotStr[str]] | Omit = omit,
        search_query_or: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RowListResponse:
        """
        A list of rows for an inference pipeline.

        Args:
          asc: Whether or not to sort on the sortColumn in ascending order.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          sort_column: Name of the column to sort on

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
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows", inference_pipeline_id=inference_pipeline_id
            ),
            body=maybe_transform(
                {
                    "column_filters": column_filters,
                    "exclude_row_id_list": exclude_row_id_list,
                    "not_search_query_and": not_search_query_and,
                    "not_search_query_or": not_search_query_or,
                    "row_id_list": row_id_list,
                    "search_query_and": search_query_and,
                    "search_query_or": search_query_or,
                },
                row_list_params.RowListParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "asc": asc,
                        "page": page,
                        "per_page": per_page,
                        "sort_column": sort_column,
                    },
                    row_list_params.RowListParams,
                ),
            ),
            cast_to=RowListResponse,
        )

    def delete(
        self,
        inference_id: str,
        *,
        inference_pipeline_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """Delete a single inference pipeline row by inference ID.

        Only project admins can
        perform this action.

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
        if not inference_id:
            raise ValueError(f"Expected a non-empty value for `inference_id` but received {inference_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows/{inference_id}",
                inference_pipeline_id=inference_pipeline_id,
                inference_id=inference_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
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

    async def retrieve(
        self,
        inference_id: str,
        *,
        inference_pipeline_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RowRetrieveResponse:
        """
        Fetch a single inference pipeline row by inference ID, including OTel steps.

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
        if not inference_id:
            raise ValueError(f"Expected a non-empty value for `inference_id` but received {inference_id!r}")
        return await self._get(
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows/{inference_id}",
                inference_pipeline_id=inference_pipeline_id,
                inference_id=inference_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RowRetrieveResponse,
        )

    async def update(
        self,
        inference_pipeline_id: str,
        *,
        inference_id: str,
        row: object,
        config: Optional[row_update_params.Config] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
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
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows", inference_pipeline_id=inference_pipeline_id
            ),
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

    async def list(
        self,
        inference_pipeline_id: str,
        *,
        asc: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        sort_column: str | Omit = omit,
        column_filters: Optional[Iterable[row_list_params.ColumnFilter]] | Omit = omit,
        exclude_row_id_list: Optional[Iterable[int]] | Omit = omit,
        not_search_query_and: Optional[SequenceNotStr[str]] | Omit = omit,
        not_search_query_or: Optional[SequenceNotStr[str]] | Omit = omit,
        row_id_list: Optional[Iterable[int]] | Omit = omit,
        search_query_and: Optional[SequenceNotStr[str]] | Omit = omit,
        search_query_or: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RowListResponse:
        """
        A list of rows for an inference pipeline.

        Args:
          asc: Whether or not to sort on the sortColumn in ascending order.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          sort_column: Name of the column to sort on

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
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows", inference_pipeline_id=inference_pipeline_id
            ),
            body=await async_maybe_transform(
                {
                    "column_filters": column_filters,
                    "exclude_row_id_list": exclude_row_id_list,
                    "not_search_query_and": not_search_query_and,
                    "not_search_query_or": not_search_query_or,
                    "row_id_list": row_id_list,
                    "search_query_and": search_query_and,
                    "search_query_or": search_query_or,
                },
                row_list_params.RowListParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "asc": asc,
                        "page": page,
                        "per_page": per_page,
                        "sort_column": sort_column,
                    },
                    row_list_params.RowListParams,
                ),
            ),
            cast_to=RowListResponse,
        )

    async def delete(
        self,
        inference_id: str,
        *,
        inference_pipeline_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """Delete a single inference pipeline row by inference ID.

        Only project admins can
        perform this action.

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
        if not inference_id:
            raise ValueError(f"Expected a non-empty value for `inference_id` but received {inference_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template(
                "/inference-pipelines/{inference_pipeline_id}/rows/{inference_id}",
                inference_pipeline_id=inference_pipeline_id,
                inference_id=inference_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class RowsResourceWithRawResponse:
    def __init__(self, rows: RowsResource) -> None:
        self._rows = rows

        self.retrieve = to_raw_response_wrapper(
            rows.retrieve,
        )
        self.update = to_raw_response_wrapper(
            rows.update,
        )
        self.list = to_raw_response_wrapper(
            rows.list,
        )
        self.delete = to_raw_response_wrapper(
            rows.delete,
        )


class AsyncRowsResourceWithRawResponse:
    def __init__(self, rows: AsyncRowsResource) -> None:
        self._rows = rows

        self.retrieve = async_to_raw_response_wrapper(
            rows.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            rows.update,
        )
        self.list = async_to_raw_response_wrapper(
            rows.list,
        )
        self.delete = async_to_raw_response_wrapper(
            rows.delete,
        )


class RowsResourceWithStreamingResponse:
    def __init__(self, rows: RowsResource) -> None:
        self._rows = rows

        self.retrieve = to_streamed_response_wrapper(
            rows.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            rows.update,
        )
        self.list = to_streamed_response_wrapper(
            rows.list,
        )
        self.delete = to_streamed_response_wrapper(
            rows.delete,
        )


class AsyncRowsResourceWithStreamingResponse:
    def __init__(self, rows: AsyncRowsResource) -> None:
        self._rows = rows

        self.retrieve = async_to_streamed_response_wrapper(
            rows.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            rows.update,
        )
        self.list = async_to_streamed_response_wrapper(
            rows.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            rows.delete,
        )
