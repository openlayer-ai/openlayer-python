# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

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
from ...types.storage import presigned_url_create_params
from ...types.storage.presigned_url_create_response import PresignedURLCreateResponse

__all__ = ["PresignedURLResource", "AsyncPresignedURLResource"]


class PresignedURLResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> PresignedURLResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return PresignedURLResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> PresignedURLResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return PresignedURLResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        object_name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PresignedURLCreateResponse:
        """
        Retrieve a presigned url to post storage artifacts.

        Args:
          object_name: The name of the object.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/storage/presigned-url",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"object_name": object_name}, presigned_url_create_params.PresignedURLCreateParams
                ),
            ),
            cast_to=PresignedURLCreateResponse,
        )


class AsyncPresignedURLResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncPresignedURLResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncPresignedURLResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncPresignedURLResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncPresignedURLResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        object_name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PresignedURLCreateResponse:
        """
        Retrieve a presigned url to post storage artifacts.

        Args:
          object_name: The name of the object.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/storage/presigned-url",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"object_name": object_name}, presigned_url_create_params.PresignedURLCreateParams
                ),
            ),
            cast_to=PresignedURLCreateResponse,
        )


class PresignedURLResourceWithRawResponse:
    def __init__(self, presigned_url: PresignedURLResource) -> None:
        self._presigned_url = presigned_url

        self.create = to_raw_response_wrapper(
            presigned_url.create,
        )


class AsyncPresignedURLResourceWithRawResponse:
    def __init__(self, presigned_url: AsyncPresignedURLResource) -> None:
        self._presigned_url = presigned_url

        self.create = async_to_raw_response_wrapper(
            presigned_url.create,
        )


class PresignedURLResourceWithStreamingResponse:
    def __init__(self, presigned_url: PresignedURLResource) -> None:
        self._presigned_url = presigned_url

        self.create = to_streamed_response_wrapper(
            presigned_url.create,
        )


class AsyncPresignedURLResourceWithStreamingResponse:
    def __init__(self, presigned_url: AsyncPresignedURLResource) -> None:
        self._presigned_url = presigned_url

        self.create = async_to_streamed_response_wrapper(
            presigned_url.create,
        )
