# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .presigned_url import (
    PresignedURLResource,
    AsyncPresignedURLResource,
    PresignedURLResourceWithRawResponse,
    AsyncPresignedURLResourceWithRawResponse,
    PresignedURLResourceWithStreamingResponse,
    AsyncPresignedURLResourceWithStreamingResponse,
)

__all__ = ["StorageResource", "AsyncStorageResource"]


class StorageResource(SyncAPIResource):
    @cached_property
    def presigned_url(self) -> PresignedURLResource:
        return PresignedURLResource(self._client)

    @cached_property
    def with_raw_response(self) -> StorageResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return StorageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> StorageResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return StorageResourceWithStreamingResponse(self)


class AsyncStorageResource(AsyncAPIResource):
    @cached_property
    def presigned_url(self) -> AsyncPresignedURLResource:
        return AsyncPresignedURLResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncStorageResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncStorageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncStorageResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncStorageResourceWithStreamingResponse(self)


class StorageResourceWithRawResponse:
    def __init__(self, storage: StorageResource) -> None:
        self._storage = storage

    @cached_property
    def presigned_url(self) -> PresignedURLResourceWithRawResponse:
        return PresignedURLResourceWithRawResponse(self._storage.presigned_url)


class AsyncStorageResourceWithRawResponse:
    def __init__(self, storage: AsyncStorageResource) -> None:
        self._storage = storage

    @cached_property
    def presigned_url(self) -> AsyncPresignedURLResourceWithRawResponse:
        return AsyncPresignedURLResourceWithRawResponse(self._storage.presigned_url)


class StorageResourceWithStreamingResponse:
    def __init__(self, storage: StorageResource) -> None:
        self._storage = storage

    @cached_property
    def presigned_url(self) -> PresignedURLResourceWithStreamingResponse:
        return PresignedURLResourceWithStreamingResponse(self._storage.presigned_url)


class AsyncStorageResourceWithStreamingResponse:
    def __init__(self, storage: AsyncStorageResource) -> None:
        self._storage = storage

    @cached_property
    def presigned_url(self) -> AsyncPresignedURLResourceWithStreamingResponse:
        return AsyncPresignedURLResourceWithStreamingResponse(self._storage.presigned_url)
