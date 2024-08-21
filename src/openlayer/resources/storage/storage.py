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
        return StorageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> StorageResourceWithStreamingResponse:
        return StorageResourceWithStreamingResponse(self)


class AsyncStorageResource(AsyncAPIResource):
    @cached_property
    def presigned_url(self) -> AsyncPresignedURLResource:
        return AsyncPresignedURLResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncStorageResourceWithRawResponse:
        return AsyncStorageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncStorageResourceWithStreamingResponse:
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
