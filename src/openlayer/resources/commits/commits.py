# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from .test_results import TestResultsResource, AsyncTestResultsResource

from ..._compat import cached_property

import warnings
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Any, Mapping, cast, overload
from typing_extensions import Literal
from ..._utils import extract_files, maybe_transform, required_args, deepcopy_minimal, strip_not_given
from ..._types import NotGiven, Timeout, Headers, NoneType, Query, Body, NOT_GIVEN, FileTypes, BinaryResponseContent
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._base_client import (
    SyncAPIClient,
    AsyncAPIClient,
    _merge_mappings,
    AsyncPaginator,
    make_request_options,
    HttpxBinaryResponseContent,
)
from ...types import shared_params
from .test_results import (
    TestResultsResource,
    AsyncTestResultsResource,
    TestResultsResourceWithRawResponse,
    AsyncTestResultsResourceWithRawResponse,
    TestResultsResourceWithStreamingResponse,
    AsyncTestResultsResourceWithStreamingResponse,
)

__all__ = ["CommitsResource", "AsyncCommitsResource"]


class CommitsResource(SyncAPIResource):
    @cached_property
    def test_results(self) -> TestResultsResource:
        return TestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> CommitsResourceWithRawResponse:
        return CommitsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> CommitsResourceWithStreamingResponse:
        return CommitsResourceWithStreamingResponse(self)


class AsyncCommitsResource(AsyncAPIResource):
    @cached_property
    def test_results(self) -> AsyncTestResultsResource:
        return AsyncTestResultsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncCommitsResourceWithRawResponse:
        return AsyncCommitsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncCommitsResourceWithStreamingResponse:
        return AsyncCommitsResourceWithStreamingResponse(self)


class CommitsResourceWithRawResponse:
    def __init__(self, commits: CommitsResource) -> None:
        self._commits = commits

    @cached_property
    def test_results(self) -> TestResultsResourceWithRawResponse:
        return TestResultsResourceWithRawResponse(self._commits.test_results)


class AsyncCommitsResourceWithRawResponse:
    def __init__(self, commits: AsyncCommitsResource) -> None:
        self._commits = commits

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithRawResponse:
        return AsyncTestResultsResourceWithRawResponse(self._commits.test_results)


class CommitsResourceWithStreamingResponse:
    def __init__(self, commits: CommitsResource) -> None:
        self._commits = commits

    @cached_property
    def test_results(self) -> TestResultsResourceWithStreamingResponse:
        return TestResultsResourceWithStreamingResponse(self._commits.test_results)


class AsyncCommitsResourceWithStreamingResponse:
    def __init__(self, commits: AsyncCommitsResource) -> None:
        self._commits = commits

    @cached_property
    def test_results(self) -> AsyncTestResultsResourceWithStreamingResponse:
        return AsyncTestResultsResourceWithStreamingResponse(self._commits.test_results)
