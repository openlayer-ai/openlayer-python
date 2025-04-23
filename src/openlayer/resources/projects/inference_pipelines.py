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
from ...types.projects import inference_pipeline_list_params, inference_pipeline_create_params
from ...types.projects.inference_pipeline_list_response import InferencePipelineListResponse
from ...types.projects.inference_pipeline_create_response import InferencePipelineCreateResponse

__all__ = ["InferencePipelinesResource", "AsyncInferencePipelinesResource"]


class InferencePipelinesResource(SyncAPIResource):
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

    def create(
        self,
        project_id: str,
        *,
        description: Optional[str],
        name: str,
        project: Optional[inference_pipeline_create_params.Project] | NotGiven = NOT_GIVEN,
        workspace: Optional[inference_pipeline_create_params.Workspace] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineCreateResponse:
        """
        Create an inference pipeline in a project.

        Args:
          description: The inference pipeline description.

          name: The inference pipeline name.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return self._post(
            f"/projects/{project_id}/inference-pipelines",
            body=maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "project": project,
                    "workspace": workspace,
                },
                inference_pipeline_create_params.InferencePipelineCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=InferencePipelineCreateResponse,
        )

    def list(
        self,
        project_id: str,
        *,
        name: str | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineListResponse:
        """
        List the inference pipelines in a project.

        Args:
          name: Filter list of items by name.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return self._get(
            f"/projects/{project_id}/inference-pipelines",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "name": name,
                        "page": page,
                        "per_page": per_page,
                    },
                    inference_pipeline_list_params.InferencePipelineListParams,
                ),
            ),
            cast_to=InferencePipelineListResponse,
        )


class AsyncInferencePipelinesResource(AsyncAPIResource):
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

    async def create(
        self,
        project_id: str,
        *,
        description: Optional[str],
        name: str,
        project: Optional[inference_pipeline_create_params.Project] | NotGiven = NOT_GIVEN,
        workspace: Optional[inference_pipeline_create_params.Workspace] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineCreateResponse:
        """
        Create an inference pipeline in a project.

        Args:
          description: The inference pipeline description.

          name: The inference pipeline name.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return await self._post(
            f"/projects/{project_id}/inference-pipelines",
            body=await async_maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "project": project,
                    "workspace": workspace,
                },
                inference_pipeline_create_params.InferencePipelineCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=InferencePipelineCreateResponse,
        )

    async def list(
        self,
        project_id: str,
        *,
        name: str | NotGiven = NOT_GIVEN,
        page: int | NotGiven = NOT_GIVEN,
        per_page: int | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> InferencePipelineListResponse:
        """
        List the inference pipelines in a project.

        Args:
          name: Filter list of items by name.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not project_id:
            raise ValueError(f"Expected a non-empty value for `project_id` but received {project_id!r}")
        return await self._get(
            f"/projects/{project_id}/inference-pipelines",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "name": name,
                        "page": page,
                        "per_page": per_page,
                    },
                    inference_pipeline_list_params.InferencePipelineListParams,
                ),
            ),
            cast_to=InferencePipelineListResponse,
        )


class InferencePipelinesResourceWithRawResponse:
    def __init__(self, inference_pipelines: InferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.create = to_raw_response_wrapper(
            inference_pipelines.create,
        )
        self.list = to_raw_response_wrapper(
            inference_pipelines.list,
        )


class AsyncInferencePipelinesResourceWithRawResponse:
    def __init__(self, inference_pipelines: AsyncInferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.create = async_to_raw_response_wrapper(
            inference_pipelines.create,
        )
        self.list = async_to_raw_response_wrapper(
            inference_pipelines.list,
        )


class InferencePipelinesResourceWithStreamingResponse:
    def __init__(self, inference_pipelines: InferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.create = to_streamed_response_wrapper(
            inference_pipelines.create,
        )
        self.list = to_streamed_response_wrapper(
            inference_pipelines.list,
        )


class AsyncInferencePipelinesResourceWithStreamingResponse:
    def __init__(self, inference_pipelines: AsyncInferencePipelinesResource) -> None:
        self._inference_pipelines = inference_pipelines

        self.create = async_to_streamed_response_wrapper(
            inference_pipelines.create,
        )
        self.list = async_to_streamed_response_wrapper(
            inference_pipelines.list,
        )
