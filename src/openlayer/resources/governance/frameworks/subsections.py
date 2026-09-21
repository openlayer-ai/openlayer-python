# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ...._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.governance.frameworks import subsection_list_rules_params
from ....types.governance.frameworks.subsection_list_rules_response import SubsectionListRulesResponse

__all__ = ["SubsectionsResource", "AsyncSubsectionsResource"]


class SubsectionsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> SubsectionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return SubsectionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> SubsectionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return SubsectionsResourceWithStreamingResponse(self)

    def list_rules(
        self,
        subsection_id: str,
        *,
        framework_id: str,
        include_results: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SubsectionListRulesResponse:
        """
        List the rules mapped to a subsection of a framework document.

        A subsection is usually the level at which a standard states an individual
        requirement, so this is the endpoint to use when you want to show which rules
        cover a specific clause.

        Args:
          include_results: Whether to include each rule's results inline, in a `results` array.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          status: Only include items whose rule result has this compliance status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        if not subsection_id:
            raise ValueError(f"Expected a non-empty value for `subsection_id` but received {subsection_id!r}")
        return self._get(
            path_template(
                "/frameworks/{framework_id}/subsections/{subsection_id}/rules",
                framework_id=framework_id,
                subsection_id=subsection_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "include_results": include_results,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "status": status,
                    },
                    subsection_list_rules_params.SubsectionListRulesParams,
                ),
            ),
            cast_to=SubsectionListRulesResponse,
        )


class AsyncSubsectionsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncSubsectionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncSubsectionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSubsectionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncSubsectionsResourceWithStreamingResponse(self)

    async def list_rules(
        self,
        subsection_id: str,
        *,
        framework_id: str,
        include_results: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SubsectionListRulesResponse:
        """
        List the rules mapped to a subsection of a framework document.

        A subsection is usually the level at which a standard states an individual
        requirement, so this is the endpoint to use when you want to show which rules
        cover a specific clause.

        Args:
          include_results: Whether to include each rule's results inline, in a `results` array.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          status: Only include items whose rule result has this compliance status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        if not subsection_id:
            raise ValueError(f"Expected a non-empty value for `subsection_id` but received {subsection_id!r}")
        return await self._get(
            path_template(
                "/frameworks/{framework_id}/subsections/{subsection_id}/rules",
                framework_id=framework_id,
                subsection_id=subsection_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "include_results": include_results,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "status": status,
                    },
                    subsection_list_rules_params.SubsectionListRulesParams,
                ),
            ),
            cast_to=SubsectionListRulesResponse,
        )


class SubsectionsResourceWithRawResponse:
    def __init__(self, subsections: SubsectionsResource) -> None:
        self._subsections = subsections

        self.list_rules = to_raw_response_wrapper(
            subsections.list_rules,
        )


class AsyncSubsectionsResourceWithRawResponse:
    def __init__(self, subsections: AsyncSubsectionsResource) -> None:
        self._subsections = subsections

        self.list_rules = async_to_raw_response_wrapper(
            subsections.list_rules,
        )


class SubsectionsResourceWithStreamingResponse:
    def __init__(self, subsections: SubsectionsResource) -> None:
        self._subsections = subsections

        self.list_rules = to_streamed_response_wrapper(
            subsections.list_rules,
        )


class AsyncSubsectionsResourceWithStreamingResponse:
    def __init__(self, subsections: AsyncSubsectionsResource) -> None:
        self._subsections = subsections

        self.list_rules = async_to_streamed_response_wrapper(
            subsections.list_rules,
        )
