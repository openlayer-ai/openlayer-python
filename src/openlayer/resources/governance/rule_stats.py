# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
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
from ...types.governance import rule_stat_retrieve_params
from ...types.governance.rule_stat_retrieve_response import RuleStatRetrieveResponse

__all__ = ["RuleStatsResource", "AsyncRuleStatsResource"]


class RuleStatsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> RuleStatsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return RuleStatsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> RuleStatsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return RuleStatsResourceWithStreamingResponse(self)

    def retrieve(
        self,
        workspace_id: str,
        *,
        framework_id: str | Omit = omit,
        project_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleStatRetrieveResponse:
        """
        Get a compliance roll-up for a workspace: how many rules exist, and how many of
        their results are passing, failing, pending, or due for renewal.

        Counts respect the filters you pass, so `frameworkId` gives you a single
        framework's overall compliance and `projectId` gives you a single project's.

        Args:
          framework_id: Only include items belonging to this framework.

          project_id: Only include items that apply to this project.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._get(
            path_template("/workspaces/{workspace_id}/rule-stats", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "framework_id": framework_id,
                        "project_id": project_id,
                    },
                    rule_stat_retrieve_params.RuleStatRetrieveParams,
                ),
            ),
            cast_to=RuleStatRetrieveResponse,
        )


class AsyncRuleStatsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncRuleStatsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncRuleStatsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncRuleStatsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncRuleStatsResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        workspace_id: str,
        *,
        framework_id: str | Omit = omit,
        project_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleStatRetrieveResponse:
        """
        Get a compliance roll-up for a workspace: how many rules exist, and how many of
        their results are passing, failing, pending, or due for renewal.

        Counts respect the filters you pass, so `frameworkId` gives you a single
        framework's overall compliance and `projectId` gives you a single project's.

        Args:
          framework_id: Only include items belonging to this framework.

          project_id: Only include items that apply to this project.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._get(
            path_template("/workspaces/{workspace_id}/rule-stats", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "framework_id": framework_id,
                        "project_id": project_id,
                    },
                    rule_stat_retrieve_params.RuleStatRetrieveParams,
                ),
            ),
            cast_to=RuleStatRetrieveResponse,
        )


class RuleStatsResourceWithRawResponse:
    def __init__(self, rule_stats: RuleStatsResource) -> None:
        self._rule_stats = rule_stats

        self.retrieve = to_raw_response_wrapper(
            rule_stats.retrieve,
        )


class AsyncRuleStatsResourceWithRawResponse:
    def __init__(self, rule_stats: AsyncRuleStatsResource) -> None:
        self._rule_stats = rule_stats

        self.retrieve = async_to_raw_response_wrapper(
            rule_stats.retrieve,
        )


class RuleStatsResourceWithStreamingResponse:
    def __init__(self, rule_stats: RuleStatsResource) -> None:
        self._rule_stats = rule_stats

        self.retrieve = to_streamed_response_wrapper(
            rule_stats.retrieve,
        )


class AsyncRuleStatsResourceWithStreamingResponse:
    def __init__(self, rule_stats: AsyncRuleStatsResource) -> None:
        self._rule_stats = rule_stats

        self.retrieve = async_to_streamed_response_wrapper(
            rule_stats.retrieve,
        )
