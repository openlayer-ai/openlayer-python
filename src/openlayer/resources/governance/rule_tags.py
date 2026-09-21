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
from ...types.governance import rule_tag_list_params
from ...types.governance.rule_tag_list_response import RuleTagListResponse

__all__ = ["RuleTagsResource", "AsyncRuleTagsResource"]


class RuleTagsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> RuleTagsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return RuleTagsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> RuleTagsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return RuleTagsResourceWithStreamingResponse(self)

    def list(
        self,
        workspace_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleTagListResponse:
        """
        List the rule tags in a workspace.

        Tags group rules across frameworks, for example by team or by control family.
        Use the ids returned here with the `tags` filter on
        [List rules](/api-reference/rest/governance/list-rules).

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._get(
            path_template("/workspaces/{workspace_id}/rule-tags", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "page": page,
                        "per_page": per_page,
                    },
                    rule_tag_list_params.RuleTagListParams,
                ),
            ),
            cast_to=RuleTagListResponse,
        )


class AsyncRuleTagsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncRuleTagsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncRuleTagsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncRuleTagsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncRuleTagsResourceWithStreamingResponse(self)

    async def list(
        self,
        workspace_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleTagListResponse:
        """
        List the rule tags in a workspace.

        Tags group rules across frameworks, for example by team or by control family.
        Use the ids returned here with the `tags` filter on
        [List rules](/api-reference/rest/governance/list-rules).

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._get(
            path_template("/workspaces/{workspace_id}/rule-tags", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "page": page,
                        "per_page": per_page,
                    },
                    rule_tag_list_params.RuleTagListParams,
                ),
            ),
            cast_to=RuleTagListResponse,
        )


class RuleTagsResourceWithRawResponse:
    def __init__(self, rule_tags: RuleTagsResource) -> None:
        self._rule_tags = rule_tags

        self.list = to_raw_response_wrapper(
            rule_tags.list,
        )


class AsyncRuleTagsResourceWithRawResponse:
    def __init__(self, rule_tags: AsyncRuleTagsResource) -> None:
        self._rule_tags = rule_tags

        self.list = async_to_raw_response_wrapper(
            rule_tags.list,
        )


class RuleTagsResourceWithStreamingResponse:
    def __init__(self, rule_tags: RuleTagsResource) -> None:
        self._rule_tags = rule_tags

        self.list = to_streamed_response_wrapper(
            rule_tags.list,
        )


class AsyncRuleTagsResourceWithStreamingResponse:
    def __init__(self, rule_tags: AsyncRuleTagsResource) -> None:
        self._rule_tags = rule_tags

        self.list = async_to_streamed_response_wrapper(
            rule_tags.list,
        )
