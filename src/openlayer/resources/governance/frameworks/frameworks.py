# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Optional
from typing_extensions import Literal

import httpx

from .sections import (
    SectionsResource,
    AsyncSectionsResource,
    SectionsResourceWithRawResponse,
    AsyncSectionsResourceWithRawResponse,
    SectionsResourceWithStreamingResponse,
    AsyncSectionsResourceWithStreamingResponse,
)
from ...._types import Body, Omit, Query, Headers, NotGiven, SequenceNotStr, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from .documents import (
    DocumentsResource,
    AsyncDocumentsResource,
    DocumentsResourceWithRawResponse,
    AsyncDocumentsResourceWithRawResponse,
    DocumentsResourceWithStreamingResponse,
    AsyncDocumentsResourceWithStreamingResponse,
)
from ...._compat import cached_property
from .subsections import (
    SubsectionsResource,
    AsyncSubsectionsResource,
    SubsectionsResourceWithRawResponse,
    AsyncSubsectionsResourceWithRawResponse,
    SubsectionsResourceWithStreamingResponse,
    AsyncSubsectionsResourceWithStreamingResponse,
)
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.governance import (
    framework_list_params,
    framework_create_params,
    framework_export_params,
    framework_update_params,
    framework_list_rules_params,
    framework_list_projects_params,
    framework_list_project_rule_stats_params,
)
from ....types.governance.framework_list_response import FrameworkListResponse
from ....types.governance.framework_create_response import FrameworkCreateResponse
from ....types.governance.framework_export_response import FrameworkExportResponse
from ....types.governance.framework_update_response import FrameworkUpdateResponse
from ....types.governance.framework_retrieve_response import FrameworkRetrieveResponse
from ....types.governance.framework_list_rules_response import FrameworkListRulesResponse
from ....types.governance.framework_list_projects_response import FrameworkListProjectsResponse
from ....types.governance.framework_list_project_rule_stats_response import FrameworkListProjectRuleStatsResponse

__all__ = ["FrameworksResource", "AsyncFrameworksResource"]


class FrameworksResource(SyncAPIResource):
    @cached_property
    def documents(self) -> DocumentsResource:
        return DocumentsResource(self._client)

    @cached_property
    def sections(self) -> SectionsResource:
        return SectionsResource(self._client)

    @cached_property
    def subsections(self) -> SubsectionsResource:
        return SubsectionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> FrameworksResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return FrameworksResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FrameworksResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return FrameworksResourceWithStreamingResponse(self)

    def create(
        self,
        workspace_id: str,
        *,
        name: str,
        description: Optional[str] | Omit = omit,
        enabled: bool | Omit = omit,
        project_selector: Optional[framework_create_params.ProjectSelector] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkCreateResponse:
        """
        Create a custom framework in a workspace.

        Args:
          name: The framework name.

          description: A short description of the framework.

          enabled: Whether the framework is active. Rules of a disabled framework are not evaluated
              and do not count towards compliance.

          project_selector: Determines which projects the framework applies to. An empty or `null` `match`
              array applies the framework to every project in the workspace.

          tags: Free-form labels on the framework.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._post(
            path_template("/workspaces/{workspace_id}/frameworks", workspace_id=workspace_id),
            body=maybe_transform(
                {
                    "name": name,
                    "description": description,
                    "enabled": enabled,
                    "project_selector": project_selector,
                    "tags": tags,
                },
                framework_create_params.FrameworkCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkCreateResponse,
        )

    def retrieve(
        self,
        framework_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkRetrieveResponse:
        """
        Retrieve a framework.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._get(
            path_template("/frameworks/{framework_id}", framework_id=framework_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkRetrieveResponse,
        )

    def update(
        self,
        framework_id: str,
        *,
        avatar: Optional[framework_update_params.Avatar] | Omit = omit,
        description: Optional[str] | Omit = omit,
        enabled: bool | Omit = omit,
        extended_description: Optional[Dict[str, object]] | Omit = omit,
        href: Optional[str] | Omit = omit,
        name: str | Omit = omit,
        project_selector: Optional[framework_update_params.ProjectSelector] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkUpdateResponse:
        """
        Update a framework.

        Args:
          avatar: The icon shown for the framework.

          description: A short description of the framework.

          enabled: Whether the framework is active. Rules of a disabled framework are not evaluated
              and do not count towards compliance.

          extended_description: A longer, rich-text description, as a TipTap JSON document.

          href: A link to the external standard or regulation the framework is based on.

          name: The framework name.

          project_selector: Determines which projects the framework applies to. An empty or `null` `match`
              array applies the framework to every project in the workspace.

          tags: Free-form labels on the framework.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._put(
            path_template("/frameworks/{framework_id}", framework_id=framework_id),
            body=maybe_transform(
                {
                    "avatar": avatar,
                    "description": description,
                    "enabled": enabled,
                    "extended_description": extended_description,
                    "href": href,
                    "name": name,
                    "project_selector": project_selector,
                    "tags": tags,
                },
                framework_update_params.FrameworkUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkUpdateResponse,
        )

    def list(
        self,
        workspace_id: str,
        *,
        asc: bool | Omit = omit,
        completion_operator: Literal["is", ">", ">=", "<", "<=", "!="] | Omit = omit,
        completion_value: int | Omit = omit,
        enabled: bool | Omit = omit,
        include_rule_stats: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        search_query: str | Omit = omit,
        sort_column: Literal[
            "name", "enabled", "dateCreated", "dateUpdated", "overallCompletion", "projectCompletionBuckets"
        ]
        | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListResponse:
        """
        List the frameworks in a workspace.

        Args:
          asc: Whether to sort in ascending order.

          completion_operator: How to compare each framework's completion percentage with `completionValue`.
              Must be sent together with `completionValue`.

          completion_value: The completion percentage to compare against, from 0 to 100.

          enabled: Only include frameworks that are enabled (or disabled).

          include_rule_stats: Whether to include a `ruleStats` object on each framework, with its rule result
              status counts and its per-project completion buckets. Computed over the returned
              page only.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          search_query: Filter by a free-text search over names and descriptions.

          sort_column: The column to sort on.

          tags: Only include frameworks carrying all of these tags.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._get(
            path_template("/workspaces/{workspace_id}/frameworks", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "asc": asc,
                        "completion_operator": completion_operator,
                        "completion_value": completion_value,
                        "enabled": enabled,
                        "include_rule_stats": include_rule_stats,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "search_query": search_query,
                        "sort_column": sort_column,
                        "tags": tags,
                    },
                    framework_list_params.FrameworkListParams,
                ),
            ),
            cast_to=FrameworkListResponse,
        )

    def export(
        self,
        framework_id: str,
        *,
        project_id: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkExportResponse:
        """
        Export a framework as an audit-ready zip archive.

        Args:
          project_id: Scope the export to this project. It must belong to the framework. Omit it for
              the workspace-wide view across every project in the framework, including
              workspace-scoped rules.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._post(
            path_template("/frameworks/{framework_id}/export", framework_id=framework_id),
            body=maybe_transform({"project_id": project_id}, framework_export_params.FrameworkExportParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkExportResponse,
        )

    def list_project_rule_stats(
        self,
        framework_id: str,
        *,
        asc: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        sort_column: Literal[
            "projectName",
            "total",
            "overallCompletion",
            "totalPassing",
            "totalFailing",
            "totalSkipped",
            "totalRunning",
            "totalError",
            "totalPending",
            "totalDueSoon",
        ]
        | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListProjectRuleStatsResponse:
        """
        List a framework's compliance stats per project.

        Args:
          asc: Whether to sort in ascending order.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          sort_column: The column to sort on.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._get(
            path_template("/frameworks/{framework_id}/project-rule-stats", framework_id=framework_id),
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
                    framework_list_project_rule_stats_params.FrameworkListProjectRuleStatsParams,
                ),
            ),
            cast_to=FrameworkListProjectRuleStatsResponse,
        )

    def list_projects(
        self,
        framework_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListProjectsResponse:
        """
        List the projects a framework applies to.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._get(
            path_template("/frameworks/{framework_id}/projects", framework_id=framework_id),
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
                    framework_list_projects_params.FrameworkListProjectsParams,
                ),
            ),
            cast_to=FrameworkListProjectsResponse,
        )

    def list_rules(
        self,
        framework_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListRulesResponse:
        """
        List the rules in a framework.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return self._get(
            path_template("/frameworks/{framework_id}/rules", framework_id=framework_id),
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
                    framework_list_rules_params.FrameworkListRulesParams,
                ),
            ),
            cast_to=FrameworkListRulesResponse,
        )


class AsyncFrameworksResource(AsyncAPIResource):
    @cached_property
    def documents(self) -> AsyncDocumentsResource:
        return AsyncDocumentsResource(self._client)

    @cached_property
    def sections(self) -> AsyncSectionsResource:
        return AsyncSectionsResource(self._client)

    @cached_property
    def subsections(self) -> AsyncSubsectionsResource:
        return AsyncSubsectionsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncFrameworksResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncFrameworksResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFrameworksResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncFrameworksResourceWithStreamingResponse(self)

    async def create(
        self,
        workspace_id: str,
        *,
        name: str,
        description: Optional[str] | Omit = omit,
        enabled: bool | Omit = omit,
        project_selector: Optional[framework_create_params.ProjectSelector] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkCreateResponse:
        """
        Create a custom framework in a workspace.

        Args:
          name: The framework name.

          description: A short description of the framework.

          enabled: Whether the framework is active. Rules of a disabled framework are not evaluated
              and do not count towards compliance.

          project_selector: Determines which projects the framework applies to. An empty or `null` `match`
              array applies the framework to every project in the workspace.

          tags: Free-form labels on the framework.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._post(
            path_template("/workspaces/{workspace_id}/frameworks", workspace_id=workspace_id),
            body=await async_maybe_transform(
                {
                    "name": name,
                    "description": description,
                    "enabled": enabled,
                    "project_selector": project_selector,
                    "tags": tags,
                },
                framework_create_params.FrameworkCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkCreateResponse,
        )

    async def retrieve(
        self,
        framework_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkRetrieveResponse:
        """
        Retrieve a framework.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._get(
            path_template("/frameworks/{framework_id}", framework_id=framework_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkRetrieveResponse,
        )

    async def update(
        self,
        framework_id: str,
        *,
        avatar: Optional[framework_update_params.Avatar] | Omit = omit,
        description: Optional[str] | Omit = omit,
        enabled: bool | Omit = omit,
        extended_description: Optional[Dict[str, object]] | Omit = omit,
        href: Optional[str] | Omit = omit,
        name: str | Omit = omit,
        project_selector: Optional[framework_update_params.ProjectSelector] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkUpdateResponse:
        """
        Update a framework.

        Args:
          avatar: The icon shown for the framework.

          description: A short description of the framework.

          enabled: Whether the framework is active. Rules of a disabled framework are not evaluated
              and do not count towards compliance.

          extended_description: A longer, rich-text description, as a TipTap JSON document.

          href: A link to the external standard or regulation the framework is based on.

          name: The framework name.

          project_selector: Determines which projects the framework applies to. An empty or `null` `match`
              array applies the framework to every project in the workspace.

          tags: Free-form labels on the framework.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._put(
            path_template("/frameworks/{framework_id}", framework_id=framework_id),
            body=await async_maybe_transform(
                {
                    "avatar": avatar,
                    "description": description,
                    "enabled": enabled,
                    "extended_description": extended_description,
                    "href": href,
                    "name": name,
                    "project_selector": project_selector,
                    "tags": tags,
                },
                framework_update_params.FrameworkUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkUpdateResponse,
        )

    async def list(
        self,
        workspace_id: str,
        *,
        asc: bool | Omit = omit,
        completion_operator: Literal["is", ">", ">=", "<", "<=", "!="] | Omit = omit,
        completion_value: int | Omit = omit,
        enabled: bool | Omit = omit,
        include_rule_stats: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        search_query: str | Omit = omit,
        sort_column: Literal[
            "name", "enabled", "dateCreated", "dateUpdated", "overallCompletion", "projectCompletionBuckets"
        ]
        | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListResponse:
        """
        List the frameworks in a workspace.

        Args:
          asc: Whether to sort in ascending order.

          completion_operator: How to compare each framework's completion percentage with `completionValue`.
              Must be sent together with `completionValue`.

          completion_value: The completion percentage to compare against, from 0 to 100.

          enabled: Only include frameworks that are enabled (or disabled).

          include_rule_stats: Whether to include a `ruleStats` object on each framework, with its rule result
              status counts and its per-project completion buckets. Computed over the returned
              page only.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          search_query: Filter by a free-text search over names and descriptions.

          sort_column: The column to sort on.

          tags: Only include frameworks carrying all of these tags.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._get(
            path_template("/workspaces/{workspace_id}/frameworks", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "asc": asc,
                        "completion_operator": completion_operator,
                        "completion_value": completion_value,
                        "enabled": enabled,
                        "include_rule_stats": include_rule_stats,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "search_query": search_query,
                        "sort_column": sort_column,
                        "tags": tags,
                    },
                    framework_list_params.FrameworkListParams,
                ),
            ),
            cast_to=FrameworkListResponse,
        )

    async def export(
        self,
        framework_id: str,
        *,
        project_id: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkExportResponse:
        """
        Export a framework as an audit-ready zip archive.

        Args:
          project_id: Scope the export to this project. It must belong to the framework. Omit it for
              the workspace-wide view across every project in the framework, including
              workspace-scoped rules.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._post(
            path_template("/frameworks/{framework_id}/export", framework_id=framework_id),
            body=await async_maybe_transform({"project_id": project_id}, framework_export_params.FrameworkExportParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FrameworkExportResponse,
        )

    async def list_project_rule_stats(
        self,
        framework_id: str,
        *,
        asc: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        sort_column: Literal[
            "projectName",
            "total",
            "overallCompletion",
            "totalPassing",
            "totalFailing",
            "totalSkipped",
            "totalRunning",
            "totalError",
            "totalPending",
            "totalDueSoon",
        ]
        | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListProjectRuleStatsResponse:
        """
        List a framework's compliance stats per project.

        Args:
          asc: Whether to sort in ascending order.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          sort_column: The column to sort on.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._get(
            path_template("/frameworks/{framework_id}/project-rule-stats", framework_id=framework_id),
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
                    framework_list_project_rule_stats_params.FrameworkListProjectRuleStatsParams,
                ),
            ),
            cast_to=FrameworkListProjectRuleStatsResponse,
        )

    async def list_projects(
        self,
        framework_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListProjectsResponse:
        """
        List the projects a framework applies to.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._get(
            path_template("/frameworks/{framework_id}/projects", framework_id=framework_id),
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
                    framework_list_projects_params.FrameworkListProjectsParams,
                ),
            ),
            cast_to=FrameworkListProjectsResponse,
        )

    async def list_rules(
        self,
        framework_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> FrameworkListRulesResponse:
        """
        List the rules in a framework.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        return await self._get(
            path_template("/frameworks/{framework_id}/rules", framework_id=framework_id),
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
                    framework_list_rules_params.FrameworkListRulesParams,
                ),
            ),
            cast_to=FrameworkListRulesResponse,
        )


class FrameworksResourceWithRawResponse:
    def __init__(self, frameworks: FrameworksResource) -> None:
        self._frameworks = frameworks

        self.create = to_raw_response_wrapper(
            frameworks.create,
        )
        self.retrieve = to_raw_response_wrapper(
            frameworks.retrieve,
        )
        self.update = to_raw_response_wrapper(
            frameworks.update,
        )
        self.list = to_raw_response_wrapper(
            frameworks.list,
        )
        self.export = to_raw_response_wrapper(
            frameworks.export,
        )
        self.list_project_rule_stats = to_raw_response_wrapper(
            frameworks.list_project_rule_stats,
        )
        self.list_projects = to_raw_response_wrapper(
            frameworks.list_projects,
        )
        self.list_rules = to_raw_response_wrapper(
            frameworks.list_rules,
        )

    @cached_property
    def documents(self) -> DocumentsResourceWithRawResponse:
        return DocumentsResourceWithRawResponse(self._frameworks.documents)

    @cached_property
    def sections(self) -> SectionsResourceWithRawResponse:
        return SectionsResourceWithRawResponse(self._frameworks.sections)

    @cached_property
    def subsections(self) -> SubsectionsResourceWithRawResponse:
        return SubsectionsResourceWithRawResponse(self._frameworks.subsections)


class AsyncFrameworksResourceWithRawResponse:
    def __init__(self, frameworks: AsyncFrameworksResource) -> None:
        self._frameworks = frameworks

        self.create = async_to_raw_response_wrapper(
            frameworks.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            frameworks.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            frameworks.update,
        )
        self.list = async_to_raw_response_wrapper(
            frameworks.list,
        )
        self.export = async_to_raw_response_wrapper(
            frameworks.export,
        )
        self.list_project_rule_stats = async_to_raw_response_wrapper(
            frameworks.list_project_rule_stats,
        )
        self.list_projects = async_to_raw_response_wrapper(
            frameworks.list_projects,
        )
        self.list_rules = async_to_raw_response_wrapper(
            frameworks.list_rules,
        )

    @cached_property
    def documents(self) -> AsyncDocumentsResourceWithRawResponse:
        return AsyncDocumentsResourceWithRawResponse(self._frameworks.documents)

    @cached_property
    def sections(self) -> AsyncSectionsResourceWithRawResponse:
        return AsyncSectionsResourceWithRawResponse(self._frameworks.sections)

    @cached_property
    def subsections(self) -> AsyncSubsectionsResourceWithRawResponse:
        return AsyncSubsectionsResourceWithRawResponse(self._frameworks.subsections)


class FrameworksResourceWithStreamingResponse:
    def __init__(self, frameworks: FrameworksResource) -> None:
        self._frameworks = frameworks

        self.create = to_streamed_response_wrapper(
            frameworks.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            frameworks.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            frameworks.update,
        )
        self.list = to_streamed_response_wrapper(
            frameworks.list,
        )
        self.export = to_streamed_response_wrapper(
            frameworks.export,
        )
        self.list_project_rule_stats = to_streamed_response_wrapper(
            frameworks.list_project_rule_stats,
        )
        self.list_projects = to_streamed_response_wrapper(
            frameworks.list_projects,
        )
        self.list_rules = to_streamed_response_wrapper(
            frameworks.list_rules,
        )

    @cached_property
    def documents(self) -> DocumentsResourceWithStreamingResponse:
        return DocumentsResourceWithStreamingResponse(self._frameworks.documents)

    @cached_property
    def sections(self) -> SectionsResourceWithStreamingResponse:
        return SectionsResourceWithStreamingResponse(self._frameworks.sections)

    @cached_property
    def subsections(self) -> SubsectionsResourceWithStreamingResponse:
        return SubsectionsResourceWithStreamingResponse(self._frameworks.subsections)


class AsyncFrameworksResourceWithStreamingResponse:
    def __init__(self, frameworks: AsyncFrameworksResource) -> None:
        self._frameworks = frameworks

        self.create = async_to_streamed_response_wrapper(
            frameworks.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            frameworks.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            frameworks.update,
        )
        self.list = async_to_streamed_response_wrapper(
            frameworks.list,
        )
        self.export = async_to_streamed_response_wrapper(
            frameworks.export,
        )
        self.list_project_rule_stats = async_to_streamed_response_wrapper(
            frameworks.list_project_rule_stats,
        )
        self.list_projects = async_to_streamed_response_wrapper(
            frameworks.list_projects,
        )
        self.list_rules = async_to_streamed_response_wrapper(
            frameworks.list_rules,
        )

    @cached_property
    def documents(self) -> AsyncDocumentsResourceWithStreamingResponse:
        return AsyncDocumentsResourceWithStreamingResponse(self._frameworks.documents)

    @cached_property
    def sections(self) -> AsyncSectionsResourceWithStreamingResponse:
        return AsyncSectionsResourceWithStreamingResponse(self._frameworks.sections)

    @cached_property
    def subsections(self) -> AsyncSubsectionsResourceWithStreamingResponse:
        return AsyncSubsectionsResourceWithStreamingResponse(self._frameworks.subsections)
