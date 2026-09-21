# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Optional
from typing_extensions import Literal

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
from ...types.governance import rule_list_params, rule_create_params, rule_update_params
from ...types.governance.rule_list_response import RuleListResponse
from ...types.governance.rule_create_response import RuleCreateResponse
from ...types.governance.rule_update_response import RuleUpdateResponse
from ...types.governance.rule_retrieve_response import RuleRetrieveResponse

__all__ = ["RulesResource", "AsyncRulesResource"]


class RulesResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> RulesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return RulesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> RulesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return RulesResourceWithStreamingResponse(self)

    def create(
        self,
        workspace_id: str,
        *,
        name: str,
        scope: Literal["project", "workspace"],
        type: Literal["platform", "evidence"],
        assignee_id: Optional[str] | Omit = omit,
        automation_params: Optional[Dict[str, object]] | Omit = omit,
        automation_type: Optional[str] | Omit = omit,
        deactivated: bool | Omit = omit,
        description: Optional[str] | Omit = omit,
        evidence_type: Optional[Literal["document", "text", "url", "categoryValue"]] | Omit = omit,
        renewal_cadence_days: Optional[int] | Omit = omit,
        tag_ids: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleCreateResponse:
        """Create a governance rule in a workspace.

        A rule is one requirement.

        Its `type` decides how it is satisfied, and the two
        types accept different fields:

        - `platform` rules are evaluated automatically from the state of your workspace.
          Set `automationType` to the signal to check. Their `scope` must be `project`,
          and `evidenceType` and `renewalCadenceDays` must be omitted or `null`.
        - `evidence` rules are satisfied by attaching evidence. Set `evidenceType` to
          the kind of evidence that satisfies them. `automationType` and
          `automationParams` must be omitted or `null`.

        A new rule belongs to no framework. Map it to one from the Openlayer app.

        Args:
          name: The rule name.

          scope: Whether the rule is evaluated once for the whole workspace, or once per project
              the rule's frameworks apply to.

          type: `platform` rules are evaluated automatically from the state of your Openlayer
              workspace. `evidence` rules are satisfied by attaching evidence.

          assignee_id: The user responsible for satisfying the rule.

          automation_params: Configuration for the platform check, when the automation takes parameters.

          automation_type: Which workspace signal a platform rule checks, for example
              `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. `null` for
              evidence rules.

          deactivated: Whether the rule is excluded from compliance calculations.

          description: What the rule requires.

          evidence_type: The kind of evidence that satisfies the rule. `null` for platform rules.

          renewal_cadence_days: How often evidence must be renewed, in days. Once evidence is older than this,
              the rule result becomes `due_soon` and then `failing`.

          tag_ids: The ids of the rule tags to associate with the rule. Replaces the rule's tags.
              Read them back from `tags`, and list the tags available in the workspace with
              `GET /workspaces/{workspaceId}/rule-tags`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._post(
            path_template("/workspaces/{workspace_id}/rules", workspace_id=workspace_id),
            body=maybe_transform(
                {
                    "name": name,
                    "scope": scope,
                    "type": type,
                    "assignee_id": assignee_id,
                    "automation_params": automation_params,
                    "automation_type": automation_type,
                    "deactivated": deactivated,
                    "description": description,
                    "evidence_type": evidence_type,
                    "renewal_cadence_days": renewal_cadence_days,
                    "tag_ids": tag_ids,
                },
                rule_create_params.RuleCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleCreateResponse,
        )

    def retrieve(
        self,
        rule_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleRetrieveResponse:
        """
        Retrieve a governance rule by its id, including the frameworks it belongs to and
        its tags.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        return self._get(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleRetrieveResponse,
        )

    def update(
        self,
        rule_id: str,
        *,
        assignee_id: Optional[str] | Omit = omit,
        deactivated: bool | Omit = omit,
        description: Optional[str] | Omit = omit,
        name: str | Omit = omit,
        renewal_cadence_days: Optional[int] | Omit = omit,
        tag_ids: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleUpdateResponse:
        """Update a governance rule.

        Only the fields you send are changed.

        Rules that ship with Openlayer report `immutable: true` and cannot be edited.

        A rule's `scope`, `type`, `evidenceType`, and automation are fixed once it
        exists -- create a new rule instead of converting one.

        Args:
          assignee_id: The user responsible for satisfying the rule.

          deactivated: Whether the rule is excluded from compliance calculations.

          description: What the rule requires.

          name: The rule name.

          renewal_cadence_days: How often evidence must be renewed, in days. Once evidence is older than this,
              the rule result becomes `due_soon` and then `failing`.

          tag_ids: The ids of the rule tags to associate with the rule. Replaces the rule's tags.
              Read them back from `tags`, and list the tags available in the workspace with
              `GET /workspaces/{workspaceId}/rule-tags`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        return self._put(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            body=maybe_transform(
                {
                    "assignee_id": assignee_id,
                    "deactivated": deactivated,
                    "description": description,
                    "name": name,
                    "renewal_cadence_days": renewal_cadence_days,
                    "tag_ids": tag_ids,
                },
                rule_update_params.RuleUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleUpdateResponse,
        )

    def list(
        self,
        workspace_id: str,
        *,
        asc: bool | Omit = omit,
        assignee_id: str | Omit = omit,
        deactivated: bool | Omit = omit,
        enabled_framework_only: bool | Omit = omit,
        framework_id: str | Omit = omit,
        group: Literal["open", "excluded", "done"] | Omit = omit,
        include_results: bool | Omit = omit,
        include_unframed: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        scope: Literal["project", "workspace"] | Omit = omit,
        search_query: str | Omit = omit,
        sort_by: Literal["name", "status", "frameworks", "scope", "dateCreated"] | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        type: Literal["platform", "evidence"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleListResponse:
        """
        List the governance rules in a workspace.

        A rule is a single requirement Openlayer tracks. `platform` rules are evaluated
        automatically from the state of your workspace; `evidence` rules are satisfied
        by attaching evidence. A rule can belong to several frameworks at once, and
        rules that belong to none are returned too unless you pass
        `includeUnframed=false`.

        Pass `includeResults=true` to get each rule's compliance results inline instead
        of fetching them separately.

        Args:
          asc: Whether to sort in ascending order.

          assignee_id: Only include rules assigned to this user.

          deactivated: Only include rules that are deactivated (or active).

          enabled_framework_only: Only include items belonging to at least one enabled framework.

          framework_id: Only include items belonging to this framework.

          group: Only include rules in one bucket of the compliance workflow. `open` covers rules
              that still need attention, `done` covers rules that are fully satisfied, and
              `excluded` covers rules that have been deactivated.

          include_results: Whether to include each rule's results inline, in a `results` array.

          include_unframed: Whether to include rules that are not part of any framework.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          scope: Only include rules with this scope.

          search_query: Filter by a free-text search over names and descriptions.

          sort_by: The field to sort on.

          status: Only include items whose rule result has this compliance status.

          tags: Only include rules carrying all of these rule tags. Pass tag ids, which you can
              look up with [List rule tags](/api-reference/rest/governance/list-rule-tags).

          type: Only include rules of this type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._get(
            path_template("/workspaces/{workspace_id}/rules", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "asc": asc,
                        "assignee_id": assignee_id,
                        "deactivated": deactivated,
                        "enabled_framework_only": enabled_framework_only,
                        "framework_id": framework_id,
                        "group": group,
                        "include_results": include_results,
                        "include_unframed": include_unframed,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "scope": scope,
                        "search_query": search_query,
                        "sort_by": sort_by,
                        "status": status,
                        "tags": tags,
                        "type": type,
                    },
                    rule_list_params.RuleListParams,
                ),
            ),
            cast_to=RuleListResponse,
        )

    def delete(
        self,
        rule_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete a governance rule and its rule results.

        Only rules you created can be deleted. Rules that ship with Openlayer report
        `immutable: true` and cannot be deleted -- exclude one from compliance by
        setting `deactivated` with `PUT /rules/{ruleId}` instead.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncRulesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncRulesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncRulesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncRulesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncRulesResourceWithStreamingResponse(self)

    async def create(
        self,
        workspace_id: str,
        *,
        name: str,
        scope: Literal["project", "workspace"],
        type: Literal["platform", "evidence"],
        assignee_id: Optional[str] | Omit = omit,
        automation_params: Optional[Dict[str, object]] | Omit = omit,
        automation_type: Optional[str] | Omit = omit,
        deactivated: bool | Omit = omit,
        description: Optional[str] | Omit = omit,
        evidence_type: Optional[Literal["document", "text", "url", "categoryValue"]] | Omit = omit,
        renewal_cadence_days: Optional[int] | Omit = omit,
        tag_ids: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleCreateResponse:
        """Create a governance rule in a workspace.

        A rule is one requirement.

        Its `type` decides how it is satisfied, and the two
        types accept different fields:

        - `platform` rules are evaluated automatically from the state of your workspace.
          Set `automationType` to the signal to check. Their `scope` must be `project`,
          and `evidenceType` and `renewalCadenceDays` must be omitted or `null`.
        - `evidence` rules are satisfied by attaching evidence. Set `evidenceType` to
          the kind of evidence that satisfies them. `automationType` and
          `automationParams` must be omitted or `null`.

        A new rule belongs to no framework. Map it to one from the Openlayer app.

        Args:
          name: The rule name.

          scope: Whether the rule is evaluated once for the whole workspace, or once per project
              the rule's frameworks apply to.

          type: `platform` rules are evaluated automatically from the state of your Openlayer
              workspace. `evidence` rules are satisfied by attaching evidence.

          assignee_id: The user responsible for satisfying the rule.

          automation_params: Configuration for the platform check, when the automation takes parameters.

          automation_type: Which workspace signal a platform rule checks, for example
              `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. `null` for
              evidence rules.

          deactivated: Whether the rule is excluded from compliance calculations.

          description: What the rule requires.

          evidence_type: The kind of evidence that satisfies the rule. `null` for platform rules.

          renewal_cadence_days: How often evidence must be renewed, in days. Once evidence is older than this,
              the rule result becomes `due_soon` and then `failing`.

          tag_ids: The ids of the rule tags to associate with the rule. Replaces the rule's tags.
              Read them back from `tags`, and list the tags available in the workspace with
              `GET /workspaces/{workspaceId}/rule-tags`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._post(
            path_template("/workspaces/{workspace_id}/rules", workspace_id=workspace_id),
            body=await async_maybe_transform(
                {
                    "name": name,
                    "scope": scope,
                    "type": type,
                    "assignee_id": assignee_id,
                    "automation_params": automation_params,
                    "automation_type": automation_type,
                    "deactivated": deactivated,
                    "description": description,
                    "evidence_type": evidence_type,
                    "renewal_cadence_days": renewal_cadence_days,
                    "tag_ids": tag_ids,
                },
                rule_create_params.RuleCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleCreateResponse,
        )

    async def retrieve(
        self,
        rule_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleRetrieveResponse:
        """
        Retrieve a governance rule by its id, including the frameworks it belongs to and
        its tags.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        return await self._get(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleRetrieveResponse,
        )

    async def update(
        self,
        rule_id: str,
        *,
        assignee_id: Optional[str] | Omit = omit,
        deactivated: bool | Omit = omit,
        description: Optional[str] | Omit = omit,
        name: str | Omit = omit,
        renewal_cadence_days: Optional[int] | Omit = omit,
        tag_ids: Optional[SequenceNotStr[str]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleUpdateResponse:
        """Update a governance rule.

        Only the fields you send are changed.

        Rules that ship with Openlayer report `immutable: true` and cannot be edited.

        A rule's `scope`, `type`, `evidenceType`, and automation are fixed once it
        exists -- create a new rule instead of converting one.

        Args:
          assignee_id: The user responsible for satisfying the rule.

          deactivated: Whether the rule is excluded from compliance calculations.

          description: What the rule requires.

          name: The rule name.

          renewal_cadence_days: How often evidence must be renewed, in days. Once evidence is older than this,
              the rule result becomes `due_soon` and then `failing`.

          tag_ids: The ids of the rule tags to associate with the rule. Replaces the rule's tags.
              Read them back from `tags`, and list the tags available in the workspace with
              `GET /workspaces/{workspaceId}/rule-tags`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        return await self._put(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            body=await async_maybe_transform(
                {
                    "assignee_id": assignee_id,
                    "deactivated": deactivated,
                    "description": description,
                    "name": name,
                    "renewal_cadence_days": renewal_cadence_days,
                    "tag_ids": tag_ids,
                },
                rule_update_params.RuleUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleUpdateResponse,
        )

    async def list(
        self,
        workspace_id: str,
        *,
        asc: bool | Omit = omit,
        assignee_id: str | Omit = omit,
        deactivated: bool | Omit = omit,
        enabled_framework_only: bool | Omit = omit,
        framework_id: str | Omit = omit,
        group: Literal["open", "excluded", "done"] | Omit = omit,
        include_results: bool | Omit = omit,
        include_unframed: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        scope: Literal["project", "workspace"] | Omit = omit,
        search_query: str | Omit = omit,
        sort_by: Literal["name", "status", "frameworks", "scope", "dateCreated"] | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        tags: SequenceNotStr[str] | Omit = omit,
        type: Literal["platform", "evidence"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleListResponse:
        """
        List the governance rules in a workspace.

        A rule is a single requirement Openlayer tracks. `platform` rules are evaluated
        automatically from the state of your workspace; `evidence` rules are satisfied
        by attaching evidence. A rule can belong to several frameworks at once, and
        rules that belong to none are returned too unless you pass
        `includeUnframed=false`.

        Pass `includeResults=true` to get each rule's compliance results inline instead
        of fetching them separately.

        Args:
          asc: Whether to sort in ascending order.

          assignee_id: Only include rules assigned to this user.

          deactivated: Only include rules that are deactivated (or active).

          enabled_framework_only: Only include items belonging to at least one enabled framework.

          framework_id: Only include items belonging to this framework.

          group: Only include rules in one bucket of the compliance workflow. `open` covers rules
              that still need attention, `done` covers rules that are fully satisfied, and
              `excluded` covers rules that have been deactivated.

          include_results: Whether to include each rule's results inline, in a `results` array.

          include_unframed: Whether to include rules that are not part of any framework.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          scope: Only include rules with this scope.

          search_query: Filter by a free-text search over names and descriptions.

          sort_by: The field to sort on.

          status: Only include items whose rule result has this compliance status.

          tags: Only include rules carrying all of these rule tags. Pass tag ids, which you can
              look up with [List rule tags](/api-reference/rest/governance/list-rule-tags).

          type: Only include rules of this type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._get(
            path_template("/workspaces/{workspace_id}/rules", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "asc": asc,
                        "assignee_id": assignee_id,
                        "deactivated": deactivated,
                        "enabled_framework_only": enabled_framework_only,
                        "framework_id": framework_id,
                        "group": group,
                        "include_results": include_results,
                        "include_unframed": include_unframed,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "scope": scope,
                        "search_query": search_query,
                        "sort_by": sort_by,
                        "status": status,
                        "tags": tags,
                        "type": type,
                    },
                    rule_list_params.RuleListParams,
                ),
            ),
            cast_to=RuleListResponse,
        )

    async def delete(
        self,
        rule_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete a governance rule and its rule results.

        Only rules you created can be deleted. Rules that ship with Openlayer report
        `immutable: true` and cannot be deleted -- exclude one from compliance by
        setting `deactivated` with `PUT /rules/{ruleId}` instead.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_id:
            raise ValueError(f"Expected a non-empty value for `rule_id` but received {rule_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template("/rules/{rule_id}", rule_id=rule_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class RulesResourceWithRawResponse:
    def __init__(self, rules: RulesResource) -> None:
        self._rules = rules

        self.create = to_raw_response_wrapper(
            rules.create,
        )
        self.retrieve = to_raw_response_wrapper(
            rules.retrieve,
        )
        self.update = to_raw_response_wrapper(
            rules.update,
        )
        self.list = to_raw_response_wrapper(
            rules.list,
        )
        self.delete = to_raw_response_wrapper(
            rules.delete,
        )


class AsyncRulesResourceWithRawResponse:
    def __init__(self, rules: AsyncRulesResource) -> None:
        self._rules = rules

        self.create = async_to_raw_response_wrapper(
            rules.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            rules.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            rules.update,
        )
        self.list = async_to_raw_response_wrapper(
            rules.list,
        )
        self.delete = async_to_raw_response_wrapper(
            rules.delete,
        )


class RulesResourceWithStreamingResponse:
    def __init__(self, rules: RulesResource) -> None:
        self._rules = rules

        self.create = to_streamed_response_wrapper(
            rules.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            rules.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            rules.update,
        )
        self.list = to_streamed_response_wrapper(
            rules.list,
        )
        self.delete = to_streamed_response_wrapper(
            rules.delete,
        )


class AsyncRulesResourceWithStreamingResponse:
    def __init__(self, rules: AsyncRulesResource) -> None:
        self._rules = rules

        self.create = async_to_streamed_response_wrapper(
            rules.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            rules.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            rules.update,
        )
        self.list = async_to_streamed_response_wrapper(
            rules.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            rules.delete,
        )
