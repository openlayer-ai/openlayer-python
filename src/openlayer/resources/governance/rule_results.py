# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional
from typing_extensions import Literal

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
from ...types.governance import (
    rule_result_list_params,
    rule_result_update_params,
    rule_result_list_evidence_params,
    rule_result_create_evidence_params,
)
from ...types.governance.rule_result_list_response import RuleResultListResponse
from ...types.governance.rule_result_update_response import RuleResultUpdateResponse
from ...types.governance.rule_result_retrieve_response import RuleResultRetrieveResponse
from ...types.governance.rule_result_list_evidence_response import RuleResultListEvidenceResponse
from ...types.governance.rule_result_create_evidence_response import RuleResultCreateEvidenceResponse

__all__ = ["RuleResultsResource", "AsyncRuleResultsResource"]


class RuleResultsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> RuleResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return RuleResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> RuleResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return RuleResultsResourceWithStreamingResponse(self)

    def retrieve(
        self,
        rule_result_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultRetrieveResponse:
        """
        Retrieve a rule result by its id.

        Alongside the status, the response carries the evaluation and renewal dates that
        explain it: `dateLastEvaluated` and `dateOfNextEvaluation` for platform rules,
        `dateOfLatestEvidence` and `dateOfRenewal` for evidence rules.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return self._get(
            path_template("/rule-results/{rule_result_id}", rule_result_id=rule_result_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultRetrieveResponse,
        )

    def update(
        self,
        rule_result_id: str,
        *,
        assignee_id: Optional[str] | Omit = omit,
        blocked_by: Iterable[rule_result_update_params.BlockedBy] | Omit = omit,
        blocking: Iterable[rule_result_update_params.Blocking] | Omit = omit,
        deactivated: bool | Omit = omit,
        deactivated_reason: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultUpdateResponse:
        """Update a rule result.

        Only the fields you send are changed.

        Use this to assign an owner, or to exclude a single result from compliance
        without deactivating the rule everywhere. `deactivatedReason` is required when
        setting `deactivated` to `true`.

        A result's `status` is computed by Openlayer and cannot be set directly.

        Args:
          assignee_id: The user responsible for this result.

          blocked_by: Rule results that must pass before this one can be satisfied.

          blocking: Rule results that this one blocks.

          deactivated: Whether this result is excluded from compliance calculations.

          deactivated_reason: Why the result was excluded.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return self._patch(
            path_template("/rule-results/{rule_result_id}", rule_result_id=rule_result_id),
            body=maybe_transform(
                {
                    "assignee_id": assignee_id,
                    "blocked_by": blocked_by,
                    "blocking": blocking,
                    "deactivated": deactivated,
                    "deactivated_reason": deactivated_reason,
                },
                rule_result_update_params.RuleResultUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultUpdateResponse,
        )

    def list(
        self,
        workspace_id: str,
        *,
        enabled_framework_only: bool | Omit = omit,
        framework_id: str | Omit = omit,
        include_unframed: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        rule_id: str | Omit = omit,
        scope: Literal["project", "workspace"] | Omit = omit,
        search_query: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        type: Literal["platform", "evidence"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultListResponse:
        """
        List rule results across a workspace.

        A rule result is the compliance status of one rule for one entity: a project for
        project-scoped rules, or the workspace itself for workspace-scoped rules. This
        is the endpoint to poll or export when you want your current compliance state,
        filtered to a framework, a project, or a status.

        Args:
          enabled_framework_only: Only include items belonging to at least one enabled framework.

          framework_id: Only include items belonging to this framework.

          include_unframed: Whether to include rules that are not part of any framework.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          rule_id: Only include results of this rule.

          scope: Only include rules with this scope.

          search_query: Filter by a free-text search over names and descriptions.

          status: Only include items whose rule result has this compliance status.

          type: Only include rules of this type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return self._get(
            path_template("/workspaces/{workspace_id}/rule-results", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "enabled_framework_only": enabled_framework_only,
                        "framework_id": framework_id,
                        "include_unframed": include_unframed,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "rule_id": rule_id,
                        "scope": scope,
                        "search_query": search_query,
                        "status": status,
                        "type": type,
                    },
                    rule_result_list_params.RuleResultListParams,
                ),
            ),
            cast_to=RuleResultListResponse,
        )

    def create_evidence(
        self,
        rule_result_id: str,
        *,
        description: Optional[str] | Omit = omit,
        name: Optional[str] | Omit = omit,
        storage_uri: Optional[str] | Omit = omit,
        text: Optional[str] | Omit = omit,
        url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultCreateEvidenceResponse:
        """
        Attach evidence to a rule result, satisfying an evidence rule.

        Send the field that matches the rule's `evidenceType`: `storageUri` for an
        uploaded document, `text` for a written statement, or `url` for a link.

        For a document, upload the file first with `POST /storage/presigned-url` and
        send the resulting storage URI as `storageUri`.

        Attaching evidence re-evaluates the rule result. If the rule sets
        `renewalCadenceDays`, the renewal window restarts from this evidence.

        Args:
          description: A description of what the evidence shows.

          name: The evidence name.

          storage_uri: Where the uploaded file is stored. Set when the rule's `evidenceType` is
              `document`.

          text: The evidence text. Set when the rule's `evidenceType` is `text`.

          url: A link to the evidence. Set when the rule's `evidenceType` is `url`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return self._post(
            path_template("/rule-results/{rule_result_id}/evidence", rule_result_id=rule_result_id),
            body=maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "storage_uri": storage_uri,
                    "text": text,
                    "url": url,
                },
                rule_result_create_evidence_params.RuleResultCreateEvidenceParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultCreateEvidenceResponse,
        )

    def list_evidence(
        self,
        rule_result_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultListEvidenceResponse:
        """
        List the evidence attached to a rule result.

        Which field carries the evidence depends on the rule's `evidenceType`:
        `storageUri` for uploaded documents, `text` for written statements, and `url`
        for links.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return self._get(
            path_template("/rule-results/{rule_result_id}/evidence", rule_result_id=rule_result_id),
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
                    rule_result_list_evidence_params.RuleResultListEvidenceParams,
                ),
            ),
            cast_to=RuleResultListEvidenceResponse,
        )


class AsyncRuleResultsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncRuleResultsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncRuleResultsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncRuleResultsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncRuleResultsResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        rule_result_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultRetrieveResponse:
        """
        Retrieve a rule result by its id.

        Alongside the status, the response carries the evaluation and renewal dates that
        explain it: `dateLastEvaluated` and `dateOfNextEvaluation` for platform rules,
        `dateOfLatestEvidence` and `dateOfRenewal` for evidence rules.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return await self._get(
            path_template("/rule-results/{rule_result_id}", rule_result_id=rule_result_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultRetrieveResponse,
        )

    async def update(
        self,
        rule_result_id: str,
        *,
        assignee_id: Optional[str] | Omit = omit,
        blocked_by: Iterable[rule_result_update_params.BlockedBy] | Omit = omit,
        blocking: Iterable[rule_result_update_params.Blocking] | Omit = omit,
        deactivated: bool | Omit = omit,
        deactivated_reason: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultUpdateResponse:
        """Update a rule result.

        Only the fields you send are changed.

        Use this to assign an owner, or to exclude a single result from compliance
        without deactivating the rule everywhere. `deactivatedReason` is required when
        setting `deactivated` to `true`.

        A result's `status` is computed by Openlayer and cannot be set directly.

        Args:
          assignee_id: The user responsible for this result.

          blocked_by: Rule results that must pass before this one can be satisfied.

          blocking: Rule results that this one blocks.

          deactivated: Whether this result is excluded from compliance calculations.

          deactivated_reason: Why the result was excluded.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return await self._patch(
            path_template("/rule-results/{rule_result_id}", rule_result_id=rule_result_id),
            body=await async_maybe_transform(
                {
                    "assignee_id": assignee_id,
                    "blocked_by": blocked_by,
                    "blocking": blocking,
                    "deactivated": deactivated,
                    "deactivated_reason": deactivated_reason,
                },
                rule_result_update_params.RuleResultUpdateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultUpdateResponse,
        )

    async def list(
        self,
        workspace_id: str,
        *,
        enabled_framework_only: bool | Omit = omit,
        framework_id: str | Omit = omit,
        include_unframed: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        rule_id: str | Omit = omit,
        scope: Literal["project", "workspace"] | Omit = omit,
        search_query: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        type: Literal["platform", "evidence"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultListResponse:
        """
        List rule results across a workspace.

        A rule result is the compliance status of one rule for one entity: a project for
        project-scoped rules, or the workspace itself for workspace-scoped rules. This
        is the endpoint to poll or export when you want your current compliance state,
        filtered to a framework, a project, or a status.

        Args:
          enabled_framework_only: Only include items belonging to at least one enabled framework.

          framework_id: Only include items belonging to this framework.

          include_unframed: Whether to include rules that are not part of any framework.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          rule_id: Only include results of this rule.

          scope: Only include rules with this scope.

          search_query: Filter by a free-text search over names and descriptions.

          status: Only include items whose rule result has this compliance status.

          type: Only include rules of this type.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not workspace_id:
            raise ValueError(f"Expected a non-empty value for `workspace_id` but received {workspace_id!r}")
        return await self._get(
            path_template("/workspaces/{workspace_id}/rule-results", workspace_id=workspace_id),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "enabled_framework_only": enabled_framework_only,
                        "framework_id": framework_id,
                        "include_unframed": include_unframed,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "rule_id": rule_id,
                        "scope": scope,
                        "search_query": search_query,
                        "status": status,
                        "type": type,
                    },
                    rule_result_list_params.RuleResultListParams,
                ),
            ),
            cast_to=RuleResultListResponse,
        )

    async def create_evidence(
        self,
        rule_result_id: str,
        *,
        description: Optional[str] | Omit = omit,
        name: Optional[str] | Omit = omit,
        storage_uri: Optional[str] | Omit = omit,
        text: Optional[str] | Omit = omit,
        url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultCreateEvidenceResponse:
        """
        Attach evidence to a rule result, satisfying an evidence rule.

        Send the field that matches the rule's `evidenceType`: `storageUri` for an
        uploaded document, `text` for a written statement, or `url` for a link.

        For a document, upload the file first with `POST /storage/presigned-url` and
        send the resulting storage URI as `storageUri`.

        Attaching evidence re-evaluates the rule result. If the rule sets
        `renewalCadenceDays`, the renewal window restarts from this evidence.

        Args:
          description: A description of what the evidence shows.

          name: The evidence name.

          storage_uri: Where the uploaded file is stored. Set when the rule's `evidenceType` is
              `document`.

          text: The evidence text. Set when the rule's `evidenceType` is `text`.

          url: A link to the evidence. Set when the rule's `evidenceType` is `url`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return await self._post(
            path_template("/rule-results/{rule_result_id}/evidence", rule_result_id=rule_result_id),
            body=await async_maybe_transform(
                {
                    "description": description,
                    "name": name,
                    "storage_uri": storage_uri,
                    "text": text,
                    "url": url,
                },
                rule_result_create_evidence_params.RuleResultCreateEvidenceParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=RuleResultCreateEvidenceResponse,
        )

    async def list_evidence(
        self,
        rule_result_id: str,
        *,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> RuleResultListEvidenceResponse:
        """
        List the evidence attached to a rule result.

        Which field carries the evidence depends on the rule's `evidenceType`:
        `storageUri` for uploaded documents, `text` for written statements, and `url`
        for links.

        Args:
          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not rule_result_id:
            raise ValueError(f"Expected a non-empty value for `rule_result_id` but received {rule_result_id!r}")
        return await self._get(
            path_template("/rule-results/{rule_result_id}/evidence", rule_result_id=rule_result_id),
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
                    rule_result_list_evidence_params.RuleResultListEvidenceParams,
                ),
            ),
            cast_to=RuleResultListEvidenceResponse,
        )


class RuleResultsResourceWithRawResponse:
    def __init__(self, rule_results: RuleResultsResource) -> None:
        self._rule_results = rule_results

        self.retrieve = to_raw_response_wrapper(
            rule_results.retrieve,
        )
        self.update = to_raw_response_wrapper(
            rule_results.update,
        )
        self.list = to_raw_response_wrapper(
            rule_results.list,
        )
        self.create_evidence = to_raw_response_wrapper(
            rule_results.create_evidence,
        )
        self.list_evidence = to_raw_response_wrapper(
            rule_results.list_evidence,
        )


class AsyncRuleResultsResourceWithRawResponse:
    def __init__(self, rule_results: AsyncRuleResultsResource) -> None:
        self._rule_results = rule_results

        self.retrieve = async_to_raw_response_wrapper(
            rule_results.retrieve,
        )
        self.update = async_to_raw_response_wrapper(
            rule_results.update,
        )
        self.list = async_to_raw_response_wrapper(
            rule_results.list,
        )
        self.create_evidence = async_to_raw_response_wrapper(
            rule_results.create_evidence,
        )
        self.list_evidence = async_to_raw_response_wrapper(
            rule_results.list_evidence,
        )


class RuleResultsResourceWithStreamingResponse:
    def __init__(self, rule_results: RuleResultsResource) -> None:
        self._rule_results = rule_results

        self.retrieve = to_streamed_response_wrapper(
            rule_results.retrieve,
        )
        self.update = to_streamed_response_wrapper(
            rule_results.update,
        )
        self.list = to_streamed_response_wrapper(
            rule_results.list,
        )
        self.create_evidence = to_streamed_response_wrapper(
            rule_results.create_evidence,
        )
        self.list_evidence = to_streamed_response_wrapper(
            rule_results.list_evidence,
        )


class AsyncRuleResultsResourceWithStreamingResponse:
    def __init__(self, rule_results: AsyncRuleResultsResource) -> None:
        self._rule_results = rule_results

        self.retrieve = async_to_streamed_response_wrapper(
            rule_results.retrieve,
        )
        self.update = async_to_streamed_response_wrapper(
            rule_results.update,
        )
        self.list = async_to_streamed_response_wrapper(
            rule_results.list,
        )
        self.create_evidence = async_to_streamed_response_wrapper(
            rule_results.create_evidence,
        )
        self.list_evidence = async_to_streamed_response_wrapper(
            rule_results.list_evidence,
        )
