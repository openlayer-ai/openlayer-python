# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from .rules import (
    RulesResource,
    AsyncRulesResource,
    RulesResourceWithRawResponse,
    AsyncRulesResourceWithRawResponse,
    RulesResourceWithStreamingResponse,
    AsyncRulesResourceWithStreamingResponse,
)
from ..._compat import cached_property
from .rule_tags import (
    RuleTagsResource,
    AsyncRuleTagsResource,
    RuleTagsResourceWithRawResponse,
    AsyncRuleTagsResourceWithRawResponse,
    RuleTagsResourceWithStreamingResponse,
    AsyncRuleTagsResourceWithStreamingResponse,
)
from .rule_stats import (
    RuleStatsResource,
    AsyncRuleStatsResource,
    RuleStatsResourceWithRawResponse,
    AsyncRuleStatsResourceWithRawResponse,
    RuleStatsResourceWithStreamingResponse,
    AsyncRuleStatsResourceWithStreamingResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from .rule_results import (
    RuleResultsResource,
    AsyncRuleResultsResource,
    RuleResultsResourceWithRawResponse,
    AsyncRuleResultsResourceWithRawResponse,
    RuleResultsResourceWithStreamingResponse,
    AsyncRuleResultsResourceWithStreamingResponse,
)
from .frameworks.frameworks import (
    FrameworksResource,
    AsyncFrameworksResource,
    FrameworksResourceWithRawResponse,
    AsyncFrameworksResourceWithRawResponse,
    FrameworksResourceWithStreamingResponse,
    AsyncFrameworksResourceWithStreamingResponse,
)

__all__ = ["GovernanceResource", "AsyncGovernanceResource"]


class GovernanceResource(SyncAPIResource):
    @cached_property
    def frameworks(self) -> FrameworksResource:
        return FrameworksResource(self._client)

    @cached_property
    def rules(self) -> RulesResource:
        return RulesResource(self._client)

    @cached_property
    def rule_results(self) -> RuleResultsResource:
        return RuleResultsResource(self._client)

    @cached_property
    def rule_stats(self) -> RuleStatsResource:
        return RuleStatsResource(self._client)

    @cached_property
    def rule_tags(self) -> RuleTagsResource:
        return RuleTagsResource(self._client)

    @cached_property
    def with_raw_response(self) -> GovernanceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return GovernanceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> GovernanceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return GovernanceResourceWithStreamingResponse(self)


class AsyncGovernanceResource(AsyncAPIResource):
    @cached_property
    def frameworks(self) -> AsyncFrameworksResource:
        return AsyncFrameworksResource(self._client)

    @cached_property
    def rules(self) -> AsyncRulesResource:
        return AsyncRulesResource(self._client)

    @cached_property
    def rule_results(self) -> AsyncRuleResultsResource:
        return AsyncRuleResultsResource(self._client)

    @cached_property
    def rule_stats(self) -> AsyncRuleStatsResource:
        return AsyncRuleStatsResource(self._client)

    @cached_property
    def rule_tags(self) -> AsyncRuleTagsResource:
        return AsyncRuleTagsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncGovernanceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncGovernanceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncGovernanceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncGovernanceResourceWithStreamingResponse(self)


class GovernanceResourceWithRawResponse:
    def __init__(self, governance: GovernanceResource) -> None:
        self._governance = governance

    @cached_property
    def frameworks(self) -> FrameworksResourceWithRawResponse:
        return FrameworksResourceWithRawResponse(self._governance.frameworks)

    @cached_property
    def rules(self) -> RulesResourceWithRawResponse:
        return RulesResourceWithRawResponse(self._governance.rules)

    @cached_property
    def rule_results(self) -> RuleResultsResourceWithRawResponse:
        return RuleResultsResourceWithRawResponse(self._governance.rule_results)

    @cached_property
    def rule_stats(self) -> RuleStatsResourceWithRawResponse:
        return RuleStatsResourceWithRawResponse(self._governance.rule_stats)

    @cached_property
    def rule_tags(self) -> RuleTagsResourceWithRawResponse:
        return RuleTagsResourceWithRawResponse(self._governance.rule_tags)


class AsyncGovernanceResourceWithRawResponse:
    def __init__(self, governance: AsyncGovernanceResource) -> None:
        self._governance = governance

    @cached_property
    def frameworks(self) -> AsyncFrameworksResourceWithRawResponse:
        return AsyncFrameworksResourceWithRawResponse(self._governance.frameworks)

    @cached_property
    def rules(self) -> AsyncRulesResourceWithRawResponse:
        return AsyncRulesResourceWithRawResponse(self._governance.rules)

    @cached_property
    def rule_results(self) -> AsyncRuleResultsResourceWithRawResponse:
        return AsyncRuleResultsResourceWithRawResponse(self._governance.rule_results)

    @cached_property
    def rule_stats(self) -> AsyncRuleStatsResourceWithRawResponse:
        return AsyncRuleStatsResourceWithRawResponse(self._governance.rule_stats)

    @cached_property
    def rule_tags(self) -> AsyncRuleTagsResourceWithRawResponse:
        return AsyncRuleTagsResourceWithRawResponse(self._governance.rule_tags)


class GovernanceResourceWithStreamingResponse:
    def __init__(self, governance: GovernanceResource) -> None:
        self._governance = governance

    @cached_property
    def frameworks(self) -> FrameworksResourceWithStreamingResponse:
        return FrameworksResourceWithStreamingResponse(self._governance.frameworks)

    @cached_property
    def rules(self) -> RulesResourceWithStreamingResponse:
        return RulesResourceWithStreamingResponse(self._governance.rules)

    @cached_property
    def rule_results(self) -> RuleResultsResourceWithStreamingResponse:
        return RuleResultsResourceWithStreamingResponse(self._governance.rule_results)

    @cached_property
    def rule_stats(self) -> RuleStatsResourceWithStreamingResponse:
        return RuleStatsResourceWithStreamingResponse(self._governance.rule_stats)

    @cached_property
    def rule_tags(self) -> RuleTagsResourceWithStreamingResponse:
        return RuleTagsResourceWithStreamingResponse(self._governance.rule_tags)


class AsyncGovernanceResourceWithStreamingResponse:
    def __init__(self, governance: AsyncGovernanceResource) -> None:
        self._governance = governance

    @cached_property
    def frameworks(self) -> AsyncFrameworksResourceWithStreamingResponse:
        return AsyncFrameworksResourceWithStreamingResponse(self._governance.frameworks)

    @cached_property
    def rules(self) -> AsyncRulesResourceWithStreamingResponse:
        return AsyncRulesResourceWithStreamingResponse(self._governance.rules)

    @cached_property
    def rule_results(self) -> AsyncRuleResultsResourceWithStreamingResponse:
        return AsyncRuleResultsResourceWithStreamingResponse(self._governance.rule_results)

    @cached_property
    def rule_stats(self) -> AsyncRuleStatsResourceWithStreamingResponse:
        return AsyncRuleStatsResourceWithStreamingResponse(self._governance.rule_stats)

    @cached_property
    def rule_tags(self) -> AsyncRuleTagsResourceWithStreamingResponse:
        return AsyncRuleTagsResourceWithStreamingResponse(self._governance.rule_tags)
