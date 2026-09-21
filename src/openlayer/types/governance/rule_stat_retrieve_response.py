# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = [
    "RuleStatRetrieveResponse",
    "RuleResults",
    "RuleResultsByRuleScope",
    "RuleResultsByRuleScopeProject",
    "RuleResultsByRuleScopeWorkspace",
    "RuleResultsByRuleType",
    "RuleResultsByRuleTypeEvidence",
    "RuleResultsByRuleTypePlatform",
    "Rules",
    "RulesByScope",
    "RulesByType",
]


class RuleResultsByRuleScopeProject(BaseModel):
    total: int
    """The total number of rule results."""

    total_due_soon: int = FieldInfo(alias="totalDueSoon")
    """The number of rule results whose evidence is about to expire."""

    total_error: int = FieldInfo(alias="totalError")
    """The number of rule results that errored during evaluation."""

    total_failing: int = FieldInfo(alias="totalFailing")
    """The number of failing rule results."""

    total_passing: int = FieldInfo(alias="totalPassing")
    """The number of passing rule results."""

    total_pending: int = FieldInfo(alias="totalPending")
    """The number of rule results that have not been satisfied yet."""

    total_running: int = FieldInfo(alias="totalRunning")
    """The number of rule results currently being evaluated."""

    total_skipped: int = FieldInfo(alias="totalSkipped")
    """The number of skipped rule results."""


class RuleResultsByRuleScopeWorkspace(BaseModel):
    total: int
    """The total number of rule results."""

    total_due_soon: int = FieldInfo(alias="totalDueSoon")
    """The number of rule results whose evidence is about to expire."""

    total_error: int = FieldInfo(alias="totalError")
    """The number of rule results that errored during evaluation."""

    total_failing: int = FieldInfo(alias="totalFailing")
    """The number of failing rule results."""

    total_passing: int = FieldInfo(alias="totalPassing")
    """The number of passing rule results."""

    total_pending: int = FieldInfo(alias="totalPending")
    """The number of rule results that have not been satisfied yet."""

    total_running: int = FieldInfo(alias="totalRunning")
    """The number of rule results currently being evaluated."""

    total_skipped: int = FieldInfo(alias="totalSkipped")
    """The number of skipped rule results."""


class RuleResultsByRuleScope(BaseModel):
    project: Optional[RuleResultsByRuleScopeProject] = None

    workspace: Optional[RuleResultsByRuleScopeWorkspace] = None


class RuleResultsByRuleTypeEvidence(BaseModel):
    total: int
    """The total number of rule results."""

    total_due_soon: int = FieldInfo(alias="totalDueSoon")
    """The number of rule results whose evidence is about to expire."""

    total_error: int = FieldInfo(alias="totalError")
    """The number of rule results that errored during evaluation."""

    total_failing: int = FieldInfo(alias="totalFailing")
    """The number of failing rule results."""

    total_passing: int = FieldInfo(alias="totalPassing")
    """The number of passing rule results."""

    total_pending: int = FieldInfo(alias="totalPending")
    """The number of rule results that have not been satisfied yet."""

    total_running: int = FieldInfo(alias="totalRunning")
    """The number of rule results currently being evaluated."""

    total_skipped: int = FieldInfo(alias="totalSkipped")
    """The number of skipped rule results."""


class RuleResultsByRuleTypePlatform(BaseModel):
    total: int
    """The total number of rule results."""

    total_due_soon: int = FieldInfo(alias="totalDueSoon")
    """The number of rule results whose evidence is about to expire."""

    total_error: int = FieldInfo(alias="totalError")
    """The number of rule results that errored during evaluation."""

    total_failing: int = FieldInfo(alias="totalFailing")
    """The number of failing rule results."""

    total_passing: int = FieldInfo(alias="totalPassing")
    """The number of passing rule results."""

    total_pending: int = FieldInfo(alias="totalPending")
    """The number of rule results that have not been satisfied yet."""

    total_running: int = FieldInfo(alias="totalRunning")
    """The number of rule results currently being evaluated."""

    total_skipped: int = FieldInfo(alias="totalSkipped")
    """The number of skipped rule results."""


class RuleResultsByRuleType(BaseModel):
    evidence: Optional[RuleResultsByRuleTypeEvidence] = None

    platform: Optional[RuleResultsByRuleTypePlatform] = None


class RuleResults(BaseModel):
    """
    Counts of rule results, after any filters in the request, with breakdowns by the type and scope of the rule each result belongs to.
    """

    total: int
    """The total number of rule results."""

    total_due_soon: int = FieldInfo(alias="totalDueSoon")
    """The number of rule results whose evidence is about to expire."""

    total_error: int = FieldInfo(alias="totalError")
    """The number of rule results that errored during evaluation."""

    total_failing: int = FieldInfo(alias="totalFailing")
    """The number of failing rule results."""

    total_passing: int = FieldInfo(alias="totalPassing")
    """The number of passing rule results."""

    total_pending: int = FieldInfo(alias="totalPending")
    """The number of rule results that have not been satisfied yet."""

    total_running: int = FieldInfo(alias="totalRunning")
    """The number of rule results currently being evaluated."""

    total_skipped: int = FieldInfo(alias="totalSkipped")
    """The number of skipped rule results."""

    by_rule_scope: Optional[RuleResultsByRuleScope] = FieldInfo(alias="byRuleScope", default=None)

    by_rule_type: Optional[RuleResultsByRuleType] = FieldInfo(alias="byRuleType", default=None)


class RulesByScope(BaseModel):
    """Rule counts by scope."""

    project: Optional[int] = None

    workspace: Optional[int] = None


class RulesByType(BaseModel):
    """Rule counts by type."""

    evidence: Optional[int] = None

    platform: Optional[int] = None


class Rules(BaseModel):
    """Counts of the rules themselves, after any filters in the request."""

    by_scope: Optional[RulesByScope] = FieldInfo(alias="byScope", default=None)
    """Rule counts by scope."""

    by_type: Optional[RulesByType] = FieldInfo(alias="byType", default=None)
    """Rule counts by type."""

    total: Optional[int] = None
    """The total number of rules."""


class RuleStatRetrieveResponse(BaseModel):
    rule_results: RuleResults = FieldInfo(alias="ruleResults")
    """
    Counts of rule results, after any filters in the request, with breakdowns by the
    type and scope of the rule each result belongs to.
    """

    rules: Rules
    """Counts of the rules themselves, after any filters in the request."""
