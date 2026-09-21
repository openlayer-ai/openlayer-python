# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = [
    "FrameworkRetrieveResponse",
    "Avatar",
    "ProjectSelector",
    "ProjectSelectorMatch",
    "RuleStats",
    "RuleStatsProjectCompletion",
    "RuleStatsRuleResults",
]


class Avatar(BaseModel):
    """The icon shown for the framework."""

    type: Literal["emoji", "imageUrl", "builtinImage"]

    value: str


class ProjectSelectorMatch(BaseModel):
    property: Literal["taskType", "riskLevel", "riskTotalScore", "name", "ownerId", "modelTypes"]
    """The project property to match against."""

    value: object
    """The value to match against.

    Pass an array to match any of several values, or `null` to match projects where
    the property is unset. Omit it for `exists` and `notExists`.
    """

    operator: Optional[str] = None
    """How to compare the project property with `value`.

    One of `equals`, `notEquals`, `contains`, `notContains`, `startsWith`,
    `endsWith`, `in`, `notIn`, `greaterThan`, `greaterThanOrEqual`, `lessThan`,
    `lessThanOrEqual`, `equalsIgnoreCase`, `containsIgnoreCase`, `matches`,
    `exists`, or `notExists`.
    """


class ProjectSelector(BaseModel):
    """Determines which projects the framework applies to.

    An empty or `null` `match` array applies the framework to every project in the workspace.
    """

    match: Optional[List[ProjectSelectorMatch]] = None
    """Match criteria, ANDed together."""


class RuleStatsProjectCompletion(BaseModel):
    """
    How many of the framework's projects fall into each completion band, where a project's completion is the share of its rule results that are passing or skipped. Projects with no evaluated results count as `low`.

    Zeroed when the request carries `projectId`: the bands compare a framework's projects against each other, which says nothing about a single project.
    """

    high: int
    """Projects at 80% completion or above."""

    low: int
    """Projects below 20% completion."""

    mid: int
    """Projects at or above 20% but below 80% completion."""


class RuleStatsRuleResults(BaseModel):
    """
    Rule result counts by status for this framework, matching what `/workspaces/{workspaceId}/rule-stats?frameworkId=<id>` reports. Narrowed to a single project when the request also carries `projectId`.
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


class RuleStats(BaseModel):
    """Compliance roll-up for the framework.

    Present only on `GET /workspaces/{workspaceId}/frameworks` when the request sets `includeRuleStats=true`.
    """

    project_completion: RuleStatsProjectCompletion = FieldInfo(alias="projectCompletion")
    """
    How many of the framework's projects fall into each completion band, where a
    project's completion is the share of its rule results that are passing or
    skipped. Projects with no evaluated results count as `low`.

    Zeroed when the request carries `projectId`: the bands compare a framework's
    projects against each other, which says nothing about a single project.
    """

    rule_results: RuleStatsRuleResults = FieldInfo(alias="ruleResults")
    """
    Rule result counts by status for this framework, matching what
    `/workspaces/{workspaceId}/rule-stats?frameworkId=<id>` reports. Narrowed to a
    single project when the request also carries `projectId`.
    """


class FrameworkRetrieveResponse(BaseModel):
    id: str
    """The framework id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    enabled: bool
    """Whether the framework is active.

    Rules of a disabled framework are not evaluated and do not count towards
    compliance.
    """

    name: str
    """The framework name."""

    tags: List[str]
    """Free-form labels on the framework."""

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the framework belongs to."""

    avatar: Optional[Avatar] = None
    """The icon shown for the framework."""

    built_in_slug: Optional[str] = FieldInfo(alias="builtInSlug", default=None)
    """
    Identifies a framework that ships with Openlayer, for example `eu_ai_act`,
    `iso_42001`, `nist_ai_rmf`, or `traiga`. `null` for frameworks you create
    yourself.
    """

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The user who created the framework. `null` for built-in frameworks."""

    description: Optional[str] = None
    """A short description of the framework."""

    extended_description: Optional[Dict[str, object]] = FieldInfo(alias="extendedDescription", default=None)
    """A longer, rich-text description, as a TipTap JSON document."""

    href: Optional[str] = None
    """A link to the external standard or regulation the framework is based on."""

    immutable: Optional[bool] = None
    """Whether the framework definition is managed by Openlayer and cannot be edited."""

    project_selector: Optional[ProjectSelector] = FieldInfo(alias="projectSelector", default=None)
    """Determines which projects the framework applies to.

    An empty or `null` `match` array applies the framework to every project in the
    workspace.
    """

    rule_stats: Optional[RuleStats] = FieldInfo(alias="ruleStats", default=None)
    """Compliance roll-up for the framework.

    Present only on `GET /workspaces/{workspaceId}/frameworks` when the request sets
    `includeRuleStats=true`.
    """
