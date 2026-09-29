# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = [
    "FrameworkListRulesResponse",
    "Item",
    "ItemFramework",
    "ItemFrameworkAvatar",
    "ItemResult",
    "ItemResultBlockedBy",
    "ItemResultBlocking",
    "ItemResultsSummary",
    "ItemTag",
]


class ItemFrameworkAvatar(BaseModel):
    """The icon shown for the framework."""

    type: Literal["emoji", "imageUrl", "builtinImage"]

    value: str


class ItemFramework(BaseModel):
    id: str
    """The framework id."""

    avatar: Optional[ItemFrameworkAvatar] = None
    """The icon shown for the framework."""

    built_in_slug: Optional[str] = FieldInfo(alias="builtInSlug", default=None)
    """
    Identifies a framework that ships with Openlayer, for example `eu_ai_act`,
    `iso_42001`, `nist_ai_rmf`, or `traiga`. `null` for frameworks you create
    yourself.
    """

    enabled: bool
    """Whether the framework is active.

    Rules of a disabled framework are not evaluated and do not count towards
    compliance.
    """

    name: str
    """The framework name."""


class ItemResultBlockedBy(BaseModel):
    id: Optional[str] = None

    status: Optional[Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]] = None
    """The compliance status of the rule for this entity.

    Computed by Openlayer and can't be set directly.
    """


class ItemResultBlocking(BaseModel):
    id: Optional[str] = None

    status: Optional[Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]] = None
    """The compliance status of the rule for this entity.

    Computed by Openlayer and can't be set directly.
    """


class ItemResult(BaseModel):
    """
    The compliance status of one rule for one entity: a project for project-scoped rules, or the workspace for workspace-scoped rules.
    """

    id: str
    """The rule result id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    deactivated: bool
    """Whether this result is excluded from compliance calculations.

    Excludes just this result, without deactivating the rule everywhere.
    """

    rule_id: str = FieldInfo(alias="ruleId")
    """The rule this result belongs to."""

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """The compliance status of the rule for this entity.

    Computed by Openlayer and can't be set directly.
    """

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the rule result belongs to."""

    assignee_id: Optional[str] = FieldInfo(alias="assigneeId", default=None)
    """The user responsible for this result."""

    blocked_by: Optional[List[ItemResultBlockedBy]] = FieldInfo(alias="blockedBy", default=None)
    """Rule results that must pass before this one can be satisfied."""

    blocking: Optional[List[ItemResultBlocking]] = None
    """Rule results that this one blocks."""

    date_last_evaluated: Optional[datetime] = FieldInfo(alias="dateLastEvaluated", default=None)
    """When the rule was last evaluated. Platform rules only."""

    date_of_latest_evidence: Optional[datetime] = FieldInfo(alias="dateOfLatestEvidence", default=None)
    """When the most recent piece of evidence was attached. Evidence rules only."""

    date_of_next_evaluation: Optional[datetime] = FieldInfo(alias="dateOfNextEvaluation", default=None)
    """When the rule will next be evaluated. Platform rules only."""

    date_of_renewal: Optional[datetime] = FieldInfo(alias="dateOfRenewal", default=None)
    """When the evidence must be renewed. Evidence rules with a renewal cadence only."""

    deactivated_reason: Optional[str] = FieldInfo(alias="deactivatedReason", default=None)
    """Why the result was excluded. Required when setting `deactivated` to `true`."""

    project_id: Optional[str] = FieldInfo(alias="projectId", default=None)
    """The project this result was evaluated for. `null` for workspace-scoped rules."""

    status_message: Optional[str] = FieldInfo(alias="statusMessage", default=None)
    """A human-readable explanation of the status."""


class ItemResultsSummary(BaseModel):
    """
    Pass-rate counts across all of the rule's entities, independent of any status filter applied to the request.
    """

    passing: Optional[int] = None

    total: Optional[int] = None


class ItemTag(BaseModel):
    """
    A label that groups rules across frameworks, for example by team or control family.
    """

    id: str
    """The rule tag id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The user who created the tag. `null` for tags that ship with Openlayer."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    immutable: bool
    """Whether the tag is managed by Openlayer. These tags can't be deleted."""

    name: str
    """The tag name."""

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the tag belongs to."""

    color: Optional[str] = None
    """The color the tag is displayed with."""


class Item(BaseModel):
    """A single requirement Openlayer tracks.

    `platform` rules are evaluated automatically from the state of your workspace, and `evidence` rules are satisfied by attaching evidence. A rule can belong to several frameworks, or to none.
    """

    id: str
    """The rule id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    name: str
    """The rule name."""

    scope: Literal["project", "workspace"]
    """
    Whether the rule is evaluated once for the whole workspace, or once per project
    the rule's frameworks apply to. Must be `project` for platform rules. Fixed once
    the rule is created.
    """

    type: Literal["platform", "evidence"]
    """
    `platform` rules are evaluated automatically from the state of your Openlayer
    workspace. `evidence` rules are satisfied by attaching evidence. Fixed once the
    rule is created.
    """

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the rule belongs to."""

    assignee_id: Optional[str] = FieldInfo(alias="assigneeId", default=None)
    """The user responsible for satisfying the rule."""

    automation_params: Optional[Dict[str, object]] = FieldInfo(alias="automationParams", default=None)
    """Configuration for the platform check, when the automation takes parameters.

    Omit or `null` for evidence rules. Fixed once the rule is created.
    """

    automation_type: Optional[str] = FieldInfo(alias="automationType", default=None)
    """
    Which workspace signal a platform rule checks, for example
    `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. Set it for
    platform rules; omit or `null` for evidence rules. Fixed once the rule is
    created.
    """

    deactivated: Optional[bool] = None
    """Whether the rule is excluded from compliance calculations."""

    description: Optional[str] = None
    """What the rule requires."""

    evidence_type: Optional[Literal["document", "text", "url", "categoryValue"]] = FieldInfo(
        alias="evidenceType", default=None
    )
    """The kind of evidence that satisfies the rule.

    Set it for evidence rules; omit or `null` for platform rules. Fixed once the
    rule is created.
    """

    frameworks: Optional[List[ItemFramework]] = None
    """The frameworks that include this rule."""

    immutable: Optional[bool] = None
    """Whether the rule is managed by Openlayer.

    These rules can't be renamed or deleted; set `deactivated` to exclude one from
    compliance instead.
    """

    renewal_cadence_days: Optional[int] = FieldInfo(alias="renewalCadenceDays", default=None)
    """How often evidence must be renewed, in days.

    Once evidence is older than this, the rule result becomes `due_soon` and then
    `failing`. The window restarts whenever evidence is attached. Omit or `null` for
    platform rules.
    """

    results: Optional[List[ItemResult]] = None
    """The rule's results, one per entity the rule is evaluated against.

    Only returned when `includeResults` is `true`.
    """

    results_summary: Optional[ItemResultsSummary] = FieldInfo(alias="resultsSummary", default=None)
    """
    Pass-rate counts across all of the rule's entities, independent of any status
    filter applied to the request.
    """

    tags: Optional[List[ItemTag]] = None
    """The rule tags associated with the rule."""


class FrameworkListRulesResponse(BaseModel):
    items: List[Item]
