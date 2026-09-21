# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["RuleResultListResponse", "Item", "ItemBlockedBy", "ItemBlocking"]


class ItemBlockedBy(BaseModel):
    id: Optional[str] = None

    status: Optional[Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]] = None
    """The compliance status of the rule for this entity."""


class ItemBlocking(BaseModel):
    id: Optional[str] = None

    status: Optional[Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]] = None
    """The compliance status of the rule for this entity."""


class Item(BaseModel):
    id: str
    """The rule result id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    deactivated: bool
    """Whether this result is excluded from compliance calculations."""

    rule_id: str = FieldInfo(alias="ruleId")
    """The rule this result belongs to."""

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """The compliance status of the rule for this entity."""

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the rule result belongs to."""

    assignee_id: Optional[str] = FieldInfo(alias="assigneeId", default=None)
    """The user responsible for this result."""

    blocked_by: Optional[List[ItemBlockedBy]] = FieldInfo(alias="blockedBy", default=None)
    """Rule results that must pass before this one can be satisfied."""

    blocking: Optional[List[ItemBlocking]] = None
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
    """Why the result was excluded."""

    project_id: Optional[str] = FieldInfo(alias="projectId", default=None)
    """The project this result was evaluated for. `null` for workspace-scoped rules."""

    status_message: Optional[str] = FieldInfo(alias="statusMessage", default=None)
    """A human-readable explanation of the status."""


class RuleResultListResponse(BaseModel):
    items: List[Item]
