# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = [
    "FrameworkListProjectRuleStatsResponse",
    "Item",
    "ItemByRuleType",
    "ItemByRuleTypeEvidence",
    "ItemByRuleTypePlatform",
]


class ItemByRuleTypeEvidence(BaseModel):
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


class ItemByRuleTypePlatform(BaseModel):
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


class ItemByRuleType(BaseModel):
    """The same counts, broken down by the type of the rule each result belongs to."""

    evidence: Optional[ItemByRuleTypeEvidence] = None

    platform: Optional[ItemByRuleTypePlatform] = None


class Item(BaseModel):
    project_id: str = FieldInfo(alias="projectId")
    """The project id."""

    project_name: str = FieldInfo(alias="projectName")
    """The project name."""

    task_type: str = FieldInfo(alias="taskType")
    """The project's task type."""

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

    by_rule_type: Optional[ItemByRuleType] = FieldInfo(alias="byRuleType", default=None)
    """The same counts, broken down by the type of the rule each result belongs to."""


class FrameworkListProjectRuleStatsResponse(BaseModel):
    items: List[Item]
