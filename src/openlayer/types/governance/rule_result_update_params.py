# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional
from typing_extensions import Literal, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["RuleResultUpdateParams", "BlockedBy", "Blocking"]


class RuleResultUpdateParams(TypedDict, total=False):
    assignee_id: Annotated[Optional[str], PropertyInfo(alias="assigneeId")]
    """The user responsible for this result."""

    blocked_by: Annotated[Iterable[BlockedBy], PropertyInfo(alias="blockedBy")]
    """Rule results that must pass before this one can be satisfied."""

    blocking: Iterable[Blocking]
    """Rule results that this one blocks."""

    deactivated: bool
    """Whether this result is excluded from compliance calculations."""

    deactivated_reason: Annotated[Optional[str], PropertyInfo(alias="deactivatedReason")]
    """Why the result was excluded."""


class BlockedBy(TypedDict, total=False):
    id: str

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """The compliance status of the rule for this entity."""


class Blocking(TypedDict, total=False):
    id: str

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """The compliance status of the rule for this entity."""
