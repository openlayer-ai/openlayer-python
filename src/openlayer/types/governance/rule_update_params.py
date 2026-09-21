# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["RuleUpdateParams"]


class RuleUpdateParams(TypedDict, total=False):
    assignee_id: Annotated[Optional[str], PropertyInfo(alias="assigneeId")]
    """The user responsible for satisfying the rule."""

    deactivated: bool
    """Whether the rule is excluded from compliance calculations."""

    description: Optional[str]
    """What the rule requires."""

    name: str
    """The rule name."""

    renewal_cadence_days: Annotated[Optional[int], PropertyInfo(alias="renewalCadenceDays")]
    """How often evidence must be renewed, in days.

    Once evidence is older than this, the rule result becomes `due_soon` and then
    `failing`.
    """

    tag_ids: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="tagIds")]
    """The ids of the rule tags to associate with the rule.

    Replaces the rule's tags. Read them back from `tags`, and list the tags
    available in the workspace with `GET /workspaces/{workspaceId}/rule-tags`.
    """
