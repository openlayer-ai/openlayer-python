# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["RuleCreateParams"]


class RuleCreateParams(TypedDict, total=False):
    name: Required[str]
    """The rule name."""

    scope: Required[Literal["project", "workspace"]]
    """
    Whether the rule is evaluated once for the whole workspace, or once per project
    the rule's frameworks apply to.
    """

    type: Required[Literal["platform", "evidence"]]
    """
    `platform` rules are evaluated automatically from the state of your Openlayer
    workspace. `evidence` rules are satisfied by attaching evidence.
    """

    assignee_id: Annotated[Optional[str], PropertyInfo(alias="assigneeId")]
    """The user responsible for satisfying the rule."""

    automation_params: Annotated[Optional[Dict[str, object]], PropertyInfo(alias="automationParams")]
    """Configuration for the platform check, when the automation takes parameters."""

    automation_type: Annotated[Optional[str], PropertyInfo(alias="automationType")]
    """
    Which workspace signal a platform rule checks, for example
    `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. `null` for
    evidence rules.
    """

    deactivated: bool
    """Whether the rule is excluded from compliance calculations."""

    description: Optional[str]
    """What the rule requires."""

    evidence_type: Annotated[
        Optional[Literal["document", "text", "url", "categoryValue"]], PropertyInfo(alias="evidenceType")
    ]
    """The kind of evidence that satisfies the rule. `null` for platform rules."""

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
