# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ...._models import BaseModel

__all__ = [
    "DocumentListResponse",
    "Item",
    "ItemSection",
    "ItemSectionRule",
    "ItemSectionSubsection",
    "ItemSectionSubsectionRule",
]


class ItemSectionRule(BaseModel):
    id: str
    """The rule id."""

    name: str
    """The rule name."""

    scope: Literal["project", "workspace"]
    """
    Whether the rule is evaluated once for the whole workspace, or once per project
    the rule's frameworks apply to.
    """

    type: Literal["platform", "evidence"]
    """
    `platform` rules are evaluated automatically from the state of your Openlayer
    workspace. `evidence` rules are satisfied by attaching evidence.
    """

    automation_params: Optional[Dict[str, object]] = FieldInfo(alias="automationParams", default=None)
    """Configuration for the platform check, when the automation takes parameters."""

    automation_type: Optional[str] = FieldInfo(alias="automationType", default=None)
    """
    Which workspace signal a platform rule checks, for example
    `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. `null` for
    evidence rules.
    """

    date_created: Optional[datetime] = FieldInfo(alias="dateCreated", default=None)
    """The creation date."""

    date_updated: Optional[datetime] = FieldInfo(alias="dateUpdated", default=None)
    """The last update date."""

    deactivated: Optional[bool] = None
    """Whether the rule is excluded from compliance calculations."""

    description: Optional[str] = None
    """What the rule requires."""

    evidence_type: Optional[Literal["document", "text", "url", "categoryValue"]] = FieldInfo(
        alias="evidenceType", default=None
    )
    """The kind of evidence that satisfies the rule. `null` for platform rules."""

    immutable: Optional[bool] = None
    """Whether the rule is managed by Openlayer and cannot be edited."""

    renewal_cadence_days: Optional[int] = FieldInfo(alias="renewalCadenceDays", default=None)
    """How often evidence must be renewed, in days.

    Once evidence is older than this, the rule result becomes `due_soon` and then
    `failing`.
    """


class ItemSectionSubsectionRule(BaseModel):
    id: str
    """The rule id."""

    name: str
    """The rule name."""

    scope: Literal["project", "workspace"]
    """
    Whether the rule is evaluated once for the whole workspace, or once per project
    the rule's frameworks apply to.
    """

    type: Literal["platform", "evidence"]
    """
    `platform` rules are evaluated automatically from the state of your Openlayer
    workspace. `evidence` rules are satisfied by attaching evidence.
    """

    automation_params: Optional[Dict[str, object]] = FieldInfo(alias="automationParams", default=None)
    """Configuration for the platform check, when the automation takes parameters."""

    automation_type: Optional[str] = FieldInfo(alias="automationType", default=None)
    """
    Which workspace signal a platform rule checks, for example
    `monitoring_mode_enabled`, `test_setup`, or `project_owner_set`. `null` for
    evidence rules.
    """

    date_created: Optional[datetime] = FieldInfo(alias="dateCreated", default=None)
    """The creation date."""

    date_updated: Optional[datetime] = FieldInfo(alias="dateUpdated", default=None)
    """The last update date."""

    deactivated: Optional[bool] = None
    """Whether the rule is excluded from compliance calculations."""

    description: Optional[str] = None
    """What the rule requires."""

    evidence_type: Optional[Literal["document", "text", "url", "categoryValue"]] = FieldInfo(
        alias="evidenceType", default=None
    )
    """The kind of evidence that satisfies the rule. `null` for platform rules."""

    immutable: Optional[bool] = None
    """Whether the rule is managed by Openlayer and cannot be edited."""

    renewal_cadence_days: Optional[int] = FieldInfo(alias="renewalCadenceDays", default=None)
    """How often evidence must be renewed, in days.

    Once evidence is older than this, the rule result becomes `due_soon` and then
    `failing`.
    """


class ItemSectionSubsection(BaseModel):
    id: str
    """The subsection id."""

    number: str
    """The subsection number as it appears in the source standard."""

    section_id: str = FieldInfo(alias="sectionId")
    """The section the subsection belongs to."""

    sort_order: int = FieldInfo(alias="sortOrder")
    """The position of the subsection within its section."""

    title: str
    """The subsection title."""

    rule_count: Optional[int] = FieldInfo(alias="ruleCount", default=None)
    """How many rules are linked to this subsection."""

    rules: Optional[List[ItemSectionSubsectionRule]] = None
    """The rules linked to this subsection."""

    text: Optional[str] = None
    """The subsection text. This is the requirement your rules are mapped against."""


class ItemSection(BaseModel):
    id: str
    """The section id."""

    document_id: str = FieldInfo(alias="documentId")
    """The document the section belongs to."""

    number: str
    """The section number as it appears in the source standard."""

    sort_order: int = FieldInfo(alias="sortOrder")
    """The position of the section within the document."""

    title: str
    """The section title."""

    rule_count: Optional[int] = FieldInfo(alias="ruleCount", default=None)
    """How many rules are linked to this section, including its subsections.

    Use it to decide whether to fetch the section's rules.
    """

    rules: Optional[List[ItemSectionRule]] = None
    """The rules linked directly to this section."""

    subsections: Optional[List[ItemSectionSubsection]] = None
    """The section's subsections, in display order."""

    text: Optional[str] = None
    """The section text."""


class Item(BaseModel):
    id: str
    """The document id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    framework_id: str = FieldInfo(alias="frameworkId")
    """The framework the document belongs to."""

    title: str
    """The document title."""

    sections: Optional[List[ItemSection]] = None
    """The document's sections, in display order.

    Only returned when retrieving a single document.
    """


class DocumentListResponse(BaseModel):
    items: List[Item]
