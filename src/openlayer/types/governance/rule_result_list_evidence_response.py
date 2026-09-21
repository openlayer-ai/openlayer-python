# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["RuleResultListEvidenceResponse", "Item"]


class Item(BaseModel):
    id: str
    """The evidence id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The user who attached the evidence."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    description: Optional[str] = None
    """A description of what the evidence shows."""

    name: Optional[str] = None
    """The evidence name."""

    storage_uri: Optional[str] = FieldInfo(alias="storageUri", default=None)
    """Where the uploaded file is stored.

    Set when the rule's `evidenceType` is `document`.
    """

    text: Optional[str] = None
    """The evidence text. Set when the rule's `evidenceType` is `text`."""

    url: Optional[str] = None
    """A link to the evidence. Set when the rule's `evidenceType` is `url`."""


class RuleResultListEvidenceResponse(BaseModel):
    items: List[Item]
