# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["RuleTagListResponse", "Item"]


class Item(BaseModel):
    id: str
    """The rule tag id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The user who created the tag. `null` for tags that ship with Openlayer."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last update date."""

    immutable: bool
    """Whether the tag is managed by Openlayer and cannot be edited or deleted."""

    name: str
    """The tag name."""

    workspace_id: str = FieldInfo(alias="workspaceId")
    """The id of the workspace the tag belongs to."""

    color: Optional[str] = None
    """The color the tag is displayed with."""


class RuleTagListResponse(BaseModel):
    items: List[Item]
