# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["InviteCreateResponse", "Item", "ItemCreator", "ItemWorkspace"]


class ItemCreator(BaseModel):
    id: Optional[str] = None
    """The invite creator id."""

    name: Optional[str] = None
    """The invite creator name."""

    username: Optional[str] = None
    """The invite creator username."""


class ItemWorkspace(BaseModel):
    id: str

    date_created: datetime = FieldInfo(alias="dateCreated")

    member_count: int = FieldInfo(alias="memberCount")

    name: str

    slug: str


class Item(BaseModel):
    id: str
    """The invite id."""

    creator: ItemCreator

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The invite creation date."""

    email: str
    """The invite email."""

    role: Literal["ADMIN", "MEMBER", "VIEWER"]
    """The invite role."""

    status: Literal["accepted", "pending"]
    """The invite status."""

    workspace: ItemWorkspace


class InviteCreateResponse(BaseModel):
    items: List[Item]
