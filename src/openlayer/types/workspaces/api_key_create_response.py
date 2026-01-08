# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["APIKeyCreateResponse"]


class APIKeyCreateResponse(BaseModel):
    id: str
    """The API key id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The API key creation date."""

    date_last_used: Optional[datetime] = FieldInfo(alias="dateLastUsed", default=None)
    """The API key last use date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The API key last update date."""

    secure_key: str = FieldInfo(alias="secureKey")
    """The API key value."""

    name: Optional[str] = None
    """The API key name."""
