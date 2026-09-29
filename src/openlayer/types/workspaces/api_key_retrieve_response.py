# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["APIKeyRetrieveResponse"]


class APIKeyRetrieveResponse(BaseModel):
    id: str
    """The API key id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The API key creation date."""

    date_last_used: Optional[datetime] = FieldInfo(alias="dateLastUsed", default=None)
    """The API key last use date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The API key last update date."""

    secure_key: str = FieldInfo(alias="secureKey")
    """An obfuscated hint of the API key value.

    When a key is created or rotated this also holds the full secret, for backward
    compatibility; prefer `secret`.
    """

    status: Literal["active", "rotating", "expired"]
    """The key's lifecycle state.

    `active`: the current secret authenticates. `rotating`: the key was rotated and
    the previous secret still authenticates until `previousKeyExpiresAt`. `expired`:
    `expiresAt` has passed and no secret authenticates.
    """

    expires_at: Optional[datetime] = FieldInfo(alias="expiresAt", default=None)
    """When the key stops authenticating.

    `null` means the key never expires. Set when the key is created or rotated, and
    must be in the future. When the request is authenticated with an API key that
    expires, the result can't be later than that key's expiry.
    """

    last_rotated_at: Optional[datetime] = FieldInfo(alias="lastRotatedAt", default=None)
    """When the key was last rotated."""

    name: Optional[str] = None
    """The API key name."""

    previous_key_expires_at: Optional[datetime] = FieldInfo(alias="previousKeyExpiresAt", default=None)
    """While `status` is `rotating`, when the previous secret stops authenticating."""

    secret: Optional[str] = None
    """The full API key.

    Only present in the response that creates or rotates the key, and never shown
    again.
    """
