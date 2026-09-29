# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["APIKeyCreateParams"]


class APIKeyCreateParams(TypedDict, total=False):
    expires_at: Annotated[Union[str, datetime, None], PropertyInfo(alias="expiresAt", format="iso8601")]
    """When the key stops authenticating.

    `null` means the key never expires. Set when the key is created or rotated, and
    must be in the future. When the request is authenticated with an API key that
    expires, the result can't be later than that key's expiry. On create, omit it to
    inherit that expiry. On rotate, omit it to keep the current one. It can't be
    changed with an update; rotate the key instead.
    """

    name: Optional[str]
    """The API key name."""
