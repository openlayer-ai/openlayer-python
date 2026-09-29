# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from datetime import datetime
from typing_extensions import Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["APIKeyRotateParams"]


class APIKeyRotateParams(TypedDict, total=False):
    workspace_id: Required[Annotated[str, PropertyInfo(alias="workspaceId")]]

    expires_at: Annotated[Union[str, datetime, None], PropertyInfo(alias="expiresAt", format="iso8601")]
    """When the key stops authenticating.

    `null` means the key never expires. Set when the key is created or rotated, and
    must be in the future. When the request is authenticated with an API key that
    expires, the result can't be later than that key's expiry.
    """

    grace_period_hours: Annotated[int, PropertyInfo(alias="gracePeriodHours")]
    """Hours the previous secret keeps authenticating."""
