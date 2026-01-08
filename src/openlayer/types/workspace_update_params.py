# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["WorkspaceUpdateParams"]


class WorkspaceUpdateParams(TypedDict, total=False):
    invite_code: Annotated[str, PropertyInfo(alias="inviteCode")]
    """The workspace invite code."""

    name: str
    """The workspace name."""

    slug: str
    """The workspace slug."""
