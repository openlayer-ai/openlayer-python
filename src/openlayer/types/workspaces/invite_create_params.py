# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, TypedDict

from ..._types import SequenceNotStr

__all__ = ["InviteCreateParams"]


class InviteCreateParams(TypedDict, total=False):
    emails: SequenceNotStr[str]

    role: Literal["ADMIN", "MEMBER", "VIEWER"]
    """The member role."""
