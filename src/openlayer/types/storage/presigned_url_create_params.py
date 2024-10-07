# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["PresignedURLCreateParams"]


class PresignedURLCreateParams(TypedDict, total=False):
    object_name: Required[Annotated[str, PropertyInfo(alias="objectName")]]
    """The name of the object."""
