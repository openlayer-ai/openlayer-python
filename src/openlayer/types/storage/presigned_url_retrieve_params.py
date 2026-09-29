# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["PresignedURLRetrieveParams"]


class PresignedURLRetrieveParams(TypedDict, total=False):
    storage_uri: Required[Annotated[str, PropertyInfo(alias="storageUri")]]
    """
    The object's storage uri, for example `outputs.storageUri` from a framework
    export's background task.
    """
