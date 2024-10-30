# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["PresignedURLCreateResponse"]


class PresignedURLCreateResponse(BaseModel):
    storage_uri: str = FieldInfo(alias="storageUri")
    """The storage URI to send back to the backend after the upload was completed."""

    url: str
    """The presigned url."""

    fields: Optional[object] = None
    """Fields to include in the body of the upload. Only needed by s3"""
