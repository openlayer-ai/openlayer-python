# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from ..._models import BaseModel

__all__ = ["PresignedURLRetrieveResponse"]


class PresignedURLRetrieveResponse(BaseModel):
    url: str
    """The presigned url. Short-lived -- download promptly."""
