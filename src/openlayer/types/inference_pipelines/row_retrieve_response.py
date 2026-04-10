# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._models import BaseModel

__all__ = ["RowRetrieveResponse"]


class RowRetrieveResponse(BaseModel):
    row: Optional[object] = None

    success: Optional[bool] = None
