# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List

from ..._models import BaseModel

__all__ = ["RowListResponse", "Item"]


class Item(BaseModel):
    openlayer_row_id: int


class RowListResponse(BaseModel):
    items: List[Item]
