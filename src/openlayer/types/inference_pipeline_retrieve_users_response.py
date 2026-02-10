# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List
from datetime import datetime

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["InferencePipelineRetrieveUsersResponse", "Item"]


class Item(BaseModel):
    id: str
    """The unique user identifier"""

    cost: float
    """Total cost for this user"""

    date_of_first_record: datetime = FieldInfo(alias="dateOfFirstRecord")
    """Timestamp of the user's first event/trace"""

    date_of_last_record: datetime = FieldInfo(alias="dateOfLastRecord")
    """Timestamp of the user's last event/trace"""

    records: int
    """Total number of traces/rows for this user"""

    sessions: int
    """Count of unique sessions for this user"""

    tokens: float
    """Total token count for this user"""


class InferencePipelineRetrieveUsersResponse(BaseModel):
    items: List[Item]
    """Array of user aggregation data"""
