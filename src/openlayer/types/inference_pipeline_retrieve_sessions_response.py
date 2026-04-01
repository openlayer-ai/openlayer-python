# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List
from datetime import datetime

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["InferencePipelineRetrieveSessionsResponse", "Item"]


class Item(BaseModel):
    id: str
    """The unique session identifier"""

    cost: float
    """Total cost for the session"""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """Latest/most recent timestamp in the session"""

    date_of_first_record: datetime = FieldInfo(alias="dateOfFirstRecord")
    """Timestamp of the first request in the session"""

    date_of_last_record: datetime = FieldInfo(alias="dateOfLastRecord")
    """Timestamp of the last request in the session"""

    duration: float
    """Duration between first and last request (in milliseconds)"""

    first_record: Dict[str, object] = FieldInfo(alias="firstRecord")
    """The complete first record in the session"""

    last_record: Dict[str, object] = FieldInfo(alias="lastRecord")
    """The complete last record in the session"""

    latency: float
    """Total latency for the session (in milliseconds)"""

    records: int
    """Total number of records/traces in the session"""

    tokens: float
    """Total token count for the session"""

    user_ids: List[str] = FieldInfo(alias="userIds")
    """List of unique user IDs that participated in this session"""


class InferencePipelineRetrieveSessionsResponse(BaseModel):
    items: List[Item]
    """Array of session aggregation data"""
