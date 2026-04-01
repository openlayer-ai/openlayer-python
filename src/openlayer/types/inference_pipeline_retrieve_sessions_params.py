# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, Annotated, TypeAlias, TypedDict

from .._types import SequenceNotStr
from .._utils import PropertyInfo

__all__ = [
    "InferencePipelineRetrieveSessionsParams",
    "ColumnFilter",
    "ColumnFilterSetColumnFilter",
    "ColumnFilterNumericColumnFilter",
    "ColumnFilterStringColumnFilter",
]


class InferencePipelineRetrieveSessionsParams(TypedDict, total=False):
    asc: bool
    """Whether or not to sort on the sortColumn in ascending order."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    sort_column: Annotated[str, PropertyInfo(alias="sortColumn")]
    """Name of the column to sort on"""

    column_filters: Annotated[Optional[Iterable[ColumnFilter]], PropertyInfo(alias="columnFilters")]

    exclude_row_id_list: Annotated[Optional[Iterable[int]], PropertyInfo(alias="excludeRowIdList")]

    not_search_query_and: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="notSearchQueryAnd")]

    not_search_query_or: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="notSearchQueryOr")]

    row_id_list: Annotated[Optional[Iterable[int]], PropertyInfo(alias="rowIdList")]

    search_query_and: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="searchQueryAnd")]

    search_query_or: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="searchQueryOr")]


class ColumnFilterSetColumnFilter(TypedDict, total=False):
    measurement: Required[str]
    """The name of the column."""

    operator: Required[Literal["contains_none", "contains_any", "contains_all", "one_of", "none_of"]]

    value: Required[SequenceNotStr[Union[str, float]]]


class ColumnFilterNumericColumnFilter(TypedDict, total=False):
    measurement: Required[str]
    """The name of the column."""

    operator: Required[Literal[">", ">=", "is", "<", "<=", "!="]]

    value: Required[Optional[float]]


class ColumnFilterStringColumnFilter(TypedDict, total=False):
    measurement: Required[str]
    """The name of the column."""

    operator: Required[Literal["is", "!="]]

    value: Required[Union[str, bool]]


ColumnFilter: TypeAlias = Union[
    ColumnFilterSetColumnFilter, ColumnFilterNumericColumnFilter, ColumnFilterStringColumnFilter
]
