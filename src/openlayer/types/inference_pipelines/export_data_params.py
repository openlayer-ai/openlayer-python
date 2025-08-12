# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ExportDataParams"]


class ExportDataParams(TypedDict, total=False):
    start: Required[int]
    """Start timestamp (Unix timestamp in seconds) for the data export range."""

    end: Required[int]
    """End timestamp (Unix timestamp in seconds) for the data export range."""

    fmt: Required[Literal["json", "csv"]]
    """Export format. Supported formats: 'json', 'csv'."""
