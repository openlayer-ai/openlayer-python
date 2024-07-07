# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Required, TypedDict

__all__ = ["InferencePipelineCreateParams"]


class InferencePipelineCreateParams(TypedDict, total=False):
    description: Required[Optional[str]]
    """The inference pipeline description."""

    name: Required[str]
    """The inference pipeline name."""
