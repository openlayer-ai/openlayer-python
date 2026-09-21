# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["RuleResultCreateEvidenceParams"]


class RuleResultCreateEvidenceParams(TypedDict, total=False):
    description: Optional[str]
    """A description of what the evidence shows."""

    name: Optional[str]
    """The evidence name."""

    storage_uri: Annotated[Optional[str], PropertyInfo(alias="storageUri")]
    """Where the uploaded file is stored.

    Set when the rule's `evidenceType` is `document`.
    """

    text: Optional[str]
    """The evidence text. Set when the rule's `evidenceType` is `text`."""

    url: Optional[str]
    """A link to the evidence. Set when the rule's `evidenceType` is `url`."""
