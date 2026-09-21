# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Iterable, Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["FrameworkUpdateParams", "Avatar", "ProjectSelector", "ProjectSelectorMatch"]


class FrameworkUpdateParams(TypedDict, total=False):
    avatar: Optional[Avatar]
    """The icon shown for the framework."""

    description: Optional[str]
    """A short description of the framework."""

    enabled: bool
    """Whether the framework is active.

    Rules of a disabled framework are not evaluated and do not count towards
    compliance.
    """

    extended_description: Annotated[Optional[Dict[str, object]], PropertyInfo(alias="extendedDescription")]
    """A longer, rich-text description, as a TipTap JSON document."""

    href: Optional[str]
    """A link to the external standard or regulation the framework is based on."""

    name: str
    """The framework name."""

    project_selector: Annotated[Optional[ProjectSelector], PropertyInfo(alias="projectSelector")]
    """Determines which projects the framework applies to.

    An empty or `null` `match` array applies the framework to every project in the
    workspace.
    """

    tags: SequenceNotStr[str]
    """Free-form labels on the framework."""


class Avatar(TypedDict, total=False):
    """The icon shown for the framework."""

    type: Required[Literal["emoji", "imageUrl", "builtinImage"]]

    value: Required[str]


class ProjectSelectorMatch(TypedDict, total=False):
    property: Required[Literal["taskType", "riskLevel", "riskTotalScore", "name", "ownerId", "modelTypes"]]
    """The project property to match against."""

    value: Required[object]
    """The value to match against.

    Pass an array to match any of several values, or `null` to match projects where
    the property is unset. Omit it for `exists` and `notExists`.
    """

    operator: str
    """How to compare the project property with `value`.

    One of `equals`, `notEquals`, `contains`, `notContains`, `startsWith`,
    `endsWith`, `in`, `notIn`, `greaterThan`, `greaterThanOrEqual`, `lessThan`,
    `lessThanOrEqual`, `equalsIgnoreCase`, `containsIgnoreCase`, `matches`,
    `exists`, or `notExists`.
    """


class ProjectSelector(TypedDict, total=False):
    """Determines which projects the framework applies to.

    An empty or `null` `match` array applies the framework to every project in the workspace.
    """

    match: Optional[Iterable[ProjectSelectorMatch]]
    """Match criteria, ANDed together."""
