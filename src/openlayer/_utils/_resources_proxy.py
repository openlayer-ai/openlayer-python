from __future__ import annotations

from typing import Any
from typing_extensions import override

from ._proxy import LazyProxy


class ResourcesProxy(LazyProxy[Any]):
    """A proxy for the `openlayer.resources` module.

    This is used so that we can lazily import `openlayer.resources` only when
    needed *and* so that users can just import `openlayer` and reference `openlayer.resources`
    """

    @override
    def __load__(self) -> Any:
        import importlib

        mod = importlib.import_module("openlayer.resources")
        return mod


resources = ResourcesProxy().__as_proxied__()
