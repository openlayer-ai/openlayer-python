"""Auto-instrumentation registry and orchestrator.

This module powers ``openlayer.lib.init(auto_instrument=True)`` — when called,
it walks a registry of supported LLM SDK integrations, detects which ones are
installed (via ``importlib.util.find_spec``), and applies their patch function
so that every newly-constructed client is auto-traced.

The two helper functions ``_patch_class_init`` and ``_unpatch_class_init`` are
the shared patch/unpatch primitives used by each ``<x>_tracer.py`` module to
wrap an SDK's client class ``__init__``. They are idempotent and reversible.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# Attribute marker placed on a client *class* once its __init__ has been
# wrapped. Distinct from `_openlayer_patched` which marks individual *instances*
# after a per-client wrap. Both are needed: the class marker prevents re-wrapping
# __init__, the instance marker prevents re-wrapping methods if the user mixes
# auto-instrument with an explicit trace_<x>(client) call on the same instance.
_CLASS_PATCHED_ATTR = "_openlayer_class_patched"


def _patch_class_init(
    cls: Type[Any],
    wrap_fn: Callable[[Any], Any],
) -> None:
    """Wrap ``cls.__init__`` so every newly-constructed instance is auto-traced.

    Idempotent: re-applying on an already-patched class is a no-op. The wrap
    failure is caught and logged so a broken integration doesn't break client
    construction.
    """
    if getattr(cls, _CLASS_PATCHED_ATTR, False):
        return

    original_init = cls.__init__

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        try:
            wrap_fn(self)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Openlayer: failed to auto-trace %s instance: %s",
                cls.__name__,
                e,
            )

    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    setattr(cls, _CLASS_PATCHED_ATTR, True)


def _unpatch_class_init(cls: Type[Any]) -> None:
    """Restore the original ``cls.__init__``. No-op if not patched."""
    if not getattr(cls, _CLASS_PATCHED_ATTR, False):
        return
    wrapped = cls.__init__
    original = getattr(wrapped, "__wrapped__", None)
    if original is not None:
        cls.__init__ = original  # type: ignore[method-assign]
    try:
        delattr(cls, _CLASS_PATCHED_ATTR)
    except AttributeError:
        pass


@dataclass(frozen=True)
class IntegrationSpec:
    """Declarative entry in the auto-instrumentation registry."""

    name: str
    """Public name users pass in ``auto_instrument=[...]``."""

    probe: str
    """Top-level package name used with ``importlib.util.find_spec`` to detect
    whether the SDK is installed. Only the first dotted segment is probed."""

    patch: Callable[[], None]
    """Idempotent function that patches the SDK's client class(es)."""

    unpatch: Optional[Callable[[], None]] = None
    """Optional restorer. Some integrations (litellm, portkey) don't have one
    today; they're still registered so the patch path works."""


# ----------------------------- Lazy import helpers ----------------------------- #
#
# The registry uses lambdas that import the tracer module on first call. This
# keeps the import-time cost of `openlayer.lib` low — we don't load every
# LLM SDK's tracer module just because the user did `from openlayer.lib import init`.


def _patch_via(module_name: str, attr: str) -> Callable[[], None]:
    """Return a lazy patcher: imports ``.<module_name>`` and calls its ``attr``."""

    def _do_patch() -> None:
        mod = importlib.import_module(f".{module_name}", package=__package__)
        getattr(mod, attr)()

    return _do_patch


# ----------------------------- Registry ----------------------------- #

REGISTRY: Tuple[IntegrationSpec, ...] = (
    IntegrationSpec(
        "openai",
        "openai",
        _patch_via("openai_tracer", "_patch_openai"),
        _patch_via("openai_tracer", "_unpatch_openai"),
    ),
    IntegrationSpec(
        "anthropic",
        "anthropic",
        _patch_via("anthropic_tracer", "_patch_anthropic"),
        _patch_via("anthropic_tracer", "_unpatch_anthropic"),
    ),
    IntegrationSpec(
        "mistral",
        "mistralai",
        _patch_via("mistral_tracer", "_patch_mistral"),
        _patch_via("mistral_tracer", "_unpatch_mistral"),
    ),
    IntegrationSpec(
        "groq",
        "groq",
        _patch_via("groq_tracer", "_patch_groq"),
        _patch_via("groq_tracer", "_unpatch_groq"),
    ),
    # Two Google Gemini entries, because the two SDKs are separate packages that
    # can be installed side by side:
    #   - `gemini`       -> LEGACY google-generativeai / genai.GenerativeModel
    #                       (maintenance mode)
    #   - `google_genai` -> CURRENT google-genai / genai.Client(), incl. Vertex
    # `_is_installed` probes the full dotted path, which matters here: `google` is
    # a namespace package, so one can be importable without the other.
    IntegrationSpec(
        "gemini",
        "google.generativeai",
        _patch_via("gemini_tracer", "_patch_gemini"),
        _patch_via("gemini_tracer", "_unpatch_gemini"),
    ),
    IntegrationSpec(
        "google_genai",
        "google.genai",
        _patch_via("google_genai_tracer", "_patch_google_genai"),
        _patch_via("google_genai_tracer", "_unpatch_google_genai"),
    ),
    IntegrationSpec(
        "oci",
        "oci",
        _patch_via("oci_tracer", "_patch_oci"),
        _patch_via("oci_tracer", "_unpatch_oci"),
    ),
    IntegrationSpec(
        "azure_content_understanding",
        "azure.ai.contentunderstanding",
        _patch_via("azure_content_understanding_tracer", "_patch_acu"),
        _patch_via("azure_content_understanding_tracer", "_unpatch_acu"),
    ),
    IntegrationSpec(
        "litellm",
        "litellm",
        _patch_via("litellm_tracer", "trace_litellm"),
        None,
    ),
    IntegrationSpec(
        "portkey",
        "portkey_ai",
        _patch_via("portkey_tracer", "trace_portkey"),
        _patch_via("portkey_tracer", "_unpatch_portkey"),
    ),
    IntegrationSpec(
        "google_adk",
        "google.adk",
        _patch_via("google_adk_tracer", "trace_google_adk"),
        _patch_via("google_adk_tracer", "unpatch_google_adk"),
    ),
)


_REGISTRY_BY_NAME: Dict[str, IntegrationSpec] = {s.name: s for s in REGISTRY}


def _is_installed(probe: str) -> bool:
    """Return True if the full dotted ``probe`` path is importable.

    Uses ``find_spec`` so the SDK itself is not imported at probe time — only
    the patch function imports it, and only when we've committed to patching.
    The full path matters for namespace-package collisions: e.g. ``google``
    can be installed via ``google.generativeai`` without ``google.adk`` being
    available, so we have to probe ``google.adk`` not just ``google``.
    """
    try:
        return importlib.util.find_spec(probe) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def auto_instrument(
    targets: Union[bool, List[str]] = True,
) -> Dict[str, bool]:
    """Detect and patch installed LLM SDKs so new client instances are auto-traced.

    Args:
        targets: ``True`` (default) patches every installed supported SDK.
            ``False`` is a no-op. A list of names patches only that subset
            (e.g. ``["openai", "anthropic"]``).

    Returns:
        Dict mapping integration name to a boolean: ``True`` if patched
        successfully, ``False`` if skipped (not installed, or patch raised).
    """
    if targets is False:
        return {}

    if targets is True:
        enabled = {s.name for s in REGISTRY}
    else:
        enabled = set(targets)
        unknown = enabled - set(_REGISTRY_BY_NAME)
        if unknown:
            logger.warning(
                "Openlayer: unknown auto_instrument targets: %s",
                sorted(unknown),
            )
            enabled &= set(_REGISTRY_BY_NAME)

    results: Dict[str, bool] = {}
    for spec in REGISTRY:
        if spec.name not in enabled:
            continue
        if not _is_installed(spec.probe):
            results[spec.name] = False
            logger.debug("Openlayer: skipped %s (not installed)", spec.name)
            continue
        try:
            spec.patch()
            results[spec.name] = True
            logger.info("Openlayer: auto-instrumented %s", spec.name)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Openlayer: failed to instrument %s: %s", spec.name, e)
            results[spec.name] = False
    return results


def unpatch_all() -> None:
    """Restore the original ``__init__`` for every patched integration.

    Integrations without an unpatch function are skipped. After this call,
    previously-constructed (and already-traced) client instances are unaffected
    — only future construction goes through the restored ``__init__``.
    """
    for spec in REGISTRY:
        if spec.unpatch is None:
            continue
        try:
            spec.unpatch()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Openlayer: failed to unpatch %s: %s", spec.name, e)
