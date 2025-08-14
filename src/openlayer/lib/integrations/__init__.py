"""Integrations for Openlayer."""

# Core integrations that are always available
__all__ = []

# Optional imports - only import if dependencies are available
try:
    from .langchain_callback import OpenlayerHandler, AsyncOpenlayerHandler

    __all__.extend(["OpenlayerHandler", "AsyncOpenlayerHandler"])
except ImportError:
    pass

try:
    from .openai_agents import OpenlayerTracerProcessor

    __all__.extend(["OpenlayerTracerProcessor"])
except ImportError:
    pass

try:
    from .oci_tracer import trace_oci_genai

    __all__.extend(["trace_oci_genai"])
except ImportError:
    pass
