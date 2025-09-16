"""Openlayer lib."""

__all__ = [
    "configure",
    "trace",
    "trace_anthropic", 
    "trace_openai",
    "trace_openai_assistant_thread_run",
    "trace_mistral",
    "trace_groq",
    "trace_async_openai",
    "trace_async",
    "trace_bedrock",
    "trace_oci_genai",
    "trace_oci",  # Alias for backward compatibility
    "trace_litellm",
    "update_current_trace",
    "update_current_step",
    # User and session context functions
    "set_user_session_context",
    "update_trace_user_session",
    "get_current_user_id",
    "get_current_session_id",
    "clear_user_session_context",
]

# ---------------------------------- Tracing --------------------------------- #
from .tracing import tracer
from .tracing.context import (
    set_user_session_context,
    update_trace_user_session,
    get_current_user_id,
    get_current_session_id,
    clear_user_session_context,
)

configure = tracer.configure
trace = tracer.trace
trace_async = tracer.trace_async
update_current_trace = tracer.update_current_trace
update_current_step = tracer.update_current_step


def trace_anthropic(client):
    """Trace Anthropic chat completions."""
    # pylint: disable=import-outside-toplevel
    import anthropic

    from .integrations import anthropic_tracer

    if not isinstance(client, anthropic.Anthropic):
        raise ValueError("Invalid client. Please provide an Anthropic client.")
    return anthropic_tracer.trace_anthropic(client)


def trace_openai(client):
    """Trace OpenAI chat completions."""
    # pylint: disable=import-outside-toplevel
    import openai

    from .integrations import openai_tracer

    if not isinstance(client, (openai.Client, openai.AzureOpenAI)):
        raise ValueError("Invalid client. Please provide an OpenAI client.")
    return openai_tracer.trace_openai(client)


def trace_async_openai(client):
    """Trace OpenAI chat completions."""
    # pylint: disable=import-outside-toplevel
    import openai

    from .integrations import async_openai_tracer

    if not isinstance(client, (openai.AsyncOpenAI, openai.AsyncAzureOpenAI)):
        raise ValueError("Invalid client. Please provide an OpenAI client.")
    return async_openai_tracer.trace_async_openai(client)


def trace_openai_assistant_thread_run(client, run):
    """Trace OpenAI Assistant thread run."""
    # pylint: disable=import-outside-toplevel
    from .integrations import openai_tracer

    return openai_tracer.trace_openai_assistant_thread_run(client, run)


def trace_mistral(client):
    """Trace Mistral chat completions."""
    # pylint: disable=import-outside-toplevel
    import mistralai

    from .integrations import mistral_tracer

    if not isinstance(client, mistralai.Mistral):
        raise ValueError("Invalid client. Please provide a Mistral client.")
    return mistral_tracer.trace_mistral(client)


def trace_groq(client):
    """Trace Groq queries."""
    # pylint: disable=import-outside-toplevel
    import groq

    from .integrations import groq_tracer

    if not isinstance(client, groq.Groq):
        raise ValueError("Invalid client. Please provide a Groq client.")
    return groq_tracer.trace_groq(client)


def trace_bedrock(client):
    """Trace AWS Bedrock model invocations."""
    # pylint: disable=import-outside-toplevel
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for Bedrock tracing. Install with: pip install boto3"
        )

    from .integrations import bedrock_tracer

    # Check if it's a boto3 client for bedrock-runtime service
    if (
        not hasattr(client, "_service_model")
        or client._service_model.service_name != "bedrock-runtime"
    ):
        raise ValueError(
            "Invalid client. Please provide a boto3 bedrock-runtime client."
        )
    return bedrock_tracer.trace_bedrock(client)



def trace_oci_genai(client, estimate_tokens: bool = True):
    """Trace OCI GenAI chat completions.
    
    Args:
        client: OCI GenAI client.
        estimate_tokens: Whether to estimate tokens when not available. Defaults to True.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import oci
    except ImportError:
        raise ImportError("oci is required for OCI GenAI tracing. Install with: pip install oci")

    from .integrations import oci_tracer

    if not isinstance(client, oci.generative_ai_inference.GenerativeAiInferenceClient):
        raise ValueError("Invalid client. Please provide an OCI GenAI client.")

    return oci_tracer.trace_oci_genai(client, estimate_tokens=estimate_tokens)


# --------------------------------- OCI GenAI -------------------------------- #
# Alias for backward compatibility
trace_oci = trace_oci_genai


# --------------------------------- LiteLLM ---------------------------------- #
def trace_litellm():
    """Enable tracing for LiteLLM completions.
    
    This function patches litellm.completion to automatically trace all completions
    made through the LiteLLM library, which provides a unified interface to 100+ LLM APIs.
    
    Example:
        >>> import litellm
        >>> from openlayer.lib import trace_litellm
        >>> 
        >>> # Enable tracing
        >>> trace_litellm()
        >>> 
        >>> # Use LiteLLM normally - tracing happens automatically
        >>> response = litellm.completion(
        ...     model="gpt-3.5-turbo",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     inference_id="custom-id-123"  # Optional Openlayer parameter
        ... )
    """
    # pylint: disable=import-outside-toplevel
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LiteLLM tracing. Install with: pip install litellm"
        )

    from .integrations import litellm_tracer

    return litellm_tracer.trace_litellm()
