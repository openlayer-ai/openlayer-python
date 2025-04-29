"""Openlayer lib."""

__all__ = [
    "trace",
    "trace_anthropic",
    "trace_openai",
    "trace_openai_assistant_thread_run",
    "trace_mistral",
    "trace_groq",
    "trace_async_openai",
    "trace_async",
]

# ---------------------------------- Tracing --------------------------------- #
from .tracing import tracer

trace = tracer.trace
trace_async = tracer.trace_async


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
