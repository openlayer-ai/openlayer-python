"""Openlayer lib.
"""

__all__ = [
    "trace",
    "trace_openai",
    "trace_openai_assistant_thread_run",
]

# ---------------------------------- Tracing --------------------------------- #
from .tracing import tracer

trace = tracer.trace


def trace_openai(client):
    """Trace OpenAI chat completions."""
    # pylint: disable=import-outside-toplevel
    import openai

    from .integrations import openai_tracer

    if not isinstance(client, (openai.Client, openai.AzureOpenAI)):
        raise ValueError("Invalid client. Please provide an OpenAI client.")
    return openai_tracer.trace_openai(client)


def trace_openai_assistant_thread_run(client, run):
    """Trace OpenAI Assistant thread run."""
    # pylint: disable=import-outside-toplevel
    from .integrations import openai_tracer

    return openai_tracer.trace_openai_assistant_thread_run(client, run)
