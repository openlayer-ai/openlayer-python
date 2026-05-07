"""Shared parsing helpers for OpenAI-shaped embedding tracers (OpenAI, AsyncOpenAI, LiteLLM)."""

from typing import Any, Dict, List, Optional, Tuple, Union


def parse_embedding_response(
    response: Any,
) -> Tuple[Union[List[float], List[List[float]]], int, int]:
    """Extract (embeddings, dimensions, count) from an OpenAI-shaped EmbeddingResponse.

    For a single input, returns the vector directly.
    For a batch, returns a list of vectors.
    """
    try:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])
        if not data:
            return [], 0, 0
        embeddings = [
            item["embedding"] if isinstance(item, dict) else item.embedding
            for item in data
        ]
        if not embeddings:
            return [], 0, 0
        if len(embeddings) == 1:
            return embeddings[0], len(embeddings[0]), 1
        return embeddings, len(embeddings[0]), len(embeddings)
    except Exception:
        return [], 0, 0


def get_embedding_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract embedding-relevant model parameters from create() kwargs."""
    return {
        "dimensions": kwargs.get("dimensions"),
        "encoding_format": kwargs.get("encoding_format"),
        "user": kwargs.get("user"),
    }


def build_embedding_step_kwargs(
    response: Any,
    call_kwargs: Dict[str, Any],
    start_time: float,
    end_time: float,
    *,
    name: str,
    provider: str,
    inference_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the kwargs to pass to ``tracer.add_embedding_step_to_trace``.

    Common boilerplate for OpenAI-shaped responses (OpenAI sync/async, LiteLLM).
    Callers may layer extra fields (cost, extra_metadata, model_parameters) on
    top of the returned dict before invoking the tracer helper.
    """
    model_name = getattr(response, "model", call_kwargs.get("model", "unknown"))
    embeddings, dim, count = parse_embedding_response(response)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens) if usage else prompt_tokens

    step_kwargs: Dict[str, Any] = {
        "name": name,
        "end_time": end_time,
        "inputs": {"input": call_kwargs.get("input")},
        "output": embeddings,
        "latency": (end_time - start_time) * 1000,
        "tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "model": model_name,
        "model_parameters": get_embedding_model_parameters(call_kwargs),
        "embedding_dimensions": dim,
        "embedding_count": count,
        "raw_output": (
            response.model_dump()
            if hasattr(response, "model_dump")
            else str(response)
        ),
        "provider": provider,
        "metadata": {"provider": provider},
    }
    # Only include id when truthy: passing id=None would overwrite the step's
    # auto-generated UUID via step.log() → setattr().
    if inference_id:
        step_kwargs["id"] = inference_id
    return step_kwargs
