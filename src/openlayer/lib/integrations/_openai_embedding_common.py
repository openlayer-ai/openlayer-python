"""Shared parsing helpers for OpenAI sync + async embedding tracers."""

from typing import Any, Dict, List, Tuple, Union


def parse_embedding_response(
    response: Any,
) -> Tuple[Union[List[float], List[List[float]]], int, int]:
    """Extract (embeddings, dimensions, count) from an OpenAI EmbeddingResponse.

    For a single input, returns the vector directly.
    For a batch, returns a list of vectors.
    """
    try:
        data = getattr(response, "data", None)
        if data is None:
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
