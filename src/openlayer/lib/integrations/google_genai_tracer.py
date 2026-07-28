"""Module with methods used to trace the unified Google Gen AI SDK.

Targets the CURRENT Google Gen AI SDK — package ``google-genai``, module
``google.genai``, client ``genai.Client()`` — in both AI Studio mode and Vertex
AI mode (``vertexai=True``). Generation lives on a sub-object here
(``client.models.generate_content``), not on the client itself.

The LEGACY ``google-generativeai`` SDK (``genai.GenerativeModel``) is handled by
``gemini_tracer.py`` instead. Both packages can be installed side by side, so
both tracers are registered independently in ``_auto.py``.

Coverage: ``generate_content`` and ``generate_content_stream``, sync and async
(``client.aio.models.*``). ``client.chats`` is covered transitively —
``Chats(modules=client.models)`` shares the patched ``Models`` instance and
``Chat.send_message`` delegates to ``generate_content``.
"""

import json
import logging
import sys
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

try:
    from google import genai

    HAVE_GOOGLE_GENAI = True
except ImportError:
    HAVE_GOOGLE_GENAI = False

if TYPE_CHECKING:
    from google import genai

from ..tracing import tracer

logger = logging.getLogger(__name__)

# Cost lookup on the backend is an exact ``(provider.lower(), model)`` match with
# no aliasing, so ``provider`` has to BE a real cost-table slug rather than a
# display label. "gemini" is the LiteLLM-sourced slug; the friendlier "Google"
# label misses newer models (e.g. gemini-3-pro-preview) and carries stale
# OpenRouter-sourced prices for the ``-latest`` aliases.
PROVIDER = "gemini"

# Kept identical to the legacy tracer's step name so spans from the two Google
# SDKs don't fragment dashboards.
STEP_NAME = "Gemini Generation"

# Checked via sys.modules rather than imported: google_adk_tracer is ~66KB, and a
# guarded lazy import would construct an ImportError on every LLM call in the
# common case where ADK isn't installed.
_ADK_TRACER_MODULE = "openlayer.lib.integrations.google_adk_tracer"

_MODEL_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "candidate_count",
    "stop_sequences",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_mime_type",
)


# ----------------------------- Public API ----------------------------- #
def trace_google_genai(
    client: "genai.Client",
) -> "genai.Client":
    """Patch a ``google.genai.Client`` to trace content generation.

    The following information is collected for each generation:
    - start_time / end_time / latency
    - tokens, prompt_tokens, completion_tokens (thinking tokens included, see
      ``_extract_usage``)
    - model, model_parameters
    - inputs, output, raw_output
    - metadata: ``llm_system`` (``google_vertex`` or ``google_ai_studio``), GCP
      project/location in Vertex mode, the raw token split, and
      ``timeToFirstToken`` when streaming.

    Parameters
    ----------
    client : genai.Client
        The Google Gen AI client to patch. Works for both
        ``genai.Client(api_key=...)`` and
        ``genai.Client(vertexai=True, project=..., location=...)``.

    Returns
    -------
    genai.Client
        The same client, patched in place.
    """
    if not HAVE_GOOGLE_GENAI:
        raise ImportError("google-genai is required for Google Gen AI tracing. Install with: pip install google-genai")

    if getattr(client, "_openlayer_patched", False) is True:
        return client

    # `client.models` / `client.aio` are read-only properties backed by instances
    # built eagerly in Client.__init__, so we patch the attribute ON the
    # sub-object. Assigning `client.models = ...` would raise.
    _patch_models(getattr(client, "models", None), client, is_async=False)
    try:
        aio = getattr(client, "aio", None)
        _patch_models(getattr(aio, "models", None), client, is_async=True)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Openlayer: could not patch the async Google Gen AI surface: %s", e)

    try:
        client._openlayer_patched = True  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        # Marking the sub-objects (done in _patch_models) already makes this
        # idempotent, so a read-only client is not fatal.
        logger.debug("Openlayer: could not mark the Google Gen AI client as patched")

    return client


def _patch_google_genai() -> None:
    """Patch ``google.genai.Client.__init__`` so every newly-constructed client
    is auto-traced. Idempotent."""
    if not HAVE_GOOGLE_GENAI:
        return
    # pylint: disable=import-outside-toplevel
    from ._auto import _patch_class_init

    _patch_class_init(genai.Client, trace_google_genai)


def _unpatch_google_genai() -> None:
    if not HAVE_GOOGLE_GENAI:
        return
    # pylint: disable=import-outside-toplevel
    from ._auto import _unpatch_class_init

    _unpatch_class_init(genai.Client)


# ----------------------------- Patching ----------------------------- #
def _patch_models(models_obj: Any, client: Any, is_async: bool) -> None:
    """Wrap ``generate_content`` / ``generate_content_stream`` on a ``Models`` or
    ``AsyncModels`` instance. Idempotent per instance."""
    if models_obj is None or getattr(models_obj, "_openlayer_patched", False) is True:
        return

    original_generate = getattr(models_obj, "generate_content", None)
    original_stream = getattr(models_obj, "generate_content_stream", None)

    if original_generate is not None:
        models_obj.generate_content = (
            _wrap_generate_content_async(original_generate, client)
            if is_async
            else _wrap_generate_content(original_generate, client)
        )

    if original_stream is not None:
        models_obj.generate_content_stream = (
            _wrap_stream_async(original_stream, client) if is_async else _wrap_stream(original_stream, client)
        )

    models_obj._openlayer_patched = True  # pylint: disable=protected-access


def _adk_span_active() -> bool:
    """True when the Google ADK tracer already has an LLM step open.

    ADK's ``Gemini.api_client`` is a ``google.genai.Client``, and its
    ``generate_content_async`` calls ``client.aio.models.generate_content``. Since
    ``google-adk`` depends on ``google-genai``, both integrations activate for ADK
    users, and without this check every ADK LLM call would emit two spans and be
    billed twice.

    The ADK tracer sets ``_current_llm_step`` inside the ``create_step`` block
    that wraps the call, so our patched method runs in a context where it's
    visible (async generators have no independent context, so each ``__anext__``
    runs in the driver's context).
    """
    module = sys.modules.get(_ADK_TRACER_MODULE)
    if module is None:
        return False
    context_var = getattr(module, "_current_llm_step", None)
    if context_var is None:
        return False
    try:
        return context_var.get() is not None
    except LookupError:
        return False


def _wrap_generate_content(original: Any, client: Any) -> Any:
    @wraps(original)
    def traced_generate_content(*args, **kwargs):
        # Popped before any call-through so the SDK never sees the extra kwarg.
        inference_id = kwargs.pop("inference_id", None)
        if _adk_span_active():
            return original(*args, **kwargs)

        start_time = time.time()
        response = original(*args, **kwargs)
        end_time = time.time()

        try:
            _add_span(
                client=client,
                kwargs=kwargs,
                output=parse_non_streaming_output_data(response),
                usage=getattr(response, "usage_metadata", None),
                raw_output=_serialize(response),
                model_version=getattr(response, "model_version", None),
                start_time=start_time,
                end_time=end_time,
                inference_id=inference_id,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to trace the generate content request with Openlayer. %s", e)

        return response

    return traced_generate_content


def _wrap_generate_content_async(original: Any, client: Any) -> Any:
    @wraps(original)
    async def traced_generate_content_async(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        if _adk_span_active():
            return await original(*args, **kwargs)

        start_time = time.time()
        response = await original(*args, **kwargs)
        end_time = time.time()

        try:
            _add_span(
                client=client,
                kwargs=kwargs,
                output=parse_non_streaming_output_data(response),
                usage=getattr(response, "usage_metadata", None),
                raw_output=_serialize(response),
                model_version=getattr(response, "model_version", None),
                start_time=start_time,
                end_time=end_time,
                inference_id=inference_id,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to trace the async generate content request with Openlayer. %s", e)

        return response

    return traced_generate_content_async


def _wrap_stream(original: Any, client: Any) -> Any:
    @wraps(original)
    def traced_generate_content_stream(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        if _adk_span_active():
            return original(*args, **kwargs)
        stream = original(*args, **kwargs)
        return stream_chunks(stream, client=client, kwargs=kwargs, inference_id=inference_id)

    return traced_generate_content_stream


def _wrap_stream_async(original: Any, client: Any) -> Any:
    @wraps(original)
    async def traced_generate_content_stream_async(*args, **kwargs):
        # ``AsyncModels.generate_content_stream`` is a coroutine function that
        # returns an AsyncIterator — callers ``await`` it and *then* ``async for``.
        # So this wrapper must be ``async def`` returning an async generator; an
        # async generator here would break the ``await``.
        inference_id = kwargs.pop("inference_id", None)
        if _adk_span_active():
            return await original(*args, **kwargs)
        stream = await original(*args, **kwargs)
        return stream_chunks_async(stream, client=client, kwargs=kwargs, inference_id=inference_id)

    return traced_generate_content_stream_async


# ----------------------------- Streaming ----------------------------- #
def stream_chunks(
    stream: Any,
    client: Any,
    kwargs: Dict[str, Any],
    inference_id: Optional[str] = None,
):
    """Yield chunks through unchanged while accumulating a span."""
    collected_output_data: List[str] = []
    raw_outputs: List[Any] = []
    usage = None
    model_version = None
    start_time = time.time()
    first_token_time = None
    end_time = None

    try:
        for i, chunk in enumerate(stream):
            if i == 0:
                first_token_time = time.time()

            # usage_metadata is present on EVERY chunk and is cumulative, so the
            # last one wins. Summing would multiply-count.
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage is not None:
                usage = chunk_usage

            chunk_model_version = getattr(chunk, "model_version", None)
            if chunk_model_version:
                model_version = chunk_model_version

            text = _safe_text(chunk)
            if text:
                collected_output_data.append(text)

            try:
                raw_outputs.append(_serialize(chunk))
            except Exception as e:  # pylint: disable=broad-except
                logger.debug("Failed to serialize chunk: %s", e)

            yield chunk

        end_time = time.time()
    finally:
        # ``finally`` so the span is still emitted when the consumer abandons the
        # iterator part-way (GeneratorExit) or the stream raises.
        if end_time is None:
            end_time = time.time()
        try:
            _add_span(
                client=client,
                kwargs=kwargs,
                output="".join(collected_output_data),
                usage=usage,
                raw_output=raw_outputs,
                model_version=model_version,
                start_time=start_time,
                end_time=end_time,
                inference_id=inference_id,
                metadata_extra={
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)
                },
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to trace the streaming generate content request with Openlayer. %s", e)


async def stream_chunks_async(
    stream: Any,
    client: Any,
    kwargs: Dict[str, Any],
    inference_id: Optional[str] = None,
):
    """Async counterpart of :func:`stream_chunks`."""
    collected_output_data: List[str] = []
    raw_outputs: List[Any] = []
    usage = None
    model_version = None
    start_time = time.time()
    first_token_time = None
    end_time = None

    try:
        i = 0
        # NOTE: a manual counter, not ``enumerate`` — ``enumerate`` returns a
        # plain iterator and cannot be driven by ``async for``.
        async for chunk in stream:
            if i == 0:
                first_token_time = time.time()
            i += 1

            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage is not None:
                usage = chunk_usage

            chunk_model_version = getattr(chunk, "model_version", None)
            if chunk_model_version:
                model_version = chunk_model_version

            text = _safe_text(chunk)
            if text:
                collected_output_data.append(text)

            try:
                raw_outputs.append(_serialize(chunk))
            except Exception as e:  # pylint: disable=broad-except
                logger.debug("Failed to serialize chunk: %s", e)

            yield chunk

        end_time = time.time()
    finally:
        if end_time is None:
            end_time = time.time()
        try:
            _add_span(
                client=client,
                kwargs=kwargs,
                output="".join(collected_output_data),
                usage=usage,
                raw_output=raw_outputs,
                model_version=model_version,
                start_time=start_time,
                end_time=end_time,
                inference_id=inference_id,
                metadata_extra={
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)
                },
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to trace the async streaming generate content request with Openlayer. %s", e)


# ----------------------------- Span assembly ----------------------------- #
def _add_span(
    client: Any,
    kwargs: Dict[str, Any],
    output: Union[str, Dict[str, Any], None],
    usage: Any,
    raw_output: Any,
    model_version: Optional[str],
    start_time: float,
    end_time: float,
    inference_id: Optional[str] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Build the trace arguments and add the chat completion step."""
    requested_model = kwargs.get("model")
    model = _normalize_model_name(requested_model)
    prompt_tokens, completion_tokens, total_tokens, token_extras = _extract_usage(usage)

    metadata = _mode_metadata(client)
    metadata.update(token_extras)
    if model_version:
        metadata["modelVersion"] = model_version
    if requested_model and requested_model != model:
        # Keeps the caller's original string recoverable after normalization.
        metadata["requestedModel"] = requested_model
    if metadata_extra:
        metadata.update(metadata_extra)

    trace_args = create_trace_args(
        end_time=end_time,
        inputs={"prompt": _format_input_messages(kwargs.get("contents"), kwargs.get("config"))},
        output=output,
        latency=(end_time - start_time) * 1000,
        tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        model_parameters=get_model_parameters(kwargs.get("config")),
        raw_output=raw_output,
        id=inference_id,
        metadata=metadata,
    )
    add_to_trace(**trace_args)


def create_trace_args(
    end_time: float,
    inputs: Dict[str, Any],
    output: Union[str, Dict[str, Any], None],
    latency: float,
    tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    model_parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raw_output: Optional[Any] = None,
    id: Optional[str] = None,  # pylint: disable=redefined-builtin
) -> Dict[str, Any]:
    """Returns a dictionary with the trace arguments."""
    trace_args = {
        "end_time": end_time,
        "inputs": inputs,
        "output": output,
        "latency": latency,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "model_parameters": model_parameters,
        "raw_output": raw_output,
        "metadata": metadata if metadata else {},
    }
    if id:
        trace_args["id"] = id
    return trace_args


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    tracer.add_chat_completion_step_to_trace(**kwargs, name=STEP_NAME, provider=PROVIDER)


def _mode_metadata(client: Any) -> Dict[str, Any]:
    """Distinguish Vertex mode from AI Studio mode.

    The same ``Client`` class serves both, so the mode is only knowable from the
    instance. ``llm_system`` matches what the Google ADK tracer already emits.
    """
    metadata: Dict[str, Any] = {}
    try:
        is_vertex = bool(getattr(client, "vertexai", False))
    except Exception:  # pylint: disable=broad-except
        is_vertex = False

    metadata["llm_system"] = "google_vertex" if is_vertex else "google_ai_studio"

    if is_vertex:
        api_client = getattr(client, "_api_client", None)
        if api_client is not None:
            project = getattr(api_client, "project", None)
            location = getattr(api_client, "location", None)
            if project:
                metadata["gcp_project"] = project
            if location:
                metadata["gcp_location"] = location

    return metadata


# ----------------------------- Extraction helpers ----------------------------- #
def _normalize_model_name(model: Optional[str]) -> str:
    """Reduce a Gemini model identifier to the bare slug the cost table uses.

    Cost rows are keyed on bare slugs, so anything else silently prices at $0
    (verified: ``models/gemini-2.5-flash`` returned $0.00 where bare
    ``gemini-2.5-flash`` returned $2.80 on identical tokens).

    Handles every shape the API accepts::

        gemini-2.5-flash                                              -> gemini-2.5-flash
        models/gemini-2.5-flash                                       -> gemini-2.5-flash
        publishers/google/models/gemini-2.5-flash                     -> gemini-2.5-flash
        projects/p/locations/l/publishers/google/models/gemini-2.5-flash -> gemini-2.5-flash

    Version suffixes are deliberately preserved — upstream coverage for them is
    inconsistent, and rewriting them would misreport which model actually ran.
    """
    if not model or not isinstance(model, str):
        return "unknown"
    return model.rsplit("/", 1)[-1] or model


def _extract_usage(usage: Any) -> Tuple[int, int, int, Dict[str, Any]]:
    """Return ``(prompt_tokens, completion_tokens, total_tokens, extras)``.

    Thinking tokens are the subtlety here. ``candidates_token_count`` EXCLUDES
    ``thoughts_token_count``, and thinking is on by default for 2.5-series
    models, so the naive ``prompt + candidates`` badly undercounts — measured 14
    for a call that really consumed 757 (prompt 8, candidates 6, thoughts 743).

    So ``tokens`` uses ``total_token_count`` (documented as prompt + candidates +
    tool_use_prompt + thoughts), and thinking tokens are folded into
    ``completion_tokens`` because Vertex bills them as output and the backend
    prices from ``completionTokens``. The raw split is preserved in metadata.
    """
    if usage is None:
        return 0, 0, 0, {}

    def _count(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    prompt_tokens = _count("prompt_token_count") or 0
    candidates_tokens = _count("candidates_token_count") or 0
    thoughts_tokens = _count("thoughts_token_count") or 0
    tool_use_tokens = _count("tool_use_prompt_token_count") or 0
    cached_tokens = _count("cached_content_token_count") or 0
    total_tokens = _count("total_token_count")

    completion_tokens = candidates_tokens + thoughts_tokens
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens + tool_use_tokens

    extras: Dict[str, Any] = {}
    if candidates_tokens:
        extras["candidatesTokens"] = candidates_tokens
    if thoughts_tokens:
        extras["thoughtsTokens"] = thoughts_tokens
    if tool_use_tokens:
        extras["toolUsePromptTokens"] = tool_use_tokens
    if cached_tokens:
        extras["cachedContentTokens"] = cached_tokens

    return prompt_tokens, completion_tokens, total_tokens, extras


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Normalize a ``GenerateContentConfig`` (pydantic) or its dict form."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True)
        except Exception:  # pylint: disable=broad-except
            pass
    return dict(getattr(config, "__dict__", None) or {})


def get_model_parameters(config: Any) -> Dict[str, Any]:
    """Gets the model parameters from the request config.

    Unlike the legacy SDK's loose ``generation_config`` kwarg, the unified SDK
    carries these on a pydantic ``GenerateContentConfig``.
    """
    config_dict = _config_to_dict(config)
    parameters: Dict[str, Any] = {key: config_dict.get(key) for key in _MODEL_PARAM_KEYS}

    thinking_config = config_dict.get("thinking_config")
    if isinstance(thinking_config, dict):
        if thinking_config.get("thinking_budget") is not None:
            parameters["thinking_budget"] = thinking_config["thinking_budget"]
        if thinking_config.get("include_thoughts") is not None:
            parameters["include_thoughts"] = thinking_config["include_thoughts"]

    return parameters


def _safe_text(obj: Any) -> Optional[str]:
    """Read ``.text`` without letting the SDK's accessor raise or warn through.

    ``GenerateContentResponse.text`` raises when the response holds no text part
    (function calls, safety blocks), so it can never be read bare.
    """
    try:
        text = getattr(obj, "text", None)
    except Exception:  # pylint: disable=broad-except
        return None
    return text if isinstance(text, str) else None


def parse_non_streaming_output_data(
    response: Any,
) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming generation."""
    text = _safe_text(response)
    if text and text.strip():
        return text.strip()

    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts: List[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    text_parts.append(part_text)
                    continue
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    return {
                        "name": getattr(function_call, "name", None) or "",
                        "arguments": dict(getattr(function_call, "args", None) or {}),
                    }
            if text_parts:
                return " ".join(text_parts).strip()
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Could not parse Google Gen AI output data: %s", e)

    return None


def _format_input_messages(contents: Any, config: Any = None) -> List[Dict[str, Any]]:
    """Format request contents into a messages array.

    ``config.system_instruction`` is surfaced as a leading system message, since
    the unified SDK carries it on the config rather than in ``contents``.
    """
    messages: List[Dict[str, Any]] = []

    system_instruction = _config_to_dict(config).get("system_instruction")
    if system_instruction:
        system_message = _to_message(system_instruction)
        system_message["role"] = "system"
        messages.append(system_message)

    if contents is None:
        return messages

    if isinstance(contents, (str, bytes)):
        messages.append({"role": "user", "content": _as_text(contents)})
        return messages

    if isinstance(contents, list):
        for item in contents:
            messages.append(_to_message(item))
        return messages

    messages.append(_to_message(contents))
    return messages


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # pylint: disable=broad-except
            return str(value)
    return value if isinstance(value, str) else str(value)


def _to_message(item: Any) -> Dict[str, Any]:
    """Convert a ``Content``, ``Part``, dict or scalar into a chat message."""
    if isinstance(item, (str, bytes)):
        return {"role": "user", "content": _as_text(item)}

    role = None
    parts = None
    if isinstance(item, dict):
        role = item.get("role")
        parts = item.get("parts")
        if parts is None and "content" in item:
            return {"role": role or "user", "content": item["content"]}
    else:
        role = getattr(item, "role", None)
        parts = getattr(item, "parts", None)

    if parts:
        text_parts: List[str] = []
        for part in parts:
            part_text = _part_to_text(part)
            if part_text:
                text_parts.append(part_text)
        if text_parts:
            return {"role": role or "user", "content": " ".join(text_parts)}

    # A bare Part (no role/parts of its own) is still worth capturing.
    single = _part_to_text(item)
    if single:
        return {"role": role or "user", "content": single}

    return {"role": role or "user", "content": str(item)}


def _part_to_text(part: Any) -> Optional[str]:
    """Best-effort text for a single ``Part``-like object."""
    if isinstance(part, (str, bytes)):
        return _as_text(part)

    if isinstance(part, dict):
        if isinstance(part.get("text"), str):
            return part["text"]
        if part.get("function_call") is not None:
            return json.dumps({"function_call": part["function_call"]}, default=str)
        if part.get("function_response") is not None:
            return json.dumps({"function_response": part["function_response"]}, default=str)
        if part.get("inline_data") is not None:
            return "<inline_data>"
        return None

    text = getattr(part, "text", None)
    if isinstance(text, str):
        return text

    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        return json.dumps(
            {
                "function_call": {
                    "name": getattr(function_call, "name", None) or "",
                    "args": dict(getattr(function_call, "args", None) or {}),
                }
            },
            default=str,
        )

    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        return json.dumps(
            {
                "function_response": {
                    "name": getattr(function_response, "name", None) or "",
                    "response": getattr(function_response, "response", None),
                }
            },
            default=str,
        )

    if getattr(part, "inline_data", None) is not None:
        return "<inline_data>"

    return None


def _serialize(obj: Any) -> Any:
    """Serialize a response or chunk to something JSON-safe."""
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            # mode="json" is what makes the result safe to publish — it coerces
            # datetimes, enums and bytes that the default mode leaves as objects.
            return model_dump(mode="json", exclude_none=True)
        except Exception:  # pylint: disable=broad-except
            pass

    model_dump_json = getattr(obj, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return json.loads(model_dump_json(exclude_none=True))
        except Exception:  # pylint: disable=broad-except
            pass

    if isinstance(obj, dict):
        return obj

    return {"response": str(obj)}
