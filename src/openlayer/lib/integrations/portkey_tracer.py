"""Module with methods used to trace Portkey AI Gateway chat completions."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union, TYPE_CHECKING

try:
    from portkey_ai import Portkey
    HAVE_PORTKEY = True
except ImportError:
    HAVE_PORTKEY = False

if TYPE_CHECKING:
    from portkey_ai import Portkey

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_portkey() -> None:
    """Patch Portkey's chat.completions.create to trace completions.

    The following information is collected for each completion:
    - start_time: The time when the completion was requested.
    - end_time: The time when the completion was received.
    - latency: The time it took to generate the completion.
    - tokens: The total number of tokens used to generate the completion.
    - prompt_tokens: The number of tokens in the prompt.
    - completion_tokens: The number of tokens in the completion.
    - model: The model used to generate the completion.
    - model_parameters: The parameters used to configure the model.
    - raw_output: The raw output of the model.
    - inputs: The inputs used to generate the completion.
    - Portkey-specific metadata (base URL, x-portkey-* headers if available)

    Returns
    -------
    None
        This function patches portkey.chat.completions.create in place.

    Example
    -------
    >>> from portkey_ai import Portkey
    >>> from openlayer.lib import trace_portkey
    >>>
    >>> # Enable tracing
    >>> trace_portkey()
    >>> 
    >>> # Use Portkey normally - tracing happens automatically
    >>> portkey = Portkey(api_key = "YOUR_PORTKEY_API_KEY")
    >>> response = portkey.chat.completions.create(
    ...     model = "@YOUR_PROVIDER_SLUG/MODEL_NAME",
    ...     messages = [
    ...         {"role": "system", "content": "You are a helpful assistant."},
    ...         {"role": "user", "content": "What is Portkey"}
    ...     ],
    ...     inference_id="custom-id-123"  # Optional Openlayer parameter
    ...     max_tokens = 512
    ... )
    """
    if not HAVE_PORTKEY:
        raise ImportError(
            "Portkey library is not installed. Please install it with: pip install portkey-ai"
        )

    # Patch instances on initialization rather than class-level attributes.
    # Some SDKs initialize 'chat' lazily on the instance.
    original_init = Portkey.__init__

    @wraps(original_init)
    def traced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            # Avoid double-patching
            if getattr(self, "_openlayer_portkey_patched", False):
                return
            # Access chat to ensure it's constructed, then wrap create
            chat = getattr(self, "chat", None)
            if chat is None or not hasattr(chat, "completions") or not hasattr(chat.completions, "create"):
                # If the structure isn't present, skip gracefully and log diagnostics
                logger.debug(
                    "Openlayer Portkey tracer: Portkey client missing expected attributes (chat/completions/create). "
                    "Tracing not applied for this instance."
                )
                return
            original_create = chat.completions.create

            @wraps(original_create)
            def traced_create(*c_args, **c_kwargs):
                inference_id = c_kwargs.pop("inference_id", None)
                stream = c_kwargs.get("stream", False)
                if stream:
                    return handle_streaming_create(
                        self,
                        *c_args,
                        create_func=original_create,
                        inference_id=inference_id,
                        **c_kwargs,
                    )
                return handle_non_streaming_create(
                    self,
                    *c_args,
                    create_func=original_create,
                    inference_id=inference_id,
                    **c_kwargs,
                )

            self.chat.completions.create = traced_create
            setattr(self, "_openlayer_portkey_patched", True)
            logger.debug("Openlayer Portkey tracer: successfully patched Portkey client instance for tracing.")
        except Exception as e:
            logger.debug("Failed to patch Portkey client instance for tracing: %s", e)

    Portkey.__init__ = traced_init
    logger.info("Openlayer Portkey tracer: tracing enabled (instance-level patch).")


def handle_streaming_create(
    client: "Portkey",
    *args,
    create_func: callable,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """
    Handles streaming chat.completions.create routed via Portkey.

    Parameters
    ----------
    client : Portkey
        The Portkey client instance making the request.
    *args :
        Positional arguments passed to the create function.
    create_func : callable
        The create function to call (typically chat.completions.create).
    inference_id : Optional[str], default None
        Optional inference ID for tracking this request.
    **kwargs :
        Additional keyword arguments forwarded to create_func.

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    # Portkey is OpenAI-compatible; request and chunks follow OpenAI spec
    # create_func is a bound method; do not pass client again
    chunks = create_func(*args, **kwargs)
    return stream_chunks(
        chunks=chunks,
        kwargs=kwargs,
        client=client,
        inference_id=inference_id,
    )


def stream_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, Any],
    client: "Portkey",
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    collected_function_call = {"name": "", "arguments": ""}
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = None
    latency = None
    model_name = kwargs.get("model", "unknown")
    provider = "unknown"
    latest_usage_data = {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None}
    latest_chunk_metadata: Dict[str, Any] = {}

    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(chunk.model_dump() if hasattr(chunk, "model_dump") else str(chunk))

            if i == 0:
                first_token_time = time.time()
                # Try to detect provider at first chunk
                provider = detect_provider(chunk, client, model_name)
            if i > 0:
                num_of_completion_tokens = i + 1

            # Extract usage from chunk if available
            chunk_usage = extract_usage(chunk)
            if any(v is not None for v in chunk_usage.values()):
                latest_usage_data = chunk_usage

            # Update metadata from latest chunk (headers/etc.)
            chunk_metadata = extract_portkey_unit_metadata(chunk, model_name)
            if chunk_metadata:
                latest_chunk_metadata.update(chunk_metadata)

            # Extract delta from chunk (OpenAI-compatible)
            delta = get_delta_from_chunk(chunk)

            if delta and getattr(delta, "content", None):
                collected_output_data.append(delta.content)
            elif delta and getattr(delta, "function_call", None):
                if delta.function_call.name:
                    collected_function_call["name"] += delta.function_call.name
                if delta.function_call.arguments:
                    collected_function_call["arguments"] += delta.function_call.arguments
            elif delta and getattr(delta, "tool_calls", None):
                tool_call = delta.tool_calls[0]
                if getattr(tool_call.function, "name", None):
                    collected_function_call["name"] += tool_call.function.name
                if getattr(tool_call.function, "arguments", None):
                    collected_function_call["arguments"] += tool_call.function.arguments

            yield chunk
        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to yield Portkey chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [m for m in collected_output_data if m is not None]
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                if collected_function_call["arguments"]:
                    try:
                        collected_function_call["arguments"] = json.loads(collected_function_call["arguments"])
                    except json.JSONDecodeError:
                        pass
                output_data = collected_function_call

            # Calculate usage and cost at end of stream (prioritize actual usage if present)
            completion_tokens_calculated, prompt_tokens_calculated, total_tokens_calculated, cost_calculated = calculate_streaming_usage_and_cost(
                chunks=raw_outputs,
                messages=kwargs.get("messages", []),
                output_content=output_data,
                model_name=model_name,
                latest_usage_data=latest_usage_data,
                latest_chunk_metadata=latest_chunk_metadata,
            )

            usage_data = latest_usage_data if any(v is not None for v in latest_usage_data.values()) else {}
            final_prompt_tokens = prompt_tokens_calculated if prompt_tokens_calculated is not None else usage_data.get("prompt_tokens", 0)
            final_completion_tokens = completion_tokens_calculated if completion_tokens_calculated is not None else usage_data.get("completion_tokens", num_of_completion_tokens)
            final_total_tokens = total_tokens_calculated if total_tokens_calculated is not None else usage_data.get("total_tokens", (final_prompt_tokens or 0) + (final_completion_tokens or 0))
            final_cost = cost_calculated if cost_calculated is not None else latest_chunk_metadata.get("cost", None)

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs.get("messages", [])},
                output=output_data,
                latency=latency,
                tokens=final_total_tokens,
                prompt_tokens=final_prompt_tokens,
                completion_tokens=final_completion_tokens,
                model=model_name,
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                cost=final_cost,
                metadata={
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None),
                    "provider": provider,
                    "portkey_model": model_name,
                    **extract_portkey_metadata(client),
                    **latest_chunk_metadata,
                },
            )
            add_to_trace(**trace_args)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to trace the Portkey streaming completion. %s", e)


def handle_non_streaming_create(
    client: "Portkey",
    *args,
    create_func: callable,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Handles non-streaming chat.completions.create routed via Portkey.

    Parameters
    ----------
    client : Portkey
        The Portkey client instance used for routing the request.
    *args :
        Positional arguments for the create function.
    create_func : callable
        The function used to create the chat completion. This is a bound method, so do not pass client again.
    inference_id : Optional[str], optional
        A unique identifier for the inference or trace, by default None.
    **kwargs :
        Additional keyword arguments passed to the create function (e.g., "messages", "model", etc.).

    Returns
    -------
    Any
        The completion response as returned by the create function.
    """
    start_time = time.time()
    # create_func is a bound method; do not pass client again
    response = create_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_non_streaming_output_data(response)

        # Usage (if provided by upstream provider via Portkey)
        usage_data = extract_usage(response)
        model_name = getattr(response, "model", kwargs.get("model", "unknown"))
        provider = detect_provider(response, client, model_name)
        extra_metadata = extract_portkey_unit_metadata(response, model_name)
        cost = extra_metadata.get("cost", None)

        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": kwargs.get("messages", [])},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=usage_data.get("total_tokens"),
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            model=model_name,
            model_parameters=get_model_parameters(kwargs),
            raw_output=response.model_dump() if hasattr(response, "model_dump") else str(response),
            id=inference_id,
            cost=cost,
            metadata={
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "provider": provider,
                "portkey_model": model_name,
                **extract_portkey_metadata(client),
                **extra_metadata,
            },
        )
        add_to_trace(**trace_args)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the Portkey non-streaming completion. %s", e)

    return response


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs (OpenAI-compatible)."""
    return {
        "temperature": kwargs.get("temperature", 1),
        "top_p": kwargs.get("top_p", 1),
        "max_tokens": kwargs.get("max_tokens", None),
        "n": kwargs.get("n", 1),
        "stream": kwargs.get("stream", False),
        "stop": kwargs.get("stop", None),
        "presence_penalty": kwargs.get("presence_penalty", 0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0),
        "logit_bias": kwargs.get("logit_bias", None),
        "logprobs": kwargs.get("logprobs", False),
        "top_logprobs": kwargs.get("top_logprobs", None),
        "parallel_tool_calls": kwargs.get("parallel_tool_calls", True),
        "seed": kwargs.get("seed", None),
        "response_format": kwargs.get("response_format", None),
        "timeout": kwargs.get("timeout", None),
        "api_base": kwargs.get("api_base", None),
        "api_version": kwargs.get("api_version", None),
    }


def create_trace_args(
    end_time: float,
    inputs: Dict[str, Any],
    output: Union[str, Dict[str, Any], None],
    latency: float,
    tokens: Optional[int],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    model: str,
    model_parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raw_output: Optional[Union[str, Dict[str, Any]]] = None,
    id: Optional[str] = None,
    cost: Optional[float] = None,
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
    if cost is not None:
        trace_args["cost"] = cost
    return trace_args


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    provider = kwargs.get("metadata", {}).get("provider", "Portkey")
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Portkey Chat Completion", provider=provider)


def parse_non_streaming_output_data(response: Any) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion (OpenAI-compatible)."""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                return None
            content = getattr(message, "content", None)
            function_call = getattr(message, "function_call", None)
            tool_calls = getattr(message, "tool_calls", None)
            if content:
                return content.strip()
            if function_call:
                return {
                    "name": function_call.name,
                    "arguments": json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments,
                }
            if tool_calls:
                return {
                    "name": tool_calls[0].function.name,
                    "arguments": json.loads(tool_calls[0].function.arguments) if isinstance(tool_calls[0].function.arguments, str) else tool_calls[0].function.arguments,
                }
    except Exception as e:
        logger.debug("Error parsing Portkey output data: %s", e)
    return None


def extract_portkey_metadata(client: "Portkey") -> Dict[str, Any]:
    """Extract Portkey-specific metadata from a Portkey client.

    Attempts to read base URL and redact x-portkey-* headers if present.
    Works defensively across SDK versions.
    """
    metadata: Dict[str, Any] = {"isPortkey": True}
    # Base URL or host
    for attr in ("base_url", "baseURL", "host", "custom_host"):
        try:
            val = getattr(client, attr, None)
            if val:
                metadata["portkeyBaseUrl"] = str(val)
                break
        except Exception:
            continue

    # Headers
    possible_header_attrs = ("default_headers", "headers", "_default_headers", "_headers", "custom_headers", "allHeaders")
    redacted: Dict[str, Any] = {}
    for attr in possible_header_attrs:
        try:
            headers = getattr(client, attr, None)
            if _is_dict_like(headers):
                for k, v in headers.items():
                    if isinstance(k, str) and k.lower().startswith("x-portkey-"):
                        if k.lower() in {"x-portkey-api-key", "x-portkey-virtual-key"}:
                            redacted[k] = "***"
                        else:
                            redacted[k] = v
        except Exception:
            continue
    if redacted:
        metadata["portkeyHeaders"] = redacted
    else:
        logger.debug(
            "Openlayer Portkey tracer: No x-portkey-* headers detected on client; provider/config metadata may be limited."
        )
    return metadata


def extract_portkey_unit_metadata(unit: Any, model_name: str) -> Dict[str, Any]:
    """Extract metadata from a response or chunk unit (headers, ids)."""
    metadata: Dict[str, Any] = {}
    try:
        # Extract system fingerprint if available (OpenAI-compatible)
        if hasattr(unit, "system_fingerprint"):
            metadata["system_fingerprint"] = unit.system_fingerprint
        if hasattr(unit, "service_tier"):
            metadata["service_tier"] = unit.service_tier
        
        # Response headers may be present on the object
        headers_obj = None
        if hasattr(unit, "_response_headers"):
            headers_obj = getattr(unit, "_response_headers")
        elif hasattr(unit, "response_headers"):
            headers_obj = getattr(unit, "response_headers")
        elif hasattr(unit, "_headers"):
            headers_obj = getattr(unit, "_headers")

        if _is_dict_like(headers_obj):
            headers = {str(k): v for k, v in headers_obj.items()}
            metadata["response_headers"] = headers
            # Known Portkey header hints (names are lower-cased defensively)
            lower = {k.lower(): v for k, v in headers.items()}
            if "x-portkey-trace-id" in lower:
                metadata["portkey_trace_id"] = lower["x-portkey-trace-id"]
            if "x-portkey-cache-status" in lower:
                metadata["portkey_cache_status"] = lower["x-portkey-cache-status"]
            if "x-portkey-retry-attempt-count" in lower:
                metadata["portkey_retry_attempt_count"] = lower["x-portkey-retry-attempt-count"]
            if "x-portkey-last-used-option-index" in lower:
                metadata["portkey_last_used_option_index"] = lower["x-portkey-last-used-option-index"]
    except Exception:
        pass
    # Attach model for convenience
    if model_name:
        metadata["portkey_model"] = model_name
    return metadata


def extract_usage(obj: Any) -> Dict[str, Optional[int]]:
    """Extract usage from a response or chunk object.
    
    This function attempts to extract token usage information from various
    locations where it might be stored, including:
    - Direct `usage` attribute
    - `model_dump()` dictionary (for streaming chunks)
    
    Parameters
    ----------
    obj : Any
        The response or chunk object to extract usage from.
    
    Returns
    -------
    Dict[str, Optional[int]]
        Dictionary with keys: total_tokens, prompt_tokens, completion_tokens.
        Values are None if usage information is not found.
    """
    try:
        # Check for direct usage attribute (works for both response and chunk)
        if hasattr(obj, "usage") and obj.usage is not None:
            usage = obj.usage
            return {
                "total_tokens": getattr(usage, "total_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        
        # Check if object model dump has usage (primarily for streaming chunks)
        if hasattr(obj, "model_dump"):
            obj_dict = obj.model_dump()
            if _supports_membership_check(obj_dict) and "usage" in obj_dict and obj_dict["usage"]:
                usage = obj_dict["usage"]
                return {
                    "total_tokens": usage.get("total_tokens", None),
                    "prompt_tokens": usage.get("prompt_tokens", None),
                    "completion_tokens": usage.get("completion_tokens", None),
                }
    except Exception:
        pass
    return {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None}


def calculate_streaming_usage_and_cost(
    chunks: Any,
    messages: Any,
    output_content: Any,
    model_name: str,
    latest_usage_data: Dict[str, Optional[int]],
    latest_chunk_metadata: Dict[str, Any],
):
    """Calculate usage and cost at the end of streaming."""
    try:
        # Priority 1: Actual usage provided in chunks
        if latest_usage_data and latest_usage_data.get("total_tokens") and latest_usage_data.get("total_tokens") > 0:
            return (
                latest_usage_data.get("completion_tokens"),
                latest_usage_data.get("prompt_tokens"),
                latest_usage_data.get("total_tokens"),
                latest_chunk_metadata.get("cost"),
            )

        # Priority 2: Look for usage embedded in final chunk dicts (if raw dicts)
        if isinstance(chunks, list):
            for chunk_data in reversed(chunks):
                if _supports_membership_check(chunk_data) and "usage" in chunk_data and chunk_data["usage"]:
                    usage = chunk_data["usage"]
                    if usage.get("total_tokens", 0) > 0:
                        return (
                            usage.get("completion_tokens"),
                            usage.get("prompt_tokens"),
                            usage.get("total_tokens"),
                            latest_chunk_metadata.get("cost"),
                        )

        # Priority 3: Estimate tokens
        completion_tokens = None
        prompt_tokens = None
        total_tokens = None
        cost = None

        # Estimate completion tokens
        if isinstance(output_content, str):
            completion_tokens = max(1, len(output_content) // 4)
        elif _is_dict_like(output_content):
            json_str = json.dumps(output_content) if output_content else "{}"
            completion_tokens = max(1, len(json_str) // 4)
        else:
            # Fallback: count chunks present
            try:
                completion_tokens = len([c for c in chunks if c])
            except Exception:
                completion_tokens = None

        # Estimate prompt tokens from messages
        if messages:
            total_chars = 0
            try:
                for message in messages:
                    if _supports_membership_check(message) and "content" in message:
                        total_chars += len(str(message["content"]))
            except Exception:
                total_chars = 0
            prompt_tokens = max(1, total_chars // 4) if total_chars > 0 else 0
        else:
            prompt_tokens = 0

        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        # Cost from metadata if present; otherwise simple heuristic for some models
        cost = latest_chunk_metadata.get("cost")
        if cost is None and total_tokens and model_name:
            ml = model_name.lower()
            if "gpt-3.5-turbo" in ml:
                cost = (prompt_tokens * 0.0005 / 1000.0) + (completion_tokens * 0.0015 / 1000.0)

        return completion_tokens, prompt_tokens, total_tokens, cost
    except Exception as e:
        logger.error("Error calculating streaming usage: %s", e)
        return None, None, None, None


def _extract_provider_from_object(obj: Any) -> Optional[str]:
    """Extract provider from a response or chunk object.
    
    Checks response_metadata for provider information.
    Returns None if no provider is found.
    """
    try:
        # Check response_metadata
        if hasattr(obj, "response_metadata") and _is_dict_like(obj.response_metadata):
            if "provider" in obj.response_metadata:
                return obj.response_metadata["provider"]
    except Exception:
        pass
    return None


def detect_provider(obj: Any, client: "Portkey", model_name: str) -> str:
    """Detect provider from a response or chunk object.
    
    Parameters
    ----------
    obj : Any
        The response or chunk object to extract provider information from.
    client : Portkey
        The Portkey client instance.
    model_name : str
        The model name to use as a fallback for provider detection.
    
    Returns
    -------
    str
        The detected provider name.
    """
    # First: check Portkey headers on the client (authoritative)
    provider = _provider_from_portkey_headers(client)
    if provider:
        return provider
    # Next: check object metadata if any
    provider = _extract_provider_from_object(obj)
    if provider:
        return provider
    # Fallback to model name heuristics
    return detect_provider_from_model_name(model_name)


def detect_provider_from_model_name(model_name: str) -> str:
    """Detect provider from model name patterns."""
    model_lower = (model_name or "").lower()
    if model_lower.startswith(("gpt-", "o1-", "text-davinci", "text-curie", "text-babbage", "text-ada")):
        return "OpenAI"
    if model_lower.startswith(("claude-", "claude")):
        return "Anthropic"
    if "gemini" in model_lower or "palm" in model_lower:
        return "Google"
    if "llama" in model_lower or "meta-" in model_lower:
        return "Meta"
    if model_lower.startswith("mistral") or "mixtral" in model_lower:
        return "Mistral"
    if model_lower.startswith("command"):
        return "Cohere"
    return "Portkey"


def get_delta_from_chunk(chunk: Any) -> Any:
    """Extract delta from chunk, handling different response formats."""
    try:
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta"):
                return choice.delta
    except Exception:
        pass
    return None


def _provider_from_portkey_headers(client: "Portkey") -> Optional[str]:
    """Get provider from Portkey headers on the client."""
    header_sources = ("default_headers", "headers", "_default_headers", "_headers")
    for attr in header_sources:
        try:
            headers = getattr(client, attr, None)
            if _is_dict_like(headers):
                for k, v in headers.items():
                    if isinstance(k, str) and k.lower() == "x-portkey-provider" and v:
                        return str(v)
        except Exception:
            continue
    return None


def _is_dict_like(obj: Any) -> bool:
    """Check if an object is dict-like (has .items() method).
    
    This is more robust than isinstance(obj, dict) as it handles
    custom dict-like objects (e.g., CaseInsensitiveDict, custom headers).
    """
    return hasattr(obj, "items") and callable(getattr(obj, "items", None))


def _supports_membership_check(obj: Any) -> bool:
    """Check if an object supports membership testing (e.g., 'key in obj').
    
    This checks for __contains__ method or if it's dict-like.
    """
    return hasattr(obj, "__contains__") or _is_dict_like(obj)
