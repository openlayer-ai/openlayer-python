"""Module with methods used to trace Google Gemini LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union, TYPE_CHECKING

try:
    import google.generativeai as genai

    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False

if TYPE_CHECKING:
    import google.generativeai as genai

from ..tracing import tracer

logger = logging.getLogger(__name__)


def _clean_model_name(model_name: str) -> str:
    """Remove 'models/' prefix from Gemini model names.
    
    Parameters
    ----------
    model_name : str
        The raw model name from the client (e.g., "models/gemini-pro").
    
    Returns
    -------
    str
        The cleaned model name (e.g., "gemini-pro").
    """
    if model_name and model_name.startswith("models/"):
        return model_name[7:]  # Remove "models/" prefix (7 characters)
    return model_name


def trace_gemini(
    client: "genai.GenerativeModel",
) -> "genai.GenerativeModel":
    """Patch the Google Gemini client to trace chat completions.

    The following information is collected for each chat completion:
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
    - metadata: Additional metadata about the completion. For example, the time it
    took to generate the first token, when streaming.

    Parameters
    ----------
    client : genai.GenerativeModel
        The Gemini GenerativeModel to patch.

    Returns
    -------
    genai.GenerativeModel
        The patched Gemini client.
    """
    if not HAVE_GEMINI:
        raise ImportError(
            "Google Generative AI library is not installed. Please install it with: pip install google-generativeai"
        )

    # Store original methods
    original_generate_content = client.generate_content
    original_generate_content_async = client.generate_content_async

    @wraps(original_generate_content)
    def traced_generate_content(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)
        clean_model_name = _clean_model_name(client.model_name)

        if stream:
            return handle_streaming_generate(
                original_generate_content,
                *args,
                inference_id=inference_id,
                model_name=clean_model_name,
                **kwargs,
            )
        return handle_non_streaming_generate(
            original_generate_content,
            *args,
            inference_id=inference_id,
            model_name=clean_model_name,
            **kwargs,
        )

    @wraps(original_generate_content_async)
    async def traced_generate_content_async(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)
        clean_model_name = _clean_model_name(client.model_name)

        if stream:
            return handle_streaming_generate_async(
                original_generate_content_async,
                *args,
                inference_id=inference_id,
                model_name=clean_model_name,
                **kwargs,
            )
        return await handle_non_streaming_generate_async(
            original_generate_content_async,
            *args,
            inference_id=inference_id,
            model_name=clean_model_name,
            **kwargs,
        )

    # Patch the methods
    client.generate_content = traced_generate_content
    client.generate_content_async = traced_generate_content_async

    return client


def handle_streaming_generate(
    generate_func: callable,
    *args,
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    **kwargs,
) -> Iterator[Any]:
    """Handles the generate_content method when streaming is enabled.

    Parameters
    ----------
    generate_func : callable
        The generate_content method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None
    model_name : str
        The model name from the client

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    chunks = generate_func(*args, **kwargs)
    return stream_chunks(
        chunks=chunks,
        kwargs=kwargs,
        inference_id=inference_id,
        model_name=model_name,
        contents=args[0] if args else kwargs.get("contents"),
    )


def stream_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, any],
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    contents: Any = None,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = 0
    num_of_prompt_tokens = 0
    latency = None

    try:
        i = 0
        for i, chunk in enumerate(chunks):
            # Store raw output
            try:
                raw_outputs.append(_serialize_chunk(chunk))
            except Exception as e:
                logger.debug("Failed to serialize chunk: %s", e)

            if i == 0:
                first_token_time = time.time()

            # Extract text content from chunk
            if hasattr(chunk, "text"):
                collected_output_data.append(chunk.text)

            # Extract token counts if available
            if hasattr(chunk, "usage_metadata"):
                usage = chunk.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    num_of_prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    num_of_completion_tokens = usage.candidates_token_count

            yield chunk

        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [message for message in collected_output_data if message is not None]
            output_data = "".join(collected_output_data) if collected_output_data else ""

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": _format_input_messages(contents)},
                output=output_data,
                latency=latency,
                tokens=num_of_prompt_tokens + num_of_completion_tokens,
                prompt_tokens=num_of_prompt_tokens,
                completion_tokens=num_of_completion_tokens,
                model=model_name,
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={"timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)},
            )
            add_to_trace(**trace_args)

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the generate content request with Openlayer. %s",
                e,
            )


async def handle_streaming_generate_async(
    generate_func: callable,
    *args,
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    **kwargs,
):
    """Handles the async generate_content method when streaming is enabled.

    Parameters
    ----------
    generate_func : callable
        The async generate_content method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None
    model_name : str
        The model name from the client

    Returns
    -------
    AsyncIterator[Any]
        An async generator that yields the chunks of the completion.
    """
    chunks = await generate_func(*args, **kwargs)
    return stream_chunks_async(
        chunks=chunks,
        kwargs=kwargs,
        inference_id=inference_id,
        model_name=model_name,
        contents=args[0] if args else kwargs.get("contents"),
    )


async def stream_chunks_async(
    chunks,
    kwargs: Dict[str, any],
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    contents: Any = None,
):
    """Streams the chunks of the async completion and traces the completion."""
    collected_output_data = []
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = 0
    num_of_prompt_tokens = 0
    latency = None

    try:
        i = 0
        async for i, chunk in enumerate(chunks):
            # Store raw output
            try:
                raw_outputs.append(_serialize_chunk(chunk))
            except Exception as e:
                logger.debug("Failed to serialize chunk: %s", e)

            if i == 0:
                first_token_time = time.time()

            # Extract text content from chunk
            if hasattr(chunk, "text"):
                collected_output_data.append(chunk.text)

            # Extract token counts if available
            if hasattr(chunk, "usage_metadata"):
                usage = chunk.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    num_of_prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    num_of_completion_tokens = usage.candidates_token_count

            yield chunk

        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [message for message in collected_output_data if message is not None]
            output_data = "".join(collected_output_data) if collected_output_data else ""

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": _format_input_messages(contents)},
                output=output_data,
                latency=latency,
                tokens=num_of_prompt_tokens + num_of_completion_tokens,
                prompt_tokens=num_of_prompt_tokens,
                completion_tokens=num_of_completion_tokens,
                model=model_name,
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={"timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)},
            )
            add_to_trace(**trace_args)

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the async generate content request with Openlayer. %s",
                e,
            )


def handle_non_streaming_generate(
    generate_func: callable,
    *args,
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    **kwargs,
) -> Any:
    """Handles the generate_content method when streaming is disabled.

    Parameters
    ----------
    generate_func : callable
        The generate_content method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None
    model_name : str
        The model name from the client

    Returns
    -------
    Any
        The generation response.
    """
    start_time = time.time()
    response = generate_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_non_streaming_output_data(response)

        # Extract token counts
        num_of_prompt_tokens = 0
        num_of_completion_tokens = 0
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                num_of_prompt_tokens = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                num_of_completion_tokens = usage.candidates_token_count

        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": _format_input_messages(args[0] if args else kwargs.get("contents"))},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=num_of_prompt_tokens + num_of_completion_tokens,
            prompt_tokens=num_of_prompt_tokens,
            completion_tokens=num_of_completion_tokens,
            model=model_name,
            model_parameters=get_model_parameters(kwargs),
            raw_output=_serialize_response(response),
            id=inference_id,
        )

        add_to_trace(**trace_args)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the generate content request with Openlayer. %s", e)

    return response


async def handle_non_streaming_generate_async(
    generate_func: callable,
    *args,
    inference_id: Optional[str] = None,
    model_name: str = "gemini",
    **kwargs,
) -> Any:
    """Handles the async generate_content method when streaming is disabled.

    Parameters
    ----------
    generate_func : callable
        The async generate_content method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None
    model_name : str
        The model name from the client

    Returns
    -------
    Any
        The generation response.
    """
    start_time = time.time()
    response = await generate_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_non_streaming_output_data(response)

        # Extract token counts
        num_of_prompt_tokens = 0
        num_of_completion_tokens = 0
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                num_of_prompt_tokens = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                num_of_completion_tokens = usage.candidates_token_count

        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": _format_input_messages(args[0] if args else kwargs.get("contents"))},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=num_of_prompt_tokens + num_of_completion_tokens,
            prompt_tokens=num_of_prompt_tokens,
            completion_tokens=num_of_completion_tokens,
            model=model_name,
            model_parameters=get_model_parameters(kwargs),
            raw_output=_serialize_response(response),
            id=inference_id,
        )

        add_to_trace(**trace_args)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the async generate content request with Openlayer. %s", e)

    return response


def parse_non_streaming_output_data(response: Any) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Parameters
    ----------
    response : Any
        The generation response.

    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed output data.
    """
    try:
        # Try to get text from the response
        if hasattr(response, "text"):
            return response.text.strip()

        # Try to get from candidates
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content"):
                content = candidate.content
                if hasattr(content, "parts") and content.parts:
                    # Combine all text parts
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                        elif hasattr(part, "function_call"):
                            # Handle function calls
                            func_call = part.function_call
                            return {
                                "name": func_call.name if hasattr(func_call, "name") else "",
                                "arguments": dict(func_call.args) if hasattr(func_call, "args") else {},
                            }
                    if text_parts:
                        return " ".join(text_parts).strip()
    except Exception as e:
        logger.debug("Could not parse Gemini output data: %s", e)

    return None


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs."""
    # Extract generation_config if present
    generation_config = kwargs.get("generation_config", {})

    if hasattr(generation_config, "__dict__"):
        generation_config = generation_config.__dict__

    return {
        "temperature": generation_config.get("temperature")
        if isinstance(generation_config, dict)
        else kwargs.get("temperature"),
        "top_p": generation_config.get("top_p") if isinstance(generation_config, dict) else kwargs.get("top_p"),
        "top_k": generation_config.get("top_k") if isinstance(generation_config, dict) else kwargs.get("top_k"),
        "max_output_tokens": generation_config.get("max_output_tokens")
        if isinstance(generation_config, dict)
        else kwargs.get("max_output_tokens"),
        "candidate_count": generation_config.get("candidate_count")
        if isinstance(generation_config, dict)
        else kwargs.get("candidate_count"),
        "stop_sequences": generation_config.get("stop_sequences")
        if isinstance(generation_config, dict)
        else kwargs.get("stop_sequences"),
    }


def create_trace_args(
    end_time: float,
    inputs: Dict,
    output: str,
    latency: float,
    tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    model_parameters: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    raw_output: Optional[Any] = None,
    id: Optional[str] = None,
) -> Dict:
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
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Gemini Generation", provider="Google")


def _format_input_messages(contents: Any) -> list:
    """Format input contents into messages array.

    Parameters
    ----------
    contents : Any
        The input contents, can be a string, list of messages, or Content objects.

    Returns
    -------
    list
        A list of message dictionaries with 'role' and 'content' keys.
    """
    if contents is None:
        return []

    # If it's a simple string, wrap it in a user message
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]

    # If it's a list, process each element
    if isinstance(contents, list):
        messages = []
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                # Already in message format
                messages.append(item)
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # Content object
                role = item.role if hasattr(item, "role") else "user"
                text_parts = []
                if hasattr(item, "parts"):
                    for part in item.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                content = " ".join(text_parts) if text_parts else str(item)
                messages.append({"role": role, "content": content})
            else:
                # Try to convert to string
                messages.append({"role": "user", "content": str(item)})
        return messages

    # Try to handle Content objects
    if hasattr(contents, "role") and hasattr(contents, "parts"):
        role = contents.role if hasattr(contents, "role") else "user"
        text_parts = []
        if hasattr(contents, "parts"):
            for part in contents.parts:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
        content = " ".join(text_parts) if text_parts else str(contents)
        return [{"role": role, "content": content}]

    # Fallback: convert to string
    return [{"role": "user", "content": str(contents)}]


def _serialize_chunk(chunk: Any) -> Dict[str, Any]:
    """Serialize a response chunk to a dictionary.

    Parameters
    ----------
    chunk : Any
        The response chunk to serialize.

    Returns
    -------
    Dict[str, Any]
        A dictionary representation of the chunk.
    """
    try:
        if hasattr(chunk, "to_dict"):
            return chunk.to_dict()
        elif hasattr(chunk, "__dict__"):
            return {k: str(v) for k, v in chunk.__dict__.items()}
        else:
            return {"text": str(chunk)}
    except Exception:
        return {"text": str(chunk)}


def _serialize_response(response: Any) -> Dict[str, Any]:
    """Serialize a response to a dictionary.

    Parameters
    ----------
    response : Any
        The response to serialize.

    Returns
    -------
    Dict[str, Any]
        A dictionary representation of the response.
    """
    try:
        if hasattr(response, "to_dict"):
            return response.to_dict()
        elif hasattr(response, "__dict__"):
            return {k: str(v) for k, v in response.__dict__.items()}
        else:
            return {"response": str(response)}
    except Exception:
        return {"response": str(response)}
