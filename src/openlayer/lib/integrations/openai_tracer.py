"""Module with methods used to trace OpenAI / Azure OpenAI LLMs."""

import json
import logging
import mimetypes
import re
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

try:
    import openai

    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

if TYPE_CHECKING:
    import openai

from ..tracing import tracer
from ..tracing.attachments import Attachment
from ..tracing.content import (
    AudioContent,
    ContentItem,
    FileContent,
    ImageContent,
    TextContent,
)

logger = logging.getLogger(__name__)


def trace_openai(
    client: Union["openai.OpenAI", "openai.AzureOpenAI"],
) -> Union["openai.OpenAI", "openai.AzureOpenAI"]:
    """Patch the OpenAI or AzureOpenAI client to trace chat completions and responses.

    This function patches both the Chat Completions API (client.chat.completions.create)
    and the Responses API (client.responses.create) to provide comprehensive tracing
    for both APIs while maintaining backward compatibility.

    The following information is collected for each completion/response:
    - start_time: The time when the completion/response was requested.
    - end_time: The time when the completion/response was received.
    - latency: The time it took to generate the completion/response.
    - tokens: The total number of tokens used to generate the completion/response.
    - prompt_tokens: The number of tokens in the prompt/input.
    - completion_tokens: The number of tokens in the completion/output.
    - model: The model used to generate the completion/response.
    - model_parameters: The parameters used to configure the model.
    - raw_output: The raw output of the model.
    - inputs: The inputs used to generate the completion/response.
    - metadata: Additional metadata about the completion/response. For example, the time it
    took to generate the first token, when streaming.

    Parameters
    ----------
    client : Union[openai.OpenAI, openai.AzureOpenAI]
        The OpenAI client to patch.

    Returns
    -------
    Union[openai.OpenAI, openai.AzureOpenAI]
        The patched OpenAI client.
    """
    if not HAVE_OPENAI:
        raise ImportError(
            "OpenAI library is not installed. Please install it with: pip install openai"
        )

    is_azure_openai = isinstance(client, openai.AzureOpenAI)

    # Patch Chat Completions API
    chat_create_func = client.chat.completions.create

    @wraps(chat_create_func)
    def traced_chat_create_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)

        if stream:
            return handle_streaming_create(
                *args,
                **kwargs,
                create_func=chat_create_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
            )
        return handle_non_streaming_create(
            *args,
            **kwargs,
            create_func=chat_create_func,
            inference_id=inference_id,
            is_azure_openai=is_azure_openai,
        )

    client.chat.completions.create = traced_chat_create_func

    # Patch parse method if it exists
    if hasattr(client.chat.completions, "parse"):
        parse_func = client.chat.completions.parse

        @wraps(parse_func)
        def traced_parse_func(*args, **kwargs):
            inference_id = kwargs.pop("inference_id", None)
            stream = kwargs.get("stream", False)

            if stream:
                return handle_streaming_parse(
                    *args,
                    **kwargs,
                    parse_func=parse_func,
                    inference_id=inference_id,
                    is_azure_openai=is_azure_openai,
                )
            return handle_non_streaming_parse(
                *args,
                **kwargs,
                parse_func=parse_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
            )

        client.chat.completions.parse = traced_parse_func

    # Patch Responses API (if available)
    if hasattr(client, "responses"):
        responses_create_func = client.responses.create

        @wraps(responses_create_func)
        def traced_responses_create_func(*args, **kwargs):
            inference_id = kwargs.pop("inference_id", None)
            stream = kwargs.get("stream", False)

            if stream:
                return handle_responses_streaming_create(
                    *args,
                    **kwargs,
                    create_func=responses_create_func,
                    inference_id=inference_id,
                    is_azure_openai=is_azure_openai,
                )
            return handle_responses_non_streaming_create(
                *args,
                **kwargs,
                create_func=responses_create_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
            )

        client.responses.create = traced_responses_create_func
    else:
        logger.debug("Responses API not available in this OpenAI client version")

    return client


def handle_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the create method when streaming is enabled.

    Parameters
    ----------
    create_func : callable
        The create method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    chunks = create_func(*args, **kwargs)
    return stream_chunks(
        chunks=chunks,
        kwargs=kwargs,
        inference_id=inference_id,
        is_azure_openai=is_azure_openai,
    )


def stream_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, any],
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    collected_function_call = {
        "name": "",
        "arguments": "",
    }
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = None
    latency = None
    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(chunk.model_dump())
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1

            # Skip chunks with empty choices (e.g., Azure OpenAI heartbeat chunks)
            if not chunk.choices:
                yield chunk
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                collected_output_data.append(delta.content)
            elif delta.function_call:
                if delta.function_call.name:
                    collected_function_call["name"] += delta.function_call.name
                if delta.function_call.arguments:
                    collected_function_call[
                        "arguments"
                    ] += delta.function_call.arguments
            elif delta.tool_calls:
                if delta.tool_calls[0].function.name:
                    collected_function_call["name"] += delta.tool_calls[0].function.name
                if delta.tool_calls[0].function.arguments:
                    collected_function_call["arguments"] += delta.tool_calls[
                        0
                    ].function.arguments

            yield chunk
        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [
                message for message in collected_output_data if message is not None
            ]
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                if collected_function_call["arguments"]:
                    try:
                        collected_function_call["arguments"] = json.loads(
                            collected_function_call["arguments"]
                        )
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass
                output_data = collected_function_call

            processed_messages = extract_chat_completion_messages(kwargs["messages"])
            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": processed_messages},
                output=output_data,
                latency=latency,
                tokens=num_of_completion_tokens,
                prompt_tokens=0,
                completion_tokens=num_of_completion_tokens,
                model=kwargs.get("model"),
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={
                    "timeToFirstToken": (
                        (first_token_time - start_time) * 1000
                        if first_token_time
                        else None
                    )
                },
            )
            add_to_trace(
                **trace_args,
                is_azure_openai=is_azure_openai,
            )

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the create chat completion request with Openlayer. %s",
                e,
            )


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs."""
    return {
        "frequency_penalty": kwargs.get("frequency_penalty", 0),
        "logit_bias": kwargs.get("logit_bias", None),
        "logprobs": kwargs.get("logprobs", False),
        "top_logprobs": kwargs.get("top_logprobs", None),
        "max_tokens": kwargs.get("max_tokens", None),
        "n": kwargs.get("n", 1),
        "presence_penalty": kwargs.get("presence_penalty", 0),
        "seed": kwargs.get("seed", None),
        "stop": kwargs.get("stop", None),
        "temperature": kwargs.get("temperature", 1),
        "top_p": kwargs.get("top_p", 1),
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
    raw_output: Optional[str] = None,
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


def add_to_trace(
    is_azure_openai: bool = False, api_type: str = "chat_completions", **kwargs
) -> None:
    """Add a chat completion or responses step to the trace."""
    # Remove api_type from kwargs to avoid passing it to the tracer
    kwargs.pop("api_type", None)

    if api_type == "responses":
        # Handle Responses API tracing
        if is_azure_openai:
            tracer.add_chat_completion_step_to_trace(
                **kwargs, name="Azure OpenAI Response", provider="Azure"
            )
        else:
            tracer.add_chat_completion_step_to_trace(
                **kwargs, name="OpenAI Response", provider="OpenAI"
            )
    else:
        # Handle Chat Completions API tracing (default behavior)
        if is_azure_openai:
            tracer.add_chat_completion_step_to_trace(
                **kwargs, name="Azure OpenAI Chat Completion", provider="Azure"
            )
        else:
            tracer.add_chat_completion_step_to_trace(
                **kwargs, name="OpenAI Chat Completion", provider="OpenAI"
            )


def handle_non_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Union["openai.types.chat.chat_completion.ChatCompletion", Any]:
    """Handles the create method when streaming is disabled.

    Parameters
    ----------
    create_func : callable
        The create method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    openai.types.chat.chat_completion.ChatCompletion
        The chat completion response.
    """
    start_time = time.time()
    response = create_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_non_streaming_output_data(response)
        processed_messages = extract_chat_completion_messages(kwargs["messages"])

        # Check if response contains audio (to sanitize raw_output)
        has_audio = bool(getattr(response.choices[0].message, "audio", None))

        # Sanitize raw_output to remove heavy base64 data already uploaded as attachments
        raw_output = response.model_dump()
        if has_audio:
            raw_output = _sanitize_raw_output(raw_output, has_audio=True)

        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": processed_messages},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=response.model,
            model_parameters=get_model_parameters(kwargs),
            raw_output=raw_output,
            id=inference_id,
        )

        add_to_trace(
            is_azure_openai=is_azure_openai,
            **trace_args,
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error(
            "Failed to trace the create chat completion request with Openlayer. %s", e
        )

    return response


def extract_chat_completion_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract and normalize messages from Chat Completions API format.

    Converts OpenAI message format to Openlayer format, extracting media content
    (images, audio, files) into Attachment objects wrapped in Content classes.

    OpenAI content types supported:
    - "text" -> TextContent
    - "image_url" -> ImageContent (URL or base64 data URL)
    - "input_audio" -> AudioContent (base64 encoded)
    - "file" -> FileContent

    Args:
        messages: List of messages in OpenAI Chat Completions format

    Returns:
        List of messages with content normalized to Openlayer format.
        String content is preserved as-is for backwards compatibility.
        List content is converted to ContentItem objects.
    """
    processed_messages: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        name = message.get("name")  # For function/tool messages

        # If content is a string, keep it as-is (backwards compatible)
        if isinstance(content, str):
            processed_msg: Dict[str, Any] = {"role": role, "content": content}
            if name:
                processed_msg["name"] = name
            processed_messages.append(processed_msg)
            continue

        # If content is None (e.g., assistant message with tool_calls only)
        if content is None:
            processed_messages.append(message)
            continue

        # If content is a list, process each item
        if isinstance(content, list):
            normalized_content: List[ContentItem] = [
                _normalize_content_item(item) for item in content
            ]
            processed_msg = {"role": role, "content": normalized_content}
            if name:
                processed_msg["name"] = name
            processed_messages.append(processed_msg)
        else:
            # Unknown content type, keep as-is
            processed_messages.append(message)

    return processed_messages


def _normalize_content_item(item: Dict[str, Any]) -> ContentItem:
    """Normalize a single content item from OpenAI format to Openlayer format.

    Supports both Chat Completions API and Responses API formats.

    Chat Completions API types:
    - "text" -> TextContent
    - "image_url" -> ImageContent (URL or base64 data URL)
    - "input_audio" -> AudioContent (base64 encoded)
    - "file" -> FileContent

    Responses API types:
    - "input_text" -> TextContent
    - "input_image" -> ImageContent (URL, base64 data URL, or file_id)

    Args:
        item: A content item dict from OpenAI message content array

    Returns:
        A ContentItem object (TextContent, ImageContent, etc.) or the original item
    """
    item_type = item.get("type", "")

    # Text content (both APIs)
    if item_type in ("text", "input_text"):
        text = item.get("text", "")
        return TextContent(text=text)

    # Image content (both APIs - different structures)
    elif item_type in ("image_url", "input_image"):
        return _normalize_image_content(item, item_type)

    # Audio content (Chat Completions only)
    elif item_type == "input_audio":
        audio_data = item.get("input_audio", {})
        data = audio_data.get("data", "")
        audio_format = audio_data.get("format", "wav")

        media_type = f"audio/{audio_format}"
        attachment = Attachment.from_base64(
            data_base64=data,
            name=f"audio.{audio_format}",
            media_type=media_type,
        )
        return AudioContent(attachment=attachment)

    # File content (Chat Completions API)
    elif item_type == "file":
        file_data = item.get("file", {})
        file_id = file_data.get("file_id")
        file_data_content = file_data.get("file_data")
        filename = file_data.get("filename", "file")

        if file_data_content:
            if file_data_content.startswith("data:"):
                # Parse data URL: data:application/pdf;base64,{base64_data}
                attachment = _parse_data_url_to_attachment(
                    file_data_content, default_type="file"
                )
                attachment.name = filename
            else:
                # Raw base64 data
                attachment = Attachment.from_base64(
                    data_base64=file_data_content,
                    name=filename,
                    media_type="application/octet-stream",
                )
        elif file_id:
            # Just reference the file ID (can't download without API call)
            attachment = Attachment(name=filename)
            attachment.metadata["openai_file_id"] = file_id
        else:
            attachment = Attachment(name=filename)

        return FileContent(attachment=attachment)

    # File content (Responses API) - different structure than Chat Completions
    elif item_type == "input_file":
        filename = item.get("filename", "file")
        file_data = item.get("file_data", "")
        file_id = item.get("file_id")

        if file_data and file_data.startswith("data:"):
            # Parse data URL: data:application/pdf;base64,{base64_data}
            attachment = _parse_data_url_to_attachment(file_data, default_type="file")
            attachment.name = filename
        elif file_id:
            # Just reference the file ID
            attachment = Attachment(name=filename)
            attachment.metadata["openai_file_id"] = file_id
        else:
            attachment = Attachment(name=filename)

        return FileContent(attachment=attachment)

    else:
        # Unknown type, return as TextContent with string representation
        # Log as warning so we can learn about new OpenAI content types
        logger.warning("Unknown content item type '%s', preserving as text", item_type)
        return TextContent(text=str(item))


def _normalize_image_content(item: Dict[str, Any], item_type: str) -> ImageContent:
    """Normalize image content from both Chat Completions and Responses API.

    Chat Completions: {"type": "image_url", "image_url": {"url": "..."}}
    Responses API:    {"type": "input_image", "image_url": "..."} or
                      {"type": "input_image", "file_id": "..."}

    Args:
        item: The content item dict
        item_type: The type string ("image_url" or "input_image")

    Returns:
        An ImageContent object with the appropriate attachment
    """
    if item_type == "input_image":
        # Responses API format - flat structure
        image_url = item.get("image_url")
        file_id = item.get("file_id")

        if file_id:
            # File ID reference - just store the reference
            attachment = Attachment(name="image")
            attachment.metadata["openai_file_id"] = file_id
        elif image_url:
            attachment = _create_image_attachment_from_url(image_url)
        else:
            # Fallback - empty attachment
            attachment = Attachment(name="image")
    else:
        # Chat Completions API format - nested structure
        image_url_data = item.get("image_url", {})
        url = image_url_data.get("url", "")
        attachment = _create_image_attachment_from_url(url)

    return ImageContent(attachment=attachment)


def _create_image_attachment_from_url(url: str) -> Attachment:
    """Create an Attachment from a URL (handles both regular URLs and data URLs).

    Args:
        url: The image URL (can be a regular URL or a data: URL with base64 content)

    Returns:
        An Attachment object
    """
    if url.startswith("data:"):
        # Base64 data URL
        return _parse_data_url_to_attachment(url, default_type="image")
    else:
        # External URL - infer media type from URL or default to image/jpeg
        media_type = mimetypes.guess_type(url)[0]
        if media_type is None or not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return Attachment.from_url(url, name="image", media_type=media_type)


def _parse_data_url_to_attachment(
    data_url: str, default_type: str = "image"
) -> Attachment:
    """Parse a data URL (data:image/jpeg;base64,...) into an Attachment.

    Args:
        data_url: The data URL string (e.g., "data:image/png;base64,iVBORw0KGgo...")
        default_type: Default type if parsing fails ("image", "audio", "file")

    Returns:
        An Attachment object
    """
    # Format: data:image/png;base64,iVBORw0KGgo...
    match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
    if match:
        media_type = match.group(1)
        base64_data = match.group(2)

        # Infer extension from media type
        extension = media_type.split("/")[-1]
        extension_map = {"jpeg": "jpg", "x-wav": "wav", "mpeg": "mp3"}
        extension = extension_map.get(extension, extension)

        return Attachment.from_base64(
            data_base64=base64_data,
            name=f"{default_type}.{extension}",
            media_type=media_type,
        )

    # Fallback - couldn't parse, treat as plain base64
    logger.warning("Could not parse data URL format, treating as raw base64")
    return Attachment.from_base64(
        data_base64=data_url,
        name=default_type,
        media_type=f"{default_type}/unknown",
    )


# -------------------------------- Responses API Handlers -------------------------------- #


def handle_responses_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the Responses API create method when streaming is enabled.

    Parameters
    ----------
    create_func : callable
        The Responses API create method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the response stream.
    """
    chunks = create_func(*args, **kwargs)
    return stream_responses_chunks(
        chunks=chunks,
        kwargs=kwargs,
        inference_id=inference_id,
        is_azure_openai=is_azure_openai,
    )


def stream_responses_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, any],
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the Responses API and traces the response."""
    collected_output_data = []
    collected_function_call = {
        "name": "",
        "arguments": "",
    }
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = None
    latency = None

    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(
                chunk.model_dump() if hasattr(chunk, "model_dump") else str(chunk)
            )
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1

            # Handle different types of ResponseStreamEvent
            chunk_data = extract_responses_chunk_data(chunk)

            if chunk_data.get("content"):
                collected_output_data.append(chunk_data["content"])
            elif chunk_data.get("function_call"):
                func_call = chunk_data["function_call"]
                if func_call.get("name"):
                    collected_function_call["name"] += func_call["name"]
                if func_call.get("arguments"):
                    collected_function_call["arguments"] += func_call["arguments"]

            yield chunk

        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [
                message for message in collected_output_data if message is not None
            ]
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                if collected_function_call["arguments"]:
                    try:
                        collected_function_call["arguments"] = json.loads(
                            collected_function_call["arguments"]
                        )
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass
                output_data = collected_function_call

            trace_args = create_trace_args(
                end_time=end_time,
                inputs=extract_responses_inputs(kwargs),
                output=output_data,
                latency=latency,
                tokens=num_of_completion_tokens,
                prompt_tokens=0,
                completion_tokens=num_of_completion_tokens,
                model=kwargs.get("model", "unknown"),
                model_parameters=get_responses_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={
                    "timeToFirstToken": (
                        (first_token_time - start_time) * 1000
                        if first_token_time
                        else None
                    )
                },
            )
            add_to_trace(
                **trace_args,
                is_azure_openai=is_azure_openai,
                api_type="responses",
            )

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the Responses API request with Openlayer. %s",
                e,
            )


def handle_responses_non_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Handles the Responses API create method when streaming is disabled.

    Parameters
    ----------
    create_func : callable
        The Responses API create method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Any
        The response object.
    """
    start_time = time.time()
    response = create_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_responses_output_data(response)
        usage_data = extract_responses_usage(response)

        # Check if response contains generated images (to sanitize raw_output)
        has_generated_images = False
        if hasattr(response, "output") and isinstance(response.output, list):
            has_generated_images = any(
                getattr(item, "type", None) == "image_generation_call"
                for item in response.output
            )

        # Sanitize raw_output to remove heavy base64 data already uploaded as attachments
        if hasattr(response, "model_dump"):
            raw_output = response.model_dump()
            if has_generated_images:
                raw_output = _sanitize_raw_output(raw_output, has_generated_images=True)
        else:
            raw_output = str(response)

        trace_args = create_trace_args(
            end_time=end_time,
            inputs=extract_responses_inputs(kwargs),
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=usage_data.get("total_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            model=getattr(response, "model", kwargs.get("model", "unknown")),
            model_parameters=get_responses_model_parameters(kwargs),
            raw_output=raw_output,
            id=inference_id,
        )

        add_to_trace(
            is_azure_openai=is_azure_openai,
            api_type="responses",
            **trace_args,
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the Responses API request with Openlayer. %s", e)

    return response


# -------------------------------- Responses API Helper Functions -------------------------------- #


def extract_responses_chunk_data(chunk: Any) -> Dict[str, Any]:
    """Extract content and function call data from a ResponseStreamEvent chunk.

    Args:
        chunk: A ResponseStreamEvent object

    Returns:
        Dictionary with content and/or function_call data
    """
    result = {}

    try:
        # Handle different types of response stream events
        chunk_type = getattr(chunk, "type", None)

        if chunk_type == "response.text.delta":
            # Text content delta
            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                result["content"] = chunk.delta.text
        elif chunk_type == "response.function_call.arguments.delta":
            # Function call arguments delta
            if hasattr(chunk, "delta"):
                result["function_call"] = {"arguments": chunk.delta}
        elif chunk_type == "response.function_call.name":
            # Function call name
            if hasattr(chunk, "name"):
                result["function_call"] = {"name": chunk.name}
        elif hasattr(chunk, "choices") and chunk.choices:
            # Fallback to chat-style format if available
            choice = chunk.choices[0]
            if hasattr(choice, "delta"):
                delta = choice.delta
                if hasattr(delta, "content") and delta.content:
                    result["content"] = delta.content
                elif hasattr(delta, "function_call"):
                    func_call = {}
                    if (
                        hasattr(delta.function_call, "name")
                        and delta.function_call.name
                    ):
                        func_call["name"] = delta.function_call.name
                    if (
                        hasattr(delta.function_call, "arguments")
                        and delta.function_call.arguments
                    ):
                        func_call["arguments"] = delta.function_call.arguments
                    if func_call:
                        result["function_call"] = func_call

    except Exception as e:
        logger.debug("Could not extract chunk data from ResponseStreamEvent: %s", e)

    return result


def extract_responses_inputs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inputs from Responses API parameters.

    Formats the input as a messages array similar to Chat Completions API format:
    {"prompt": [{"role": "user", "content": "..."}]}

    Handles multimodal inputs (text + images) by normalizing content items
    using the same ContentItem format as Chat Completions API.

    Responses API input formats supported:
    - Simple string: input="What is 2+2?"
    - List of messages: input=[{"role": "user", "content": "..."}]
    - Multimodal content: input=[{"role": "user", "content": [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "https://..."}
      ]}]

    Args:
        kwargs: The parameters passed to the Responses API

    Returns:
        Dictionary with prompt as a messages array
    """
    messages: List[Dict[str, Any]] = []

    # Handle different input formats for Responses API
    if "input" in kwargs:
        input_value = kwargs["input"]

        if isinstance(input_value, str):
            # Simple string input
            messages.append({"role": "user", "content": input_value})
        elif isinstance(input_value, list):
            # List of messages - process each one for multimodal content
            for msg in input_value:
                if isinstance(msg, dict):
                    processed_msg = _process_responses_message(msg)
                    messages.append(processed_msg)
                else:
                    # Non-dict item, wrap as user message
                    messages.append({"role": "user", "content": str(msg)})

    # Add instructions as system message (if present)
    if "instructions" in kwargs:
        messages.insert(0, {"role": "system", "content": kwargs["instructions"]})

    # Handle legacy/alternative input parameters
    if not messages:
        if "prompt" in kwargs:
            messages.append({"role": "user", "content": kwargs["prompt"]})

    # If no user message was added, create a fallback
    if not any(msg.get("role") == "user" for msg in messages):
        if messages:
            # Only system message, add empty user message
            messages.append({"role": "user", "content": ""})
        else:
            # No messages at all, add placeholder
            messages.append({"role": "user", "content": "No input provided"})

    return {"prompt": messages}


def _process_responses_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single message from Responses API input.

    Normalizes multimodal content items (text, images) to Openlayer format.
    String content is preserved as-is for backwards compatibility.

    Args:
        message: A message dict with 'role' and 'content' keys

    Returns:
        Processed message with normalized content
    """
    role = message.get("role", "user")
    content = message.get("content")

    # String content - keep as-is (backwards compatible)
    if isinstance(content, str):
        return {"role": role, "content": content}

    # List content - normalize each item
    if isinstance(content, list):
        normalized_content: List[ContentItem] = []
        for item in content:
            if isinstance(item, dict):
                content_item = _normalize_content_item(item)
                normalized_content.append(content_item)
            else:
                # Non-dict item, convert to TextContent
                normalized_content.append(TextContent(text=str(item)))
        return {"role": role, "content": normalized_content}

    # None or other type - return as-is
    return message


def parse_responses_output_data(
    response: Any,
) -> Union[str, List[ContentItem], Dict[str, Any], None]:
    """Parses the output data from a Responses API response.

    Handles text and image generation outputs. For multimodal outputs
    (e.g., text + generated images), returns a list of ContentItem objects.

    Args:
        response: The Response object from the Responses API

    Returns:
        The parsed output data:
        - str: For text-only responses (backwards compatible)
        - List[ContentItem]: For multimodal responses (text + images)
        - Dict: For function/tool call responses
        - None: If no output data
    """
    try:
        if not hasattr(response, "output") or not response.output:
            return None

        if not isinstance(response.output, list):
            # Handle non-list output
            if hasattr(response.output, "text"):
                return response.output.text.strip()
            return str(response.output).strip()

        content_items: List[ContentItem] = []
        text_content: Optional[str] = None

        for output_item in response.output:
            output_type = getattr(output_item, "type", None)

            if output_type == "message":
                # Text message output
                extracted_text = _extract_text_from_message_output(output_item)
                if extracted_text:
                    text_content = extracted_text
                    content_items.append(TextContent(text=extracted_text))

            elif output_type == "image_generation_call":
                # Image generation output - result contains base64 image
                image_data = getattr(output_item, "result", None)
                if image_data:
                    attachment = Attachment.from_base64(
                        data_base64=image_data,
                        name="generated_image.png",
                        media_type="image/png",
                    )
                    content_items.append(ImageContent(attachment=attachment))

            else:
                # Unknown output type - log for future support
                logger.debug("Unknown Responses API output type: %s", output_type)

        # Return appropriate format based on content
        if len(content_items) > 1:
            # Multimodal output - return list of ContentItems
            return content_items
        elif len(content_items) == 1:
            if isinstance(content_items[0], TextContent):
                # Text-only output - return string (backwards compatible)
                return text_content
            else:
                # Single non-text content item - return as list
                return content_items

    except Exception as e:
        logger.warning("Could not parse Responses API output data: %s", e)

    return None


def _extract_text_from_message_output(output_item: Any) -> Optional[str]:
    """Extract text content from a message output item.

    Args:
        output_item: A ResponseOutputMessage object

    Returns:
        The extracted text or None
    """
    if hasattr(output_item, "content") and output_item.content:
        if isinstance(output_item.content, list) and output_item.content:
            first_content = output_item.content[0]
            if hasattr(first_content, "text"):
                return first_content.text.strip()
        elif hasattr(output_item.content, "text"):
            return output_item.content.text.strip()
    elif hasattr(output_item, "text"):
        return output_item.text.strip()
    return None


def extract_responses_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from a Responses API response.

    Args:
        response: The Response object from the Responses API

    Returns:
        Dictionary with token usage information
    """
    usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    try:
        if hasattr(response, "usage"):
            usage_obj = response.usage
            # Handle ResponseUsage object with different attribute names
            usage["total_tokens"] = getattr(usage_obj, "total_tokens", 0)
            # ResponseUsage uses 'input_tokens' instead of 'prompt_tokens'
            usage["prompt_tokens"] = getattr(
                usage_obj, "input_tokens", getattr(usage_obj, "prompt_tokens", 0)
            )
            # ResponseUsage uses 'output_tokens' instead of 'completion_tokens'
            usage["completion_tokens"] = getattr(
                usage_obj, "output_tokens", getattr(usage_obj, "completion_tokens", 0)
            )
        elif hasattr(response, "token_usage"):
            # Alternative usage attribute name
            usage_obj = response.token_usage
            usage["total_tokens"] = getattr(usage_obj, "total_tokens", 0)
            usage["prompt_tokens"] = getattr(
                usage_obj, "input_tokens", getattr(usage_obj, "prompt_tokens", 0)
            )
            usage["completion_tokens"] = getattr(
                usage_obj, "output_tokens", getattr(usage_obj, "completion_tokens", 0)
            )
    except Exception as e:
        logger.debug("Could not extract token usage from Responses API response: %s", e)

    return usage


def get_responses_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from Responses API kwargs."""
    return {
        "max_output_tokens": kwargs.get("max_output_tokens"),
        "temperature": kwargs.get("temperature", 1),
        "top_p": kwargs.get("top_p", 1),
        "reasoning": kwargs.get("reasoning"),
        "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
        "max_tool_calls": kwargs.get("max_tool_calls"),
        "background": kwargs.get("background"),
        "truncation": kwargs.get("truncation"),
        "include": kwargs.get("include"),
    }


def _sanitize_raw_output(
    raw_output: Dict[str, Any],
    has_audio: bool = False,
    has_generated_images: bool = False,
) -> Dict[str, Any]:
    """Remove heavy base64 data from raw_output that's already uploaded as attachments.

    This prevents duplicating large binary data in the trace - the data is already
    stored in blob storage via attachments, so we replace it with a placeholder.

    Args:
        raw_output: The raw model output dict (from response.model_dump())
        has_audio: Whether the response contains audio data (Chat Completions API)
        has_generated_images: Whether the response contains generated images (Responses API)

    Returns:
        A sanitized copy of raw_output with heavy data replaced by placeholders
    """
    import copy

    sanitized = copy.deepcopy(raw_output)

    if has_audio:
        _sanitize_audio_data(sanitized)

    if has_generated_images:
        _sanitize_image_data(sanitized)

    return sanitized


def _sanitize_audio_data(sanitized: Dict[str, Any]) -> None:
    """Remove audio base64 data from Chat Completions response.

    Args:
        sanitized: The raw output dict to sanitize (modified in place)
    """
    try:
        for choice in sanitized.get("choices", []):
            message = choice.get("message", {})
            if message and "audio" in message and message["audio"]:
                if "data" in message["audio"]:
                    message["audio"]["data"] = "[UPLOADED_TO_STORAGE]"
    except Exception as e:
        logger.debug("Could not sanitize audio data from raw_output: %s", e)


def _sanitize_image_data(sanitized: Dict[str, Any]) -> None:
    """Remove generated image base64 data from Responses API output.

    Args:
        sanitized: The raw output dict to sanitize (modified in place)
    """
    try:
        for output_item in sanitized.get("output", []):
            if output_item.get("type") == "image_generation_call":
                if "result" in output_item:
                    output_item["result"] = "[UPLOADED_TO_STORAGE]"
    except Exception as e:
        logger.debug("Could not sanitize image data from raw_output: %s", e)


def parse_non_streaming_output_data(
    response: "openai.types.chat.chat_completion.ChatCompletion",
) -> Union[str, List[ContentItem], Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Handles text, audio, and function call outputs. For multimodal outputs
    (e.g., audio responses), returns a list of ContentItem objects.

    Parameters
    ----------
    response : openai.types.chat.chat_completion.ChatCompletion
        The chat completion response.

    Returns
    -------
    Union[str, List[ContentItem], Dict[str, Any], None]
        The parsed output data:
        - str: For text-only responses (backwards compatible)
        - List[ContentItem]: For multimodal responses (text + audio)
        - Dict: For function/tool call responses
        - None: If no output data
    """
    message = response.choices[0].message
    output_content = message.content
    output_audio = getattr(message, "audio", None)
    output_function_call = message.function_call
    output_tool_calls = message.tool_calls

    # Check for audio output (multimodal response)
    if output_audio is not None:
        content_items: List[ContentItem] = []

        # Add text content (transcript) if available
        transcript = getattr(output_audio, "transcript", None)
        if transcript:
            content_items.append(TextContent(text=transcript))

        # Add audio content
        audio_data = getattr(output_audio, "data", None)
        if audio_data:
            # Create attachment from base64 audio data
            attachment = Attachment.from_base64(
                data_base64=audio_data,
                name="output_audio.wav",
                media_type="audio/wav",
            )
            # Store additional audio metadata
            audio_id = getattr(output_audio, "id", None)
            if audio_id:
                attachment.metadata["openai_audio_id"] = audio_id
            expires_at = getattr(output_audio, "expires_at", None)
            if expires_at:
                attachment.metadata["expires_at"] = expires_at

            content_items.append(AudioContent(attachment=attachment))

        if content_items:
            return content_items

    # Text-only response (backwards compatible)
    if output_content:
        return output_content.strip()

    # Function/tool call response
    if output_function_call or output_tool_calls:
        if output_function_call:
            return {
                "name": output_function_call.name,
                "arguments": json.loads(output_function_call.arguments),
            }
        else:
            return {
                "name": output_tool_calls[0].function.name,
                "arguments": json.loads(output_tool_calls[0].function.arguments),
            }

    return None


def handle_streaming_parse(
    parse_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the parse method when streaming is enabled.

    Parameters
    ----------
    parse_func : callable
        The parse method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    chunks = parse_func(*args, **kwargs)
    return stream_parse_chunks(
        chunks=chunks,
        kwargs=kwargs,
        inference_id=inference_id,
        is_azure_openai=is_azure_openai,
    )


def stream_parse_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, any],
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the parse completion and traces the completion."""
    collected_output_data = []
    collected_function_call = {
        "name": "",
        "arguments": "",
    }
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = None
    latency = None
    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(chunk.model_dump())
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1

            # Skip chunks with empty choices (e.g., Azure OpenAI heartbeat chunks)
            if not chunk.choices:
                yield chunk
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                collected_output_data.append(delta.content)
            elif delta.function_call:
                if delta.function_call.name:
                    collected_function_call["name"] += delta.function_call.name
                if delta.function_call.arguments:
                    collected_function_call[
                        "arguments"
                    ] += delta.function_call.arguments
            elif delta.tool_calls:
                if delta.tool_calls[0].function.name:
                    collected_function_call["name"] += delta.tool_calls[0].function.name
                if delta.tool_calls[0].function.arguments:
                    collected_function_call["arguments"] += delta.tool_calls[
                        0
                    ].function.arguments

            yield chunk
        end_time = time.time()
        latency = (end_time - start_time) * 1000
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            collected_output_data = [
                message for message in collected_output_data if message is not None
            ]
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                if collected_function_call["arguments"]:
                    try:
                        collected_function_call["arguments"] = json.loads(
                            collected_function_call["arguments"]
                        )
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass
                output_data = collected_function_call

            processed_messages = extract_chat_completion_messages(kwargs["messages"])
            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": processed_messages},
                output=output_data,
                latency=latency,
                tokens=num_of_completion_tokens,
                prompt_tokens=0,
                completion_tokens=num_of_completion_tokens,
                model=kwargs.get("model"),
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={
                    "timeToFirstToken": (
                        (first_token_time - start_time) * 1000
                        if first_token_time
                        else None
                    ),
                    "method": "parse",
                    "response_format": kwargs.get("response_format"),
                },
            )
            add_to_trace(
                **trace_args,
                is_azure_openai=is_azure_openai,
            )

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the parse chat completion request with Openlayer. %s",
                e,
            )


def handle_non_streaming_parse(
    parse_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Handles the parse method when streaming is disabled.

    Parameters
    ----------
    parse_func : callable
        The parse method to handle.
    is_azure_openai : bool, optional
        Whether the client is an Azure OpenAI client, by default False
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Any
        The parsed completion response.
    """
    start_time = time.time()
    response = parse_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_structured_output_data(response)
        processed_messages = extract_chat_completion_messages(kwargs["messages"])
        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": processed_messages},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=response.model,
            model_parameters=get_model_parameters(kwargs),
            raw_output=response.model_dump(),
            id=inference_id,
            metadata={
                "method": "parse",
                "response_format": kwargs.get("response_format"),
            },
        )

        add_to_trace(
            is_azure_openai=is_azure_openai,
            **trace_args,
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error(
            "Failed to trace the parse chat completion request with Openlayer. %s", e
        )

    return response


def parse_structured_output_data(response: Any) -> Union[str, Dict[str, Any], None]:
    """Parses the structured output data from a parse method completion.

    Parameters
    ----------
    response : Any
        The parse method completion response.

    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed structured output data.
    """
    try:
        # Check if response has parsed structured data
        if hasattr(response, "parsed") and response.parsed is not None:
            # Handle Pydantic models
            if hasattr(response.parsed, "model_dump"):
                return response.parsed.model_dump()
            # Handle dict-like objects
            elif hasattr(response.parsed, "__dict__"):
                return response.parsed.__dict__
            # Handle other structured formats
            else:
                return response.parsed

        # Fallback to regular message content parsing
        return parse_non_streaming_output_data(response)

    except Exception as e:
        logger.error("Failed to parse structured output data: %s", e)
        # Final fallback to regular parsing
        return parse_non_streaming_output_data(response)


# --------------------------- OpenAI Assistants API -------------------------- #
def trace_openai_assistant_thread_run(
    client: "openai.OpenAI", run: "openai.types.beta.threads.run.Run"
) -> None:
    """Trace a run from an OpenAI assistant.

    Once the run is completed, the thread data is published to Openlayer,
    along with the latency, and number of tokens used."""
    if not HAVE_OPENAI:
        raise ImportError(
            "OpenAI library is not installed. Please install it with: pip install openai"
        )

    _type_check_run(run)

    # Do nothing if the run is not completed
    if run.status != "completed":
        return

    try:
        # Extract vars
        run_step_vars = _extract_run_vars(run)
        metadata = _extract_run_metadata(run)

        # Convert thread to prompt
        messages = client.beta.threads.messages.list(
            thread_id=run.thread_id, order="asc"
        )
        prompt = _thread_messages_to_prompt(messages)

        # Add step to the trace
        tracer.add_chat_completion_step_to_trace(
            inputs={"prompt": prompt[:-1]},  # Remove the last message (the output)
            output=prompt[-1]["content"],
            **run_step_vars,
            metadata=metadata,
            provider="OpenAI",
            name="OpenAI Assistant Run",
        )

    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to monitor run. %s", e)


def _type_check_run(run: "openai.types.beta.threads.run.Run") -> None:
    """Validate the run object."""
    if HAVE_OPENAI and not isinstance(run, openai.types.beta.threads.run.Run):
        raise ValueError(f"Expected a Run object, but got {type(run)}.")


def _extract_run_vars(run: "openai.types.beta.threads.run.Run") -> Dict[str, any]:
    """Extract the variables from the run object."""
    return {
        "start_time": run.created_at,
        "end_time": run.completed_at,
        "latency": (run.completed_at - run.created_at) * 1000,  # Convert to ms
        "prompt_tokens": run.usage.prompt_tokens,
        "completion_tokens": run.usage.completion_tokens,
        "tokens": run.usage.total_tokens,
        "model": run.model,
    }


def _extract_run_metadata(run: "openai.types.beta.threads.run.Run") -> Dict[str, any]:
    """Extract the metadata from the run object."""
    return {
        "openaiThreadId": run.thread_id,
        "openaiAssistantId": run.assistant_id,
    }


def _thread_messages_to_prompt(
    messages: List["openai.types.beta.threads.thread_message.ThreadMessage"],
) -> List[Dict[str, str]]:
    """Given list of ThreadMessage, return its contents in the `prompt` format,
    i.e., a list of dicts with 'role' and 'content' keys."""
    prompt = []
    for message in list(messages):
        role = message.role
        contents = message.content

        for content in contents:
            content_type = content.type
            if content_type == "text":
                text_content = content.text.value
            if content_type == "image_file":
                text_content = content.image_file.file_id

            prompt.append(
                {
                    "role": role,
                    "content": text_content,
                }
            )
    return prompt
