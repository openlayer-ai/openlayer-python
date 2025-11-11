"""Module with methods used to trace async OpenAI/Azure OpenAI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, AsyncIterator, Optional, Union, TYPE_CHECKING

try:
    import openai

    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

if TYPE_CHECKING:
    import openai

from .openai_tracer import (
    get_model_parameters,
    create_trace_args,
    add_to_trace,
    parse_non_streaming_output_data,
    parse_structured_output_data,
    # Import Responses API helper functions
    extract_responses_chunk_data,
    extract_responses_inputs,
    parse_responses_output_data,
    extract_responses_usage,
    get_responses_model_parameters,
)

logger = logging.getLogger(__name__)


def trace_async_openai(
    client: Union["openai.AsyncOpenAI", "openai.AsyncAzureOpenAI"],
) -> Union["openai.AsyncOpenAI", "openai.AsyncAzureOpenAI"]:
    """Patch the AsyncOpenAI or AsyncAzureOpenAI client to trace chat completions and responses.

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
    client : Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]
        The AsyncOpenAI client to patch.

    Returns
    -------
    Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]
        The patched AsyncOpenAI client.
    """
    if not HAVE_OPENAI:
        raise ImportError("OpenAI library is not installed. Please install it with: pip install openai")

    is_azure_openai = isinstance(client, openai.AsyncAzureOpenAI)

    # Patch Chat Completions API
    chat_create_func = client.chat.completions.create

    @wraps(chat_create_func)
    async def traced_chat_create_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)

        if stream:
            return handle_async_streaming_create(
                *args,
                **kwargs,
                create_func=chat_create_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
                api_type="chat_completions",
            )
        return await handle_async_non_streaming_create(
            *args,
            **kwargs,
            create_func=chat_create_func,
            inference_id=inference_id,
            is_azure_openai=is_azure_openai,
            api_type="chat_completions",
        )

    client.chat.completions.create = traced_chat_create_func

    # Patch parse method if it exists
    if hasattr(client.chat.completions, 'parse'):
        parse_func = client.chat.completions.parse

        @wraps(parse_func)
        async def traced_parse_func(*args, **kwargs):
            inference_id = kwargs.pop("inference_id", None)
            stream = kwargs.get("stream", False)

            if stream:
                return handle_async_streaming_parse(
                    *args,
                    **kwargs,
                    parse_func=parse_func,
                    inference_id=inference_id,
                    is_azure_openai=is_azure_openai,
                )
            return await handle_async_non_streaming_parse(
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
        async def traced_responses_create_func(*args, **kwargs):
            inference_id = kwargs.pop("inference_id", None)
            stream = kwargs.get("stream", False)

            if stream:
                return handle_async_responses_streaming_create(
                    *args,
                    **kwargs,
                    create_func=responses_create_func,
                    inference_id=inference_id,
                    is_azure_openai=is_azure_openai,
                )
            return await handle_async_responses_non_streaming_create(
                *args,
                **kwargs,
                create_func=responses_create_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
            )

        client.responses.create = traced_responses_create_func
    else:
        logger.debug("Responses API not available in this AsyncOpenAI client version")

    return client


async def handle_async_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    api_type: str = "chat_completions",
    **kwargs,
) -> AsyncIterator[Any]:
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
    AsyncIterator[Any]
        A generator that yields the chunks of the completion.
    """
    chunks = await create_func(*args, **kwargs)

    # Create and return a new async generator that processes chunks
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
        async for chunk in chunks:
            raw_outputs.append(chunk.model_dump())
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1
            i += 1

            delta = chunk.choices[0].delta

            if delta.content:
                collected_output_data.append(delta.content)
            elif delta.function_call:
                if delta.function_call.name:
                    collected_function_call["name"] += delta.function_call.name
                if delta.function_call.arguments:
                    collected_function_call["arguments"] += delta.function_call.arguments
            elif delta.tool_calls:
                if delta.tool_calls[0].function.name:
                    collected_function_call["name"] += delta.tool_calls[0].function.name
                if delta.tool_calls[0].function.arguments:
                    collected_function_call["arguments"] += delta.tool_calls[0].function.arguments

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
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                collected_function_call["arguments"] = json.loads(collected_function_call["arguments"])
                output_data = collected_function_call

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs["messages"]},
                output=output_data,
                latency=latency,
                tokens=num_of_completion_tokens,
                prompt_tokens=0,
                completion_tokens=num_of_completion_tokens,
                model=kwargs.get("model"),
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={"timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)},
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


async def handle_async_non_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    api_type: str = "chat_completions",
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
    response = await create_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_non_streaming_output_data(response)
        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": kwargs["messages"]},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=response.model,
            model_parameters=get_model_parameters(kwargs),
            raw_output=response.model_dump(),
            id=inference_id,
        )

        add_to_trace(
            is_azure_openai=is_azure_openai,
            **trace_args,
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the create chat completion request with Openlayer. %s", e)

    return response


# -------------------------------- Async Responses API Handlers -------------------------------- #


async def handle_async_responses_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    """Handles the Responses API create method when streaming is enabled (async version).

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
    AsyncIterator[Any]
        An async generator that yields the chunks of the response stream.
    """
    chunks = await create_func(*args, **kwargs)

    # Create and return a new async generator that processes chunks
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
        async for chunk in chunks:
            raw_outputs.append(chunk.model_dump() if hasattr(chunk, "model_dump") else str(chunk))
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1
            i += 1

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
            collected_output_data = [message for message in collected_output_data if message is not None]
            if collected_output_data:
                output_data = "".join(collected_output_data)
            else:
                if collected_function_call["arguments"]:
                    try:
                        collected_function_call["arguments"] = json.loads(collected_function_call["arguments"])
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
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None),
                    "api_type": "responses",
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


async def handle_async_responses_non_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Handles the Responses API create method when streaming is disabled (async version).

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
    response = await create_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_responses_output_data(response)
        usage_data = extract_responses_usage(response)

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
            raw_output=response.model_dump() if hasattr(response, "model_dump") else str(response),
            id=inference_id,
            metadata={"api_type": "responses"},
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


async def handle_async_streaming_parse(
    parse_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
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
    AsyncIterator[Any]
        A generator that yields the chunks of the completion.
    """
    chunks = await parse_func(*args, **kwargs)

    # Create and return a new async generator that processes chunks
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
        async for chunk in chunks:
            raw_outputs.append(chunk.model_dump())
            if i == 0:
                first_token_time = time.time()
            if i > 0:
                num_of_completion_tokens = i + 1
            i += 1

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
                collected_function_call["arguments"] = json.loads(
                    collected_function_call["arguments"]
                )
                output_data = collected_function_call

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs["messages"]},
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


async def handle_async_non_streaming_parse(
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
    response = await parse_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        output_data = parse_structured_output_data(response)
        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": kwargs["messages"]},
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
