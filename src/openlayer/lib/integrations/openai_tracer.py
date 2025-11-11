"""Module with methods used to trace OpenAI / Azure OpenAI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, List, Optional, Union, TYPE_CHECKING

try:
    import openai

    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

if TYPE_CHECKING:
    import openai

from ..tracing import tracer

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
        raise ImportError("OpenAI library is not installed. Please install it with: pip install openai")

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
                api_type="chat_completions",
            )
        return handle_non_streaming_create(
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
    api_type: str = "chat_completions",
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


def add_to_trace(is_azure_openai: bool = False, api_type: str = "chat_completions", **kwargs) -> None:
    """Add a chat completion or responses step to the trace."""
    # Remove api_type from kwargs to avoid passing it to the tracer
    kwargs.pop("api_type", None)

    if api_type == "responses":
        # Handle Responses API tracing
        if is_azure_openai:
            tracer.add_chat_completion_step_to_trace(**kwargs, name="Azure OpenAI Response", provider="Azure")
        else:
            tracer.add_chat_completion_step_to_trace(**kwargs, name="OpenAI Response", provider="OpenAI")
    else:
        # Handle Chat Completions API tracing (default behavior)
        if is_azure_openai:
            tracer.add_chat_completion_step_to_trace(**kwargs, name="Azure OpenAI Chat Completion", provider="Azure")
        else:
            tracer.add_chat_completion_step_to_trace(**kwargs, name="OpenAI Chat Completion", provider="OpenAI")


def handle_non_streaming_create(
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
    response = create_func(*args, **kwargs)
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
            raw_outputs.append(chunk.model_dump() if hasattr(chunk, "model_dump") else str(chunk))
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
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)
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
                    if hasattr(delta.function_call, "name") and delta.function_call.name:
                        func_call["name"] = delta.function_call.name
                    if hasattr(delta.function_call, "arguments") and delta.function_call.arguments:
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

    Args:
        kwargs: The parameters passed to the Responses API

    Returns:
        Dictionary with prompt as a messages array
    """
    messages = []

    # Handle different input formats for Responses API
    if "conversation" in kwargs:
        # Conversation is already in messages format
        conversation = kwargs["conversation"]
        if isinstance(conversation, list):
            messages = conversation
        else:
            # Single message, wrap it
            messages = [{"role": "user", "content": str(conversation)}]
    else:
        # Build messages array from available parameters
        if "instructions" in kwargs:
            messages.append({"role": "system", "content": kwargs["instructions"]})
        
        if "input" in kwargs:
            messages.append({"role": "user", "content": kwargs["input"]})
        elif "prompt" in kwargs:
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


def parse_responses_output_data(response: Any) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a Responses API response.

    Args:
        response: The Response object from the Responses API

    Returns:
        The parsed output data
    """
    try:
        # Handle Response object structure - check for output first (Responses API structure)
        if hasattr(response, "output") and response.output:
            if isinstance(response.output, list) and response.output:
                # Handle list of output messages
                first_output = response.output[0]
                if hasattr(first_output, "content") and first_output.content:
                    # Extract text from content list
                    if isinstance(first_output.content, list) and first_output.content:
                        text_content = first_output.content[0]
                        if hasattr(text_content, "text"):
                            return text_content.text.strip()
                    elif hasattr(first_output.content, "text"):
                        return first_output.content.text.strip()
                    else:
                        return str(first_output.content).strip()
                elif hasattr(first_output, "text"):
                    return first_output.text.strip()
            elif hasattr(response.output, "text"):
                return response.output.text.strip()
            elif hasattr(response.output, "content"):
                return str(response.output.content).strip()

        # Handle Chat Completions style structure (fallback)
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                message = choice.message
                if hasattr(message, "content") and message.content:
                    return message.content.strip()
                elif hasattr(message, "function_call"):
                    return {
                        "name": message.function_call.name,
                        "arguments": json.loads(message.function_call.arguments)
                        if message.function_call.arguments
                        else {},
                    }
                elif hasattr(message, "tool_calls") and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    return {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                    }

        # Handle direct text response
        if hasattr(response, "text") and response.text:
            return response.text.strip()

    except Exception as e:
        logger.debug("Could not parse Responses API output data: %s", e)

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
            usage["prompt_tokens"] = getattr(usage_obj, "input_tokens", getattr(usage_obj, "prompt_tokens", 0))
            # ResponseUsage uses 'output_tokens' instead of 'completion_tokens'
            usage["completion_tokens"] = getattr(usage_obj, "output_tokens", getattr(usage_obj, "completion_tokens", 0))
        elif hasattr(response, "token_usage"):
            # Alternative usage attribute name
            usage_obj = response.token_usage
            usage["total_tokens"] = getattr(usage_obj, "total_tokens", 0)
            usage["prompt_tokens"] = getattr(usage_obj, "input_tokens", getattr(usage_obj, "prompt_tokens", 0))
            usage["completion_tokens"] = getattr(usage_obj, "output_tokens", getattr(usage_obj, "completion_tokens", 0))
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


def parse_non_streaming_output_data(
    response: "openai.types.chat.chat_completion.ChatCompletion",
) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Parameters
    ----------
    response : openai.types.chat.chat_completion.ChatCompletion
        The chat completion response.
    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed output data.
    """
    output_content = response.choices[0].message.content
    output_function_call = response.choices[0].message.function_call
    output_tool_calls = response.choices[0].message.tool_calls
    if output_content:
        output_data = output_content.strip()
    elif output_function_call or output_tool_calls:
        if output_function_call:
            function_call = {
                "name": output_function_call.name,
                "arguments": json.loads(output_function_call.arguments),
            }
        else:
            function_call = {
                "name": output_tool_calls[0].function.name,
                "arguments": json.loads(output_tool_calls[0].function.arguments),
            }
        output_data = function_call
    else:
        output_data = None
    return output_data


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
        if hasattr(response, 'parsed') and response.parsed is not None:
            # Handle Pydantic models
            if hasattr(response.parsed, 'model_dump'):
                return response.parsed.model_dump()
            # Handle dict-like objects  
            elif hasattr(response.parsed, '__dict__'):
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
def trace_openai_assistant_thread_run(client: "openai.OpenAI", run: "openai.types.beta.threads.run.Run") -> None:
    """Trace a run from an OpenAI assistant.

    Once the run is completed, the thread data is published to Openlayer,
    along with the latency, and number of tokens used."""
    if not HAVE_OPENAI:
        raise ImportError("OpenAI library is not installed. Please install it with: pip install openai")

    _type_check_run(run)

    # Do nothing if the run is not completed
    if run.status != "completed":
        return

    try:
        # Extract vars
        run_step_vars = _extract_run_vars(run)
        metadata = _extract_run_metadata(run)

        # Convert thread to prompt
        messages = client.beta.threads.messages.list(thread_id=run.thread_id, order="asc")
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
