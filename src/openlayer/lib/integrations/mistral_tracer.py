"""Module with methods used to trace Mistral LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union

import mistralai

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_mistral(
    client: mistralai.Mistral,
) -> mistralai.Mistral:
    """Patch the Mistral client to trace chat completions.

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
    client : mistralai.Mistral
        The Mistral client to patch.

    Returns
    -------
    mistralai.Mistral
        The patched Mistral client.
    """
    stream_func = client.chat.stream
    create_func = client.chat.complete

    @wraps(stream_func)
    def traced_stream_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        return handle_streaming_create(
            *args,
            **kwargs,
            create_func=stream_func,
            inference_id=inference_id,
        )

    @wraps(create_func)
    def traced_create_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        return handle_non_streaming_create(
            *args,
            **kwargs,
            create_func=create_func,
            inference_id=inference_id,
        )

    client.chat.stream = traced_stream_func
    client.chat.complete = traced_create_func

    return client


def handle_streaming_create(
    create_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the create method when streaming is enabled.

    Parameters
    ----------
    create_func : callable
        The create method to handle.
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
    )


def stream_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, any],
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
            delta = chunk.data.choices[0].delta

            if delta.content:
                collected_output_data.append(delta.content)
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

            # Get usage data from the last chunk
            usage = chunk.model_dump()["data"].get("usage", {})

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs["messages"]},
                output=output_data,
                latency=latency,
                tokens=usage.get("total_tokens", num_of_completion_tokens),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", num_of_completion_tokens),
                model=kwargs.get("model"),
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={"timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)},
            )
            add_to_trace(
                **trace_args,
            )

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the create chat completion request with Openlayer. %s",
                e,
            )


def handle_non_streaming_create(
    create_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> mistralai.models.ChatCompletionResponse:
    """Handles the create method when streaming is disabled.

    Parameters
    ----------
    create_func : callable
        The create method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    mistralai.models.ChatCompletionResponse
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
            **trace_args,
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the create chat completion request with Openlayer. %s", e)

    return response


def parse_non_streaming_output_data(
    response: mistralai.models.ChatCompletionResponse,
) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Parameters
    ----------
    response : mistralai.models.ChatCompletionResponse
        The chat completion response.
    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed output data.
    """
    output_content = response.choices[0].message.content
    output_tool_calls = response.choices[0].message.tool_calls
    if output_content:
        output_data = output_content.strip()
    elif output_tool_calls:
        function_call = {
            "name": output_tool_calls[0].function.name,
            "arguments": json.loads(output_tool_calls[0].function.arguments),
        }
        output_data = function_call
    else:
        output_data = None
    return output_data


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs."""
    return {
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 1.0),
        "max_tokens": kwargs.get("max_tokens"),
        "min_tokens": kwargs.get("min_tokens"),
        "stream": kwargs.get("stream", False),
        "stop": kwargs.get("stop", None),
        "random_seed": kwargs.get("random_seed"),
        "response_format": kwargs.get("response_format", "text"),
        "safe_prompt": kwargs.get("safe_prompt", False),
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


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Mistral Chat Completion", provider="Mistral")
