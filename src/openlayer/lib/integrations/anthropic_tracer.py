"""Module with methods used to trace Anthropic LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union

import anthropic

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_anthropic(
    client: anthropic.Anthropic,
) -> anthropic.Anthropic:
    """Patch the Anthropic client to trace chat completions.

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
    client : anthropic.Anthropic
        The Anthropic client to patch.

    Returns
    -------
    anthropic.Anthropic
        The patched Anthropic client.
    """
    create_func = client.messages.create

    @wraps(create_func)
    def traced_create_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)

        if stream:
            return handle_streaming_create(
                *args,
                **kwargs,
                create_func=create_func,
                inference_id=inference_id,
            )
        return handle_non_streaming_create(
            *args,
            **kwargs,
            create_func=create_func,
            inference_id=inference_id,
        )

    client.messages.create = traced_create_func
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
    output_data = ""
    collected_output_data = []
    collected_function_call = {
        "name": "",
        "inputs": "",
    }
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = num_of_prompt_tokens = None
    latency = None
    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(chunk.model_dump())
            if i == 0:
                first_token_time = time.time()
                if chunk.type == "message_start":
                    num_of_prompt_tokens = chunk.message.usage.input_tokens
            if i > 0:
                num_of_completion_tokens = i + 1

            if chunk.type == "content_block_start":
                content_block = chunk.content_block
                if content_block.type == "tool_use":
                    collected_function_call["name"] = content_block.name
            elif chunk.type == "content_block_delta":
                delta = chunk.delta
                if delta.type == "text_delta":
                    collected_output_data.append(delta.text)
                elif delta.type == "input_json_delta":
                    collected_function_call["inputs"] += delta.partial_json

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
                collected_function_call["inputs"] = json.loads(collected_function_call["inputs"])
                output_data = collected_function_call

            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs["messages"]},
                output=output_data,
                latency=latency,
                tokens=num_of_completion_tokens,
                prompt_tokens=num_of_prompt_tokens,
                completion_tokens=num_of_completion_tokens,
                model=kwargs.get("model"),
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                metadata={"timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None)},
            )
            add_to_trace(**trace_args)

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
) -> anthropic.types.Message:
    """Handles the create method when streaming is disabled.

    Parameters
    ----------
    create_func : callable
        The create method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    anthropic.types.Message
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
            tokens=response.usage.input_tokens + response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
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
    response: anthropic.types.Message,
) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Parameters
    ----------
    response : anthropic.types.Message
        The chat completion response.
    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed output data.
    """
    output_data = None
    output_content = response.content[0]
    if output_content.type == "text":
        output_data = output_content.text
    elif output_content.type == "tool_use":
        output_data = {"id": output_content.id, "name": output_content.name, "input": output_content.input}

    return output_data


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs."""
    return {
        "max_tokens": kwargs.get("max_tokens"),
        "stop_sequences": kwargs.get("stop_sequences"),
        "temperature": kwargs.get("temperature", 1.0),
        "tool_choice": kwargs.get("tool_choice", {}),
        "tools": kwargs.get("tools", []),
        "top_k": kwargs.get("top_k"),
        "top_p": kwargs.get("top_p"),
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
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Anthropic Message Creation", provider="Anthropic")
