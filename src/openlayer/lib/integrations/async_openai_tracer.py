"""Module with methods used to trace async OpenAI/Azure OpenAI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, AsyncIterator, Optional, Union

import openai

from .openai_tracer import (
    get_model_parameters,
    create_trace_args,
    add_to_trace,
    parse_non_streaming_output_data,
)

logger = logging.getLogger(__name__)


def trace_async_openai(
    client: Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI],
) -> Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]:
    """Patch the AsyncOpenAI or AsyncAzureOpenAI client to trace chat completions.

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
    client : Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]
        The AsyncOpenAI client to patch.

    Returns
    -------
    Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]
        The patched AsyncOpenAI client.
    """
    is_azure_openai = isinstance(client, openai.AsyncAzureOpenAI)
    create_func = client.chat.completions.create

    @wraps(create_func)
    async def traced_create_func(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)

        if stream:
            return handle_async_streaming_create(
                *args,
                **kwargs,
                create_func=create_func,
                inference_id=inference_id,
                is_azure_openai=is_azure_openai,
            )
        return await handle_async_non_streaming_create(
            *args,
            **kwargs,
            create_func=create_func,
            inference_id=inference_id,
            is_azure_openai=is_azure_openai,
        )

    client.chat.completions.create = traced_create_func
    return client


async def handle_async_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
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


async def handle_async_non_streaming_create(
    create_func: callable,
    *args,
    is_azure_openai: bool = False,
    inference_id: Optional[str] = None,
    **kwargs,
) -> "openai.types.chat.chat_completion.ChatCompletion":
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
        logger.error(
            "Failed to trace the create chat completion request with Openlayer. %s", e
        )

    return response
