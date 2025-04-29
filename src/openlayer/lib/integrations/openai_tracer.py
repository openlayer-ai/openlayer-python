"""Module with methods used to trace OpenAI / Azure OpenAI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, List, Optional, Union

import openai

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_openai(
    client: Union[openai.OpenAI, openai.AzureOpenAI],
) -> Union[openai.OpenAI, openai.AzureOpenAI]:
    """Patch the OpenAI or AzureOpenAI client to trace chat completions.

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
    client : Union[openai.OpenAI, openai.AzureOpenAI]
        The OpenAI client to patch.

    Returns
    -------
    Union[openai.OpenAI, openai.AzureOpenAI]
        The patched OpenAI client.
    """
    is_azure_openai = isinstance(client, openai.AzureOpenAI)
    create_func = client.chat.completions.create

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
                is_azure_openai=is_azure_openai,
            )
        return handle_non_streaming_create(
            *args,
            **kwargs,
            create_func=create_func,
            inference_id=inference_id,
            is_azure_openai=is_azure_openai,
        )

    client.chat.completions.create = traced_create_func
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


def add_to_trace(is_azure_openai: bool = False, **kwargs) -> None:
    """Add a chat completion step to the trace."""
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
        logger.error(
            "Failed to trace the create chat completion request with Openlayer. %s", e
        )

    return response


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


# --------------------------- OpenAI Assistants API -------------------------- #
def trace_openai_assistant_thread_run(
    client: openai.OpenAI, run: "openai.types.beta.threads.run.Run"
) -> None:
    """Trace a run from an OpenAI assistant.

    Once the run is completed, the thread data is published to Openlayer,
    along with the latency, and number of tokens used."""
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
        print(f"Failed to monitor run. {e}")  # noqa: T201


def _type_check_run(run: "openai.types.beta.threads.run.Run") -> None:
    """Validate the run object."""
    if not isinstance(run, openai.types.beta.threads.run.Run):
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
