"""Module with methods used to trace AWS Bedrock LLMs."""

import io
import json
import logging
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Union

from botocore.response import StreamingBody


try:
    import boto3

    HAVE_BOTO3 = True
except ImportError:
    HAVE_BOTO3 = False

if TYPE_CHECKING:
    import boto3

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_bedrock(
    client: "boto3.client",
) -> "boto3.client":
    """Patch the Bedrock client to trace model invocations.

    The following information is collected for each model invocation:
    - start_time: The time when the invocation was requested.
    - end_time: The time when the invocation was received.
    - latency: The time it took to generate the completion.
    - tokens: The total number of tokens used to generate the completion.
    - prompt_tokens: The number of tokens in the prompt.
    - completion_tokens: The number of tokens in the completion.
    - model: The model used to generate the completion.
    - model_parameters: The parameters used to configure the model.
    - raw_output: The raw output of the model.
    - inputs: The inputs used to generate the completion.
    - metadata: Additional metadata about the completion.

    Parameters
    ----------
    client : boto3.client
        The Bedrock client to patch.

    Returns
    -------
    boto3.client
        The patched Bedrock client.
    """
    if not HAVE_BOTO3:
        raise ImportError(
            "boto3 library is not installed. Please install it with: pip install boto3"
        )

    # Patch invoke_model for non-streaming requests
    invoke_model_func = client.invoke_model
    invoke_model_stream_func = client.invoke_model_with_response_stream

    @wraps(invoke_model_func)
    def traced_invoke_model(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        return handle_non_streaming_invoke(
            *args,
            **kwargs,
            invoke_func=invoke_model_func,
            inference_id=inference_id,
        )

    @wraps(invoke_model_stream_func)
    def traced_invoke_model_stream(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        return handle_streaming_invoke(
            *args,
            **kwargs,
            invoke_func=invoke_model_stream_func,
            inference_id=inference_id,
        )

    client.invoke_model = traced_invoke_model
    client.invoke_model_with_response_stream = traced_invoke_model_stream
    return client


def handle_non_streaming_invoke(
    invoke_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Handles the invoke_model method for non-streaming requests."""
    start_time = time.time()
    response = invoke_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        # Parse the input body
        body_str = kwargs.get("body", "{}")
        if isinstance(body_str, bytes):
            body_str = body_str.decode("utf-8")
        body_data = json.loads(body_str) if isinstance(body_str, str) else body_str

        # Read the response body ONCE and preserve it
        original_body = response["body"]
        response_body_bytes = original_body.read()

        # Parse the response data for tracing
        if isinstance(response_body_bytes, bytes):
            response_body_str = response_body_bytes.decode("utf-8")
        else:
            response_body_str = response_body_bytes
        response_data = json.loads(response_body_str)

        # Create a NEW StreamingBody with the same data and type
        # This preserves the exact botocore.response.StreamingBody type
        new_stream = io.BytesIO(response_body_bytes)
        response["body"] = StreamingBody(new_stream, len(response_body_bytes))

        # Extract data for tracing
        inputs = extract_inputs_from_body(body_data)
        output_data = extract_output_data(response_data)
        tokens_info = extract_tokens_info(response_data)
        model_id = kwargs.get("modelId", "unknown")
        metadata = extract_metadata(response_data)

        trace_args = create_trace_args(
            end_time=end_time,
            inputs=inputs,
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=tokens_info.get("total_tokens", 0),
            prompt_tokens=tokens_info.get("input_tokens", 0),
            completion_tokens=tokens_info.get("output_tokens", 0),
            model=model_id,
            model_parameters=get_model_parameters(body_data),
            raw_output=response_data,
            id=inference_id,
            metadata=metadata,
        )

        add_to_trace(**trace_args)

    except Exception as e:
        logger.error(
            "Failed to trace the Bedrock model invocation with Openlayer. %s", e
        )

    # Return the response with the properly restored body
    return response


def handle_streaming_invoke(
    invoke_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the invoke_model_with_response_stream method for streaming requests.

    Parameters
    ----------
    invoke_func : callable
        The invoke_model_with_response_stream method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    response = invoke_func(*args, **kwargs)
    return stream_chunks(
        response=response,
        kwargs=kwargs,
        inference_id=inference_id,
    )


def stream_chunks(
    response: Dict[str, Any],
    kwargs: Dict[str, Any],
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    collected_tool_calls = []
    current_tool_call = None
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = num_of_prompt_tokens = None
    latency = None
    final_metadata = {}

    try:
        # Parse the input body
        body_str = kwargs.get("body", "{}")
        if isinstance(body_str, bytes):
            body_str = body_str.decode("utf-8")
        body_data = json.loads(body_str) if isinstance(body_str, str) else body_str

        stream = response["body"]
        i = 0
        for i, event in enumerate(stream):
            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                raw_outputs.append(chunk_data)

                if i == 0:
                    first_token_time = time.time()

                # Handle different event types
                if chunk_data.get("type") == "message_start":
                    # Extract prompt tokens from message start
                    usage = chunk_data.get("message", {}).get("usage", {})
                    num_of_prompt_tokens = usage.get("input_tokens", 0)

                elif chunk_data.get("type") == "content_block_start":
                    content_block = chunk_data.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        current_tool_call = {
                            "type": "tool_use",
                            "id": content_block.get("id", ""),
                            "name": content_block.get("name", ""),
                            "input": "",
                        }

                elif chunk_data.get("type") == "content_block_delta":
                    delta = chunk_data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        collected_output_data.append(delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        if current_tool_call:
                            current_tool_call["input"] += delta.get("partial_json", "")

                elif chunk_data.get("type") == "content_block_stop":
                    if current_tool_call:
                        # Parse the JSON input
                        try:
                            current_tool_call["input"] = json.loads(
                                current_tool_call["input"]
                            )
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            pass
                        collected_tool_calls.append(current_tool_call)
                        current_tool_call = None

                elif chunk_data.get("type") == "message_delta":
                    # Extract final metadata like stop_reason
                    delta = chunk_data.get("delta", {})
                    if "stop_reason" in delta:
                        final_metadata["stop_reason"] = delta["stop_reason"]
                    if "stop_sequence" in delta:
                        final_metadata["stop_sequence"] = delta["stop_sequence"]

                elif chunk_data.get("type") == "message_stop":
                    # Extract final usage information
                    usage = chunk_data.get("usage", {})
                    if usage:
                        num_of_completion_tokens = usage.get("output_tokens", 0)

                yield event

        end_time = time.time()
        latency = (end_time - start_time) * 1000

    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            # Determine output data
            if collected_output_data:
                output_data = "".join(collected_output_data)
            elif collected_tool_calls:
                output_data = (
                    collected_tool_calls[0]
                    if len(collected_tool_calls) == 1
                    else collected_tool_calls
                )
            else:
                output_data = ""

            # Extract inputs
            inputs = extract_inputs_from_body(body_data)
            model_id = kwargs.get("modelId", "unknown")

            # Calculate total tokens
            total_tokens = (num_of_prompt_tokens or 0) + (num_of_completion_tokens or 0)

            # Add streaming metadata
            metadata = {
                "timeToFirstToken": (
                    (first_token_time - start_time) * 1000 if first_token_time else None
                ),
                **final_metadata,
            }

            trace_args = create_trace_args(
                end_time=end_time,
                inputs=inputs,
                output=output_data,
                latency=latency,
                tokens=total_tokens,
                prompt_tokens=num_of_prompt_tokens or 0,
                completion_tokens=num_of_completion_tokens or 0,
                model=model_id,
                model_parameters=get_model_parameters(body_data),
                raw_output=raw_outputs,
                id=inference_id,
                metadata=metadata,
            )
            add_to_trace(**trace_args)

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the streaming Bedrock model invocation with Openlayer. %s",
                e,
            )


def extract_inputs_from_body(body_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inputs from the request body."""
    inputs = {}

    # Add messages if present
    if "messages" in body_data:
        inputs["prompt"] = body_data["messages"]

    # Add system prompt if present
    if "system" in body_data:
        inputs["system"] = body_data["system"]

    # Add tools if present
    if "tools" in body_data:
        inputs["tools"] = body_data["tools"]

    # If no messages, try to extract prompt or fallback to entire body
    if not inputs:
        if "prompt" in body_data:
            inputs["prompt"] = body_data["prompt"]
        else:
            inputs["prompt"] = body_data

    return inputs


def extract_output_data(
    response_data: Dict[str, Any],
) -> Union[str, Dict[str, Any], list, None]:
    """Extract output data from the response."""
    # Handle Anthropic model response format
    if "content" in response_data and isinstance(response_data["content"], list):
        content_list = response_data["content"]

        # If single content item, return it directly
        if len(content_list) == 1:
            content = content_list[0]
            if content.get("type") == "text":
                return content.get("text", "")
            elif content.get("type") == "tool_use":
                return {
                    "type": "tool_use",
                    "id": content.get("id"),
                    "name": content.get("name"),
                    "input": content.get("input"),
                }
            elif content.get("type") == "image":
                return {"type": "image", "source": content.get("source")}

        # Multiple content items, return the list
        else:
            output_list = []
            for content in content_list:
                if content.get("type") == "text":
                    output_list.append(content.get("text", ""))
                elif content.get("type") == "tool_use":
                    output_list.append(
                        {
                            "type": "tool_use",
                            "id": content.get("id"),
                            "name": content.get("name"),
                            "input": content.get("input"),
                        }
                    )
                elif content.get("type") == "image":
                    output_list.append(
                        {"type": "image", "source": content.get("source")}
                    )
            return output_list

    # Handle other response formats (fallback for non-Anthropic models)
    elif "completion" in response_data:
        return response_data["completion"]
    elif "text" in response_data:
        return response_data["text"]
    elif "response" in response_data:
        return response_data["response"]

    # Fallback
    return str(response_data)


def extract_tokens_info(response_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage information from the response."""
    tokens_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Handle Anthropic model response format
    if "usage" in response_data:
        usage = response_data["usage"]
        tokens_info["input_tokens"] = usage.get("input_tokens", 0)
        tokens_info["output_tokens"] = usage.get("output_tokens", 0)
        tokens_info["total_tokens"] = (
            tokens_info["input_tokens"] + tokens_info["output_tokens"]
        )

    return tokens_info


def extract_metadata(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from the response."""
    metadata = {}

    # Add stop information
    if "stop_reason" in response_data:
        metadata["stop_reason"] = response_data["stop_reason"]

    if "stop_sequence" in response_data:
        metadata["stop_sequence"] = response_data["stop_sequence"]

    # Add response ID and type
    if "id" in response_data:
        metadata["response_id"] = response_data["id"]

    if "type" in response_data:
        metadata["response_type"] = response_data["type"]

    # Add role information
    if "role" in response_data:
        metadata["role"] = response_data["role"]

    return metadata


def get_model_parameters(body_data: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the request body."""
    # Extract all possible parameters from the Bedrock API
    return {
        "max_tokens": body_data.get("max_tokens"),
        "temperature": body_data.get("temperature"),
        "top_p": body_data.get("top_p"),
        "top_k": body_data.get("top_k"),
        "stop_sequences": body_data.get("stop_sequences"),
        "anthropic_version": body_data.get("anthropic_version"),
        "anthropic_beta": body_data.get("anthropic_beta"),
        "tool_choice": body_data.get("tool_choice"),
        "tools": body_data.get("tools"),
        "system": body_data.get("system"),
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
    tracer.add_chat_completion_step_to_trace(
        **kwargs, name="AWS Bedrock Model Invocation", provider="Bedrock"
    )
