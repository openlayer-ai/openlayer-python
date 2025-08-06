"""Module with methods used to trace Oracle OCI Generative AI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union, TYPE_CHECKING

try:
    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    HAVE_OCI = True
except ImportError:
    HAVE_OCI = False

if TYPE_CHECKING:
    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_oci_genai(
    client: "GenerativeAiInferenceClient",
    estimate_tokens: bool = True,
) -> "GenerativeAiInferenceClient":
    """Patch the OCI Generative AI client to trace chat completions.

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
    client : GenerativeAiInferenceClient
        The OCI Generative AI client to patch.
    estimate_tokens : bool, optional
        Whether to estimate token counts when not provided by the OCI response.
        Defaults to True. When False, token fields will be None if not available.

    Returns
    -------
    GenerativeAiInferenceClient
        The patched OCI client.
    """
    if not HAVE_OCI:
        raise ImportError("oci library is not installed. Please install it with: pip install oci")

    chat_func = client.chat

    @wraps(chat_func)
    def traced_chat_func(*args, **kwargs):
        # Extract chat_details from args or kwargs
        chat_details = args[0] if args else kwargs.get("chat_details")

        if chat_details is None:
            raise ValueError("Could not determine chat_details from arguments.")

        # Check if streaming is enabled
        stream = False
        if hasattr(chat_details, "chat_request"):
            chat_request = chat_details.chat_request
            stream = getattr(chat_request, "is_stream", False)

        # Measure timing around the actual OCI call
        start_time = time.time()
        response = chat_func(*args, **kwargs)
        end_time = time.time()

        if stream:
            return handle_streaming_chat(
                response=response,
                chat_details=chat_details,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                estimate_tokens=estimate_tokens,
            )
        else:
            return handle_non_streaming_chat(
                response=response,
                chat_details=chat_details,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                estimate_tokens=estimate_tokens,
            )

    client.chat = traced_chat_func
    return client


def handle_streaming_chat(
    response: Iterator[Any],
    chat_details: Any,
    kwargs: Dict[str, Any],
    start_time: float,
    end_time: float,
    estimate_tokens: bool = True,
) -> Iterator[Any]:
    """Handles the chat method when streaming is enabled.

    Parameters
    ----------
    response : Iterator[Any]
        The streaming response from the OCI chat method.
    chat_details : Any
        The chat details object.
    kwargs : Dict[str, Any]
        Additional keyword arguments.

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    return stream_chunks(
        chunks=response.data.events(),
        chat_details=chat_details,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
        estimate_tokens=estimate_tokens,
    )


def stream_chunks(
    chunks: Iterator[Any],
    chat_details: Any,
    kwargs: Dict[str, Any],
    start_time: float,
    end_time: float,
    estimate_tokens: bool = True,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    collected_function_calls = []
    # Simplified streaming stats - only track essential metrics
    total_chunks = 0
    first_chunk_time = None
    last_chunk_time = None
    chunk_samples = []  # Simplified sampling
    
    end_time = None
    first_token_time = None
    num_of_completion_tokens = num_of_prompt_tokens = None
    latency = None

    try:
        for i, chunk in enumerate(chunks):
            total_chunks = i + 1
            
            # Only track timing for first and last chunks to minimize overhead
            if i == 0:
                first_token_time = time.time()
                first_chunk_time = first_token_time
                # Extract prompt tokens from first chunk if available
                if hasattr(chunk, "data") and hasattr(chunk.data, "usage"):
                    usage = chunk.data.usage
                    num_of_prompt_tokens = getattr(usage, "prompt_tokens", 0)
                else:
                    # OCI doesn't provide usage info, estimate from chat_details if enabled
                    if estimate_tokens:
                        num_of_prompt_tokens = estimate_prompt_tokens_from_chat_details(chat_details)
                    else:
                        num_of_prompt_tokens = None
                    
                # Store first chunk sample (only for debugging)
                if hasattr(chunk, "data"):
                    chunk_samples.append({"index": 0, "type": "first"})
            
            # Update completion tokens count (estimation based)
            if i > 0 and estimate_tokens:
                num_of_completion_tokens = i + 1

            # Fast content extraction - optimized for performance
            content = _extract_chunk_content(chunk)
            if content:
                if isinstance(content, dict) and "function_call" in content:
                    collected_function_calls.append(content["function_call"])
                elif content:  # Text content
                    collected_output_data.append(str(content))

            yield chunk

        # Update final timing
        last_chunk_time = time.time()
        end_time = last_chunk_time
        latency = (end_time - start_time) * 1000

    except Exception as e:
        logger.error("Failed yield chunk. %s", e)
    finally:
        # Try to add step to the trace
        try:
            # Determine output data
            if collected_output_data:
                output_data = "".join(collected_output_data)
            elif collected_function_calls:
                output_data = (
                    collected_function_calls[0] if len(collected_function_calls) == 1 else collected_function_calls
                )
            else:
                output_data = ""

            # chat_details is passed directly as parameter
            model_id = extract_model_id(chat_details)

            # Calculate total tokens - handle None values properly
            if estimate_tokens:
                total_tokens = (num_of_prompt_tokens or 0) + (num_of_completion_tokens or 0)
            else:
                total_tokens = None if num_of_prompt_tokens is None and num_of_completion_tokens is None else ((num_of_prompt_tokens or 0) + (num_of_completion_tokens or 0))

            # Simplified metadata - only essential timing info
            metadata = {
                "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None),
            }

            trace_args = create_trace_args(
                end_time=end_time,
                inputs=extract_inputs_from_chat_details(chat_details),
                output=output_data,
                latency=latency,
                tokens=total_tokens,
                prompt_tokens=num_of_prompt_tokens,
                completion_tokens=num_of_completion_tokens,
                model=model_id,
                model_parameters=get_model_parameters(chat_details),
                raw_output={
                    "streaming_summary": {
                        "total_chunks": total_chunks,
                        "duration_seconds": (last_chunk_time - first_chunk_time) if last_chunk_time and first_chunk_time else 0,
                    },
                    "complete_response": "".join(collected_output_data) if collected_output_data else None,
                },
                id=None,
                metadata=metadata,
            )
            add_to_trace(**trace_args)

        except Exception as e:
            logger.error(
                "Failed to trace the streaming OCI chat completion request with Openlayer. %s",
                e,
            )


def handle_non_streaming_chat(
    response: Any,
    chat_details: Any,
    kwargs: Dict[str, Any],
    start_time: float,
    end_time: float,
    estimate_tokens: bool = True,
) -> Any:
    """Handles the chat method when streaming is disabled.

    Parameters
    ----------
    response : Any
        The response from the OCI chat method.
    chat_details : Any
        The chat details object.
    kwargs : Dict[str, Any]
        Additional keyword arguments.

    Returns
    -------
    Any
        The chat completion response.
    """
    # Use the timing from the actual OCI call (passed as parameters)
    # start_time and end_time are already provided

    try:
        # Parse response and extract data
        output_data = parse_non_streaming_output_data(response)
        tokens_info = extract_tokens_info(response, chat_details, estimate_tokens)
        model_id = extract_model_id(chat_details)

        latency = (end_time - start_time) * 1000

        # Extract additional metadata
        additional_metadata = extract_response_metadata(response)

        trace_args = create_trace_args(
            end_time=end_time,
            inputs=extract_inputs_from_chat_details(chat_details),
            output=output_data,
            latency=latency,
            tokens=tokens_info.get("total_tokens"),
            prompt_tokens=tokens_info.get("input_tokens"),
            completion_tokens=tokens_info.get("output_tokens"),
            model=model_id,
            model_parameters=get_model_parameters(chat_details),
            raw_output=response.data.__dict__ if hasattr(response, "data") else response.__dict__,
            id=None,
            metadata=additional_metadata,
        )

        add_to_trace(**trace_args)

    except Exception as e:
        logger.error("Failed to trace the OCI chat completion request with Openlayer. %s", e)

    return response


def extract_response_metadata(response) -> Dict[str, Any]:
    """Extract additional metadata from the OCI response."""
    metadata = {}

    if not hasattr(response, "data"):
        return metadata

    try:
        data = response.data

        # Extract model_id and model_version
        if hasattr(data, "model_id"):
            metadata["model_id"] = data.model_id
        if hasattr(data, "model_version"):
            metadata["model_version"] = data.model_version

        # Extract chat response metadata
        if hasattr(data, "chat_response"):
            chat_response = data.chat_response

            # Extract time_created
            if hasattr(chat_response, "time_created"):
                metadata["time_created"] = str(chat_response.time_created)

            # Extract finish_reason from first choice
            if hasattr(chat_response, "choices") and chat_response.choices:
                choice = chat_response.choices[0]
                if hasattr(choice, "finish_reason"):
                    metadata["finish_reason"] = choice.finish_reason

                # Extract index
                if hasattr(choice, "index"):
                    metadata["choice_index"] = choice.index

            # Extract API format
            if hasattr(chat_response, "api_format"):
                metadata["api_format"] = chat_response.api_format

    except Exception as e:
        logger.debug("Error extracting response metadata: %s", e)

    return metadata


def extract_inputs_from_chat_details(chat_details) -> Dict[str, Any]:
    """Extract inputs from the chat details in a clean format."""
    inputs = {}

    if chat_details is None:
        return inputs

    try:
        if hasattr(chat_details, "chat_request"):
            chat_request = chat_details.chat_request

            # Extract messages in clean format
            if hasattr(chat_request, "messages") and chat_request.messages:
                messages = []
                for msg in chat_request.messages:
                    # Extract role and convert to OpenAI format (lowercase)
                    role = getattr(msg, "role", "USER").lower()

                    # Extract content text
                    content_text = ""
                    if hasattr(msg, "content") and msg.content:
                        # Handle content as list of content objects
                        if isinstance(msg.content, list):
                            text_parts = []
                            for content_item in msg.content:
                                if hasattr(content_item, "text"):
                                    text_parts.append(content_item.text)
                                elif isinstance(content_item, dict) and "text" in content_item:
                                    text_parts.append(content_item["text"])
                            content_text = " ".join(text_parts)
                        else:
                            content_text = str(msg.content)

                    messages.append({"role": role, "content": content_text})

                inputs["prompt"] = messages

            # Extract tools if present
            if hasattr(chat_request, "tools") and chat_request.tools:
                inputs["tools"] = chat_request.tools

    except Exception as e:
        logger.debug("Error extracting inputs: %s", e)
        inputs["prompt"] = str(chat_details)

    return inputs


def parse_non_streaming_output_data(response) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion, extracting clean text."""
    if not hasattr(response, "data"):
        return str(response)

    try:
        data = response.data

        # Handle OCI chat response structure
        if hasattr(data, "chat_response"):
            chat_response = data.chat_response
            if hasattr(chat_response, "choices") and chat_response.choices:
                choice = chat_response.choices[0]

                # Extract text from message content
                if hasattr(choice, "message") and choice.message:
                    message = choice.message
                    if hasattr(message, "content") and message.content:
                        # Handle content as list of content objects
                        if isinstance(message.content, list):
                            text_parts = []
                            for content_item in message.content:
                                if hasattr(content_item, "text"):
                                    text_parts.append(content_item.text)
                                elif isinstance(content_item, dict) and "text" in content_item:
                                    text_parts.append(content_item["text"])
                            return " ".join(text_parts)
                        else:
                            return str(message.content)

        # Handle choice-based responses (fallback)
        elif hasattr(data, "choices") and data.choices:
            choice = data.choices[0]

            # Handle message content
            if hasattr(choice, "message"):
                message = choice.message
                if hasattr(message, "content") and message.content:
                    if isinstance(message.content, list):
                        text_parts = []
                        for content_item in message.content:
                            if hasattr(content_item, "text"):
                                text_parts.append(content_item.text)
                        return " ".join(text_parts)
                    return str(message.content)
                elif hasattr(message, "function_call") and message.function_call:
                    return {
                        "function_call": {
                            "name": getattr(message.function_call, "name", ""),
                            "arguments": getattr(message.function_call, "arguments", ""),
                        }
                    }

            # Handle text content directly
            elif hasattr(choice, "text") and choice.text:
                return choice.text

        # Handle direct text responses
        elif hasattr(data, "text") and data.text:
            return data.text

        # Handle generated_text field
        elif hasattr(data, "generated_text") and data.generated_text:
            return data.generated_text

    except Exception as e:
        logger.debug("Error parsing output data: %s", e)

    return str(data)


def estimate_prompt_tokens_from_chat_details(chat_details) -> Optional[int]:
    """Estimate prompt tokens from chat details when OCI doesn't provide usage info."""
    if not chat_details:
        return None

    try:
        input_text = ""
        if hasattr(chat_details, "chat_request") and hasattr(chat_details.chat_request, "messages"):
            for msg in chat_details.chat_request.messages:
                if hasattr(msg, "content") and msg.content:
                    for content_item in msg.content:
                        if hasattr(content_item, "text"):
                            input_text += content_item.text + " "

        # Rough estimation: ~4 characters per token
        estimated_tokens = max(1, len(input_text) // 4)
        return estimated_tokens
    except Exception as e:
        logger.debug("Error estimating prompt tokens: %s", e)
        return None


def extract_tokens_info(response, chat_details=None, estimate_tokens: bool = True) -> Dict[str, Optional[int]]:
    """Extract token usage information from the response.
    
    Handles both CohereChatResponse and GenericChatResponse types from OCI.
    
    Parameters
    ----------
    response : Any
        The OCI chat response object (CohereChatResponse or GenericChatResponse)
    chat_details : Any, optional
        The chat details for token estimation if needed
    estimate_tokens : bool, optional
        Whether to estimate tokens when not available in response. Defaults to True.
        
    Returns
    -------
    Dict[str, Optional[int]]
        Dictionary with token counts. Values can be None if unavailable and estimation disabled.
    """
    tokens_info = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    try:
        # Extract token usage from OCI response (handles both CohereChatResponse and GenericChatResponse)
        if hasattr(response, "data"):
            usage = None
            
            # For CohereChatResponse: response.data.usage
            if hasattr(response.data, "usage"):
                usage = response.data.usage
            # For GenericChatResponse: response.data.chat_response.usage  
            elif hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "usage"):
                usage = response.data.chat_response.usage
                
            if usage is not None:
                # Extract tokens from usage object
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                
                tokens_info["input_tokens"] = prompt_tokens
                tokens_info["output_tokens"] = completion_tokens
                tokens_info["total_tokens"] = total_tokens or (
                    (prompt_tokens + completion_tokens) if prompt_tokens is not None and completion_tokens is not None else None
                )
                logger.debug("Found token usage info: %s", tokens_info)
                return tokens_info

            # If no usage info found, estimate based on text length only if estimation is enabled
            if estimate_tokens:
                logger.debug("No token usage found in response, estimating from text length")
                
                # Estimate input tokens from chat_details
                if chat_details:
                    try:
                        input_text = ""
                        if hasattr(chat_details, "chat_request") and hasattr(chat_details.chat_request, "messages"):
                            for msg in chat_details.chat_request.messages:
                                if hasattr(msg, "content") and msg.content:
                                    for content_item in msg.content:
                                        if hasattr(content_item, "text"):
                                            input_text += content_item.text + " "

                        # Rough estimation: ~4 characters per token
                        estimated_input_tokens = max(1, len(input_text) // 4)
                        tokens_info["input_tokens"] = estimated_input_tokens
                    except Exception as e:
                        logger.debug("Error estimating input tokens: %s", e)
                        tokens_info["input_tokens"] = None

                # Estimate output tokens from response
                try:
                    output_text = parse_non_streaming_output_data(response)
                    if isinstance(output_text, str):
                        # Rough estimation: ~4 characters per token
                        estimated_output_tokens = max(1, len(output_text) // 4)
                        tokens_info["output_tokens"] = estimated_output_tokens
                    else:
                        tokens_info["output_tokens"] = None
                except Exception as e:
                    logger.debug("Error estimating output tokens: %s", e)
                    tokens_info["output_tokens"] = None

                # Calculate total tokens only if we have estimates
                if tokens_info["input_tokens"] is not None and tokens_info["output_tokens"] is not None:
                    tokens_info["total_tokens"] = tokens_info["input_tokens"] + tokens_info["output_tokens"]
                elif tokens_info["input_tokens"] is not None or tokens_info["output_tokens"] is not None:
                    tokens_info["total_tokens"] = (tokens_info["input_tokens"] or 0) + (tokens_info["output_tokens"] or 0)
                else:
                    tokens_info["total_tokens"] = None
                    
                logger.debug("Estimated token usage: %s", tokens_info)
            else:
                logger.debug("No token usage found in response and estimation disabled, returning None values")

    except Exception as e:
        logger.debug("Error extracting/estimating token info: %s", e)
        # Always return None values on exceptions (no more fallback values)
        tokens_info = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    return tokens_info


def extract_model_id(chat_details) -> str:
    """Extract model ID from chat details."""
    if chat_details is None:
        return "unknown"

    try:
        if hasattr(chat_details, "chat_request"):
            chat_request = chat_details.chat_request
            if hasattr(chat_request, "model_id") and chat_request.model_id:
                return chat_request.model_id

        # Try to extract from serving mode
        if hasattr(chat_details, "serving_mode"):
            serving_mode = chat_details.serving_mode
            if hasattr(serving_mode, "model_id") and serving_mode.model_id:
                return serving_mode.model_id

    except Exception as e:
        logger.debug("Error extracting model ID: %s", e)

    return "unknown"


def get_model_parameters(chat_details) -> Dict[str, Any]:
    """Gets the model parameters from the chat details."""
    if chat_details is None or not hasattr(chat_details, "chat_request"):
        return {}

    try:
        chat_request = chat_details.chat_request

        return {
            "max_tokens": getattr(chat_request, "max_tokens", None),
            "temperature": getattr(chat_request, "temperature", None),
            "top_p": getattr(chat_request, "top_p", None),
            "top_k": getattr(chat_request, "top_k", None),
            "frequency_penalty": getattr(chat_request, "frequency_penalty", None),
            "presence_penalty": getattr(chat_request, "presence_penalty", None),
            "stop": getattr(chat_request, "stop", None),
            "tools": getattr(chat_request, "tools", None),
            "tool_choice": getattr(chat_request, "tool_choice", None),
            "is_stream": getattr(chat_request, "is_stream", None),
            "is_echo": getattr(chat_request, "is_echo", None),
            "log_probs": getattr(chat_request, "log_probs", None),
            "logit_bias": getattr(chat_request, "logit_bias", None),
            "num_generations": getattr(chat_request, "num_generations", None),
            "seed": getattr(chat_request, "seed", None),
        }
    except Exception as e:
        logger.debug("Error extracting model parameters: %s", e)
        return {}


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


def _extract_chunk_content(chunk) -> Optional[Union[str, Dict[str, Any]]]:
    """Fast content extraction from OCI chunk - optimized for performance."""
    try:
        if not hasattr(chunk, "data"):
            return None
            
        data = chunk.data
        
        # Fast path: Handle JSON string chunks
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                
                # Handle OCI streaming structure: message.content[0].text
                if "message" in parsed_data and "content" in parsed_data["message"]:
                    content = parsed_data["message"]["content"]
                    if isinstance(content, list) and content:
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "TEXT":
                                text = content_item.get("text")
                                if text:
                                    return text
                    elif content:
                        return str(content)
                
                # Handle function calls
                elif "function_call" in parsed_data:
                    return {
                        "function_call": {
                            "name": parsed_data["function_call"].get("name", ""),
                            "arguments": parsed_data["function_call"].get("arguments", ""),
                        }
                    }
                
                # Handle direct text field
                elif "text" in parsed_data:
                    text = parsed_data["text"]
                    if text:
                        return text
                        
            except json.JSONDecodeError:
                return None
        
        # Fast path: Handle object-based chunks
        else:
            # Handle choices-based structure
            if hasattr(data, "choices") and data.choices:
                choice = data.choices[0]
                
                # Handle delta content
                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if hasattr(delta, "content") and delta.content:
                        return delta.content
                    elif hasattr(delta, "function_call") and delta.function_call:
                        return {
                            "function_call": {
                                "name": getattr(delta.function_call, "name", ""),
                                "arguments": getattr(delta.function_call, "arguments", ""),
                            }
                        }
                
                # Handle message content
                elif hasattr(choice, "message"):
                    message = choice.message
                    if hasattr(message, "content") and message.content:
                        return message.content
                    elif hasattr(message, "function_call") and message.function_call:
                        return {
                            "function_call": {
                                "name": getattr(message.function_call, "name", ""),
                                "arguments": getattr(message.function_call, "arguments", ""),
                            }
                        }
            
            # Handle direct text responses
            elif hasattr(data, "text") and data.text:
                return data.text
                
    except Exception:
        # Silent failure for performance - don't log per chunk
        pass
        
    return None


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Oracle OCI Chat Completion", provider="OCI")
