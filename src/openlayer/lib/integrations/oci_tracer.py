"""Module with methods used to trace Oracle OCI Generative AI LLMs."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union, TYPE_CHECKING

try:
    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    from oci.generative_ai_inference.models import GenericChatRequest, ChatDetails
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
        inference_id = kwargs.pop("inference_id", None)
        
        # Extract chat_details from args or kwargs
        chat_details = args[0] if args else kwargs.get("chat_details")
        
        # Check if streaming is enabled
        stream = False
        if hasattr(chat_details, 'chat_request'):
            chat_request = chat_details.chat_request
            stream = getattr(chat_request, 'is_stream', False)
            
        if stream:
            return handle_streaming_chat(
                *args,
                **kwargs,
                chat_func=chat_func,
                inference_id=inference_id,
            )
        return handle_non_streaming_chat(
            *args,
            **kwargs,
            chat_func=chat_func,
            inference_id=inference_id,
        )

    client.chat = traced_chat_func
    return client


def handle_streaming_chat(
    chat_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the chat method when streaming is enabled.

    Parameters
    ----------
    chat_func : callable
        The chat method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    response = chat_func(*args, **kwargs)
    return stream_chunks(
        chunks=response,
        kwargs=kwargs,
        inference_id=inference_id,
    )


def stream_chunks(
    chunks: Iterator[Any],
    kwargs: Dict[str, Any],
    inference_id: Optional[str] = None,
):
    """Streams the chunks of the completion and traces the completion."""
    collected_output_data = []
    collected_function_calls = []
    raw_outputs = []
    start_time = time.time()
    end_time = None
    first_token_time = None
    num_of_completion_tokens = num_of_prompt_tokens = None
    latency = None
    
    try:
        i = 0
        for i, chunk in enumerate(chunks):
            # Store raw output
            if hasattr(chunk, 'data'):
                raw_outputs.append(chunk.data.__dict__)
            else:
                raw_outputs.append(str(chunk))
            
            if i == 0:
                first_token_time = time.time()
                # Extract prompt tokens from first chunk if available
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'usage'):
                    usage = chunk.data.usage
                    num_of_prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    
            if i > 0:
                num_of_completion_tokens = i + 1
                
            # Extract content from chunk based on OCI response structure
            try:
                if hasattr(chunk, 'data'):
                    data = chunk.data
                    
                    # Handle different response structures
                    if hasattr(data, 'choices') and data.choices:
                        choice = data.choices[0]
                        
                        # Handle delta content
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            if hasattr(delta, 'content') and delta.content:
                                collected_output_data.append(delta.content)
                            elif hasattr(delta, 'function_call') and delta.function_call:
                                collected_function_calls.append({
                                    "name": getattr(delta.function_call, 'name', ''),
                                    "arguments": getattr(delta.function_call, 'arguments', '')
                                })
                        
                        # Handle message content
                        elif hasattr(choice, 'message'):
                            message = choice.message
                            if hasattr(message, 'content') and message.content:
                                collected_output_data.append(message.content)
                            elif hasattr(message, 'function_call') and message.function_call:
                                collected_function_calls.append({
                                    "name": getattr(message.function_call, 'name', ''),
                                    "arguments": getattr(message.function_call, 'arguments', '')
                                })
                    
                    # Handle text-only responses
                    elif hasattr(data, 'text') and data.text:
                        collected_output_data.append(data.text)
                        
            except Exception as chunk_error:
                logger.debug("Error processing chunk: %s", chunk_error)
                
            yield chunk
            
        end_time = time.time()
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
                output_data = collected_function_calls[0] if len(collected_function_calls) == 1 else collected_function_calls
            else:
                output_data = ""
                
            # Extract chat_details from kwargs for input processing
            chat_details = kwargs.get("chat_details") or (args[0] if args else None)
            model_id = extract_model_id(chat_details)
            
            # Calculate total tokens
            total_tokens = (num_of_prompt_tokens or 0) + (num_of_completion_tokens or 0)
            
            # Add streaming metadata
            metadata = {
                "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None),
            }
            
            trace_args = create_trace_args(
                end_time=end_time,
                inputs=extract_inputs_from_chat_details(chat_details),
                output=output_data,
                latency=latency,
                tokens=total_tokens,
                prompt_tokens=num_of_prompt_tokens or 0,
                completion_tokens=num_of_completion_tokens or 0,
                model=model_id,
                model_parameters=get_model_parameters(chat_details),
                raw_output=raw_outputs,
                id=inference_id,
                metadata=metadata,
            )
            add_to_trace(**trace_args)
            
        except Exception as e:
            logger.error(
                "Failed to trace the streaming OCI chat completion request with Openlayer. %s",
                e,
            )


def handle_non_streaming_chat(
    chat_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Handles the chat method when streaming is disabled.

    Parameters
    ----------
    chat_func : callable
        The chat method to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Any
        The chat completion response.
    """
    start_time = time.time()
    response = chat_func(*args, **kwargs)
    end_time = time.time()
    
    try:
        # Extract chat_details for input processing
        chat_details = args[0] if args else kwargs.get("chat_details")
        
        # Parse response and extract data
        output_data = parse_non_streaming_output_data(response)
        tokens_info = extract_tokens_info(response)
        model_id = extract_model_id(chat_details)
        
        trace_args = create_trace_args(
            end_time=end_time,
            inputs=extract_inputs_from_chat_details(chat_details),
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=tokens_info.get("total_tokens", 0),
            prompt_tokens=tokens_info.get("input_tokens", 0),
            completion_tokens=tokens_info.get("output_tokens", 0),
            model=model_id,
            model_parameters=get_model_parameters(chat_details),
            raw_output=response.data.__dict__ if hasattr(response, 'data') else response.__dict__,
            id=inference_id,
        )
        
        add_to_trace(**trace_args)
        
    except Exception as e:
        logger.error("Failed to trace the OCI chat completion request with Openlayer. %s", e)
    
    return response


def extract_inputs_from_chat_details(chat_details) -> Dict[str, Any]:
    """Extract inputs from the chat details."""
    inputs = {}
    
    if chat_details is None:
        return inputs
    
    try:
        if hasattr(chat_details, 'chat_request'):
            chat_request = chat_details.chat_request
            
            # Extract messages
            if hasattr(chat_request, 'messages') and chat_request.messages:
                # Convert messages to serializable format
                messages = []
                for msg in chat_request.messages:
                    if hasattr(msg, '__dict__'):
                        messages.append(msg.__dict__)
                    else:
                        messages.append(str(msg))
                inputs["prompt"] = messages
            
            # Extract system message if present
            if hasattr(chat_request, 'system_message') and chat_request.system_message:
                inputs["system"] = chat_request.system_message
                
            # Extract tools if present
            if hasattr(chat_request, 'tools') and chat_request.tools:
                inputs["tools"] = chat_request.tools
        
    except Exception as e:
        logger.debug("Error extracting inputs: %s", e)
        inputs["prompt"] = str(chat_details)
    
    return inputs


def parse_non_streaming_output_data(response) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion."""
    if not hasattr(response, 'data'):
        return str(response)
        
    try:
        data = response.data
        
        # Handle choice-based responses
        if hasattr(data, 'choices') and data.choices:
            choice = data.choices[0]
            
            # Handle message content
            if hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content:
                    return message.content
                elif hasattr(message, 'function_call') and message.function_call:
                    return {
                        "function_call": {
                            "name": getattr(message.function_call, 'name', ''),
                            "arguments": getattr(message.function_call, 'arguments', '')
                        }
                    }
            
            # Handle text content directly
            elif hasattr(choice, 'text') and choice.text:
                return choice.text
        
        # Handle direct text responses
        elif hasattr(data, 'text') and data.text:
            return data.text
            
        # Handle generated_text field
        elif hasattr(data, 'generated_text') and data.generated_text:
            return data.generated_text
        
    except Exception as e:
        logger.debug("Error parsing output data: %s", e)
    
    return str(data)


def extract_tokens_info(response) -> Dict[str, int]:
    """Extract token usage information from the response."""
    tokens_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    try:
        if hasattr(response, 'data') and hasattr(response.data, 'usage'):
            usage = response.data.usage
            tokens_info["input_tokens"] = getattr(usage, 'prompt_tokens', 0)
            tokens_info["output_tokens"] = getattr(usage, 'completion_tokens', 0)
            tokens_info["total_tokens"] = tokens_info["input_tokens"] + tokens_info["output_tokens"]
    except Exception as e:
        logger.debug("Error extracting token info: %s", e)
    
    return tokens_info


def extract_model_id(chat_details) -> str:
    """Extract model ID from chat details."""
    if chat_details is None:
        return "unknown"
        
    try:
        if hasattr(chat_details, 'chat_request'):
            chat_request = chat_details.chat_request
            if hasattr(chat_request, 'model_id') and chat_request.model_id:
                return chat_request.model_id
        
        # Try to extract from serving mode
        if hasattr(chat_details, 'serving_mode'):
            serving_mode = chat_details.serving_mode
            if hasattr(serving_mode, 'model_id') and serving_mode.model_id:
                return serving_mode.model_id
                
    except Exception as e:
        logger.debug("Error extracting model ID: %s", e)
        
    return "unknown"


def get_model_parameters(chat_details) -> Dict[str, Any]:
    """Gets the model parameters from the chat details."""
    if chat_details is None or not hasattr(chat_details, 'chat_request'):
        return {}
        
    try:
        chat_request = chat_details.chat_request
        
        return {
            "max_tokens": getattr(chat_request, 'max_tokens', None),
            "temperature": getattr(chat_request, 'temperature', None),
            "top_p": getattr(chat_request, 'top_p', None),
            "top_k": getattr(chat_request, 'top_k', None),
            "frequency_penalty": getattr(chat_request, 'frequency_penalty', None),
            "presence_penalty": getattr(chat_request, 'presence_penalty', None),
            "stop": getattr(chat_request, 'stop', None),
            "tools": getattr(chat_request, 'tools', None),
            "tool_choice": getattr(chat_request, 'tool_choice', None),
            "is_stream": getattr(chat_request, 'is_stream', None),
            "is_echo": getattr(chat_request, 'is_echo', None),
            "log_probs": getattr(chat_request, 'log_probs', None),
            "logit_bias": getattr(chat_request, 'logit_bias', None),
            "num_generations": getattr(chat_request, 'num_generations', None),
            "seed": getattr(chat_request, 'seed', None),
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


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    tracer.add_chat_completion_step_to_trace(**kwargs, name="Oracle OCI Chat Completion", provider="OCI")