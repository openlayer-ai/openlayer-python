"""Module with methods used to trace LiteLLM completions."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union, TYPE_CHECKING

try:
    import litellm
    HAVE_LITELLM = True
except ImportError:
    HAVE_LITELLM = False

if TYPE_CHECKING:
    import litellm

from ..tracing import tracer

logger = logging.getLogger(__name__)


def trace_litellm() -> None:
    """Patch the litellm.completion function to trace completions.

    The following information is collected for each completion:
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

    Returns
    -------
    None
        This function patches litellm.completion in place.

    Example
    -------
    >>> import litellm
    >>> from openlayer.lib import trace_litellm
    >>> 
    >>> # Enable tracing
    >>> trace_litellm()
    >>> 
    >>> # Use LiteLLM normally - tracing happens automatically
    >>> response = litellm.completion(
    ...     model="gpt-3.5-turbo",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     inference_id="custom-id-123"  # Optional Openlayer parameter
    ... )
    """
    if not HAVE_LITELLM:
        raise ImportError(
            "LiteLLM library is not installed. Please install it with: pip install litellm"
        )
    
    original_completion = litellm.completion

    @wraps(original_completion)
    def traced_completion(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        stream = kwargs.get("stream", False)

        if stream:
            return handle_streaming_completion(
                *args,
                **kwargs,
                completion_func=original_completion,
                inference_id=inference_id,
            )
        return handle_non_streaming_completion(
            *args,
            **kwargs,
            completion_func=original_completion,
            inference_id=inference_id,
        )

    litellm.completion = traced_completion


def handle_streaming_completion(
    completion_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """Handles the completion function when streaming is enabled.

    Parameters
    ----------
    completion_func : callable
        The completion function to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Iterator[Any]
        A generator that yields the chunks of the completion.
    """
    # Enable usage data in streaming by setting stream_options
    # This ensures we get proper token usage data in the final chunk
    # Reference: https://docs.litellm.ai/docs/completion/usage
    if "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}
    
    chunks = completion_func(*args, **kwargs)
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
    model_name = kwargs.get("model", "unknown")
    latest_usage_data = {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None}
    provider = "unknown"
    latest_chunk_metadata = {}
    
    try:
        i = 0
        for i, chunk in enumerate(chunks):
            raw_outputs.append(chunk.model_dump() if hasattr(chunk, 'model_dump') else str(chunk))
            
            if i == 0:
                first_token_time = time.time()
                # Try to detect provider from the first chunk
                provider = detect_provider_from_chunk(chunk, model_name)
            
            # Extract usage data from this chunk if available (usually in final chunks)
            chunk_usage = extract_usage_from_chunk(chunk)
            if any(v is not None for v in chunk_usage.values()):
                latest_usage_data = chunk_usage
                
            # Always update metadata from latest chunk (for cost, headers, etc.)
            chunk_metadata = extract_litellm_metadata(chunk, model_name)
            if chunk_metadata:
                latest_chunk_metadata.update(chunk_metadata)
                
            if i > 0:
                num_of_completion_tokens = i + 1

            # Handle different chunk formats based on provider
            delta = get_delta_from_chunk(chunk)

            if delta and hasattr(delta, 'content') and delta.content:
                collected_output_data.append(delta.content)
            elif delta and hasattr(delta, 'function_call') and delta.function_call:
                if delta.function_call.name:
                    collected_function_call["name"] += delta.function_call.name
                if delta.function_call.arguments:
                    collected_function_call["arguments"] += delta.function_call.arguments
            elif delta and hasattr(delta, 'tool_calls') and delta.tool_calls:
                if delta.tool_calls[0].function.name:
                    collected_function_call["name"] += delta.tool_calls[0].function.name
                if delta.tool_calls[0].function.arguments:
                    collected_function_call["arguments"] += delta.tool_calls[0].function.arguments

            yield chunk
            
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to yield chunk. %s", e)
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
                        pass
                output_data = collected_function_call

            # Post-streaming calculations (after streaming is finished)
            completion_tokens_calculated, prompt_tokens_calculated, total_tokens_calculated, cost_calculated = calculate_streaming_usage_and_cost(
                chunks=raw_outputs,
                messages=kwargs.get("messages", []),
                output_content=output_data,
                model_name=model_name,
                latest_usage_data=latest_usage_data,
                latest_chunk_metadata=latest_chunk_metadata
            )
            
            # Use calculated values (fall back to extracted data if calculation fails)
            usage_data = latest_usage_data if any(v is not None for v in latest_usage_data.values()) else {}
            
            final_prompt_tokens = prompt_tokens_calculated if prompt_tokens_calculated is not None else usage_data.get("prompt_tokens", 0)
            final_completion_tokens = completion_tokens_calculated if completion_tokens_calculated is not None else usage_data.get("completion_tokens", num_of_completion_tokens)
            final_total_tokens = total_tokens_calculated if total_tokens_calculated is not None else usage_data.get("total_tokens", final_prompt_tokens + final_completion_tokens)
            final_cost = cost_calculated if cost_calculated is not None else latest_chunk_metadata.get('cost', None)
            
            trace_args = create_trace_args(
                end_time=end_time,
                inputs={"prompt": kwargs.get("messages", [])},
                output=output_data,
                latency=latency,
                tokens=final_total_tokens,
                prompt_tokens=final_prompt_tokens,
                completion_tokens=final_completion_tokens,
                model=model_name,
                model_parameters=get_model_parameters(kwargs),
                raw_output=raw_outputs,
                id=inference_id,
                cost=final_cost,  # Use calculated cost
                metadata={
                    "timeToFirstToken": ((first_token_time - start_time) * 1000 if first_token_time else None),
                    "provider": provider,
                    "litellm_model": model_name,
                    **latest_chunk_metadata,  # Add all LiteLLM-specific metadata
                },
            )
            add_to_trace(**trace_args)

        # pylint: disable=broad-except
        except Exception as e:
            logger.error(
                "Failed to trace the LiteLLM completion request with Openlayer. %s",
                e,
            )


def handle_non_streaming_completion(
    completion_func: callable,
    *args,
    inference_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Handles the completion function when streaming is disabled.

    Parameters
    ----------
    completion_func : callable
        The completion function to handle.
    inference_id : Optional[str], optional
        A user-generated inference id, by default None

    Returns
    -------
    Any
        The completion response.
    """
    start_time = time.time()
    response = completion_func(*args, **kwargs)
    end_time = time.time()

    # Try to add step to the trace
    try:
        model_name = kwargs.get("model", getattr(response, 'model', 'unknown'))
        provider = detect_provider_from_response(response, model_name)
        output_data = parse_non_streaming_output_data(response)
        usage_data = extract_usage_from_response(response)
        
        # Extract additional LiteLLM metadata
        extra_metadata = extract_litellm_metadata(response, model_name)
        
        # Extract cost from metadata
        cost = extra_metadata.get('cost', None)
        
        trace_args = create_trace_args(
            end_time=end_time,
            inputs={"prompt": kwargs.get("messages", [])},
            output=output_data,
            latency=(end_time - start_time) * 1000,
            tokens=usage_data.get("total_tokens"),
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            model=model_name,
            model_parameters=get_model_parameters(kwargs),
            raw_output=response.model_dump() if hasattr(response, 'model_dump') else str(response),
            id=inference_id,
            cost=cost,  # Add cost as direct parameter
            metadata={
                "provider": provider,
                "litellm_model": model_name,
                **extra_metadata,  # Add all LiteLLM-specific metadata
            },
        )

        add_to_trace(**trace_args)
        
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the LiteLLM completion request with Openlayer. %s", e)

    return response


def get_model_parameters(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the model parameters from the kwargs."""
    return {
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 1.0),
        "max_tokens": kwargs.get("max_tokens", None),
        "n": kwargs.get("n", 1),
        "stream": kwargs.get("stream", False),
        "stop": kwargs.get("stop", None),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        "logit_bias": kwargs.get("logit_bias", None),
        "logprobs": kwargs.get("logprobs", False),
        "top_logprobs": kwargs.get("top_logprobs", None),
        "parallel_tool_calls": kwargs.get("parallel_tool_calls", True),
        "seed": kwargs.get("seed", None),
        "response_format": kwargs.get("response_format", None),
        "timeout": kwargs.get("timeout", None),
        "api_base": kwargs.get("api_base", None),
        "api_version": kwargs.get("api_version", None),
    }


def create_trace_args(
    end_time: float,
    inputs: Dict[str, Any],
    output: str,
    latency: float,
    tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    model_parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raw_output: Optional[str] = None,
    id: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
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
    if cost is not None:
        trace_args["cost"] = cost
    return trace_args


def add_to_trace(**kwargs) -> None:
    """Add a chat completion step to the trace."""
    provider = kwargs.get("metadata", {}).get("provider", "LiteLLM")
    tracer.add_chat_completion_step_to_trace(**kwargs, name="LiteLLM Chat Completion", provider=provider)


def parse_non_streaming_output_data(response: Any) -> Union[str, Dict[str, Any], None]:
    """Parses the output data from a non-streaming completion.

    Parameters
    ----------
    response : Any
        The completion response.
        
    Returns
    -------
    Union[str, Dict[str, Any], None]
        The parsed output data.
    """
    try:
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content:
                    return message.content.strip()
                elif hasattr(message, 'function_call') and message.function_call:
                    return {
                        "name": message.function_call.name,
                        "arguments": json.loads(message.function_call.arguments) if isinstance(message.function_call.arguments, str) else message.function_call.arguments,
                    }
                elif hasattr(message, 'tool_calls') and message.tool_calls:
                    return {
                        "name": message.tool_calls[0].function.name,
                        "arguments": json.loads(message.tool_calls[0].function.arguments) if isinstance(message.tool_calls[0].function.arguments, str) else message.tool_calls[0].function.arguments,
                    }
    except Exception as e:
        logger.debug("Error parsing output data: %s", e)
    
    return None


def detect_provider_from_response(response: Any, model_name: str) -> str:
    """Detect the provider from the response object."""
    try:
        # First try LiteLLM's built-in provider detection
        if HAVE_LITELLM:
            try:
                provider_info = litellm.get_llm_provider(model_name)
                if provider_info and len(provider_info) > 1:
                    return provider_info[1]  # provider_info is (model, provider, dynamic_api_key, api_base)
            except Exception:
                pass
        
        # Try to get provider from response metadata/hidden params
        if hasattr(response, '_hidden_params'):
            hidden_params = response._hidden_params
            if 'custom_llm_provider' in hidden_params:
                return hidden_params['custom_llm_provider']
            if 'litellm_provider' in hidden_params:
                return hidden_params['litellm_provider']
        
        # Try other response attributes
        if hasattr(response, 'response_metadata') and 'provider' in response.response_metadata:
            return response.response_metadata['provider']
            
        # Fallback to model name detection
        return detect_provider_from_model_name(model_name)
    except Exception:
        return "unknown"


def detect_provider_from_chunk(chunk: Any, model_name: str) -> str:
    """Detect the provider from a streaming chunk."""
    try:
        # First try LiteLLM's built-in provider detection
        if HAVE_LITELLM:
            try:
                import litellm
                provider_info = litellm.get_llm_provider(model_name)
                if provider_info and len(provider_info) > 1:
                    return provider_info[1]
            except Exception:
                pass
        
        # Try to get provider from chunk metadata/hidden params
        if hasattr(chunk, '_hidden_params'):
            hidden_params = chunk._hidden_params
            if 'custom_llm_provider' in hidden_params:
                return hidden_params['custom_llm_provider']
            if 'litellm_provider' in hidden_params:
                return hidden_params['litellm_provider']
        
        # Fallback to model name detection
        return detect_provider_from_model_name(model_name)
    except Exception:
        return "unknown"


def detect_provider_from_model_name(model_name: str) -> str:
    """Detect provider from model name patterns."""
    model_lower = model_name.lower()
    
    if model_lower.startswith(('gpt-', 'o1-', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada')):
        return "OpenAI"
    elif model_lower.startswith(('claude-', 'claude')):
        return "Anthropic"
    elif 'gemini' in model_lower or 'palm' in model_lower:
        return "Google"
    elif 'llama' in model_lower:
        return "Meta"
    elif model_lower.startswith('mistral'):
        return "Mistral"
    elif model_lower.startswith('command'):
        return "Cohere"
    else:
        return "unknown"


def get_delta_from_chunk(chunk: Any) -> Any:
    """Extract delta from chunk, handling different response formats."""
    try:
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta'):
                return choice.delta
    except Exception:
        pass
    return None


def extract_usage_from_response(response: Any) -> Dict[str, Optional[int]]:
    """Extract usage data from response."""
    try:
        if hasattr(response, 'usage'):
            usage = response.usage
            return {
                "total_tokens": getattr(usage, 'total_tokens', None),
                "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                "completion_tokens": getattr(usage, 'completion_tokens', None),
            }
    except Exception:
        pass
    
    return {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None}


def calculate_streaming_usage_and_cost(chunks, messages, output_content, model_name, latest_usage_data, latest_chunk_metadata):
    """Calculate usage and cost after streaming is finished.
    
    With stream_options={"include_usage": True}, LiteLLM provides accurate usage data
    in the final streaming chunk. This function prioritizes that data over estimation.
    
    Reference: https://docs.litellm.ai/docs/completion/usage
    """
    try:
        # Priority 1: Use actual usage data from streaming chunks (with stream_options)
        if latest_usage_data and latest_usage_data.get("total_tokens") and latest_usage_data.get("total_tokens") > 0:
            logger.debug("Using actual streaming usage data from chunks")
            return (
                latest_usage_data.get("completion_tokens"),
                latest_usage_data.get("prompt_tokens"),
                latest_usage_data.get("total_tokens"),
                latest_chunk_metadata.get("cost")
            )
        
        # Priority 2: Look for usage data in the final chunk directly
        for chunk_data in reversed(chunks):  # Check from the end
            if isinstance(chunk_data, dict) and "usage" in chunk_data and chunk_data["usage"]:
                usage = chunk_data["usage"]
                if usage.get("total_tokens", 0) > 0:
                    logger.debug("Found usage data in final chunk: %s", usage)
                    return (
                        usage.get("completion_tokens"),
                        usage.get("prompt_tokens"), 
                        usage.get("total_tokens"),
                        latest_chunk_metadata.get("cost")
                    )
        
        # Priority 3: Manual calculation as fallback
        logger.debug("Falling back to manual token calculation")
        completion_tokens = None
        prompt_tokens = None
        total_tokens = None
        cost = None
        
        # 1. Calculate completion tokens from output content
        if isinstance(output_content, str):
            # Simple token estimation: ~4 characters per token (rough approximation)
            completion_tokens = max(1, len(output_content) // 4)
        elif isinstance(output_content, dict):
            # For function calls, estimate based on JSON content length
            json_str = json.dumps(output_content) if output_content else "{}"
            completion_tokens = max(1, len(json_str) // 4)
        else:
            # Fallback: count chunks with content
            completion_tokens = len([chunk for chunk in chunks if chunk])
        
        # 2. Calculate prompt tokens from input messages
        if messages:
            # Simple estimation: sum of message content lengths
            total_chars = 0
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    total_chars += len(str(message["content"]))
            prompt_tokens = max(1, total_chars // 4)
        else:
            prompt_tokens = 0
        
        # 3. Calculate total tokens
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        
        # 4. Try to get cost from metadata or estimate
        cost = latest_chunk_metadata.get("cost")
        if cost is None and total_tokens and model_name:
            # Simple cost estimation for gpt-3.5-turbo (if we know the model)
            if "gpt-3.5-turbo" in model_name.lower():
                # Approximate cost: $0.0005 per 1K prompt tokens, $0.0015 per 1K completion tokens
                estimated_cost = (prompt_tokens * 0.0005 / 1000) + (completion_tokens * 0.0015 / 1000)
                cost = estimated_cost
        
        logger.debug(
            "Calculated streaming usage: prompt=%s, completion=%s, total=%s, cost=%s",
            prompt_tokens, completion_tokens, total_tokens, cost
        )
        
        return completion_tokens, prompt_tokens, total_tokens, cost
        
    except Exception as e:
        logger.debug("Error calculating streaming usage: %s", e)
        return None, None, None, None


def extract_usage_from_chunk(chunk: Any) -> Dict[str, Optional[int]]:
    """Extract usage data from streaming chunk."""
    try:
        # Check for usage attribute
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage = chunk.usage
            return {
                "total_tokens": getattr(usage, 'total_tokens', None),
                "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                "completion_tokens": getattr(usage, 'completion_tokens', None),
            }
        
        # Check for usage in _hidden_params (LiteLLM specific)
        if hasattr(chunk, '_hidden_params'):
            hidden_params = chunk._hidden_params
            # Check if usage is a direct attribute
            if hasattr(hidden_params, 'usage') and hidden_params.usage is not None:
                usage = hidden_params.usage
                return {
                    "total_tokens": getattr(usage, 'total_tokens', None),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(usage, 'completion_tokens', None),
                }
            # Check if usage is a dictionary key
            elif isinstance(hidden_params, dict) and 'usage' in hidden_params:
                usage = hidden_params['usage']
                if usage:
                    return {
                        "total_tokens": usage.get('total_tokens', None),
                        "prompt_tokens": usage.get('prompt_tokens', None),
                        "completion_tokens": usage.get('completion_tokens', None),
                    }
        
        # Check if chunk model dump has usage
        if hasattr(chunk, 'model_dump'):
            chunk_dict = chunk.model_dump()
            if 'usage' in chunk_dict and chunk_dict['usage']:
                usage = chunk_dict['usage']
                return {
                    "total_tokens": usage.get('total_tokens', None),
                    "prompt_tokens": usage.get('prompt_tokens', None),
                    "completion_tokens": usage.get('completion_tokens', None),
                }
    except Exception:
        pass
    
    return {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None}


def extract_litellm_metadata(response: Any, model_name: str) -> Dict[str, Any]:
    """Extract LiteLLM-specific metadata from response."""
    metadata = {}
    
    try:
        # Extract hidden parameters
        if hasattr(response, '_hidden_params'):
            hidden_params = response._hidden_params
            
            # Cost information
            if 'response_cost' in hidden_params:
                metadata['cost'] = hidden_params['response_cost']
            
            # API information
            if 'api_base' in hidden_params:
                metadata['api_base'] = hidden_params['api_base']
            if 'api_version' in hidden_params:
                metadata['api_version'] = hidden_params['api_version']
                
            # Model information
            if 'model_info' in hidden_params:
                metadata['model_info'] = hidden_params['model_info']
                
            # Additional provider info
            if 'additional_args' in hidden_params:
                metadata['additional_args'] = hidden_params['additional_args']
                
            # Extract response headers if available 
            if 'additional_headers' in hidden_params:
                headers = hidden_params['additional_headers']
                if headers:
                    metadata['response_headers'] = headers
        
        # Extract system fingerprint if available
        if hasattr(response, 'system_fingerprint'):
            metadata['system_fingerprint'] = response.system_fingerprint
            
        # Extract response headers if available
        if hasattr(response, '_response_headers'):
            metadata['response_headers'] = dict(response._response_headers)
            
    except Exception as e:
        logger.debug("Error extracting LiteLLM metadata: %s", e)
    
    return metadata
