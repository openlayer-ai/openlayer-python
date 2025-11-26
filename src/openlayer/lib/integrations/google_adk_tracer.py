"""Module with methods used to trace Google Agent Development Kit (ADK).

This module provides instrumentation for Google's Agent Development Kit (ADK),
capturing agent execution, LLM calls, tool calls, and other ADK-specific events.
"""

import contextvars
import json
import logging
import sys
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

try:
    import wrapt
    HAVE_WRAPT = True
except ImportError:
    HAVE_WRAPT = False

if TYPE_CHECKING:
    try:
        from google.adk.agents.base_agent import BaseAgent
        from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
    except ImportError:
        pass

try:
    import google.adk
    HAVE_GOOGLE_ADK = True
except ImportError:
    HAVE_GOOGLE_ADK = False

from ..tracing import tracer, steps, enums

logger = logging.getLogger(__name__)


# Track wrapped methods for cleanup
_wrapped_methods = []

# Context variable to store the current user query across nested calls
_current_user_query: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'google_adk_user_query', default=None
)

# Context variable to track the parent step for agent transfers
# When set, sub-agent steps will be created as children of this step instead of the current step
_agent_transfer_parent_step: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    'google_adk_transfer_parent', default=None
)


class NoOpSpan:
    """A no-op span that does nothing.
    
    This is used to prevent ADK from creating its own telemetry spans
    while we create Openlayer steps instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the no-op span."""
        pass

    def __enter__(self) -> "NoOpSpan":
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        pass

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        """No-op set_attribute."""
        pass

    def set_attributes(self, *args: Any, **kwargs: Any) -> None:
        """No-op set_attributes."""
        pass

    def add_event(self, *args: Any, **kwargs: Any) -> None:
        """No-op add_event."""
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        """No-op set_status."""
        pass

    def update_name(self, *args: Any, **kwargs: Any) -> None:
        """No-op update_name."""
        pass

    def is_recording(self) -> bool:
        """Return False since this is a no-op span."""
        return False

    def end(self, *args: Any, **kwargs: Any) -> None:
        """No-op end."""
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        """No-op record_exception."""
        pass


class NoOpTracer:
    """A tracer that creates no-op spans to prevent ADK from creating real spans.
    
    ADK has built-in OpenTelemetry tracing. We replace it with this no-op tracer
    to prevent duplicate spans and use Openlayer's tracing instead.
    """

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        """Return a no-op context manager."""
        return NoOpSpan()

    def start_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        """Return a no-op span."""
        return NoOpSpan()

    def use_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        """Return a no-op context manager."""
        return NoOpSpan()


def trace_google_adk() -> None:
    """Enable tracing for Google Agent Development Kit (ADK).
    
    This function patches Google ADK to trace agent execution, LLM calls,
    and tool calls to Openlayer. It uses a global patching approach that
    automatically instruments all ADK agents created after this function
    is called.
    
    The following information is collected for each operation:
    - Agent execution: agent name, tools, handoffs, sub-agents
    - LLM calls: model, tokens (prompt, completion, total), messages, config
    - Tool calls: tool name, arguments, results
    - Start/end times and latency for all operations
    
    Note:
        Agent transfers (handoffs via ``transfer_to_agent``) do not create 
        separate tool steps to avoid excessive nesting. Sub-agent executions
        are nested directly under the LLM call that initiates the transfer.
    
    Requirements:
        Make sure to install Google ADK with: ``pip install google-adk``
        and wrapt with: ``pip install wrapt``
    
    Raises:
        ImportError: If google-adk or wrapt is not installed.
    
    Example:
        .. code-block:: python
        
            import os
            os.environ["OPENLAYER_API_KEY"] = "your-api-key"
            os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your-pipeline-id"
            
            from openlayer.lib.integrations import trace_google_adk
            
            # Enable tracing (must be called before creating agents)
            trace_google_adk()
            
            # Now create and run your ADK agents
            from google.adk.agents import Agent
            
            agent = Agent(
                name="Assistant",
                model="gemini-2.0-flash-exp",
                instructions="You are a helpful assistant"
            )
            
            result = await agent.run_async(...)
    """
    if not HAVE_GOOGLE_ADK:
        raise ImportError(
            "google-adk library is not installed. "
            "Please install it with: pip install google-adk"
        )
    
    if not HAVE_WRAPT:
        raise ImportError(
            "wrapt library is not installed. "
            "Please install it with: pip install wrapt"
        )
    
    logger.info("Enabling Google ADK tracing for Openlayer")
    _patch_google_adk()


def unpatch_google_adk() -> None:
    """Remove all patches from Google ADK modules.
    
    This function restores ADK's original behavior by removing all
    Openlayer instrumentation and restoring ADK's built-in tracer.
    """
    if not HAVE_GOOGLE_ADK:
        logger.warning("google-adk is not installed, nothing to unpatch")
        return
    
    logger.info("Disabling Google ADK tracing for Openlayer")
    _unpatch_google_adk()


# ----------------------------- Helper Functions ----------------------------- #


def _build_llm_request_for_trace(llm_request: Any) -> Dict[str, Any]:
    """Build a dictionary representation of the LLM request for tracing.
    
    Args:
        llm_request: The ADK LLM request object.
        
    Returns:
        Dictionary containing model, config, and contents.
    """
    from google.genai import types
    
    result = {
        "model": llm_request.model,
        "config": llm_request.config.model_dump(exclude_none=True, exclude="response_schema"),
        "contents": [],
    }
    
    # Filter out inline_data (images/files) from contents for tracing
    for content in llm_request.contents:
        parts = [
            part for part in content.parts 
            if not hasattr(part, "inline_data") or not part.inline_data
        ]
        result["contents"].append(
            types.Content(role=content.role, parts=parts).model_dump(exclude_none=True)
        )
    
    return result


def _extract_messages_from_contents(contents: list) -> Dict[str, Any]:
    """Extract and normalize messages from ADK contents format.
    
    Converts ADK's message format (with role and parts) to Openlayer's
    expected format (with role and content).
    
    Args:
        contents: List of ADK content objects.
        
    Returns:
        Dictionary with normalized messages for Openlayer.
    """
    messages = []
    
    for content in contents:
        # Normalize role: "model" -> "assistant"
        raw_role = content.get("role", "user")
        if raw_role == "model":
            role = "assistant"
        elif raw_role in ["user", "system"]:
            role = raw_role
        else:
            role = raw_role
        
        parts = content.get("parts", [])
        
        # Extract text content from parts
        text_parts = []
        for part in parts:
            if "text" in part and part.get("text") is not None:
                text_parts.append(str(part["text"]))
        
        # Combine text parts into content
        if text_parts:
            content_str = "\n".join(text_parts)
            messages.append({"role": role, "content": content_str})
    
    return {"messages": messages, "prompt": messages}


def _extract_llm_attributes(
    llm_request_dict: Dict[str, Any], 
    llm_response: Optional[Any] = None
) -> Dict[str, Any]:
    """Extract LLM attributes from request and response.
    
    Args:
        llm_request_dict: Dictionary containing the LLM request data.
        llm_response: Optional LLM response object.
        
    Returns:
        Dictionary containing extracted attributes for the step.
    """
    attributes = {}
    
    # Extract model
    if "model" in llm_request_dict:
        attributes["model"] = llm_request_dict["model"]
    
    # Extract config parameters
    if "config" in llm_request_dict:
        config = llm_request_dict["config"]
        model_parameters = {}
        
        if "temperature" in config:
            model_parameters["temperature"] = config["temperature"]
        if "max_output_tokens" in config:
            model_parameters["max_output_tokens"] = config["max_output_tokens"]
        if "top_p" in config:
            model_parameters["top_p"] = config["top_p"]
        if "top_k" in config:
            model_parameters["top_k"] = config["top_k"]
        if "candidate_count" in config:
            model_parameters["candidate_count"] = config["candidate_count"]
        if "stop_sequences" in config:
            model_parameters["stop_sequences"] = config["stop_sequences"]
        
        if model_parameters:
            attributes["model_parameters"] = model_parameters
    
    # Add system instruction as a system message if present (do this first)
    if "config" in llm_request_dict and "system_instruction" in llm_request_dict["config"]:
        system_instruction = llm_request_dict["config"]["system_instruction"]
        attributes["inputs"] = {
            "messages": [{"role": "system", "content": system_instruction}],
            "prompt": [{"role": "system", "content": system_instruction}]
        }
    
    # Extract messages and append to existing inputs
    if "contents" in llm_request_dict:
        messages_data = _extract_messages_from_contents(llm_request_dict["contents"])
        if "inputs" in attributes:
            # Append to existing system message
            attributes["inputs"]["messages"].extend(messages_data["messages"])
            attributes["inputs"]["prompt"].extend(messages_data["prompt"])
        else:
            # No system instruction, use messages as-is
            attributes["inputs"] = messages_data
    
    # Extract response data
    if llm_response:
        try:
            response_dict = (
                json.loads(llm_response) 
                if isinstance(llm_response, str) 
                else llm_response
            )
            
            # Extract tokens from usage_metadata
            if "usage_metadata" in response_dict:
                usage = response_dict["usage_metadata"]
                attributes["prompt_tokens"] = usage.get("prompt_token_count", 0)
                attributes["completion_tokens"] = usage.get("candidates_token_count", 0)
                attributes["total_tokens"] = usage.get("total_token_count", 0)
            
            # Extract response content
            if "content" in response_dict and "parts" in response_dict["content"]:
                parts = response_dict["content"]["parts"]
                text_parts = []
                
                for part in parts:
                    if "text" in part and part.get("text") is not None:
                        text_parts.append(str(part["text"]))
                
                if text_parts:
                    attributes["output"] = "\n".join(text_parts)
            
            # Store raw response
            if isinstance(llm_response, str):
                attributes["raw_output"] = llm_response
            else:
                try:
                    attributes["raw_output"] = json.dumps(response_dict)
                except (TypeError, ValueError):
                    pass
        
        except Exception as e:
            logger.debug(f"Failed to extract response attributes: {e}")
    
    return attributes


def extract_agent_attributes(instance: Any) -> Dict[str, Any]:
    """Extract agent metadata for tracing.
    
    Args:
        instance: The ADK agent instance.
        
    Returns:
        Dictionary containing agent attributes.
    """
    attributes = {}
    
    if hasattr(instance, "name"):
        attributes["agent_name"] = instance.name
    if hasattr(instance, "description"):
        attributes["description"] = instance.description
    if hasattr(instance, "model"):
        attributes["model"] = instance.model
    if hasattr(instance, "instruction"):
        attributes["instruction"] = instance.instruction
    
    # Extract tool information
    if hasattr(instance, "tools") and instance.tools:
        tools_info = []
        for tool in instance.tools:
            if hasattr(tool, "name"):
                tool_info = {"name": tool.name}
                if hasattr(tool, "description"):
                    tool_info["description"] = tool.description
                tools_info.append(tool_info)
        if tools_info:
            attributes["tools"] = tools_info
    
    # Extract sub-agents recursively
    if hasattr(instance, "sub_agents") and instance.sub_agents:
        sub_agents_info = []
        for sub_agent in instance.sub_agents:
            sub_agent_attrs = extract_agent_attributes(sub_agent)
            sub_agents_info.append(sub_agent_attrs)
        if sub_agents_info:
            attributes["sub_agents"] = sub_agents_info
    
    return attributes


# ----------------------------- Wrapper Functions ---------------------------- #


def _base_agent_run_async_wrapper() -> Any:
    """Wrapper for BaseAgent.run_async to create agent execution steps.
    
    Returns:
        Decorator function that wraps the original method.
    """
    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        async def new_function():
            agent_name = instance.name if hasattr(instance, "name") else "Unknown Agent"
            
            # Check if this is a sub-agent being called via transfer
            transfer_parent = _agent_transfer_parent_step.get()
            
            # Reset the context variable for this agent execution (only for root agents)
            if transfer_parent is None:
                _current_user_query.set(None)
            
            # Extract invocation context for session/user IDs
            invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
            
            # Build metadata with session info
            metadata = {"agent_type": "google_adk"}
            if invocation_context:
                if hasattr(invocation_context, "invocation_id"):
                    metadata["invocation_id"] = invocation_context.invocation_id
                if hasattr(invocation_context, "session") and invocation_context.session:
                    if hasattr(invocation_context.session, "id"):
                        metadata["session_id"] = invocation_context.session.id
                    if hasattr(invocation_context.session, "user_id"):
                        metadata["user_id"] = invocation_context.session.user_id
            
            # Extract agent attributes
            agent_attrs = extract_agent_attributes(instance)
            
            # Placeholder for user query - will be updated by LLM wrapper
            inputs = {
                **agent_attrs,
                "user_query": "Processing..."
            }
            
            # If we're in a transfer, create the step as a child of the transfer parent
            # Otherwise, use normal context (child of current step)
            if transfer_parent is not None:
                logger.debug(f"Creating sub-agent step as sibling: {agent_name}")
                # Temporarily set the parent step, create our step, then restore
                step_cm = tracer.create_step(
                    name=f"Agent: {agent_name}",
                    step_type=enums.StepType.USER_CALL,
                    inputs=inputs,
                    metadata=metadata,
                    parent_step=transfer_parent
                )
                # Clear the transfer parent so nested steps work normally
                _agent_transfer_parent_step.set(None)
            else:
                step_cm = tracer.create_step(
                    name=f"Agent: {agent_name}",
                    step_type=enums.StepType.USER_CALL,
                    inputs=inputs,
                    metadata=metadata
                )
            
            # Use the step as a context manager and capture the actual step object
            with step_cm as step:
                try:
                    # Execute the agent
                    async_gen = wrapped(*args, **kwargs)
                    final_response = None
                    
                    async for event in async_gen:
                        # Extract final response from events
                        if hasattr(event, 'is_final_response') and event.is_final_response():
                            if hasattr(event, 'content') and event.content:
                                try:
                                    final_response = event.content.parts[0].text.strip()
                                except (AttributeError, IndexError):
                                    final_response = str(event.content)
                        
                        yield event
                    
                    # Update user_query from context variable if it was captured by LLM wrapper
                    captured_query = _current_user_query.get()
                    if captured_query:
                        step.inputs["user_query"] = captured_query
                    else:
                        step.inputs["user_query"] = "No query provided"
                    
                    # Set output with meaningful content
                    if final_response:
                        step.output = final_response
                    else:
                        step.output = "Agent execution completed"
                    
                except Exception as e:
                    step.output = f"Error: {str(e)}"
                    logger.error(f"Error in agent execution: {e}")
                    raise
        
        return new_function()
    
    return actual_decorator


def _base_llm_flow_call_llm_async_wrapper() -> Any:
    """Wrapper for BaseLlmFlow._call_llm_async to create LLM call steps.
    
    Returns:
        Decorator function that wraps the original method.
    """
    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        async def new_function():
            # Extract invocation context for session/user IDs
            invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
            
            # Build metadata with session info
            metadata = {"llm_system": "google_vertex"}
            if invocation_context:
                if hasattr(invocation_context, "invocation_id"):
                    metadata["invocation_id"] = invocation_context.invocation_id
                if hasattr(invocation_context, "session") and invocation_context.session:
                    if hasattr(invocation_context.session, "id"):
                        metadata["session_id"] = invocation_context.session.id
                    if hasattr(invocation_context.session, "user_id"):
                        metadata["user_id"] = invocation_context.session.user_id
            
            # Extract LLM request
            llm_request = args[1] if len(args) > 1 else None
            model_name = "unknown"
            
            if llm_request and hasattr(llm_request, "model"):
                model_name = llm_request.model
            
            # Build request dict
            llm_request_dict = None
            if llm_request:
                llm_request_dict = _build_llm_request_for_trace(llm_request)
            
            # Extract initial attributes
            inputs = {}
            model_parameters = {}
            if llm_request_dict:
                attrs = _extract_llm_attributes(llm_request_dict, None)
                if "inputs" in attrs:
                    inputs = attrs["inputs"]
                if "model_parameters" in attrs:
                    model_parameters = attrs["model_parameters"]
                
                # Extract user query from the messages and store in context variable
                # This allows the parent agent step to access it
                if "inputs" in attrs and "messages" in attrs["inputs"]:
                    messages = attrs["inputs"]["messages"]
                    # Find the last user message (most recent user query)
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            user_query = msg.get("content", "")
                            if user_query and _current_user_query.get() is None:
                                # Only set if not already set (first user message)
                                _current_user_query.set(user_query)
                            break
            
            # Use tracer.create_step context manager
            with tracer.create_step(
                name=f"LLM Call: {model_name}",
                step_type=enums.StepType.CHAT_COMPLETION,
                inputs=inputs,
                metadata=metadata
            ) as step:
                # Set ChatCompletionStep attributes
                step.model = model_name
                step.provider = "Google"
                step.model_parameters = model_parameters
                
                try:
                    # Execute LLM call
                    async_gen = wrapped(*args, **kwargs)
                    collected_responses = []
                    
                    async for item in async_gen:
                        collected_responses.append(item)
                        yield item
                    
                    # The response will be finalized by _finalize_model_response_event_wrapper
                    
                except Exception as e:
                    step.output = f"Error: {str(e)}"
                    logger.error(f"Error in LLM call: {e}")
                    raise
        
        return new_function()
    
    return actual_decorator


def _call_tool_async_wrapper() -> Any:
    """Wrapper for __call_tool_async to create tool execution steps.
    
    Returns:
        Decorator function that wraps the original method.
    """
    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        async def new_function():
            # Extract tool information
            tool = args[0] if args else kwargs.get("tool")
            tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})
            tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")
            
            tool_name = getattr(tool, "name", "unknown_tool")
            tool_description = getattr(tool, "description", None)
            
            # Check if this is an agent transfer (handoff)
            is_agent_transfer = tool_name == "transfer_to_agent" or (
                tool_description and "transfer" in tool_description.lower()
            )
            
            # For agent transfers, don't create a step - let the agent step handle it
            # But set the transfer parent so the sub-agent becomes a sibling of the LLM call
            if is_agent_transfer:
                logger.debug(f"Handling agent transfer: {tool_name}")
                
                # Get the current step's parent (should be the LLM call)
                # We want the sub-agent to be a sibling of the LLM call, not a child
                # So we need to get the grandparent (the main agent step)
                current_step = tracer.get_current_step()
                if current_step and hasattr(current_step, 'parent_step') and current_step.parent_step:
                    # Set the LLM call's parent as the transfer parent
                    _agent_transfer_parent_step.set(current_step.parent_step)
                    logger.debug(f"Set transfer parent to: {current_step.parent_step.name if hasattr(current_step.parent_step, 'name') else 'unknown'}")
                
                try:
                    # Execute tool without creating a step
                    result = await wrapped(*args, **kwargs)
                    return result
                finally:
                    # Clear the transfer parent after execution
                    _agent_transfer_parent_step.set(None)
            
            # Build metadata with session info from tool_context
            metadata = {"tool_system": "google_adk"}
            if tool_description:
                metadata["description"] = tool_description
            
            # Extract session/user IDs from tool_context
            if tool_context:
                if hasattr(tool_context, "function_call_id"):
                    metadata["function_call_id"] = tool_context.function_call_id
                if hasattr(tool_context, "invocation_context"):
                    inv_ctx = tool_context.invocation_context
                    if hasattr(inv_ctx, "invocation_id"):
                        metadata["invocation_id"] = inv_ctx.invocation_id
                    if hasattr(inv_ctx, "session") and inv_ctx.session:
                        if hasattr(inv_ctx.session, "id"):
                            metadata["session_id"] = inv_ctx.session.id
                        if hasattr(inv_ctx.session, "user_id"):
                            metadata["user_id"] = inv_ctx.session.user_id
            
            # Use tracer.create_step context manager
            with tracer.create_step(
                name=f"Tool: {tool_name}",
                step_type=enums.StepType.TOOL,
                inputs=tool_args,
                metadata=metadata
            ) as step:
                # Set ToolStep attributes
                step.function_name = tool_name
                step.arguments = tool_args
                
                try:
                    # Execute tool
                    result = await wrapped(*args, **kwargs)
                    
                    # Set output
                    if isinstance(result, dict):
                        step.output = result
                    else:
                        step.output = str(result)
                    
                    return result
                    
                except Exception as e:
                    step.output = f"Error: {str(e)}"
                    logger.error(f"Error in tool execution: {e}")
                    raise
        
        return new_function()
    
    return actual_decorator


def _finalize_model_response_event_wrapper() -> Any:
    """Wrapper for _finalize_model_response_event to update LLM steps.
    
    This is called by ADK after an LLM response completes. We use it to
    update the current step with final token counts and response content.
    
    Returns:
        Decorator function that wraps the original method.
    """
    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        # Call the original method
        result = wrapped(*args, **kwargs)
        
        # Extract response data
        llm_request = args[0] if len(args) > 0 else kwargs.get("llm_request")
        llm_response = args[1] if len(args) > 1 else kwargs.get("llm_response")
        
        # Note: In a real implementation, we would update the current step here
        # For now, we just pass through since step management is handled in the
        # LLM wrapper itself
        
        return result
    
    return actual_decorator


# ----------------------------- Patching Functions --------------------------- #


def _patch(
    module_name: str, 
    object_name: str, 
    method_name: str, 
    wrapper_function: Any
) -> None:
    """Helper to apply a patch and keep track of it.
    
    Args:
        module_name: The module containing the object to patch.
        object_name: The class or object name to patch.
        method_name: The method name to patch.
        wrapper_function: The wrapper function to apply.
    """
    try:
        module = __import__(module_name, fromlist=[object_name])
        obj = getattr(module, object_name)
        wrapt.wrap_function_wrapper(obj, method_name, wrapper_function())
        _wrapped_methods.append((obj, method_name))
        logger.debug(f"Successfully wrapped {module_name}.{object_name}.{method_name}")
    except Exception as e:
        logger.warning(f"Could not wrap {module_name}.{object_name}.{method_name}: {e}")


def _patch_module_function(
    module_name: str, 
    function_name: str, 
    wrapper_function: Any
) -> None:
    """Helper to patch module-level functions.
    
    Args:
        module_name: The module containing the function.
        function_name: The function name to patch.
        wrapper_function: The wrapper function to apply.
    """
    try:
        module = __import__(module_name, fromlist=[function_name])
        wrapt.wrap_function_wrapper(module, function_name, wrapper_function())
        _wrapped_methods.append((module, function_name))
        logger.debug(f"Successfully wrapped {module_name}.{function_name}")
    except Exception as e:
        logger.warning(f"Could not wrap {module_name}.{function_name}: {e}")


def _patch_google_adk() -> None:
    """Apply all patches to Google ADK modules."""
    logger.debug("Applying Google ADK patches for Openlayer instrumentation")
    
    # First, disable ADK's own tracer by replacing it with our NoOpTracer
    noop_tracer = NoOpTracer()
    try:
        import google.adk.telemetry as adk_telemetry
        adk_telemetry.tracer = noop_tracer
        logger.debug("Replaced ADK's tracer with NoOpTracer")
    except Exception as e:
        logger.warning(f"Failed to replace ADK tracer: {e}")
    
    # Also replace the tracer in modules that have already imported it
    modules_to_patch = [
        "google.adk.runners",
        "google.adk.agents.base_agent",
        "google.adk.flows.llm_flows.base_llm_flow",
        "google.adk.flows.llm_flows.functions",
    ]
    
    for module_name in modules_to_patch:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                if hasattr(module, "tracer"):
                    module.tracer = noop_tracer
                    logger.debug(f"Replaced tracer in {module_name}")
            except Exception as e:
                logger.warning(f"Failed to replace tracer in {module_name}: {e}")
    
    # Patch agent execution
    _patch(
        "google.adk.agents.base_agent",
        "BaseAgent",
        "run_async",
        _base_agent_run_async_wrapper
    )
    
    # Patch LLM calls
    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_call_llm_async",
        _base_llm_flow_call_llm_async_wrapper
    )
    
    # Patch LLM response finalization
    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_finalize_model_response_event",
        _finalize_model_response_event_wrapper
    )
    
    # Patch tool execution
    _patch_module_function(
        "google.adk.flows.llm_flows.functions",
        "__call_tool_async",
        _call_tool_async_wrapper
    )
    
    logger.info("Google ADK patching complete")


def _unpatch_google_adk() -> None:
    """Remove all patches from Google ADK modules."""
    logger.debug("Removing Google ADK patches")
    
    # Restore ADK's tracer
    try:
        import google.adk.telemetry as adk_telemetry
        from opentelemetry import trace
        
        adk_telemetry.tracer = trace.get_tracer("gcp.vertex.agent")
        logger.debug("Restored ADK's built-in tracer")
    except Exception as e:
        logger.warning(f"Failed to restore ADK tracer: {e}")
    
    # Unwrap all methods
    for obj, method_name in _wrapped_methods:
        try:
            if hasattr(getattr(obj, method_name), "__wrapped__"):
                original = getattr(obj, method_name).__wrapped__
                setattr(obj, method_name, original)
                logger.debug(f"Successfully unwrapped {obj}.{method_name}")
        except Exception as e:
            logger.warning(f"Failed to unwrap {obj}.{method_name}: {e}")
    
    _wrapped_methods.clear()
    logger.info("Google ADK unpatching complete")

