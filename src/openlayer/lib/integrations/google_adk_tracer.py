"""Module with methods used to trace Google Agent Development Kit (ADK).

This module provides instrumentation for Google's Agent Development Kit (ADK),
capturing agent execution, LLM calls, tool calls, callbacks, and other
ADK-specific events.

The following callbacks are traced as Function Call steps:
- before_agent_callback: Called before the agent starts processing a request
- after_agent_callback: Called after the agent finishes processing a request
- before_model_callback: Called before each LLM model invocation
- after_model_callback: Called after each LLM model invocation
- before_tool_callback: Called before each tool execution
- after_tool_callback: Called after each tool execution

Reference:
    https://google.github.io/adk-docs/callbacks/#the-callback-mechanism-interception-and-control
"""

import asyncio
import contextvars
import json
import logging
import sys
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

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
from ..tracing.tracer import _current_step as _tracer_current_step

logger = logging.getLogger(__name__)

# Store original callbacks for restoration
_original_callbacks: Dict[str, Any] = {}


# Track wrapped methods for cleanup
_wrapped_methods = []

# Context variable to store the current user query across nested calls
_current_user_query: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "google_adk_user_query", default=None
)

# Context variable to track the parent step for agent transfers
# When set, sub-agent steps will be created as children of this step instead of the current step
_agent_transfer_parent_step: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "google_adk_transfer_parent", default=None
)

# Context variable to store the current LLM step for updating with response data
_current_llm_step: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("google_adk_llm_step", default=None)

# Context variable to store the current LLM request for callbacks
_current_llm_request: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "google_adk_llm_request", default=None
)

# Context variable to store the agent step for proper callback hierarchy
# Callbacks should be siblings of LLM calls, not children
_current_agent_step: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "google_adk_agent_step", default=None
)


# Configuration for whether to disable ADK's built-in OpenTelemetry tracing
# When False (default), ADK's OTel tracing works alongside Openlayer tracing
# When True, ADK's tracing is replaced with no-ops (legacy behavior)
_disable_adk_otel_tracing: bool = False


class NoOpSpan:
    """A no-op span that does nothing.

    This is used when users want to disable ADK's OpenTelemetry tracing
    and only use Openlayer's tracing.
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
    """A tracer that creates no-op spans.

    This is only used when users explicitly want to disable ADK's
    OpenTelemetry tracing via disable_adk_otel=True.
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


def trace_google_adk(disable_adk_otel: bool = False) -> None:
    """Enable tracing for Google Agent Development Kit (ADK).

    This function patches Google ADK to trace agent execution, LLM calls,
    and tool calls to Openlayer. It uses a global patching approach that
    automatically instruments all ADK agents created after this function
    is called.

    By default, ADK's built-in OpenTelemetry tracing remains active, allowing
    you to send telemetry to both Google Cloud (via ADK's OTel integration)
    and Openlayer simultaneously. This is useful when you want to use Google
    Cloud's tracing features (Cloud Trace, Cloud Monitoring, Cloud Logging)
    alongside Openlayer's observability platform.

    The following information is collected for each operation:
    - Agent execution: agent name, tools, handoffs, sub-agents
    - LLM calls: model, tokens (prompt, completion, total), messages, config
    - Tool calls: tool name, arguments, results
    - All 6 ADK callbacks: before_agent, after_agent, before_model, after_model,
      before_tool, after_tool
    - Start/end times and latency for all operations

    Args:
        disable_adk_otel: If True, disables ADK's built-in OpenTelemetry tracing.
            When False (default), ADK's OTel tracing works alongside Openlayer,
            allowing you to send data to both Google Cloud and Openlayer.
            Set to True only if you want Openlayer as your sole observability tool.

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

            # Enable tracing with ADK's OTel also active (default)
            # Data goes to both Google Cloud (if configured) and Openlayer
            trace_google_adk()

            # OR: Enable tracing with ONLY Openlayer (disable ADK's OTel)
            # trace_google_adk(disable_adk_otel=True)

            # Now create and run your ADK agents
            from google.adk.agents import Agent

            agent = Agent(name="Assistant", model="gemini-2.5-flash", instructions="You are a helpful assistant")

            result = await agent.run_async(...)
    """
    global _disable_adk_otel_tracing

    if not HAVE_GOOGLE_ADK:
        raise ImportError("google-adk library is not installed. Please install it with: pip install google-adk")

    if not HAVE_WRAPT:
        raise ImportError("wrapt library is not installed. Please install it with: pip install wrapt")

    _disable_adk_otel_tracing = disable_adk_otel

    if disable_adk_otel:
        logger.info("Enabling Google ADK tracing for Openlayer (ADK's OpenTelemetry tracing will be disabled)")
    else:
        logger.info(
            "Enabling Google ADK tracing for Openlayer (ADK's OpenTelemetry tracing remains active for Google Cloud)"
        )

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


def _sort_steps_by_time(step: Any, recursive: bool = True) -> None:
    """Sort nested steps by start_time for correct chronological order.

    This ensures that steps appear in the order they were executed,
    not the order they were created/added to the parent.

    Args:
        step: The step whose nested steps should be sorted.
        recursive: If True, also sort nested steps within children.
    """
    if not hasattr(step, "steps") or not step.steps:
        return

    # Sort by start_time
    step.steps.sort(key=lambda s: getattr(s, "start_time", 0) or 0)

    # Recursively sort children if requested
    if recursive:
        for child_step in step.steps:
            _sort_steps_by_time(child_step, recursive=True)


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
        parts = [part for part in content.parts if not hasattr(part, "inline_data") or not part.inline_data]
        result["contents"].append(types.Content(role=content.role, parts=parts).model_dump(exclude_none=True))

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


def _extract_llm_attributes(llm_request_dict: Dict[str, Any], llm_response: Optional[Any] = None) -> Dict[str, Any]:
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
            "prompt": [{"role": "system", "content": system_instruction}],
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
            response_dict = json.loads(llm_response) if isinstance(llm_response, str) else llm_response

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

    This wrapper:
    - Creates a AgentCallStep for the agent execution
    - Automatically wraps agent callbacks for tracing
    - Captures the final response and user query

    Returns:
        Decorator function that wraps the original method.
    """

    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        async def new_function():
            agent_name = instance.name if hasattr(instance, "name") else "Unknown Agent"

            # Wrap agent callbacks for tracing (if not already wrapped)
            _wrap_agent_callbacks(instance)

            # Check if this is a sub-agent being called via transfer
            transfer_parent = _agent_transfer_parent_step.get()

            # Reset the context variable for this agent execution (only for root agents)
            if transfer_parent is None:
                _current_user_query.set(None)

            # Extract invocation context for session/user IDs
            invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")

            # Build metadata with session info
            metadata = {"agent_type": "google_adk"}

            # Add callback info to metadata (all 6 ADK callback types)
            has_callbacks = []
            callback_attrs = [
                ("before_agent_callback", "before_agent"),
                ("after_agent_callback", "after_agent"),
                ("before_model_callback", "before_model"),
                ("after_model_callback", "after_model"),
                ("before_tool_callback", "before_tool"),
                ("after_tool_callback", "after_tool"),
            ]
            for attr, name in callback_attrs:
                if hasattr(instance, attr) and getattr(instance, attr):
                    has_callbacks.append(name)
            if has_callbacks:
                metadata["callbacks"] = has_callbacks

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
            inputs = {**agent_attrs, "user_query": "Processing..."}

            # If we're in a transfer, create the step as a child of the transfer parent
            # Otherwise, use normal context (child of current step)
            transfer_token = None
            if transfer_parent is not None:
                logger.debug(f"Creating sub-agent step as sibling: {agent_name}")
                # Temporarily set current step to transfer parent so new step becomes its child
                transfer_token = _tracer_current_step.set(transfer_parent)
                # Clear the transfer parent so nested steps work normally
                _agent_transfer_parent_step.set(None)

            step_cm = tracer.create_step(
                name=f"Agent: {agent_name}", step_type=enums.StepType.AGENT, inputs=inputs, metadata=metadata
            )

            # Use the step as a context manager and capture the actual step object
            # Note: The step is created when entering the with block with the correct parent
            with step_cm as step:
                # Store the agent step so callbacks can use it as parent
                # This ensures callbacks are siblings of LLM calls, not children
                _current_agent_step.set(step)

                try:
                    # Execute the agent
                    async_gen = wrapped(*args, **kwargs)
                    final_response = None
                    user_query_updated = False

                    async for event in async_gen:
                        # Update user_query as soon as it's available from LLM wrapper
                        # This ensures it's captured even if generator is abandoned early
                        if not user_query_updated:
                            captured_query = _current_user_query.get()
                            if captured_query:
                                step.inputs["user_query"] = captured_query
                                user_query_updated = True

                        # Extract final response from events
                        if hasattr(event, "is_final_response") and event.is_final_response():
                            if hasattr(event, "content") and event.content:
                                try:
                                    final_response = event.content.parts[0].text.strip()
                                    # Update step output IMMEDIATELY when captured
                                    # This ensures it's set even if generator is abandoned
                                    if final_response:
                                        step.output = final_response
                                except (AttributeError, IndexError):
                                    final_response = str(event.content)
                                    if final_response:
                                        step.output = final_response

                        yield event

                    # Fallback: Update user_query if not already set
                    if not user_query_updated:
                        captured_query = _current_user_query.get()
                        if captured_query:
                            step.inputs["user_query"] = captured_query
                        else:
                            step.inputs["user_query"] = "No query provided"

                    # Fallback: Set default output if none was captured
                    if not step.output:
                        step.output = "Agent execution completed"

                except Exception as e:
                    step.output = f"Error: {str(e)}"
                    logger.error(f"Error in agent execution: {e}")
                    raise
                finally:
                    # Sort all nested steps recursively by start_time to ensure chronological order
                    # This fixes the issue where callbacks appear after LLM calls/tools
                    # even though they executed before/after them
                    _sort_steps_by_time(step, recursive=True)
                    logger.debug(f"Sorted nested steps by start_time (recursive)")

                    # Clear the agent step context
                    _current_agent_step.set(None)

            # Restore the current step context if we changed it for transfer
            # This must be done AFTER the with block exits
            if transfer_token is not None:
                _tracer_current_step.reset(transfer_token)

        return new_function()

    return actual_decorator


def _extract_usage_from_response(response: Any) -> Dict[str, int]:
    """Extract token usage from an LLM response object.

    Args:
        response: The LLM response object (can be various types).

    Returns:
        Dictionary with prompt_tokens, completion_tokens, total_tokens.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        # Check if response has usage_metadata attribute directly
        if hasattr(response, "usage_metadata"):
            usage_metadata = response.usage_metadata
            if usage_metadata:
                usage["prompt_tokens"] = getattr(usage_metadata, "prompt_token_count", 0) or 0
                usage["completion_tokens"] = getattr(usage_metadata, "candidates_token_count", 0) or 0
                usage["total_tokens"] = getattr(usage_metadata, "total_token_count", 0) or 0

        # Check for dict-based response
        elif isinstance(response, dict):
            if "usage_metadata" in response:
                um = response["usage_metadata"]
                usage["prompt_tokens"] = um.get("prompt_token_count", 0) or 0
                usage["completion_tokens"] = um.get("candidates_token_count", 0) or 0
                usage["total_tokens"] = um.get("total_token_count", 0) or 0

        # Try to get from model_dump if available (Pydantic model)
        elif hasattr(response, "model_dump"):
            try:
                resp_dict = response.model_dump()
                if "usage_metadata" in resp_dict:
                    um = resp_dict["usage_metadata"]
                    usage["prompt_tokens"] = um.get("prompt_token_count", 0) or 0
                    usage["completion_tokens"] = um.get("candidates_token_count", 0) or 0
                    usage["total_tokens"] = um.get("total_token_count", 0) or 0
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Failed to extract usage metadata: {e}")

    return usage


def _extract_output_from_response(response: Any) -> Optional[str]:
    """Extract text output from an LLM response.

    Args:
        response: The LLM response object.

    Returns:
        Extracted text content or None.
    """
    try:
        # Check for content attribute with parts
        if hasattr(response, "content") and response.content:
            content = response.content
            if hasattr(content, "parts") and content.parts:
                text_parts = []
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(str(part.text))
                if text_parts:
                    return "\n".join(text_parts)

        # Check for dict-based response
        if isinstance(response, dict):
            if "content" in response and "parts" in response.get("content", {}):
                parts = response["content"]["parts"]
                text_parts = []
                for part in parts:
                    if "text" in part and part.get("text") is not None:
                        text_parts.append(str(part["text"]))
                if text_parts:
                    return "\n".join(text_parts)

        # Try model_dump
        if hasattr(response, "model_dump"):
            try:
                resp_dict = response.model_dump()
                if "content" in resp_dict and resp_dict["content"]:
                    content = resp_dict["content"]
                    if "parts" in content:
                        text_parts = []
                        for part in content["parts"]:
                            if "text" in part and part.get("text") is not None:
                                text_parts.append(str(part["text"]))
                        if text_parts:
                            return "\n".join(text_parts)
            except Exception:
                pass

        # Fallback to text attribute
        if hasattr(response, "text") and response.text:
            return str(response.text)

    except Exception as e:
        logger.debug(f"Failed to extract output from response: {e}")

    return None


def _base_llm_flow_call_llm_async_wrapper() -> Any:
    """Wrapper for BaseLlmFlow._call_llm_async to create LLM call steps.

    This wrapper:
    - Creates a ChatCompletionStep for the LLM call
    - Captures input messages and model parameters
    - Extracts usage metadata (tokens) from the response
    - Stores the step in context for callback access

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

            # Store request in context for callbacks
            _current_llm_request.set(llm_request)

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
                metadata=metadata,
            ) as step:
                # Set ChatCompletionStep attributes
                step.model = model_name
                step.provider = "Google"
                step.model_parameters = model_parameters

                # Store step in context for later updates (e.g., by callbacks)
                _current_llm_step.set(step)

                try:
                    # Execute LLM call
                    async_gen = wrapped(*args, **kwargs)
                    collected_responses = []
                    last_response = None

                    async for item in async_gen:
                        collected_responses.append(item)
                        last_response = item
                        yield item

                    # Extract usage metadata from the last response
                    if last_response is not None:
                        usage = _extract_usage_from_response(last_response)
                        if usage["total_tokens"] > 0 or usage["prompt_tokens"] > 0:
                            step.prompt_tokens = usage["prompt_tokens"]
                            step.completion_tokens = usage["completion_tokens"]
                            step.tokens = usage["total_tokens"]
                            logger.debug(
                                f"Captured token usage: prompt={usage['prompt_tokens']}, "
                                f"completion={usage['completion_tokens']}, "
                                f"total={usage['total_tokens']}"
                            )

                        # Extract output text
                        output_text = _extract_output_from_response(last_response)
                        if output_text:
                            step.output = output_text

                        # Store raw response for debugging
                        try:
                            if hasattr(last_response, "model_dump"):
                                step.raw_output = json.dumps(last_response.model_dump(exclude_none=True))
                            elif isinstance(last_response, dict):
                                step.raw_output = json.dumps(last_response)
                        except Exception:
                            pass

                except Exception as e:
                    step.output = f"Error: {str(e)}"
                    logger.error(f"Error in LLM call: {e}")
                    raise
                finally:
                    # Sort nested steps by start_time for correct chronological order
                    _sort_steps_by_time(step, recursive=True)

                    # Clear context variables
                    _current_llm_step.set(None)
                    _current_llm_request.set(None)

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
                if current_step and hasattr(current_step, "parent_step") and current_step.parent_step:
                    # Set the LLM call's parent as the transfer parent
                    _agent_transfer_parent_step.set(current_step.parent_step)
                    logger.debug(
                        f"Set transfer parent to: {current_step.parent_step.name if hasattr(current_step.parent_step, 'name') else 'unknown'}"
                    )

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
                name=f"Tool: {tool_name}", step_type=enums.StepType.TOOL, inputs=tool_args, metadata=metadata
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
                finally:
                    # Sort nested steps by start_time for correct chronological order
                    _sort_steps_by_time(step, recursive=True)

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

        # Extract response data and update step if we have one
        llm_response = args[1] if len(args) > 1 else kwargs.get("llm_response")
        current_step = _current_llm_step.get()

        if current_step is not None and llm_response is not None:
            try:
                # Extract and update usage metadata
                usage = _extract_usage_from_response(llm_response)
                if usage["total_tokens"] > 0 or usage["prompt_tokens"] > 0:
                    current_step.prompt_tokens = usage["prompt_tokens"]
                    current_step.completion_tokens = usage["completion_tokens"]
                    current_step.tokens = usage["total_tokens"]

                # Extract and update output if not already set
                if not current_step.output:
                    output_text = _extract_output_from_response(llm_response)
                    if output_text:
                        current_step.output = output_text
            except Exception as e:
                logger.debug(f"Error updating step from finalize: {e}")

        return result

    return actual_decorator


# ----------------------------- Callback Wrappers ----------------------------- #


def _extract_callback_inputs(callback_type: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract inputs for a callback based on its type.

    Args:
        callback_type: Type of callback (before_agent, after_agent, before_model,
            after_model, before_tool, after_tool).
        args: Positional arguments passed to the callback.
        kwargs: Keyword arguments passed to the callback.

    Returns:
        Dictionary of inputs for tracing.
    """
    inputs: Dict[str, Any] = {}

    # Extract callback_context (first arg for most callbacks)
    callback_context = args[0] if args else kwargs.get("callback_context")
    if callback_context:
        if hasattr(callback_context, "agent_name"):
            inputs["agent_name"] = callback_context.agent_name
        if hasattr(callback_context, "invocation_id"):
            inputs["invocation_id"] = callback_context.invocation_id
        if hasattr(callback_context, "state") and callback_context.state:
            # Include a subset of state keys for debugging
            try:
                state_keys = list(callback_context.state.keys())[:10]
                inputs["state_keys"] = state_keys
            except Exception:
                pass

    # Type-specific extraction
    if callback_type == "before_agent":
        # before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]
        pass  # callback_context already extracted above

    elif callback_type == "after_agent":
        # after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]
        pass  # callback_context already extracted above

    elif callback_type == "before_model":
        # before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest)
        #   -> Optional[LlmResponse]
        llm_request = args[1] if len(args) > 1 else kwargs.get("llm_request")
        if llm_request:
            if hasattr(llm_request, "model"):
                inputs["model"] = llm_request.model
            if hasattr(llm_request, "config"):
                try:
                    inputs["config"] = llm_request.config.model_dump(exclude_none=True, exclude="response_schema")
                except Exception:
                    pass

    elif callback_type == "after_model":
        # after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse)
        #   -> Optional[LlmResponse]
        llm_response = args[1] if len(args) > 1 else kwargs.get("llm_response")
        if llm_response:
            # Extract usage from response
            usage = _extract_usage_from_response(llm_response)
            if usage["total_tokens"] > 0:
                inputs["usage"] = usage
            # Extract output text
            output_text = _extract_output_from_response(llm_response)
            if output_text:
                inputs["response_preview"] = output_text[:200] + "..." if len(output_text) > 200 else output_text

    elif callback_type == "before_tool":
        # before_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext)
        #   -> Optional[dict]
        tool = args[0] if args else kwargs.get("tool")
        tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})
        tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")

        if tool:
            if hasattr(tool, "name"):
                inputs["tool_name"] = tool.name
            if hasattr(tool, "description"):
                inputs["tool_description"] = tool.description
        if tool_args:
            inputs["tool_args"] = tool_args
        if tool_context and hasattr(tool_context, "function_call_id"):
            inputs["function_call_id"] = tool_context.function_call_id

    elif callback_type == "after_tool":
        # after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext,
        #   tool_response: dict) -> Optional[dict]
        tool = args[0] if args else kwargs.get("tool")
        tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})
        tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")
        tool_response = args[3] if len(args) > 3 else kwargs.get("tool_response")

        if tool and hasattr(tool, "name"):
            inputs["tool_name"] = tool.name
        if tool_args:
            inputs["tool_args"] = tool_args
        if tool_response:
            # Include a preview of the response
            try:
                if isinstance(tool_response, dict):
                    inputs["tool_response"] = tool_response
                else:
                    response_str = str(tool_response)
                    inputs["tool_response_preview"] = (
                        response_str[:200] + "..." if len(response_str) > 200 else response_str
                    )
            except Exception:
                pass

    return inputs


def _create_callback_wrapper(callback_name: str, callback_type: str) -> Callable:
    """Create a wrapper function for ADK callbacks.

    This creates a wrapper that traces callback execution as a Function Call step.

    Callback hierarchy and timing:
    - Model callbacks (before_model, after_model) are placed at the Agent level
      as siblings of LLM calls
    - Tool callbacks (before_tool, after_tool) are placed at the LLM level
      as siblings of Tool steps
    - "before_*" callbacks have their start_time adjusted to appear before
      their associated operation when sorted

    Supported callback types:
    - before_agent: Called before the agent starts processing
    - after_agent: Called after the agent finishes processing
    - before_model: Called before each LLM model invocation
    - after_model: Called after each LLM model invocation
    - before_tool: Called before each tool execution
    - after_tool: Called after each tool execution

    Reference:
        https://google.github.io/adk-docs/callbacks/#the-callback-mechanism-interception-and-control

    Args:
        callback_name: Human-readable name for the callback.
        callback_type: Type of callback.

    Returns:
        A wrapper function that traces the callback.
    """
    # Determine the parent step for this callback:
    # - Model callbacks (before_model, after_model) → Agent step (siblings of LLM calls)
    # - Tool callbacks (before_tool, after_tool) → LLM step (siblings of Tool steps)
    use_agent_parent = callback_type in ("before_model", "after_model")
    use_llm_parent = callback_type in ("before_tool", "after_tool")

    # "before_*" callbacks need their start_time adjusted to appear before
    # their associated operation (since they're actually called after the operation starts)
    is_before_callback = callback_type.startswith("before_")

    def wrapper(original_callback: Callable) -> Callable:
        """Wrap the original callback with tracing."""
        if original_callback is None:
            return None

        # Handle async callbacks
        if asyncio.iscoroutinefunction(original_callback):

            async def async_traced_callback(*args, **kwargs):
                # Extract inputs based on callback type
                inputs = _extract_callback_inputs(callback_type, args, kwargs)

                # Determine the parent step and get reference time for ordering
                saved_token = None
                reference_step = None

                if use_agent_parent:
                    # Model callbacks → Agent step (siblings of LLM calls)
                    agent_step = _current_agent_step.get()
                    if agent_step is not None:
                        saved_token = _tracer_current_step.set(agent_step)
                    # Reference for timing is the current LLM step
                    reference_step = _current_llm_step.get()
                elif use_llm_parent:
                    # Tool callbacks → LLM step (siblings of Tool steps)
                    llm_step = _current_llm_step.get()
                    if llm_step is not None:
                        saved_token = _tracer_current_step.set(llm_step)
                        reference_step = llm_step

                try:
                    # Create a step for the callback
                    with tracer.create_step(
                        name=f"Callback: {callback_name}",
                        step_type=enums.StepType.USER_CALL,
                        inputs=inputs,
                        metadata={"callback_type": callback_type, "is_callback": True},
                    ) as step:
                        # Adjust start_time for "before_*" callbacks to appear before
                        # their associated operation when sorted by time
                        if is_before_callback and reference_step is not None:
                            ref_start = getattr(reference_step, "start_time", None)
                            if ref_start is not None:
                                # Set start_time to be 1ms before the reference operation
                                step.start_time = ref_start - 0.001

                        try:
                            result = await original_callback(*args, **kwargs)

                            # Set output based on result
                            if result is not None:
                                if hasattr(result, "model_dump"):
                                    try:
                                        step.output = result.model_dump(exclude_none=True)
                                    except Exception:
                                        step.output = str(result)
                                elif isinstance(result, dict):
                                    step.output = result
                                else:
                                    step.output = str(result)
                            else:
                                step.output = "Callback completed (no modification)"

                            return result
                        except Exception as e:
                            step.output = f"Error: {str(e)}"
                            raise
                finally:
                    # Restore the previous current step
                    if saved_token is not None:
                        _tracer_current_step.reset(saved_token)

            return async_traced_callback
        else:
            # Handle sync callbacks
            def sync_traced_callback(*args, **kwargs):
                # Extract inputs based on callback type
                inputs = _extract_callback_inputs(callback_type, args, kwargs)

                # Determine the parent step and get reference time for ordering
                saved_token = None
                reference_step = None

                if use_agent_parent:
                    # Model callbacks → Agent step (siblings of LLM calls)
                    agent_step = _current_agent_step.get()
                    if agent_step is not None:
                        saved_token = _tracer_current_step.set(agent_step)
                    # Reference for timing is the current LLM step
                    reference_step = _current_llm_step.get()
                elif use_llm_parent:
                    # Tool callbacks → LLM step (siblings of Tool steps)
                    llm_step = _current_llm_step.get()
                    if llm_step is not None:
                        saved_token = _tracer_current_step.set(llm_step)
                        reference_step = llm_step

                try:
                    # Create a step for the callback
                    with tracer.create_step(
                        name=f"Callback: {callback_name}",
                        step_type=enums.StepType.USER_CALL,
                        inputs=inputs,
                        metadata={"callback_type": callback_type, "is_callback": True},
                    ) as step:
                        # Adjust start_time for "before_*" callbacks to appear before
                        # their associated operation when sorted by time
                        if is_before_callback and reference_step is not None:
                            ref_start = getattr(reference_step, "start_time", None)
                            if ref_start is not None:
                                # Set start_time to be 1ms before the reference operation
                                step.start_time = ref_start - 0.001

                        try:
                            result = original_callback(*args, **kwargs)

                            # Set output based on result
                            if result is not None:
                                if hasattr(result, "model_dump"):
                                    try:
                                        step.output = result.model_dump(exclude_none=True)
                                    except Exception:
                                        step.output = str(result)
                                elif isinstance(result, dict):
                                    step.output = result
                                else:
                                    step.output = str(result)
                            else:
                                step.output = "Callback completed (no modification)"

                            return result
                        except Exception as e:
                            step.output = f"Error: {str(e)}"
                            raise
                finally:
                    # Restore the previous current step
                    if saved_token is not None:
                        _tracer_current_step.reset(saved_token)

            return sync_traced_callback

    return wrapper


def _wrap_agent_callbacks(agent: Any) -> None:
    """Wrap an agent's callbacks with tracing wrappers.

    This function wraps all 6 ADK callback types on an agent instance:
    - before_agent_callback: Called before the agent starts processing
    - after_agent_callback: Called after the agent finishes processing
    - before_model_callback: Called before each LLM model invocation
    - after_model_callback: Called after each LLM model invocation
    - before_tool_callback: Called before each tool execution
    - after_tool_callback: Called after each tool execution

    Reference:
        https://google.github.io/adk-docs/callbacks/#the-callback-mechanism-interception-and-control

    Args:
        agent: The ADK agent instance to wrap callbacks for.
    """
    agent_name = getattr(agent, "name", "unknown")
    agent_id = id(agent)

    # Define all callback types to wrap
    callback_configs = [
        ("before_agent_callback", "before_agent"),
        ("after_agent_callback", "after_agent"),
        ("before_model_callback", "before_model"),
        ("after_model_callback", "after_model"),
        ("before_tool_callback", "before_tool"),
        ("after_tool_callback", "after_tool"),
    ]

    for callback_attr, callback_type in callback_configs:
        if hasattr(agent, callback_attr):
            original = getattr(agent, callback_attr)
            if original is not None and not getattr(original, "_openlayer_wrapped", False):
                wrapper = _create_callback_wrapper(f"{callback_type.replace('_', ' ')} ({agent_name})", callback_type)
                wrapped = wrapper(original)
                wrapped._openlayer_wrapped = True
                wrapped._openlayer_original = original
                setattr(agent, callback_attr, wrapped)
                _original_callbacks[f"{agent_id}_{callback_type}"] = original
                logger.debug(f"Wrapped {callback_attr} for agent: {agent_name}")

    # Recursively wrap sub-agents
    if hasattr(agent, "sub_agents") and agent.sub_agents:
        for sub_agent in agent.sub_agents:
            _wrap_agent_callbacks(sub_agent)


def _unwrap_agent_callbacks(agent: Any) -> None:
    """Remove callback wrappers from an agent.

    Args:
        agent: The ADK agent instance to unwrap callbacks for.
    """
    agent_id = id(agent)

    # All callback attribute names
    callback_attrs = [
        "before_agent_callback",
        "after_agent_callback",
        "before_model_callback",
        "after_model_callback",
        "before_tool_callback",
        "after_tool_callback",
    ]

    # Restore original callbacks
    for callback_name in callback_attrs:
        if hasattr(agent, callback_name):
            callback = getattr(agent, callback_name)
            if callback and hasattr(callback, "_openlayer_original"):
                setattr(agent, callback_name, callback._openlayer_original)

    # Clean up stored originals
    for key in list(_original_callbacks.keys()):
        if key.startswith(f"{agent_id}_"):
            del _original_callbacks[key]

    # Recursively unwrap sub-agents
    if hasattr(agent, "sub_agents") and agent.sub_agents:
        for sub_agent in agent.sub_agents:
            _unwrap_agent_callbacks(sub_agent)


# ----------------------------- Patching Functions --------------------------- #


def _patch(module_name: str, object_name: str, method_name: str, wrapper_function: Any) -> None:
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


def _patch_module_function(module_name: str, function_name: str, wrapper_function: Any) -> None:
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
    """Apply all patches to Google ADK modules.

    This function:
    - Optionally disables ADK's built-in OpenTelemetry tracing (if configured)
    - Patches agent execution (run_async)
    - Patches LLM calls (_call_llm_async)
    - Patches LLM response finalization
    - Patches tool execution

    By default, ADK's OpenTelemetry tracing remains active, allowing users
    to send telemetry to both Google Cloud and Openlayer. ADK uses OTel
    exporters configured via google.adk.telemetry.get_gcp_exporters() or
    standard OTEL_EXPORTER_OTLP_* environment variables.

    Callbacks (before_model, after_model, before_tool) are wrapped
    dynamically when agents run, not through static patching.

    Reference:
        ADK Telemetry: https://github.com/google/adk-python/tree/main/src/google/adk/telemetry
    """
    logger.debug("Applying Google ADK patches for Openlayer instrumentation")

    # Only disable ADK's tracer if explicitly requested
    # By default, keep ADK's OTel tracing active for Google Cloud integration
    if _disable_adk_otel_tracing:
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
    else:
        logger.debug(
            "Keeping ADK's OpenTelemetry tracing active. "
            "Telemetry will be sent to both Google Cloud (if configured) and Openlayer."
        )

    # Patch agent execution
    _patch("google.adk.agents.base_agent", "BaseAgent", "run_async", _base_agent_run_async_wrapper)

    # Patch LLM calls
    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_call_llm_async",
        _base_llm_flow_call_llm_async_wrapper,
    )

    # Patch LLM response finalization
    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_finalize_model_response_event",
        _finalize_model_response_event_wrapper,
    )

    # Patch tool execution
    _patch_module_function("google.adk.flows.llm_flows.functions", "__call_tool_async", _call_tool_async_wrapper)

    if _disable_adk_otel_tracing:
        logger.info("Google ADK patching complete. ADK's OTel tracing disabled, using Openlayer only.")
    else:
        logger.info(
            "Google ADK patching complete. ADK's OTel tracing active (Google Cloud) + Openlayer tracing enabled."
        )


def _unpatch_google_adk() -> None:
    """Remove all patches from Google ADK modules.

    This function:
    - Restores ADK's built-in OpenTelemetry tracing (if it was disabled)
    - Removes all method patches
    - Clears stored original callbacks
    """
    global _disable_adk_otel_tracing

    logger.debug("Removing Google ADK patches")

    # Restore ADK's tracer only if we disabled it
    if _disable_adk_otel_tracing:
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

    # Clear stored original callbacks
    _original_callbacks.clear()

    # Reset the flag
    _disable_adk_otel_tracing = False

    logger.info("Google ADK unpatching complete")
