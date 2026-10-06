"""Module with methods used to trace Google Agent Development Kit (ADK).

This module provides instrumentation for Google's Agent Development Kit (ADK),
capturing agent execution, LLM calls, tool calls, callbacks, and other
ADK-specific events.

Each ``Runner.run_async`` call (one user turn) becomes a single trace rooted at an
``Agent turn`` step. Every agent that runs during the turn is nested under it,
including agents reached through a transfer, which ADK 2.x runs in separate
asyncio tasks. Each transfer ADK makes is recorded as a Handoff step, and the
turn records ``starting_agent``, ``final_agent``, ``handoff_count`` and
``handoff_path`` as trace metadata.

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
import importlib
import json
import logging
import math
import sys
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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

from ..tracing import tracer, steps, enums, traces
from ..tracing.tracer import _rag_context as _tracer_rag_context
from ..tracing.tracer import _current_step as _tracer_current_step
from ..tracing.tracer import _current_trace as _tracer_current_trace
from ..tracing.tracer import _safe_reset_contextvar
from .google_genai_tracer import PROVIDER as GEMINI_PROVIDER
from .google_genai_tracer import _extract_usage as _extract_gemini_usage
from .google_genai_tracer import _normalize_model_name

logger = logging.getLogger(__name__)

# Store original callbacks for restoration
_original_callbacks: Dict[str, Any] = {}


# Track wrapped methods for cleanup
_wrapped_methods = []

# Module-level idempotency guard. _patch_google_adk() wraps many methods via
# wrapt; without this, a second call (e.g. a repeated init(auto_instrument=True))
# would stack a wrapper layer on every ADK method. Reset by _unpatch_google_adk().
_google_adk_patched = False

# Context variable to store the current user query across nested calls
_current_user_query: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "google_adk_user_query", default=None
)

# ADK's built-in agent-transfer tool. Handoffs are read from what ADK actually did
# (``event.actions.transfer_to_agent``), which also covers custom tools and
# callbacks that set that action. A call to this tool that did transfer is shown
# as the Handoff step only, not also as a Tool step.
ADK_TRANSFER_TOOL_NAME = "transfer_to_agent"

# Where ADK defines the function that runs a single tool call. It moved twice in
# google-adk 2.x; the first one that exists is patched.
_TOOL_CALL_TARGETS: List[Tuple[str, str]] = [
    ("google.adk.flows.llm_flows.tools._caller", "_call_tool_async"),  # google-adk >= 2.10
    ("google.adk.flows.llm_flows._tool_caller", "_call_tool_async"),  # google-adk 2.9
    ("google.adk.flows.llm_flows.functions", "__call_tool_async"),  # google-adk 1.x to 2.8
]


class _AdkTurn:
    """State for one ``Runner.run_async`` call (one user turn).

    ADK 2.x runs every agent node in its own asyncio task, so a ContextVar set
    inside one agent is invisible to the next. The ContextVar only holds a
    reference to this object, set before ADK creates any task, and every task
    mutates the same instance.

    A Runner started from inside a turn (AgentTool runs its agent that way) gets
    a nested scope: its handoffs are kept apart from the outer turn's, and it
    writes nothing to the turn step or the trace.
    """

    def __init__(
        self,
        step: Any,
        trace: Optional[traces.Trace],
        user_query: Optional[str] = None,
        nested: bool = False,
    ) -> None:
        self.step = step
        self.trace = trace
        self.user_query = user_query
        self.nested = nested
        # First agent that emitted content. On a resume, ADK 1.x re-enters the
        # root agent, which silently hands over to the paused sub-agent, so the
        # first agent entered (the fallback) isn't necessarily who started.
        self.starting_agent: Optional[str] = None
        self.first_entered_agent: Optional[str] = None
        self.final_agent: Optional[str] = None
        self.handoffs: List[Tuple[str, str]] = []
        # (agent, parent step) for agents a transfer handed control to that
        # haven't started yet. The parent is the transferring agent's parent, so
        # a transfer inside a workflow agent or an AgentTool stays nested there.
        self.pending_transfers: List[Tuple[str, Any]] = []
        self.root_attributes_set = False
        # This turn's position in _trace_turn_summaries[trace].
        self.trace_slot: Optional[int] = None

    def path(self) -> List[str]:
        starting_agent = self.starting_agent or self.first_entered_agent
        path = [starting_agent] if starting_agent else []
        path.extend(to_agent for _, to_agent in self.handoffs)
        return path

    def summary(self) -> Dict[str, Any]:
        path = self.path()
        return {
            "starting_agent": path[0] if path else None,
            # With no content event at all (e.g. an empty model reply), whoever
            # last received control is the best guess.
            "final_agent": self.final_agent or (path[-1] if path else None),
            "handoff_count": len(self.handoffs),
            "handoff_path": path,
        }


_current_turn: contextvars.ContextVar[Optional[_AdkTurn]] = contextvars.ContextVar("google_adk_turn", default=None)

# Turn summaries per trace, in order. A trace can hold several turns when the
# Runner is called more than once inside one @trace function; its columns then
# describe all of them. Values hold no reference to the trace, so entries go
# away with it.
_trace_turn_summaries: "weakref.WeakKeyDictionary[traces.Trace, List[Dict[str, Any]]]" = weakref.WeakKeyDictionary()

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

# Everything a turn sets in the context. The Runner wrapper swaps these in for
# each step of ADK's generator and back out before handing an event to the
# caller, so the caller's context never holds turn state while the turn is
# suspended (see _runner_run_async_wrapper).
_TURN_CONTEXT_VARS: Tuple["contextvars.ContextVar[Any]", ...] = (
    _tracer_current_step,
    _tracer_current_trace,
    _tracer_rag_context,
    _current_turn,
    _current_agent_step,
    _current_llm_step,
    _current_llm_request,
    _current_user_query,
)


def _snapshot_context() -> List[Any]:
    return [var.get(None) for var in _TURN_CONTEXT_VARS]


def _apply_context(values: List[Any]) -> None:
    for var, value in zip(_TURN_CONTEXT_VARS, values):
        var.set(value)


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
        Each ``Runner.run_async`` call is one trace, rooted at an
        ``Agent turn: <app_name>`` step. Every transfer ADK makes (through
        ``transfer_to_agent``, or a tool or callback that sets
        ``actions.transfer_to_agent``) becomes a ``Handoff: <from> → <to>``
        step, and the agent that receives control follows the transferring
        agent at the same level. ``starting_agent``, ``final_agent``,
        ``handoff_count`` and ``handoff_path`` are added to the trace metadata,
        so they can be used as columns in tests; when one ``@trace`` function
        runs several turns, they cover all of them. Agents used as tools
        (``AgentTool``) and workflow agents (``SequentialAgent``,
        ``ParallelAgent``, ``LoopAgent``) are nested agent steps, not handoffs.

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


@asynccontextmanager
async def _aclosing(agen: AsyncGenerator[Any, None]) -> AsyncIterator[AsyncGenerator[Any, None]]:
    """Close a wrapped ADK generator when our wrapper closes.

    Re-yielding without this leaves ADK's generator to the event loop's
    finalizer, which closes it from another task. ADK's OpenTelemetry spans then
    fail to detach their context and log "Failed to detach context".
    ``contextlib.aclosing`` needs Python 3.10.
    """
    try:
        yield agen
    finally:
        await agen.aclose()


def _content_text(content: Any) -> Optional[str]:
    """Join the text parts of a ``types.Content``; None when it has no text.

    Thought parts (the model's reasoning) are left out, as ADK does for
    ``output_key`` and AgentTool results. Gemini can answer STOP with a Content
    whose ``parts`` is None.
    """
    parts = getattr(content, "parts", None) or []
    text = "\n".join(
        part.text for part in parts if getattr(part, "text", None) and not getattr(part, "thought", None)
    ).strip()
    return text or None


def _reply_text(content: Any) -> Optional[str]:
    """The answer a final response carries, for agent and turn outputs.

    A final response can carry only a tool result: on google-adk 1.x an AgentTool
    with ``skip_summarization`` ends the turn on its function response.
    """
    text = _content_text(content)
    if text:
        return text
    for part in getattr(content, "parts", None) or []:
        function_response = getattr(part, "function_response", None)
        response = getattr(function_response, "response", None)
        if response is None:
            continue
        if isinstance(response, dict) and list(response) == ["result"]:
            response = response["result"]
        return response if isinstance(response, str) else json.dumps(response, default=str)
    return None


def _event_transfer_target(event: Any) -> Optional[str]:
    """The agent ADK hands control to after this event, if any."""
    target = getattr(getattr(event, "actions", None), "transfer_to_agent", None)
    return str(target) if target else None


def _response_json(response: Any) -> str:
    """Serialize an LLM response for ``raw_output``.

    ``inline_data`` bytes (generated images or audio) are left out, as on the
    request side. mode="json" base64-encodes the remaining bytes, such as the
    thought signatures on function calls.
    """
    content = getattr(response, "content", None)
    parts = getattr(content, "parts", None)
    if content is not None and parts and any(getattr(getattr(p, "inline_data", None), "data", None) for p in parts):
        parts = [
            part.model_copy(update={"inline_data": part.inline_data.model_copy(update={"data": None})})
            if getattr(getattr(part, "inline_data", None), "data", None)
            else part
            for part in parts
        ]
        response = response.model_copy(update={"content": content.model_copy(update={"parts": parts})})
    return json.dumps(response.model_dump(mode="json", exclude_none=True))


def _llm_provider(model_name: Optional[str]) -> str:
    """Return the cost-lookup provider slug for a model name.

    ``BaseLlmFlow._call_llm_async`` is model-agnostic (LiteLlm and
    Claude-on-Vertex agents reach it too), so only Gemini models get the
    "gemini" slug the backend prices with. Other models keep the old label.
    """
    if _normalize_model_name(model_name).startswith("gemini"):
        return GEMINI_PROVIDER
    return "Google"


def _record_step_error(step: Any, error: BaseException) -> None:
    """Record exception info on a step's metadata without overwriting output."""
    if step is None:
        return
    try:
        if getattr(step, "metadata", None) is None:
            step.metadata = {}
        step.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    except Exception:  # pragma: no cover - defensive: never break unwinding
        logger.debug("Failed to record error on step metadata", exc_info=True)


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


def _extract_llm_attributes(llm_request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract LLM attributes from a request.

    Args:
        llm_request_dict: Dictionary containing the LLM request data.

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

    return attributes


def _extract_tool_info(tool: Any) -> Optional[Dict[str, Any]]:
    """Extract info from a single tool entry in an ADK agent's tools list.

    ADK agents can have three kinds of tool entries:
    1. Raw callables (Python functions) — have ``__name__`` but not ``name``
    2. BaseTool subclass instances — have ``.name`` and ``.description``
    3. AgentTool instances — BaseTool with an ``.agent`` attribute wrapping
       another agent (the "agent-as-a-tool" pattern)

    For AgentTool, we recursively extract the wrapped agent's own tools
    so the trace shows the full tool hierarchy.

    Args:
        tool: A tool entry from an ADK agent's ``tools`` list.

    Returns:
        Dictionary with tool metadata, or None if the tool cannot be identified.
    """
    tool_info: Optional[Dict[str, Any]] = None

    # Case 1: AgentTool (must check before generic BaseTool)
    if hasattr(tool, "agent") and hasattr(tool, "name"):
        tool_info = {
            "name": tool.name,
            "type": "agent_tool",
        }
        if hasattr(tool, "description") and tool.description:
            tool_info["description"] = tool.description

        # Recursively extract the wrapped agent's tools
        wrapped_agent = tool.agent
        if hasattr(wrapped_agent, "tools") and wrapped_agent.tools:
            agent_tools = []
            for inner_tool in wrapped_agent.tools:
                inner_info = _extract_tool_info(inner_tool)
                if inner_info:
                    agent_tools.append(inner_info)
            if agent_tools:
                tool_info["agent_tools"] = agent_tools

    # Case 2: BaseTool subclass (has .name attribute set by BaseTool.__init__)
    elif hasattr(tool, "name") and tool.name:
        tool_info = {"name": tool.name}
        if hasattr(tool, "description") and tool.description:
            tool_info["description"] = tool.description

    # Case 3: Raw callable (plain Python function or lambda)
    elif callable(tool):
        name = getattr(tool, "__name__", None) or getattr(tool, "__qualname__", "unknown_tool")
        tool_info = {"name": name}
        doc = getattr(tool, "__doc__", None)
        if doc:
            tool_info["description"] = doc.strip().split("\n")[0]

    return tool_info


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
    # ADK agents store tools as a mix of raw callables (functions),
    # BaseTool instances, and AgentTool wrappers. We need to handle all three.
    if hasattr(instance, "tools") and instance.tools:
        tools_info = []
        for tool in instance.tools:
            tool_info = _extract_tool_info(tool)
            if tool_info:
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


def _runner_run_async_wrapper() -> Any:
    """Wrapper for Runner.run_async to make one trace per user turn.

    The turn step is set as the current step before ADK creates any asyncio
    task, so the tasks ADK 2.x spawns for each agent inherit it. It also holds
    the turn-level output and the handoff summary.

    The turn's context (current step and trace, the turn itself, ...) is only in
    place while ADK's generator runs. Before each event goes to the caller, the
    caller's own values are put back, and the turn's are put back in before
    asking ADK for the next event. Otherwise a caller that stops iterating early
    (``break`` on the final response) would keep the turn as its current step:
    asyncio closes an abandoned generator from another task, so the turn could
    never be unset, and every later turn or trace in that task would be nested
    into it. Swapping on every step also makes the turn independent of which
    task resumes it, e.g. one ``ensure_future(agen.__anext__())`` per event.

    Returns:
        Decorator function that wraps the original method.
    """

    def actual_decorator(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        async def new_function():
            # A Runner started inside a turn (AgentTool runs its agent through
            # one) is part of a tool call in the current turn, not a new turn.
            if _current_turn.get() is not None:
                async for event in _run_nested_runner(wrapped, args, kwargs):
                    yield event
                return

            app_name = getattr(instance, "app_name", None) or "google_adk"
            metadata: Dict[str, Any] = {"agent_type": "google_adk", "app_name": app_name}
            if kwargs.get("session_id"):
                metadata["session_id"] = kwargs["session_id"]
            if kwargs.get("user_id"):
                metadata["user_id"] = kwargs["user_id"]
            user_query = _content_text(kwargs.get("new_message"))

            caller = _snapshot_context()
            try:
                with tracer.create_step(
                    name=f"Agent turn: {app_name}",
                    step_type=enums.StepType.AGENT,
                    inputs={"user_query": user_query or "No query provided"},
                    metadata=metadata,
                ) as step:
                    turn = _AdkTurn(step, tracer.get_current_trace(), user_query=user_query)
                    _current_turn.set(turn)
                    inner = _snapshot_context()
                    _apply_context(caller)

                    async_gen = wrapped(*args, **kwargs)
                    error: Optional[BaseException] = None
                    try:
                        while True:
                            caller = _snapshot_context()
                            _apply_context(inner)
                            try:
                                event = await async_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            finally:
                                inner = _snapshot_context()
                                _apply_context(caller)
                            _record_turn_event(turn, event)
                            yield event
                    except BaseException as e:
                        # GeneratorExit when the caller stops early; anything
                        # else is a failure ADK or the caller raised.
                        error = e
                        raise
                    finally:
                        caller = _snapshot_context()
                        _apply_context(inner)
                        try:
                            await async_gen.aclose()
                        finally:
                            _finish_turn(turn, error)
                    # Leaving create_step with the turn's context in place
                    # completes the trace (and publishes it when the turn is the
                    # root step), whichever task or context this runs in.
            finally:
                _apply_context(caller)

        return new_function()

    return actual_decorator


async def _run_nested_runner(wrapped: Any, args: tuple, kwargs: dict) -> AsyncGenerator[Any, None]:
    """Run a Runner started inside a turn, with its own handoff scope.

    Its agents nest under the current step (the AgentTool's Tool step), and
    transfers between them aren't the outer turn's handoffs.
    """
    scope = _AdkTurn(
        tracer.get_current_step(),
        None,
        user_query=_content_text(kwargs.get("new_message")),
        nested=True,
    )
    token = _current_turn.set(scope)
    try:
        async with _aclosing(wrapped(*args, **kwargs)) as async_gen:
            async for event in async_gen:
                yield event
    finally:
        _safe_reset_contextvar(_current_turn, token)


def _record_turn_event(turn: _AdkTurn, event: Any) -> None:
    """Update the turn step from an event the Runner yields.

    The summary and end time are refreshed on every event: a caller that returns
    on the final response (e.g. a @trace handler) can publish its trace before
    the turn's generator is closed.
    """
    try:
        invocation_id = getattr(event, "invocation_id", None)
        if invocation_id and "invocation_id" not in turn.step.metadata:
            turn.step.metadata["invocation_id"] = invocation_id

        author = getattr(event, "author", None)
        if author and author != "user":
            content = getattr(event, "content", None)
            target = _event_transfer_target(event)
            if target:
                if turn.starting_agent is None:
                    turn.starting_agent = author
                # Control moved on, even if the target never says anything.
                turn.final_agent = target
            elif getattr(content, "parts", None):
                # Events with no parts are skipped: ADK emits empty closing events
                # from every agent on the way out (e.g. on resume), which don't
                # say who answered.
                if turn.starting_agent is None:
                    turn.starting_agent = author
                turn.final_agent = author
                is_final = getattr(event, "is_final_response", None)
                if callable(is_final) and is_final():
                    text = _reply_text(content)
                    if text:
                        turn.step.output = text
        _update_turn_step(turn)
    except Exception:  # pragma: no cover - defensive: never break the user's loop
        logger.debug("Failed to record a Google ADK turn event", exc_info=True)


def _update_turn_step(turn: _AdkTurn) -> None:
    """Write the handoff summary to the turn step and the trace's columns."""
    summary = turn.summary()
    step_summary = {**summary, "handoff_path": " > ".join(summary["handoff_path"])}
    turn.step.metadata.update(step_summary)
    now = time.time()
    turn.step.end_time = now
    turn.step.latency = (now - turn.step.start_time) * 1000

    # The turn step's metadata only becomes columns when it is the root step;
    # the trace metadata also covers turns run inside a user's @trace function.
    trace = turn.trace
    if trace is None:
        return
    summaries = _trace_turn_summaries.setdefault(trace, [])
    if turn.trace_slot is None:
        turn.trace_slot = len(summaries)
        summaries.append(summary)
    else:
        summaries[turn.trace_slot] = summary
    # Assigned directly: Trace.update_metadata skips None values, which would
    # keep a previous turn's value next to this one's.
    if trace.metadata is None:
        trace.metadata = {}
    trace.metadata.update(_combine_turn_summaries(summaries))


def _combine_turn_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Columns for a trace that holds one or more turns, in order."""
    path: List[str] = []
    for summary in summaries:
        for agent in summary["handoff_path"]:
            if not path or path[-1] != agent:
                path.append(agent)
    starting_agents = [s["starting_agent"] for s in summaries if s["starting_agent"]]
    final_agents = [s["final_agent"] for s in summaries if s["final_agent"]]
    return {
        "starting_agent": starting_agents[0] if starting_agents else None,
        "final_agent": final_agents[-1] if final_agents else None,
        "handoff_count": sum(s["handoff_count"] for s in summaries),
        "handoff_path": " > ".join(path),
    }


def _finish_turn(turn: _AdkTurn, error: Optional[BaseException]) -> None:
    """Close the turn step: output, error and the final summary."""
    try:
        if error is None or isinstance(error, GeneratorExit):
            if not turn.step.output:
                turn.step.output = "Agent execution completed"
        else:
            # Failed or cancelled: record why, and leave the output unset so the
            # row doesn't look like a completed turn.
            _record_step_error(turn.step, error)
        _update_turn_step(turn)
        _sort_steps_by_time(turn.step, recursive=True)
    except Exception:  # pragma: no cover - defensive: never break unwinding
        logger.debug("Failed to finish a Google ADK turn", exc_info=True)


def _record_handoff(
    turn: Optional[_AdkTurn],
    agent_step: Any,
    parent_step: Any,
    from_agent: str,
    to_agent: str,
    event: Any,
) -> None:
    """Record a transfer ADK made: a Handoff step under the transferring agent.

    Called for the event the transferring agent authors with
    ``actions.transfer_to_agent`` set, which is what ADK acts on. It covers the
    built-in ``transfer_to_agent`` tool, custom tools and callbacks that set the
    action, and several transfer calls in one reply (ADK keeps the last one).
    """
    metadata: Dict[str, Any] = {"tool_system": "google_adk"}
    invocation_id = getattr(event, "invocation_id", None)
    if invocation_id:
        metadata["invocation_id"] = invocation_id
    handoff = steps.step_factory(
        step_type=enums.StepType.HANDOFF,
        name=f"Handoff: {from_agent} → {to_agent}",
        inputs={"agent_name": to_agent},
        output={"transferred_to": to_agent},
        metadata=metadata,
    )
    handoff.start_time = handoff.end_time = time.time()
    handoff.latency = 0
    handoff.from_component = from_agent
    handoff.to_component = to_agent
    agent_step.add_nested_step(handoff)

    if turn is not None:
        turn.handoffs.append((from_agent, to_agent))
        turn.pending_transfers.append((to_agent, parent_step))


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

            # An agent that a transfer handed control to is nested under the
            # transferring agent's parent, not under whatever step is current
            # (on ADK 1.x that's still inside the transferring agent's step, on
            # 2.x it's wherever ADK's scheduler task was created).
            turn = _current_turn.get()
            transfer_parent = None
            if turn is not None:
                if turn.first_entered_agent is None:
                    turn.first_entered_agent = agent_name
                for index, (pending_agent, pending_parent) in enumerate(turn.pending_transfers):
                    if pending_agent == agent_name:
                        del turn.pending_transfers[index]
                        transfer_parent = pending_parent
                        break

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

            # The first agent of a turn describes it, as the root agent step did
            # before turns had their own step: its attributes are the turn's
            # input columns.
            if turn is not None and not turn.nested and not turn.root_attributes_set:
                turn.root_attributes_set = True
                turn.step.inputs = {**agent_attrs, "user_query": turn.step.inputs.get("user_query")}
                if has_callbacks:
                    turn.step.metadata["callbacks"] = has_callbacks

            # Every agent in a turn answers the message the Runner was given. The
            # LLM wrapper's last user message is only a fallback: on ADK 2.x a
            # transferred agent's is ADK's "For context: ..." transcript.
            turn_query = turn.user_query if turn is not None else None
            inputs = {**agent_attrs, "user_query": turn_query or "Processing..."}

            # The parent this agent's step is created under, which is also where
            # an agent it transfers to goes.
            parent_step = transfer_parent if transfer_parent is not None else tracer.get_current_step()

            transfer_token = None
            if transfer_parent is not None:
                logger.debug(f"Creating transferred agent step under its transferring agent's parent: {agent_name}")
                transfer_token = _tracer_current_step.set(transfer_parent)

            try:
                with tracer.create_step(
                    name=f"Agent: {agent_name}", step_type=enums.StepType.AGENT, inputs=inputs, metadata=metadata
                ) as step:
                    # Store the agent step so callbacks and tool calls can use it as
                    # parent. This keeps them siblings of LLM calls, not children.
                    agent_step_token = _current_agent_step.set(step)

                    user_query_updated = turn_query is not None
                    transferred = False
                    failed = False
                    try:
                        async with _aclosing(wrapped(*args, **kwargs)) as async_gen:
                            async for event in async_gen:
                                # Update user_query as soon as it's available from LLM wrapper
                                # This ensures it's captured even if generator is abandoned early
                                if not user_query_updated:
                                    captured_query = _current_user_query.get()
                                    if captured_query:
                                        step.inputs["user_query"] = captured_query
                                        user_query_updated = True

                                author = getattr(event, "author", None)
                                target = _event_transfer_target(event)
                                if target and author == agent_name:
                                    _record_handoff(turn, step, parent_step, agent_name, target, event)
                                    transferred = True
                                elif hasattr(event, "is_final_response") and event.is_final_response():
                                    # On ADK 1.x the agents this one transferred to
                                    # run inside it and their replies pass through
                                    # here; they aren't this agent's output. Replies
                                    # from its own sub-agents before any transfer
                                    # are (workflow and custom agents).
                                    if turn is None or not transferred or author == agent_name:
                                        final_response = _reply_text(getattr(event, "content", None))
                                        if final_response:
                                            step.output = final_response

                                yield event

                    except BaseException as e:
                        # GeneratorExit is ADK (or the caller) closing the
                        # generator early, which ADK 2.x does after a transfer.
                        if not isinstance(e, GeneratorExit):
                            failed = True
                            _record_step_error(step, e)
                            logger.debug("Agent execution raised; propagating: %s", e)
                        raise
                    finally:
                        # Fallbacks live here because ADK 2.x closes an agent's
                        # generator early (e.g. after a transfer), which skips any
                        # code after the loop.
                        if not user_query_updated:
                            captured_query = _current_user_query.get()
                            step.inputs["user_query"] = captured_query or "No query provided"
                        if not step.output and not failed:
                            step.output = "Agent execution completed"

                        # Sort all nested steps recursively by start_time to ensure chronological order
                        # This fixes the issue where callbacks appear after LLM calls/tools
                        # even though they executed before/after them
                        _sort_steps_by_time(step, recursive=True)

                        # Restore the enclosing agent step (None for a root agent)
                        _safe_reset_contextvar(_current_agent_step, agent_step_token)
            finally:
                # Restore the current step context if we changed it for transfer.
                # This must be done AFTER the with block exits.
                _safe_reset_contextvar(_tracer_current_step, transfer_token)

        return new_function()

    return actual_decorator


def _extract_usage_from_response(response: Any) -> Dict[str, Any]:
    """Extract token usage from an LLM response object.

    Args:
        response: The LLM response object (can be various types).

    Thinking tokens (``thoughts_token_count``) are counted as completion tokens,
    because Gemini bills them as output. Same accounting as the Gen AI tracer.
    Adapters for other APIs (LiteLlm, ApigeeLlm's chat-completions mode) already
    include reasoning tokens in ``candidates_token_count``; the totals show it,
    and then they aren't added again.

    Returns:
        Dictionary with prompt_tokens, completion_tokens, total_tokens, and
        "breakdown" (the raw token split, for step metadata).
    """
    usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "breakdown": {}}

    try:
        if hasattr(response, "usage_metadata"):
            usage_metadata = response.usage_metadata
        elif isinstance(response, dict):
            usage_metadata = response.get("usage_metadata")
        elif hasattr(response, "model_dump"):
            usage_metadata = response.model_dump().get("usage_metadata")
        else:
            usage_metadata = None

        if isinstance(usage_metadata, dict):
            from google.genai import types

            usage_metadata = types.GenerateContentResponseUsageMetadata.model_validate(usage_metadata)

        if usage_metadata:
            prompt_tokens, completion_tokens, total_tokens, breakdown = _extract_gemini_usage(usage_metadata)
            candidates_tokens = getattr(usage_metadata, "candidates_token_count", None) or 0
            tool_use_tokens = getattr(usage_metadata, "tool_use_prompt_token_count", None) or 0
            reported_total = getattr(usage_metadata, "total_token_count", None)
            if (
                completion_tokens > candidates_tokens
                and isinstance(reported_total, int)
                and prompt_tokens + candidates_tokens + tool_use_tokens >= reported_total
            ):
                completion_tokens = candidates_tokens
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = completion_tokens
            usage["total_tokens"] = total_tokens
            usage["breakdown"] = breakdown
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
                attrs = _extract_llm_attributes(llm_request_dict)
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
                # Set ChatCompletionStep attributes. The backend prices by an exact
                # (provider, model) match, so Gemini models get the "gemini" slug
                # and a bare model name (no Vertex resource path).
                step.provider = _llm_provider(model_name)
                step.model = _normalize_model_name(model_name) if step.provider == GEMINI_PROVIDER else model_name
                step.model_parameters = model_parameters

                # Store step in context for later updates (e.g., by callbacks)
                _current_llm_step.set(step)

                try:
                    # Execute LLM call
                    last_response = None

                    async with _aclosing(wrapped(*args, **kwargs)) as async_gen:
                        async for item in async_gen:
                            last_response = item
                            yield item

                    # Extract usage metadata from the last response
                    if last_response is not None:
                        usage = _extract_usage_from_response(last_response)
                        if usage["total_tokens"] > 0 or usage["prompt_tokens"] > 0:
                            step.prompt_tokens = usage["prompt_tokens"]
                            step.completion_tokens = usage["completion_tokens"]
                            step.tokens = usage["total_tokens"]
                            step.metadata.update(usage["breakdown"])
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
                                step.raw_output = _response_json(last_response)
                            elif isinstance(last_response, dict):
                                step.raw_output = json.dumps(last_response)
                        except Exception:
                            pass

                except Exception as e:
                    _record_step_error(step, e)
                    logger.debug("LLM call raised; propagating: %s", e)
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
    """Wrapper for ADK's tool-call function to create tool steps.

    Tool steps are nested under the calling agent's step, as siblings of its LLM
    calls. Handoffs are recorded from ADK's events instead (see
    ``_record_handoff``).

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

            # Nest under the agent step rather than the LLM call that requested
            # the tool. The platform doesn't store usage for an LLM step that has
            # child steps.
            parent_token = None
            agent_step = _current_agent_step.get()
            if agent_step is not None:
                parent_token = _tracer_current_step.set(agent_step)

            try:
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

                    except Exception as e:
                        _record_step_error(step, e)
                        logger.debug("Tool execution raised; propagating: %s", e)
                        raise
                    finally:
                        # Sort nested steps by start_time for correct chronological order
                        _sort_steps_by_time(step, recursive=True)
            finally:
                _safe_reset_contextvar(_tracer_current_step, parent_token)

            # A transfer_to_agent call that did transfer is shown by the Handoff
            # step the transferring agent records from ADK's event, so its Tool
            # step is dropped. A call that failed (e.g. no agent_name) stays.
            if tool_name == ADK_TRANSFER_TOOL_NAME and getattr(
                getattr(tool_context, "actions", None), "transfer_to_agent", None
            ):
                parent = agent_step if agent_step is not None else tracer.get_current_step()
                if parent is not None and step in parent.steps:
                    parent.steps.remove(step)
            return result

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
                    current_step.metadata.update(usage["breakdown"])

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
                inputs["usage"] = {key: value for key, value in usage.items() if key != "breakdown"}
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
    - All callbacks except the agent ones are placed at the Agent level, as
      siblings of LLM calls and Tool steps
    - before_model has its start_time moved just before the LLM call, because
      ADK invokes it after the LLM step has started. Tool callbacks already
      run before and after the tool step.

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
    # Model and tool callbacks → Agent step (siblings of LLM calls and Tool steps)
    use_agent_parent = callback_type in ("before_model", "after_model", "before_tool", "after_tool")

    # before_model runs after the LLM step has started, so its start_time is
    # moved just before the LLM call's.
    is_before_callback = callback_type == "before_model"

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
                    agent_step = _current_agent_step.get()
                    if agent_step is not None:
                        saved_token = _tracer_current_step.set(agent_step)
                    # Reference for timing is the current LLM step
                    reference_step = _current_llm_step.get()

                try:
                    # Create a step for the callback
                    with tracer.create_step(
                        name=f"Callback: {callback_name}",
                        step_type=enums.StepType.USER_CALL,
                        inputs=inputs,
                        metadata={"callback_type": callback_type, "is_callback": True},
                    ) as step:
                        # Move before_model just ahead of its LLM call when sorted
                        # by time. The smallest possible step keeps it after
                        # everything that really started earlier (e.g. the
                        # previous tool's after_tool callback).
                        if is_before_callback and reference_step is not None:
                            ref_start = getattr(reference_step, "start_time", None)
                            if ref_start is not None:
                                step.start_time = math.nextafter(ref_start, -math.inf)

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
                            _record_step_error(step, e)
                            raise
                finally:
                    # Restore the previous current step. Wrapped defensively
                    # because the await chain can cross asyncio Contexts and
                    # would otherwise raise a chained ValueError from this
                    # cleanup path during exception unwinding (OPEN-10343).
                    _safe_reset_contextvar(_tracer_current_step, saved_token)

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
                    agent_step = _current_agent_step.get()
                    if agent_step is not None:
                        saved_token = _tracer_current_step.set(agent_step)
                    # Reference for timing is the current LLM step
                    reference_step = _current_llm_step.get()

                try:
                    # Create a step for the callback
                    with tracer.create_step(
                        name=f"Callback: {callback_name}",
                        step_type=enums.StepType.USER_CALL,
                        inputs=inputs,
                        metadata={"callback_type": callback_type, "is_callback": True},
                    ) as step:
                        # Move before_model just ahead of its LLM call when sorted
                        # by time. The smallest possible step keeps it after
                        # everything that really started earlier (e.g. the
                        # previous tool's after_tool callback).
                        if is_before_callback and reference_step is not None:
                            ref_start = getattr(reference_step, "start_time", None)
                            if ref_start is not None:
                                step.start_time = math.nextafter(ref_start, -math.inf)

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
                            _record_step_error(step, e)
                            raise
                finally:
                    # Restore the previous current step. Wrapped defensively
                    # for symmetry with the async path; see OPEN-10343.
                    _safe_reset_contextvar(_tracer_current_step, saved_token)

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


def _patch_tool_execution() -> None:
    """Patch the function ADK runs each tool call through.

    It moved in google-adk 2.9 and 2.10 (see ``_TOOL_CALL_TARGETS``).
    ``functions.py`` re-exports the moved function, but ADK calls it from the
    module that defines it, so only that module can be patched. If it moves
    again, the re-export's ``__module__`` still names the defining module.
    """
    for module_name, function_name in _TOOL_CALL_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, function_name):
            _patch_module_function(module_name, function_name, _call_tool_async_wrapper)
            return
    try:
        functions = importlib.import_module("google.adk.flows.llm_flows.functions")
    except ImportError:
        functions = None
    reexported = getattr(functions, "_call_tool_async", None)
    defining_module = getattr(reexported, "__module__", None)
    if defining_module and hasattr(sys.modules.get(defining_module), "_call_tool_async"):
        _patch_module_function(defining_module, "_call_tool_async", _call_tool_async_wrapper)
        return
    logger.warning(
        "Could not find Google ADK's tool-call function (tried %s); tool calls and handoffs won't be traced.",
        ", ".join(f"{module}.{function}" for module, function in _TOOL_CALL_TARGETS),
    )


def _patch_google_adk() -> None:
    """Apply all patches to Google ADK modules.

    This function:
    - Optionally disables ADK's built-in OpenTelemetry tracing (if configured)
    - Patches the runner (Runner.run_async), one trace per user turn
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
    global _google_adk_patched
    if _google_adk_patched:
        logger.debug("Google ADK already patched; skipping (idempotent).")
        return

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

    # Patch the runner (one trace per user turn)
    _patch("google.adk.runners", "Runner", "run_async", _runner_run_async_wrapper)

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

    # Patch tool execution (including transfer_to_agent handoffs)
    _patch_tool_execution()

    _google_adk_patched = True

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
    global _disable_adk_otel_tracing, _google_adk_patched

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

    # Reset the flags so a subsequent trace_google_adk() re-patches cleanly
    _disable_adk_otel_tracing = False
    _google_adk_patched = False

    logger.info("Google ADK unpatching complete")
