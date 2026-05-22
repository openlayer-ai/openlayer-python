"""Openlayer tracing integration for the Claude Agent SDK.

Wraps ``claude_agent_sdk.query`` (and ``ClaudeSDKClient``) so each call becomes
an Openlayer trace with nested steps for assistant turns, tool calls, and
subagents.

The wrapper is a pure *observer* of the stream: every message yielded by the
underlying ``query()`` is forwarded to the caller unchanged and in order. We
emit Openlayer steps as a side effect of observation.

See ``docs/superpowers/specs/2026-05-12-claude-agent-sdk-integration-design.md``
for the design rationale.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from dataclasses import dataclass, field, replace as _dataclass_replace
from typing import Any, AsyncIterator, Dict, List, Optional

from ..tracing import tracer as _tracer
from ..tracing.enums import StepType

logger = logging.getLogger(__name__)

# We do NOT import ``claude_agent_sdk`` at module load time so this file is
# importable even when the optional dependency is absent — matching the
# pattern of every other integration in this directory.
try:
    import claude_agent_sdk as _cas  # type: ignore[import-not-found]

    HAVE_CLAUDE_AGENT_SDK = True
except ImportError:  # pragma: no cover - exercised by the optional-dep tests
    _cas = None  # type: ignore[assignment]
    HAVE_CLAUDE_AGENT_SDK = False


# ---------------------------- Configuration ---------------------------- #


@dataclass
class _Config:
    """Module-level configuration set by ``trace_claude_agent_sdk``."""

    inference_pipeline_id: Optional[str] = None
    truncate_tool_output_chars: int = 8192
    capture_thinking: bool = True
    redact_mcp_env: bool = True


_config = _Config()


# ----------------------------- Trace state ----------------------------- #


@dataclass
class _TraceState:
    """Per-query trace state.

    Holds the root step and per-tool-use bookkeeping used to nest subagent
    messages and to bracket tool calls across PreToolUse / PostToolUse hooks.
    """

    root_step: Any
    pending_tools: Dict[str, Any] = field(default_factory=dict)
    tool_to_parent_step: Dict[str, Any] = field(default_factory=dict)
    turn_counter: int = 0
    session_id: Optional[str] = None
    model: Optional[str] = None
    user_prompt: Optional[str] = None
    # Lookup of subagent definitions (built from ``options.agents``) so the
    # PreToolUse hook can attach a subagent's prompt/tools/model to the
    # AGENT step opened for its dispatch.
    agents_defined: Optional[Dict[str, Any]] = None


_current_state: contextvars.ContextVar[Optional[_TraceState]] = contextvars.ContextVar(
    "openlayer_claude_agent_sdk_state", default=None
)


def _require_cas() -> None:
    """Raise ``ImportError`` if the optional SDK is not installed."""
    if not HAVE_CLAUDE_AGENT_SDK:
        raise ImportError(
            "claude-agent-sdk is not installed. "
            "Install with: pip install 'claude-agent-sdk>=0.1.81'"
        )


# ------------------------------ Helpers ------------------------------ #


def _redact_mcp_servers(mcp_servers: Any) -> Any:
    """Strip ``env`` and ``headers`` from MCP server config dicts.

    These typically contain credentials and must not be persisted.
    """
    if not mcp_servers:
        return mcp_servers
    if isinstance(mcp_servers, list):
        return [
            {k: v for k, v in s.items() if k not in {"env", "headers"}}
            if isinstance(s, dict)
            else s
            for s in mcp_servers
        ]
    return mcp_servers


_ROOT_STEP_NAME = "Claude Agent SDK query"


def _serialize_system_prompt(sp: Any) -> Any:
    """Coerce ``options.system_prompt`` (str | preset dict | dataclass) into a
    JSON-serializable value, truncated to 4096 chars for string forms."""
    if sp is None:
        return None
    if isinstance(sp, str):
        return _truncate(sp, 4096)
    if isinstance(sp, dict):
        return sp
    # Preset / file dataclass — best-effort attribute pluck
    keys = ("type", "preset", "append", "excludeDynamicSections", "path")
    out = {k: getattr(sp, k) for k in keys if hasattr(sp, k)}
    return out or str(sp)


def _serialize_agent_definitions(agents: Any) -> Any:
    """``options.agents`` is a dict[str, AgentDefinition]. Capture each
    definition's description, prompt (truncated), and tools list."""
    if not agents:
        return None
    out: Dict[str, Any] = {}
    for name, defn in agents.items():
        out[name] = {
            "description": getattr(defn, "description", None),
            "prompt": _truncate(getattr(defn, "prompt", None), 4096),
            "tools": getattr(defn, "tools", None),
            "model": getattr(defn, "model", None),
        }
    return out


def _capture_options_metadata(root_step: Any, options: Any) -> None:
    """Snapshot user-provided options onto the root step metadata.

    Captures the user's ``system_prompt``, subagent definitions, and a handful
    of other useful configuration values. Called once per query, with the
    *original* (pre-hook-injection) options so we record what the user passed,
    not what we forwarded to the SDK.
    """
    if options is None:
        return
    metadata: Dict[str, Any] = {}

    sp = getattr(options, "system_prompt", None)
    serialized_sp = _serialize_system_prompt(sp)
    if serialized_sp is not None:
        metadata["system_prompt"] = serialized_sp

    agents = _serialize_agent_definitions(getattr(options, "agents", None))
    if agents:
        metadata["agents_defined"] = agents

    opt_capture: Dict[str, Any] = {}
    for opt in (
        "model",
        "fallback_model",
        "max_turns",
        "max_budget_usd",
        "permission_mode",
        "cwd",
        "allowed_tools",
        "disallowed_tools",
        "continue_conversation",
        "resume",
        "fork_session",
    ):
        val = getattr(options, opt, None)
        if val is None or val == [] or val == {}:
            continue
        # Convert Path to str
        opt_capture[opt] = str(val) if hasattr(val, "__fspath__") else val
    if opt_capture:
        metadata["options"] = opt_capture

    if metadata:
        root_step.log(metadata=metadata)


def _truncate(value: Any, max_chars: int) -> Any:
    """Coerce ``value`` to a string and truncate to ``max_chars``.

    Tool outputs can be arbitrary objects; we serialize via ``json.dumps`` when
    possible, falling back to ``str()``. The original length is preserved in
    the truncation marker so downstream UI can hint at omission.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = str(value)
    if len(value) > max_chars:
        return value[:max_chars] + f"... [truncated, full length {len(value)}]"
    return value


def _parse_mcp_metadata(tool_name: str) -> Dict[str, Any]:
    """Parse ``mcp__<server>__<tool>`` tool names into server/tool metadata."""
    if not tool_name or not tool_name.startswith("mcp__"):
        return {}
    parts = tool_name.split("__", 2)
    if len(parts) == 3:
        return {"mcp_server": parts[1], "mcp_tool_name": parts[2]}
    return {}


# --------------------------- Tool-step lifecycle --------------------------- #


class _ToolStepHandle:
    """Holds the live ``create_step`` context manager for an in-flight tool call.

    The Openlayer tracer exposes step lifecycle as a context manager
    (``tracer.create_step``). For tools we need to open the step in
    ``PreToolUse`` and close it from ``PostToolUse`` (which may execute on a
    different stack frame within the same coroutine). We bypass ``with`` by
    invoking ``__enter__`` / ``__exit__`` manually.

    Caveat: between ``__enter__`` and ``__exit__`` the step is "current" on the
    contextvar stack. Sequential pre/post pairs per ``tool_use_id`` are well-
    behaved, but truly concurrent tool calls could interleave and produce
    nested-looking steps. The SDK fires pre/post serially per tool, so this is
    acceptable for the v1 integration.
    """

    def __init__(self, step_cm):
        self._cm = step_cm
        self.step = step_cm.__enter__()
        self.start_time = time.time()
        self._closed = False

    def log(self, **kwargs: Any) -> None:
        self.step.log(**kwargs)

    def end(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Ensure latency is recorded even if PostToolUse fires before any
        # nested children would set it via the tracer's natural path.
        if self.step.latency is None:
            self.step.latency = (time.time() - self.start_time) * 1000.0
        self._cm.__exit__(None, None, None)


# ------------------------------- Hooks ------------------------------- #


async def _pre_tool_use_hook(input_data, tool_use_id, context):  # noqa: D401
    """Composed PreToolUse hook — opens a TOOL step for this tool call."""
    state = _current_state.get()
    if state is None or tool_use_id is None:
        return {}
    try:
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input") or {}
        metadata: Dict[str, Any] = {"tool_use_id": tool_use_id}
        metadata.update(_parse_mcp_metadata(tool_name))

        # The built-in ``Agent`` tool is how the SDK dispatches subagents. We
        # represent those calls as AGENT steps (not TOOL) so the subagent's
        # nested chat/tool steps live inside an agent boundary in the trace.
        # The dispatched subagent's name typically rides in ``subagent_type``.
        is_subagent_dispatch = tool_name == "Agent"
        step_type = StepType.AGENT if is_subagent_dispatch else StepType.TOOL
        if is_subagent_dispatch and isinstance(tool_input, dict):
            subagent_type = tool_input.get("subagent_type")
            description = tool_input.get("description")
            display_name = (
                f"Agent: {subagent_type}" if subagent_type else "Agent (subagent)"
            )
            metadata["subagent_type"] = subagent_type
            metadata["subagent_description"] = description
            # If the user registered this subagent via ``options.agents``,
            # surface its definition (prompt/tools/model) on the step so
            # reviewers see what the spawned subagent was configured to do.
            agents_defined = state.agents_defined or {}
            defn = agents_defined.get(subagent_type)
            if defn is not None:
                metadata["agent_definition"] = defn
        else:
            display_name = tool_name

        # If this tool was spawned by a parent Agent tool (subagent case),
        # nest beneath the parent's still-open ToolStep.
        parent_handle = state.pending_tools.get(tool_use_id)
        if parent_handle is not None:
            # Already exists — rare reentrant case. Skip to avoid leaking step.
            return {}

        parent_for_subagent = state.tool_to_parent_step.get(tool_use_id)
        pushed_token = None
        if parent_for_subagent is not None:
            pushed_token = _tracer._current_step.set(parent_for_subagent)
        try:
            cm = _tracer.create_step(
                name=display_name,
                step_type=step_type,
                inputs=tool_input,
                metadata=metadata,
            )
            handle = _ToolStepHandle(cm)
        finally:
            # Pop the temporary parent push immediately. We don't want
            # subsequent operations (e.g. message observation between
            # PreToolUse and PostToolUse) to be siblings of the tool step.
            if pushed_token is not None:
                _tracer._safe_reset_contextvar(_tracer._current_step, pushed_token)

        state.pending_tools[tool_use_id] = handle
    except Exception:  # never break user's hook execution
        logger.exception(
            "Openlayer PreToolUse hook failed for tool_use_id=%s", tool_use_id
        )
    return {}


async def _post_tool_use_hook(input_data, tool_use_id, context):  # noqa: D401
    """Composed PostToolUse hook — finalizes the TOOL step on success."""
    state = _current_state.get()
    if state is None or tool_use_id is None:
        return {}
    try:
        handle = state.pending_tools.pop(tool_use_id, None)
        if handle is None:
            return {}
        raw_output = (
            input_data.get("tool_response")
            or input_data.get("tool_output")
            or input_data.get("output")
        )
        output = _truncate(raw_output, _config.truncate_tool_output_chars)
        handle.log(output=output, metadata={"is_error": False})
        handle.end()
        # Note: ``state.tool_to_parent_step`` is now populated only by
        # ``_observe_assistant`` (it maps ``subagent_tool_use_id -> parent
        # AGENT step``). Nothing to write here.
    except Exception:
        logger.exception(
            "Openlayer PostToolUse hook failed for tool_use_id=%s", tool_use_id
        )
    return {}


async def _post_tool_use_failure_hook(input_data, tool_use_id, context):  # noqa: D401
    """Composed PostToolUseFailure hook — finalizes the TOOL step on error."""
    state = _current_state.get()
    if state is None or tool_use_id is None:
        return {}
    try:
        handle = state.pending_tools.pop(tool_use_id, None)
        if handle is None:
            return {}
        err = (
            input_data.get("error")
            or input_data.get("tool_response")
            or input_data.get("tool_output")
        )
        handle.log(
            output=_truncate(err, _config.truncate_tool_output_chars),
            metadata={"is_error": True},
        )
        handle.end()
    except Exception:
        logger.exception(
            "Openlayer PostToolUseFailure hook failed for tool_use_id=%s",
            tool_use_id,
        )
    return {}


def _inject_openlayer_hooks(options: Any) -> Any:
    """Return a copy of ``options`` with our internal hooks merged in.

    User-provided hooks are preserved untouched. Our hooks are appended after
    the user's so user hooks have first crack at any synchronous influence on
    the agent (e.g. ``permissionDecision`` decisions).
    """
    _require_cas()
    if options is None:
        options = _cas.ClaudeAgentOptions()

    HookMatcher = _cas.HookMatcher  # type: ignore[attr-defined]

    user_hooks: Dict[str, list] = dict(getattr(options, "hooks", None) or {})

    def _append(event: str, matcher: Any) -> None:
        user_hooks[event] = list(user_hooks.get(event) or []) + [matcher]

    _append("PreToolUse", HookMatcher(hooks=[_pre_tool_use_hook]))
    _append("PostToolUse", HookMatcher(hooks=[_post_tool_use_hook]))
    _append("PostToolUseFailure", HookMatcher(hooks=[_post_tool_use_failure_hook]))

    try:
        return _dataclass_replace(options, hooks=user_hooks)
    except TypeError:
        # ``options`` is not a dataclass or doesn't accept ``hooks`` in
        # ``replace``. Fall back to mutating in place (safe for our use-case
        # since we just instantiated it).
        try:
            setattr(options, "hooks", user_hooks)
        except Exception:
            logger.debug("Could not set hooks on options; tool tracing disabled")
        return options


# --------------------------- Observers --------------------------- #


def _observe_system_init(msg: Any, state: _TraceState) -> None:
    """Capture ``SystemMessage(subtype='init')`` into the root step metadata."""
    if getattr(msg, "subtype", None) != "init":
        return
    data = getattr(msg, "data", {}) or {}
    state.session_id = data.get("session_id")
    state.model = data.get("model")
    mcp_servers = data.get("mcp_servers")
    if _config.redact_mcp_env:
        mcp_servers = _redact_mcp_servers(mcp_servers)
    agent_config = {
        "model": data.get("model"),
        "tools": data.get("tools"),
        "mcp_servers": mcp_servers,
        "skills": data.get("skills"),
        "slash_commands": data.get("slash_commands"),
        "plugins": data.get("plugins"),
        "permission_mode": data.get("permissionMode"),
        "cwd": data.get("cwd"),
        "claude_code_version": data.get("claude_code_version"),
        "api_key_source": data.get("apiKeySource"),
        "output_style": data.get("output_style"),
    }
    state.root_step.log(
        metadata={
            "session_id": state.session_id,
            "agent_config": agent_config,
        }
    )


def _observe_result(msg: Any, state: _TraceState) -> None:
    """Capture ``ResultMessage`` — finalizes root cost / tokens / latency."""
    usage = getattr(msg, "usage", None) or {}
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    total_tokens = input_tokens + output_tokens
    duration_ms = getattr(msg, "duration_ms", None)

    # ``AgentStep`` doesn't define cost / tokens / prompt_tokens attributes, so
    # the recommended Step.log() route is metadata. We stash the same values
    # there. ``post_process_trace`` spreads root metadata into the trace_data
    # payload, so these show up at trace-level as well.
    state.root_step.output = getattr(msg, "result", None) or ""
    state.root_step.latency = duration_ms
    # We surface cost / tokens / per-turn token breakdowns via metadata because
    # base ``Step`` (and thus ``AgentStep``) doesn't define those attributes —
    # ``Step.log()`` would silently filter them otherwise. ``post_process_trace``
    # spreads root metadata into the trace-level row, so these become visible
    # at the trace level. The UI reads ``promptTokens`` / ``completionTokens``
    # in camelCase, so we duplicate the snake_case names with camelCase aliases
    # to make sure they land in the right column.
    state.root_step.log(
        metadata={
            "cost": getattr(msg, "total_cost_usd", None),
            "tokens": total_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "promptTokens": input_tokens,
            "completionTokens": output_tokens,
            "model": _resolve_root_model(state, msg),
            "provider": "anthropic",
            "rawOutput": _serialize_result_message(msg),
            "session_id": getattr(msg, "session_id", None) or state.session_id,
            "num_turns": getattr(msg, "num_turns", None),
            "stop_reason": getattr(msg, "stop_reason", None),
            "subtype": getattr(msg, "subtype", None),
            "is_error": getattr(msg, "is_error", False),
            "duration_api_ms": getattr(msg, "duration_api_ms", None),
            "model_usage": getattr(msg, "model_usage", None),
            "permission_denials": getattr(msg, "permission_denials", None),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        }
    )


def _resolve_root_model(state: _TraceState, result_msg: Any) -> Optional[str]:
    """Best-effort model identifier for the root step.

    ResultMessage doesn't carry a ``model`` field directly. We try, in order:
    the first key of ``model_usage`` (the SDK's per-model token breakdown),
    then the model recorded on the init ``SystemMessage`` (cached on state).
    """
    model_usage = getattr(result_msg, "model_usage", None)
    if isinstance(model_usage, dict) and model_usage:
        first = next(iter(model_usage.keys()), None)
        if first:
            return first
    return getattr(state, "model", None)


def _serialize_assistant_message(msg: Any, content: List[Any]) -> Optional[str]:
    """Render an AssistantMessage's full content array to JSON for ``raw_output``.

    Captures every block (text, thinking, tool_use) so users can inspect the
    complete model response even when text is empty. Tries to introspect
    dataclass-style block objects; falls back to ``str(block)``.
    """
    try:
        blocks: List[Dict[str, Any]] = []
        for b in content:
            bname = type(b).__name__
            if bname in ("TextBlock", "FakeTextBlock"):
                blocks.append({"type": "text", "text": getattr(b, "text", "")})
            elif bname in ("ThinkingBlock", "FakeThinkingBlock"):
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": getattr(b, "thinking", ""),
                        "signature": getattr(b, "signature", None),
                    }
                )
            elif bname in ("ToolUseBlock", "FakeToolUseBlock"):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(b, "id", ""),
                        "name": getattr(b, "name", ""),
                        "input": getattr(b, "input", None),
                    }
                )
            else:
                blocks.append({"type": bname, "repr": str(b)})
        return json.dumps(
            {
                "model": getattr(msg, "model", None),
                "stop_reason": getattr(msg, "stop_reason", None),
                "usage": getattr(msg, "usage", None),
                "parent_tool_use_id": getattr(msg, "parent_tool_use_id", None),
                "content": blocks,
            },
            default=str,
        )
    except Exception:
        return None


def _serialize_result_message(msg: Any) -> Optional[str]:
    """Render a ``ResultMessage`` to a JSON-ish string for ``rawOutput`` display."""
    try:
        return json.dumps(
            {
                "subtype": getattr(msg, "subtype", None),
                "result": getattr(msg, "result", None),
                "session_id": getattr(msg, "session_id", None),
                "duration_ms": getattr(msg, "duration_ms", None),
                "duration_api_ms": getattr(msg, "duration_api_ms", None),
                "num_turns": getattr(msg, "num_turns", None),
                "stop_reason": getattr(msg, "stop_reason", None),
                "is_error": getattr(msg, "is_error", None),
                "total_cost_usd": getattr(msg, "total_cost_usd", None),
                "usage": getattr(msg, "usage", None),
                "model_usage": getattr(msg, "model_usage", None),
                "permission_denials": getattr(msg, "permission_denials", None),
            },
            default=str,
        )
    except Exception:
        return None


def _resolve_subagent_parent(msg: Any, state: _TraceState) -> Any:
    """Return the spawning Agent ToolStep for a subagent message, or ``None``.

    When the SDK delegates work to a subagent via the ``Agent`` tool, the
    subsequent messages carry ``parent_tool_use_id`` pointing at the tool-use
    block. We open the tool step in ``_pre_tool_use_hook`` and keep it open
    while subagent messages stream; this helper finds the right step so we can
    push it onto the contextvar stack before opening child steps.
    """
    ptid = getattr(msg, "parent_tool_use_id", None)
    if ptid and ptid in state.pending_tools:
        return state.pending_tools[ptid].step
    return None


def _observe_assistant(msg: Any, state: _TraceState) -> None:
    """Emit a CHAT_COMPLETION step for each ``AssistantMessage``.

    Concatenates ``TextBlock`` content into ``output`` and ``ThinkingBlock``
    content into ``metadata.thinking`` (when ``capture_thinking`` is enabled).
    Tracks the IDs of any ``ToolUseBlock``s for cross-reference with later
    tool steps.
    """
    state.turn_counter += 1
    content = list(getattr(msg, "content", None) or [])
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_use_blocks: List[Dict[str, Any]] = []
    for block in content:
        bname = type(block).__name__
        if bname in ("TextBlock", "FakeTextBlock"):
            text_parts.append(getattr(block, "text", "") or "")
        elif bname in ("ThinkingBlock", "FakeThinkingBlock"):
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif bname in ("ToolUseBlock", "FakeToolUseBlock"):
            tool_use_blocks.append(
                {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", None),
                }
            )

    usage = getattr(msg, "usage", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = (input_tokens or 0) + (output_tokens or 0)
    text_output = "\n".join(p for p in text_parts if p)

    # Output: prefer text; fall back to tool-use summary; then thinking; then a
    # marker so users see *something* in the UI instead of an empty step.
    if text_output:
        output_for_ui: str = text_output
    elif tool_use_blocks:
        names = ", ".join(b["name"] for b in tool_use_blocks)
        output_for_ui = f"[tool call: {names}]"
    elif thinking_parts and _config.capture_thinking:
        output_for_ui = "[thinking]\n" + "\n".join(thinking_parts)
    else:
        output_for_ui = "[no content]"

    # For root-level assistant turns, surface the user's original prompt as
    # the step's input so users see what triggered this turn. For subagent
    # turns we omit it (they're triggered by the parent's Agent tool call,
    # not by a user prompt).
    is_subagent_turn = getattr(msg, "parent_tool_use_id", None) is not None
    step_inputs = None
    if not is_subagent_turn and state.user_prompt is not None:
        step_inputs = {"prompt": state.user_prompt}

    # If this is a subagent message that declares tool calls, register each
    # declared tool_use_id -> parent AGENT step so a later PreToolUse hook
    # firing for that tool nests the TOOL step inside the subagent's AGENT
    # boundary. We can't rely on the contextvar stack for this: the SDK may
    # run subagent hooks in a different asyncio task than the one that opened
    # the AGENT step, and contextvars are task-scoped — without an explicit
    # registration, the tool ends up at the root.
    if is_subagent_turn and tool_use_blocks:
        parent_agent_step = _resolve_subagent_parent(msg, state)
        if parent_agent_step is not None:
            for b in tool_use_blocks:
                if b.get("id"):
                    state.tool_to_parent_step[b["id"]] = parent_agent_step

    # Subagent nesting: if this assistant message belongs to a subagent run,
    # temporarily push the spawning tool's step onto the contextvar stack so
    # the new chat-completion step is created beneath it.
    subagent_parent = _resolve_subagent_parent(msg, state)
    pushed_token = None
    if subagent_parent is not None:
        pushed_token = _tracer._current_step.set(subagent_parent)
    try:
        with _tracer.create_step(
            name=f"assistant turn {state.turn_counter}",
            step_type=StepType.CHAT_COMPLETION,
            inputs=step_inputs,
        ) as chat_step:
            chat_step.log(
                output=output_for_ui,
                model=getattr(msg, "model", None),
                provider="anthropic",
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                tokens=total_tokens,
                raw_output=_serialize_assistant_message(msg, content),
                metadata={
                    "thinking": (
                        "\n".join(thinking_parts)
                        if (thinking_parts and _config.capture_thinking)
                        else None
                    ),
                    "tool_calls": tool_use_blocks or None,
                    "stop_reason": getattr(msg, "stop_reason", None),
                    "parent_tool_use_id": getattr(msg, "parent_tool_use_id", None),
                    "message_id": getattr(msg, "message_id", None),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                    "cache_creation_input_tokens": usage.get(
                        "cache_creation_input_tokens"
                    ),
                },
            )
    finally:
        if pushed_token is not None:
            _tracer._safe_reset_contextvar(_tracer._current_step, pushed_token)


def _observe(msg: Any, state: _TraceState) -> None:
    """Dispatch a message to the appropriate observer.

    Matches by class name so the ``Fake*`` mocks in the test suite are handled
    alongside the real ``claude_agent_sdk`` classes.
    """
    type_name = type(msg).__name__
    if type_name in ("SystemMessage", "FakeSystemMessage"):
        _observe_system_init(msg, state)
    elif type_name in ("AssistantMessage", "FakeAssistantMessage"):
        _observe_assistant(msg, state)
    elif type_name in ("ResultMessage", "FakeResultMessage"):
        _observe_result(msg, state)


# ----------------------------- Public API ----------------------------- #


async def traced_query(
    *,
    prompt: Any,
    options: Any = None,
    inference_pipeline_id: Optional[str] = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Wrap ``claude_agent_sdk.query()`` and emit an Openlayer trace.

    Every message from the underlying ``query()`` stream is yielded to the
    caller unchanged. Openlayer steps are emitted as a side effect.

    Args:
        prompt: The prompt argument forwarded to ``claude_agent_sdk.query``.
        options: Optional ``ClaudeAgentOptions`` forwarded to ``query``.
        inference_pipeline_id: Optional pipeline override; falls back to
            ``OPENLAYER_INFERENCE_PIPELINE_ID``.
        **kwargs: Any additional kwargs (e.g. ``transport``) are passed through.

    Yields:
        The same messages the underlying ``query()`` would have yielded, in the
        same order.
    """
    _require_cas()
    # Get hold of the *original* query — when our global monkey-patch is in
    # effect, ``_cas.query`` points back at ``patched_query`` which would
    # otherwise recurse into us.
    query_fn = getattr(_cas.query, "_openlayer_original", _cas.query)
    # Snapshot the user's options BEFORE we inject our hooks so the captured
    # metadata reflects what the user actually configured, not our mutations.
    original_options = options
    options = _inject_openlayer_hooks(options)
    resolved_pipeline = inference_pipeline_id or _config.inference_pipeline_id
    with _tracer.create_step(
        name=_ROOT_STEP_NAME,
        step_type=StepType.AGENT,
        inputs={"prompt": prompt},
        inference_pipeline_id=resolved_pipeline,
    ) as root_step:
        _capture_options_metadata(root_step, original_options)
        state = _TraceState(
            root_step=root_step,
            user_prompt=prompt if isinstance(prompt, str) else None,
            agents_defined=_serialize_agent_definitions(
                getattr(original_options, "agents", None)
            ),
        )
        token = _current_state.set(state)
        try:
            async for msg in query_fn(prompt=prompt, options=options, **kwargs):
                try:
                    _observe(msg, state)
                except Exception:  # never break the user's stream
                    logger.exception(
                        "Openlayer observation failed for message %r",
                        type(msg).__name__,
                    )
                yield msg
        finally:
            _current_state.reset(token)


def trace_claude_agent_sdk(
    *,
    inference_pipeline_id: Optional[str] = None,
    truncate_tool_output_chars: int = 8192,
    capture_thinking: bool = True,
    redact_mcp_env: bool = True,
) -> None:
    """Auto-instrument ``claude_agent_sdk.query`` so every call is traced.

    Call this once at startup before any code does ``from claude_agent_sdk
    import query``. After the call, ordinary use of the SDK is automatically
    wrapped: every ``async for m in query(...)`` becomes an Openlayer trace.

    The patch is idempotent — calling this function multiple times is safe and
    only updates configuration on subsequent calls.

    Args:
        inference_pipeline_id: Optional default Openlayer inference pipeline ID.
            Falls back to the ``OPENLAYER_INFERENCE_PIPELINE_ID`` env var when
            unset (handled by the tracer at publish time).
        truncate_tool_output_chars: Maximum number of characters of tool output
            to capture in each TOOL step. Excess is truncated with a marker.
            Defaults to 8192.
        capture_thinking: Whether to capture ``ThinkingBlock`` content into
            chat-completion step metadata. Defaults to True.
        redact_mcp_env: Whether to strip ``env`` and ``headers`` from MCP
            server configs in trace metadata. Defaults to True.
    """
    _require_cas()
    global _config
    _config = _Config(
        inference_pipeline_id=inference_pipeline_id,
        truncate_tool_output_chars=truncate_tool_output_chars,
        capture_thinking=capture_thinking,
        redact_mcp_env=redact_mcp_env,
    )

    if getattr(_cas.query, "_openlayer_patched", False):
        # Already patched — config update above is sufficient.
        return

    original_query = _cas.query

    async def patched_query(*args, **kwargs):
        # Forward through traced_query; ``query_fn`` resolves to
        # ``original_query`` via ``_openlayer_original`` to avoid recursion.
        async for m in traced_query(*args, **kwargs):
            yield m

    patched_query._openlayer_patched = True  # type: ignore[attr-defined]
    patched_query._openlayer_original = original_query  # type: ignore[attr-defined]
    _cas.query = patched_query  # type: ignore[assignment]

    _patch_claude_sdk_client()


# ----------------------- ClaudeSDKClient patching ----------------------- #


def _patch_claude_sdk_client() -> None:
    """Monkey-patch ``ClaudeSDKClient`` so its query/receive_response are traced.

    The client API splits the request lifecycle across two methods:
    ``client.query(prompt, session_id=...)`` (which only enqueues the prompt)
    and ``async for m in client.receive_response()`` (which streams the
    response). We open the root step when ``query`` is called and close it
    when the matching ``receive_response`` generator exhausts.
    """
    Client = _cas.ClaudeSDKClient  # type: ignore[attr-defined]
    if getattr(Client, "_openlayer_patched", False):
        return

    original_init = Client.__init__
    original_query = Client.query
    original_receive_response = Client.receive_response

    def patched_init(self, options=None, transport=None):  # type: ignore[no-redef]
        # Stash the original (pre-injection) options for metadata capture at
        # query() time.
        self._openlayer_original_options = options
        if options is not None:
            options = _inject_openlayer_hooks(options)
        else:
            options = _inject_openlayer_hooks(None)
        original_init(self, options=options, transport=transport)
        self._openlayer_state = None
        self._openlayer_state_cm = None
        self._openlayer_token = None

    async def patched_query(self, prompt, session_id="default"):  # type: ignore[no-redef]
        # Open a root step for this prompt; the matching receive_response will
        # close it. If the user calls query() multiple times without iterating
        # receive_response() in between, the previous state will be replaced —
        # but at that point the previous trace is orphaned anyway, so we just
        # log a debug message rather than try to recover.
        if getattr(self, "_openlayer_state", None) is not None:
            logger.debug(
                "ClaudeSDKClient.query() called before previous receive_response() "
                "was consumed; previous trace will be orphaned."
            )
        cm = _tracer.create_step(
            name=_ROOT_STEP_NAME,
            step_type=StepType.AGENT,
            inputs={"prompt": prompt, "session_id": session_id},
            inference_pipeline_id=_config.inference_pipeline_id,
        )
        root_step = cm.__enter__()
        original_options = getattr(self, "_openlayer_original_options", None)
        _capture_options_metadata(root_step, original_options)
        state = _TraceState(
            root_step=root_step,
            user_prompt=prompt if isinstance(prompt, str) else None,
            agents_defined=_serialize_agent_definitions(
                getattr(original_options, "agents", None)
            )
            if original_options is not None
            else None,
        )
        self._openlayer_state = state
        self._openlayer_state_cm = cm
        self._openlayer_token = _current_state.set(state)
        return await original_query(self, prompt, session_id=session_id)

    async def patched_receive_response(self):  # type: ignore[no-redef]
        state = getattr(self, "_openlayer_state", None)
        try:
            async for msg in original_receive_response(self):
                if state is not None:
                    try:
                        _observe(msg, state)
                    except Exception:
                        logger.exception(
                            "Openlayer observation failed for client message %r",
                            type(msg).__name__,
                        )
                yield msg
        finally:
            if state is not None and getattr(self, "_openlayer_state_cm", None):
                token = self._openlayer_token
                self._openlayer_state = None
                cm = self._openlayer_state_cm
                self._openlayer_state_cm = None
                self._openlayer_token = None
                try:
                    if token is not None:
                        _tracer._safe_reset_contextvar(_current_state, token)
                finally:
                    cm.__exit__(None, None, None)

    Client.__init__ = patched_init
    Client.query = patched_query
    Client.receive_response = patched_receive_response
    Client._openlayer_patched = True
