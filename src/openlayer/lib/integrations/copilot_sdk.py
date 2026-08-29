"""Openlayer tracing integration for the GitHub Copilot SDK.

The Copilot SDK (``github-copilot-sdk`` on PyPI, imported as ``copilot``) drives
the Copilot CLI over JSON-RPC and exposes a rich session event stream. This
module subscribes to that stream and turns each ``send()`` into an Openlayer
trace with nested steps for assistant turns, tool calls and subagents.

Why we buffer instead of building steps live
--------------------------------------------
``assistant.usage`` -- which carries every token count and the premium-request
figure -- is ``ephemeral`` and absent from ``session.get_events()``, so we have
to listen live. But Copilot fires tool calls *concurrently*: three
``tool.execution_start`` events arrive before any completion, and the
completions come back out of order. Driving step context managers straight from
the live callbacks would therefore nest sibling tools inside one another. We
buffer live and build the whole trace in one deterministic, correctly-nested
pass when the session goes idle.

See ``docs/superpowers/specs/2026-08-27-github-copilot-sdk-integration-design.md``.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..tracing import enums, tracer

logger = logging.getLogger(__name__)

__all__ = [
    "CopilotConfig",
    "CopilotTraceCollector",
    "openlayer_event_handler",
    "trace_copilot",
    "untrace_copilot",
]

ROOT_STEP_NAME = "GitHub Copilot"

# High-frequency events that carry no step-level meaning. Dropped at ingest so
# the buffer stays small -- a single session emits hundreds of these.
_IGNORED_EVENT_TYPES = frozenset(
    {
        "assistant.message_delta",
        "assistant.reasoning_delta",
        "assistant.streaming_delta",
        "assistant.tool_call_delta",
        "tool.execution_partial_result",
        "session.background_tasks_changed",
    }
)

# Copilot routes to several model families. The Openlayer cost service keys on
# the *real* provider, and Copilot's model ids match its slugs verbatim
# (verified: anthropic/claude-haiku-4.5 and openai/gpt-5.4 both resolve, while
# github/* returns "No cost data found"). Labelling these "github" would yield a
# silent $0, so an unrecognized prefix omits the provider entirely -- landing
# unpriced is recoverable, landing priced-wrong is not.
_PROVIDER_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini-", "google"),
    ("grok-", "xai"),
)


# --------------------------------------------------------------------------- #
# Event accessors (tolerant of dict / dataclass and snake_case / camelCase)
# --------------------------------------------------------------------------- #


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _event_type(event: Any) -> str:
    """Return an event's ``type`` as a plain string.

    The live Python binding hands us ``SessionEventType.SESSION_START`` (an
    Enum), not ``"session.start"`` -- so every comparison silently fails unless
    we coerce. Recorded fixtures carry the serialized string form, which is why
    this needs an explicit test rather than only fixture coverage.
    """
    raw = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    if raw is None:
        return ""
    value = getattr(raw, "value", raw)
    return value if isinstance(value, str) else str(value)


def _event_data(event: Any) -> Dict[str, Any]:
    """Return an event's ``data`` payload as a plain dict."""
    if isinstance(event, dict):
        data = event.get("data")
    else:
        data = getattr(event, "data", None)
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if hasattr(data, "__dict__"):
        return dict(vars(data))
    return {}


def _field(data: Any, name: str) -> Any:
    """Read ``name`` from a payload tolerating snake_case and camelCase.

    The Python binding yields ``tool_call_id``; the TypeScript binding (and the
    fixtures captured from it) yields ``toolCallId``. Accepting both keeps a
    single code path and makes the two SDK implementations mirror each other.
    """
    if not isinstance(data, dict):
        return None
    if name in data:
        return data[name]
    return data.get(_camel(name))


def _envelope(event: Any, name: str) -> Any:
    """Read a top-level envelope field (``agent_id``, ``timestamp``, ...)."""
    if isinstance(event, dict):
        return _field(event, name)
    value = getattr(event, name, None)
    if value is None:
        value = getattr(event, _camel(name), None)
    return value


def _timestamp(event: Any) -> Optional[float]:
    """Convert an ISO-8601 event timestamp to epoch seconds."""
    raw = _envelope(event, "timestamp")
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _duration_seconds(value: Any) -> Optional[float]:
    """Copilot durations arrive as timedelta (Python) or ms number (TS)."""
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Coerce SDK dataclasses and enums into JSON-serializable structures.

    Live events carry dataclasses (``ToolExecutionCompleteResult``) and enums
    where the recorded fixtures carry plain dicts and strings. Anything we put
    on a step has to survive JSON serialization at publish time.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _depth > 6:  # defensive: never recurse into a cyclic SDK object forever
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, enum.Enum):
        return _jsonable(value.value, _depth + 1)
    # Also unwrap enum-like objects whose only public attribute is ``value``,
    # so a binding that hands back a lookalike is not serialized as {"value": x}.
    attrs = vars(value) if hasattr(value, "__dict__") else {}
    public = {k for k in attrs if not k.startswith("_")}
    if public == {"value"}:
        return _jsonable(attrs["value"], _depth + 1)
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonable(v, _depth + 1)
            for k, v in vars(value).items()
            if not k.startswith("__")
        }
    return str(value)


def _truncate(value: Any, max_chars: int) -> Any:
    """Cap oversized tool payloads; results can contain whole files."""
    value = _jsonable(value)
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…<truncated %d chars>" % (len(value) - max_chars)
    return value


# --------------------------------------------------------------------------- #
# Mapping helpers
# --------------------------------------------------------------------------- #


def _provider_for_model(model: Optional[str]) -> Optional[str]:
    """Map a Copilot model id to the real underlying provider slug."""
    if not model:
        return None
    lowered = str(model).lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            return provider
    logger.debug("Openlayer: no provider mapping for Copilot model %r", model)
    return None


def _usage_details(usage: Dict[str, Any]) -> Dict[str, int]:
    """Build the non-overlapping token partition the cost backend prices.

    Copilot reports ``input_tokens`` as a *superset* that already contains
    ``cache_read_tokens`` and ``cache_write_tokens``. The backend sums a price
    per recognized key, so the granular categories must be broken out and
    subtracted from the base or cached tokens get charged twice. Reasoning stays
    folded into ``output_tokens``, matching the convention already established
    in ``langchain_callback._build_usage_details``.
    """
    input_total = int(_field(usage, "input_tokens") or 0)
    output_total = int(_field(usage, "output_tokens") or 0)
    cache_read = int(_field(usage, "cache_read_tokens") or 0)
    cache_write = int(_field(usage, "cache_write_tokens") or 0)

    details: Dict[str, int] = {}
    for key, value in (
        ("input_tokens", input_total - cache_read - cache_write),
        ("output_tokens", output_total),
        ("cached_tokens", cache_read),
        ("cache_creation_tokens", cache_write),
    ):
        if value and value > 0:
            details[key] = value
    return details


# GitHub meters every call in AIU ("AI Units") and reports the total on
# ``copilot_usage.total_nano_aiu``. Decoding the per-token rates GitHub ships
# alongside it (``copilot_usage._token_details``) for ``claude-haiku-4.5`` gives
# 100 / 500 / 10 / 125 AIU per 1M input / output / cache-read / cache-write
# tokens -- exactly Anthropic's published list prices scaled by 100. So one AIU
# is one US cent, and ``total_nano_aiu / 1e11`` reproduced the priced cost of an
# observed call to twelve decimal places ($0.01675375 both ways).
#
# Confirmed across two vendors: ``gpt-5-mini`` bills 25 / 200 / 2.5 AIU per 1M
# input / output / cache-read tokens, which is OpenAI's $0.25 / $2.00 / $0.025
# list pricing under the same 1 AIU = $0.01 constant. Live rows carry both
# figures and they agree to twelve decimal places.
#
# Still not proven for *every* vendor Copilot may route to, so this is used as a
# *fallback* for models we cannot map to a provider, and as a cross-check
# alongside the priced cost -- never in place of Openlayer's own pricing when the
# provider is known.
_NANO_AIU_PER_USD = 1e11


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Normalize a nested payload to a dict.

    Live events nest dataclasses (``copilot_usage`` is
    ``AssistantUsageCopilotUsage``, not a dict) where the recorded fixtures nest
    plain dicts -- so an ``isinstance(..., dict)`` guard silently skips real
    data. Same trap as the enum-typed ``type`` field.
    """
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _metered_cost_usd(usage: Dict[str, Any]) -> Optional[float]:
    """GitHub's own metered value of a call, in USD, or None if unavailable."""
    copilot_usage = _as_mapping(_field(usage, "copilot_usage"))
    nano_aiu = _field(copilot_usage, "total_nano_aiu")
    if not nano_aiu:
        return None
    try:
        return float(nano_aiu) / _NANO_AIU_PER_USD
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Configuration and buffered records
# --------------------------------------------------------------------------- #


@dataclass
class CopilotConfig:
    """Tunable per-integration configuration."""

    inference_pipeline_id: Optional[str] = None
    truncate_tool_output_chars: int = 8192
    capture_reasoning: bool = True


@dataclass
class _Interaction:
    """One ``send()`` -- the unit that becomes a single Openlayer trace."""

    interaction_id: str
    user_prompt: str = ""
    events: List[Any] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


@dataclass
class _TurnRecord:
    """One LLM call: an ``assistant.turn_start`` .. ``assistant.message`` cycle."""

    agent_id: Optional[str]
    turn_id: Any
    model: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output: str = ""
    reasoning: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    api_call_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    parent_tool_call_id: Optional[str] = None


@dataclass
class _ToolRecord:
    """One tool execution, or one subagent dispatch."""

    tool_call_id: str
    name: str = "tool"
    arguments: Any = None
    result: Any = None
    error: Any = None
    success: Optional[bool] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    parent_tool_call_id: Optional[str] = None
    agent_id: Optional[str] = None
    mcp_server_name: Optional[str] = None
    is_subagent: bool = False
    subagent_name: Optional[str] = None
    subagent_metadata: Dict[str, Any] = field(default_factory=dict)
    permission: Optional[str] = None


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


class CopilotTraceCollector:
    """Buffers a Copilot session's events and builds traces when it goes idle.

    One instance per session. Wire it up via
    ``client.create_session(on_event=collector.handle)``.
    """

    def __init__(self, config: Optional[CopilotConfig] = None) -> None:
        self.config = config or CopilotConfig()
        self.session_id: Optional[str] = None
        self.session_model: Optional[str] = None
        self.interactions: Dict[str, _Interaction] = {}
        self.built_count = 0

        # Correlation tables. Needed because assistant.turn_end and
        # assistant.usage carry no interaction_id at all.
        self._turn_to_interaction: Dict[Tuple[Any, Any], str] = {}
        self._tool_to_interaction: Dict[str, str] = {}
        self._api_call_to_interaction: Dict[str, str] = {}
        self._agent_to_interaction: Dict[str, str] = {}
        self._current_interaction_id: Optional[str] = None

    # ------------------------------ ingest ------------------------------ #

    def handle(self, event: Any) -> None:
        """Entry point wired to ``create_session(on_event=...)``.

        Must never raise: it runs inside the SDK's own event dispatch, and
        observability must not be able to break the customer's agent.
        """
        try:
            self._handle(event)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Openlayer: failed to handle Copilot event", exc_info=True)

    def flush(self) -> None:
        """Build any still-open interactions (e.g. on disconnect)."""
        try:
            self._build_all()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Openlayer: failed to flush Copilot traces", exc_info=True)

    def _handle(self, event: Any) -> None:
        event_type = _event_type(event)
        if not event_type or event_type in _IGNORED_EVENT_TYPES:
            return

        data = _event_data(event)

        if event_type == "session.start":
            self.session_id = _field(data, "session_id") or self.session_id
            return
        if event_type == "session.model_change":
            self.session_model = _field(data, "model") or self.session_model
            return
        if event_type in ("session.idle", "session.shutdown"):
            # "The session went quiescent" -- not "this send finished". Two
            # overlapping send() calls are serialized by Copilot and produce a
            # single idle with both interactions open, so build them all.
            self._build_all()
            return

        agent_id = _envelope(event, "agent_id")

        if event_type == "user.message" and not agent_id:
            # Root-agent user messages open an interaction. Subagents emit their
            # own user.message; those belong to the dispatching interaction.
            interaction_id = _field(data, "interaction_id") or "_auto_%d" % len(self.interactions)
            interaction = _Interaction(
                interaction_id=interaction_id,
                user_prompt=_field(data, "content") or "",
                start_time=_timestamp(event) or time.time(),
            )
            self.interactions[interaction_id] = interaction
            self._current_interaction_id = interaction_id
            interaction.events.append(event)
            return

        interaction_id = self._route(data, event)
        if interaction_id is None:
            return
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return
        interaction.events.append(event)
        self._remember_correlations(event_type, data, interaction_id, event)

    def _route(self, data: Dict[str, Any], event: Any) -> Optional[str]:
        """Find the interaction an event belongs to.

        Order matters. ``assistant.usage`` carries no ``interaction_id`` and no
        ``turn_id``, and its ``api_call_id`` is only learned from the
        ``assistant.message`` that arrives *after* it -- so it must be routable
        by ``parent_tool_call_id`` or ``agent_id`` before any fallback.
        """
        explicit = _field(data, "interaction_id")
        if explicit and explicit in self.interactions:
            return explicit
        for key, table in (
            ("tool_call_id", self._tool_to_interaction),
            ("parent_tool_call_id", self._tool_to_interaction),
            ("api_call_id", self._api_call_to_interaction),
        ):
            value = _field(data, key)
            if value and value in table:
                return table[value]
        agent_id = _envelope(event, "agent_id")
        if agent_id and agent_id in self._agent_to_interaction:
            return self._agent_to_interaction[agent_id]
        turn_key = (agent_id, _field(data, "turn_id"))
        if turn_key in self._turn_to_interaction:
            return self._turn_to_interaction[turn_key]
        return self._current_interaction_id

    def _remember_correlations(
        self, event_type: str, data: Dict[str, Any], interaction_id: str, event: Any
    ) -> None:
        tool_call_id = _field(data, "tool_call_id")
        if tool_call_id:
            self._tool_to_interaction[tool_call_id] = interaction_id
        api_call_id = _field(data, "api_call_id")
        if api_call_id:
            self._api_call_to_interaction[api_call_id] = interaction_id
        # subagent.started is the first event tying an agentId to an
        # interaction, and it arrives before any of that subagent's own events.
        if event_type == "subagent.started" and tool_call_id:
            self._agent_to_interaction[tool_call_id] = interaction_id
        if event_type == "assistant.turn_start":
            agent_id = _envelope(event, "agent_id")
            self._turn_to_interaction[(agent_id, _field(data, "turn_id"))] = interaction_id

    # ------------------------------ assembly ------------------------------ #

    def _build_all(self) -> None:
        for interaction in list(self.interactions.values()):
            if interaction.events:
                self.build_interaction(interaction)
        self.interactions.clear()
        self._current_interaction_id = None

    def _assemble(
        self, interaction: _Interaction
    ) -> Tuple[Dict[Tuple[Any, Any], _TurnRecord], Dict[str, _ToolRecord], List[_TurnRecord]]:
        """Fold buffered events into turn and tool records."""
        turns: Dict[Tuple[Any, Any], _TurnRecord] = {}
        tools: Dict[str, _ToolRecord] = {}
        order: List[_TurnRecord] = []
        # assistant.usage arrives *before* the assistant.message it belongs to
        # and carries no turn_id, so park it by api_call_id until the message
        # names the turn. Never join on adjacency.
        pending_usage: Dict[str, Dict[str, Any]] = {}

        for event in interaction.events:
            event_type = _event_type(event)
            data = _event_data(event)
            agent_id = _envelope(event, "agent_id")
            ts = _timestamp(event)

            if event_type == "assistant.turn_start":
                key = (agent_id, _field(data, "turn_id"))
                turn = _TurnRecord(
                    agent_id=agent_id,
                    turn_id=_field(data, "turn_id"),
                    model=_field(data, "model"),
                    start_time=ts,
                )
                turns[key] = turn
                order.append(turn)

            elif event_type == "model.call_start":
                key = (agent_id, _field(data, "turn_id"))
                turn = turns.get(key)
                if turn is not None:
                    turn.model = _field(data, "model") or turn.model
                    if ts:
                        turn.start_time = turn.start_time or ts

            elif event_type == "assistant.usage":
                api_call_id = _field(data, "api_call_id")
                if api_call_id:
                    pending_usage[api_call_id] = data

            elif event_type == "assistant.message":
                key = (agent_id, _field(data, "turn_id"))
                turn = turns.get(key)
                if turn is None:
                    turn = _TurnRecord(
                        agent_id=agent_id, turn_id=_field(data, "turn_id"), start_time=ts
                    )
                    turns[key] = turn
                    order.append(turn)
                content = _field(data, "content") or ""
                turn.output = (turn.output + content) if turn.output else content
                turn.model = _field(data, "model") or turn.model
                turn.end_time = ts or turn.end_time
                turn.parent_tool_call_id = _field(data, "parent_tool_call_id")
                api_call_id = _field(data, "api_call_id")
                turn.api_call_id = api_call_id
                reasoning = _field(data, "reasoning_text")
                if reasoning:
                    turn.reasoning = reasoning
                # Join usage to message on api_call_id -- verified 1:1 across
                # every turn, including inside subagents.
                if api_call_id and api_call_id in pending_usage:
                    turn.usage = pending_usage.pop(api_call_id)

            elif event_type == "assistant.reasoning":
                key = (agent_id, _field(data, "turn_id"))
                turn = turns.get(key)
                text = _field(data, "text") or _field(data, "content")
                if turn is not None and text and not turn.reasoning:
                    turn.reasoning = text

            elif event_type == "assistant.turn_end":
                key = (agent_id, _field(data, "turn_id"))
                turn = turns.get(key)
                if turn is not None and ts:
                    turn.end_time = ts

            elif event_type == "model.call_failure":
                key = (agent_id, _field(data, "turn_id"))
                turn = turns.get(key)
                if turn is not None:
                    turn.error = {
                        "message": _field(data, "error_message"),
                        "code": _field(data, "error_code"),
                        "type": _field(data, "error_type"),
                        "status_code": _field(data, "status_code"),
                    }

            elif event_type == "tool.execution_start":
                tool_call_id = _field(data, "tool_call_id")
                if not tool_call_id:
                    continue
                tools[tool_call_id] = _ToolRecord(
                    tool_call_id=tool_call_id,
                    name=_field(data, "tool_name") or "tool",
                    arguments=_field(data, "arguments"),
                    start_time=ts,
                    parent_tool_call_id=_field(data, "parent_tool_call_id"),
                    agent_id=agent_id,
                    mcp_server_name=_field(data, "mcp_server_name"),
                )

            elif event_type == "tool.execution_complete":
                tool_call_id = _field(data, "tool_call_id")
                tool = tools.get(tool_call_id) if tool_call_id else None
                if tool is None:
                    continue
                tool.success = _field(data, "success")
                tool.error = _field(data, "error")
                tool.result = _field(data, "result")
                tool.end_time = ts

            elif event_type == "subagent.started":
                tool_call_id = _field(data, "tool_call_id")
                tool = tools.get(tool_call_id) if tool_call_id else None
                if tool is None:
                    continue
                tool.is_subagent = True
                tool.subagent_name = _field(data, "agent_display_name") or _field(
                    data, "agent_name"
                )
                tool.subagent_metadata.update(
                    {
                        "agent_name": _field(data, "agent_name"),
                        "agent_description": _field(data, "agent_description"),
                        "model": _field(data, "model"),
                    }
                )

            elif event_type == "subagent.completed":
                tool_call_id = _field(data, "tool_call_id")
                tool = tools.get(tool_call_id) if tool_call_id else None
                if tool is None:
                    continue
                tool.subagent_metadata.update(
                    {
                        "total_tokens": _field(data, "total_tokens"),
                        "total_tool_calls": _field(data, "total_tool_calls"),
                        "duration_ms": _field(data, "duration_ms"),
                        "cancelled": _field(data, "cancelled"),
                    }
                )

            elif event_type == "permission.completed":
                tool_call_id = _field(data, "tool_call_id")
                tool = tools.get(tool_call_id) if tool_call_id else None
                if tool is not None:
                    tool.permission = _field(data, "decision") or _field(data, "outcome")

        return turns, tools, order

    # ------------------------------ building ------------------------------ #

    def build_interaction(self, interaction: _Interaction) -> None:
        """Build and publish one Openlayer trace from a buffered interaction."""
        turns, tools, order = self._assemble(interaction)

        # The root's output is the LAST root-agent assistant message. Openlayer
        # builds a row's output from the root step, so an empty root silently
        # yields an unusable row.
        final_output = ""
        for turn in reversed(order):
            if turn.agent_id is None and turn.output:
                final_output = turn.output
                break

        # A record's parent is its parent_tool_call_id, else the tool call that
        # dispatched its agent, else the root. Never parentId -- that is a
        # chronological chain, not a tree.
        children: Dict[Optional[str], List[Any]] = {}

        def parent_of(record: Any) -> Optional[str]:
            explicit = getattr(record, "parent_tool_call_id", None)
            if explicit:
                return explicit
            agent_id = getattr(record, "agent_id", None)
            if agent_id and agent_id in tools:
                return agent_id
            return None

        for record in list(order) + list(tools.values()):
            children.setdefault(parent_of(record), []).append(record)
        for bucket in children.values():
            bucket.sort(key=lambda r: r.start_time or 0.0)

        end_times = [r.end_time for r in list(order) + list(tools.values()) if r.end_time]
        root_end = max(end_times) if end_times else None

        with tracer.create_step(
            name=ROOT_STEP_NAME,
            step_type=enums.StepType.AGENT,
            inputs={"prompt": interaction.user_prompt},
            output=final_output,
            metadata={
                "copilot_session_id": self.session_id,
                "copilot_interaction_id": interaction.interaction_id,
                "model": self.session_model,
            },
            inference_pipeline_id=self.config.inference_pipeline_id,
        ) as root:
            root.log(start_time=interaction.start_time)
            if root_end:
                root.log(
                    end_time=root_end,
                    latency=(root_end - interaction.start_time) * 1000.0,
                )
            self._emit(children, None)

        self.built_count += 1

    def _emit(self, children: Dict[Optional[str], List[Any]], parent_key: Optional[str]) -> None:
        """Recursively open and close steps in correct nesting order.

        Because this pass is synchronous and properly nested, the tracer's step
        stack yields the right tree -- the concurrency hazard was removed by
        deferring construction until every event was in hand.
        """
        for record in children.get(parent_key, []):
            if isinstance(record, _TurnRecord):
                self._emit_turn(record)
            else:
                self._emit_tool(record, children)

    def _emit_turn(self, turn: _TurnRecord) -> None:
        usage = turn.usage or {}
        metadata: Dict[str, Any] = {"turn_id": turn.turn_id}
        if turn.agent_id:
            metadata["agent_id"] = turn.agent_id

        premium = _field(usage, "cost")
        if premium is not None:
            # NOT dollars -- Copilot premium-request units, a flat per-model
            # multiplier. Kept as metadata so the step's real cost is priced by
            # Openlayer from provider+model instead.
            metadata["copilot_premium_requests"] = premium
        copilot_usage = _as_mapping(_field(usage, "copilot_usage"))
        nano_aiu = _field(copilot_usage, "total_nano_aiu")
        if nano_aiu is not None:
            metadata["copilot_nano_aiu"] = nano_aiu
        metered_cost = _metered_cost_usd(usage)
        if metered_cost is not None:
            # GitHub's own metered value of this call in USD. Recorded even when
            # Openlayer prices the step, so the two figures can be compared.
            metadata["copilot_metered_cost_usd"] = metered_cost
        for key in ("finish_reason", "api_endpoint", "service_request_id", "initiator"):
            value = _field(usage, key)
            if value is not None:
                metadata["copilot_" + key] = value
        if turn.reasoning and self.config.capture_reasoning:
            metadata["reasoning"] = turn.reasoning
        if turn.error:
            metadata["error"] = turn.error

        details = _usage_details(usage)
        name = "turn %s" % turn.turn_id if turn.turn_id is not None else "assistant turn"

        with tracer.create_step(
            name=name,
            step_type=enums.StepType.CHAT_COMPLETION,
            inputs={"prompt": []},
            output=turn.output,
            metadata=_jsonable(metadata),
            inference_pipeline_id=self.config.inference_pipeline_id,
        ) as step:
            provider = _provider_for_model(turn.model)
            step.log(
                model=turn.model,
                provider=provider,
                prompt_tokens=details.get("input_tokens"),
                completion_tokens=details.get("output_tokens"),
                tokens=sum(details.values()) or None,
                usage_details=details or None,
            )
            if provider is None and metered_cost is not None:
                # Without a provider Openlayer cannot price the step, and an
                # unpriced step reads as $0. GitHub's own metered figure is far
                # better than nothing, so a model we have no mapping for still
                # lands costed.
                step.log(cost=metered_cost)
            if turn.start_time:
                step.log(start_time=turn.start_time)
            end = turn.end_time or _duration_and_end(turn)
            if end and turn.start_time:
                step.log(end_time=end, latency=(end - turn.start_time) * 1000.0)

    def _emit_tool(self, tool: _ToolRecord, children: Dict[Optional[str], List[Any]]) -> None:
        metadata: Dict[str, Any] = {
            "tool_call_id": tool.tool_call_id,
            "success": tool.success,
        }
        if tool.error:
            metadata["error"] = tool.error
        if tool.mcp_server_name:
            metadata["mcp_server_name"] = tool.mcp_server_name
        if tool.permission:
            metadata["permission"] = tool.permission

        if tool.is_subagent:
            step_type = enums.StepType.AGENT
            name = "subagent: %s" % (tool.subagent_name or "unknown")
            metadata.update(tool.subagent_metadata)
        else:
            step_type = enums.StepType.TOOL
            name = tool.name

        output = _truncate(tool.result, self.config.truncate_tool_output_chars)

        with tracer.create_step(
            name=name,
            step_type=step_type,
            inputs=_jsonable(tool.arguments),
            output=output,
            metadata=_jsonable(metadata),
            inference_pipeline_id=self.config.inference_pipeline_id,
        ) as step:
            if tool.start_time:
                step.log(start_time=tool.start_time)
            if tool.end_time and tool.start_time:
                step.log(
                    end_time=tool.end_time,
                    latency=(tool.end_time - tool.start_time) * 1000.0,
                )
            self._emit(children, tool.tool_call_id)


def _duration_and_end(turn: _TurnRecord) -> Optional[float]:
    """Derive a turn end time from usage duration when turn_end was missed."""
    duration = _duration_seconds(_field(turn.usage or {}, "duration"))
    if duration and turn.start_time:
        return turn.start_time + duration
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


_OPENLAYER_HANDLER_ATTR = "_openlayer_copilot_handler"


def _is_openlayer_handler(handler: Any) -> bool:
    """True if ``handler`` was produced by :func:`openlayer_event_handler`."""
    return bool(getattr(handler, _OPENLAYER_HANDLER_ATTR, False))


def _compose_handlers(
    user_handler: Optional[Callable[[Any], None]], our_handler: Callable[[Any], None]
) -> Callable[[Any], None]:
    """Run both handlers; neither may break the other."""

    def composed(event: Any) -> None:
        if user_handler is not None:
            try:
                user_handler(event)
            except Exception:
                logger.debug("Openlayer: user on_event handler raised", exc_info=True)
        our_handler(event)

    return composed


def openlayer_event_handler(**config_kwargs: Any) -> Callable[[Any], None]:
    """Return an ``on_event`` handler that traces one Copilot session.

    The returned callable exposes ``.flush()`` and ``.collector`` so callers who
    build sessions themselves can publish a partial trace for a session that is
    torn down without ever going idle.

    Example:
        >>> from openlayer.lib.integrations.copilot_sdk import openlayer_event_handler
        >>> handler = openlayer_event_handler()
        >>> session = await client.create_session(on_event=handler)
        >>> ...
        >>> handler.flush()  # optional: only needed on an abnormal teardown
    """
    collector = CopilotTraceCollector(CopilotConfig(**config_kwargs))

    # A closure rather than ``collector.handle`` directly: bound methods do not
    # accept attribute assignment, and we want ``.flush`` / ``.collector`` on
    # the returned callable while keeping it a plain function that can be passed
    # straight to ``on_event``.
    def handler(event: Any) -> None:
        collector.handle(event)

    handler.flush = collector.flush  # type: ignore[attr-defined]
    handler.collector = collector  # type: ignore[attr-defined]
    setattr(handler, _OPENLAYER_HANDLER_ATTR, True)
    return handler


def _wrap_disconnect(session: Any, collector: CopilotTraceCollector) -> None:
    """Flush any still-open interaction when the session is disconnected.

    ``session.idle`` is the normal trigger, but a session torn down mid-flight
    never emits it, and the buffered work would otherwise be dropped silently.
    Flushing is idempotent: ``_build_all`` clears the buffer, so a disconnect
    after a normal idle publishes nothing extra.
    """
    original_disconnect = getattr(session, "disconnect", None)
    if original_disconnect is None:
        return

    async def patched_disconnect(*args: Any, **kwargs: Any) -> Any:
        try:
            collector.flush()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Openlayer: failed to flush on disconnect", exc_info=True)
        return await original_disconnect(*args, **kwargs)

    try:
        session.disconnect = patched_disconnect
    except (AttributeError, TypeError):  # pragma: no cover - frozen session object
        logger.debug("Openlayer: could not wrap session.disconnect", exc_info=True)


_PATCHED = False
_ORIGINAL_CREATE_SESSION: Optional[Callable[..., Any]] = None


def trace_copilot(**config_kwargs: Any) -> None:
    """Enable Openlayer tracing for every GitHub Copilot SDK session.

    Monkey-patches ``copilot.CopilotClient.create_session`` so each session gets
    an Openlayer event handler, composed with any ``on_event`` the caller
    supplies. Idempotent.

    Requirements:
        ``github-copilot-sdk>=1.0.11`` must be installed:
        ``pip install 'github-copilot-sdk>=1.0.11'``

    Args:
        inference_pipeline_id: Optional Openlayer inference pipeline ID. Falls
            back to the ``OPENLAYER_INFERENCE_PIPELINE_ID`` env var.
        truncate_tool_output_chars: Max characters of tool output per TOOL step.
            Defaults to 8192.
        capture_reasoning: Whether to capture reasoning text into chat-step
            metadata. Defaults to True.

    Example:
        >>> from openlayer.lib import trace_copilot
        >>> trace_copilot()
        >>> client = CopilotClient()
        >>> session = await client.create_session()
        >>> await session.send_and_wait("Summarize this repo")
    """
    global _PATCHED, _ORIGINAL_CREATE_SESSION
    if _PATCHED:
        return
    try:
        import copilot
    except ImportError as exc:
        raise ImportError(
            "github-copilot-sdk is required for Copilot SDK tracing. "
            "Install with: pip install 'github-copilot-sdk>=1.0.11'"
        ) from exc

    _ORIGINAL_CREATE_SESSION = copilot.CopilotClient.create_session
    original = _ORIGINAL_CREATE_SESSION

    async def patched_create_session(self: Any, **kwargs: Any) -> Any:
        existing = kwargs.get("on_event")
        if _is_openlayer_handler(existing):
            # The caller already passed an ``openlayer_event_handler()``. Adding
            # a second collector would build the same interaction twice and
            # publish duplicate rows -- and mixing the two entry points is easy
            # to do by following the quickstart and then the "trace specific
            # sessions" snippet. Defer to theirs.
            session = await original(self, **kwargs)
            _wrap_disconnect(session, existing.collector)
            return session
        handler = openlayer_event_handler(**config_kwargs)
        kwargs["on_event"] = _compose_handlers(existing, handler)
        session = await original(self, **kwargs)
        _wrap_disconnect(session, handler.collector)  # type: ignore[attr-defined]
        return session

    copilot.CopilotClient.create_session = patched_create_session
    _PATCHED = True


def untrace_copilot() -> None:
    """Undo :func:`trace_copilot`. Primarily for tests."""
    global _PATCHED, _ORIGINAL_CREATE_SESSION
    if not _PATCHED or _ORIGINAL_CREATE_SESSION is None:
        return
    import copilot

    copilot.CopilotClient.create_session = _ORIGINAL_CREATE_SESSION
    _ORIGINAL_CREATE_SESSION = None
    _PATCHED = False
