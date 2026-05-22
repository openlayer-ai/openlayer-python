"""Test helpers for claude-agent-sdk mock streams.

These dataclasses duck-type the real ``claude_agent_sdk`` message types just
enough that our integration's observer (which dispatches by ``type(msg).__name__``)
can be exercised without instantiating the real SDK objects. The ``Fake`` prefix
is matched explicitly in the dispatcher so tests stay self-contained.
"""

from __future__ import annotations

from typing import Any, List, Optional, AsyncIterator
from dataclasses import field, dataclass


@dataclass
class FakeTextBlock:
    """Mirrors ``claude_agent_sdk.TextBlock``."""

    text: str


@dataclass
class FakeThinkingBlock:
    """Mirrors ``claude_agent_sdk.ThinkingBlock``."""

    thinking: str
    signature: str = "sig"


@dataclass
class FakeToolUseBlock:
    """Mirrors ``claude_agent_sdk.ToolUseBlock``."""

    id: str
    name: str
    input: dict


@dataclass
class FakeToolResultBlock:
    """Mirrors ``claude_agent_sdk.ToolResultBlock``."""

    tool_use_id: str
    content: Any
    is_error: Optional[bool] = None


@dataclass
class FakeSystemMessage:
    """Mirrors ``claude_agent_sdk.SystemMessage``."""

    subtype: str
    data: dict


@dataclass
class FakeAssistantMessage:
    """Mirrors ``claude_agent_sdk.AssistantMessage``."""

    content: List[Any]
    model: str = "claude-opus-4-7"
    parent_tool_use_id: Optional[str] = None
    usage: Optional[dict] = None
    message_id: Optional[str] = None
    stop_reason: Optional[str] = None
    session_id: Optional[str] = None
    uuid: Optional[str] = None


@dataclass
class FakeUserMessage:
    """Mirrors ``claude_agent_sdk.UserMessage``."""

    content: Any
    parent_tool_use_id: Optional[str] = None
    uuid: Optional[str] = None
    tool_use_result: Optional[dict] = None


@dataclass
class FakeResultMessage:
    """Mirrors ``claude_agent_sdk.ResultMessage``."""

    subtype: str
    duration_ms: int = 1000
    duration_api_ms: int = 800
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "sess_test"
    total_cost_usd: float = 0.001
    usage: dict = field(
        default_factory=lambda: {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    )
    result: Optional[str] = "Done"
    model_usage: Optional[dict] = None
    stop_reason: Optional[str] = "end_turn"
    permission_denials: Optional[list] = None
    uuid: Optional[str] = None


async def make_stream(messages: list) -> AsyncIterator:
    """Yield a sequence of messages as an async iterator.

    This matches the shape ``claude_agent_sdk.query()`` returns.
    """
    for msg in messages:
        yield msg


def init_system_message(
    session_id: str = "sess_test",
    model: str = "claude-opus-4-7",
    tools: Optional[list] = None,
    mcp_servers: Optional[list] = None,
    skills: Optional[list] = None,
) -> FakeSystemMessage:
    """Build a ``SystemMessage(subtype='init')`` matching the SDK's init shape."""
    return FakeSystemMessage(
        subtype="init",
        data={
            "session_id": session_id,
            "model": model,
            "tools": tools or ["Read", "Bash"],
            "mcp_servers": mcp_servers or [],
            "skills": skills or [],
            "slash_commands": [],
            "plugins": [],
            "permissionMode": "default",
            "cwd": "/tmp",
            "claude_code_version": "test-0.1.81",
            "apiKeySource": "ANTHROPIC_API_KEY",
            "output_style": "default",
        },
    )
