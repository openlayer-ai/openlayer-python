"""Module with the enums used in the tracing module."""

import enum


class StepType(enum.Enum):
    """Types of steps in a trace."""

    AGENT = "agent"
    CHAT_COMPLETION = "chat_completion"
    GUARDRAIL = "guardrail"
    HANDOFF = "handoff"
    RETRIEVER = "retriever"
    TOOL = "tool"
    USER_CALL = "user_call"


class ContentType(enum.Enum):
    """Types of content in multimodal messages."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
