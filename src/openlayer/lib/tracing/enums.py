"""Module with the enums used in the tracing module."""

import enum


class StepType(enum.Enum):
    AGENT = "agent"
    CHAT_COMPLETION = "chat_completion"
    GUARDRAIL = "guardrail"
    HANDOFF = "handoff"
    RETRIEVER = "retriever"
    TOOL = "tool"
    USER_CALL = "user_call"
