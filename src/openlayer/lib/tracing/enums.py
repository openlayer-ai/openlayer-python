"""Module with the enums used in the tracing module."""

import enum


class StepType(enum.Enum):
    USER_CALL = "user_call"
    CHAT_COMPLETION = "chat_completion"
