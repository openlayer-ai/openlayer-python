"""OpenLayer tracing module."""

from .tracer import (
    trace,
    trace_async,
    update_current_trace,
    update_current_step,
    log_context,
    log_output,
    configure,
    get_current_trace,
    get_current_step,
    create_step,
)

from .context import (
    set_user_session_context,
    update_trace_user_session,
    get_current_user_id,
    get_current_session_id,
    clear_user_session_context,
)

__all__ = [
    # Core tracing functions
    "trace",
    "trace_async", 
    "update_current_trace",
    "update_current_step",
    "log_context",
    "log_output",
    "configure",
    "get_current_trace",
    "get_current_step", 
    "create_step",
    
    # User and session context functions
    "set_user_session_context",
    "update_trace_user_session",
    "get_current_user_id",
    "get_current_session_id",
    "clear_user_session_context",
]

