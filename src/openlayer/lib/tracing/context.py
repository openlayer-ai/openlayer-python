"""
Streamlined user and session context management for Openlayer tracing.

This module provides simple functions to set user_id and session_id in middleware
and override them anywhere in your traced code.
"""

import contextvars
import threading
from typing import Optional, Union

# Sentinel object to distinguish between "not provided" and "explicitly None"
_NOT_PROVIDED = object()

# Context variables for user and session tracking
_user_id_context = contextvars.ContextVar("openlayer_user_id", default=None)
_session_id_context = contextvars.ContextVar("openlayer_session_id", default=None)

# Thread-local fallback for environments where contextvars don't work well
_thread_local = threading.local()


class UserSessionContext:
    """Internal class to manage user and session context."""
    
    @staticmethod
    def set_user_id(user_id: Union[str, int, None]) -> None:
        """Set the user ID for the current context."""
        user_id_str = str(user_id) if user_id is not None else None
        _user_id_context.set(user_id_str)
        
        # Thread-local fallback
        _thread_local.user_id = user_id_str
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get the current user ID."""
        try:
            return _user_id_context.get(None)
        except LookupError:
            # Fallback to thread-local
            return getattr(_thread_local, 'user_id', None)
    
    @staticmethod
    def set_session_id(session_id: Union[str, None]) -> None:
        """Set the session ID for the current context."""
        _session_id_context.set(session_id)
        
        # Thread-local fallback
        _thread_local.session_id = session_id
    
    @staticmethod
    def get_session_id() -> Optional[str]:
        """Get the current session ID."""
        try:
            return _session_id_context.get(None)
        except LookupError:
            # Fallback to thread-local
            return getattr(_thread_local, 'session_id', None)
    
    @staticmethod
    def clear_context() -> None:
        """Clear all user and session context."""
        _user_id_context.set(None)
        _session_id_context.set(None)
        
        # Clear thread-local
        for attr in ['user_id', 'session_id']:
            if hasattr(_thread_local, attr):
                delattr(_thread_local, attr)


# ----------------------------- Public API Functions ----------------------------- #

def set_user_session_context(
    user_id: Union[str, int, None] = None,
    session_id: Union[str, None] = None,
) -> None:
    """Set user and session context for tracing (typically called in middleware).
    
    This function should be called once per request in your middleware to establish
    default user_id and session_id values that will be automatically included in all traces.
    
    Args:
        user_id: The user identifier
        session_id: The session identifier
    
    Example:
        >>> from openlayer.lib.tracing import set_user_session_context
        >>> 
        >>> # In your middleware or request handler
        >>> def middleware(request):
        ...     set_user_session_context(
        ...         user_id=request.user.id,
        ...         session_id=request.session.session_key
        ...     )
        ...     # Now all traced functions will automatically include these values
    """
    if user_id is not None:
        UserSessionContext.set_user_id(user_id)
    if session_id is not None:
        UserSessionContext.set_session_id(session_id)


def update_trace_user_session(
    user_id: Union[str, int, None] = _NOT_PROVIDED,
    session_id: Union[str, None] = _NOT_PROVIDED,
) -> None:
    """Update user_id and/or session_id for the current trace context.
    
    This can be called anywhere in your traced code to override the user_id
    and/or session_id set in middleware. Inspired by Langfuse's updateActiveTrace pattern.
    
    Args:
        user_id: The user identifier to set (optional). Pass None to clear.
        session_id: The session identifier to set (optional). Pass None to clear.
        
    Example:
        >>> from openlayer.lib.tracing import update_trace_user_session
        >>> 
        >>> @trace()
        >>> def process_request():
        ...     # Override user_id for this specific trace
        ...     update_trace_user_session(user_id="different_user_123")
        ...     return "result"
        >>> 
        >>> @trace()
        >>> def start_new_session():
        ...     # Start a new session for this trace
        ...     update_trace_user_session(session_id="new_session_456")
        ...     return "result"
        >>> 
        >>> @trace()
        >>> def switch_user_and_session():
        ...     # Update both at once
        ...     update_trace_user_session(
        ...         user_id="admin_user_789",
        ...         session_id="admin_session_abc"
        ...     )
        ...     return "result"
        >>> 
        >>> @trace()
        >>> def clear_user():
        ...     # Clear user_id (set to None)
        ...     update_trace_user_session(user_id=None)
        ...     return "result"
    """
    # Use sentinel object to distinguish between "not provided" and "explicitly None"
    if user_id is not _NOT_PROVIDED:
        UserSessionContext.set_user_id(user_id)
    if session_id is not _NOT_PROVIDED:
        UserSessionContext.set_session_id(session_id)


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context."""
    return UserSessionContext.get_user_id()


def get_current_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return UserSessionContext.get_session_id()


def clear_user_session_context() -> None:
    """Clear all user and session context."""
    UserSessionContext.clear_context()
